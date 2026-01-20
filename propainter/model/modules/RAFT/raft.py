import torch
from torch import nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import coords_grid, upflow8


# Setup autocast with proper version handling
def _get_autocast_context(enabled: bool, device: str = 'cuda'):
    """Get appropriate autocast context manager for current PyTorch version."""
    try:
        # Try PyTorch 2.0+ API first (torch.amp.autocast)
        from torch.amp import autocast as amp_autocast
        return amp_autocast(device, enabled=enabled)
    except (ImportError, TypeError):
        try:
            # Fallback to PyTorch 1.10+ API (torch.cuda.amp.autocast)
            # When using old API, device parameter is not supported
            from torch.cuda.amp import autocast as cuda_autocast
            return cuda_autocast(enabled=enabled)
        except ImportError:
            # Dummy autocast for very old PyTorch versions
            class DummyAutocast:
                def __init__(self, enabled):
                    self.enabled = enabled
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyAutocast(enabled)


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self._coords_cache = {}

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if "dropout" not in args._get_kwargs():
            args.dropout = 0

        if "alternate_corr" not in args._get_kwargs():
            args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = SmallEncoder(
                output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = BasicEncoder(
                output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords_base = self._get_coords_grid(H // 8, W // 8, img.device)
        coords0 = coords_base
        coords1 = coords_base.expand(N, -1, -1, -1)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def _get_coords_grid(self, ht, wd, device):
        key = (device, ht, wd)
        coords = self._coords_cache.get(key)
        if coords is None:
            coords = coords_grid(1, ht, wd, device=device)
            self._coords_cache[key] = coords
        return coords

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=True):
        """Estimate optical flow between pair of frames
        
        Args:
            image1: First image tensor [B, C, H, W], expected in range [0, 1] or [0, 255]
            image2: Second image tensor [B, C, H, W], same format as image1
            iters: Number of update iterations (default: 12, use 20 for test mode)
            flow_init: Initial flow estimate [B, 2, H, W] (optional)
            test_mode: Whether to use test mode (higher accuracy, longer runtime)
        
        Returns:
            Tuple of (flow_low, flow) where flow is [B, 2, H, W] optical flow
        
        Note:
            Input images are expected in normalized range [0, 1] or [0, 255].
            The network normalizes internally if needed: image = 2 * (image / 255.0) - 1.0
            Current implementation assumes inputs are pre-normalized or in raw uint8 range.
        """
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # Use proper autocast context (handles both new and old PyTorch APIs)
        amp_enabled = getattr(self.args, 'mixed_precision', False)
        device = 'cuda' if image1.is_cuda else 'cpu'

        # run the feature network
        with _get_autocast_context(amp_enabled, device):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with _get_autocast_context(amp_enabled, device):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = [] if not test_mode else None
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with _get_autocast_context(amp_enabled, device):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            if flow_predictions is not None:
                flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
