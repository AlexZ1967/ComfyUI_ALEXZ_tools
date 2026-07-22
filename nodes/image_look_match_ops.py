"""Batch, mask, resize, and numerical helpers for Look Match nodes."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_nn_func

from ..utils import color_match_utils

_EPS = 1e-6
_LONG_SIDE_TARGET = {"as_is": None, "1440p": 1440, "1080p": 1080, "720p": 720}

def _pad_batch_last(batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Pad tensor batch by repeating last item until `batch_size`."""
    return color_match_utils.pad_batch_last(batch, batch_size)


def _resolve_compute_device(requested: str) -> tuple[torch.device, str | None]:
    """Resolve requested compute device and return optional warning code."""
    req = str(requested).lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        return torch.device("cpu"), "cuda_requested_but_unavailable"
    if req == "cpu":
        return torch.device("cpu"), None
    if torch.cuda.is_available():
        return torch.device("cuda"), None
    return torch.device("cpu"), None


def _split_alpha(image_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split RGB and alpha channels from BHWC image batch."""
    if image_batch.shape[-1] > 3:
        return image_batch[..., :3], image_batch[..., 3:4]
    return image_batch, None


def _prepare_optional_mask_batch(mask: torch.Tensor | None, batch_size: int, height: int, width: int) -> torch.Tensor | None:
    """Normalize optional mask to BHW float tensor in [0, 1]."""
    if mask is None:
        return None
    mm = _pad_batch_last(mask, batch_size).float()
    if mm.dim() == 4:
        if mm.shape[-1] == 1:
            mm = mm[..., 0]
        elif mm.shape[1] == 1:
            mm = mm[:, 0]
        else:
            mm = mm[..., 0]
    if mm.dim() == 2:
        mm = mm.unsqueeze(0)
    max_val = float(mm.max().item()) if mm.numel() else 0.0
    if max_val > 1.0:
        mm = mm / 255.0
    mm = mm.clamp(0.0, 1.0)
    if mm.shape[-2] != height or mm.shape[-1] != width:
        mm = torch_nn_func.interpolate(mm.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
    return mm


def _resize_mask_hw(mask_hw: torch.Tensor | None, height: int, width: int) -> torch.Tensor | None:
    """Resize HxW mask to target HxW using nearest interpolation."""
    if mask_hw is None:
        return None
    if mask_hw.shape[-2] == height and mask_hw.shape[-1] == width:
        return mask_hw
    return torch_nn_func.interpolate(mask_hw.unsqueeze(0).unsqueeze(0), size=(height, width), mode="nearest").squeeze(0).squeeze(0)


def _downscale_hwc_long_side(image_hwc: torch.Tensor, mode: str) -> tuple[torch.Tensor, dict]:
    """Downscale HWC tensor by long side target without upscaling."""
    h = int(image_hwc.shape[0])
    w = int(image_hwc.shape[1])
    target = _LONG_SIDE_TARGET.get(str(mode), None)
    if target is None:
        return image_hwc, {"optimized_size": [h, w], "scale": 1.0}
    long_side = max(h, w)
    if long_side <= int(target):
        return image_hwc, {"optimized_size": [h, w], "scale": 1.0}
    scale = float(target) / float(long_side)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = torch_nn_func.interpolate(
        image_hwc.permute(2, 0, 1).unsqueeze(0),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)
    return resized, {"optimized_size": [nh, nw], "scale": scale}


def _resize_hwc_to(image_hwc: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize HWC tensor to requested size with bilinear filtering."""
    if int(image_hwc.shape[0]) == int(height) and int(image_hwc.shape[1]) == int(width):
        return image_hwc
    return torch_nn_func.interpolate(
        image_hwc.permute(2, 0, 1).unsqueeze(0),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)


def _luma(rgb_hwc: torch.Tensor) -> torch.Tensor:
    """Compute luma channel from RGB image."""
    return (
        rgb_hwc[..., 0] * 0.2126
        + rgb_hwc[..., 1] * 0.7152
        + rgb_hwc[..., 2] * 0.0722
    )


def _stage_alpha(weight: float) -> float:
    """Convert stage weight to bounded blend alpha."""
    w = max(0.0, float(weight))
    return float(1.0 - torch.exp(torch.tensor(-w)).item())


def _blend_stage(base: torch.Tensor, stage: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend stage output with base image."""
    a = float(max(0.0, min(1.0, alpha)))
    if a <= 0.0:
        return base
    if a >= 1.0:
        return stage
    return base * (1.0 - a) + stage * a


def _compose_fit_mask(subject: torch.Tensor | None, sky: torch.Tensor | None, ground: torch.Tensor | None) -> torch.Tensor | None:
    """Compose weighted fit mask from optional region masks."""
    template = subject if subject is not None else (sky if sky is not None else ground)
    if template is None:
        return None
    m = torch.full_like(template, 0.25)
    if subject is not None:
        m = m + subject * 1.0
    if sky is not None:
        m = m + sky * 0.5
    if ground is not None:
        m = m + ground * 0.5
    return m.clamp(0.0, 1.5)


def _weighted_mean_channel(img_hwc: torch.Tensor, mask_hw: torch.Tensor | None) -> torch.Tensor:
    """Compute per-channel weighted mean for HWC tensor."""
    if mask_hw is None:
        return img_hwc.mean(dim=(0, 1))
    w = mask_hw.unsqueeze(-1).clamp(0.0, 1.5)
    denom = w.sum().clamp_min(_EPS)
    return (img_hwc * w).sum(dim=(0, 1)) / denom


def _weighted_luma_mean(img_hwc: torch.Tensor, mask_hw: torch.Tensor | None) -> torch.Tensor:
    """Compute weighted luma mean."""
    lum = _luma(img_hwc)
    if mask_hw is None:
        return lum.mean()
    w = mask_hw.clamp(0.0, 1.5)
    return (lum * w).sum() / w.sum().clamp_min(_EPS)


def _fit_exposure_gain(src_hwc: torch.Tensor, ref_hwc: torch.Tensor, fit_mask_hw: torch.Tensor | None) -> torch.Tensor:
    """Fit exposure + white-balance gain vector."""
    src_mean = _weighted_mean_channel(src_hwc, fit_mask_hw)
    ref_mean = _weighted_mean_channel(ref_hwc, fit_mask_hw)
    wb_gain = (ref_mean / src_mean.clamp_min(_EPS)).clamp(0.2, 5.0)
    wb_gain = wb_gain / wb_gain.mean().clamp_min(_EPS)
    src_l = _weighted_luma_mean(src_hwc, fit_mask_hw)
    ref_l = _weighted_luma_mean(ref_hwc, fit_mask_hw)
    exp_gain = (ref_l / src_l.clamp_min(_EPS)).clamp(0.25, 4.0)
    return (wb_gain * exp_gain).clamp(0.2, 5.0)


def _ensure_strict_knots(knots: torch.Tensor) -> torch.Tensor:
    """Enforce monotonic knot vector for piecewise mapping."""
    kk = knots.clamp(0.0, 1.0)
    kk = torch.cummax(kk, dim=0).values
    n = int(kk.numel())
    if n > 1:
        kk = (kk + torch.linspace(0.0, 1e-4, n, device=kk.device, dtype=kk.dtype)).clamp(0.0, 1.0)
    kk[0] = 0.0
    kk[-1] = 1.0
    return kk


def _piecewise_linear_map(values: torch.Tensor, xk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    """Map values by monotonic piecewise linear curve."""
    xk = _ensure_strict_knots(xk)
    yk = yk.clamp(0.0, 1.0)
    flat = values.reshape(-1)
    idx = torch.bucketize(flat, xk)
    last = int(xk.numel() - 1)
    idx1 = idx.clamp(1, last)
    idx0 = idx1 - 1
    x0 = xk[idx0]
    x1 = xk[idx1]
    y0 = yk[idx0]
    y1 = yk[idx1]
    t = (flat - x0) / (x1 - x0).clamp_min(_EPS)
    out = y0 + t * (y1 - y0)
    out = torch.where(idx == 0, yk[0], out)
    out = torch.where(idx >= xk.numel(), yk[-1], out)
    return out.reshape(values.shape).clamp(0.0, 1.0)


def _gradient_energy(luma_hw: torch.Tensor) -> float:
    """Estimate average local detail energy from luminance gradients."""
    if luma_hw.shape[0] < 2 or luma_hw.shape[1] < 2:
        return 0.0
    gx = (luma_hw[:, 1:] - luma_hw[:, :-1]).abs().mean()
    gy = (luma_hw[1:, :] - luma_hw[:-1, :]).abs().mean()
    return float((gx + gy).item())
