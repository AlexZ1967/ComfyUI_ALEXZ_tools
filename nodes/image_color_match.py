"""
Module: nodes/image_color_match.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Color Match To Reference node implementation.

Purpose:
    Implements color-transfer methods, quality metrics, and JSON export for image-to-reference matching.
"""

import json

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        """Fallback tqdm wrapper that returns the iterable unchanged when tqdm is unavailable."""
        return iterable if iterable is not None else []

from ..utils import color_match_utils
from ..utils.color_match_utils import normalize_mask

_LOGGER = logging.getLogger("ImageColorMatchToReference")
_SSIM_WINDOW_CACHE = {}
_LPIPS_CACHE = {}
_VGG_CACHE = {}


def _pad_batch_last(batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Compatibility wrapper around shared batch-padding helper."""
    return color_match_utils.pad_batch_last(batch, batch_size)


def _linear_fit_torch_batch(
    img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper around shared linear-fit helper."""
    return color_match_utils.linear_fit_torch_batch(img, ref, mask)


def _mean_std_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Compatibility wrapper around shared mean/std helper."""
    return color_match_utils.mean_std_match_torch_batch(img, ref, mask)


def _linear_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Compatibility wrapper around shared linear-match helper."""
    return color_match_utils.linear_match_torch_batch(img, ref, mask)


def _lab_match_torch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor], mode: str):
    """Match color statistics in Lab space and return corrected RGB tensor."""
    if mask is None:
        mask_t = torch.ones((img.shape[0], img.shape[1]), dtype=img.dtype, device=img.device)
    else:
        mask_t = normalize_mask(mask)
    out = color_match_utils.apply_color_match(
        img.unsqueeze(0),
        ref.unsqueeze(0),
        mask_t,
        mode,
        mask_white_is_keep=True,
    )
    return out[0]


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB values [0,1] to linear RGB."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB values to sRGB [0,1]."""
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * torch.pow(torch.clamp(rgb, min=0.0), 1.0 / 2.4) - 0.055,
    )


def _rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor (...,3) in [0,1] to CIE Lab."""
    rgb_lin = _srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
    m_rgb_xyz = rgb_lin.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.matmul(rgb_lin, m_rgb_xyz.t())
    white = xyz.new_tensor([0.95047, 1.0, 1.08883])
    xyz_n = xyz / white
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    f = torch.where(
        xyz_n > delta3,
        torch.pow(torch.clamp(xyz_n, min=0.0), 1.0 / 3.0),
        xyz_n / (3.0 * delta * delta) + (4.0 / 29.0),
    )
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([l, a, b], dim=-1)


def _lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """Convert CIE Lab tensor (...,3) back to RGB in [0,1]."""
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def invf(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, t ** 3, 3.0 * (delta ** 2) * (t - 4.0 / 29.0))

    white = lab.new_tensor([0.95047, 1.0, 1.08883])
    xyz = torch.stack([invf(fx), invf(fy), invf(fz)], dim=-1) * white
    m_xyz_rgb = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb_lin = torch.matmul(xyz, m_xyz_rgb.t())
    rgb = _linear_to_srgb(rgb_lin)
    return torch.clamp(rgb, 0.0, 1.0)


def _rgb_to_oklab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (...,3) in [0,1] to Oklab."""
    rgb_lin = _srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
    m_rgb_lms = rgb.new_tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = torch.matmul(rgb_lin, m_rgb_lms.t())
    lms_cbrt = torch.pow(torch.clamp(lms, min=0.0), 1.0 / 3.0)
    m_lms_oklab = rgb.new_tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return torch.matmul(lms_cbrt, m_lms_oklab.t())


def _oklab_to_rgb_torch(oklab: torch.Tensor) -> torch.Tensor:
    """Convert Oklab tensor (...,3) to RGB in [0,1]."""
    m_oklab_lms = oklab.new_tensor(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    )
    lms_cbrt = torch.matmul(oklab, m_oklab_lms.t())
    lms = lms_cbrt ** 3
    m_lms_rgb = oklab.new_tensor(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )
    rgb_lin = torch.matmul(lms, m_lms_rgb.t())
    rgb = _linear_to_srgb(rgb_lin)
    return torch.clamp(rgb, 0.0, 1.0)


def _interp1d_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D linear interpolation on tensors with monotonic xp."""
    idx = torch.searchsorted(xp, x, right=False)
    max_idx = xp.shape[0] - 1
    idx0 = torch.clamp(idx - 1, 0, max_idx)
    idx1 = torch.clamp(idx, 0, max_idx)
    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]
    denom = torch.clamp(x1 - x0, min=1e-8)
    w = (x - x0) / denom
    y = y0 + w * (y1 - y0)
    y = torch.where(idx <= 0, fp[0], y)
    y = torch.where(idx >= xp.shape[0], fp[-1], y)
    return y


def _hist_match_channel_torch(
    src: torch.Tensor,
    ref: torch.Tensor,
    keep: torch.Tensor,
    bins: int,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Histogram CDF matching for one channel using torch ops only."""
    src_vals = src[keep]
    ref_vals = ref[keep]
    if src_vals.numel() < 10 or ref_vals.numel() < 10:
        return src
    vmin, vmax = value_range
    src_hist = torch.histc(src_vals.float(), bins=bins, min=vmin, max=vmax)
    ref_hist = torch.histc(ref_vals.float(), bins=bins, min=vmin, max=vmax)
    src_cdf = torch.cumsum(src_hist, dim=0)
    ref_cdf = torch.cumsum(ref_hist, dim=0)
    src_cdf = src_cdf / torch.clamp(src_cdf[-1], min=1e-8)
    ref_cdf = ref_cdf / torch.clamp(ref_cdf[-1], min=1e-8)
    bin_edges = torch.linspace(vmin, vmax, steps=bins + 1, device=src.device, dtype=torch.float32)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    interp_values = _interp1d_torch(src_cdf, ref_cdf, centers)
    src_flat = src.contiguous().view(-1).float()
    indices = torch.bucketize(src_flat, bin_edges, right=True) - 1
    indices = torch.clamp(indices, 0, bins - 1)
    matched = interp_values[indices].reshape_as(src).to(dtype=src.dtype)
    return matched


def _adain_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """AdaIN-style matching using masked mean/std statistics."""
    return _mean_std_match_batch(img, ref, mask)


def _tone_curve_match_batch(
    img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor], num_points: int = 5
) -> torch.Tensor:
    """Tone-curve luminance matching (quantile-based) in torch."""
    bsz, h, w, _ = img.shape
    out = img.clone()
    keep_all = torch.ones((h, w), dtype=torch.bool, device=img.device)
    quantiles = torch.linspace(0.05, 0.95, steps=num_points, device=img.device, dtype=torch.float32)
    zero = torch.tensor([0.0], device=img.device, dtype=torch.float32)
    one = torch.tensor([1.0], device=img.device, dtype=torch.float32)
    src_gray = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    ref_gray = ref[..., 0] * 0.299 + ref[..., 1] * 0.587 + ref[..., 2] * 0.114
    for b in range(bsz):
        keep = keep_all if mask is None else (mask[b] > 0.5)
        src_vals = src_gray[b][keep]
        ref_vals = ref_gray[b][keep]
        if src_vals.numel() < 10 or ref_vals.numel() < 10:
            continue
        src_q = torch.quantile(src_vals.float(), quantiles)
        ref_q = torch.quantile(ref_vals.float(), quantiles)
        src_points = torch.cat([zero, src_q, one], dim=0)
        ref_points = torch.cat([zero, ref_q, one], dim=0)
        lum = src_gray[b].float()
        mapped = _interp1d_torch(lum.contiguous().view(-1), src_points, ref_points).view(h, w)
        scale = mapped / torch.clamp(lum, min=1e-6)
        scale = torch.clamp(scale, 0.5, 2.0).to(dtype=img.dtype)
        out[b] = torch.clamp(img[b] * scale.unsqueeze(-1), 0.0, 1.0)
    return out


def _optimal_transport_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Per-channel 1D optimal transport matching in torch."""
    bsz, h, w, _ = img.shape
    out = img.clone()
    keep_all = torch.ones((h, w), dtype=torch.bool, device=img.device)
    for b in range(bsz):
        keep = keep_all if mask is None else (mask[b] > 0.5)
        if keep.sum() < 10:
            continue
        for c in range(3):
            src = img[b, :, :, c]
            tar = ref[b, :, :, c]
            src_vals = src[keep]
            ref_vals = tar[keep]
            if src_vals.numel() < 10 or ref_vals.numel() < 10:
                continue
            src_sorted, src_order = torch.sort(src_vals)
            ref_sorted, _ = torch.sort(ref_vals)
            src_q = torch.linspace(0.0, 1.0, steps=src_sorted.numel(), device=img.device, dtype=torch.float32)
            ref_q = torch.linspace(0.0, 1.0, steps=ref_sorted.numel(), device=img.device, dtype=torch.float32)
            mapped_sorted = _interp1d_torch(src_q, ref_q, ref_sorted.float()).to(dtype=img.dtype)
            mapped = torch.empty_like(src_vals)
            mapped[src_order] = mapped_sorted
            out[b, :, :, c][keep] = mapped
    return torch.clamp(out, 0.0, 1.0)


def _cdf_match_batch_space(
    space_img: torch.Tensor,
    space_ref: torch.Tensor,
    mask: Optional[torch.Tensor],
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> torch.Tensor:
    """Apply per-channel CDF matching in arbitrary 3-channel color space."""
    bsz, h, w, _ = space_img.shape
    out = space_img.clone()
    keep_all = torch.ones((h, w), dtype=torch.bool, device=space_img.device)
    for b in range(bsz):
        keep = keep_all if mask is None else (mask[b] > 0.5)
        if keep.sum() < 10:
            continue
        for c in range(3):
            out[b, :, :, c] = _hist_match_channel_torch(
                space_img[b, :, :, c], space_ref[b, :, :, c], keep, bins=256, value_range=ranges[c]
            )
    return out


def _lab_cdf_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Torch-only Lab CDF matching."""
    lab_img = _rgb_to_lab_torch(img)
    lab_ref = _rgb_to_lab_torch(ref)
    ranges = ((0.0, 100.0), (-127.0, 127.0), (-127.0, 127.0))
    matched = _cdf_match_batch_space(lab_img, lab_ref, mask, ranges)
    matched[..., 0] = torch.clamp(matched[..., 0], 0.0, 100.0)
    matched[..., 1] = torch.clamp(matched[..., 1], -127.0, 127.0)
    matched[..., 2] = torch.clamp(matched[..., 2], -127.0, 127.0)
    return _lab_to_rgb_torch(matched)


def _oklab_cdf_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Torch-only Oklab CDF matching."""
    oklab_img = _rgb_to_oklab_torch(img)
    oklab_ref = _rgb_to_oklab_torch(ref)
    ranges = ((0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))
    matched = _cdf_match_batch_space(oklab_img, oklab_ref, mask, ranges)
    matched[..., 0] = torch.clamp(matched[..., 0], 0.0, 1.0)
    matched[..., 1] = torch.clamp(matched[..., 1], -0.5, 0.5)
    matched[..., 2] = torch.clamp(matched[..., 2], -0.5, 0.5)
    return _oklab_to_rgb_torch(matched)


def _mean_std_fit_torch_batch(
    img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper around shared mean/std-fit helper."""
    return color_match_utils.mean_std_fit_torch_batch(img, ref, mask)


def _lut_grid_colors(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create flattened LUT input colors in .cube order (R fastest)."""
    levels = torch.linspace(0.0, 1.0, steps=size, device=device, dtype=dtype)
    bb, gg, rr = torch.meshgrid(levels, levels, levels, indexing="ij")
    return torch.stack([rr, gg, bb], dim=-1).reshape(-1, 3)


def _poly2_features(colors: torch.Tensor) -> torch.Tensor:
    """Build second-order polynomial RGB features for LUT baking."""
    r = colors[:, 0:1]
    g = colors[:, 1:2]
    b = colors[:, 2:3]
    ones = torch.ones_like(r)
    return torch.cat([ones, r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], dim=1)


def _fit_poly2_color_map(
    src: torch.Tensor, dst: torch.Tensor, mask: Optional[torch.Tensor], max_samples: int = 50000
) -> torch.Tensor:
    """Fit polynomial RGB map from src->dst colors for LUT baking."""
    src_flat = src.reshape(-1, 3)
    dst_flat = dst.reshape(-1, 3)
    if mask is not None:
        keep = (mask > 0.5).reshape(-1)
        if int(keep.sum().item()) >= 10:
            src_flat = src_flat[keep]
            dst_flat = dst_flat[keep]
    if src_flat.shape[0] > max_samples:
        idx = torch.randperm(src_flat.shape[0], device=src.device)[:max_samples]
        src_flat = src_flat[idx]
        dst_flat = dst_flat[idx]
    x = _poly2_features(src_flat.float())
    y = dst_flat.float()
    xtx = x.t().matmul(x)
    xty = x.t().matmul(y)
    ridge = torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype) * 1e-4
    beta = torch.linalg.solve(xtx + ridge, xty)
    return beta


def _apply_poly2_color_map(colors: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply fitted polynomial RGB map to flattened LUT colors."""
    x = _poly2_features(colors.float())
    out = x.matmul(beta).to(dtype=colors.dtype)
    return torch.clamp(out, 0.0, 1.0)


def _sanitize_lut_name(name: str) -> str:
    """Sanitize LUT filename base."""
    base = (name or "").strip()
    if not base:
        return "color_match_lut"
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in base)
    return safe.strip("_") or "color_match_lut"


def _write_cube_file(path: Path, colors: torch.Tensor, size: int, title: str):
    """Write LUT colors to .cube file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by ComfyUI_ALEXZ_tools Color Match To Reference",
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {int(size)}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    flat = colors.detach().cpu().float().numpy()
    for r, g, b in flat:
        lines.append(f"{float(r):.6f} {float(g):.6f} {float(b):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _vgg19_features(device: torch.device):
    """Load and cache VGG19 feature extractor used by perceptual matching."""
    key = ("vgg19_f12", device.type, device.index if device.type == "cuda" else -1)
    if key in _VGG_CACHE:
        return _VGG_CACHE[key]
    try:
        from torchvision.models import VGG19_Weights, vgg19
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "torchvision is required for preset=perceptual. "
            "Use ComfyUI default environment or install torchvision."
        ) from exc
    _LOGGER.info("Loading VGG19 (perceptual_vgg)...")
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:12].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    _VGG_CACHE[key] = vgg
    return vgg


def _perceptual_vgg(img: torch.Tensor, ref: torch.Tensor, steps: int, lr: float):
    """Optimize image parameters against VGG perceptual features for high-quality matching."""
    device = img.device
    inf_ctx = torch.inference_mode(False) if torch.is_inference_mode_enabled() else nullcontext()

    with inf_ctx:
        vgg = _vgg19_features(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

        def prep(x):
            """Normalize and resize an image tensor for VGG feature extraction."""
            x = x.float().permute(2, 0, 1).unsqueeze(0)
            return (x - mean) / std

        with torch.no_grad():
            feat_ref = vgg(prep(ref.detach().clone()))

        W = torch.eye(3, device=device, dtype=img.dtype).requires_grad_(True)
        b = torch.zeros(3, device=device, dtype=img.dtype).requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=lr)

        steps_int = max(1, int(steps))
        img_work = img.detach().clone()
        with torch.enable_grad():
            iterator = tqdm(range(steps_int), desc="perceptual_vgg", leave=False)
            for _ in iterator:
                opt.zero_grad(set_to_none=True)
                x = torch.clamp(torch.einsum("hwc,dc->hwd", img_work, W) + b, 0.0, 1.0)
                feat_x = vgg(prep(x))
                loss = torch.mean((feat_x - feat_ref) ** 2)
                loss.backward()
                opt.step()

    corrected = torch.clamp(torch.einsum("hwc,dc->hwd", img, W.detach()) + b.detach(), 0.0, 1.0)
    params = {
        "matrix": W.detach().cpu().tolist(),
        "bias": b.detach().cpu().tolist(),
        "loss_final": float(loss.detach().cpu()),
        "steps": int(steps),
        "lr": float(lr),
    }
    return corrected, params


def _perceptual_vgg_fast(img: torch.Tensor, ref: torch.Tensor, steps: int, lr: float, max_side: int = 256):
    """Run a lightweight perceptual optimization pass tuned for speed."""
    h, w, _ = img.shape
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        img_small = F.interpolate(
            img.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        ref_small = F.interpolate(
            ref.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
    else:
        img_small, ref_small = img, ref

    fast_steps = max(1, min(int(steps), 5))
    _, params = _perceptual_vgg(img_small, ref_small, fast_steps, lr)

    W = torch.tensor(params["matrix"], device=img.device, dtype=img.dtype)
    b = torch.tensor(params["bias"], device=img.device, dtype=img.dtype)
    corrected_full = torch.clamp(torch.einsum("hwc,dc->hwd", img, W) + b, 0.0, 1.0)
    params["mode"] = "perceptual_vgg_fast"
    params["used_scale"] = scale
    params["used_steps"] = fast_steps
    return corrected_full, params


def _ssim_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5):
    """Create a Gaussian convolution window for SSIM computation."""
    key = (channels, device.type, str(dtype), size, sigma)
    if key in _SSIM_WINDOW_CACHE:
        return _SSIM_WINDOW_CACHE[key]
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel = kernel_2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    _SSIM_WINDOW_CACHE[key] = kernel
    return kernel


def _ssim_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute SSIM similarity score for two image tensors."""
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    channels = a.shape[1]
    window = _ssim_window(channels, a.device, a.dtype)
    mu1 = F.conv2d(a, window, padding=window.shape[-1] // 2, groups=channels)
    mu2 = F.conv2d(b, window, padding=window.shape[-1] // 2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(a * a, window, padding=window.shape[-1] // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=window.shape[-1] // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=window.shape[-1] // 2, groups=channels) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().clamp(0.0, 1.0).item())


def _downscale_max_side(img: torch.Tensor, max_side: int = 256) -> torch.Tensor:
    """Downscale an image tensor so its longest side does not exceed the given limit."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = float(max_side) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    out = F.interpolate(
        img.permute(2, 0, 1).unsqueeze(0),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def _lpips_model(device: torch.device):
    """Load and cache LPIPS model weights for perceptual scoring."""
    key = ("alex", device.type)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    model = lpips.LPIPS(net="alex").to(device).eval()
    _LPIPS_CACHE[key] = model
    return model


def _lpips_alex_distance(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """Compute LPIPS AlexNet perceptual distance between image tensors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _lpips_model(device)
    if model is None:
        return None
    aa = _downscale_max_side(a, 256).permute(2, 0, 1).unsqueeze(0).to(device)
    bb = _downscale_max_side(b, 256).permute(2, 0, 1).unsqueeze(0).to(device)
    aa = aa * 2.0 - 1.0
    bb = bb * 2.0 - 1.0
    with torch.inference_mode():
        dist = model(aa, bb)
    return float(dist.item())


def _delta_e76_mean(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """Compute mean DeltaE76 color distance in Lab space."""
    if color_match_utils.cv2 is None:
        return None
    a_np = np.clip(a.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    b_np = np.clip(b.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    a_lab = color_match_utils.cv2.cvtColor(a_np, color_match_utils.cv2.COLOR_RGB2LAB)
    b_lab = color_match_utils.cv2.cvtColor(b_np, color_match_utils.cv2.COLOR_RGB2LAB)
    delta = np.linalg.norm(a_lab - b_lab, axis=2)
    return float(np.mean(delta))


def _quality_metrics(img: torch.Tensor, ref: torch.Tensor):
    """Compute quality metrics used for before/after color-match reporting."""
    mse = float(torch.mean((img - ref) ** 2).item())
    ssim = _ssim_similarity(img, ref)
    de = _delta_e76_mean(img, ref)
    lpips_a = _lpips_alex_distance(img, ref)
    return {
        "mse": mse,
        "ssim": ssim,
        "delta_e76": de,
        "lpips_alex": lpips_a,
    }


def _auto_optimal_candidate_metrics(candidate: torch.Tensor, ref: torch.Tensor, strategy: str) -> dict:
    """Compute candidate metrics used to pick auto_optimal method."""
    mse = float(torch.mean((candidate - ref) ** 2).item())
    ssim = None
    lpips_a = None
    if strategy in ("mse_ssim", "mse_ssim_lpips"):
        cand_s = _downscale_max_side(candidate, 256)
        ref_s = _downscale_max_side(ref, 256)
        ssim = _ssim_similarity(cand_s, ref_s)
    if strategy == "mse_ssim_lpips":
        lpips_a = _lpips_alex_distance(candidate, ref)
    return {
        "mse": mse,
        "ssim": ssim,
        "lpips_alex": lpips_a,
    }


def _auto_optimal_score(metrics: dict, strategy: str) -> float:
    """Compose scalar auto_optimal score from candidate metrics."""
    score = float(metrics["mse"])
    if strategy in ("mse_ssim", "mse_ssim_lpips") and metrics.get("ssim") is not None:
        score += 0.05 * (1.0 - float(metrics["ssim"]))
    if strategy == "mse_ssim_lpips" and metrics.get("lpips_alex") is not None:
        score += 0.05 * float(metrics["lpips_alex"])
    return score


def _empty_quality_metrics() -> dict:
    """Return empty quality metrics payload when metric evaluation is disabled."""
    return {
        "mse": None,
        "ssim": None,
        "delta_e76": None,
        "lpips_alex": None,
    }


def _improvement_pct(before: dict, after: dict) -> dict:
    """Convert metric pairs into percentage-improvement values."""
    res = {}
    for k in ("mse", "delta_e76", "lpips_alex"):
        bv = before.get(k)
        av = after.get(k)
        if bv is None or av is None:
            res[k] = None
        else:
            denom = max(float(bv), 1e-6)
            res[k] = round((float(bv) - float(av)) / denom * 100.0, 3)
    bv = before.get("ssim")
    av = after.get("ssim")
    if bv is None or av is None:
        res["ssim"] = None
    else:
        denom = max(float(bv), 1e-6)
        res["ssim"] = round((float(av) - float(bv)) / denom * 100.0, 3)
    return res


class ImageColorMatchToReference:
    """ComfyUI node that matches image color grading to a reference frame."""
    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Базовое изображение (образец)."}),
                "image": ("IMAGE", {"tooltip": "Изображение, которое нужно подогнать по цвету."}),
                "preset": (
                    ["mean_std", "linear", "tone_curve", "adain", "optimal_transport", "lab_cdf", "oklab_cdf", "auto_optimal", "perceptual_vgg_fast"],
                    {
                        "default": "linear",
                        "tooltip": "Метод: mean_std=стд, linear=линейная, tone_curve=кривая, adain=AdaIN норм, optimal_transport=Wasserstein, lab_cdf=Lab, oklab_cdf=Oklab, auto_optimal=автовыбор linear/oklab_cdf, perceptual_vgg_fast=VGG.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Сила применения коррекции (0..1).",
                    },
                ),
            },
            "optional": {
                "match_mask": ("MASK", {"tooltip": "Где считать статистику (белое=учитывать)."}),
                "apply_mask": ("MASK", {"tooltip": "Где применять коррекцию (белое=применить, чёрное=оставить исходное). Маски повышают время обработки."}),
                "preserve_alpha": ("BOOLEAN", {"default": True, "tooltip": "Если вход RGBA — сохранить альфу из исходника."}),
                "compute_quality_metrics": ("BOOLEAN", {"default": True, "tooltip": "Считать MSE/SSIM/DeltaE/LPIPS. Отключите для ускорения батчей."}),
                "auto_optimal_metric": (
                    ["mse", "mse_ssim", "mse_ssim_lpips"],
                    {"default": "mse_ssim", "tooltip": "Критерий выбора для auto_optimal. mse_ssim_lpips точнее, но медленнее."},
                ),
                "export_lut": ("BOOLEAN", {"default": False, "tooltip": "Экспортировать LUT .cube для каждой пары вход/референс."}),
                "lut_size": ("INT", {"default": 33, "min": 8, "max": 65, "tooltip": "Размер 3D LUT (типично 17/33)."}),
                "lut_output_dir": ("STRING", {"default": "", "tooltip": "Папка для .cube. Пусто = ./output/color_luts."}),
                "lut_name": ("STRING", {"default": "color_match", "tooltip": "Базовое имя LUT файла (без расширения)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "match_json")
    FUNCTION = "match"
    CATEGORY = "image/color"

    def match(
        self,
        reference,
        image,
        preset,
        match_mask=None,
        apply_mask=None,
        preserve_alpha=True,
        compute_quality_metrics=True,
        auto_optimal_metric="mse_ssim",
        export_lut=False,
        lut_size=33,
        lut_output_dir="",
        lut_name="color_match",
        strength=1.0,
    ):
        """Execute the node and return processed outputs for ComfyUI."""
        batch_size = max(reference.shape[0], image.shape[0])
        reference_batch = _pad_batch_last(reference, batch_size)
        image_batch = _pad_batch_last(image, batch_size)
        ref_h, ref_w = reference_batch.shape[1], reference_batch.shape[2]

        alpha_batch = None
        reference_rgb = reference_batch
        image_rgb = image_batch
        if reference_batch.shape[3] > 3:
            alpha_batch = reference_batch[..., 3:4]
            reference_rgb = reference_batch[..., :3]
        if image_batch.shape[3] > 3:
            if alpha_batch is None:
                alpha_batch = image_batch[..., 3:4]
            image_rgb = image_batch[..., :3]

        if image_rgb.shape[1] != ref_h or image_rgb.shape[2] != ref_w:
            image_rgb = color_match_utils.resize_images_to_size(image_rgb, ref_h, ref_w)
        if alpha_batch is not None and (alpha_batch.shape[1] != ref_h or alpha_batch.shape[2] != ref_w):
            alpha_batch = color_match_utils.resize_images_to_size(alpha_batch, ref_h, ref_w)

        match_mask_batch, apply_mask_batch, match_mask_valid = color_match_utils.prepare_match_and_apply_masks(
            match_mask, apply_mask, batch_size, ref_h, ref_w, reference_rgb.device, reference_rgb.dtype
        )
        if match_mask is not None:
            empty_mask_idx = (~match_mask_valid).nonzero(as_tuple=False).flatten().tolist()
            if empty_mask_idx:
                _LOGGER.warning(
                    "ColorMatch: match_mask has no white pixels for frames %s; returning original image for those frames.",
                    empty_mask_idx,
                )

        corrected_batch = None
        auto_mode_batch = None
        auto_linear_batch = None
        auto_oklab_batch = None
        mean_std_scale_batch = None
        mean_std_offset_batch = None
        if preset == "mean_std":
            corrected_batch = _mean_std_match_batch(image_rgb, reference_rgb, match_mask_batch)
            mean_std_scale_batch, mean_std_offset_batch = _mean_std_fit_torch_batch(
                image_rgb, reference_rgb, match_mask_batch
            )
        elif preset == "linear":
            corrected_batch = _linear_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "tone_curve":
            corrected_batch = _tone_curve_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "adain":
            corrected_batch = _adain_match_batch(image_rgb, reference_rgb, match_mask_batch)
            mean_std_scale_batch, mean_std_offset_batch = _mean_std_fit_torch_batch(
                image_rgb, reference_rgb, match_mask_batch
            )
        elif preset == "optimal_transport":
            corrected_batch = _optimal_transport_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "lab_cdf":
            corrected_batch = _lab_cdf_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "oklab_cdf":
            corrected_batch = _oklab_cdf_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "auto_optimal":
            auto_linear_batch = _linear_match_batch(image_rgb, reference_rgb, match_mask_batch)
            auto_oklab_batch = _oklab_cdf_match_batch(image_rgb, reference_rgb, match_mask_batch)
            if auto_optimal_metric == "mse":
                mse_linear = torch.mean((auto_linear_batch - reference_rgb) ** 2, dim=(1, 2, 3))
                mse_oklab = torch.mean((auto_oklab_batch - reference_rgb) ** 2, dim=(1, 2, 3))
                choose_oklab = mse_oklab + 1e-6 < mse_linear
                auto_mode_batch = ["oklab_cdf" if bool(v) else "linear" for v in choose_oklab.tolist()]
                corrected_batch = torch.where(choose_oklab[:, None, None, None], auto_oklab_batch, auto_linear_batch)

        scale_batch, offset_batch = _linear_fit_torch_batch(image_rgb, reference_rgb, match_mask_batch)
        lut_dir = Path(lut_output_dir).expanduser() if str(lut_output_dir).strip() else (Path.cwd() / "output" / "color_luts")
        lut_base = _sanitize_lut_name(lut_name)
        lut_size_int = int(max(8, min(65, int(lut_size))))

        matched_list = []
        json_list = []
        iterator = tqdm(range(batch_size), desc=f"ColorMatch[{preset}]", unit="img")
        for idx in iterator:
            ref_t = reference_rgb[idx]
            img_t = image_rgb[idx]
            mm_t = match_mask_batch[idx] if match_mask_batch is not None else None
            am_t = apply_mask_batch[idx] if apply_mask_batch is not None else None
            match_valid = bool(match_mask_valid[idx].item())

            deep_params = None
            if not match_valid:
                corrected_t = img_t
                mode = f"{preset}:empty_match_mask"
                deep_params = {
                    "warning": "empty_match_mask",
                }
            elif preset == "auto_optimal":
                if corrected_batch is not None:
                    corrected_t = corrected_batch[idx]
                    chosen = auto_mode_batch[idx]
                    mode = f"auto_optimal:{chosen}"
                    deep_params = {
                        "auto_optimal": {
                            "strategy": auto_optimal_metric,
                            "score_linear": None,
                            "score_oklab_cdf": None,
                            "selected": chosen,
                        }
                    }
                else:
                    linear_t = auto_linear_batch[idx]
                    oklab_t = auto_oklab_batch[idx]
                    linear_metrics = _auto_optimal_candidate_metrics(linear_t, ref_t, auto_optimal_metric)
                    oklab_metrics = _auto_optimal_candidate_metrics(oklab_t, ref_t, auto_optimal_metric)
                    score_linear = _auto_optimal_score(linear_metrics, auto_optimal_metric)
                    score_oklab = _auto_optimal_score(oklab_metrics, auto_optimal_metric)
                    if score_oklab + 1e-8 < score_linear:
                        corrected_t = oklab_t
                        chosen = "oklab_cdf"
                    else:
                        corrected_t = linear_t
                        chosen = "linear"
                    mode = f"auto_optimal:{chosen}"
                    deep_params = {
                        "auto_optimal": {
                            "strategy": auto_optimal_metric,
                            "linear": linear_metrics,
                            "oklab_cdf": oklab_metrics,
                            "score_linear": round(float(score_linear), 6),
                            "score_oklab_cdf": round(float(score_oklab), 6),
                            "selected": chosen,
                        }
                    }
            elif corrected_batch is not None:
                corrected_t = corrected_batch[idx]
                mode = preset
            elif preset == "perceptual_vgg_fast":
                corrected_t, deep_params = _perceptual_vgg_fast(img_t, ref_t, 5, 0.05)
                mode = "perceptual_vgg_fast"
            else:
                corrected_t = img_t
                mode = "none"

            if strength < 1.0:
                corrected_t = img_t * (1.0 - strength) + corrected_t * strength
            corrected_for_lut = corrected_t.clone()

            if match_valid:
                scale_t, offset_t = scale_batch[idx], offset_batch[idx]
            else:
                scale_t = torch.ones_like(scale_batch[idx])
                offset_t = torch.zeros_like(offset_batch[idx])
            resolve_params = {
                "scale": [round(float(s), 5) for s in scale_t],
                "offset": [round(float(o), 5) for o in offset_t],
                "gamma": [1.0, 1.0, 1.0],
            }
            fusion_params = {
                "gain": resolve_params["scale"],
                "lift": resolve_params["offset"],
                "gamma": resolve_params["gamma"],
            }

            if am_t is not None:
                mask_apply = am_t[..., None]
                corrected_t = corrected_t * mask_apply + img_t * (1.0 - mask_apply)

            corrected_t = torch.clamp(corrected_t, 0.0, 1.0)
            if compute_quality_metrics:
                metrics_before = _quality_metrics(img_t, ref_t)
                metrics_after = _quality_metrics(corrected_t, ref_t)
                improvement = _improvement_pct(metrics_before, metrics_after)
            else:
                metrics_before = _empty_quality_metrics()
                metrics_after = _empty_quality_metrics()
                improvement = _empty_quality_metrics()
            matched_t = corrected_t
            if alpha_batch is not None and preserve_alpha:
                matched_t = torch.cat([matched_t, alpha_batch[idx]], dim=-1)

            stats = {
                "ref_mean": [round(float(x), 4) for x in ref_t.reshape(-1, 3).mean(dim=0)],
                "img_mean": [round(float(x), 4) for x in img_t.reshape(-1, 3).mean(dim=0)],
                "ref_std": [round(float(x), 4) for x in ref_t.reshape(-1, 3).std(dim=0)],
                "img_std": [round(float(x), 4) for x in img_t.reshape(-1, 3).std(dim=0)],
                "mask_used": mm_t is not None,
            }
            payload = {
                "status": "ok",
                "preset": preset,
                "mode": mode,
                "resolve": resolve_params,
                "fusion": fusion_params,
                "linear": {
                    "scale": resolve_params["scale"],
                    "offset": resolve_params["offset"],
                },
                "deep": deep_params,
                "stats": stats,
                "quality": {
                    "before": metrics_before,
                    "after": metrics_after,
                    "improvement_pct": improvement,
                },
            }

            if export_lut:
                lut_payload = {"exported": False, "path": None, "size": int(lut_size_int), "method": None, "error": None}
                try:
                    lut_colors = _lut_grid_colors(lut_size_int, img_t.device, img_t.dtype)
                    if not match_valid:
                        lut_out = lut_colors
                        lut_method = "identity_empty_match_mask"
                    else:
                        lut_method = "baked_poly2"
                        exact_scale = None
                        exact_offset = None
                        if mode == "linear" or mode == "auto_optimal:linear":
                            exact_scale = scale_t
                            exact_offset = offset_t
                        elif mode == "mean_std" or mode == "adain":
                            exact_scale = mean_std_scale_batch[idx]
                            exact_offset = mean_std_offset_batch[idx]
                        if exact_scale is not None and exact_offset is not None:
                            mix_scale = (1.0 - float(strength)) + float(strength) * exact_scale
                            mix_offset = float(strength) * exact_offset
                            lut_out = torch.clamp(lut_colors * mix_scale[None, :] + mix_offset[None, :], 0.0, 1.0)
                            lut_method = "exact_linear"
                        else:
                            beta = _fit_poly2_color_map(img_t, corrected_for_lut, mm_t)
                            lut_out = _apply_poly2_color_map(lut_colors, beta)
                    file_name = f"{lut_base}.cube" if batch_size == 1 else f"{lut_base}_{idx:04d}.cube"
                    lut_path = (lut_dir / file_name).resolve()
                    _write_cube_file(lut_path, lut_out, lut_size_int, f"{lut_base}_{idx:04d}")
                    lut_payload["exported"] = True
                    lut_payload["path"] = str(lut_path)
                    lut_payload["method"] = lut_method
                except Exception as exc:
                    lut_payload["error"] = str(exc)
                payload["lut"] = lut_payload

            json_list.append(json.dumps(payload, ensure_ascii=True))
            matched_list.append(matched_t.cpu())

        return (
            torch.stack(matched_list, dim=0),
            json_list,
        )
