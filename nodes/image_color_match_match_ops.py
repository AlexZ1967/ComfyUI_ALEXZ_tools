"""Color-transfer algorithms for Color Match To Reference."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..utils import color_match_utils
from ..utils.color_match_utils import normalize_mask
from .image_color_match_color_ops import (
    _fit_mvgd_affine,
    _hist_match_channel_torch,
    _interp1d_torch,
    _lab_to_rgb_torch,
    _mkl_transfer_matrix,
    _oklab_to_rgb_torch,
    _rgb_to_lab_torch,
    _rgb_to_oklab_torch,
)
from .image_color_match_metrics_ops import _perceptual_vgg_fast

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


def _hist_match_rgb_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Per-channel RGB histogram matching in torch."""
    bsz, h, w, _ = img.shape
    out = img.clone()
    keep_all = torch.ones((h, w), dtype=torch.bool, device=img.device)
    for b in range(bsz):
        keep = keep_all if mask is None else (mask[b] > 0.5)
        if keep.sum() < 10:
            continue
        for c in range(3):
            out[b, :, :, c] = _hist_match_channel_torch(
                img[b, :, :, c], ref[b, :, :, c], keep, bins=256, value_range=(0.0, 1.0)
            )
    return torch.clamp(out, 0.0, 1.0)


def _reinhard_lab_fast_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Fast Reinhard-like transfer: mean/std alignment in Lab space."""
    lab_img = _rgb_to_lab_torch(img)
    lab_ref = _rgb_to_lab_torch(ref)
    if mask is None:
        weights = torch.ones(
            (img.shape[0], img.shape[1], img.shape[2], 1),
            dtype=img.dtype,
            device=img.device,
        )
    else:
        weights = (mask > 0.5).to(dtype=img.dtype, device=img.device).unsqueeze(-1)
    counts = weights.sum(dim=(1, 2), keepdim=True)
    empty = counts <= 0.5
    if torch.any(empty):
        weights = torch.where(empty, torch.ones_like(weights), weights)
        counts = weights.sum(dim=(1, 2), keepdim=True)
    counts = counts.clamp_min(1.0)
    mean_img = (lab_img * weights).sum(dim=(1, 2), keepdim=True) / counts
    mean_ref = (lab_ref * weights).sum(dim=(1, 2), keepdim=True) / counts
    var_img = (((lab_img - mean_img) ** 2) * weights).sum(dim=(1, 2), keepdim=True) / counts
    var_ref = (((lab_ref - mean_ref) ** 2) * weights).sum(dim=(1, 2), keepdim=True) / counts
    std_img = torch.sqrt(torch.clamp(var_img, min=1e-6))
    std_ref = torch.sqrt(torch.clamp(var_ref, min=0.0))
    matched = (lab_img - mean_img) * (std_ref / std_img) + mean_ref
    matched[..., 0] = torch.clamp(matched[..., 0], 0.0, 100.0)
    matched[..., 1] = torch.clamp(matched[..., 1], -127.0, 127.0)
    matched[..., 2] = torch.clamp(matched[..., 2], -127.0, 127.0)
    return _lab_to_rgb_torch(matched)


def _covariance_transfer_batch(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: Optional[torch.Tensor],
    solver: str,
) -> torch.Tensor:
    """Apply MKL/MVGD-like 3x3 color transfer in batch mode."""
    bsz, h, w, _ = img.shape
    out = img.clone()
    keep_all = torch.ones((h, w), dtype=torch.bool, device=img.device)
    eye = torch.eye(3, device=img.device, dtype=torch.float32)
    for b in range(bsz):
        keep = keep_all if mask is None else (mask[b] > 0.5)
        if keep.sum() < 16:
            continue
        src_vals = img[b][keep].float()
        ref_vals = ref[b][keep].float()
        if src_vals.shape[0] < 16 or ref_vals.shape[0] < 16:
            continue
        flat = img[b].reshape(-1, 3).float()
        try:
            if solver == "mkl":
                src_mean = src_vals.mean(dim=0)
                ref_mean = ref_vals.mean(dim=0)
                src_centered = src_vals - src_mean
                ref_centered = ref_vals - ref_mean
                denom_src = float(max(int(src_vals.shape[0]) - 1, 1))
                denom_ref = float(max(int(ref_vals.shape[0]) - 1, 1))
                src_cov = (src_centered.t() @ src_centered) / denom_src + 1e-6 * eye
                ref_cov = (ref_centered.t() @ ref_centered) / denom_ref + 1e-6 * eye
                a = _mkl_transfer_matrix(src_cov, ref_cov, eps=1e-6)
                b_vec = ref_mean - torch.matmul(a, src_mean)
            else:
                a, b_vec = _fit_mvgd_affine(src_vals, ref_vals, ridge=1e-4)
            mapped = torch.matmul(flat, a.t()) + b_vec[None, :]
            out[b] = mapped.view(h, w, 3).to(dtype=img.dtype)
        except RuntimeError:
            continue
    return torch.clamp(out, 0.0, 1.0)


def _mkl_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """MKL-like multivariate covariance transfer."""
    return _covariance_transfer_batch(img, ref, mask, solver="mkl")


def _mvgd_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """MVGD-like affine fit in RGB space."""
    return _covariance_transfer_batch(img, ref, mask, solver="mvgd")


def _hm_mkl_hm_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Compound transfer: histogram -> MKL -> histogram."""
    stage1 = _hist_match_rgb_batch(img, ref, mask)
    stage2 = _mkl_match_batch(stage1, ref, mask)
    return _hist_match_rgb_batch(stage2, ref, mask)


def _hm_mvgd_hm_match_batch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Compound transfer: histogram -> MVGD -> histogram."""
    stage1 = _hist_match_rgb_batch(img, ref, mask)
    stage2 = _mvgd_match_batch(stage1, ref, mask)
    return _hist_match_rgb_batch(stage2, ref, mask)


def _grid_match_batch(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: Optional[torch.Tensor],
    grid: int,
    matcher_fn,
) -> torch.Tensor:
    """Apply batch matcher independently per spatial tile."""
    grid_i = int(max(1, grid))
    if grid_i <= 1:
        return matcher_fn(img, ref, mask)
    _, h, w, _ = img.shape
    out = torch.empty_like(img)
    for gy in range(grid_i):
        y0 = (gy * h) // grid_i
        y1 = ((gy + 1) * h) // grid_i
        if y1 <= y0:
            continue
        for gx in range(grid_i):
            x0 = (gx * w) // grid_i
            x1 = ((gx + 1) * w) // grid_i
            if x1 <= x0:
                continue
            img_tile = img[:, y0:y1, x0:x1, :]
            ref_tile = ref[:, y0:y1, x0:x1, :]
            mask_tile = None if mask is None else mask[:, y0:y1, x0:x1]
            out[:, y0:y1, x0:x1, :] = matcher_fn(img_tile, ref_tile, mask_tile)
    return torch.clamp(out, 0.0, 1.0)


def _run_auto_fallback_single(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: Optional[torch.Tensor],
    method: str,
    *,
    lab_cdf_fn: Optional[Callable] = None,
    oklab_cdf_fn: Optional[Callable] = None,
    perceptual_fn: Optional[Callable] = None,
) -> tuple[torch.Tensor, Optional[dict]]:
    """Run one fallback method for auto_optimal on a single frame."""
    lab_cdf_fn = lab_cdf_fn or _lab_cdf_match_batch
    oklab_cdf_fn = oklab_cdf_fn or _oklab_cdf_match_batch
    perceptual_fn = perceptual_fn or _perceptual_vgg_fast
    mask_b = None if mask is None else mask.unsqueeze(0)
    if method == "lab_cdf":
        return lab_cdf_fn(img.unsqueeze(0), ref.unsqueeze(0), mask_b)[0], None
    if method == "oklab_cdf":
        return oklab_cdf_fn(img.unsqueeze(0), ref.unsqueeze(0), mask_b)[0], None
    if method == "perceptual_vgg_fast":
        return perceptual_fn(img, ref, 5, 0.05)
    return img, None


def _mean_std_fit_torch_batch(
    img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper around shared mean/std-fit helper."""
    return color_match_utils.mean_std_fit_torch_batch(img, ref, mask)
