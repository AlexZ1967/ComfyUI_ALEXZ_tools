"""Tonal masks and detail-preserving blends for adaptive descreening."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from . import image_descreen_adaptive_ops as descreen_ops

_rgb_to_gray = descreen_ops.rgb_to_gray

def _pil_resample_rgb(image_rgb: np.ndarray, scale: float, *, pre_blur_px: float) -> np.ndarray:
    return descreen_ops.pil_resample_rgb_with_mode(image_rgb, scale, pre_blur_px=pre_blur_px, resample_mode="lanczos")


def _edge_preserve_blend(sharp_rgb: np.ndarray, smooth_rgb: np.ndarray) -> np.ndarray:
    """Blend smooth descreened result with sharper FFT-only result near real contours."""
    sharp = sharp_rgb.astype(np.float32)
    smooth = smooth_rgb.astype(np.float32)
    gray = _rgb_to_gray(sharp)
    base = np.asarray(
        Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.6)),
        dtype=np.float32,
    )
    gy, gx = np.gradient(base)
    grad = np.sqrt(gx * gx + gy * gy)
    p70 = float(np.percentile(grad, 70.0))
    p98 = float(np.percentile(grad, 98.0))
    if p98 <= p70:
        p70 = float(np.min(grad))
        p98 = float(np.max(grad))
    mask = np.clip((grad - p70) / max(1e-6, p98 - p70), 0.0, 1.0)
    mask = np.asarray(
        Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float32,
    ) / 255.0
    blended = smooth * (1.0 - mask[..., None]) + sharp * mask[..., None]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _build_transition_cleanup_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Emphasize midtone transition regions where halftone survives notch filtering most often."""
    gray = _rgb_to_gray(image_rgb.astype(np.float32))
    blur = np.asarray(
        Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.3)),
        dtype=np.float32,
    )
    gy, gx = np.gradient(blur)
    grad = np.sqrt(gx * gx + gy * gy)
    p65 = float(np.percentile(grad, 65.0))
    p98 = float(np.percentile(grad, 98.0))
    if p98 <= p65:
        p65 = float(np.min(grad))
        p98 = float(np.max(grad))
    grad_mask = np.clip((grad - p65) / max(1e-6, p98 - p65), 0.0, 1.0)
    tone = np.clip(gray / 255.0, 0.0, 1.0)
    tone_mask = np.clip(1.0 - np.abs(tone - 0.5) / 0.5, 0.0, 1.0) ** 1.5
    mask = grad_mask * tone_mask
    mask = np.asarray(
        Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float32,
    ) / 255.0
    return np.clip(mask, 0.0, 1.0)


def _transition_cleanup_blend(
    image_rgb: np.ndarray,
    *,
    scale_percent: float,
    strength: float,
    pre_blur_px: float,
) -> tuple[np.ndarray, float]:
    """Apply mild scale-descreen only inside transition zones."""
    if float(strength) <= 0.0 or float(scale_percent) >= 99.95:
        return image_rgb.copy(), 0.0
    smooth = _pil_resample_rgb(
        image_rgb,
        max(0.05, float(scale_percent) / 100.0),
        pre_blur_px=pre_blur_px,
    ).astype(np.float32)
    sharp = image_rgb.astype(np.float32)
    mask = _build_transition_cleanup_mask(image_rgb) * float(np.clip(strength, 0.0, 1.0))
    blended = sharp * (1.0 - mask[..., None]) + smooth * mask[..., None]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8), float(mask.mean())


def _build_tonal_hybrid_mask(
    image_rgb: np.ndarray,
    *,
    cleanup_strength: float,
    midtone_weight: float,
    transition_weight: float,
    shadow_protect: float,
    highlight_protect: float,
) -> np.ndarray:
    """Build a broader cleanup mask focused on midtones and tone transitions."""
    gray = _rgb_to_gray(image_rgb.astype(np.float32))
    tone = np.clip(gray / 255.0, 0.0, 1.0)
    mid_mask = np.clip(1.0 - np.abs(tone - 0.5) / 0.5, 0.0, 1.0) ** 1.1

    blur = np.asarray(
        Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.4)),
        dtype=np.float32,
    )
    gy, gx = np.gradient(blur)
    grad = np.sqrt(gx * gx + gy * gy)
    p45 = float(np.percentile(grad, 45.0))
    p95 = float(np.percentile(grad, 95.0))
    if p95 <= p45:
        p45 = float(np.min(grad))
        p95 = float(np.max(grad))
    transition_mask = np.clip((grad - p45) / max(1e-6, p95 - p45), 0.0, 1.0)
    transition_mask *= 0.35 + 0.65 * mid_mask

    base_mask = np.clip(
        float(midtone_weight) * mid_mask + float(transition_weight) * transition_mask,
        0.0,
        1.0,
    )
    shadow_hold = np.clip((0.34 - tone) / 0.34, 0.0, 1.0)
    highlight_hold = np.clip((tone - 0.72) / (1.0 - 0.72), 0.0, 1.0)
    protect = 1.0 - np.clip(float(shadow_protect) * shadow_hold + float(highlight_protect) * highlight_hold, 0.0, 1.0)
    mask = base_mask * protect * float(np.clip(cleanup_strength, 0.0, 1.0))
    mask = np.asarray(
        Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float32,
    ) / 255.0
    return np.clip(mask, 0.0, 1.0)


def _restore_large_scale_detail(
    base_rgb: np.ndarray,
    blended_rgb: np.ndarray,
    *,
    detail_restore_strength: float,
) -> np.ndarray:
    """Restore contour-scale detail without bringing back fine halftone dots aggressively."""
    if float(detail_restore_strength) <= 0.0:
        return blended_rgb.copy()
    base = base_rgb.astype(np.float32)
    blended = blended_rgb.astype(np.float32)
    base_gray = _rgb_to_gray(base)
    edge_mask = _build_transition_cleanup_mask(base_rgb)
    low = np.asarray(
        Image.fromarray(np.clip(base, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.2)),
        dtype=np.float32,
    )
    detail = base - low
    detail *= (0.25 + 0.75 * edge_mask[..., None])
    restored = blended + float(np.clip(detail_restore_strength, 0.0, 1.0)) * detail
    return np.clip(restored, 0.0, 255.0).astype(np.uint8)
