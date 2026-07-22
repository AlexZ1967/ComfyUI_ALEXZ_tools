"""Algorithms and orchestration shared by adaptive descreen nodes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import image_descreen_adaptive_ops as descreen_ops
from .image_descreen_adaptive_tonal_ops import (
    _build_tonal_hybrid_mask,
    _edge_preserve_blend,
    _restore_large_scale_detail,
    _transition_cleanup_blend,
)
from .image_descreen_adaptive_fft_ops import (
    _apply_fft_notch_rgb,
    _build_fft_preview,
    _expand_fft_peaks_with_harmonics,
    _fft_peak,
    _find_fft_peaks,
    _screen_energy,
)


def _to_rgb_batch(image: torch.Tensor) -> np.ndarray:
    """Convert Comfy IMAGE tensor to uint8 RGB batch."""
    return descreen_ops.to_rgb_batch(image)


def _to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB array [H,W,3] to Comfy IMAGE tensor [1,H,W,3]."""
    return descreen_ops.to_tensor(image_rgb)


def _rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to float32 luma."""
    return descreen_ops.rgb_to_gray(image_rgb)


def _clip_roi(x: int, y: int, w: int, h: int, *, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """Clip ROI rectangle to image bounds."""
    return descreen_ops.clip_roi(x, y, w, h, image_w=image_w, image_h=image_h)


def _select_roi_rect(
    image_w: int,
    image_h: int,
    *,
    roi_mode: str,
    roi_size_percent: float,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
) -> tuple[int, int, int, int]:
    """Resolve analysis ROI rectangle."""
    return descreen_ops.select_roi_rect(image_w, image_h, roi_mode=roi_mode, roi_size_percent=roi_size_percent, roi_x=roi_x, roi_y=roi_y, roi_w=roi_w, roi_h=roi_h)


def _pil_resample_rgb(image_rgb: np.ndarray, scale: float, *, pre_blur_px: float) -> np.ndarray:
    """Downscale/upscale RGB image with optional pre-blur."""
    return _pil_resample_rgb_with_mode(image_rgb, scale, pre_blur_px=pre_blur_px, resample_mode="lanczos")


def _resolve_resample_mode(resample_mode: str) -> Image.Resampling:
    """Map user-facing resample mode string to PIL enum."""
    return descreen_ops.resolve_resample_mode(resample_mode)


def _pil_resample_rgb_with_mode(
    image_rgb: np.ndarray,
    scale: float,
    *,
    pre_blur_px: float,
    resample_mode: str,
) -> np.ndarray:
    """Downscale/upscale RGB image with optional pre-blur and explicit resampler."""
    return descreen_ops.pil_resample_rgb_with_mode(image_rgb, scale, pre_blur_px=pre_blur_px, resample_mode=resample_mode)


def _pil_downscale_rgb_with_mode(
    image_rgb: np.ndarray,
    scale: float,
    *,
    pre_blur_px: float,
    resample_mode: str,
) -> np.ndarray:
    """Downscale RGB image only, keeping the reduced result."""
    return descreen_ops.pil_resample_rgb_with_mode(image_rgb, scale, pre_blur_px=pre_blur_px, resample_mode=resample_mode, restore_size=False)


def _apply_fixed_percent_downscale_batch(
    image: torch.Tensor,
    *,
    scale_percent: float,
    resample_mode: str,
) -> torch.Tensor:
    """Apply fixed downscale-only descreening to the whole batch."""
    return descreen_ops.apply_fixed_percent_downscale_batch(image, scale_percent=scale_percent, resample_mode=resample_mode)






















def _fft_notch_one(
    image_rgb: np.ndarray,
    *,
    apply_mode: str,
    roi_mode: str,
    roi_size_percent: float,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    auto_peak_count: int,
    harmonic_count: int,
    notch_radius: float,
    notch_strength: float,
    notch_tangent_scale: float,
    protect_low_freq: float,
    min_period_px: float,
    max_period_px: float,
    nms_radius: int,
    post_blur_px: float,
    transition_cleanup_target_px: float,
    transition_cleanup_strength: float,
    transition_cleanup_pre_blur_px: float,
    post_scale_target_px: float,
    post_scale_pre_blur_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply FFT notch descreening to one image and return diagnostics."""
    h, w, _ = image_rgb.shape
    rx, ry, rw, rh = _select_roi_rect(
        w,
        h,
        roi_mode=roi_mode,
        roi_size_percent=roi_size_percent,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
    )
    roi_rgb = image_rgb[ry : ry + rh, rx : rx + rw]
    roi_gray = _rgb_to_gray(roi_rgb)
    peaks, _mag = _find_fft_peaks(
        roi_gray,
        peak_count=auto_peak_count,
        protect_low_freq=protect_low_freq,
        min_period_px=min_period_px,
        max_period_px=max_period_px,
        nms_radius=nms_radius,
    )
    if not peaks:
        preview = _build_compare_preview(roi_rgb, roi_rgb)
        result = {
            "mode": "fft_notch",
            "roi": {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)},
            "apply_mode": str(apply_mode),
            "peaks": [],
            "peak_count_found": 0,
            "estimated_period_px": None,
            "notch_radius": float(notch_radius),
            "notch_strength": float(notch_strength),
            "notch_tangent_scale": float(notch_tangent_scale),
            "protect_low_freq": float(protect_low_freq),
            "post_blur_px": float(post_blur_px),
            "transition_cleanup_target_px": float(transition_cleanup_target_px),
            "transition_cleanup_strength": float(transition_cleanup_strength),
            "transition_cleanup_pre_blur_px": float(transition_cleanup_pre_blur_px),
            "transition_cleanup_scale_percent": 100.0,
            "transition_cleanup_mask_mean": 0.0,
            "post_scale_target_px": float(post_scale_target_px),
            "post_scale_pre_blur_px": float(post_scale_pre_blur_px),
            "post_scale_percent": 100.0,
            "warning": "No FFT peaks were detected in the selected ROI.",
        }
        return image_rgb.copy(), preview, result

    base_peaks = list(peaks)
    peaks = _expand_fft_peaks_with_harmonics(
        peaks,
        harmonic_count=harmonic_count,
    )
    primary_peak = base_peaks[0]
    processed_rgb = _apply_fft_notch_rgb(
        image_rgb,
        peaks,
        apply_mode=apply_mode,
        notch_radius=notch_radius,
        notch_strength=notch_strength,
        notch_tangent_scale=notch_tangent_scale,
        post_blur_px=post_blur_px,
    )
    fft_only_rgb = processed_rgb.copy()
    transition_cleanup_scale_percent = 100.0
    transition_cleanup_mask_mean = 0.0
    if float(transition_cleanup_target_px) > 0.0 and float(transition_cleanup_strength) > 0.0:
        transition_cleanup_scale_percent = _predict_descreen_scale_percent(
            float(primary_peak["period_px"]),
            target_screen_px=transition_cleanup_target_px,
        )
        processed_rgb, transition_cleanup_mask_mean = _transition_cleanup_blend(
            processed_rgb,
            scale_percent=transition_cleanup_scale_percent,
            strength=transition_cleanup_strength,
            pre_blur_px=transition_cleanup_pre_blur_px,
        )
    post_scale_percent = 100.0
    if float(post_scale_target_px) > 0.0:
        post_scale_percent = _predict_descreen_scale_percent(
            float(primary_peak["period_px"]),
            target_screen_px=post_scale_target_px,
        )
        if post_scale_percent < 99.95:
            processed_rgb = _pil_resample_rgb(
                processed_rgb,
                post_scale_percent / 100.0,
                pre_blur_px=post_scale_pre_blur_px,
            )
            processed_rgb = _edge_preserve_blend(fft_only_rgb, processed_rgb)
    preview = _build_fft_preview(
        roi_gray,
        peaks,
        notch_radius=notch_radius,
        notch_strength=notch_strength,
        notch_tangent_scale=notch_tangent_scale,
    )
    result = {
        "mode": "fft_notch",
        "roi": {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)},
        "apply_mode": str(apply_mode),
        "peak_count_found": int(len(peaks)),
        "estimated_period_px": float(primary_peak["period_px"]),
        "screen_angle_deg": float(primary_peak["angle_deg"]),
        "base_peak_count_found": int(len(base_peaks)),
        "notch_radius": float(notch_radius),
        "notch_strength": float(notch_strength),
        "notch_tangent_scale": float(notch_tangent_scale),
        "harmonic_count": int(harmonic_count),
        "protect_low_freq": float(protect_low_freq),
        "min_period_px": float(min_period_px),
        "max_period_px": float(max_period_px),
        "nms_radius": int(nms_radius),
        "post_blur_px": float(post_blur_px),
        "transition_cleanup_target_px": float(transition_cleanup_target_px),
        "transition_cleanup_strength": float(transition_cleanup_strength),
        "transition_cleanup_pre_blur_px": float(transition_cleanup_pre_blur_px),
        "transition_cleanup_scale_percent": float(transition_cleanup_scale_percent),
        "transition_cleanup_mask_mean": float(transition_cleanup_mask_mean),
        "post_scale_target_px": float(post_scale_target_px),
        "post_scale_pre_blur_px": float(post_scale_pre_blur_px),
        "post_scale_percent": float(post_scale_percent),
        "post_scale_edge_preserve": bool(float(post_scale_target_px) > 0.0 and post_scale_percent < 99.95),
        "peaks": [
            {
                "dx": int(item["dx"]),
                "dy": int(item["dy"]),
                "fx": float(item["fx"]),
                "fy": float(item["fy"]),
                "period_px": float(item["period_px"]),
                "angle_deg": float(item["angle_deg"]),
                "magnitude": float(item["magnitude"]),
            }
            for item in peaks
        ],
    }
    return processed_rgb, preview, result


def _structure_energy(gray: np.ndarray) -> float:
    """Estimate useful structure retention via low-frequency gradient energy."""
    return descreen_ops.structure_energy(gray)


def _build_compare_preview(original_rgb: np.ndarray, processed_rgb: np.ndarray) -> np.ndarray:
    """Create side-by-side ROI preview for quick visual comparison."""
    return descreen_ops.build_compare_preview(original_rgb, processed_rgb)


def _build_multi_compare_preview(*images_rgb: np.ndarray) -> np.ndarray:
    """Create side-by-side preview for two or more RGB images of equal height."""
    return descreen_ops.build_compare_preview(*images_rgb)


def _format_percent_caption(percent: float) -> str:
    """Format scale percent compactly for labels."""
    if abs(float(percent) - round(float(percent))) < 1e-6:
        return f"{int(round(float(percent)))}%"
    return f"{float(percent):.2f}%"


def _draw_sheet_caption(draw: ImageDraw.ImageDraw, x: int, caption: str) -> None:
    """Draw a slightly enlarged caption without requiring external fonts."""
    base_y = 5
    draw.text((x + 4, base_y), caption, fill=(30, 30, 30))
    # Double-pass offset keeps labels readable while staying compact.
    draw.text((x + 5, base_y), caption, fill=(30, 30, 30))


def _build_scale_sheet_preview(variants: list[dict[str, Any]]) -> np.ndarray:
    """Build contact sheet from scale variants with percent labels."""
    return descreen_ops.build_scale_sheet_preview(variants)


def _predict_descreen_scale_percent(
    period_px: float,
    *,
    target_screen_px: float,
    min_scale_percent: float = 5.0,
    max_scale_percent: float = 100.0,
) -> float:
    """Predict a practical scale percent that collapses the visible raster."""
    return descreen_ops.predict_descreen_scale_percent(period_px, target_screen_px=target_screen_px, min_scale_percent=min_scale_percent, max_scale_percent=max_scale_percent)












def _tonal_hybrid_one(
    image_rgb: np.ndarray,
    *,
    apply_mode: str,
    roi_mode: str,
    roi_size_percent: float,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    auto_peak_count: int,
    harmonic_count: int,
    notch_radius: float,
    notch_tangent_scale: float,
    notch_strength: float,
    protect_low_freq: float,
    min_period_px: float,
    max_period_px: float,
    nms_radius: int,
    post_blur_px: float,
    target_screen_px: float,
    cleanup_strength: float,
    midtone_weight: float,
    transition_weight: float,
    shadow_protect: float,
    highlight_protect: float,
    detail_restore_strength: float,
    pre_blur_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hybrid descreen tuned for tone transitions and midtones."""
    fft_rgb, _preview, base_result = _fft_notch_one(
        image_rgb,
        apply_mode=apply_mode,
        roi_mode=roi_mode,
        roi_size_percent=roi_size_percent,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
        auto_peak_count=auto_peak_count,
        harmonic_count=harmonic_count,
        notch_radius=notch_radius,
        notch_strength=notch_strength,
        notch_tangent_scale=notch_tangent_scale,
        protect_low_freq=protect_low_freq,
        min_period_px=min_period_px,
        max_period_px=max_period_px,
        nms_radius=nms_radius,
        post_blur_px=post_blur_px,
        transition_cleanup_target_px=0.0,
        transition_cleanup_strength=0.0,
        transition_cleanup_pre_blur_px=0.0,
        post_scale_target_px=0.0,
        post_scale_pre_blur_px=0.0,
    )
    period_px = float(base_result["estimated_period_px"]) if base_result.get("estimated_period_px") else 0.0
    scale_percent = _predict_descreen_scale_percent(period_px, target_screen_px=target_screen_px)
    smooth_rgb = _pil_resample_rgb(
        fft_rgb,
        scale_percent / 100.0,
        pre_blur_px=pre_blur_px,
    )
    cleanup_mask = _build_tonal_hybrid_mask(
        fft_rgb,
        cleanup_strength=cleanup_strength,
        midtone_weight=midtone_weight,
        transition_weight=transition_weight,
        shadow_protect=shadow_protect,
        highlight_protect=highlight_protect,
    )
    blended = fft_rgb.astype(np.float32) * (1.0 - cleanup_mask[..., None]) + smooth_rgb.astype(np.float32) * cleanup_mask[..., None]
    blended_rgb = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    final_rgb = _restore_large_scale_detail(
        fft_rgb,
        blended_rgb,
        detail_restore_strength=detail_restore_strength,
    )

    rx = int(base_result["roi"]["x"])
    ry = int(base_result["roi"]["y"])
    rw = int(base_result["roi"]["w"])
    rh = int(base_result["roi"]["h"])
    roi_original = image_rgb[ry : ry + rh, rx : rx + rw]
    roi_fft = fft_rgb[ry : ry + rh, rx : rx + rw]
    roi_final = final_rgb[ry : ry + rh, rx : rx + rw]
    preview = _build_multi_compare_preview(roi_original, roi_fft, roi_final)

    result = dict(base_result)
    result["mode"] = "tonal_hybrid"
    result["target_screen_px"] = float(target_screen_px)
    result["scale_percent"] = float(scale_percent)
    result["cleanup_strength"] = float(cleanup_strength)
    result["midtone_weight"] = float(midtone_weight)
    result["transition_weight"] = float(transition_weight)
    result["shadow_protect"] = float(shadow_protect)
    result["highlight_protect"] = float(highlight_protect)
    result["detail_restore_strength"] = float(detail_restore_strength)
    result["pre_blur_px"] = float(pre_blur_px)
    result["cleanup_mask_mean"] = float(cleanup_mask.mean())
    return final_rgb, preview, result


def _analyze_one(
    image_rgb: np.ndarray,
    *,
    roi_mode: str,
    roi_size_percent: float,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    min_scale_percent: float,
    max_scale_percent: float,
    step_percent: float,
    target_screen_px: float,
    detail_weight: float,
    pre_blur_px: float,
    resample_mode: str,
    sheet_zone_mode: str,
    sheet_zone_size_percent: float,
    sheet_zone_x: int,
    sheet_zone_y: int,
    sheet_zone_w: int,
    sheet_zone_h: int,
    sheet_range_up_percent: float,
    sheet_step_percent: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Analyze one image and return processed image, compare preview, and metrics."""
    h, w, _ = image_rgb.shape
    rx, ry, rw, rh = _select_roi_rect(
        w,
        h,
        roi_mode=roi_mode,
        roi_size_percent=roi_size_percent,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
    )
    roi_rgb = image_rgb[ry : ry + rh, rx : rx + rw]
    roi_gray = _rgb_to_gray(roi_rgb)
    peak = _fft_peak(roi_gray)
    dx = int(peak["dx"])
    dy = int(peak["dy"])
    period_px = float(peak["period_px"])

    orig_screen = max(1e-6, _screen_energy(roi_gray, dx, dy))
    orig_structure = max(1e-6, _structure_energy(roi_gray))
    predicted_percent = float(np.clip((float(target_screen_px) / max(period_px, 1e-6)) * 100.0, min_scale_percent, max_scale_percent))

    candidates = []
    current = float(min_scale_percent)
    while current <= float(max_scale_percent) + 1e-6:
        scale = current / 100.0
        roi_processed = _pil_resample_rgb_with_mode(
            roi_rgb,
            scale,
            pre_blur_px=pre_blur_px,
            resample_mode=resample_mode,
        )
        roi_proc_gray = _rgb_to_gray(roi_processed)
        screen_ratio = _screen_energy(roi_proc_gray, dx, dy) / orig_screen
        detail_ratio = _structure_energy(roi_proc_gray) / orig_structure
        detail_loss = max(0.0, 1.0 - float(detail_ratio))
        score = float(screen_ratio) + float(detail_weight) * float(detail_loss)
        candidates.append(
            {
                "percent": round(current, 4),
                "screen_ratio": float(screen_ratio),
                "detail_ratio": float(detail_ratio),
                "detail_loss": float(detail_loss),
                "score": float(score),
                "roi_processed": roi_processed,
            }
        )
        current += float(step_percent)

    best = min(candidates, key=lambda item: item["score"])
    recommended_percent = float(best["percent"])
    recommended_scale = recommended_percent / 100.0
    processed_rgb = _pil_resample_rgb_with_mode(
        image_rgb,
        recommended_scale,
        pre_blur_px=pre_blur_px,
        resample_mode=resample_mode,
    )

    sheet_mode = str(sheet_zone_mode or "analysis_roi").strip().lower()
    if sheet_mode == "analysis_roi":
        sx, sy, sw, sh = rx, ry, rw, rh
    else:
        lookup_mode = "center_square" if sheet_mode == "sheet_center_square" else sheet_mode
        sx, sy, sw, sh = _select_roi_rect(
            w,
            h,
            roi_mode=lookup_mode,
            roi_size_percent=sheet_zone_size_percent,
            roi_x=sheet_zone_x,
            roi_y=sheet_zone_y,
            roi_w=sheet_zone_w,
            roi_h=sheet_zone_h,
        )
    sheet_rgb = image_rgb[sy : sy + sh, sx : sx + sw]

    sheet_step = max(0.1, float(sheet_step_percent))
    base_percent = math.floor(float(predicted_percent) / sheet_step) * sheet_step
    base_percent = max(sheet_step, float(base_percent))
    max_sheet_percent = float(base_percent) + max(0.0, float(sheet_range_up_percent))
    sheet_variants: list[dict[str, Any]] = []
    current_sheet = float(base_percent)
    while current_sheet <= max_sheet_percent + 1e-6:
        roi_scaled = _pil_downscale_rgb_with_mode(
            sheet_rgb,
            current_sheet / 100.0,
            pre_blur_px=pre_blur_px,
            resample_mode=resample_mode,
        )
        sheet_variants.append(
            {
                "percent": round(current_sheet, 4),
                "image": roi_scaled,
            }
        )
        current_sheet += sheet_step
    compare_preview = _build_scale_sheet_preview(sheet_variants)

    candidate_table = []
    for item in candidates:
        candidate_table.append(
            {
                "percent": float(item["percent"]),
                "screen_ratio": float(item["screen_ratio"]),
                "detail_ratio": float(item["detail_ratio"]),
                "detail_loss": float(item["detail_loss"]),
                "score": float(item["score"]),
            }
        )

    result = {
        "recommended_percent": float(recommended_percent),
        "recommended_scale": float(recommended_scale),
        "predicted_percent": float(predicted_percent),
        "estimated_period_px": float(period_px),
        "screen_angle_deg": float(peak["angle_deg"]),
        "roi": {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)},
        "target_screen_px": float(target_screen_px),
        "detail_weight": float(detail_weight),
        "pre_blur_px": float(pre_blur_px),
        "resample_mode": str(resample_mode),
        "sheet_zone": {"x": int(sx), "y": int(sy), "w": int(sw), "h": int(sh)},
        "sheet_zone_mode": str(sheet_mode),
        "sheet_base_percent": float(base_percent),
        "sheet_range_up_percent": float(sheet_range_up_percent),
        "sheet_step_percent": float(sheet_step),
        "sheet_scales": [float(item["percent"]) for item in sheet_variants],
        "candidates": candidate_table,
    }
    return processed_rgb, compare_preview, result


def _estimate_period_one(
    image_rgb: np.ndarray,
    *,
    roi_mode: str,
    roi_size_percent: float,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    target_screen_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate raster period and predicted base scale from one ROI."""
    h, w, _ = image_rgb.shape
    rx, ry, rw, rh = _select_roi_rect(
        w,
        h,
        roi_mode=roi_mode,
        roi_size_percent=roi_size_percent,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
    )
    roi_rgb = image_rgb[ry : ry + rh, rx : rx + rw]
    peak = _fft_peak(_rgb_to_gray(roi_rgb))
    predicted_percent = _predict_descreen_scale_percent(
        float(peak["period_px"]),
        target_screen_px=target_screen_px,
    )
    result = {
        "roi": {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)},
        "estimated_period_px": float(peak["period_px"]),
        "screen_angle_deg": float(peak["angle_deg"]),
        "target_screen_px": float(target_screen_px),
        "predicted_scale_percent": float(predicted_percent),
    }
    return roi_rgb, result


def _build_scale_preview_one(
    image_rgb: np.ndarray,
    *,
    resample_mode: str,
    roi_mode: str,
    roi_size_percent: float,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    base_percent: float,
    range_up_percent: float,
    step_percent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a scale sheet from one ROI using a fixed base percent."""
    h, w, _ = image_rgb.shape
    rx, ry, rw, rh = _select_roi_rect(
        w,
        h,
        roi_mode=roi_mode,
        roi_size_percent=roi_size_percent,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
    )
    roi_rgb = image_rgb[ry : ry + rh, rx : rx + rw]
    step = max(0.1, float(step_percent))
    current = max(step, float(base_percent))
    max_percent = current + max(0.0, float(range_up_percent))
    variants: list[dict[str, Any]] = []
    while current <= max_percent + 1e-6:
        scaled = _pil_downscale_rgb_with_mode(
            roi_rgb,
            current / 100.0,
            pre_blur_px=0.0,
            resample_mode=resample_mode,
        )
        variants.append({"percent": round(current, 4), "image": scaled})
        current += step
    preview = _build_scale_sheet_preview(variants)
    result = {
        "roi": {"x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)},
        "resample_mode": str(resample_mode),
        "base_percent": float(base_percent),
        "range_up_percent": float(range_up_percent),
        "step_percent": float(step),
        "sheet_scales": [float(item["percent"]) for item in variants],
    }
    return preview, result
