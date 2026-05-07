"""
Module: nodes/image_descreen_adaptive.py
Author: AlexZ1967
Last updated: 2026-03-27

Description:
    Utilities for practical scale-based halftone descreening:
    estimate raster period, build visual scale previews, run the
    legacy all-in-one adaptive node, and apply a chosen fixed scale.

Purpose:
    Provides ComfyUI nodes for pragmatic descreen workflows where the user
    measures raster period, reviews a scale sheet, and then applies a fixed
    final downscale to one image or a batch.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def _to_rgb_batch(image: torch.Tensor) -> np.ndarray:
    """Convert Comfy IMAGE tensor to uint8 RGB batch."""
    if image.ndim != 4:
        raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")
    arr = image.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.shape[-1] < 3:
        raise ValueError("Expected IMAGE tensor with at least 3 channels.")
    rgb = np.clip(arr[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb


def _to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB array [H,W,3] to Comfy IMAGE tensor [1,H,W,3]."""
    return torch.from_numpy(image_rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to float32 luma."""
    arr = image_rgb.astype(np.float32)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _clip_roi(x: int, y: int, w: int, h: int, *, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """Clip ROI rectangle to image bounds."""
    x0 = max(0, min(int(x), image_w - 1))
    y0 = max(0, min(int(y), image_h - 1))
    width = max(8, int(w))
    height = max(8, int(h))
    x1 = max(x0 + 1, min(x0 + width, image_w))
    y1 = max(y0 + 1, min(y0 + height, image_h))
    return x0, y0, x1 - x0, y1 - y0


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
    mode = str(roi_mode or "center_square").strip().lower()
    if mode == "full_frame":
        return (0, 0, int(image_w), int(image_h))
    if mode == "manual_rect":
        return _clip_roi(roi_x, roi_y, roi_w, roi_h, image_w=image_w, image_h=image_h)

    side = max(32, int(round(min(image_w, image_h) * float(roi_size_percent) / 100.0)))
    side = min(side, image_w, image_h)
    x = max(0, (image_w - side) // 2)
    y = max(0, (image_h - side) // 2)
    return (x, y, side, side)


def _pil_resample_rgb(image_rgb: np.ndarray, scale: float, *, pre_blur_px: float) -> np.ndarray:
    """Downscale/upscale RGB image with optional pre-blur."""
    return _pil_resample_rgb_with_mode(image_rgb, scale, pre_blur_px=pre_blur_px, resample_mode="lanczos")


def _resolve_resample_mode(resample_mode: str) -> Image.Resampling:
    """Map user-facing resample mode string to PIL enum."""
    mode = str(resample_mode or "lanczos").strip().lower()
    if mode == "bicubic":
        return Image.Resampling.BICUBIC
    return Image.Resampling.LANCZOS


def _pil_resample_rgb_with_mode(
    image_rgb: np.ndarray,
    scale: float,
    *,
    pre_blur_px: float,
    resample_mode: str,
) -> np.ndarray:
    """Downscale/upscale RGB image with optional pre-blur and explicit resampler."""
    pil = Image.fromarray(image_rgb)
    if float(pre_blur_px) > 0.0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(pre_blur_px)))
    resample = _resolve_resample_mode(resample_mode)
    width, height = pil.size
    down_w = max(1, int(round(width * float(scale))))
    down_h = max(1, int(round(height * float(scale))))
    down = pil.resize((down_w, down_h), resample)
    up = down.resize((width, height), resample)
    return np.asarray(up, dtype=np.uint8)


def _pil_downscale_rgb_with_mode(
    image_rgb: np.ndarray,
    scale: float,
    *,
    pre_blur_px: float,
    resample_mode: str,
) -> np.ndarray:
    """Downscale RGB image only, keeping the reduced result."""
    pil = Image.fromarray(image_rgb)
    if float(pre_blur_px) > 0.0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(pre_blur_px)))
    resample = _resolve_resample_mode(resample_mode)
    width, height = pil.size
    down_w = max(1, int(round(width * float(scale))))
    down_h = max(1, int(round(height * float(scale))))
    down = pil.resize((down_w, down_h), resample)
    return np.asarray(down, dtype=np.uint8)


def _apply_fixed_percent_downscale_batch(
    image: torch.Tensor,
    *,
    scale_percent: float,
    resample_mode: str,
) -> torch.Tensor:
    """Apply fixed downscale-only descreening to the whole batch."""
    scale = float(scale_percent) / 100.0
    if scale <= 0.0:
        raise ValueError("scale_percent must be greater than 0.")
    rgb_batch = _to_rgb_batch(image)
    processed_batch = []
    for rgb in rgb_batch:
        processed_rgb = _pil_downscale_rgb_with_mode(
            rgb,
            scale,
            pre_blur_px=0.0,
            resample_mode=resample_mode,
        )
        processed_batch.append(torch.from_numpy(processed_rgb.astype(np.float32) / 255.0))
    return torch.stack(processed_batch, dim=0)


def _fft_peak(gray: np.ndarray) -> dict[str, Any]:
    """Estimate dominant screen frequency peak from grayscale ROI."""
    a = gray.astype(np.float32)
    a = (a - float(a.mean())) / (float(a.std()) + 1e-6)
    h, w = a.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    windowed = a * wy * wx
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.abs(spectrum)
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mag[rr < max(8, int(min(h, w) * 0.02))] = 0.0
    mag[rr > int(min(h, w) * 0.45)] = 0.0

    peak_index = int(np.argmax(mag))
    peak_y, peak_x = np.unravel_index(peak_index, mag.shape)
    dy = int(peak_y - cy)
    dx = int(peak_x - cx)
    if dy < 0 or (dy == 0 and dx < 0):
        dy = -dy
        dx = -dx

    fy = float(dy) / float(h)
    fx = float(dx) / float(w)
    freq = math.hypot(fx, fy)
    period_px = 1.0 / max(freq, 1e-6)
    angle_deg = math.degrees(math.atan2(float(dy), float(dx))) if (dx or dy) else 0.0

    return {
        "dx": int(dx),
        "dy": int(dy),
        "fx": float(fx),
        "fy": float(fy),
        "period_px": float(period_px),
        "angle_deg": float(angle_deg),
    }


def _fft_log_magnitude(gray: np.ndarray) -> np.ndarray:
    """Return shifted log-magnitude spectrum for grayscale ROI."""
    a = gray.astype(np.float32)
    a = (a - float(a.mean())) / (float(a.std()) + 1e-6)
    h, w = a.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(a * wy * wx))
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def _find_fft_peaks(
    gray: np.ndarray,
    *,
    peak_count: int,
    protect_low_freq: float,
    min_period_px: float,
    max_period_px: float,
    nms_radius: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Find dominant unique FFT peaks for halftone screen detection."""
    mag = _fft_log_magnitude(gray)
    work = mag.copy()
    h, w = work.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    work[rr < max(6.0, float(min(h, w)) * float(protect_low_freq))] = 0.0
    work[rr > float(min(h, w)) * 0.48] = 0.0
    work[(yy < cy) | ((yy == cy) & (xx <= cx))] = 0.0

    peaks: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(8, int(peak_count) * 12)
    while len(peaks) < int(peak_count) and attempts < max_attempts:
        attempts += 1
        peak_index = int(np.argmax(work))
        peak_value = float(work.flat[peak_index])
        if peak_value <= 0.0:
            break
        py, px = np.unravel_index(peak_index, work.shape)
        dy = int(py - cy)
        dx = int(px - cx)
        fy = float(dy) / float(h)
        fx = float(dx) / float(w)
        freq = math.hypot(fx, fy)
        period_px = 1.0 / max(freq, 1e-6)
        if float(min_period_px) <= float(period_px) <= float(max_period_px):
            peaks.append(
                {
                    "dx": int(dx),
                    "dy": int(dy),
                    "fx": float(fx),
                    "fy": float(fy),
                    "period_px": float(period_px),
                    "angle_deg": float(math.degrees(math.atan2(float(dy), float(dx)))),
                    "magnitude": float(peak_value),
                }
            )
        for sy, sx in ((py, px), (cy - dy, cx - dx)):
            y0 = max(0, int(sy) - int(nms_radius))
            y1 = min(h, int(sy) + int(nms_radius) + 1)
            x0 = max(0, int(sx) - int(nms_radius))
            x1 = min(w, int(sx) + int(nms_radius) + 1)
            work[y0:y1, x0:x1] = 0.0
    return peaks, mag


def _build_fft_notch_mask(
    shape: tuple[int, int],
    peaks: list[dict[str, Any]],
    *,
    notch_radius: float,
    notch_strength: float,
    notch_tangent_scale: float,
) -> np.ndarray:
    """Build multiplicative FFT notch mask from detected peak frequencies."""
    h, w = int(shape[0]), int(shape[1])
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    radius = max(0.5, float(notch_radius))
    tangent_radius = max(radius, radius * max(1.0, float(notch_tangent_scale)))
    strength = float(np.clip(float(notch_strength), 0.0, 1.0))
    for peak in peaks:
        py = float(cy) + float(peak["fy"]) * float(h)
        px = float(cx) + float(peak["fx"]) * float(w)
        qy = float(cy) - float(peak["fy"]) * float(h)
        qx = float(cx) - float(peak["fx"]) * float(w)
        ux = float(peak["fx"])
        uy = float(peak["fy"])
        norm = math.hypot(ux, uy)
        if norm <= 1e-8:
            ux, uy = 1.0, 0.0
            norm = 1.0
        ux /= norm
        uy /= norm
        tx = -uy
        ty = ux
        for sy, sx in ((py, px), (qy, qx)):
            dx = xx - sx
            dy = yy - sy
            dr = dx * ux + dy * uy
            dt = dx * tx + dy * ty
            dist = (dr * dr) / (radius * radius) + (dt * dt) / (tangent_radius * tangent_radius)
            notch = 1.0 - strength * np.exp(-0.5 * dist)
            mask *= notch.astype(np.float32)
    return np.clip(mask, 0.0, 1.0)


def _expand_fft_peaks_with_harmonics(
    peaks: list[dict[str, Any]],
    *,
    harmonic_count: int,
    max_frequency: float = 0.48,
) -> list[dict[str, Any]]:
    """Expand detected base peaks with harmonic multiples."""
    if int(harmonic_count) <= 1:
        return list(peaks)
    expanded: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for peak in peaks:
        base_fx = float(peak["fx"])
        base_fy = float(peak["fy"])
        for harmonic in range(1, int(harmonic_count) + 1):
            fx = base_fx * float(harmonic)
            fy = base_fy * float(harmonic)
            freq = math.hypot(fx, fy)
            if freq <= 0.0 or freq >= float(max_frequency):
                continue
            key = (int(round(fx * 100000.0)), int(round(fy * 100000.0)))
            if key in seen:
                continue
            seen.add(key)
            item = dict(peak)
            item["fx"] = float(fx)
            item["fy"] = float(fy)
            item["period_px"] = 1.0 / max(freq, 1e-6)
            item["harmonic"] = int(harmonic)
            expanded.append(item)
    return expanded


def _apply_fft_notch_gray(
    gray: np.ndarray,
    peaks: list[dict[str, Any]],
    *,
    notch_radius: float,
    notch_strength: float,
    notch_tangent_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FFT notch filtering to one grayscale image."""
    work = gray.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(work))
    mask = _build_fft_notch_mask(
        work.shape,
        peaks,
        notch_radius=notch_radius,
        notch_strength=notch_strength,
        notch_tangent_scale=notch_tangent_scale,
    )
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return np.real(filtered).astype(np.float32), mask


def _apply_fft_notch_rgb(
    image_rgb: np.ndarray,
    peaks: list[dict[str, Any]],
    *,
    apply_mode: str,
    notch_radius: float,
    notch_strength: float,
    notch_tangent_scale: float,
    post_blur_px: float,
) -> np.ndarray:
    """Apply FFT notch filtering with color-preserving luminance modes."""
    work_rgb = image_rgb.astype(np.float32)
    mode = str(apply_mode or "log_luma_preserve_color").strip().lower()
    if mode == "rgb_channels_legacy":
        channels = []
        for idx in range(3):
            channel = work_rgb[..., idx]
            filtered, _mask = _apply_fft_notch_gray(
                channel,
                peaks,
                notch_radius=notch_radius,
                notch_strength=notch_strength,
                notch_tangent_scale=notch_tangent_scale,
            )
            channels.append(filtered)
        stacked = np.stack(channels, axis=-1)
    else:
        luma = _rgb_to_gray(work_rgb)
        if mode == "log_luma_preserve_color":
            work_plane = np.log1p(np.clip(luma, 0.0, None))
            filtered_plane, _mask = _apply_fft_notch_gray(
                work_plane,
                peaks,
                notch_radius=notch_radius,
                notch_strength=notch_strength,
                notch_tangent_scale=notch_tangent_scale,
            )
            filtered_luma = np.expm1(filtered_plane)
        else:
            filtered_luma, _mask = _apply_fft_notch_gray(
                luma,
                peaks,
                notch_radius=notch_radius,
                notch_strength=notch_strength,
                notch_tangent_scale=notch_tangent_scale,
            )
        filtered_luma = np.clip(filtered_luma, 0.0, 255.0)
        # Soft luminance ratio is more stable in near-black tones than direct division.
        soft = 8.0
        ratio = (filtered_luma + soft) / (np.maximum(luma, 0.0) + soft)
        stacked = work_rgb * ratio[..., None]
    stacked = np.clip(stacked, 0.0, 255.0).astype(np.uint8)
    if float(post_blur_px) > 0.0:
        pil = Image.fromarray(stacked)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(post_blur_px)))
        stacked = np.asarray(pil, dtype=np.uint8)
    return stacked


def _normalize_preview_plane(plane: np.ndarray) -> np.ndarray:
    """Normalize 2D float plane to uint8 grayscale preview."""
    work = plane.astype(np.float32)
    p1 = float(np.percentile(work, 1.0))
    p99 = float(np.percentile(work, 99.0))
    if p99 <= p1:
        p1, p99 = float(work.min()), float(work.max())
    norm = (work - p1) / max(1e-6, p99 - p1)
    return np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)


def _overlay_peaks_on_spectrum(spectrum_u8: np.ndarray, peaks: list[dict[str, Any]]) -> np.ndarray:
    """Draw detected peak markers over spectrum preview."""
    preview = np.repeat(spectrum_u8[..., None], 3, axis=2)
    h, w = spectrum_u8.shape
    cy, cx = h // 2, w // 2
    for peak in peaks:
        py = int(round(float(cy) + float(peak["fy"]) * float(h)))
        px = int(round(float(cx) + float(peak["fx"]) * float(w)))
        qy = int(round(float(cy) - float(peak["fy"]) * float(h)))
        qx = int(round(float(cx) - float(peak["fx"]) * float(w)))
        for sy, sx in ((py, px), (qy, qx)):
            y0 = max(0, sy - 2)
            y1 = min(h, sy + 3)
            x0 = max(0, sx - 2)
            x1 = min(w, sx + 3)
            preview[y0:y1, x0:x1, 0] = 255
            preview[y0:y1, x0:x1, 1] = 64
            preview[y0:y1, x0:x1, 2] = 64
    return preview


def _build_fft_preview(
    roi_gray: np.ndarray,
    peaks: list[dict[str, Any]],
    *,
    notch_radius: float,
    notch_strength: float,
    notch_tangent_scale: float,
) -> np.ndarray:
    """Build side-by-side FFT preview: before peaks, mask, after."""
    before = _fft_log_magnitude(roi_gray)
    filtered_gray, mask = _apply_fft_notch_gray(
        roi_gray,
        peaks,
        notch_radius=notch_radius,
        notch_strength=notch_strength,
        notch_tangent_scale=notch_tangent_scale,
    )
    after = _fft_log_magnitude(filtered_gray)
    before_u8 = _normalize_preview_plane(before)
    mask_u8 = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
    after_u8 = _normalize_preview_plane(after)
    before_rgb = _overlay_peaks_on_spectrum(before_u8, peaks)
    mask_rgb = np.repeat(mask_u8[..., None], 3, axis=2)
    after_rgb = np.repeat(after_u8[..., None], 3, axis=2)
    h = before_rgb.shape[0]
    gap = np.full((h, 8, 3), 245, dtype=np.uint8)
    return np.concatenate([before_rgb, gap, mask_rgb, gap, after_rgb], axis=1)


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


def _screen_energy(gray: np.ndarray, dx: int, dy: int, radius: int = 3) -> float:
    """Measure residual energy around conjugate FFT peaks."""
    a = gray.astype(np.float32)
    a = (a - float(a.mean())) / (float(a.std()) + 1e-6)
    h, w = a.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(a * wy * wx))
    power = np.abs(spectrum) ** 2
    cy, cx = h // 2, w // 2
    points = [
        (cy + int(dy), cx + int(dx)),
        (cy - int(dy), cx - int(dx)),
    ]
    energy = 0.0
    for py, px in points:
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)
        energy += float(power[y0:y1, x0:x1].sum())
    return energy


def _structure_energy(gray: np.ndarray) -> float:
    """Estimate useful structure retention via low-frequency gradient energy."""
    pil = Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8))
    smooth = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    gy, gx = np.gradient(smooth)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _build_compare_preview(original_rgb: np.ndarray, processed_rgb: np.ndarray) -> np.ndarray:
    """Create side-by-side ROI preview for quick visual comparison."""
    h, w, _ = original_rgb.shape
    gap = np.full((h, 8, 3), 245, dtype=np.uint8)
    return np.concatenate([original_rgb, gap, processed_rgb], axis=1)


def _build_multi_compare_preview(*images_rgb: np.ndarray) -> np.ndarray:
    """Create side-by-side preview for two or more RGB images of equal height."""
    if not images_rgb:
        raise ValueError("Expected at least one image for preview.")
    h = int(images_rgb[0].shape[0])
    gap = np.full((h, 8, 3), 245, dtype=np.uint8)
    parts = []
    for idx, item in enumerate(images_rgb):
        if idx > 0:
            parts.append(gap)
        parts.append(item)
    return np.concatenate(parts, axis=1)


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
    if not variants:
        raise ValueError("Expected at least one scale variant for scale-sheet preview.")
    images = [item["image"].astype(np.uint8) for item in variants]
    captions = [_format_percent_caption(float(item["percent"])) for item in variants]
    margin_top = 28
    gap = 10
    h = max(int(im.shape[0]) for im in images) + margin_top
    w = sum(int(im.shape[1]) for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for caption, im in zip(captions, images):
        pil_im = Image.fromarray(im.astype(np.uint8))
        canvas.paste(pil_im, (x, margin_top))
        _draw_sheet_caption(draw, x, caption)
        x += int(im.shape[1]) + gap
    return np.asarray(canvas, dtype=np.uint8)


def _predict_descreen_scale_percent(
    period_px: float,
    *,
    target_screen_px: float,
    min_scale_percent: float = 5.0,
    max_scale_percent: float = 100.0,
) -> float:
    """Predict a practical scale percent that collapses the visible raster."""
    if float(period_px) <= 0.0 or float(target_screen_px) <= 0.0:
        return 100.0
    predicted = (float(target_screen_px) / float(period_px)) * 100.0
    return float(np.clip(predicted, float(min_scale_percent), float(max_scale_percent)))


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


class ImageEstimateRasterPeriod:
    """Estimate raster period and predict a practical base downscale percent."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Скан/изображение с заметным печатным растром."}),
            },
            "optional": {
                "roi_mode": (
                    ["center_square", "full_frame", "manual_rect"],
                    {
                        "default": "center_square",
                        "tooltip": "Область анализа для оценки шага растра.",
                    },
                ),
                "roi_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального ROI в процентах от меньшей стороны кадра.",
                    },
                ),
                "roi_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата ROI для manual_rect."}),
                "roi_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата ROI для manual_rect."}),
                "roi_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина ROI для manual_rect."}),
                "roi_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота ROI для manual_rect."}),
                "target_screen_px": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Целевой остаточный шаг растра в пикселях для расчета базового процента.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("roi_preview", "estimated_period_px", "predicted_scale_percent", "analysis_json")
    FUNCTION = "estimate"
    CATEGORY = "image/restoration"

    def estimate(
        self,
        image: torch.Tensor,
        roi_mode: str = "center_square",
        roi_size_percent: float = 40.0,
        roi_x: int = 0,
        roi_y: int = 0,
        roi_w: int = 256,
        roi_h: int = 256,
        target_screen_px: float = 1.0,
    ):
        """Estimate raster period for one image or batch, returning first ROI preview."""
        rgb_batch = _to_rgb_batch(image)
        preview_first = None
        analysis = []
        for idx, rgb in enumerate(rgb_batch):
            roi_preview, result = _estimate_period_one(
                rgb,
                roi_mode=roi_mode,
                roi_size_percent=roi_size_percent,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_w=roi_w,
                roi_h=roi_h,
                target_screen_px=target_screen_px,
            )
            result["batch_index"] = int(idx)
            analysis.append(result)
            if preview_first is None:
                preview_first = _to_tensor(roi_preview)
        if preview_first is None:
            raise ValueError("Empty image batch.")
        first = analysis[0]
        payload: Any = first if len(analysis) == 1 else analysis
        return (
            preview_first,
            float(first["estimated_period_px"]),
            float(first["predicted_scale_percent"]),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


class ImageDescreenScalePreview:
    """Build a visual scale sheet from a chosen base percent and ROI."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Скан/изображение для визуального подбора scale."}),
                "base_percent": (
                    "FLOAT",
                    {
                        "default": 26.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Базовый процент, от которого строится подборочная лестница.",
                    },
                ),
            },
            "optional": {
                "resample_mode": (
                    ["lanczos", "bicubic"],
                    {
                        "default": "lanczos",
                        "tooltip": "Ресемплер для preview-вариантов.",
                    },
                ),
                "roi_mode": (
                    ["center_square", "full_frame", "manual_rect"],
                    {
                        "default": "center_square",
                        "tooltip": "Зона preview, по которой строится scale-sheet.",
                    },
                ),
                "roi_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального ROI в процентах от меньшей стороны кадра.",
                    },
                ),
                "roi_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата ROI для manual_rect."}),
                "roi_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата ROI для manual_rect."}),
                "roi_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина ROI для manual_rect."}),
                "roi_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота ROI для manual_rect."}),
                "range_up_percent": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "На сколько процентов вверх строить лестницу от base_percent.",
                    },
                ),
                "step_percent": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Шаг процентов между вариантами в scale-sheet.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scale_sheet", "analysis_json")
    FUNCTION = "preview"
    CATEGORY = "image/restoration"

    def preview(
        self,
        image: torch.Tensor,
        base_percent: float = 26.0,
        resample_mode: str = "lanczos",
        roi_mode: str = "center_square",
        roi_size_percent: float = 40.0,
        roi_x: int = 0,
        roi_y: int = 0,
        roi_w: int = 256,
        roi_h: int = 256,
        range_up_percent: float = 10.0,
        step_percent: float = 2.0,
    ):
        """Build preview for the first image in batch and return diagnostics for all."""
        rgb_batch = _to_rgb_batch(image)
        preview_first = None
        analysis = []
        for idx, rgb in enumerate(rgb_batch):
            scale_sheet, result = _build_scale_preview_one(
                rgb,
                resample_mode=resample_mode,
                roi_mode=roi_mode,
                roi_size_percent=roi_size_percent,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_w=roi_w,
                roi_h=roi_h,
                base_percent=base_percent,
                range_up_percent=range_up_percent,
                step_percent=step_percent,
            )
            result["batch_index"] = int(idx)
            analysis.append(result)
            if preview_first is None:
                preview_first = _to_tensor(scale_sheet)
        if preview_first is None:
            raise ValueError("Empty image batch.")
        payload: Any = analysis[0] if len(analysis) == 1 else analysis
        return (
            preview_first,
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


class ImageDescreenAdaptiveScale:
    """Legacy all-in-one node: estimate, preview, and apply adaptive scale at once."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Скан/изображение с заметным печатным растром."}),
            },
            "optional": {
                "resample_mode": (
                    ["lanczos", "bicubic"],
                    {
                        "default": "lanczos",
                        "tooltip": "Ресемплер для scale-descreen. На некоторых типографских сканах лучше работает Lanczos, на других Bicubic.",
                    },
                ),
                "roi_mode": (
                    ["center_square", "full_frame", "manual_rect"],
                    {
                        "default": "center_square",
                        "tooltip": "Область анализа для оценки шага растра. center_square обычно самый практичный режим.",
                    },
                ),
                "roi_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального квадратного ROI в процентах от меньшей стороны кадра.",
                    },
                ),
                "roi_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата ROI для manual_rect."}),
                "roi_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата ROI для manual_rect."}),
                "roi_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина ROI для manual_rect."}),
                "roi_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота ROI для manual_rect."}),
                "min_scale_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Нижняя граница поиска optimal scale в процентах.",
                    },
                ),
                "max_scale_percent": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Верхняя граница поиска optimal scale в процентах.",
                    },
                ),
                "step_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Шаг перебора между scale-кандидатами.",
                    },
                ),
                "target_screen_px": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Целевой остаточный размер шага растра после уменьшения. Меньше = агрессивнее подавление растра.",
                    },
                ),
                "detail_weight": (
                    "FLOAT",
                    {
                        "default": 1.25,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Вес штрафа за потерю полезной структуры. Больше = больше сохраняем детали, меньше = агрессивнее убираем растр.",
                    },
                ),
                "pre_blur_px": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Опциональный blur перед downscale. По умолчанию 0.0, т.к. часто лучший результат на сканах достигается без blur.",
                    },
                ),
                "sheet_zone_mode": (
                    ["analysis_roi", "center_square", "full_frame", "manual_rect"],
                    {
                        "default": "analysis_roi",
                        "tooltip": "Отдельная зона для preview scale-sheet. analysis_roi использует тот же ROI, что и анализ; manual_rect задается параметрами ниже.",
                    },
                ),
                "sheet_zone_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального квадрата для scale-sheet, если sheet_zone_mode=center_square.",
                    },
                ),
                "sheet_zone_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата зоны scale-sheet для manual_rect."}),
                "sheet_zone_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата зоны scale-sheet для manual_rect."}),
                "sheet_zone_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина зоны scale-sheet для manual_rect."}),
                "sheet_zone_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота зоны scale-sheet для manual_rect."}),
                "sheet_range_up_percent": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "На сколько процентов вверх от расчетного базового scale строить подборочную лестницу в preview sheet.",
                    },
                ),
                "sheet_step_percent": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Шаг процентов между вариантами в preview sheet.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "scale_sheet", "recommended_percent", "estimated_period_px", "analysis_json")
    FUNCTION = "descreen"
    CATEGORY = "image/restoration"

    def descreen(
        self,
        image: torch.Tensor,
        resample_mode: str = "lanczos",
        roi_mode: str = "center_square",
        roi_size_percent: float = 40.0,
        roi_x: int = 0,
        roi_y: int = 0,
        roi_w: int = 256,
        roi_h: int = 256,
        min_scale_percent: float = 8.0,
        max_scale_percent: float = 20.0,
        step_percent: float = 1.0,
        target_screen_px: float = 1.0,
        detail_weight: float = 1.25,
        pre_blur_px: float = 0.0,
        sheet_zone_mode: str = "analysis_roi",
        sheet_zone_size_percent: float = 40.0,
        sheet_zone_x: int = 0,
        sheet_zone_y: int = 0,
        sheet_zone_w: int = 256,
        sheet_zone_h: int = 256,
        sheet_range_up_percent: float = 10.0,
        sheet_step_percent: float = 2.0,
    ):
        """Estimate halftone period, choose optimal scale, and process the image."""
        rgb_batch = _to_rgb_batch(image)
        processed_batch = []
        preview_first = None
        analysis = []

        for idx, rgb in enumerate(rgb_batch):
            processed_rgb, compare_preview, result = _analyze_one(
                rgb,
                roi_mode=roi_mode,
                roi_size_percent=roi_size_percent,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_w=roi_w,
                roi_h=roi_h,
                min_scale_percent=min_scale_percent,
                max_scale_percent=max_scale_percent,
                step_percent=step_percent,
                target_screen_px=target_screen_px,
                detail_weight=detail_weight,
                pre_blur_px=pre_blur_px,
                resample_mode=resample_mode,
                sheet_zone_mode=sheet_zone_mode,
                sheet_zone_size_percent=sheet_zone_size_percent,
                sheet_zone_x=sheet_zone_x,
                sheet_zone_y=sheet_zone_y,
                sheet_zone_w=sheet_zone_w,
                sheet_zone_h=sheet_zone_h,
                sheet_range_up_percent=sheet_range_up_percent,
                sheet_step_percent=sheet_step_percent,
            )
            result["batch_index"] = int(idx)
            analysis.append(result)
            processed_batch.append(torch.from_numpy(processed_rgb.astype(np.float32) / 255.0))
            if preview_first is None:
                preview_first = _to_tensor(compare_preview)

        if preview_first is None:
            raise ValueError("Empty image batch.")

        processed_tensor = torch.stack(processed_batch, dim=0)
        first = analysis[0]
        payload: Any = analysis[0] if len(analysis) == 1 else analysis
        return (
            processed_tensor,
            preview_first,
            float(first["recommended_percent"]),
            float(first["estimated_period_px"]),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


class ImageDescreenApplyPercent:
    """Apply a known descreen percent as final downscale-only output."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Изображение или батч, к которому нужно применить уже найденный descreen percent."}),
                "scale_percent": (
                    "FLOAT",
                    {
                        "default": 13.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Готовый descreen percent. Удобно подключать recommended_percent из Descreen By Adaptive Scale.",
                    },
                ),
            },
            "optional": {
                "resample_mode": (
                    ["lanczos", "bicubic"],
                    {
                        "default": "lanczos",
                        "tooltip": "Ресемплер для уменьшения итоговой картинки. Lanczos на части сканов держит форму лучше, Bicubic может вести себя мягче.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "applied_percent", "analysis_json")
    FUNCTION = "apply"
    CATEGORY = "image/restoration"

    def apply(
        self,
        image: torch.Tensor,
        scale_percent: float = 13.0,
        resample_mode: str = "lanczos",
    ):
        """Apply fixed downscale-only descreen percent to the whole image batch."""
        processed = _apply_fixed_percent_downscale_batch(
            image,
            scale_percent=scale_percent,
            resample_mode=resample_mode,
        )
        payload = {
            "mode": "fixed_percent_downscale_only",
            "applied_percent": float(scale_percent),
            "applied_scale": float(scale_percent) / 100.0,
            "resample_mode": str(resample_mode),
            "batch_size": int(processed.shape[0]),
            "return_to_original_size": False,
        }
        return (
            processed,
            float(scale_percent),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )
