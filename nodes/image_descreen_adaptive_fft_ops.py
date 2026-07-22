"""FFT analysis and notch filtering for adaptive descreening."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from . import image_descreen_adaptive_ops as descreen_ops

_rgb_to_gray = descreen_ops.rgb_to_gray


def _fft_peak(gray: np.ndarray) -> dict[str, Any]:
    """Estimate dominant screen frequency peak from grayscale ROI."""
    work = gray.astype(np.float32)
    work = (work - float(work.mean())) / (float(work.std()) + 1e-6)
    height, width = work.shape
    spectrum = np.fft.fftshift(
        np.fft.fft2(work * np.hanning(height)[:, None] * np.hanning(width)[None, :])
    )
    magnitude = np.abs(spectrum)
    cy, cx = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    magnitude[radius < max(8, int(min(height, width) * 0.02))] = 0.0
    magnitude[radius > int(min(height, width) * 0.45)] = 0.0
    py, px = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    dy, dx = int(py - cy), int(px - cx)
    if dy < 0 or (dy == 0 and dx < 0):
        dy, dx = -dy, -dx
    fy, fx = float(dy) / float(height), float(dx) / float(width)
    frequency = math.hypot(fx, fy)
    return {
        "dx": dx,
        "dy": dy,
        "fx": fx,
        "fy": fy,
        "period_px": 1.0 / max(frequency, 1e-6),
        "angle_deg": math.degrees(math.atan2(float(dy), float(dx))) if (dx or dy) else 0.0,
    }


def _fft_log_magnitude(gray: np.ndarray) -> np.ndarray:
    """Return shifted log-magnitude spectrum for grayscale ROI."""
    work = gray.astype(np.float32)
    work = (work - float(work.mean())) / (float(work.std()) + 1e-6)
    height, width = work.shape
    spectrum = np.fft.fftshift(
        np.fft.fft2(work * np.hanning(height)[:, None] * np.hanning(width)[None, :])
    )
    return np.log1p(np.abs(spectrum)).astype(np.float32)


def _screen_energy(gray: np.ndarray, dx: int, dy: int, radius: int = 3) -> float:
    """Measure residual energy around conjugate FFT peaks."""
    work = gray.astype(np.float32)
    work = (work - float(work.mean())) / (float(work.std()) + 1e-6)
    height, width = work.shape
    spectrum = np.fft.fftshift(
        np.fft.fft2(work * np.hanning(height)[:, None] * np.hanning(width)[None, :])
    )
    power = np.abs(spectrum) ** 2
    cy, cx = height // 2, width // 2
    energy = 0.0
    for py, px in ((cy + int(dy), cx + int(dx)), (cy - int(dy), cx - int(dx))):
        energy += float(
            power[
                max(0, py - radius) : min(height, py + radius + 1),
                max(0, px - radius) : min(width, px + radius + 1),
            ].sum()
        )
    return energy


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
