"""
Module: nodes/image_descreen_adaptive.py
Author: AlexZ1967
Last updated: 2026-03-27

Description:
    Estimate halftone screen period from an ROI, choose an adaptive
    downscale percentage to suppress visible raster, and apply a fixed
    descreen percent to a full image/batch.

Purpose:
    Provides ComfyUI nodes that analyze periodic screen artifacts in a scan,
    search candidate downscale percentages, and then apply either the chosen
    adaptive scale or a user-specified fixed scale to the full image/batch.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter


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
    pil = Image.fromarray(image_rgb, mode="RGB")
    if float(pre_blur_px) > 0.0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(pre_blur_px)))
    width, height = pil.size
    down_w = max(1, int(round(width * float(scale))))
    down_h = max(1, int(round(height * float(scale))))
    down = pil.resize((down_w, down_h), Image.Resampling.LANCZOS)
    up = down.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(up, dtype=np.uint8)


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
        "period_px": float(period_px),
        "angle_deg": float(angle_deg),
    }


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
    pil = Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8), mode="L")
    smooth = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    gy, gx = np.gradient(smooth)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _build_compare_preview(original_rgb: np.ndarray, processed_rgb: np.ndarray) -> np.ndarray:
    """Create side-by-side ROI preview for quick visual comparison."""
    h, w, _ = original_rgb.shape
    gap = np.full((h, 8, 3), 245, dtype=np.uint8)
    return np.concatenate([original_rgb, gap, processed_rgb], axis=1)


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
        roi_processed = _pil_resample_rgb(roi_rgb, scale, pre_blur_px=pre_blur_px)
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
    processed_rgb = _pil_resample_rgb(image_rgb, recommended_scale, pre_blur_px=pre_blur_px)
    compare_preview = _build_compare_preview(roi_rgb, best["roi_processed"])

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
        "candidates": candidate_table,
    }
    return processed_rgb, compare_preview, result


def _apply_fixed_percent_batch(
    image: torch.Tensor,
    *,
    scale_percent: float,
    pre_blur_px: float,
) -> torch.Tensor:
    """Apply fixed downscale/upscale descreening to the whole batch."""
    scale = float(scale_percent) / 100.0
    if scale <= 0.0:
        raise ValueError("scale_percent must be greater than 0.")
    rgb_batch = _to_rgb_batch(image)
    processed_batch = []
    for rgb in rgb_batch:
        processed_rgb = _pil_resample_rgb(rgb, scale, pre_blur_px=pre_blur_px)
        processed_batch.append(torch.from_numpy(processed_rgb.astype(np.float32) / 255.0))
    return torch.stack(processed_batch, dim=0)


class ImageDescreenAdaptiveScale:
    """Estimate and apply an adaptive downscale percentage for halftone descreening."""

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
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "roi_preview", "recommended_percent", "estimated_period_px", "analysis_json")
    FUNCTION = "descreen"
    CATEGORY = "image/restoration"

    def descreen(
        self,
        image: torch.Tensor,
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
    """Apply a known descreen percent to a full image or batch."""

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
                "pre_blur_px": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Опциональный blur перед downscale. Обычно 0.0 или очень малое значение.",
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
        pre_blur_px: float = 0.0,
    ):
        """Apply fixed descreen percent to the whole image batch."""
        processed = _apply_fixed_percent_batch(
            image,
            scale_percent=scale_percent,
            pre_blur_px=pre_blur_px,
        )
        payload = {
            "mode": "fixed_percent",
            "applied_percent": float(scale_percent),
            "applied_scale": float(scale_percent) / 100.0,
            "pre_blur_px": float(pre_blur_px),
            "batch_size": int(processed.shape[0]),
        }
        return (
            processed,
            float(scale_percent),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )
