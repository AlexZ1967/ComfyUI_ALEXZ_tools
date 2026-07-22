"""Pure image, ROI, and resampling operations for adaptive descreen nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def to_rgb_batch(image: torch.Tensor) -> np.ndarray:
    """Convert a Comfy IMAGE tensor to a uint8 RGB batch."""
    if image.ndim != 4:
        raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")
    arr = image.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.shape[-1] < 3:
        raise ValueError("Expected IMAGE tensor with at least 3 channels.")
    return np.clip(arr[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert one uint8 RGB image to a Comfy IMAGE tensor."""
    return torch.from_numpy(image_rgb.astype(np.float32) / 255.0).unsqueeze(0)


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB pixels to float32 luma."""
    arr = image_rgb.astype(np.float32)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def clip_roi(x: int, y: int, w: int, h: int, *, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """Clip an ROI rectangle to image bounds."""
    x0 = max(0, min(int(x), image_w - 1))
    y0 = max(0, min(int(y), image_h - 1))
    x1 = max(x0 + 1, min(x0 + max(8, int(w)), image_w))
    y1 = max(y0 + 1, min(y0 + max(8, int(h)), image_h))
    return x0, y0, x1 - x0, y1 - y0


def select_roi_rect(image_w: int, image_h: int, *, roi_mode: str, roi_size_percent: float, roi_x: int, roi_y: int, roi_w: int, roi_h: int) -> tuple[int, int, int, int]:
    """Resolve full-frame, manual, or centered-square analysis ROI."""
    mode = str(roi_mode or "center_square").strip().lower()
    if mode == "full_frame":
        return 0, 0, int(image_w), int(image_h)
    if mode == "manual_rect":
        return clip_roi(roi_x, roi_y, roi_w, roi_h, image_w=image_w, image_h=image_h)
    side = min(max(32, int(round(min(image_w, image_h) * float(roi_size_percent) / 100.0))), image_w, image_h)
    return max(0, (image_w - side) // 2), max(0, (image_h - side) // 2), side, side


def resolve_resample_mode(resample_mode: str) -> Image.Resampling:
    """Map the public resample setting to a PIL enum."""
    return Image.Resampling.BICUBIC if str(resample_mode or "lanczos").strip().lower() == "bicubic" else Image.Resampling.LANCZOS


def pil_resample_rgb_with_mode(image_rgb: np.ndarray, scale: float, *, pre_blur_px: float, resample_mode: str, restore_size: bool = True) -> np.ndarray:
    """Resample RGB image, optionally restoring its original dimensions."""
    pil = Image.fromarray(image_rgb)
    if float(pre_blur_px) > 0.0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(pre_blur_px)))
    width, height = pil.size
    target = max(1, int(round(width * float(scale)))), max(1, int(round(height * float(scale))))
    result = pil.resize(target, resolve_resample_mode(resample_mode))
    if restore_size:
        result = result.resize((width, height), resolve_resample_mode(resample_mode))
    return np.asarray(result, dtype=np.uint8)


def apply_fixed_percent_downscale_batch(image: torch.Tensor, *, scale_percent: float, resample_mode: str) -> torch.Tensor:
    """Apply fixed downscale-only descreening to a complete IMAGE batch."""
    scale = float(scale_percent) / 100.0
    if scale <= 0.0:
        raise ValueError("scale_percent must be greater than 0.")
    return torch.stack([torch.from_numpy(pil_resample_rgb_with_mode(rgb, scale, pre_blur_px=0.0, resample_mode=resample_mode, restore_size=False).astype(np.float32) / 255.0) for rgb in to_rgb_batch(image)], dim=0)


def structure_energy(gray: np.ndarray) -> float:
    """Estimate retained large-scale structure using smooth gradient energy."""
    smooth = np.asarray(Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    gy, gx = np.gradient(smooth)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def predict_descreen_scale_percent(period_px: float, *, target_screen_px: float, min_scale_percent: float = 5.0, max_scale_percent: float = 100.0) -> float:
    """Predict a bounded downscale percent from a measured raster period."""
    if float(period_px) <= 0.0 or float(target_screen_px) <= 0.0:
        return 100.0
    return float(np.clip((float(target_screen_px) / float(period_px)) * 100.0, float(min_scale_percent), float(max_scale_percent)))


def build_compare_preview(*images_rgb: np.ndarray) -> np.ndarray:
    """Build a horizontal comparison strip from equally high RGB images."""
    if not images_rgb:
        raise ValueError("Expected at least one image for preview.")
    gap = np.full((int(images_rgb[0].shape[0]), 8, 3), 245, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for index, image in enumerate(images_rgb):
        if index:
            parts.append(gap)
        parts.append(image)
    return np.concatenate(parts, axis=1)


def build_scale_sheet_preview(variants: list[dict[str, Any]]) -> np.ndarray:
    """Build labelled contact sheet for fixed-scale preview variants."""
    if not variants:
        raise ValueError("Expected at least one scale variant for scale-sheet preview.")
    images = [item["image"].astype(np.uint8) for item in variants]
    margin_top, gap = 28, 10
    canvas = Image.new("RGB", (sum(int(image.shape[1]) for image in images) + gap * (len(images) - 1), max(int(image.shape[0]) for image in images) + margin_top), (245, 245, 245))
    draw, x = ImageDraw.Draw(canvas), 0
    for item, image in zip(variants, images):
        percent = float(item["percent"])
        caption = f"{int(round(percent))}%" if abs(percent - round(percent)) < 1e-6 else f"{percent:.2f}%"
        canvas.paste(Image.fromarray(image), (x, margin_top))
        draw.text((x + 4, 5), caption, fill=(30, 30, 30))
        draw.text((x + 5, 5), caption, fill=(30, 30, 30))
        x += int(image.shape[1]) + gap
    return np.asarray(canvas, dtype=np.uint8)

