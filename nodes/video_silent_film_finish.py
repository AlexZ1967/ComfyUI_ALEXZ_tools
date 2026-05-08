"""
Module: nodes/video_silent_film_finish.py
Author: AlexZ1967
Last updated: 2026-05-08

Description:
    Silent-film finish node implementation.

Purpose:
    Applies projection-era and print-era artifacts such as monochrome toning,
    global flicker, gate weave, softness, and grain to a video frame batch.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F


def _validate_video_batch(images: torch.Tensor) -> torch.Tensor:
    """Validate Comfy image batch contract and clamp to normalized range."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("image must be a torch.Tensor batch in THWC layout.")
    if images.dim() != 4:
        raise ValueError(f"Expected image batch in THWC layout, got shape={tuple(images.shape)}")
    if images.size(0) < 1:
        raise ValueError("Silent film finish requires at least one frame.")
    if images.size(-1) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got {images.size(-1)}.")
    return images.detach().float().clamp(0.0, 1.0)


def _temporal_signal(
    length: int,
    seed: int,
    scale: float,
    smoothing: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a deterministic smoothed zero-mean temporal signal."""
    if length <= 0 or scale <= 1e-8:
        return torch.zeros((max(0, length),), device=device, dtype=dtype)
    smoothing = max(0.0, min(0.995, float(smoothing)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(length, generator=generator, dtype=torch.float32)
    out = torch.zeros(length, dtype=torch.float32)
    prev = 0.0
    drive = max(0.02, 1.0 - smoothing)
    for idx in range(length):
        prev = prev * smoothing + float(noise[idx].item()) * drive
        out[idx] = prev
    out = out - out.mean()
    std = float(out.std(unbiased=False).item())
    if std > 1e-8:
        out = out / std
    out = out * float(scale)
    return out.to(device=device, dtype=dtype)


def _soft_clip_highlights(gray: torch.Tensor, rolloff: float) -> torch.Tensor:
    """Compress highlights above a soft threshold."""
    rolloff = max(0.0, float(rolloff))
    if rolloff <= 1e-8:
        return gray
    threshold = 0.68
    upper = torch.clamp(gray - threshold, min=0.0)
    denom = 1.0 + upper * (3.5 * rolloff) / max(1e-6, 1.0 - threshold)
    compressed = threshold + upper / denom
    return torch.where(gray > threshold, compressed, gray)


def _apply_gate_weave(
    images: torch.Tensor,
    shift_x_px: torch.Tensor,
    shift_y_px: torch.Tensor,
) -> torch.Tensor:
    """Apply per-frame translational gate weave in pixels."""
    if images.numel() == 0:
        return images
    frame_count, height, width, channels = images.shape
    batch = images.permute(0, 3, 1, 2).contiguous()
    theta = torch.zeros((frame_count, 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift_x_px / max(1.0, float(width))
    theta[:, 1, 2] = 2.0 * shift_y_px / max(1.0, float(height))
    grid = F.affine_grid(theta, batch.size(), align_corners=False)
    woven = F.grid_sample(
        batch,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return woven.permute(0, 2, 3, 1).contiguous().view(frame_count, height, width, channels)


def _build_blur_levels(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build two blur levels: a soft optical blur and a slightly stronger defocus blur."""
    batch = gray.permute(0, 3, 1, 2).contiguous()
    soft_blur = F.avg_pool2d(batch, kernel_size=3, stride=1, padding=1)
    soft_blur = F.avg_pool2d(soft_blur, kernel_size=3, stride=1, padding=1)

    defocus_blur = F.avg_pool2d(soft_blur, kernel_size=5, stride=1, padding=2)
    downscaled = F.interpolate(soft_blur, scale_factor=0.5, mode="bilinear", align_corners=False)
    downscaled = F.avg_pool2d(downscaled, kernel_size=3, stride=1, padding=1)
    downscaled = F.interpolate(
        downscaled,
        size=batch.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    defocus_blur = defocus_blur * 0.6 + downscaled * 0.4
    return (
        soft_blur.permute(0, 2, 3, 1).contiguous(),
        defocus_blur.permute(0, 2, 3, 1).contiguous(),
    )


def _apply_softness(gray: torch.Tensor, softness: float) -> torch.Tensor:
    """Blend the image with a gentle spatial blur."""
    softness = max(0.0, min(1.0, float(softness)))
    if softness <= 1e-8:
        return gray
    soft_blur, _defocus_blur = _build_blur_levels(gray)
    out = gray * (1.0 - 0.78 * softness) + soft_blur * (0.78 * softness)
    return out


def _blur_batch(gray: torch.Tensor) -> torch.Tensor:
    """Build a reusable softly blurred version of the batch."""
    soft_blur, _defocus_blur = _build_blur_levels(gray)
    return soft_blur


def _apply_focus_drift(
    gray_rgb: torch.Tensor,
    base_softness: float,
    focus_signal: torch.Tensor,
    focus_drift_strength: float,
) -> torch.Tensor:
    """Modulate apparent focus by varying blur amount per frame."""
    focus_drift_strength = max(0.0, min(1.0, float(focus_drift_strength)))
    if focus_drift_strength <= 1e-8:
        return _apply_softness(gray_rgb, base_softness)

    focus_signal = torch.clamp(focus_signal, -1.0, 1.0).view(-1, 1, 1, 1)
    base_softness = max(0.0, min(1.0, float(base_softness)))

    soft_blur, defocus_blur = _build_blur_levels(gray_rgb)
    base_mix = min(0.88, 0.72 * base_softness + 0.08)
    softened = gray_rgb * (1.0 - base_mix) + soft_blur * base_mix

    sharpen_amount = torch.clamp(-focus_signal, 0.0, 1.0) * min(0.35, 2.4 * focus_drift_strength)
    defocus_amount = torch.clamp(focus_signal, 0.0, 1.0) * min(0.42, 3.2 * focus_drift_strength)

    sharpened = softened * (1.0 - sharpen_amount) + gray_rgb * sharpen_amount
    return sharpened * (1.0 - defocus_amount) + defocus_blur * defocus_amount


def _tone_gray(gray: torch.Tensor, tone_mode: str) -> torch.Tensor:
    """Convert single-channel gray image to a toned RGB image."""
    if tone_mode == "neutral_bw":
        return gray.repeat(1, 1, 1, 3)
    if tone_mode == "warm_print":
        return torch.cat(
            [
                torch.clamp(gray * 1.02, 0.0, 1.0),
                torch.clamp(gray * 0.985, 0.0, 1.0),
                torch.clamp(gray * 0.90, 0.0, 1.0),
            ],
            dim=-1,
        )
    if tone_mode == "sepia_print":
        return torch.cat(
            [
                torch.clamp(gray * 1.07, 0.0, 1.0),
                torch.clamp(gray * 0.96, 0.0, 1.0),
                torch.clamp(gray * 0.78, 0.0, 1.0),
            ],
            dim=-1,
        )
    if tone_mode == "cool_nitrate":
        return torch.cat(
            [
                torch.clamp(gray * 0.90, 0.0, 1.0),
                torch.clamp(gray * 0.97, 0.0, 1.0),
                torch.clamp(gray * 1.04, 0.0, 1.0),
            ],
            dim=-1,
        )
    raise ValueError(f"Unsupported tone_mode: {tone_mode}")


def _apply_grain(
    image_rgb: torch.Tensor,
    grain_strength: float,
    grain_size: int,
    seed: int,
) -> torch.Tensor:
    """Apply coarse monochrome grain to RGB channels."""
    grain_strength = max(0.0, float(grain_strength))
    if grain_strength <= 1e-8:
        return image_rgb

    frame_count, height, width, _channels = image_rgb.shape
    grain_size = int(max(1, grain_size))
    small_h = max(1, height // grain_size)
    small_w = max(1, width // grain_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn((frame_count, 1, small_h, small_w), generator=generator, dtype=torch.float32)
    noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
    noise = noise.to(device=image_rgb.device, dtype=image_rgb.dtype)
    noise = noise.permute(0, 2, 3, 1)
    noise = noise * grain_strength
    out = image_rgb + noise
    return torch.clamp(out, 0.0, 1.0)


def _parse_cadence_json(cadence_json, frame_count: int) -> tuple[str, dict | None]:
    """Parse cadence metadata and validate that it matches the current batch."""
    text = str(cadence_json or "").strip()
    if not text:
        return "none", None
    try:
        data = json.loads(text)
    except Exception:
        return "invalid_json", None
    if not isinstance(data, dict):
        return "invalid_payload", None
    if str(data.get("schema_name", "")) != "alexz.video_silent_film_cadence":
        return "wrong_schema", None
    group_ids = data.get("output_group_ids")
    fps_values = data.get("fps_values_full")
    phase_biases = data.get("phase_biases_full")
    if not isinstance(group_ids, list) or not isinstance(fps_values, list) or not isinstance(phase_biases, list):
        return "missing_sync_fields", None
    if len(group_ids) != int(frame_count):
        return "frame_mismatch", None
    max_group = max([int(x) for x in group_ids], default=-1)
    if max_group < 0 or len(fps_values) <= max_group or len(phase_biases) <= max_group:
        return "group_mismatch", None
    return "cadence_locked", data


def _expand_group_series(
    group_ids: list[int],
    group_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand one value per cadence group to one value per output frame."""
    index = torch.tensor([int(x) for x in group_ids], device=device, dtype=torch.long)
    return group_values.to(device=device, dtype=dtype)[index]


class VideoSilentFilmFinish:
    """ComfyUI node that adds print/projection characteristics of silent film."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Батч кадров видео в THWC."}),
                "tone_mode": (
                    ["neutral_bw", "warm_print", "sepia_print", "cool_nitrate"],
                    {
                        "default": "neutral_bw",
                        "tooltip": "Тонировка копии: нейтральная ЧБ, теплый print, сепия или холодный nitrate.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.97,
                        "min": 0.2,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Глобальный контраст после перевода в монохром.",
                    },
                ),
                "midtone_gamma": (
                    "FLOAT",
                    {
                        "default": 0.94,
                        "min": 0.3,
                        "max": 2.5,
                        "step": 0.01,
                        "tooltip": "Гамма средних тонов. Ниже 1.0 дает более светлый print-подобный midtone lift.",
                    },
                ),
                "black_lift": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "min": -0.3,
                        "max": 0.4,
                        "step": 0.005,
                        "tooltip": "Подъем черного для менее современного контраста.",
                    },
                ),
                "highlight_rolloff": (
                    "FLOAT",
                    {
                        "default": 0.36,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Мягкая компрессия светов.",
                    },
                ),
                "softness": (
                    "FLOAT",
                    {
                        "default": 0.26,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Легкая оптическая мягкость.",
                    },
                ),
                "focus_drift_strength": (
                    "FLOAT",
                    {
                        "default": 0.09,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Очень деликатное пульсирующее изменение резкости. Часть кадров становится чуть резче базового уровня, часть чуть мягче, без modern autofocus hunting.",
                    },
                ),
                "flicker_strength": (
                    "FLOAT",
                    {
                        "default": 0.055,
                        "min": 0.0,
                        "max": 0.3,
                        "step": 0.005,
                        "tooltip": "Глобальный flicker экспозиции между кадрами.",
                    },
                ),
                "breathing_strength": (
                    "FLOAT",
                    {
                        "default": 0.022,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.005,
                        "tooltip": "Медленное плавание яркости поверх flicker.",
                    },
                ),
                "gate_weave_px": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Небольшой сдвиг кадра в пикселях, как у старого проектора.",
                    },
                ),
                "grain_strength": (
                    "FLOAT",
                    {
                        "default": 0.028,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.005,
                        "tooltip": "Сила монохромного зерна.",
                    },
                ),
                "grain_size": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Размер зерна: 1 мельче, больше значения грубее.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1925,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Seed для flicker, weave и grain.",
                    },
                ),
            },
            "optional": {
                "cadence_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Опциональный cadence_json из Silent Film Cadence. Если подключен, flicker, gate weave и focus drift будут синхронизированы с виртуальной скоростью ручной съемки.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "finish_json")
    FUNCTION = "apply"
    CATEGORY = "video/stylize"

    def apply(
        self,
        image,
        tone_mode,
        contrast,
        midtone_gamma,
        black_lift,
        highlight_rolloff,
        softness,
        focus_drift_strength,
        flicker_strength,
        breathing_strength,
        gate_weave_px,
        grain_strength,
        grain_size,
        seed,
        cadence_json="",
    ):
        """Apply silent-film print/projection finishing to a video batch."""
        images = _validate_video_batch(image)
        alpha = images[..., 3:4] if images.size(-1) == 4 else None
        rgb = images[..., :3]

        frame_count = int(rgb.size(0))
        device = rgb.device
        dtype = rgb.dtype

        sync_mode, cadence_meta = _parse_cadence_json(cadence_json, frame_count=frame_count)
        if cadence_meta is not None:
            group_ids = [int(x) for x in cadence_meta.get("output_group_ids", [])]
            group_count = max(group_ids) + 1 if group_ids else 0
            fps_values = torch.tensor(cadence_meta.get("fps_values_full", []), dtype=dtype, device=device)
            phase_biases = torch.tensor(cadence_meta.get("phase_biases_full", []), dtype=dtype, device=device)

            fps_center = torch.clamp(fps_values.mean(), min=1e-6)
            fps_deviation = (fps_center - fps_values) / fps_center
            cadence_flicker = _temporal_signal(group_count, seed + 101, flicker_strength, 0.22, device, dtype)
            cadence_breathing = _temporal_signal(group_count, seed + 211, breathing_strength, 0.93, device, dtype)
            cadence_gate_x = _temporal_signal(group_count, seed + 11, gate_weave_px, 0.58, device, dtype)
            cadence_gate_y = _temporal_signal(group_count, seed + 23, gate_weave_px * 0.8, 0.63, device, dtype)
            cadence_focus = _temporal_signal(group_count, seed + 307, 1.0, 0.36, device, dtype)

            cadence_flicker = cadence_flicker + fps_deviation * float(flicker_strength) * 0.75
            cadence_gate_x = cadence_gate_x + phase_biases * float(gate_weave_px) * 0.85
            cadence_gate_y = cadence_gate_y - phase_biases * float(gate_weave_px) * 0.45
            cadence_focus = torch.tanh(cadence_focus + phase_biases * 0.55)

            gate_x = _expand_group_series(group_ids, cadence_gate_x, device, dtype)
            gate_y = _expand_group_series(group_ids, cadence_gate_y, device, dtype)
            flicker_fast = _expand_group_series(group_ids, cadence_flicker, device, dtype)
            flicker_slow = _expand_group_series(group_ids, cadence_breathing, device, dtype)
            focus_signal = _expand_group_series(group_ids, cadence_focus, device, dtype)
        else:
            gate_x = _temporal_signal(frame_count, seed + 11, gate_weave_px, 0.55, device, dtype)
            gate_y = _temporal_signal(frame_count, seed + 23, gate_weave_px * 0.8, 0.60, device, dtype)
            flicker_fast = _temporal_signal(frame_count, seed + 101, flicker_strength, 0.18, device, dtype)
            flicker_slow = _temporal_signal(frame_count, seed + 211, breathing_strength, 0.92, device, dtype)
            focus_signal = torch.tanh(_temporal_signal(frame_count, seed + 307, 1.0, 0.48, device, dtype))

        woven_rgb = _apply_gate_weave(rgb, gate_x, gate_y)
        woven_alpha = _apply_gate_weave(alpha, gate_x, gate_y) if alpha is not None else None

        gray = (
            woven_rgb[..., 0:1] * 0.299
            + woven_rgb[..., 1:2] * 0.587
            + woven_rgb[..., 2:3] * 0.114
        )
        gamma = max(0.3, float(midtone_gamma))
        gray = torch.pow(torch.clamp(gray, 0.0, 1.0), gamma)
        gray = (gray - 0.5) * float(contrast) + 0.5 + float(black_lift)
        gray = torch.clamp(gray, 0.0, 1.0)
        gray = _soft_clip_highlights(gray, highlight_rolloff)
        gray_rgb = gray.repeat(1, 1, 1, 3)
        gray_rgb = _apply_focus_drift(gray_rgb, softness, focus_signal, focus_drift_strength)
        gray = gray_rgb[..., 0:1]

        exposure = 1.0 + flicker_fast + flicker_slow
        exposure = exposure.view(frame_count, 1, 1, 1)
        gray = torch.clamp(gray * exposure, 0.0, 1.0)

        toned = _tone_gray(gray, tone_mode)
        toned = _apply_grain(toned, grain_strength, grain_size, seed + 1001)

        if woven_alpha is not None:
            output = torch.cat([toned, torch.clamp(woven_alpha, 0.0, 1.0)], dim=-1)
        else:
            output = toned
        output = torch.clamp(output, 0.0, 1.0)

        payload = {
            "schema_name": "alexz.video_silent_film_finish",
            "schema_version": 1,
            "status": "ok",
            "tone_mode": tone_mode,
            "contrast": float(contrast),
            "midtone_gamma": float(midtone_gamma),
            "black_lift": float(black_lift),
            "highlight_rolloff": float(highlight_rolloff),
            "softness": float(softness),
            "focus_drift_strength": float(focus_drift_strength),
            "flicker_strength": float(flicker_strength),
            "breathing_strength": float(breathing_strength),
            "gate_weave_px": float(gate_weave_px),
            "grain_strength": float(grain_strength),
            "grain_size": int(grain_size),
            "seed": int(seed),
            "frame_count": frame_count,
            "sync_mode": sync_mode,
            "focus_preview": [round(float(x), 4) for x in focus_signal[:16].tolist()],
            "gate_x_preview": [round(float(x), 4) for x in gate_x[:16].tolist()],
            "gate_y_preview": [round(float(y), 4) for y in gate_y[:16].tolist()],
            "exposure_preview": [round(float(x), 4) for x in exposure.view(-1)[:16].tolist()],
        }
        return (output, json.dumps(payload, ensure_ascii=False, indent=2))
