"""
Module: nodes/video_silent_film_cadence.py
Author: AlexZ1967
Last updated: 2026-05-08

Description:
    Silent-film cadence emulation node implementation.

Purpose:
    Rebuilds a 25 fps image batch into a lower effective cadence with optional
    drifting fps and temporal exposure blur while preserving output frame count.
"""

from __future__ import annotations

import json
import math
import random

import torch


def _validate_video_batch(images: torch.Tensor) -> torch.Tensor:
    """Validate Comfy image batch contract and normalize dtype range assumptions."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("image must be a torch.Tensor batch in THWC layout.")
    if images.dim() != 4:
        raise ValueError(f"Expected image batch in THWC layout, got shape={tuple(images.shape)}")
    if images.size(0) < 2:
        raise ValueError("Silent-film cadence emulation requires at least 2 frames.")
    if images.size(-1) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got {images.size(-1)}.")
    return images.detach().float().clamp(0.0, 1.0)


def _lerp_frame(images: torch.Tensor, position: float) -> torch.Tensor:
    """Sample one frame at a fractional index via linear interpolation."""
    frame_count = int(images.size(0))
    pos = max(0.0, min(float(position), float(frame_count - 1)))
    left = int(math.floor(pos))
    right = min(frame_count - 1, left + 1)
    frac = pos - float(left)
    if right == left or frac <= 1e-6:
        return images[left]
    return images[left] * (1.0 - frac) + images[right] * frac


def _temporal_exposure_sample(
    images: torch.Tensor,
    center_position: float,
    exposure_frames: float,
    blur_samples: int,
) -> torch.Tensor:
    """Sample one output frame by integrating several nearby temporal samples."""
    if blur_samples <= 1 or exposure_frames <= 1e-4:
        return _lerp_frame(images, center_position)

    offsets = torch.linspace(-0.5, 0.5, steps=int(blur_samples), dtype=images.dtype, device=images.device)
    weights = 1.0 - offsets.abs() * 2.0
    weights = torch.clamp(weights, min=1e-4)
    weights = weights / weights.sum()

    acc = torch.zeros_like(images[0])
    for offset, weight in zip(offsets.tolist(), weights.tolist()):
        sample = _lerp_frame(images, center_position + offset * exposure_frames)
        acc = acc + sample * float(weight)
    return acc


def _build_capture_intervals(
    total_duration: float,
    target_fps_min: float,
    target_fps_max: float,
    fps_drift_strength: float,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Build interval boundaries for the virtual low-fps capture cadence."""
    rng = random.Random(int(seed))
    starts = [0.0]
    fps_values = []
    phase_biases = []
    current_time = 0.0
    state = 0.0
    phase_state = 0.0
    midpoint = 0.5 * (target_fps_min + target_fps_max)
    span = max(0.0, target_fps_max - target_fps_min)
    drift_strength = max(0.0, min(1.0, float(fps_drift_strength)))

    while current_time < total_duration - 1e-9:
        if span <= 1e-6:
            fps = midpoint
        else:
            state = state * 0.72 + (rng.uniform(-1.0, 1.0) * drift_strength)
            state = max(-1.0, min(1.0, state))
            fps = midpoint + 0.5 * span * state
            fps = max(target_fps_min, min(target_fps_max, fps))
        fps_values.append(float(fps))
        phase_state = phase_state * 0.55 + rng.uniform(-1.0, 1.0) * 0.16 * drift_strength
        phase_state = max(-0.18, min(0.18, phase_state))
        phase_biases.append(float(phase_state))
        current_time = min(total_duration, current_time + (1.0 / max(1e-6, fps)))
        starts.append(current_time)
    return starts, fps_values, phase_biases


def _assign_frames_to_groups(frame_count: int, output_fps: float, starts: list[float]) -> list[int]:
    """Assign each output frame center to one virtual capture interval."""
    output_times = ((torch.arange(frame_count, dtype=torch.float32) + 0.5) / float(output_fps)).tolist()
    group_ids: list[int] = []
    group_idx = 0
    for t in output_times:
        while group_idx + 1 < len(starts) - 1 and t >= starts[group_idx + 1] - 1e-9:
            group_idx += 1
        group_ids.append(group_idx)
    return group_ids


def _summarize_group_sizes(group_ids: list[int]) -> list[int]:
    """Collapse per-frame group ids into consecutive output counts per group."""
    if not group_ids:
        return []
    sizes = []
    current = group_ids[0]
    count = 1
    for item in group_ids[1:]:
        if item == current:
            count += 1
            continue
        sizes.append(count)
        current = item
        count = 1
    sizes.append(count)
    return sizes


def _interval_output_span_map(starts: list[float], output_fps: float) -> list[float]:
    """Estimate how many 25 fps display frames each virtual capture interval spans."""
    spans = []
    for idx in range(max(0, len(starts) - 1)):
        duration = max(0.0, float(starts[idx + 1] - starts[idx]))
        spans.append(max(1.0, duration * float(output_fps)))
    return spans


class VideoSilentFilmCadence:
    """ComfyUI node that emulates 1920s-style low cadence while keeping 25 fps output."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Батч кадров видео в THWC. Обычно 25 fps исходник."}),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "FPS входного батча. Для вашей задачи обычно 25.",
                    },
                ),
                "playback_mode": (
                    ["preserve_duration_25fps", "undercrank_projected_25fps"],
                    {
                        "default": "preserve_duration_25fps",
                        "tooltip": "preserve_duration_25fps сохраняет длину клипа. undercrank_projected_25fps делает аутентичное ускорение движения как при проекции немого кино на 25 fps.",
                    },
                ),
                "target_fps_min": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Нижняя граница виртуального каденса немого кино.",
                    },
                ),
                "target_fps_max": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Верхняя граница виртуального каденса. Выходной батч по-прежнему останется 25 fps.",
                    },
                ),
                "fps_drift_strength": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Сколько плавания разрешать внутри диапазона target_fps_min/max.",
                    },
                ),
                "shutter_fraction": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.01,
                        "tooltip": "Доля виртуального интервала, идущая в экспозиционный blur.",
                    },
                ),
                "motion_blur_strength": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Сила темпорального blur для удерживаемых кадров.",
                    },
                ),
                "blur_samples": (
                    "INT",
                    {
                        "default": 7,
                        "min": 1,
                        "max": 17,
                        "step": 2,
                        "tooltip": "Сколько суб-сэмплов усреднять внутри экспозиции.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1925,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Seed для детерминированного плавания fps.",
                    },
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "cadence_json")
    FUNCTION = "apply"
    CATEGORY = "video/stylize"

    def apply(
        self,
        image,
        source_fps,
        playback_mode,
        target_fps_min,
        target_fps_max,
        fps_drift_strength,
        shutter_fraction,
        motion_blur_strength,
        blur_samples,
        seed,
    ):
        """Apply silent-film cadence emulation while keeping output length unchanged."""
        images = _validate_video_batch(image)

        source_fps = float(source_fps)
        playback_mode = str(playback_mode)
        target_fps_min = float(target_fps_min)
        target_fps_max = float(target_fps_max)
        if source_fps <= 0.0:
            raise ValueError("source_fps must be > 0.")
        if playback_mode not in {"preserve_duration_25fps", "undercrank_projected_25fps"}:
            raise ValueError(f"Unsupported playback_mode: {playback_mode}")
        if target_fps_min <= 0.0 or target_fps_max <= 0.0:
            raise ValueError("target_fps_min and target_fps_max must be > 0.")
        if target_fps_min > target_fps_max:
            raise ValueError("target_fps_min must be <= target_fps_max.")

        frame_count = int(images.size(0))
        total_duration = float(frame_count) / source_fps
        starts, fps_values, phase_biases = _build_capture_intervals(
            total_duration=total_duration,
            target_fps_min=target_fps_min,
            target_fps_max=target_fps_max,
            fps_drift_strength=fps_drift_strength,
            seed=seed,
        )
        group_ids = _assign_frames_to_groups(frame_count=frame_count, output_fps=source_fps, starts=starts)
        group_sizes = _summarize_group_sizes(group_ids)
        interval_span_map = _interval_output_span_map(starts, output_fps=source_fps)
        blur_scale = max(0.0, float(motion_blur_strength))
        shutter_fraction = max(0.0, float(shutter_fraction))
        blur_samples = int(max(1, blur_samples))
        
        def _sample_group_frame(group_idx: int, phase: float) -> torch.Tensor:
            start_t = starts[group_idx]
            end_t = starts[group_idx + 1]
            interval_duration = max(1e-6, end_t - start_t)
            phase = max(0.0, min(1.0, phase))
            compressed_phase = phase + phase_biases[group_idx]
            compressed_phase = max(0.06, min(0.94, compressed_phase))
            sample_center_t = start_t + interval_duration * compressed_phase
            center_pos = sample_center_t * source_fps - 0.5
            interval_frames = interval_duration * source_fps
            exposure_frames = interval_frames * shutter_fraction * blur_scale
            if blur_scale > 0.0 and interval_span_map:
                exposure_frames = max(
                    exposure_frames,
                    0.28 * float(interval_span_map[group_idx]) * blur_scale,
                )
            return _temporal_exposure_sample(
                images=images,
                center_position=center_pos,
                exposure_frames=exposure_frames,
                blur_samples=blur_samples,
            )

        if playback_mode == "undercrank_projected_25fps":
            output_frames = []
            for group_idx in range(len(starts) - 1):
                output_frames.append(_sample_group_frame(group_idx, phase=0.5))
            intra_group_motion = 0.0
        else:
            output_times = ((torch.arange(frame_count, dtype=torch.float32) + 0.5) / float(source_fps)).tolist()
            output_frames = []
            group_idx = 0
            intra_group_motion = min(0.45, 0.35 * blur_scale)
            for t in output_times:
                while group_idx + 1 < len(starts) - 1 and t >= starts[group_idx + 1] - 1e-9:
                    group_idx += 1
                start_t = starts[group_idx]
                end_t = starts[group_idx + 1]
                interval_duration = max(1e-6, end_t - start_t)
                phase = (t - start_t) / interval_duration
                phase = 0.5 + (phase - 0.5) * intra_group_motion
                output_frames.append(_sample_group_frame(group_idx, phase=phase))

        output = torch.stack(output_frames, dim=0).clamp(0.0, 1.0)

        avg_effective_fps = float((len(starts) - 1) / total_duration) if total_duration > 0 else 0.0
        output_duration_seconds = float(len(output_frames) / source_fps) if source_fps > 0 else 0.0
        payload = {
            "schema_name": "alexz.video_silent_film_cadence",
            "schema_version": 1,
            "status": "ok",
            "source_fps": float(source_fps),
            "output_fps": float(source_fps),
            "playback_mode": playback_mode,
            "target_fps_min": float(target_fps_min),
            "target_fps_max": float(target_fps_max),
            "fps_drift_strength": float(fps_drift_strength),
            "shutter_fraction": float(shutter_fraction),
            "motion_blur_strength": float(motion_blur_strength),
            "blur_samples": int(blur_samples),
            "seed": int(seed),
            "input_frame_count": frame_count,
            "output_frame_count": int(len(output_frames)),
            "group_count": int(len(starts) - 1),
            "average_effective_fps": avg_effective_fps,
            "input_duration_seconds": total_duration,
            "output_duration_seconds": output_duration_seconds,
            "group_sizes_preview": group_sizes[:32],
            "max_group_size": int(max(group_sizes) if group_sizes else 0),
            "min_group_size": int(min(group_sizes) if group_sizes else 0),
            "average_group_size": float(sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0,
            "unique_frame_ratio": float((len(starts) - 1) / frame_count),
            "fps_values_preview": [round(x, 4) for x in fps_values[:32]],
            "phase_biases_preview": [round(x, 4) for x in phase_biases[:32]],
            "intra_group_motion": float(intra_group_motion),
            "recommended_video_combine_fps": float(source_fps),
        }
        return (output, json.dumps(payload, ensure_ascii=False, indent=2))
