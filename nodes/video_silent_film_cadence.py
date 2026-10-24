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
import logging
import math
import random
from bisect import bisect_right
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None


# Передаточное отношение фиксировано; скорость ручки определяется каденсом.
_CRANK_FRAMES_PER_TURN = 8
_LOGGER = logging.getLogger("VideoSilentFilmCadence")


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


def _frame_to_numpy(images: torch.Tensor, index: int) -> np.ndarray:
    """Convert one batched frame to float32 HWC numpy."""
    return images[index].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def _gray_u8_from_frame(frame_np: np.ndarray) -> np.ndarray:
    """Convert RGB(A) float frame in [0,1] to uint8 grayscale for optical flow."""
    rgb = frame_np[..., :3]
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gray = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    return gray


def _dense_pair_flow(frame_a_np: np.ndarray, frame_b_np: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Estimate forward/backward dense flow between adjacent frames."""
    gray_a = _gray_u8_from_frame(frame_a_np)
    gray_b = _gray_u8_from_frame(frame_b_np)
    height, width = gray_a.shape[:2]
    scale = 1.0
    long_side = max(height, width)
    if long_side > max_side:
        scale = float(max_side) / float(long_side)
        new_w = max(8, int(round(width * scale)))
        new_h = max(8, int(round(height * scale)))
        gray_a_small = cv2.resize(gray_a, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray_b_small = cv2.resize(gray_b, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_h, new_w = height, width
        gray_a_small = gray_a
        gray_b_small = gray_b

    flow_fw = cv2.calcOpticalFlowFarneback(gray_a_small, gray_b_small, None, 0.5, 3, 17, 3, 5, 1.1, 0)
    flow_bw = cv2.calcOpticalFlowFarneback(gray_b_small, gray_a_small, None, 0.5, 3, 17, 3, 5, 1.1, 0)
    if new_w != width or new_h != height:
        scale_x = float(width) / float(new_w)
        scale_y = float(height) / float(new_h)
        flow_fw = cv2.resize(flow_fw, (width, height), interpolation=cv2.INTER_LINEAR)
        flow_bw = cv2.resize(flow_bw, (width, height), interpolation=cv2.INTER_LINEAR)
        flow_fw[..., 0] *= scale_x
        flow_fw[..., 1] *= scale_y
        flow_bw[..., 0] *= scale_x
        flow_bw[..., 1] *= scale_y
    return flow_fw.astype(np.float32, copy=False), flow_bw.astype(np.float32, copy=False)


def _warp_np_with_flow(frame_np: np.ndarray, flow_np: np.ndarray, amount: float) -> np.ndarray:
    """Warp one frame along a dense optical flow field by the given fraction."""
    height, width = frame_np.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x - flow_np[..., 0] * float(amount)
    map_y = grid_y - flow_np[..., 1] * float(amount)
    warped = cv2.remap(
        frame_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped.astype(np.float32, copy=False)


def _flow_consistency_confidence(flow: np.ndarray, reverse_flow: np.ndarray) -> np.ndarray:
    """Return source-space confidence from forward/backward flow consistency."""
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    reverse_at_destination = cv2.remap(
        reverse_flow, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    residual = np.sum((flow + reverse_at_destination) ** 2, axis=-1)
    magnitude = np.sum(flow ** 2 + reverse_at_destination ** 2, axis=-1)
    confidence = np.exp(-residual / (0.5 + 0.01 * magnitude))
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return (confidence * valid).astype(np.float32)


def _lerp_frame_flow(
    images: torch.Tensor,
    position: float,
    frame_cache: dict[int, np.ndarray],
    flow_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    confidence_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> torch.Tensor:
    """Sample one frame at a fractional index using dense-flow intermediate warps."""
    frame_count = int(images.size(0))
    pos = max(0.0, min(float(position), float(frame_count - 1)))
    left = int(math.floor(pos))
    right = min(frame_count - 1, left + 1)
    frac = pos - float(left)
    if right == left or frac <= 1e-6:
        return images[left]
    if left not in frame_cache:
        frame_cache[left] = _frame_to_numpy(images, left)
    if right not in frame_cache:
        frame_cache[right] = _frame_to_numpy(images, right)
    if left not in flow_cache:
        flow_cache[left] = _dense_pair_flow(frame_cache[left], frame_cache[right])
    flow_fw, flow_bw = flow_cache[left]
    warped_left = _warp_np_with_flow(frame_cache[left], flow_fw, frac)
    warped_right = _warp_np_with_flow(frame_cache[right], flow_bw, 1.0 - frac)
    if confidence_cache is not None and left in confidence_cache:
        confidence_left, confidence_right = confidence_cache[left]
    else:
        confidence_left = _flow_consistency_confidence(flow_fw, flow_bw)
        confidence_right = _flow_consistency_confidence(flow_bw, flow_fw)
        if confidence_cache is not None:
            confidence_cache[left] = (confidence_left, confidence_right)
    confidence = np.minimum(
        _warp_np_with_flow(confidence_left, flow_fw, frac),
        _warp_np_with_flow(confidence_right, flow_bw, 1.0 - frac),
    )
    # Несогласованные контуры и перекрытия не должны деформировать исходные детали.
    confidence = np.minimum(confidence, np.minimum(confidence_left, confidence_right))
    disagreement = np.mean(np.abs(warped_left[..., :3] - warped_right[..., :3]), axis=-1)
    original_difference = np.mean(
        np.abs(frame_cache[left][..., :3] - frame_cache[right][..., :3]), axis=-1
    )
    confidence *= np.exp(-np.maximum(0.0, disagreement - original_difference) / 0.03)
    confidence = cv2.GaussianBlur(confidence, (5, 5), 0)[..., None]
    flow_blended = warped_left * (1.0 - frac) + warped_right * frac
    simple_blended = frame_cache[left] * (1.0 - frac) + frame_cache[right] * frac
    blended = flow_blended * confidence + simple_blended * (1.0 - confidence)
    return torch.from_numpy(blended).to(device=images.device, dtype=images.dtype)


def _temporal_exposure_sample(
    sample_fn,
    frame_template: torch.Tensor,
    center_position: float,
    exposure_frames: float,
    blur_samples: int,
) -> torch.Tensor:
    """Sample one output frame by integrating several nearby temporal samples."""
    if blur_samples <= 1 or exposure_frames <= 1e-4:
        return sample_fn(center_position)

    # Коэффициенты малы и нужны как Python-числа: не создаём синхронизацию с GPU на каждом кадре.
    offsets = torch.linspace(-0.5, 0.5, steps=int(blur_samples), dtype=frame_template.dtype)
    weights = 1.0 - offsets.abs() * 2.0
    weights = torch.clamp(weights, min=1e-4)
    weights = weights / weights.sum()

    acc = torch.zeros_like(frame_template)
    for offset, weight in zip(offsets.tolist(), weights.tolist()):
        sample = sample_fn(center_position + offset * exposure_frames)
        acc = acc + sample * float(weight)
    return acc


def _select_compute_device(mode: str, height: int, width: int) -> torch.device:
    """Select CUDA with a bounded working-set budget while respecting ComfyUI's device."""
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported compute_device: {mode}")
    if mode == "cpu" or (mode == "auto" and height * width < 256 * 256):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if mode == "cuda":
            _LOGGER.warning("[Silent Film Cadence] CUDA недоступна: используется CPU.")
        return torch.device("cpu")
    try:
        from comfy import model_management
    except ImportError:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = model_management.get_torch_device()
    if device.type != "cuda":
        return torch.device("cpu")
    free, _ = torch.cuda.mem_get_info(device)
    available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    required = height * width * 384 + 512 * 1024**2
    if available < required:
        _LOGGER.warning("[Silent Film Cadence] Недостаточно свободной VRAM для рабочего кэша: используется CPU.")
        return torch.device("cpu")
    return device


def _trim_frame_caches(*caches: dict, limit: int = 4) -> None:
    """Bound sequential frame and flow caches without changing sampling results."""
    for cache in caches:
        while len(cache) > limit:
            del cache[next(iter(cache))]


class _TorchExposureSampler:
    """Warp and blend on a torch device using the existing CPU Farneback estimator."""

    def __init__(self, images, device, frame_cache, flow_cache, confidence_cache):
        """Allocate reusable sampling grids and bounded per-pair device caches."""
        self.images = images
        self.device = device
        self.frame_cache = frame_cache
        self.flow_cache = flow_cache
        self.confidence_cache = confidence_cache
        self.tensor_frames = {}
        self.tensor_pairs = {}
        height, width = images.shape[1:3]
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32), indexing="ij",
        )
        self.pixel_grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
        self.scale = torch.tensor([2.0 / width, 2.0 / height], device=device, dtype=torch.float32).view(1, 2, 1, 1)
        kernel = torch.tensor([1, 4, 6, 4, 1], device=device, dtype=torch.float32) / 16.0
        self.blur_kernel = (kernel[:, None] * kernel[None, :]).view(1, 1, 5, 5)

    def _frame(self, index):
        """Upload only a small window of source frames."""
        if index not in self.tensor_frames:
            self.tensor_frames[index] = self.images[index].to(self.device).permute(2, 0, 1).unsqueeze(0)
            _trim_frame_caches(self.tensor_frames)
        return self.tensor_frames[index]

    def _warp(self, frame, flow, amount):
        """Match OpenCV's linear remap coordinate quantization on the torch device."""
        coords = torch.round((self.pixel_grid - flow * amount) * 32.0) / 32.0
        grid = ((coords + 0.5) * self.scale - 1.0).permute(0, 2, 3, 1)
        return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def _confidence(self, flow, reverse):
        """Evaluate forward/backward consistency on the same device as the warps."""
        reverse_at_destination = self._warp(reverse, flow, -1.0)
        residual = (flow + reverse_at_destination).square().sum(dim=1, keepdim=True)
        magnitude = (flow.square() + reverse_at_destination.square()).sum(dim=1, keepdim=True)
        confidence = torch.exp(-residual / (0.5 + 0.01 * magnitude))
        coords = self.pixel_grid + flow
        height, width = flow.shape[-2:]
        valid = (coords[:, 0:1] >= 0) & (coords[:, 0:1] <= width - 1)
        valid &= (coords[:, 1:2] >= 0) & (coords[:, 1:2] <= height - 1)
        return confidence * valid

    def sample(self, position: float, use_flow: bool) -> torch.Tensor:
        """Sample a fractional frame with bounded device caches and flow confidence."""
        pos = max(0.0, min(float(position), float(len(self.images) - 1)))
        left = int(math.floor(pos))
        right = min(len(self.images) - 1, left + 1)
        frac = pos - left
        first = self._frame(left)
        if left == right or frac <= 1e-6:
            return first[0].permute(1, 2, 0)
        second = self._frame(right)
        simple = first * (1.0 - frac) + second * frac
        if not use_flow:
            return simple[0].permute(1, 2, 0)
        if left not in self.tensor_pairs:
            for index in (left, right):
                if index not in self.frame_cache:
                    self.frame_cache[index] = _frame_to_numpy(self.images, index)
            if left not in self.flow_cache:
                self.flow_cache[left] = _dense_pair_flow(self.frame_cache[left], self.frame_cache[right])
            forward, backward = self.flow_cache[left]
            def upload(array):
                """Convert HWC or HW flow data to a device tensor."""
                if array.ndim == 2:
                    array = array[..., None]
                return torch.from_numpy(array).to(self.device).permute(2, 0, 1).unsqueeze(0)

            forward_tensor, backward_tensor = upload(forward), upload(backward)
            self.tensor_pairs[left] = (
                forward_tensor, backward_tensor,
                self._confidence(forward_tensor, backward_tensor),
                self._confidence(backward_tensor, forward_tensor),
                (first[:, :3] - second[:, :3]).abs().mean(dim=1, keepdim=True),
            )
            _trim_frame_caches(self.tensor_pairs, self.frame_cache, self.flow_cache, self.confidence_cache)
        forward, backward, conf_left, conf_right, original_difference = self.tensor_pairs[left]
        warped_left = self._warp(first, forward, frac)
        warped_right = self._warp(second, backward, 1.0 - frac)
        confidence = torch.minimum(
            self._warp(conf_left, forward, frac), self._warp(conf_right, backward, 1.0 - frac),
        )
        confidence = torch.minimum(confidence, torch.minimum(conf_left, conf_right))
        disagreement = (warped_left[:, :3] - warped_right[:, :3]).abs().mean(dim=1, keepdim=True)
        confidence *= torch.exp(-torch.clamp(disagreement - original_difference, min=0.0) / 0.03)
        padding = "reflect" if min(confidence.shape[-2:]) > 2 else "replicate"
        confidence = F.conv2d(F.pad(confidence, (2, 2, 2, 2), mode=padding), self.blur_kernel)
        warped = warped_left * (1.0 - frac) + warped_right * frac
        blended = warped * confidence + simple * (1.0 - confidence)
        return blended[0].permute(1, 2, 0)


def _build_capture_intervals(
    total_duration: float,
    target_fps_min: float,
    target_fps_max: float,
    fps_drift_strength: float,
    seed: int,
    cadence_model: str = "legacy_random",
    crank_frames_per_turn: int = 8,
    crank_irregularity: float = 0.15,
) -> tuple[list[float], list[float], list[float]]:
    """Build interval boundaries for the virtual low-fps capture cadence."""
    if cadence_model == "hand_crank":
        return _build_hand_crank_intervals(
            total_duration, target_fps_min, target_fps_max, fps_drift_strength,
            seed, crank_frames_per_turn, crank_irregularity,
        )
    if cadence_model != "legacy_random":
        raise ValueError(f"Unsupported cadence_model: {cadence_model}")
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


def _build_hand_crank_intervals(
    total_duration: float,
    target_fps_min: float,
    target_fps_max: float,
    fps_drift_strength: float,
    seed: int,
    crank_frames_per_turn: int,
    crank_irregularity: float,
) -> tuple[list[float], list[float], list[float]]:
    """Build bounded cyclic cadence tied to film advance per crank revolution."""
    if int(crank_frames_per_turn) != crank_frames_per_turn or not 2 <= crank_frames_per_turn <= 32:
        raise ValueError("crank_frames_per_turn must be an integer between 2 and 32.")
    if not math.isfinite(float(crank_irregularity)) or not 0.0 <= crank_irregularity <= 1.0:
        raise ValueError("crank_irregularity must be between 0 and 1.")
    rng = random.Random(int(seed))
    initial_phase = rng.random()
    previous_tempo = rng.uniform(-1.0, 1.0)
    next_tempo = rng.uniform(-1.0, 1.0)
    midpoint = 0.5 * (target_fps_min + target_fps_max)
    amplitude = 0.5 * (target_fps_max - target_fps_min) * max(0.0, min(1.0, float(fps_drift_strength)))
    irregularity = float(crank_irregularity)
    starts = [0.0]
    fps_values = []
    current_time = 0.0
    frame_index = 0
    previous_turn = 0
    while current_time < total_duration - 1e-9:
        # Оборот связан с продвижением плёнки, а не с независимым таймером.
        revolutions = initial_phase + frame_index / float(crank_frames_per_turn)
        turn = int(math.floor(revolutions))
        if turn > previous_turn:
            previous_tempo = next_tempo
            next_tempo = rng.uniform(-1.0, 1.0)
            previous_turn = turn
        phase = revolutions - turn
        smooth_phase = 0.5 - 0.5 * math.cos(math.tau * phase / 2.0)
        tempo = previous_tempo * (1.0 - smooth_phase) + next_tempo * smooth_phase
        cycle = math.sin(math.tau * phase)
        # Даже при максимальной нерегулярности основой остаётся цикл ручки.
        drift_mix = 0.35 * irregularity
        variation = cycle * (1.0 - drift_mix) + tempo * drift_mix
        fps = max(target_fps_min, min(target_fps_max, midpoint + amplitude * variation))
        fps_values.append(float(fps))
        current_time = min(total_duration, current_time + 1.0 / fps)
        starts.append(current_time)
        frame_index += 1
    # Не добавляем независимое случайное смещение экспозиции поверх физического каденса.
    return starts, fps_values, [0.0] * len(fps_values)


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


def _crank_phase_at_time(starts: list[float], time_seconds: float, initial_phase: float) -> float:
    """Interpolate crank revolutions from the same intervals used for capture."""
    time_seconds = max(starts[0], min(starts[-1], float(time_seconds)))
    index = min(len(starts) - 2, max(0, bisect_right(starts, time_seconds) - 1))
    duration = max(1e-9, starts[index + 1] - starts[index])
    fraction = (time_seconds - starts[index]) / duration
    return initial_phase + (index + fraction) / _CRANK_FRAMES_PER_TURN


def _apply_capture_shake(
    frame: torch.Tensor,
    crank_phase: float,
    strength: float,
    base_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply small crank-locked image-plane camera translation and roll."""
    if strength <= 0.0:
        return frame
    height, width, channels = frame.shape
    angle = math.tau * (crank_phase % 1.0)
    shift_x = strength * 0.003 * width * (0.8 * math.sin(angle) + 0.2 * math.sin(2.0 * angle))
    shift_y = strength * 0.002 * height * (0.8 * math.cos(angle) + 0.2 * math.sin(2.0 * angle))
    roll = math.radians(0.08) * strength * math.sin(angle + math.pi / 4.0)
    if base_grid is None:
        identity = torch.eye(2, 3, device=frame.device, dtype=frame.dtype).unsqueeze(0)
        base_grid = F.affine_grid(identity, (1, channels, height, width), align_corners=False)
    x, y = base_grid[..., 0], base_grid[..., 1]
    cosine, sine = math.cos(roll), math.sin(roll)
    # Учитываем пропорции кадра, чтобы наклон не превращался в растяжение.
    grid = torch.stack((
        cosine * x + sine * (height / width) * y - 2.0 * shift_x / width,
        -sine * (width / height) * x + cosine * y - 2.0 * shift_y / height,
    ), dim=-1)
    batch = frame.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0].permute(1, 2, 0)


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
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "FPS входного батча. Текущий дефолт под 24 fps исходник с выводом в 25 fps.",
                    },
                ),
                "playback_mode": (
                    ["preserve_duration_25fps", "undercrank_projected_25fps"],
                    {
                        "default": "undercrank_projected_25fps",
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
                        "default": 17.5,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Верхняя граница виртуального каденса. Выходной батч по-прежнему останется 25 fps.",
                    },
                ),
                "fps_drift_strength": (
                    "FLOAT",
                    {
                        "default": 0.82,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Сила изменения скорости внутри target_fps_min/max: 0 = ровный темп, 1 = полный диапазон. В hand_crank управляет пульсацией ручки; в legacy_random — случайным дрейфом.",
                    },
                ),
                "shutter_fraction": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.01,
                        "tooltip": "Доля виртуального интервала, идущая в экспозиционный blur.",
                    },
                ),
                "motion_blur_strength": (
                    "FLOAT",
                    {
                        "default": 1.22,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Сила темпорального blur для удерживаемых кадров.",
                    },
                ),
                "blur_mode": (
                    ["simple", "flow_integrated"],
                    {
                        "default": "flow_integrated",
                        "tooltip": "simple = быстрое усреднение соседних кадров. flow_integrated = motion-compensated blur для более мягкой длинной экспозиции.",
                    },
                ),
                "blur_samples": (
                    "INT",
                    {
                        "default": 9,
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
            "optional": {
                "cadence_model": (
                    ["legacy_random", "hand_crank"],
                    {
                        "default": "legacy_random",
                        "tooltip": "hand_crank = циклическая неравномерность вращения ручки. legacy_random = прежний случайный дрейф для совместимости старых workflow.",
                    },
                ),
                "crank_frames_per_turn": (
                    "INT",
                    {
                        "default": 8, "min": 2, "max": 32, "step": 1,
                        "tooltip": "Устаревшее поле совместимости, скрыто в интерфейсе. Модель использует фиксированные 8 кадров на оборот; сохранённое значение игнорируется.",
                    },
                ),
                "crank_irregularity": (
                    "FLOAT",
                    {
                        "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Только hand_crank. Неровность темпа между оборотами: 0 = точно повторяющийся цикл; 0.1–0.25 = небольшая живая нерегулярность; 1 = заметный дрейф с сохранением цикла.",
                    },
                ),
                "camera_shake_strength": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Только hand_crank. Покачивание камеры при съёмке, связанное с ручкой и учитываемое внутри экспозиции: 0 = выключено; 0.15–0.30 = небольшое; 1 = сильное. Не заменяет дрожание проектора gate_weave_px в Finish.",
                    },
                ),
                "compute_device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": "auto = GPU для крупных кадров при наличии свободной VRAM; cpu = прежняя обработка; cuda = запрос GPU. Optical flow Farneback остаётся на CPU, деформации, смешивание и покачивание ускоряются на GPU. При нехватке VRAM используется CPU.",
                    },
                ),
            },
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
        blur_mode,
        blur_samples,
        seed,
        cadence_model="legacy_random",
        crank_frames_per_turn=8,
        crank_irregularity=0.15,
        camera_shake_strength=0.0,
        compute_device="auto",
    ):
        """Apply virtual capture cadence, exposure blur, and optional crank-locked shake."""
        started_at = perf_counter()
        _LOGGER.info("[Silent Film Cadence] Этап 1/4: проверка входных кадров и параметров.")
        images = _validate_video_batch(image)
        camera_shake_strength = float(camera_shake_strength)
        if not math.isfinite(camera_shake_strength) or not 0.0 <= camera_shake_strength <= 1.0:
            raise ValueError("camera_shake_strength must be between 0 and 1.")
        # Сохраняем аргумент и место виджета для старых workflow, но механизм теперь фиксирован.
        crank_frames_per_turn = _CRANK_FRAMES_PER_TURN

        source_fps = float(source_fps)
        playback_mode = str(playback_mode)
        blur_mode = str(blur_mode)
        target_fps_min = float(target_fps_min)
        target_fps_max = float(target_fps_max)
        if source_fps <= 0.0:
            raise ValueError("source_fps must be > 0.")
        if playback_mode not in {"preserve_duration_25fps", "undercrank_projected_25fps"}:
            raise ValueError(f"Unsupported playback_mode: {playback_mode}")
        if blur_mode not in {"simple", "flow_integrated"}:
            raise ValueError(f"Unsupported blur_mode: {blur_mode}")
        if target_fps_min <= 0.0 or target_fps_max <= 0.0:
            raise ValueError("target_fps_min and target_fps_max must be > 0.")
        if target_fps_min > target_fps_max:
            raise ValueError("target_fps_min must be <= target_fps_max.")

        frame_count = int(images.size(0))
        _LOGGER.info(
            "[Silent Film Cadence] Вход: %d кадров, %dx%d, source_fps=%.2f; режим=%s, модель=%s.",
            frame_count, images.size(2), images.size(1), source_fps, playback_mode, cadence_model,
        )
        total_duration = float(frame_count) / source_fps
        _LOGGER.info("[Silent Film Cadence] Этап 2/4: построение виртуального каденса и экспозиций.")
        starts, fps_values, phase_biases = _build_capture_intervals(
            total_duration=total_duration,
            target_fps_min=target_fps_min,
            target_fps_max=target_fps_max,
            fps_drift_strength=fps_drift_strength,
            seed=seed,
            cadence_model=cadence_model,
            crank_frames_per_turn=crank_frames_per_turn,
            crank_irregularity=crank_irregularity,
        )
        group_ids = _assign_frames_to_groups(frame_count=frame_count, output_fps=source_fps, starts=starts)
        group_sizes = _summarize_group_sizes(group_ids)
        interval_span_map = _interval_output_span_map(starts, output_fps=source_fps)
        blur_scale = max(0.0, float(motion_blur_strength))
        shutter_fraction = max(0.0, float(shutter_fraction))
        blur_samples = int(max(1, blur_samples))
        actual_blur_mode = blur_mode
        frame_cache: dict[int, np.ndarray] = {}
        flow_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        confidence_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if blur_mode == "flow_integrated" and cv2 is None:
            actual_blur_mode = "simple_fallback_no_cv2"
            _LOGGER.warning("[Silent Film Cadence] OpenCV недоступен: вместо flow_integrated используется simple.")
        actual_camera_shake = camera_shake_strength if cadence_model == "hand_crank" else 0.0
        initial_crank_phase = random.Random(int(seed)).random()
        work_device = _select_compute_device(str(compute_device), images.size(1), images.size(2))

        def _prepare_sampling(device):
            """Prepare only per-frame device storage, not the entire video batch."""
            sampler = _TorchExposureSampler(images, device, frame_cache, flow_cache, confidence_cache) if device.type == "cuda" else None
            template = images[0].to(device)
            grid = None
            if actual_camera_shake > 0.0:
                height, width, channels = images.shape[1:]
                identity = torch.eye(2, 3, device=device, dtype=images.dtype).unsqueeze(0)
                grid = F.affine_grid(identity, (1, channels, height, width), align_corners=False)
            return sampler, template, grid

        try:
            torch_sampler, frame_template, capture_grid = _prepare_sampling(work_device)
        except torch.OutOfMemoryError:
            if work_device.type != "cuda":
                raise
            _LOGGER.warning("[Silent Film Cadence] Не удалось выделить VRAM: используется CPU.")
            work_device = torch.device("cpu")
            torch_sampler, frame_template, capture_grid = _prepare_sampling(work_device)
        _LOGGER.info(
            "[Silent Film Cadence] Обработка: %s; optical flow: %s.",
            work_device, "CPU Farneback" if actual_blur_mode == "flow_integrated" else "не используется",
        )

        output_count = len(starts) - 1 if playback_mode == "undercrank_projected_25fps" else frame_count
        _LOGGER.info(
            "[Silent Film Cadence] Этап 3/4: экспозиционное размытие и движение камеры; "
            "blur=%s, отсчётов=%d, покачивание=%.2f, выходных кадров=%d.",
            actual_blur_mode, blur_samples, actual_camera_shake, output_count,
        )

        def _sample_position(position: float) -> torch.Tensor:
            if torch_sampler is not None:
                sample = torch_sampler.sample(position, actual_blur_mode == "flow_integrated")
            elif actual_blur_mode == "flow_integrated":
                sample = _lerp_frame_flow(images, position, frame_cache, flow_cache, confidence_cache)
            else:
                sample = _lerp_frame(images, position)
            sample = sample.to(work_device)
            _trim_frame_caches(frame_cache, flow_cache, confidence_cache)
            if actual_camera_shake > 0.0:
                sample_time = (position + 0.5) / source_fps
                crank_phase = _crank_phase_at_time(starts, sample_time, initial_crank_phase)
                sample = _apply_capture_shake(sample, crank_phase, actual_camera_shake, capture_grid)
            return sample
        
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
                sample_fn=_sample_position,
                frame_template=frame_template,
                center_position=center_pos,
                exposure_frames=exposure_frames,
                blur_samples=blur_samples,
            )

        def _render_frame(group_idx, phase):
            """Retry only the current exposure on CPU after a CUDA allocation failure."""
            nonlocal work_device, torch_sampler, frame_template, capture_grid
            try:
                result = _sample_group_frame(group_idx, phase)
            except torch.OutOfMemoryError:
                if work_device.type != "cuda":
                    raise
                _LOGGER.warning("[Silent Film Cadence] VRAM закончилась: текущий кадр и остаток батча рассчитываются на CPU.")
                torch_sampler = capture_grid = frame_template = None
                work_device = torch.device("cpu")
                torch_sampler, frame_template, capture_grid = _prepare_sampling(work_device)
                result = _sample_group_frame(group_idx, phase)
            return result.to(images.device)

        # Заполняем один батч сразу: список кадров плюс stack/clamp удваивали пиковую RAM.
        output_frames = torch.empty((output_count, *images.shape[1:]), dtype=images.dtype, device=images.device)
        with tqdm(total=output_count, desc="[Silent Film Cadence] Экспозиция", unit="кадр", dynamic_ncols=True, mininterval=0.5) as progress:
            if playback_mode == "undercrank_projected_25fps":
                for group_idx in range(len(starts) - 1):
                    output_frames[group_idx].copy_(_render_frame(group_idx, phase=0.5))
                    progress.update(1)
                intra_group_motion = 0.0
            else:
                output_times = ((torch.arange(frame_count, dtype=torch.float32) + 0.5) / float(source_fps)).tolist()
                group_idx = 0
                intra_group_motion = min(0.45, 0.35 * blur_scale)
                for output_index, t in enumerate(output_times):
                    while group_idx + 1 < len(starts) - 1 and t >= starts[group_idx + 1] - 1e-9:
                        group_idx += 1
                    start_t = starts[group_idx]
                    end_t = starts[group_idx + 1]
                    interval_duration = max(1e-6, end_t - start_t)
                    phase = (t - start_t) / interval_duration
                    phase = 0.5 + (phase - 0.5) * intra_group_motion
                    output_frames[output_index].copy_(_render_frame(group_idx, phase=phase))
                    progress.update(1)

        _LOGGER.info("[Silent Film Cadence] Этап 4/4: сборка выходного батча и cadence_json.")
        output = output_frames.clamp_(0.0, 1.0)

        avg_effective_fps = float((len(starts) - 1) / total_duration) if total_duration > 0 else 0.0
        output_duration_seconds = float(len(output_frames) / source_fps) if source_fps > 0 else 0.0
        payload = {
            "schema_name": "alexz.video_silent_film_cadence",
            "schema_version": 1,
            "status": "ok",
            "source_fps": float(source_fps),
            "output_fps": float(source_fps),
            "playback_mode": playback_mode,
            "blur_mode": blur_mode,
            "actual_blur_mode": actual_blur_mode,
            "compute_device": str(compute_device),
            "actual_compute_device": str(work_device),
            "target_fps_min": float(target_fps_min),
            "target_fps_max": float(target_fps_max),
            "fps_drift_strength": float(fps_drift_strength),
            "cadence_model": str(cadence_model),
            "crank_frames_per_turn": int(crank_frames_per_turn),
            "crank_irregularity": float(crank_irregularity),
            "camera_shake_strength": camera_shake_strength,
            "actual_camera_shake_strength": actual_camera_shake,
            "capture_motion_model": "crank_locked" if actual_camera_shake > 0.0 else "none",
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
            "output_group_ids": [int(x) for x in range(len(output_frames))]
            if playback_mode == "undercrank_projected_25fps"
            else [int(x) for x in group_ids],
            "fps_values_full": [round(float(x), 6) for x in fps_values],
            "phase_biases_full": [round(float(x), 6) for x in phase_biases],
            "fps_values_preview": [round(x, 4) for x in fps_values[:32]],
            "phase_biases_preview": [round(x, 4) for x in phase_biases[:32]],
            "intra_group_motion": float(intra_group_motion),
            "recommended_video_combine_fps": float(source_fps),
        }
        cadence_json = json.dumps(payload, ensure_ascii=False, indent=2)
        _LOGGER.info(
            "[Silent Film Cadence] Готово: %d → %d кадров за %.1f с; виртуальный fps=%.2f.",
            frame_count, len(output_frames), perf_counter() - started_at, avg_effective_fps,
        )
        return (output, cadence_json)
