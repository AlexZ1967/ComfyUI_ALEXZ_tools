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
import logging
import math
from time import perf_counter

import torch
import torch.nn.functional as F
import psutil
from tqdm import tqdm

from ..utils.video_memory import allocate_video, release_video_pages, use_disk, check_interrupt


_LOGGER = logging.getLogger("VideoSilentFilmFinish")


def _log_during_progress(level, message, *args):
    """Clear active progress bars before logging and redraw them afterwards."""
    with tqdm.external_write_mode():
        _LOGGER.log(level, message, *args)


def _validate_video_batch(images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Validate Comfy image batch contract and clamp to normalized range."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("image must be a torch.Tensor batch in THWC layout.")
    if images.dim() != 4:
        raise ValueError(f"Expected image batch in THWC layout, got shape={tuple(images.shape)}")
    if images.size(0) < 1:
        raise ValueError("Silent film finish requires at least one frame.")
    if images.size(-1) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got {images.size(-1)}.")
    images = images.detach()
    return images.float().clamp(0.0, 1.0) if normalize else images


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


def _lens_edge_mask(height, width, edge_start, device, dtype, edge_end=1.0):
    """Build a circular falloff using half the frame width as the unit radius."""
    # Единый масштаб обеих осей сохраняет круговые контуры на прямоугольном кадре.
    half_width, half_height = (width - 1) / 2.0, (height - 1) / 2.0
    width_radius = max(half_width, 1.0)
    x = (torch.arange(width, device=device, dtype=dtype) - half_width) / width_radius
    y = (torch.arange(height, device=device, dtype=dtype) - half_height) / width_radius
    radius = torch.sqrt(y[:, None].square() + x[None, :].square())
    transition = ((radius - edge_start) / (edge_end - edge_start)).clamp(0.0, 1.0)
    mask = transition.square() * (3.0 - 2.0 * transition)
    return mask.view(1, height, width, 1)


def _lens_gaussian_blur(rgb, sigma):
    """Blur at one radius using a bounded kernel and a reduced working resolution."""
    _count, height, width, channels = rgb.shape
    batch = rgb.permute(0, 3, 1, 2)
    scale = min(1.0, 3.0 / sigma)
    if scale < 1.0:
        blur_size = (max(1, round(height * scale)), max(1, round(width * scale)))
        batch = F.interpolate(batch, size=blur_size, mode="area")
        sigma *= scale
    radius = math.ceil(3.0 * sigma)
    axis = torch.arange(-radius, radius + 1, device=rgb.device, dtype=rgb.dtype)
    kernel = torch.exp(-0.5 * (axis / sigma).square())
    kernel = kernel / kernel.sum()
    padding = "reflect" if min(batch.shape[-2:]) > radius else "replicate"
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    blurred = F.conv2d(F.pad(batch, (radius, radius, 0, 0), mode=padding), horizontal, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode=padding), vertical, groups=channels)
    if scale < 1.0:
        blurred = F.interpolate(blurred, size=(height, width), mode="bilinear", align_corners=False)
    blurred = blurred.permute(0, 2, 3, 1)
    return blurred


def _apply_variable_lens_blur(rgb, mask, maximum_sigma):
    """Approximate a spatially varying Gaussian by interpolating adjacent radius levels."""
    # Маска задаёт локальный радиус, а не прозрачность одинаково размытой копии.
    sigma_map = mask * maximum_sigma
    levels = [0.0]
    sigma = min(0.5, maximum_sigma)
    while sigma < maximum_sigma:
        levels.append(sigma)
        sigma *= 2.0
    levels.append(maximum_sigma)
    output = torch.zeros_like(rgb)
    for index, sigma in enumerate(levels):
        check_interrupt()
        if index == 0:
            weight = (1.0 - sigma_map / levels[1]).clamp(0.0, 1.0)
        else:
            weight = ((sigma_map - levels[index - 1]) / (sigma - levels[index - 1])).clamp(0.0, 1.0)
            if index + 1 < len(levels):
                weight = torch.minimum(weight, ((levels[index + 1] - sigma_map) / (levels[index + 1] - sigma)).clamp(0.0, 1.0))
        blurred = rgb if index == 0 else _lens_gaussian_blur(rgb, sigma)
        output.addcmul_(blurred, weight)
    return torch.where(mask == 0.0, rgb, output)


def _apply_lens_edges(rgb, vignette_strength, edge_softness, edge_start, mask=None, edge_end=1.0):
    """Apply time-invariant lens falloff and peripheral blur before projection and grain."""
    if vignette_strength == 0.0 and edge_softness == 0.0:
        return rgb
    _count, height, width, channels = rgb.shape
    if mask is None:
        mask = _lens_edge_mask(height, width, edge_start, rgb.device, rgb.dtype, edge_end)
    if edge_softness > 0.0:
        maximum_sigma = width / 960.0 * 24.0 * edge_softness
        rgb = _apply_variable_lens_blur(rgb, mask, maximum_sigma)
    if vignette_strength > 0.0:
        rgb = rgb * (1.0 - mask * vignette_strength)
    return rgb


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


def _normalize_field(field: torch.Tensor) -> torch.Tensor:
    """Normalize a per-frame scalar field to zero mean and unit variance."""
    mean = field.mean(dim=(1, 2, 3), keepdim=True)
    field = field - mean
    std = field.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    return field / std


def _scaled_noise(
    frame_count: int,
    height: int,
    width: int,
    scale_divisor: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a correlated noise field by upscaling lower-resolution noise."""
    scale_divisor = max(1.0, float(scale_divisor))
    small_h = max(1, int(round(height / scale_divisor)))
    small_w = max(1, int(round(width / scale_divisor)))
    noise = torch.randn((frame_count, 1, small_h, small_w), generator=generator, dtype=torch.float32)
    noise = noise.to(device=device, dtype=dtype)
    noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
    return noise


def _build_emulsion_grain_field(
    frame_count: int,
    height: int,
    width: int,
    grain_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a clustered grain field closer to emulsion structure than white noise."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    grain_size = max(1.0, float(grain_size))
    fine = _scaled_noise(frame_count, height, width, grain_size, generator, device, dtype)
    mid = _scaled_noise(frame_count, height, width, grain_size * 2.0, generator, device, dtype)
    coarse = _scaled_noise(frame_count, height, width, grain_size * 4.0, generator, device, dtype)
    envelope = _scaled_noise(frame_count, height, width, grain_size * 6.0, generator, device, dtype)

    fine = _normalize_field(fine)
    mid = _normalize_field(mid)
    coarse = _normalize_field(coarse)
    envelope = _normalize_field(envelope)

    clump_gain = 0.78 + 0.48 * torch.sigmoid(envelope * 1.3)
    grain = fine * clump_gain + mid * 0.42 + coarse * 0.24
    grain = _normalize_field(grain)
    return torch.tanh(grain * 1.25)


def _apply_grain_overlay(
    image_rgb: torch.Tensor,
    grain_strength: float,
    grain_size: int,
    seed: int,
) -> torch.Tensor:
    """Legacy fast grain overlay for previews and lighter workflows."""
    grain_strength = max(0.0, float(grain_strength))
    if grain_strength <= 1e-8:
        return image_rgb

    frame_count, height, width, _channels = image_rgb.shape
    grain_size = max(1.0, float(grain_size))
    small_h = max(1, int(height // grain_size))
    small_w = max(1, int(width // grain_size))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn((frame_count, 1, small_h, small_w), generator=generator, dtype=torch.float32)
    noise = noise.to(device=image_rgb.device, dtype=image_rgb.dtype)
    noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)
    noise = noise.permute(0, 2, 3, 1)
    out = image_rgb + noise * grain_strength
    return torch.clamp(out, 0.0, 1.0)


def _apply_grain(
    image_rgb: torch.Tensor,
    grain_mode: str,
    grain_strength: float,
    grain_size: int,
    seed: int,
    frame_offset: int | None = None,
) -> torch.Tensor:
    """Apply either fast overlay grain or a more film-like emulsion grain model."""
    grain_strength = max(0.0, float(grain_strength))
    if grain_strength <= 1e-8:
        return image_rgb

    if frame_offset is not None:
        # Каждый кадр имеет свой seed: размер порции и повтор после OOM не меняют зерно.
        output = torch.empty_like(image_rgb)
        for index in range(len(image_rgb)):
            frame_seed = (int(seed) + (frame_offset + index) * 1000003) % (2**63 - 1)
            output[index:index + 1] = _apply_grain(
                image_rgb[index:index + 1], grain_mode, grain_strength, grain_size, frame_seed,
            )
        return output

    if grain_mode == "overlay_fast":
        return _apply_grain_overlay(image_rgb, grain_strength, grain_size, seed)

    if grain_mode != "emulsion_luma":
        raise ValueError(f"Unsupported grain_mode: {grain_mode}")

    frame_count, height, width, _channels = image_rgb.shape
    device = image_rgb.device
    dtype = image_rgb.dtype
    grain_field = _build_emulsion_grain_field(
        frame_count=frame_count,
        height=height,
        width=width,
        grain_size=grain_size,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    luma = (
        image_rgb[..., 0:1] * 0.299
        + image_rgb[..., 1:2] * 0.587
        + image_rgb[..., 2:3] * 0.114
    )
    midtone_peak = 1.0 - torch.clamp(torch.abs(luma - 0.5) / 0.5, 0.0, 1.0)
    shadow_gate = torch.clamp((luma - 0.04) / 0.26, 0.0, 1.0)
    highlight_gate = torch.clamp((0.96 - luma) / 0.28, 0.0, 1.0)
    luma_weight = 0.28 + 1.06 * (midtone_peak**0.78) * shadow_gate * highlight_gate

    detail_loss = torch.clamp(grain_strength * (0.46 + 0.58 * luma_weight), 0.0, 0.26)
    blur_kernel = 3 if int(max(1, grain_size)) <= 2 else 5
    batch = image_rgb.permute(0, 3, 1, 2).contiguous()
    blurred = F.avg_pool2d(batch, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
    blurred = blurred.permute(0, 2, 3, 1).contiguous()
    eroded = image_rgb * (1.0 - detail_loss) + blurred * detail_loss

    grain_rgb = grain_field.permute(0, 2, 3, 1)
    density_response = grain_strength * (0.42 + 0.78 * luma_weight)
    multiplicative = 1.0 + grain_rgb * density_response * 0.48
    additive = grain_rgb * density_response * 0.16

    local_contrast = torch.clamp(torch.abs(image_rgb - blurred).mean(dim=-1, keepdim=True) * 5.0, 0.0, 1.0)
    grain_presence = torch.clamp(0.72 + 0.42 * luma_weight - 0.18 * local_contrast, 0.35, 1.2)
    out = eroded * (1.0 + (multiplicative - 1.0) * grain_presence) + additive * grain_presence
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


def _available_finish_ram() -> int:
    """Use ComfyUI's cgroup-aware RAM availability when it is available."""
    try:
        from comfy.system_memory import virtual_memory_available
    except ImportError:
        return int(psutil.virtual_memory().available)
    return int(virtual_memory_available())


def _finish_processing_plan(images: torch.Tensor, mode: str, batch_size: int, *, label="Silent Film Finish") -> tuple[torch.device, int, int]:
    """Budget working chunks independently of large disk-backed output storage."""
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported compute_device: {mode}")
    if int(batch_size) != batch_size or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer between 1 and 16.")
    frame_count, height, width, channels = images.shape
    batch_size = min(int(batch_size), frame_count)
    output_bytes = frame_count * height * width * channels * 4
    reserve = 512 * 1024**2
    per_frame = height * width * 512
    ram = _available_finish_ram()
    mapped = use_disk(images.shape, ram=ram, work_bytes=per_frame * int(batch_size))
    cpu_output_bytes = output_bytes if images.device.type == "cpu" and not mapped else 0
    cpu_budget = ram - cpu_output_bytes - reserve
    cpu_batch = min(int(batch_size), max(0, cpu_budget // per_frame))
    _LOGGER.info("[%s] Память: выход %.2f ГиБ (VRAM → RAM → диск), свободно RAM %.2f ГиБ; рабочая оценка %.2f ГиБ/кадр.", label, output_bytes / 1024**3, ram / 1024**3, per_frame / 1024**3)

    def cpu_plan():
        """Require CPU workspace only if CPU execution is actually selected."""
        if cpu_batch < 1:
            raise MemoryError(f"{label}: недостаточно свободной RAM для одного рабочего кадра. Освободите память или используйте CUDA/меньшее разрешение.")
        return torch.device("cpu"), cpu_batch, cpu_batch

    if mode == "cpu" or (mode == "auto" and height * width < 256 * 256):
        return cpu_plan()
    if not torch.cuda.is_available():
        if mode == "cuda":
            _LOGGER.warning("[%s] CUDA недоступна: используется CPU.", label)
        return cpu_plan()
    try:
        from comfy import model_management
    except ImportError:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = model_management.get_torch_device()
    if device.type != "cuda":
        return cpu_plan()
    free, _ = torch.cuda.mem_get_info(device)
    available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    # Если целый выход не помещается вместе с рабочим кадром, считаем на GPU,
    # а результат отправляем в RAM/на диск вместо необязательного CPU fallback.
    output_fits = available - reserve - output_bytes >= per_frame
    gpu_output_bytes = output_bytes if output_fits else 0
    host_per_frame = height * width * channels * 4 * 2
    gpu_batch = min(int(batch_size), (available - gpu_output_bytes - reserve) // per_frame, max(0, cpu_budget // host_per_frame))
    _LOGGER.info("[%s] VRAM доступно %.2f ГиБ; порция CUDA %d, предел batch_size=%d.", label, available / 1024**3, max(0, gpu_batch), batch_size)
    if gpu_batch < 1:
        _LOGGER.warning("[%s] Недостаточно свободной VRAM: используется CPU.", label)
        return cpu_plan()
    return device, int(gpu_batch), cpu_batch


def _cpu_fallback_size(images, requested):
    """Recheck CPU workspace after an output may have consumed host RAM."""
    size = min(requested, len(images), max(0, (_available_finish_ram() - 512 * 1024**2) // (images.size(1) * images.size(2) * 512)))
    if size < 1:
        raise MemoryError("Silent Film: после CUDA OOM недостаточно RAM для безопасного CPU fallback.")
    return int(size)


def _finish_chunk(images, gate_x, gate_y, focus_signal, exposure, parameters, frame_offset):
    """Process one bounded chunk while preserving clip-global signals and frame seeds."""
    alpha = images[..., 3:4] if images.size(-1) == 4 else None
    rgb = images[..., :3]
    vignette_strength = parameters.get("vignette_strength", 0.0)
    edge_softness = parameters.get("edge_softness", 0.0)
    edge_start = parameters.get("edge_start", 0.6)
    edge_end = parameters.get("edge_end", 1.0)
    if vignette_strength > 0.0 or edge_softness > 0.0:
        masks = parameters.setdefault("_lens_masks", {})
        key = (rgb.device, rgb.dtype, rgb.size(1), rgb.size(2), edge_start, edge_end)
        if key not in masks:
            masks[key] = _lens_edge_mask(rgb.size(1), rgb.size(2), edge_start, rgb.device, rgb.dtype, edge_end)
        rgb = _apply_lens_edges(rgb, vignette_strength, edge_softness, edge_start, masks[key])
    woven_rgb = _apply_gate_weave(rgb, gate_x, gate_y)
    woven_alpha = _apply_gate_weave(alpha, gate_x, gate_y) if alpha is not None else None
    gray = woven_rgb[..., 0:1] * 0.299 + woven_rgb[..., 1:2] * 0.587 + woven_rgb[..., 2:3] * 0.114
    gray = torch.pow(torch.clamp(gray, 0.0, 1.0), max(0.3, float(parameters["midtone_gamma"])))
    gray = (gray - 0.5) * float(parameters["contrast"]) + 0.5 + float(parameters["black_lift"])
    gray = _soft_clip_highlights(torch.clamp(gray, 0.0, 1.0), parameters["highlight_rolloff"])
    # До тонировки каналы одинаковы: фокус достаточно считать один раз по яркости.
    gray = _apply_focus_drift(
        gray, parameters["softness"], focus_signal, parameters["focus_drift_strength"],
    )
    gray = torch.clamp(gray * exposure.view(-1, 1, 1, 1), 0.0, 1.0)
    toned = _tone_gray(gray, parameters["tone_mode"])
    effective_size = float(parameters["grain_size"])
    grain_strength = parameters["grain_strength"]
    if parameters.get("grain_size_mode", "pixels") == "relative_width":
        effective_size *= toned.size(2) / 1920.0
        # Субпиксельное зерно ослабевает при усреднении по площади пикселя.
        grain_strength *= min(1.0, effective_size)
    toned = _apply_grain(
        toned, parameters["grain_mode"], grain_strength, effective_size,
        parameters["seed"] + 1001, frame_offset=frame_offset,
    )
    if woven_alpha is not None:
        toned = torch.cat([toned, torch.clamp(woven_alpha, 0.0, 1.0)], dim=-1)
    return torch.clamp(toned, 0.0, 1.0)


def _run_finish_chunk(images, start, end, device, gate_x, gate_y, focus_signal, exposure, parameters):
    """Keep all temporary chunk tensors inside a scope released before OOM retries."""
    chunk = images[start:end].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    return _finish_chunk(
        chunk, gate_x[start:end].to(device), gate_y[start:end].to(device),
        focus_signal[start:end].to(device), exposure[start:end].to(device),
        parameters, start,
    )


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
                "vignette_strength": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Плавное затемнение углов объективом до тональной обработки: 0 = выключено, 0.12 = до 12% по RGB в углах. Старт 0.10–0.15. Центр сохраняется; эффект постоянен во времени.",
                    },
                ),
                "edge_softness": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Расфокус относительно ширины: максимальная sigma = ширина × 0.025 × edge_softness; локально умножается на маску. 0 выключает, 0.02–0.04 — деликатный старт, 0.15 уже заметно (7.2 px при 1920; 14.4 px при 3840), 0.5–1 — сильный. Сравнивайте кадры при одинаковом размере показа. Область задаётся edge_start/end. Центр и альфа сохраняются; зерно добавляется после оптики.",
                    },
                ),
                "edge_start": (
                    "FLOAT",
                    {
                        "default": 0.6, "min": 0.0, "max": 4.0, "step": 0.01,
                        "tooltip": "Начало плавного усиления виньетки и мягкости: 0 = от центра, 0.6 = 60% половины ширины от центра; 1 соответствует середине бокового края. Полная сила достигается на edge_end; разность edge_end − edge_start задаёт ширину перехода. Старт 0.6.",
                    },
                ),
                "edge_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01,
                     "tooltip": "Радиус полной силы виньетки и расфокуса; должен быть больше edge_start. 1 = половина ширины кадра. Пример: start 0.8, end 1.2 — переход от 40% до 60% ширины от центра. Больше разность — мягче и шире переход."},
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
                "grain_mode": (
                    ["emulsion_luma", "overlay_fast"],
                    {
                        "default": "emulsion_luma",
                        "tooltip": "Режим зерна. emulsion_luma не просто шумит картинку, а слегка съедает микродеталь и сильнее проявляется в полутонах.",
                    },
                ),
                "grain_size_mode": (
                    ["pixels", "relative_width"],
                    {"default": "pixels", "tooltip": "pixels сохраняет прежний масштаб в пикселях. relative_width масштабирует grain_size относительно ширины 1920 px: размер 3 даёт около 6 px при 3840 и 1 px при 640. Субпиксельное зерно ослабевает. Для новых видео рекомендуется relative_width."},
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
                "grain_strength": (
                    "FLOAT",
                    {
                        "default": 0.032,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.005,
                        "tooltip": "Сила монохромного зерна. В emulsion_luma влияет и на зернистость, и на потерю микродетали.",
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
                "compute_device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": "auto = CUDA для крупных кадров при наличии VRAM; cpu = порционная обработка на CPU; cuda = запрос GPU. При нехватке VRAM порция уменьшается, затем используется CPU.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 4, "min": 1, "max": 16, "step": 1,
                        "tooltip": "Максимум кадров в рабочей порции. 4 — старт; 1–2 — меньше RAM/VRAM. Фактический размер автоматически ограничивается доступной памятью. На число выходных кадров и рисунок зерна не влияет.",
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
        grain_mode,
        grain_strength,
        grain_size,
        seed,
        cadence_json="",
        compute_device="auto",
        batch_size=4,
        vignette_strength=0.0,
        edge_softness=0.0,
        edge_start=0.6,
        grain_size_mode="pixels",
        edge_end=1.0,
    ):
        """Apply silent-film print/projection finishing to a video batch."""
        if grain_size_mode not in ("pixels", "relative_width"):
            raise ValueError("grain_size_mode must be pixels or relative_width")
        started_at = perf_counter()
        _LOGGER.info("[Silent Film Finish] Этап 1/7: проверка входного батча.")
        # Нормализуем только рабочую порцию: здесь не создаём копию всего ролика.
        images = _validate_video_batch(image, normalize=False)
        vignette_strength, edge_softness, edge_start = float(vignette_strength), float(edge_softness), float(edge_start)
        edge_end = float(edge_end)
        for name, value, maximum in (
            ("vignette_strength", vignette_strength, 1.0),
            ("edge_softness", edge_softness, 1.0),
            ("edge_start", edge_start, 4.0),
            ("edge_end", edge_end, 4.0),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}.")

        if edge_end <= edge_start:
            raise ValueError("edge_end must be greater than edge_start.")

        frame_count = int(images.size(0))
        device = torch.device("cpu")
        dtype = torch.float32
        work_device, current_batch_size, cpu_batch_size = _finish_processing_plan(images, str(compute_device), batch_size)
        _LOGGER.info(
            "[Silent Film Finish] Вход: %d кадров, %dx%d; тон=%s, зерно=%s.",
            frame_count, images.size(2), images.size(1), tone_mode, grain_mode,
        )
        _LOGGER.info("[Silent Film Finish] Обработка: %s; рабочая порция: %d кадров.", work_device, current_batch_size)
        if vignette_strength > 0.0 or edge_softness > 0.0:
            _LOGGER.info("[Silent Film Finish] Оптика: виньетка=%.2f, мягкость краёв=%.2f, начало=%.2f, конец=%.2f.", vignette_strength, edge_softness, edge_start, edge_end)

        _LOGGER.info("[Silent Film Finish] Этап 2/7: синхронизация и подготовка сигналов проекции.")
        sync_mode, cadence_meta = _parse_cadence_json(cadence_json, frame_count=frame_count)
        _LOGGER.info(
            "[Silent Film Finish] Синхронизация: %s; %s.",
            sync_mode, "используется cadence_json" if cadence_meta is not None else "независимые сигналы проекции",
        )
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

        exposure = 1.0 + flicker_fast + flicker_slow
        parameters = {
            "tone_mode": tone_mode, "contrast": contrast, "midtone_gamma": midtone_gamma,
            "black_lift": black_lift, "highlight_rolloff": highlight_rolloff,
            "softness": softness, "focus_drift_strength": focus_drift_strength,
            "grain_mode": grain_mode, "grain_strength": grain_strength,
            "grain_size": grain_size, "grain_size_mode": grain_size_mode, "seed": seed,
            "vignette_strength": vignette_strength, "edge_softness": edge_softness, "edge_start": edge_start, "edge_end": edge_end,
        }
        # Один выходной батч вместо списка порций и последующего удвоения памяти через cat.
        output = allocate_video(images.shape, device=images.device, label="Silent Film Finish", work_device=work_device,
                                work_bytes=images.size(1) * images.size(2) * 512 * current_batch_size)
        start = 0
        chunk_count = 0
        _LOGGER.info( "[Silent Film Finish] Этап 3/7: оптика и дрожание проектора; в каждой порции.")
        _LOGGER.info( "[Silent Film Finish] Этап 4/7: ЧБ, контраст, света, мягкость и фокус; в каждой порции.")
        _LOGGER.info( "[Silent Film Finish] Этап 5/7: экспозиция и тонировка; в каждой порции.")
        _LOGGER.info( "[Silent Film Finish] Этап 6/7: зерно %s; в каждой порции.", parameters["grain_mode"])
        with tqdm(total=frame_count, desc="[Silent Film Finish] Обработка", unit="кадр", dynamic_ncols=True, mininterval=0.5) as progress:
            while start < frame_count:
                check_interrupt()
                end = min(frame_count, start + current_batch_size)
                result = None
                try:
                    result = _run_finish_chunk(
                        images, start, end, work_device, gate_x, gate_y, focus_signal, exposure, parameters,
                    )
                    output[start:end].copy_(result)
                    check_interrupt()
                    release_video_pages(output, start, end, written=True)
                    release_video_pages(images, start, end)
                except torch.OutOfMemoryError:
                    result = None
                    if work_device.type != "cuda":
                        raise
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        _log_during_progress(logging.WARNING, "[Silent Film Finish] VRAM закончилась: повтор порции с %d кадров, начиная с кадра %d.", current_batch_size, start + 1)
                    else:
                        if cpu_batch_size < 1:
                            raise MemoryError("Silent Film Finish: после CUDA OOM недостаточно RAM для безопасного CPU fallback.")
                        work_device = torch.device("cpu")
                        current_batch_size = _cpu_fallback_size(images, cpu_batch_size)
                        _log_during_progress(logging.WARNING, "[Silent Film Finish] Один кадр не помещается в VRAM: остаток батча рассчитывается на CPU.")
                    continue
                del result
                chunk_count += 1
                progress.update(end - start)
                start = end
        _LOGGER.info("[Silent Film Finish] Этап 7/7: завершение выходного батча и finish_json.")

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
            "grain_mode": grain_mode,
            "grain_strength": float(grain_strength),
            "grain_size": int(grain_size),
            "grain_size_mode": grain_size_mode,
            "effective_grain_size_px": float(grain_size) * (images.size(2) / 1920.0 if grain_size_mode == "relative_width" else 1.0),
            "vignette_strength": vignette_strength,
            "edge_softness": edge_softness,
            "edge_start": edge_start, "edge_end": edge_end,
            "seed": int(seed),
            "frame_count": frame_count,
            "compute_device": str(compute_device),
            "actual_compute_device": str(work_device),
            "batch_size": int(batch_size),
            "actual_batch_size": current_batch_size,
            "chunk_count": chunk_count,
            "grain_seed_mode": "per_frame",
            "sync_mode": sync_mode,
            "focus_preview": [round(float(x), 4) for x in focus_signal[:16].tolist()],
            "gate_x_preview": [round(float(x), 4) for x in gate_x[:16].tolist()],
            "gate_y_preview": [round(float(y), 4) for y in gate_y[:16].tolist()],
            "exposure_preview": [round(float(x), 4) for x in exposure.view(-1)[:16].tolist()],
        }
        finish_json = json.dumps(payload, ensure_ascii=False, indent=2)
        _LOGGER.info("[Silent Film Finish] Готово: %d кадров за %.1f с.", frame_count, perf_counter() - started_at)
        return (output, finish_json)
