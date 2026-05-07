"""
Module: nodes/video_inpaint.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Remove Static Watermark from Video node implementation.

Purpose:
    Runs crop-first ProPainter-based inpainting and composites restored output back to full frames.
"""

import gc

import glob
import json
import os
import re
import shutil

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from ..utils.interrupt import check_interrupt as shared_check_interrupt
from ..utils.color_match_utils import (
    apply_color_match as _apply_color_match,
    ensure_mask_batch as _ensure_mask_batch,
    normalize_mask as _normalize_mask,
    resize_images_to_size as _resize_images_to_size,
    resize_mask_to_output as _resize_mask_to_output,
)
from ..propainter.propainter_inference import (
    ProPainterConfig,
    feature_propagation,
    process_inpainting,
)
from ..propainter.utils.cudnn_utils import configure_cudnn
from ..propainter.utils.image_utils import (
    ImageConfig,
    convert_image_to_frames,
    handle_output,
    prepare_frames_and_masks,
)
from ..propainter.utils.model_utils import initialize_models


STREAM_CHUNK_DEFAULT = 30
STREAM_START_DEFAULT = 0
STREAM_END_DEFAULT = 0
STREAM_STRIDE_DEFAULT = 1


class _LazyModelManagementProxy:
    """Lazy proxy for Comfy model_management preserving module-level monkeypatching."""

    def __init__(self) -> None:
        object.__setattr__(self, "_overrides", {})

    def _load(self):
        from comfy import model_management as comfy_model_management

        return comfy_model_management

    def __getattribute__(self, name):
        if name in {"_overrides", "_load", "__dict__", "__class__", "__getattr__", "__getattribute__", "__setattr__"}:
            return object.__getattribute__(self, name)
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._load(), name)

    def get_torch_device(self):
        """Resolve Comfy torch device only when runtime actually needs it."""
        return self._load().get_torch_device()

    def __setattr__(self, name, value) -> None:
        overrides = object.__getattribute__(self, "_overrides")
        overrides[name] = value


model_management = _LazyModelManagementProxy()


def _load_folder_paths():
    """Import Comfy folder_paths lazily to keep module import headless-safe."""
    import folder_paths as comfy_folder_paths

    return comfy_folder_paths


def _check_interrupt() -> None:
    """Raise an interrupt error when ComfyUI requests execution cancellation."""
    shared_check_interrupt()


def _masked_output_looks_invalid(
    output_images: torch.Tensor,
    source_images: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    """Detect obviously broken inpaint output inside the masked region.

    FP16 inference on some devices can collapse the inpaint region to near-black.
    We only trigger fallback when the masked area is overwhelmingly dark compared to
    the corresponding source region or contains non-finite values.
    """
    if output_images.numel() == 0:
        return True
    if not torch.isfinite(output_images).all():
        return True

    mask = _normalize_mask(mask)
    mask = _ensure_mask_batch(mask, output_images.size(dim=0))
    mask = _resize_mask_to_output(mask, output_images.shape[1], output_images.shape[2])
    if source_images.shape[1:3] != output_images.shape[1:3]:
        source_images = _resize_images_to_size(
            source_images, output_images.shape[1], output_images.shape[2]
        )
    selected = mask > 0.5
    if not bool(selected.any()):
        return False

    selected_rgb = selected.unsqueeze(-1).expand_as(output_images)
    masked_output = output_images[selected_rgb]
    masked_source = source_images[selected_rgb]
    if masked_output.numel() == 0 or masked_source.numel() == 0:
        return False

    out_mean = float(masked_output.mean().item())
    src_mean = float(masked_source.mean().item())
    near_black_ratio = float((masked_output < 0.02).float().mean().item())
    dynamic_range = float((masked_output.max() - masked_output.min()).item())

    if near_black_ratio > 0.985 and src_mean > 0.05:
        return True
    if out_mean < 0.02 and src_mean > 0.08:
        return True
    if dynamic_range < 0.01 and out_mean < 0.05 and src_mean > 0.08:
        return True
    return False


def _updated_frames_to_images(updated_frames: torch.Tensor) -> torch.Tensor:
    """Convert ProPainter updated frames from [-1, 1] BTCHW to [0, 1] THWC."""
    if updated_frames.dim() != 5:
        raise ValueError(f"Expected updated_frames in BTCHW layout, got shape={tuple(updated_frames.shape)}")
    frames = updated_frames[0].detach().float().clamp(-1.0, 1.0)
    frames = (frames + 1.0) * 0.5
    return frames.permute(0, 2, 3, 1).contiguous()


def _check_inputs(frames: torch.Tensor, masks: torch.Tensor) -> None:
    """Validate frame and mask tensor shapes before running inpaint pipeline."""
    if frames.size(dim=0) <= 1:
        raise ValueError(f"Image length must be greater than 1, but got: {frames.size(dim=0)}")
    if masks.size(dim=0) != 1 and frames.size(dim=0) != masks.size(dim=0):
        raise ValueError(
            "Image and Mask must have the same length or Mask length must be 1. "
            f"Got Image: {frames.size(dim=0)}, Mask: {masks.size(dim=0)}"
        )
    if frames.size(dim=1) != masks.size(dim=1) or frames.size(dim=2) != masks.size(dim=2):
        raise ValueError(
            "Image and Mask must have the same spatial dimensions. "
            f"Got Image: ({frames.size(dim=1)}, {frames.size(dim=2)}), "
            f"Mask: ({masks.size(dim=1)}, {masks.size(dim=2)})"
        )


def _mask_to_bbox(mask: torch.Tensor) -> tuple[int, int, int, int, str]:
    """Convert binary mask into an expanded bounding box for inpaint cropping."""
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 2:
        mask_np = mask_np[np.newaxis, ...]
    union = mask_np.max(axis=0)
    ys, xs = np.where(union > 0.05)
    height, width = union.shape
    if xs.size == 0 or ys.size == 0:
        return 0, 0, width, height, "empty_mask"
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1, "ok"


def _pre_crop_inputs(
    frames: torch.Tensor,
    mask: torch.Tensor,
    padding: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int], str]:
    """Prepare frames and mask tensors for crop-first inpaint processing."""
    mask = _normalize_mask(mask)
    mask = _ensure_mask_batch(mask, frames.size(dim=0))
    x0, y0, x1, y1, status = _mask_to_bbox(mask)

    height = int(frames.shape[1])
    width = int(frames.shape[2])
    if padding > 0:
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(width, x1 + padding)
        y1 = min(height, y1 + padding)

    cropped_frames = frames[:, y0:y1, x0:x1, :]
    cropped_mask = mask[:, y0:y1, x0:x1]
    return cropped_frames, cropped_mask, (x0, y0, x1, y1), status


def _crop_frames_with_bbox(
    frames: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a frame batch to the selected bounding box region."""
    x0, y0, x1, y1 = bbox
    return frames[:, y0:y1, x0:x1, :], mask[:, y0:y1, x0:x1]




def _sanitize_prefix(prefix: str) -> str:
    """Sanitize filename prefix to keep saved outputs filesystem-safe."""
    if prefix is None:
        return "patch_"
    prefix = prefix.strip()
    return prefix if prefix else "patch_"


def _sanitize_prefix_with_default(prefix: str, default: str) -> str:
    """Sanitize filename prefix and fallback to a default when empty."""
    if prefix is None:
        return default
    prefix = prefix.strip()
    return prefix if prefix else default


def _feather_alpha_mask(alpha_mask: torch.Tensor, feather_radius: int) -> torch.Tensor:
    """Return a softened alpha mask for patch compositing.

    The input mask uses the node convention: white = replace. Feathering is applied
    only inside the selected region so the exterior remains fully transparent.
    """
    alpha_mask = alpha_mask.detach().float().cpu()
    if alpha_mask.dim() == 2:
        alpha_mask = alpha_mask.unsqueeze(0)
    if feather_radius <= 0:
        return alpha_mask.clamp(0.0, 1.0)

    feather_radius = int(max(1, feather_radius))
    out = []
    for idx in range(alpha_mask.size(0)):
        binary = (alpha_mask[idx].numpy() > 0.5).astype(np.uint8)
        if binary.max() == 0:
            out.append(torch.from_numpy(binary.astype(np.float32)))
            continue
        dist_in = ndimage.distance_transform_edt(binary)
        soft = np.clip(dist_in / float(feather_radius), 0.0, 1.0).astype(np.float32)
        out.append(torch.from_numpy(soft))
    return torch.stack(out, dim=0)


def _ensure_dir(path: str) -> None:
    """Create directory (and parents) when it does not exist yet."""
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _save_rgba_sequence(
    rgb_frames: torch.Tensor,
    alpha_mask: torch.Tensor,
    output_dir: str,
    prefix: str,
    label: str,
    start_index: int = 0,
    feather_radius: int = 0,
) -> None:
    """Save RGBA tensor batch to numbered PNG sequence on disk."""
    if not output_dir:
        return
    _ensure_dir(output_dir)
    prefix = _sanitize_prefix(prefix)

    rgb_frames = rgb_frames.detach().cpu()
    alpha_mask = _feather_alpha_mask(alpha_mask, feather_radius)
    if alpha_mask.dim() == 2:
        alpha_mask = alpha_mask.unsqueeze(0)
    if alpha_mask.size(0) == 1 and rgb_frames.size(0) > 1:
        alpha_mask = alpha_mask.repeat(rgb_frames.size(0), 1, 1)

    for idx in range(rgb_frames.size(0)):
        rgb = rgb_frames[idx].clamp(0.0, 1.0).numpy()
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        alpha = alpha_mask[idx].clamp(0.0, 1.0).numpy()
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        rgba_u8 = (rgba * 255.0).round().astype(np.uint8)
        filename = f"{prefix}{label}{idx + start_index:04d}.png"
        Image.fromarray(rgba_u8).save(os.path.join(output_dir, filename))


def _save_rgba_frame(
    rgb_u8: np.ndarray,
    alpha_u8: np.ndarray,
    output_dir: str,
    filename: str,
) -> None:
    """Save a single RGBA frame tensor as PNG file."""
    if not output_dir:
        return
    _ensure_dir(output_dir)
    rgba = np.concatenate([rgb_u8, alpha_u8[..., None]], axis=-1)
    Image.fromarray(rgba).save(os.path.join(output_dir, filename))


def _save_rgb_frame(
    rgb_u8: np.ndarray,
    output_dir: str,
    filename: str,
) -> None:
    """Save a single RGB frame tensor as PNG file."""
    if not output_dir:
        return
    _ensure_dir(output_dir)
    Image.fromarray(rgb_u8).save(os.path.join(output_dir, filename))


def _save_mask_frame(
    mask_u8: np.ndarray,
    output_dir: str,
    filename: str,
) -> None:
    """Save a single mask tensor as grayscale PNG file."""
    if not output_dir:
        return
    _ensure_dir(output_dir)
    Image.fromarray(mask_u8).save(os.path.join(output_dir, filename))


def _save_transform_json(output_dir: str, prefix: str, transform_json: str) -> None:
    """Save crop/transform metadata to JSON file for downstream tools."""
    if not output_dir:
        return
    _ensure_dir(output_dir)
    prefix = _sanitize_prefix(prefix)
    filename = f"{prefix}transform.json"
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
        handle.write(transform_json)


def _list_numbered_frames(directory: str, prefix: str, label: str) -> list[str]:
    """List numerically ordered frame files from a directory."""
    if not directory:
        return []
    prefix = _sanitize_prefix(prefix)
    pattern = os.path.join(directory, f"{prefix}{label}*.png")
    entries = []
    regex = re.compile(rf"^{re.escape(prefix)}{re.escape(label)}(\d+)\.png$")
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        match = regex.match(name)
        if not match:
            continue
        entries.append((int(match.group(1)), path))
    entries.sort(key=lambda item: item[0])
    return [path for _, path in entries]


def _purge_cached_inputs(directory: str, prefix: str) -> None:
    """Remove stale cached frame files before writing new input data."""
    if not directory:
        return
    prefix = _sanitize_prefix(prefix)
    patterns = [
        os.path.join(directory, f"{prefix}input_*.png"),
        os.path.join(directory, f"{prefix}mask_*.png"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                continue


def _load_rgba_frame(path: str) -> np.ndarray:
    """Load RGBA image file from disk into normalized torch tensor."""
    with Image.open(path) as img:
        img = img.convert("RGBA")
        return np.asarray(img, dtype=np.uint8)


def _build_preview_composite(
    cache_dir: str,
    output_dir: str,
    prefix: str,
    index: int,
) -> torch.Tensor | None:
    """Build side-by-side preview showing source, mask, and inpaint result."""
    if not cache_dir or not output_dir:
        return None
    prefix = _sanitize_prefix(prefix)
    input_path = os.path.join(cache_dir, f"{prefix}input_{index:04d}.png")
    output_path = os.path.join(output_dir, f"{prefix}{index:04d}.png")
    if not os.path.exists(input_path) or not os.path.exists(output_path):
        return None

    input_rgba = _load_rgba_frame(input_path).astype(np.float32) / 255.0
    output_rgba = _load_rgba_frame(output_path).astype(np.float32) / 255.0

    alpha = output_rgba[..., 3:4]
    comp_rgb = output_rgba[..., :3] * alpha + input_rgba[..., :3] * (1.0 - alpha)
    comp_rgba = np.concatenate([comp_rgb, np.ones_like(alpha)], axis=-1)

    preview_image = torch.from_numpy(comp_rgba).unsqueeze(0)
    return preview_image


def _stream_write_fullframes(
    video_path: str,
    output_dir: str,
    output_name: str,
    fullframe_prefix: str,
    bbox: tuple[int, int, int, int],
    pre_crop: bool,
    stream_start: int,
    stream_end: int,
    stream_stride: int,
    total_frames: int | None,
) -> None:
    """Stream ffmpeg output directly into the full-frame video writer."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for fullframe output.") from exc

    if not output_dir:
        return

    patch_prefix = _sanitize_prefix(output_name)
    full_prefix = _sanitize_prefix_with_default(fullframe_prefix, "fullframe_")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    if stream_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, stream_start)

    frame_index = stream_start
    patch_index = 0
    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(total=total_frames, desc="Fullframe composite", leave=False)
    except Exception:
        progress = None

    try:
        while True:
            _check_interrupt()
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if stream_end > 0 and frame_index >= stream_end:
                break
            if (frame_index - stream_start) % stream_stride != 0:
                frame_index += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            patch_path = os.path.join(output_dir, f"{patch_prefix}{patch_index:04d}.png")
            if os.path.exists(patch_path):
                patch_rgba = _load_rgba_frame(patch_path)
                patch_rgb = patch_rgba[..., :3].astype(np.float32)
                alpha = patch_rgba[..., 3:4].astype(np.float32) / 255.0

                if pre_crop:
                    x0, y0, x1, y1 = bbox
                    region = frame_rgb[y0:y1, x0:x1, :].astype(np.float32)
                    if patch_rgb.shape[:2] != region.shape[:2]:
                        patch_rgb = cv2.resize(
                            patch_rgb, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_LINEAR
                        )
                        alpha = cv2.resize(
                            alpha, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                        alpha = alpha[..., None] if alpha.ndim == 2 else alpha
                    region = patch_rgb * alpha + region * (1.0 - alpha)
                    frame_rgb[y0:y1, x0:x1, :] = np.clip(region, 0.0, 255.0).astype(np.uint8)
                else:
                    if patch_rgb.shape[:2] != frame_rgb.shape[:2]:
                        patch_rgb = cv2.resize(
                            patch_rgb, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
                        )
                        alpha = cv2.resize(
                            alpha, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                        alpha = alpha[..., None] if alpha.ndim == 2 else alpha
                    frame_rgb = np.clip(
                        patch_rgb * alpha + frame_rgb.astype(np.float32) * (1.0 - alpha),
                        0.0,
                        255.0,
                    ).astype(np.uint8)

            out_name = f"{full_prefix}{patch_index:04d}.png"
            _save_rgb_frame(frame_rgb, output_dir, out_name)

            patch_index += 1
            frame_index += 1
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        cap.release()


def _format_resolve_edit_position(
    norm_x: float,
    norm_y: float,
    background_width: int,
    background_height: int,
    overlay_width: int,
    overlay_height: int,
) -> dict:
    """Format frame index and fps to Resolve-compatible timeline position."""
    scale_x = (background_width * background_width) / max(1.0, float(overlay_width))
    scale_y = (background_height * background_height) / max(1.0, float(overlay_height))
    pos_x = (norm_x - 0.5) * scale_x
    pos_y = (norm_y - 0.5) * scale_y
    return {"x": round(float(pos_x), 3), "y": round(float(pos_y), 3)}


def _format_crop_json(
    center_x: float,
    center_y: float,
    crop_width: int,
    crop_height: int,
    background_width: int,
    background_height: int,
    status: str,
) -> str:
    """Build JSON payload describing crop transform and output placement."""
    payload = {
        "status": status,
        "overlay_scale": {"x": None, "y": None},
        "overlay_rotation_angle": None,
        "overlay_position_pixels": {"x": None, "y": None},
        "overlay_position": {"x": None, "y": None},
        "fusion_position": {"x": None, "y": None},
        "resolve_position_edit": {"x": None, "y": None},
        "pixel_center": {"top_left": {"x": None, "y": None}, "center": {"x": None, "y": None}},
        "normalized_center": {"top_left": {"x": None, "y": None}, "bottom_left": {"x": None, "y": None}},
        "color_space": "srgb",
        "alpha_mode": "straight",
        "levels": "full",
    }
    if status != "ok":
        return json.dumps(payload, ensure_ascii=True)

    bg_w = max(1.0, float(background_width))
    bg_h = max(1.0, float(background_height))
    norm_x = float(center_x / bg_w)
    norm_y_bottom = float(1.0 - (center_y / bg_h))
    norm_y_top = float(center_y / bg_h)
    center_origin_x = float(center_x - (bg_w * 0.5))
    center_origin_y = float(center_y - (bg_h * 0.5))

    payload.update({
        "overlay_scale": {"x": 1.0, "y": 1.0},
        "overlay_rotation_angle": 0.0,
        "overlay_position_pixels": {"x": round(float(center_x), 3), "y": round(float(center_y), 3)},
        "overlay_position": {"x": round(norm_x, 6), "y": round(norm_y_bottom, 6)},
        "fusion_position": {"x": round(norm_x, 6), "y": round(norm_y_bottom, 6)},
        "resolve_position_edit": _format_resolve_edit_position(
            norm_x,
            norm_y_bottom,
            background_width,
            background_height,
            crop_width,
            crop_height,
        ),
        "pixel_center": {
            "top_left": {"x": round(float(center_x), 3), "y": round(float(center_y), 3)},
            "center": {"x": round(center_origin_x, 3), "y": round(center_origin_y, 3)},
        },
        "normalized_center": {
            "top_left": {"x": round(norm_x, 6), "y": round(norm_y_top, 6)},
            "bottom_left": {"x": round(norm_x, 6), "y": round(norm_y_bottom, 6)},
        },
    })
    return json.dumps(payload, ensure_ascii=True)


def _crop_outputs(
    output_images: torch.Tensor,
    mask: torch.Tensor,
    background_width: int,
    background_height: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Extract crop outputs and metadata from inpaint pipeline results."""
    mask = _normalize_mask(mask)
    mask = _ensure_mask_batch(mask, output_images.size(dim=0))
    mask = _resize_mask_to_output(mask, background_height, background_width)

    x0, y0, x1, y1, status = _mask_to_bbox(mask)
    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)

    cropped_images = output_images[:, y0:y1, x0:x1, :]
    cropped_mask = mask[:, y0:y1, x0:x1].clamp(0.0, 1.0)
    alpha = cropped_mask.unsqueeze(-1)
    rgba = torch.cat([cropped_images, alpha], dim=-1)

    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    transform_json = _format_crop_json(
        center_x,
        center_y,
        crop_width,
        crop_height,
        background_width,
        background_height,
        status,
    )
    return rgba, cropped_mask, transform_json


def _compose_outputs_from_bbox(
    output_images: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int],
    background_width: int,
    background_height: int,
    status: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Composite cropped inpaint result back onto full-size output frames."""
    mask = _normalize_mask(mask)
    mask = _ensure_mask_batch(mask, output_images.size(dim=0))
    mask = _resize_mask_to_output(mask, output_images.shape[1], output_images.shape[2])
    mask = mask.clamp(0.0, 1.0)

    alpha = mask.unsqueeze(-1)
    rgba = torch.cat([output_images, alpha], dim=-1)

    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    transform_json = _format_crop_json(
        center_x,
        center_y,
        output_images.shape[2],
        output_images.shape[1],
        background_width,
        background_height,
        status,
    )
    return rgba, mask, transform_json


class VideoInpaintWatermark:
    """ComfyUI node that removes static watermarks from video frames."""
    def _run_propainter_chunk(
        self,
        frames_np: np.ndarray,
        frames: torch.Tensor,
        mask_full: torch.Tensor,
        image_config: ImageConfig,
        inpaint_config: ProPainterConfig,
        throughput_mode: str,
    ) -> torch.Tensor:
        """Run one ProPainter pass and return output images in [0, 1] NHWC."""
        device = inpaint_config.device
        frames_tensor, flow_masks_tensor, masks_dilated_tensor, original_frames = (
            prepare_frames_and_masks(frames_np, mask_full, image_config, device)
        )
        models = initialize_models(device, inpaint_config.fp16)
        updated_frames, updated_masks, pred_flows_bi = process_inpainting(
            models,
            frames_tensor,
            flow_masks_tensor,
            masks_dilated_tensor,
            inpaint_config,
        )
        propagated_images = None
        if updated_frames.dim() == 5 and updated_frames.numel() > 0:
            propagated_images = _updated_frames_to_images(updated_frames)
        composed_frames = feature_propagation(
            models.inpaint_model,
            updated_frames,
            updated_masks,
            masks_dilated_tensor,
            pred_flows_bi,
            original_frames,
            inpaint_config,
        )
        output_images, _flow_masks, _masks_dilated = handle_output(
            composed_frames, flow_masks_tensor, masks_dilated_tensor
        )
        if (
            propagated_images is not None
            and _masked_output_looks_invalid(output_images, frames, mask_full)
            and not _masked_output_looks_invalid(propagated_images, frames, mask_full)
        ):
            print("[VideoInpaint] feature_propagation output looked invalid; using image_propagation result for this chunk.")
            output_images = propagated_images
        del frames_tensor, flow_masks_tensor, masks_dilated_tensor, updated_frames, updated_masks, pred_flows_bi
        if torch.cuda.is_available() and throughput_mode != "enable":
            torch.cuda.empty_cache()
        return output_images

    def _stream_video(
        self,
        video: str,
        mask: torch.Tensor,
        mask_dilates: int,
        flow_mask_dilates: int,
        ref_stride: int,
        neighbor_length: int,
        subvideo_length: int,
        raft_iter: int,
        fp16: str,
        throughput_mode: str,
        cudnn_benchmark: str,
        tf32: str,
        crop_padding: int,
        feather_radius: int,
        color_match_mode: str,
        cache_dir: str,
        output_dir: str,
        output_name: str,
        preview_frame: int,
        write_fullframes: bool,
        fullframe_prefix: str,
    ):
        """Stream the source video through ffmpeg and process frames chunk-by-chunk."""
        pre_crop = True
        width = 0
        height = 0
        stream_chunk = STREAM_CHUNK_DEFAULT
        stream_start = STREAM_START_DEFAULT
        stream_end = STREAM_END_DEFAULT
        stream_stride = STREAM_STRIDE_DEFAULT
        if not video:
            raise ValueError("video is required.")
        if not output_dir:
            raise ValueError("output_dir is required.")
        if not cache_dir:
            raise ValueError("cache_dir is required.")
        folder_paths = _load_folder_paths()
        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video}")
        _purge_cached_inputs(cache_dir, output_name)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video processing.") from exc

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        if stream_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, stream_start)

        frame_index = stream_start
        cache_index = 0
        full_width = None
        full_height = None
        bbox = None
        status = "ok"
        transform_json = None
        mask_crop_u8 = None

        try:
            while True:
                _check_interrupt()
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if stream_end > 0 and frame_index >= stream_end:
                    break
                if (frame_index - stream_start) % stream_stride != 0:
                    frame_index += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if full_width is None:
                    full_height, full_width = frame_rgb.shape[:2]
                    mask_full = _normalize_mask(mask)
                    mask_full = _ensure_mask_batch(mask_full, 1)
                    mask_full = _resize_mask_to_output(mask_full, full_height, full_width)
                    x0, y0, x1, y1, status = _mask_to_bbox(mask_full)
                    if status == "ok":
                        if pre_crop:
                            x0 = max(0, x0 - crop_padding)
                            y0 = max(0, y0 - crop_padding)
                            x1 = min(full_width, x1 + crop_padding)
                            y1 = min(full_height, y1 + crop_padding)
                    else:
                        x0, y0, x1, y1 = 0, 0, full_width, full_height

                    bbox = (x0, y0, x1, y1)
                    center_x = (x0 + x1) * 0.5
                    center_y = (y0 + y1) * 0.5
                    crop_width = max(1, x1 - x0)
                    crop_height = max(1, y1 - y0)
                    transform_json = _format_crop_json(
                        center_x,
                        center_y,
                        crop_width,
                        crop_height,
                        full_width,
                        full_height,
                        status,
                    )
                    _save_transform_json(output_dir, output_name, transform_json)

                    mask_np = mask_full[0].clamp(0.0, 1.0).cpu().numpy()
                    if pre_crop:
                        mask_np = mask_np[y0:y1, x0:x1]
                    mask_crop_u8 = (mask_np * 255.0).round().astype(np.uint8)

                if pre_crop:
                    x0, y0, x1, y1 = bbox
                    frame_rgb = frame_rgb[y0:y1, x0:x1, :]

                prefix = _sanitize_prefix(output_name)
                rgb_name = f"{prefix}input_{cache_index:04d}.png"
                mask_name = f"{prefix}mask_{cache_index:04d}.png"
                _save_rgb_frame(frame_rgb, cache_dir, rgb_name)
                _save_mask_frame(mask_crop_u8, cache_dir, mask_name)
                cache_index += 1
                frame_index += 1
        finally:
            cap.release()

        if cache_index == 0:
            dummy_image = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
            return (dummy_image, transform_json or "{}")

        if status == "empty_mask":
            prefix = _sanitize_prefix(output_name)
            for path in _list_numbered_frames(cache_dir, prefix, "input_"):
                name = os.path.basename(path)
                mask_path = os.path.join(
                    cache_dir, name.replace(f"{prefix}input_", f"{prefix}mask_")
                )
                rgb = _load_rgba_frame(path)[..., :3]
                if os.path.exists(mask_path):
                    with Image.open(mask_path) as img:
                        mask_u8 = np.asarray(img.convert("L"), dtype=np.uint8)
                else:
                    mask_u8 = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
                out_name = name.replace(f"{prefix}input_", prefix)
                _save_rgba_frame(rgb, mask_u8, output_dir, out_name)
            preview = None
            if preview_frame >= 0:
                preview_index = max(0, min(cache_index - 1, preview_frame))
                preview = _build_preview_composite(cache_dir, output_dir, output_name, preview_index)
            dummy_image = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
            if preview is not None:
                return (preview, transform_json or "{}")
            return (dummy_image, transform_json or "{}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prefix = _sanitize_prefix(output_name)
        cache_files = _list_numbered_frames(cache_dir, output_name, "input_")
        global_index = 0

        for idx in range(0, len(cache_files), max(1, stream_chunk)):
            _check_interrupt()
            chunk_paths = cache_files[idx : idx + max(1, stream_chunk)]
            frames_list = []
            masks_list = []
            for path in chunk_paths:
                name = os.path.basename(path)
                mask_path = os.path.join(
                    cache_dir, name.replace(f"{prefix}input_", f"{prefix}mask_")
                )
                rgb = _load_rgba_frame(path)[..., :3]
                if os.path.exists(mask_path):
                    with Image.open(mask_path) as img:
                        mask_u8 = np.asarray(img.convert("L"), dtype=np.uint8)
                else:
                    mask_u8 = _load_rgba_frame(path)[..., 3]
                frames_list.append(rgb)
                masks_list.append(mask_u8)

            mask_np = np.stack(masks_list, axis=0).astype(np.float32) / 255.0
            frames_chunk = frames_list
            mask_chunk = torch.from_numpy(mask_np)

            global_index = self._process_stream_chunk(
                frames_chunk,
                mask_chunk,
                mask_dilates,
                flow_mask_dilates,
                ref_stride,
                neighbor_length,
                subvideo_length,
                raft_iter,
                fp16,
                throughput_mode,
                cudnn_benchmark,
                tf32,
                pre_crop,
                feather_radius,
                color_match_mode,
                bbox,
                status,
                full_width,
                full_height,
                cache_dir="",
                output_dir=output_dir,
                output_name=output_name,
                start_index=global_index,
                inputs_already_cropped=pre_crop,
                mask_is_ready=True,
            )
            gc.collect()

        preview = None
        if preview_frame >= 0:
            preview_index = max(0, min(cache_index - 1, preview_frame))
            preview = _build_preview_composite(cache_dir, output_dir, output_name, preview_index)

        if write_fullframes:
            _stream_write_fullframes(
                video_path=video_path,
                output_dir=output_dir,
                output_name=output_name,
                fullframe_prefix=fullframe_prefix,
                bbox=bbox,
                pre_crop=pre_crop,
                stream_start=stream_start,
                stream_end=stream_end,
                stream_stride=stream_stride,
                total_frames=cache_index,
            )

        dummy_image = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
        if preview is not None:
            return (preview, transform_json or "{}")
        return (dummy_image, transform_json or "{}")

    def _process_stream_chunk(
        self,
        frames_buf: list[np.ndarray],
        mask: torch.Tensor,
        mask_dilates: int,
        flow_mask_dilates: int,
        ref_stride: int,
        neighbor_length: int,
        subvideo_length: int,
        raft_iter: int,
        fp16: str,
        throughput_mode: str,
        cudnn_benchmark: str,
        tf32: str,
        pre_crop: bool,
        feather_radius: int,
        color_match_mode: str,
        bbox: tuple[int, int, int, int],
        status: str,
        full_width: int,
        full_height: int,
        cache_dir: str,
        output_dir: str,
        output_name: str,
        start_index: int,
        inputs_already_cropped: bool = False,
        mask_is_ready: bool = False,
    ) -> int:
        """Process one streamed frame chunk through inpaint and compositing steps."""
        _check_interrupt()
        width = 0
        height = 0
        frames_np = np.stack(frames_buf, axis=0).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames_np)
        save_count = frames.size(dim=0)

        mask_full = _normalize_mask(mask)
        mask_full = _ensure_mask_batch(mask_full, frames.size(dim=0))
        if not mask_is_ready:
            mask_full = _resize_mask_to_output(mask_full, full_height, full_width)

        if save_count == 1:
            frames = frames.repeat(2, 1, 1, 1)
            mask_full = mask_full.repeat(2, 1, 1)

        if pre_crop and not inputs_already_cropped:
            frames, mask_full = _crop_frames_with_bbox(frames, mask_full, bbox)

        if cache_dir and pre_crop and not inputs_already_cropped:
            _save_rgba_sequence(frames, mask_full, cache_dir, output_name, "input_", start_index)

        _check_inputs(frames, mask_full)
        device = model_management.get_torch_device()

        if cudnn_benchmark != "default" or tf32 != "default":
            configure_cudnn(
                benchmark=cudnn_benchmark == "enable",
                allow_tf32=tf32 == "enable",
            )

        width = int(frames.shape[2])
        height = int(frames.shape[1])

        frames_np = convert_image_to_frames(frames)
        video_length = frames.size(dim=0)
        input_size = (frames_np[0].shape[1], frames_np[0].shape[0])

        image_config = ImageConfig(
            width, height, mask_dilates, flow_mask_dilates, input_size, video_length
        )
        inpaint_config = ProPainterConfig(
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
            fp16=fp16,
            video_length=video_length,
            device=device,
            process_size=image_config.process_size,
            skip_empty_cache=throughput_mode == "enable",
        )
        output_images = self._run_propainter_chunk(
            frames_np=frames_np,
            frames=frames,
            mask_full=mask_full,
            image_config=image_config,
            inpaint_config=inpaint_config,
            throughput_mode=throughput_mode,
        )

        if fp16 == "enable" and _masked_output_looks_invalid(output_images, frames, mask_full):
            print("[VideoInpaint] FP16 chunk output looked invalid inside mask; retrying in FP32.")
            retry_config = ProPainterConfig(
                ref_stride=ref_stride,
                neighbor_length=neighbor_length,
                subvideo_length=subvideo_length,
                raft_iter=raft_iter,
                fp16="disable",
                video_length=video_length,
                device=device,
                process_size=image_config.process_size,
                skip_empty_cache=throughput_mode == "enable",
            )
            output_images = self._run_propainter_chunk(
                frames_np=frames_np,
                frames=frames,
                mask_full=mask_full,
                image_config=image_config,
                inpaint_config=retry_config,
                throughput_mode=throughput_mode,
            )

        if pre_crop:
            crop_width = max(1, bbox[2] - bbox[0])
            crop_height = max(1, bbox[3] - bbox[1])
            output_images = _resize_images_to_size(output_images, crop_height, crop_width)
            output_images = _apply_color_match(output_images, frames, mask_full, color_match_mode)
            rgba, out_mask, _transform_json = _compose_outputs_from_bbox(
                output_images,
                mask_full,
                bbox,
                full_width,
                full_height,
                status,
            )
        else:
            output_images = _apply_color_match(output_images, frames, mask_full, color_match_mode)
            rgba, out_mask, _transform_json = _crop_outputs(
                output_images,
                mask_full,
                output_images.shape[2],
                output_images.shape[1],
            )

        if output_dir:
            if save_count < rgba.size(dim=0):
                rgba = rgba[:save_count]
                out_mask = out_mask[:save_count]
            _save_rgba_sequence(
                rgba[..., :3],
                out_mask,
                output_dir,
                output_name,
                "",
                start_index,
                feather_radius=feather_radius,
            )
        return start_index + save_count

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        folder_paths = _load_folder_paths()
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Маска для удаления (1 кадр или batch)."}),
                "mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение маски (0-100, типично 4–12)."}),
                "flow_mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение flow-маски (0-100, типично 4–12)."}),
                "ref_stride": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Шаг опорных кадров (1-100, типично 5–15)."}),
                "neighbor_length": ("INT", {"default": 10, "min": 2, "max": 300, "tooltip": "Соседние кадры (2-300, типично 5–15)."}),
                "subvideo_length": ("INT", {"default": 80, "min": 1, "max": 300, "tooltip": "Длина подвидео (1-300, типично 40–120)."}),
                "raft_iter": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "Итерации RAFT (1-100, типично 10–30)."}),
                "fp16": (["enable", "disable"], {"default": "disable", "tooltip": "FP16 режим. `disable` безопаснее; `enable` быстрее/меньше VRAM, но на части GPU может давать черный masked-area результат."}),
                "throughput_mode": (["enable", "disable"], {"default": "disable", "tooltip": "Пропускать очистку кэша GPU (enable быстрее, но больше памяти)."}),
                "cudnn_benchmark": (["default", "enable", "disable"], {"default": "default", "tooltip": "cuDNN benchmark (enable быстрее при фикс. размере)."}),
                "tf32": (["default", "enable", "disable"], {"default": "default", "tooltip": "TF32 матмулы (enable быстрее, чуть менее точно)."}),
                "crop_padding": ("INT", {"default": 64, "min": 0, "max": 512, "tooltip": "Паддинг вокруг маски (0-512). Для watermark-removal безопасный старт часто 64+, иначе masked-area может деградировать в черный patch."}),
                "feather_radius": ("INT", {"default": 12, "min": 0, "max": 256, "tooltip": "Мягкость края заплаты в пикселях. Убирает заметный контур patch при композите."}),
                "color_match_mode": (["none", "mean_std", "linear", "hist", "lab_l", "lab_l_cdf", "lab_full", "lab_cdf"], {"default": "none", "tooltip": "Подгонка цвета по чистой зоне (mean_std/linear/hist/lab_l/lab_l_cdf/lab_full/lab_cdf)."}),
                "cache_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "Папка для кэша обрезанного входа (RGB+mask). Обязательна."}),
                "output_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "Папка для сохранения результата (PNG с альфой). Обязательна."}),
                "output_name": ("STRING", {"default": "patch_", "multiline": False, "tooltip": "Префикс имени файлов (например: patch_ -> patch_0000.png)."}),
                "video": (sorted(files), {"video_upload": True, "tooltip": "Видео из папки input (Upload для добавления)."}),
                "preview_frame": ("INT", {"default": 0, "min": -1, "max": 1000000, "tooltip": "Кадр для превью композита (0 = первый, -1 = не выводить)."}),
                "write_fullframes": ("BOOLEAN", {"default": False, "tooltip": "Записать полные кадры с наложенным патчем."}),
                "fullframe_prefix": ("STRING", {"default": "fullframe_", "multiline": False, "tooltip": "Префикс для полных кадров (fullframe_0000.png)."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "transform_json")
    FUNCTION = "inpaint"
    CATEGORY = "video/inpaint"

    def inpaint(
        self,
        mask: torch.Tensor,
        mask_dilates: int,
        flow_mask_dilates: int,
        ref_stride: int,
        neighbor_length: int,
        subvideo_length: int,
        raft_iter: int,
        fp16: str,
        throughput_mode: str,
        cudnn_benchmark: str,
        tf32: str,
        crop_padding: int,
        feather_radius: int,
        color_match_mode: str,
        cache_dir: str,
        output_dir: str,
        output_name: str,
        video: str,
        preview_frame: int,
        write_fullframes: bool,
        fullframe_prefix: str,
    ):
        """Run the watermark inpainting pipeline and return video/image outputs."""
        return self._stream_video(
            video=video,
            mask=mask,
            mask_dilates=mask_dilates,
            flow_mask_dilates=flow_mask_dilates,
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
            fp16=fp16,
            throughput_mode=throughput_mode,
            cudnn_benchmark=cudnn_benchmark,
            tf32=tf32,
            crop_padding=crop_padding,
            feather_radius=feather_radius,
            color_match_mode=color_match_mode,
            cache_dir=cache_dir,
            output_dir=output_dir,
            output_name=output_name,
            preview_frame=preview_frame,
            write_fullframes=write_fullframes,
            fullframe_prefix=fullframe_prefix,
        )
