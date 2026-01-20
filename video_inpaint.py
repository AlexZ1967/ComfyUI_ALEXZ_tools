import json
import logging

import numpy as np
import torch
from comfy import model_management

from .propainter.propainter_inference import (
    ProPainterConfig,
    feature_propagation,
    process_inpainting,
)
from .propainter.utils.cudnn_utils import configure_cudnn
from .propainter.utils.image_utils import (
    ImageConfig,
    convert_image_to_frames,
    handle_output,
    prepare_frames_and_masks,
)
from .propainter.utils.model_utils import initialize_models
from .e2fgvi.utils.image_utils import (
    convert_image_to_frames as e2f_convert_frames,
    convert_mask_to_frames as e2f_convert_masks,
    dilate_masks as e2f_dilate_masks,
    prepare_tensors as e2f_prepare_tensors,
    resize_frames as e2f_resize_frames,
    resize_masks as e2f_resize_masks,
)
from .e2fgvi.utils.model_utils import load_model as e2f_load_model


_LOGGER = logging.getLogger("VideoInpaintWatermark")


def _check_inputs(frames: torch.Tensor, masks: torch.Tensor) -> None:
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


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    max_val = float(mask.max()) if mask.numel() else 0.0
    if max_val > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def _ensure_mask_batch(mask: torch.Tensor, frame_count: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.size(0) == 1 and frame_count > 1:
        mask = mask.repeat(frame_count, 1, 1)
    return mask


def _resize_mask_to_output(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to resize mask outputs.") from exc

    mask_np = mask.detach().cpu().numpy()
    resized = []
    for frame in mask_np:
        resized.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST))
    resized_np = np.stack(resized, axis=0).astype(np.float32)
    return torch.from_numpy(resized_np)


def _mask_to_bbox(mask: torch.Tensor) -> tuple[int, int, int, int, str]:
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 2:
        mask_np = mask_np[np.newaxis, ...]
    union = mask_np.max(axis=0)
    ys, xs = np.where(union > 0.5)
    height, width = union.shape
    if xs.size == 0 or ys.size == 0:
        return 0, 0, width, height, "empty_mask"
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1, "ok"


def _format_resolve_edit_position(
    norm_x: float,
    norm_y: float,
    background_width: int,
    background_height: int,
    overlay_width: int,
    overlay_height: int,
) -> dict:
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


class VideoInpaintWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Видео кадры (IMAGE batch)."}),
                "mask": ("MASK", {"tooltip": "Маска для удаления (1 кадр или batch)."}),
                "method": (["propainter", "e2fgvi", "e2fgvi_hq"], {"default": "propainter", "tooltip": "Алгоритм инпейнтинга."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Ширина выхода (0 = как у входа, 256–2048 типично)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Высота выхода (0 = как у входа, 256–2048 типично)."}),
                "mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение маски (0-100, типично 4–12)."}),
                "flow_mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение flow-маски (0-100, типично 4–12)."}),
                "ref_stride": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Шаг опорных кадров (1-100, типично 5–15)."}),
                "neighbor_length": ("INT", {"default": 10, "min": 2, "max": 300, "tooltip": "Соседние кадры (2-300, типично 5–15)."}),
                "subvideo_length": ("INT", {"default": 80, "min": 1, "max": 300, "tooltip": "Длина подвидео (1-300, типично 40–120)."}),
                "raft_iter": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "Итерации RAFT (1-100, типично 10–30)."}),
                "fp16": (["enable", "disable"], {"default": "enable", "tooltip": "FP16 режим (enable быстрее/меньше VRAM)."}),
                "throughput_mode": (["enable", "disable"], {"default": "disable", "tooltip": "Пропускать очистку кэша GPU (enable быстрее, но больше памяти)."}),
                "cudnn_benchmark": (["default", "enable", "disable"], {"default": "default", "tooltip": "cuDNN benchmark (enable быстрее при фикс. размере)."}),
                "tf32": (["default", "enable", "disable"], {"default": "default", "tooltip": "TF32 матмулы (enable быстрее, чуть менее точно)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "transform_json")
    FUNCTION = "inpaint"
    CATEGORY = "video/inpaint"

    def inpaint(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        method: str,
        width: int,
        height: int,
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
    ):
        if method in ("e2fgvi", "e2fgvi_hq"):
            return self._inpaint_e2fgvi(
                frames=frames,
                mask=mask,
                method=method,
                width=width,
                height=height,
                mask_dilates=mask_dilates,
                ref_stride=ref_stride,
                neighbor_length=neighbor_length,
                fp16=fp16,
                throughput_mode=throughput_mode,
            )

        _check_inputs(frames, mask)
        device = model_management.get_torch_device()

        if cudnn_benchmark != "default" or tf32 != "default":
            configure_cudnn(
                benchmark=cudnn_benchmark == "enable",
                allow_tf32=tf32 == "enable",
            )

        if width <= 0:
            width = int(frames.shape[2])
        if height <= 0:
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
        _LOGGER.info(
            "ProPainter: %s frames, process_size=%s, device=%s",
            video_length,
            image_config.process_size,
            device,
        )

        frames_tensor, flow_masks_tensor, masks_dilated_tensor, original_frames = (
            prepare_frames_and_masks(frames_np, mask, image_config, device)
        )

        models = initialize_models(device, inpaint_config.fp16)

        updated_frames, updated_masks, pred_flows_bi = process_inpainting(
            models,
            frames_tensor,
            flow_masks_tensor,
            masks_dilated_tensor,
            inpaint_config,
        )

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

        del frames_tensor, flow_masks_tensor, masks_dilated_tensor, updated_frames, updated_masks, pred_flows_bi
        if torch.cuda.is_available() and throughput_mode != "enable":
            torch.cuda.empty_cache()

        rgba, out_mask, transform_json = _crop_outputs(
            output_images,
            mask,
            output_images.shape[2],
            output_images.shape[1],
        )
        _LOGGER.info("Transform JSON: %s", transform_json)
        return (rgba, out_mask, transform_json)

    def _inpaint_e2fgvi(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        method: str,
        width: int,
        height: int,
        mask_dilates: int,
        ref_stride: int,
        neighbor_length: int,
        fp16: str,
        throughput_mode: str,
    ):
        device = model_management.get_torch_device()
        frames_np = e2f_convert_frames(frames)
        masks_np = e2f_convert_masks(mask)

        if method == "e2fgvi":
            target_size = (432, 240) if width <= 0 or height <= 0 else (width, height)
        else:
            target_size = None if width <= 0 or height <= 0 else (width, height)

        if target_size is not None:
            frames_np = e2f_resize_frames(frames_np, target_size)
            masks_np = e2f_resize_masks(masks_np, target_size)

        masks_np = e2f_dilate_masks(masks_np, mask_dilates)

        imgs, masks, binary_masks, h, w = e2f_prepare_tensors(frames_np, masks_np, device)
        video_length = imgs.size(dim=1)

        model_name = "e2fgvi_hq" if method == "e2fgvi_hq" else "e2fgvi"
        model = e2f_load_model(model_name, device, fp16 == "enable")

        neighbor_stride = max(1, int(neighbor_length))
        ref_length = max(1, int(ref_stride))
        comp_frames = [None] * video_length

        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(
                range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))
            )
            ref_ids = _get_ref_index(f, neighbor_ids, video_length, ref_length)
            selected_imgs = imgs[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:, neighbor_ids + ref_ids, :, :, :]
            masked_imgs = selected_imgs * (1 - selected_masks)

            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            if h_pad > 0:
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)
                masked_imgs = masked_imgs[:, :, :, : h + h_pad, :]
            if w_pad > 0:
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)
                masked_imgs = masked_imgs[:, :, :, :, : w + w_pad]

            with torch.inference_mode():
                pred_imgs, _ = model(masked_imgs, len(neighbor_ids))

            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

            for i, idx in enumerate(neighbor_ids):
                img = pred_imgs[i].astype("uint8") * binary_masks[idx] + frames_np[idx] * (
                    1 - binary_masks[idx]
                )
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype("float32") * 0.5 + img.astype("float32") * 0.5
                    )
                comp_frames[idx] = comp_frames[idx].astype("uint8")

            if throughput_mode != "enable" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_np = np.stack(comp_frames, axis=0).astype("float32") / 255.0
        output_images = torch.from_numpy(output_np)
        rgba, out_mask, transform_json = _crop_outputs(
            output_images,
            mask,
            output_images.shape[2],
            output_images.shape[1],
        )
        _LOGGER.info("Transform JSON: %s", transform_json)
        return (rgba, out_mask, transform_json)


def _get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_length, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_length * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_length * (ref_num // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


_LOGGER.warning(
    "Loaded VideoInpaintWatermark. NODE_CLASS_MAPPINGS=%s",
    ["VideoInpaintWatermark"],
)
