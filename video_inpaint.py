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


class VideoInpaintWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Видео кадры (IMAGE batch)."}),
                "mask": ("MASK", {"tooltip": "Маска для удаления (1 кадр или batch)."}),
                "method": (["propainter", "e2fgvi", "e2fgvi_hq"], {"default": "propainter", "tooltip": "Алгоритм инпейнтинга."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Ширина выхода, 0 = как у входа."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Высота выхода, 0 = как у входа."}),
                "mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение маски (0-100)."}),
                "flow_mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение flow-маски (0-100)."}),
                "ref_stride": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Шаг опорных кадров (1-100)."}),
                "neighbor_length": ("INT", {"default": 10, "min": 2, "max": 300, "tooltip": "Соседние кадры (2-300)."}),
                "subvideo_length": ("INT", {"default": 80, "min": 1, "max": 300, "tooltip": "Длина подвидео (1-300)."}),
                "raft_iter": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "Итерации RAFT (1-100)."}),
                "fp16": (["enable", "disable"], {"default": "enable", "tooltip": "FP16 режим."}),
                "throughput_mode": (["enable", "disable"], {"default": "disable", "tooltip": "Пропускать очистку кэша GPU (enable быстрее, но больше память)."}),
                "cudnn_benchmark": (["default", "enable", "disable"], {"default": "default", "tooltip": "cuDNN benchmark."}),
                "tf32": (["default", "enable", "disable"], {"default": "default", "tooltip": "TF32 матмулы."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
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

        return (output_images,)

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
        return (torch.from_numpy(output_np),)


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
