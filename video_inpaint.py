import logging

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
                "method": (["propainter", "e2fgvi"], {"default": "propainter", "tooltip": "Алгоритм инпейнтинга (E2FGVI пока не реализован)."}),
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
        if method == "e2fgvi":
            raise RuntimeError("E2FGVI is not implemented yet. Use method=propainter.")

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


_LOGGER.warning(
    "Loaded VideoInpaintWatermark. NODE_CLASS_MAPPINGS=%s",
    ["VideoInpaintWatermark"],
)
