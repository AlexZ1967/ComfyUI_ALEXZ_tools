import logging

import torch


_LOGGER = logging.getLogger("VideoInpaintWatermark")


def _get_propainter_node():
    try:
        from ComfyUI_ProPainter_Nodes.propainter_nodes import ProPainterInpaint
    except Exception as exc:  # pragma: no cover - optional dependency
        return None, exc
    return ProPainterInpaint(), None


class VideoInpaintWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Видео кадры (IMAGE batch)."}),
                "mask": ("MASK", {"tooltip": "Маска для удаления (1 кадр или batch)."}),
                "method": (["propainter", "e2fgvi"], {"default": "propainter", "tooltip": "Алгоритм инпейнтинга (E2FGVI требует отдельной установки)."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Ширина выхода, 0 = как у входа."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "Высота выхода, 0 = как у входа."}),
                "mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение маски (0-100)."}),
                "flow_mask_dilates": ("INT", {"default": 8, "min": 0, "max": 100, "tooltip": "Расширение flow-маски (0-100)."}),
                "ref_stride": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Шаг опорных кадров (1-100)."}),
                "neighbor_length": ("INT", {"default": 10, "min": 2, "max": 300, "tooltip": "Соседние кадры (2-300)."}),
                "subvideo_length": ("INT", {"default": 80, "min": 1, "max": 300, "tooltip": "Длина подвидео (1-300)."}),
                "raft_iter": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "Итерации RAFT (1-100)."}),
                "fp16": (["enable", "disable"], {"default": "enable", "tooltip": "FP16 режим."}),
                "throughput_mode": (["enable", "disable"], {"default": "enable", "tooltip": "Пропускать очистку кэша GPU."}),
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
            raise RuntimeError(
                "E2FGVI is not available in this setup. Install an E2FGVI ComfyUI node "
                "or use method=propainter."
            )

        propainter_node, error = _get_propainter_node()
        if propainter_node is None:
            raise RuntimeError(f"ProPainter is not available: {error}")

        if width <= 0:
            width = int(frames.shape[2])
        if height <= 0:
            height = int(frames.shape[1])

        result = propainter_node.propainter_inpainting(
            image=frames,
            mask=mask,
            width=width,
            height=height,
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
        )

        if isinstance(result, tuple):
            return (result[0],)
        return (result,)


_LOGGER.warning(
    "Loaded VideoInpaintWatermark. NODE_CLASS_MAPPINGS=%s",
    ["VideoInpaintWatermark"],
)
