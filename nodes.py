import math

import comfy.model_management
import comfy.utils
import torch


_ASPECT_RATIOS = {
    "1x1": (1, 1),
    "16x9": (16, 9),
    "9x16": (9, 16),
    "2x3": (2, 3),
    "3x2": (3, 2),
    "4x3": (4, 3),
    "3x4": (3, 4),
}

_TARGET_AREA = 1024 * 1024
_SIZE_MULTIPLE = 32
_LATENT_CHANNELS = 16


def _round_to_multiple(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def _target_size(aspect_ratio):
    ratio_w, ratio_h = _ASPECT_RATIOS[aspect_ratio]
    ratio = ratio_w / ratio_h
    width = math.sqrt(_TARGET_AREA * ratio)
    height = width / ratio
    width = _round_to_multiple(width, _SIZE_MULTIPLE)
    height = _round_to_multiple(height, _SIZE_MULTIPLE)
    return int(width), int(height)


class ImagePrepareForQwenEditOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(_ASPECT_RATIOS.keys()), {"default": "1x1"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "prepare"
    CATEGORY = "QWen"

    def prepare(self, image, aspect_ratio):
        target_width, target_height = _target_size(aspect_ratio)

        samples = image.movedim(-1, 1)
        in_height = samples.shape[2]
        in_width = samples.shape[3]
        scale = min(target_width / in_width, target_height / in_height)

        new_width = max(1, int(round(in_width * scale)))
        new_height = max(1, int(round(in_height * scale)))

        resized = comfy.utils.common_upscale(samples, new_width, new_height, "lanczos", "disabled")
        resized = resized.movedim(1, -1)

        canvas = torch.full(
            (image.shape[0], target_height, target_width, image.shape[-1]),
            0.5,
            device=image.device,
            dtype=image.dtype,
        )
        y0 = max(0, (target_height - new_height) // 2)
        x0 = max(0, (target_width - new_width) // 2)
        canvas[:, y0:y0 + new_height, x0:x0 + new_width, :] = resized

        latent = torch.zeros(
            (image.shape[0], _LATENT_CHANNELS, target_height // 8, target_width // 8),
            device=comfy.model_management.intermediate_device(),
        )

        return (canvas, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": ImagePrepareForQwenEditOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": "ImagePrepare_for_QwenEdit_outpaint",
}
