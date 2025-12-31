import logging
import math

import comfy.model_management
import comfy.utils
import torch


_ASPECT_RATIOS = {
    "as_is": None,  # Сохранить исходные пропорции
    "1x1": (1328, 1328),
    "16x9": (1664, 928),
    "9x16": (928, 1664),
    "4x3": (1472, 1104),
    "3x4": (1104, 1472),
    "3x2": (1584, 1056),
    "2x3": (1056, 1584),
}

# Целевая площадь для масштабирования (при as_is)
_TARGET_AREA = 1328 * 1328
_LATENT_CHANNELS = 4  # Стандартный VAE использует 4 канала


_LOGGER = logging.getLogger("ImagePrepare_for_QwenEdit_outpaint")


def _round_to_multiple(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


class ImagePrepareForQwenEditOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(_ASPECT_RATIOS.keys()), {"default": "as_is"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "prepare"
    CATEGORY = "image/qwen"

    def prepare(self, image, aspect_ratio):
        samples = image.movedim(-1, 1)
        in_height = samples.shape[2]
        in_width = samples.shape[3]

        if aspect_ratio == "as_is":
            # Масштабируем по площади, сохраняя пропорции
            current_area = in_width * in_height
            scale = math.sqrt(_TARGET_AREA / current_area)
            
            new_width = max(1, int(round(in_width * scale)))
            new_height = max(1, int(round(in_height * scale)))
            
            latent_width = new_width // 8
            latent_height = new_height // 8
        else:
            # Вписываем в целевой размер
            target_width, target_height = _ASPECT_RATIOS[aspect_ratio]
            scale = min(target_width / in_width, target_height / in_height)
            
            new_width = max(1, int(round(in_width * scale)))
            new_height = max(1, int(round(in_height * scale)))
            
            latent_width = target_width // 8
            latent_height = target_height // 8

        resized = comfy.utils.common_upscale(samples, new_width, new_height, "lanczos", "disabled")
        resized = resized.movedim(1, -1)

        return (resized, {"samples": torch.zeros(
            (image.shape[0], _LATENT_CHANNELS, latent_height, latent_width),
            device=comfy.model_management.intermediate_device(),
        )})


NODE_CLASS_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": ImagePrepareForQwenEditOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": "Image Prepare for QwenEdit Outpaint",
}

_LOGGER.warning(
    "Loaded ImagePrepare_for_QwenEdit_outpaint. NODE_CLASS_MAPPINGS=%s",
    list(NODE_CLASS_MAPPINGS.keys()),
)
