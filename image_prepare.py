import logging
import math

import comfy.model_management
import comfy.utils
import torch

from .utils import round_to_multiple


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
_SIZE_MULTIPLE = 32
_LATENT_CHANNELS = 4  # Стандартный VAE использует 4 канала


_LOGGER = logging.getLogger("ImagePrepare_for_QwenEdit_outpaint")


class ImagePrepareForQwenEditOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Исходное изображение для подготовки."}),
                "aspect_ratio": (
                    list(_ASPECT_RATIOS.keys()),
                    {"default": "as_is", "tooltip": "Целевое соотношение сторон для ресайза."},
                ),
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

            new_width = round_to_multiple(new_width, _SIZE_MULTIPLE)
            new_height = round_to_multiple(new_height, _SIZE_MULTIPLE)

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


_LOGGER.warning(
    "Loaded ImagePrepare_for_QwenEdit_outpaint. NODE_CLASS_MAPPINGS=%s",
    ["ImagePrepare_for_QwenEdit_outpaint"],
)
