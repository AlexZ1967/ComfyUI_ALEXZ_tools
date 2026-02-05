import torch

from .utils import ensure_hwc, image_difference


class ImageDifference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Первая картинка (эталон)."}),
                "image_b": ("IMAGE", {"tooltip": "Вторая картинка (для сравнения)."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("difference",)
    FUNCTION = "diff"
    CATEGORY = "image/utils"

    def diff(self, image_a, image_b):
        a = ensure_hwc(image_a[0] if isinstance(image_a, list) else image_a)
        b = ensure_hwc(image_b[0] if isinstance(image_b, list) else image_b)
        diff = image_difference(a, b)
        return (diff.unsqueeze(0),)
