"""Image Difference node implementation.

Provides a simple node that computes absolute per-pixel difference between two
images after automatic size normalization.
"""

import torch


from ..utils.utils import ensure_hwc, image_difference


class ImageDifference:
    """ComfyUI node that computes an absolute visual difference between two images."""
    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
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
        """Compute image difference and return the resulting visualization."""
        a = ensure_hwc(image_a[0] if isinstance(image_a, list) else image_a)
        b = ensure_hwc(image_b[0] if isinstance(image_b, list) else image_b)
        diff = image_difference(a, b)
        return (diff.unsqueeze(0),)
