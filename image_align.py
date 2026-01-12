import logging

from .alignment import align_overlay_to_background
from .constants import (
    DEFAULT_FEATURE_COUNT,
    DEFAULT_GOOD_MATCH_PERCENT,
    DEFAULT_OPACITY,
    DEFAULT_RANSAC_THRESH,
    MATCHER_TYPES,
)


_LOGGER = logging.getLogger("ImageAlignOverlayToBackground")


class ImageAlignOverlayToBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {"tooltip": "Фоновое изображение, в координатах которого выравниваем."}),
                "overlay": ("IMAGE", {"tooltip": "Изображение, которое будет масштабировано/повернуто/сдвинуто."}),
                "feature_count": ("INT", {
                    "default": DEFAULT_FEATURE_COUNT,
                    "min": 100,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "Число ключевых точек: больше точности, но медленнее.",
                }),
                "good_match_percent": ("FLOAT", {
                    "default": DEFAULT_GOOD_MATCH_PERCENT,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Доля лучших матчей для оценки трансформации.",
                }),
                "ransac_thresh": ("FLOAT", {
                    "default": DEFAULT_RANSAC_THRESH,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Порог RANSAC для выбросов (в пикселях).",
                }),
                "opacity": ("FLOAT", {
                    "default": DEFAULT_OPACITY,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Прозрачность оверлея в композите (0..1).",
                }),
                "matcher_type": (MATCHER_TYPES, {"default": "orb", "tooltip": "Алгоритм детектора/дескриптора."}),
            },
            "optional": {
                "background_mask": ("MASK", {"tooltip": "Маска области совпадений на фоне (белое=использовать)."}),
                "overlay_mask": ("MASK", {"tooltip": "Маска области совпадений на оверлее (белое=использовать)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("aligned_overlay", "composite", "difference", "transform_json")
    FUNCTION = "align"
    CATEGORY = "image/alignment"

    def align(
        self,
        background,
        overlay,
        feature_count,
        good_match_percent,
        ransac_thresh,
        opacity,
        matcher_type,
        background_mask=None,
        overlay_mask=None,
    ):
        return align_overlay_to_background(
            background=background,
            overlay=overlay,
            background_mask=background_mask,
            overlay_mask=overlay_mask,
            feature_count=feature_count,
            good_match_percent=good_match_percent,
            ransac_thresh=ransac_thresh,
            opacity=opacity,
            matcher_type=matcher_type,
            logger=_LOGGER,
        )


_LOGGER.warning(
    "Loaded ImageAlignOverlayToBackground. NODE_CLASS_MAPPINGS=%s",
    ["ImageAlignOverlayToBackground"],
)
