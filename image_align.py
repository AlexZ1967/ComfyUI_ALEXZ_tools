import logging

from .alignment import align_overlay_to_background
from .constants import (
    DEFAULT_FEATURE_COUNT,
    DEFAULT_GOOD_MATCH_PERCENT,
    DEFAULT_MIN_INLIERS,
    DEFAULT_MIN_MATCHES,
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
                    "tooltip": "Число ключевых точек (пример: 800–4000). Больше точности, но медленнее.",
                }),
                "min_matches": ("INT", {
                    "default": DEFAULT_MIN_MATCHES,
                    "min": 4,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Минимум совпадений (пример: 8–20 просто, 20–50 сложно).",
                }),
                "min_inliers": ("INT", {
                    "default": DEFAULT_MIN_INLIERS,
                    "min": 3,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Минимум inliers после RANSAC (обычно близко к min_matches).",
                }),
                "good_match_percent": ("FLOAT", {
                    "default": DEFAULT_GOOD_MATCH_PERCENT,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Доля лучших матчей (пример: 0.1–0.3 типично).",
                }),
                "ransac_thresh": ("FLOAT", {
                    "default": DEFAULT_RANSAC_THRESH,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Порог RANSAC в пикселях (пример: 2–5 точнее, 6–10 устойчивее).",
                }),
                "opacity": ("FLOAT", {
                    "default": DEFAULT_OPACITY,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Прозрачность оверлея (0..1), пример: 0.5 = 50%.",
                }),
                "matcher_type": (MATCHER_TYPES, {"default": "orb", "tooltip": "Алгоритм детектора/дескриптора."}),
                "scale_mode": (["preserve_aspect", "independent_xy"], {"default": "preserve_aspect", "tooltip": "Масштабирование: с сохранением пропорций или отдельно по X/Y."}),
                "allow_rotation": ("BOOLEAN", {"default": True, "tooltip": "Разрешить поворот оверлея."}),
                "color_mode": (["gray", "lab_l", "lab"], {"default": "gray", "tooltip": "Режим обработки цвета: серый, L-канал LAB, полный LAB."}),
                "lab_channels": (["l", "lab"], {"default": "lab", "tooltip": "Каналы LAB при color_mode=lab: только L или L+ab."}),
            },
            "optional": {
                "background_mask": ("MASK", {"tooltip": "Маска области совпадений на фоне (белое=использовать)."}),
                "overlay_mask": ("MASK", {"tooltip": "Маска области совпадений на оверлее (белое=использовать)."}),
                "use_color": ("BOOLEAN", {"default": False, "tooltip": "Устарело, используйте color_mode."}),
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
        min_matches,
        min_inliers,
        good_match_percent,
        ransac_thresh,
        opacity,
        matcher_type,
        scale_mode,
        allow_rotation,
        color_mode,
        lab_channels,
        background_mask=None,
        overlay_mask=None,
        use_color=None,
    ):
        if use_color is not None and use_color and color_mode == "gray":
            color_mode = "lab"
        if color_mode is None:
            color_mode = "lab" if use_color else "gray"
        if color_mode == "lab_l":
            color_mode = "lab"
            lab_channels = "l"
        if scale_mode == "uniform":
            _LOGGER.warning("scale_mode 'uniform' is deprecated; using 'preserve_aspect'.")
            scale_mode = "preserve_aspect"
        elif scale_mode == "free":
            _LOGGER.warning("scale_mode 'free' is deprecated; using 'independent_xy'.")
            scale_mode = "independent_xy"
        return align_overlay_to_background(
            background=background,
            overlay=overlay,
            background_mask=background_mask,
            overlay_mask=overlay_mask,
            feature_count=feature_count,
            min_matches=min_matches,
            min_inliers=min_inliers,
            good_match_percent=good_match_percent,
            ransac_thresh=ransac_thresh,
            opacity=opacity,
            matcher_type=matcher_type,
            scale_mode=scale_mode,
            allow_rotation=allow_rotation,
            color_mode=color_mode,
            lab_channels=lab_channels,
            logger=_LOGGER,
        )


_LOGGER.warning(
    "Loaded ImageAlignOverlayToBackground. NODE_CLASS_MAPPINGS=%s",
    ["ImageAlignOverlayToBackground"],
)
