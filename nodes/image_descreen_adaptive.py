"""
Module: nodes/image_descreen_adaptive.py
Author: AlexZ1967
Last updated: 2026-03-27

Description:
    Utilities for practical scale-based halftone descreening:
    estimate raster period, build visual scale previews, run the
    legacy all-in-one adaptive node, and apply a chosen fixed scale.

Purpose:
    Provides ComfyUI nodes for pragmatic descreen workflows where the user
    measures raster period, reviews a scale sheet, and then applies a fixed
    final downscale to one image or a batch.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from .image_descreen_adaptive_core import (
    _apply_fixed_percent_downscale_batch,
    _analyze_one,
    _estimate_period_one,
    _build_scale_preview_one,
    _to_rgb_batch,
    _to_tensor,
)


class ImageEstimateRasterPeriod:
    """Estimate raster period and predict a practical base downscale percent."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Скан/изображение с заметным печатным растром."}),
            },
            "optional": {
                "roi_mode": (
                    ["center_square", "full_frame", "manual_rect"],
                    {
                        "default": "center_square",
                        "tooltip": "Область анализа для оценки шага растра.",
                    },
                ),
                "roi_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального ROI в процентах от меньшей стороны кадра.",
                    },
                ),
                "roi_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата ROI для manual_rect."}),
                "roi_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата ROI для manual_rect."}),
                "roi_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина ROI для manual_rect."}),
                "roi_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота ROI для manual_rect."}),
                "target_screen_px": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Целевой остаточный шаг растра в пикселях для расчета базового процента.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("roi_preview", "estimated_period_px", "predicted_scale_percent", "analysis_json")
    FUNCTION = "estimate"
    CATEGORY = "image/restoration"

    def estimate(
        self,
        image: torch.Tensor,
        roi_mode: str = "center_square",
        roi_size_percent: float = 40.0,
        roi_x: int = 0,
        roi_y: int = 0,
        roi_w: int = 256,
        roi_h: int = 256,
        target_screen_px: float = 1.0,
    ):
        """Estimate raster period for one image or batch, returning first ROI preview."""
        rgb_batch = _to_rgb_batch(image)
        preview_first = None
        analysis = []
        for idx, rgb in enumerate(rgb_batch):
            roi_preview, result = _estimate_period_one(
                rgb,
                roi_mode=roi_mode,
                roi_size_percent=roi_size_percent,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_w=roi_w,
                roi_h=roi_h,
                target_screen_px=target_screen_px,
            )
            result["batch_index"] = int(idx)
            analysis.append(result)
            if preview_first is None:
                preview_first = _to_tensor(roi_preview)
        if preview_first is None:
            raise ValueError("Empty image batch.")
        first = analysis[0]
        payload: Any = first if len(analysis) == 1 else analysis
        return (
            preview_first,
            float(first["estimated_period_px"]),
            float(first["predicted_scale_percent"]),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


class ImageDescreenScalePreview:
    """Build a visual scale sheet from a chosen base percent and ROI."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Скан/изображение для визуального подбора scale."}),
                "base_percent": (
                    "FLOAT",
                    {
                        "default": 26.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Базовый процент, от которого строится подборочная лестница.",
                    },
                ),
            },
            "optional": {
                "resample_mode": (
                    ["lanczos", "bicubic"],
                    {
                        "default": "lanczos",
                        "tooltip": "Ресемплер для preview-вариантов.",
                    },
                ),
                "roi_mode": (
                    ["center_square", "full_frame", "manual_rect"],
                    {
                        "default": "center_square",
                        "tooltip": "Зона preview, по которой строится scale-sheet.",
                    },
                ),
                "roi_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального ROI в процентах от меньшей стороны кадра.",
                    },
                ),
                "roi_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата ROI для manual_rect."}),
                "roi_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата ROI для manual_rect."}),
                "roi_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина ROI для manual_rect."}),
                "roi_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота ROI для manual_rect."}),
                "range_up_percent": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "На сколько процентов вверх строить лестницу от base_percent.",
                    },
                ),
                "step_percent": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Шаг процентов между вариантами в scale-sheet.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scale_sheet", "analysis_json")
    FUNCTION = "preview"
    CATEGORY = "image/restoration"

    def preview(
        self,
        image: torch.Tensor,
        base_percent: float = 26.0,
        resample_mode: str = "lanczos",
        roi_mode: str = "center_square",
        roi_size_percent: float = 40.0,
        roi_x: int = 0,
        roi_y: int = 0,
        roi_w: int = 256,
        roi_h: int = 256,
        range_up_percent: float = 10.0,
        step_percent: float = 2.0,
    ):
        """Build preview for the first image in batch and return diagnostics for all."""
        rgb_batch = _to_rgb_batch(image)
        preview_first = None
        analysis = []
        for idx, rgb in enumerate(rgb_batch):
            scale_sheet, result = _build_scale_preview_one(
                rgb,
                resample_mode=resample_mode,
                roi_mode=roi_mode,
                roi_size_percent=roi_size_percent,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_w=roi_w,
                roi_h=roi_h,
                base_percent=base_percent,
                range_up_percent=range_up_percent,
                step_percent=step_percent,
            )
            result["batch_index"] = int(idx)
            analysis.append(result)
            if preview_first is None:
                preview_first = _to_tensor(scale_sheet)
        if preview_first is None:
            raise ValueError("Empty image batch.")
        payload: Any = analysis[0] if len(analysis) == 1 else analysis
        return (
            preview_first,
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


class ImageDescreenAdaptiveScale:
    """Legacy all-in-one node: estimate, preview, and apply adaptive scale at once."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Скан/изображение с заметным печатным растром."}),
            },
            "optional": {
                "resample_mode": (
                    ["lanczos", "bicubic"],
                    {
                        "default": "lanczos",
                        "tooltip": "Ресемплер для scale-descreen. На некоторых типографских сканах лучше работает Lanczos, на других Bicubic.",
                    },
                ),
                "roi_mode": (
                    ["center_square", "full_frame", "manual_rect"],
                    {
                        "default": "center_square",
                        "tooltip": "Область анализа для оценки шага растра. center_square обычно самый практичный режим.",
                    },
                ),
                "roi_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального квадратного ROI в процентах от меньшей стороны кадра.",
                    },
                ),
                "roi_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата ROI для manual_rect."}),
                "roi_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата ROI для manual_rect."}),
                "roi_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина ROI для manual_rect."}),
                "roi_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота ROI для manual_rect."}),
                "min_scale_percent": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Нижняя граница поиска optimal scale в процентах.",
                    },
                ),
                "max_scale_percent": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Верхняя граница поиска optimal scale в процентах.",
                    },
                ),
                "step_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Шаг перебора между scale-кандидатами.",
                    },
                ),
                "target_screen_px": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Целевой остаточный размер шага растра после уменьшения. Меньше = агрессивнее подавление растра.",
                    },
                ),
                "detail_weight": (
                    "FLOAT",
                    {
                        "default": 1.25,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Вес штрафа за потерю полезной структуры. Больше = больше сохраняем детали, меньше = агрессивнее убираем растр.",
                    },
                ),
                "pre_blur_px": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Опциональный blur перед downscale. По умолчанию 0.0, т.к. часто лучший результат на сканах достигается без blur.",
                    },
                ),
                "sheet_zone_mode": (
                    ["analysis_roi", "center_square", "full_frame", "manual_rect"],
                    {
                        "default": "analysis_roi",
                        "tooltip": "Отдельная зона для preview scale-sheet. analysis_roi использует тот же ROI, что и анализ; manual_rect задается параметрами ниже.",
                    },
                ),
                "sheet_zone_size_percent": (
                    "FLOAT",
                    {
                        "default": 40.0,
                        "min": 5.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Размер центрального квадрата для scale-sheet, если sheet_zone_mode=center_square.",
                    },
                ),
                "sheet_zone_x": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Левая координата зоны scale-sheet для manual_rect."}),
                "sheet_zone_y": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Верхняя координата зоны scale-sheet для manual_rect."}),
                "sheet_zone_w": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Ширина зоны scale-sheet для manual_rect."}),
                "sheet_zone_h": ("INT", {"default": 256, "min": 8, "max": 100000, "tooltip": "Высота зоны scale-sheet для manual_rect."}),
                "sheet_range_up_percent": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.5,
                        "tooltip": "На сколько процентов вверх от расчетного базового scale строить подборочную лестницу в preview sheet.",
                    },
                ),
                "sheet_step_percent": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Шаг процентов между вариантами в preview sheet.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "scale_sheet", "recommended_percent", "estimated_period_px", "analysis_json")
    FUNCTION = "descreen"
    CATEGORY = "image/restoration"

    def descreen(
        self,
        image: torch.Tensor,
        resample_mode: str = "lanczos",
        roi_mode: str = "center_square",
        roi_size_percent: float = 40.0,
        roi_x: int = 0,
        roi_y: int = 0,
        roi_w: int = 256,
        roi_h: int = 256,
        min_scale_percent: float = 8.0,
        max_scale_percent: float = 20.0,
        step_percent: float = 1.0,
        target_screen_px: float = 1.0,
        detail_weight: float = 1.25,
        pre_blur_px: float = 0.0,
        sheet_zone_mode: str = "analysis_roi",
        sheet_zone_size_percent: float = 40.0,
        sheet_zone_x: int = 0,
        sheet_zone_y: int = 0,
        sheet_zone_w: int = 256,
        sheet_zone_h: int = 256,
        sheet_range_up_percent: float = 10.0,
        sheet_step_percent: float = 2.0,
    ):
        """Estimate halftone period, choose optimal scale, and process the image."""
        rgb_batch = _to_rgb_batch(image)
        processed_batch = []
        preview_first = None
        analysis = []

        for idx, rgb in enumerate(rgb_batch):
            processed_rgb, compare_preview, result = _analyze_one(
                rgb,
                roi_mode=roi_mode,
                roi_size_percent=roi_size_percent,
                roi_x=roi_x,
                roi_y=roi_y,
                roi_w=roi_w,
                roi_h=roi_h,
                min_scale_percent=min_scale_percent,
                max_scale_percent=max_scale_percent,
                step_percent=step_percent,
                target_screen_px=target_screen_px,
                detail_weight=detail_weight,
                pre_blur_px=pre_blur_px,
                resample_mode=resample_mode,
                sheet_zone_mode=sheet_zone_mode,
                sheet_zone_size_percent=sheet_zone_size_percent,
                sheet_zone_x=sheet_zone_x,
                sheet_zone_y=sheet_zone_y,
                sheet_zone_w=sheet_zone_w,
                sheet_zone_h=sheet_zone_h,
                sheet_range_up_percent=sheet_range_up_percent,
                sheet_step_percent=sheet_step_percent,
            )
            result["batch_index"] = int(idx)
            analysis.append(result)
            processed_batch.append(torch.from_numpy(processed_rgb.astype(np.float32) / 255.0))
            if preview_first is None:
                preview_first = _to_tensor(compare_preview)

        if preview_first is None:
            raise ValueError("Empty image batch.")

        processed_tensor = torch.stack(processed_batch, dim=0)
        first = analysis[0]
        payload: Any = analysis[0] if len(analysis) == 1 else analysis
        return (
            processed_tensor,
            preview_first,
            float(first["recommended_percent"]),
            float(first["estimated_period_px"]),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )


class ImageDescreenApplyPercent:
    """Apply a known descreen percent as final downscale-only output."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Изображение или батч, к которому нужно применить уже найденный descreen percent."}),
                "scale_percent": (
                    "FLOAT",
                    {
                        "default": 13.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Готовый descreen percent. Удобно подключать recommended_percent из Descreen By Adaptive Scale.",
                    },
                ),
            },
            "optional": {
                "resample_mode": (
                    ["lanczos", "bicubic"],
                    {
                        "default": "lanczos",
                        "tooltip": "Ресемплер для уменьшения итоговой картинки. Lanczos на части сканов держит форму лучше, Bicubic может вести себя мягче.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "applied_percent", "analysis_json")
    FUNCTION = "apply"
    CATEGORY = "image/restoration"

    def apply(
        self,
        image: torch.Tensor,
        scale_percent: float = 13.0,
        resample_mode: str = "lanczos",
    ):
        """Apply fixed downscale-only descreen percent to the whole image batch."""
        processed = _apply_fixed_percent_downscale_batch(
            image,
            scale_percent=scale_percent,
            resample_mode=resample_mode,
        )
        payload = {
            "mode": "fixed_percent_downscale_only",
            "applied_percent": float(scale_percent),
            "applied_scale": float(scale_percent) / 100.0,
            "resample_mode": str(resample_mode),
            "batch_size": int(processed.shape[0]),
            "return_to_original_size": False,
        }
        return (
            processed,
            float(scale_percent),
            json.dumps(payload, ensure_ascii=True, indent=2),
        )
