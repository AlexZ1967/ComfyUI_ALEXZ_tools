"""
Module: nodes/image_color_match.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Color Match To Reference node implementation.

Purpose:
    Implements color-transfer methods, quality metrics, and JSON export for image-to-reference matching.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        """Fallback tqdm wrapper that returns the iterable unchanged when tqdm is unavailable."""
        return iterable if iterable is not None else []

from ..utils import color_match_utils
from ..utils.interrupt import check_interrupt

_LOGGER = logging.getLogger("ImageColorMatchToReference")

from .image_color_match_lut_ops import (
    _apply_poly2_color_map,
    _fit_poly2_color_map,
    _lut_grid_colors,
    _sanitize_lut_name,
    _write_cube_file,
)
from .image_color_match_match_ops import (
    _adain_match_batch,
    _grid_match_batch,
    _hist_match_rgb_batch,
    _hm_mkl_hm_match_batch,
    _hm_mvgd_hm_match_batch,
    _lab_cdf_match_batch,
    _linear_fit_torch_batch,
    _linear_match_batch,
    _mean_std_fit_torch_batch,
    _mean_std_match_batch,
    _mkl_match_batch,
    _mvgd_match_batch,
    _oklab_cdf_match_batch,
    _optimal_transport_match_batch,
    _pad_batch_last,
    _reinhard_lab_fast_batch,
    _tone_curve_match_batch,
)
from . import image_color_match_match_ops as color_match_ops
from . import image_color_match_metrics_ops as color_metrics_ops
from .image_color_match_metrics_ops import (
    _apply_skin_tone_protection,
    _auto_optimal_score,
    _empty_quality_metrics,
    _improvement_pct,
    _lpips_alex_distance,
    _perceptual_vgg_fast,
    _quality_metrics_fast,
)


def _quality_metrics(img: torch.Tensor, ref: torch.Tensor):
    """Compute full metrics while preserving the adapter-level LPIPS seam."""
    return color_metrics_ops._quality_metrics(img, ref, lpips_fn=_lpips_alex_distance)


def _auto_optimal_candidate_metrics(candidate: torch.Tensor, ref: torch.Tensor, strategy: str) -> dict:
    """Compute auto metrics while preserving the adapter-level LPIPS seam."""
    return color_metrics_ops._auto_optimal_candidate_metrics(
        candidate,
        ref,
        strategy,
        lpips_fn=_lpips_alex_distance,
    )


def _run_auto_fallback_single(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: Optional[torch.Tensor],
    method: str,
):
    """Run fallback while preserving adapter-level algorithm seams."""
    return color_match_ops._run_auto_fallback_single(
        img,
        ref,
        mask,
        method,
        lab_cdf_fn=_lab_cdf_match_batch,
        oklab_cdf_fn=_oklab_cdf_match_batch,
        perceptual_fn=_perceptual_vgg_fast,
    )


class ImageColorMatchToReference:
    """ComfyUI node that matches image color grading to a reference frame."""
    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Базовое изображение (образец)."}),
                "image": ("IMAGE", {"tooltip": "Изображение, которое нужно подогнать по цвету."}),
                "preset": (
                    [
                        "mean_std",
                        "linear",
                        "tone_curve",
                        "adain",
                        "optimal_transport",
                        "lab_cdf",
                        "oklab_cdf",
                        "auto_optimal",
                        "perceptual_vgg_fast",
                        "reinhard_lab_fast",
                        "hm",
                        "mkl",
                        "mvgd",
                        "hm-mkl-hm",
                        "hm-mvgd-hm",
                    ],
                    {
                        "default": "linear",
                        "tooltip": "Метод: mean_std/linear/tone_curve/adain=быстрые CPU-friendly; optimal_transport/lab_cdf/oklab_cdf=точнее, но тяжелее; auto_optimal=автовыбор; perceptual_vgg_fast=VGG (требует torchvision, предпочтителен GPU); reinhard_lab_fast/hm/mkl/mvgd/hm-mkl-hm/hm-mvgd-hm=экспериментальные режимы палитрового переноса.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Сила применения коррекции (0..1).",
                    },
                ),
            },
            "optional": {
                "match_mask": ("MASK", {"tooltip": "Где считать статистику (белое=учитывать)."}),
                "apply_mask": ("MASK", {"tooltip": "Где применять коррекцию (белое=применить, чёрное=оставить исходное). Маски повышают время обработки."}),
                "preserve_alpha": ("BOOLEAN", {"default": True, "tooltip": "Если вход RGBA — сохранить альфу из исходника."}),
                "compute_quality_metrics": ("BOOLEAN", {"default": True, "tooltip": "Считать MSE/SSIM/DeltaE/LPIPS. Отключите для ускорения батчей."}),
                "quality_metrics_mode": (
                    ["off", "fast", "full"],
                    {"default": "full", "tooltip": "off=без метрик, fast=MSE+SSIM, full=MSE+SSIM+DeltaE+LPIPS (LPIPS требует пакет lpips, медленнее на CPU)."},
                ),
                "auto_optimal_metric": (
                    ["mse", "mse_ssim", "mse_ssim_lpips"],
                    {"default": "mse_ssim", "tooltip": "Критерий выбора для auto_optimal. mse_ssim_lpips точнее, но медленнее."},
                ),
                "auto_temporal_stability": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Стабилизирует выбор метода в auto_optimal между соседними кадрами."},
                ),
                "auto_temporal_alpha": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Сглаживание EMA для auto_optimal (выше = стабильнее)."},
                ),
                "auto_switch_threshold": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Порог переключения режима в auto_optimal (hysteresis)."},
                ),
                "auto_quality_fallback": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "В auto_optimal включает fallback к более тяжелому методу, если качество низкое."},
                ),
                "auto_fallback_threshold": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Порог score в auto_optimal; выше порога активируется fallback."},
                ),
                "auto_fallback_margin": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0005, "tooltip": "Минимальный выигрыш score для принятия fallback-результата."},
                ),
                "auto_fallback_method": (
                    ["lab_cdf", "oklab_cdf", "perceptual_vgg_fast"],
                    {"default": "lab_cdf", "tooltip": "Метод fallback в auto_optimal при низком качестве. perceptual_vgg_fast требует torchvision и обычно GPU."},
                ),
                "skin_tone_protection": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Сохранять тона кожи, ослабляя коррекцию в skin-областях."},
                ),
                "skin_protection_strength": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Сила защиты кожи (0..1)."},
                ),
                "spatial_grid": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Локальный матчинг по сетке NxN (работает для linear/mean_std/adain/auto_optimal). Увеличение сетки повышает нагрузку на CPU/GPU."},
                ),
                "export_lut": ("BOOLEAN", {"default": False, "tooltip": "Экспортировать LUT .cube для каждой пары вход/референс."}),
                "lut_size": ("INT", {"default": 33, "min": 8, "max": 65, "tooltip": "Размер 3D LUT (типично 17/33)."}),
                "lut_output_dir": ("STRING", {"default": "", "tooltip": "Папка для .cube. Пусто = ./output/color_luts."}),
                "lut_name": ("STRING", {"default": "color_match", "tooltip": "Базовое имя LUT файла (без расширения)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "match_json")
    FUNCTION = "match"
    CATEGORY = "image/color"

    def match(
        self,
        reference,
        image,
        preset,
        match_mask=None,
        apply_mask=None,
        preserve_alpha=True,
        compute_quality_metrics=True,
        quality_metrics_mode="full",
        auto_optimal_metric="mse_ssim",
        auto_temporal_stability=False,
        auto_temporal_alpha=0.75,
        auto_switch_threshold=0.01,
        auto_quality_fallback=False,
        auto_fallback_threshold=0.05,
        auto_fallback_margin=0.001,
        auto_fallback_method="lab_cdf",
        skin_tone_protection=False,
        skin_protection_strength=0.6,
        spatial_grid=1,
        export_lut=False,
        lut_size=33,
        lut_output_dir="",
        lut_name="color_match",
        strength=1.0,
    ):
        """Execute the node and return processed outputs for ComfyUI."""
        batch_size = max(reference.shape[0], image.shape[0])
        reference_batch = _pad_batch_last(reference, batch_size)
        image_batch = _pad_batch_last(image, batch_size)
        ref_h, ref_w = reference_batch.shape[1], reference_batch.shape[2]

        alpha_batch = None
        reference_rgb = reference_batch
        image_rgb = image_batch
        if reference_batch.shape[3] > 3:
            alpha_batch = reference_batch[..., 3:4]
            reference_rgb = reference_batch[..., :3]
        if image_batch.shape[3] > 3:
            if alpha_batch is None:
                alpha_batch = image_batch[..., 3:4]
            image_rgb = image_batch[..., :3]

        if image_rgb.shape[1] != ref_h or image_rgb.shape[2] != ref_w:
            image_rgb = color_match_utils.resize_images_to_size(image_rgb, ref_h, ref_w)
        if alpha_batch is not None and (alpha_batch.shape[1] != ref_h or alpha_batch.shape[2] != ref_w):
            alpha_batch = color_match_utils.resize_images_to_size(alpha_batch, ref_h, ref_w)

        match_mask_batch, apply_mask_batch, match_mask_valid = color_match_utils.prepare_match_and_apply_masks(
            match_mask, apply_mask, batch_size, ref_h, ref_w, reference_rgb.device, reference_rgb.dtype
        )
        if match_mask is not None:
            empty_mask_idx = (~match_mask_valid).nonzero(as_tuple=False).flatten().tolist()
            if empty_mask_idx:
                _LOGGER.warning(
                    "ColorMatch: match_mask has no white pixels for frames %s; returning original image for those frames.",
                    empty_mask_idx,
                )

        spatial_grid_int = int(max(1, int(spatial_grid)))
        spatial_grid_supported = {"linear", "mean_std", "adain", "auto_optimal"}
        spatial_grid_applied = spatial_grid_int > 1 and preset in spatial_grid_supported
        if spatial_grid_int > 1 and not spatial_grid_applied:
            _LOGGER.info(
                "ColorMatch: spatial_grid=%d ignored for preset=%s (supported: linear/mean_std/adain/auto_optimal).",
                spatial_grid_int,
                preset,
            )

        corrected_batch = None
        auto_mode_batch = None
        auto_linear_batch = None
        auto_oklab_batch = None
        auto_pre_score_linear = None
        auto_pre_score_oklab = None
        mean_std_scale_batch = None
        mean_std_offset_batch = None
        if preset == "mean_std":
            if spatial_grid_applied:
                corrected_batch = _grid_match_batch(
                    image_rgb, reference_rgb, match_mask_batch, spatial_grid_int, _mean_std_match_batch
                )
            else:
                corrected_batch = _mean_std_match_batch(image_rgb, reference_rgb, match_mask_batch)
            mean_std_scale_batch, mean_std_offset_batch = _mean_std_fit_torch_batch(
                image_rgb, reference_rgb, match_mask_batch
            )
        elif preset == "linear":
            if spatial_grid_applied:
                corrected_batch = _grid_match_batch(
                    image_rgb, reference_rgb, match_mask_batch, spatial_grid_int, _linear_match_batch
                )
            else:
                corrected_batch = _linear_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "tone_curve":
            corrected_batch = _tone_curve_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "adain":
            if spatial_grid_applied:
                corrected_batch = _grid_match_batch(
                    image_rgb, reference_rgb, match_mask_batch, spatial_grid_int, _adain_match_batch
                )
            else:
                corrected_batch = _adain_match_batch(image_rgb, reference_rgb, match_mask_batch)
            mean_std_scale_batch, mean_std_offset_batch = _mean_std_fit_torch_batch(
                image_rgb, reference_rgb, match_mask_batch
            )
        elif preset == "optimal_transport":
            corrected_batch = _optimal_transport_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "lab_cdf":
            corrected_batch = _lab_cdf_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "oklab_cdf":
            corrected_batch = _oklab_cdf_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "reinhard_lab_fast":
            corrected_batch = _reinhard_lab_fast_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "hm":
            corrected_batch = _hist_match_rgb_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "mkl":
            corrected_batch = _mkl_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "mvgd":
            corrected_batch = _mvgd_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "hm-mkl-hm":
            corrected_batch = _hm_mkl_hm_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "hm-mvgd-hm":
            corrected_batch = _hm_mvgd_hm_match_batch(image_rgb, reference_rgb, match_mask_batch)
        elif preset == "auto_optimal":
            if spatial_grid_applied:
                auto_linear_batch = _grid_match_batch(
                    image_rgb, reference_rgb, match_mask_batch, spatial_grid_int, _linear_match_batch
                )
                auto_oklab_batch = _grid_match_batch(
                    image_rgb, reference_rgb, match_mask_batch, spatial_grid_int, _oklab_cdf_match_batch
                )
            else:
                auto_linear_batch = _linear_match_batch(image_rgb, reference_rgb, match_mask_batch)
                auto_oklab_batch = _oklab_cdf_match_batch(image_rgb, reference_rgb, match_mask_batch)
            if auto_optimal_metric == "mse" and not auto_temporal_stability:
                auto_pre_score_linear = torch.mean((auto_linear_batch - reference_rgb) ** 2, dim=(1, 2, 3))
                auto_pre_score_oklab = torch.mean((auto_oklab_batch - reference_rgb) ** 2, dim=(1, 2, 3))
                choose_oklab = auto_pre_score_oklab + 1e-6 < auto_pre_score_linear
                auto_mode_batch = ["oklab_cdf" if bool(v) else "linear" for v in choose_oklab.tolist()]
                corrected_batch = torch.where(choose_oklab[:, None, None, None], auto_oklab_batch, auto_linear_batch)

        scale_batch, offset_batch = _linear_fit_torch_batch(image_rgb, reference_rgb, match_mask_batch)
        lut_dir = Path(lut_output_dir).expanduser() if str(lut_output_dir).strip() else (Path.cwd() / "output" / "color_luts")
        lut_base = _sanitize_lut_name(lut_name)
        lut_size_int = int(max(8, min(65, int(lut_size))))
        effective_quality_mode = quality_metrics_mode if str(quality_metrics_mode) in ("off", "fast", "full") else "full"
        if not compute_quality_metrics:
            effective_quality_mode = "off"
        auto_prev_selected = None
        auto_ema_linear = None
        auto_ema_oklab = None
        auto_fallback_enabled = bool(auto_quality_fallback)
        auto_fallback_threshold_f = float(max(0.0, auto_fallback_threshold))
        auto_fallback_margin_f = float(max(0.0, auto_fallback_margin))
        auto_fallback_method_s = (
            auto_fallback_method
            if str(auto_fallback_method) in ("lab_cdf", "oklab_cdf", "perceptual_vgg_fast")
            else "lab_cdf"
        )
        grid_suffix = f":grid{spatial_grid_int}x{spatial_grid_int}" if spatial_grid_applied else ""

        matched_list = []
        json_list = []
        iterator = tqdm(range(batch_size), desc=f"ColorMatch[{preset}]", unit="img")
        for idx in iterator:
            check_interrupt()
            ref_t = reference_rgb[idx]
            img_t = image_rgb[idx]
            mm_t = match_mask_batch[idx] if match_mask_batch is not None else None
            am_t = apply_mask_batch[idx] if apply_mask_batch is not None else None
            match_valid = bool(match_mask_valid[idx].item())

            deep_params = None
            if not match_valid:
                corrected_t = img_t
                mode = f"{preset}:empty_match_mask"
                deep_params = {
                    "warning": "empty_match_mask",
                }
            elif preset == "auto_optimal":
                if corrected_batch is not None:
                    corrected_t = corrected_batch[idx]
                    chosen = auto_mode_batch[idx]
                    selected_score = (
                        float(auto_pre_score_linear[idx]) if chosen == "linear" else float(auto_pre_score_oklab[idx])
                    )
                    fallback_selected = None
                    fallback_score = None
                    fallback_meta = None
                    if auto_fallback_enabled and selected_score > auto_fallback_threshold_f:
                        fallback_t, fallback_meta = _run_auto_fallback_single(
                            img_t, ref_t, mm_t, auto_fallback_method_s
                        )
                        fallback_metrics = _auto_optimal_candidate_metrics(
                            fallback_t, ref_t, auto_optimal_metric
                        )
                        fallback_score = _auto_optimal_score(fallback_metrics, auto_optimal_metric)
                        if fallback_score + auto_fallback_margin_f < selected_score:
                            corrected_t = fallback_t
                            fallback_selected = auto_fallback_method_s
                    mode = f"auto_optimal:{chosen}{grid_suffix}"
                    if fallback_selected is not None:
                        mode = f"{mode}:fallback:{fallback_selected}"
                    deep_params = {
                        "auto_optimal": {
                            "strategy": auto_optimal_metric,
                            "score_linear": None if auto_pre_score_linear is None else round(float(auto_pre_score_linear[idx]), 6),
                            "score_oklab_cdf": None if auto_pre_score_oklab is None else round(float(auto_pre_score_oklab[idx]), 6),
                            "selected": chosen,
                            "temporal_stability": bool(auto_temporal_stability),
                            "spatial_grid": int(spatial_grid_int if spatial_grid_applied else 1),
                            "fallback_enabled": bool(auto_fallback_enabled),
                            "fallback_method": auto_fallback_method_s,
                            "fallback_threshold": auto_fallback_threshold_f,
                            "fallback_margin": auto_fallback_margin_f,
                            "fallback_score": None if fallback_score is None else round(float(fallback_score), 6),
                            "fallback_applied": bool(fallback_selected is not None),
                            "fallback_meta": fallback_meta,
                        }
                    }
                else:
                    linear_t = auto_linear_batch[idx]
                    oklab_t = auto_oklab_batch[idx]
                    linear_metrics = _auto_optimal_candidate_metrics(linear_t, ref_t, auto_optimal_metric)
                    oklab_metrics = _auto_optimal_candidate_metrics(oklab_t, ref_t, auto_optimal_metric)
                    score_linear = _auto_optimal_score(linear_metrics, auto_optimal_metric)
                    score_oklab = _auto_optimal_score(oklab_metrics, auto_optimal_metric)
                    ema_linear = score_linear
                    ema_oklab = score_oklab
                    if auto_temporal_stability:
                        alpha = float(max(0.0, min(0.99, auto_temporal_alpha)))
                        if auto_ema_linear is None:
                            auto_ema_linear = score_linear
                            auto_ema_oklab = score_oklab
                        else:
                            auto_ema_linear = alpha * auto_ema_linear + (1.0 - alpha) * score_linear
                            auto_ema_oklab = alpha * auto_ema_oklab + (1.0 - alpha) * score_oklab
                        ema_linear = auto_ema_linear
                        ema_oklab = auto_ema_oklab
                    if ema_oklab + 1e-8 < ema_linear:
                        candidate = "oklab_cdf"
                    else:
                        candidate = "linear"
                    chosen = candidate
                    if auto_temporal_stability and auto_prev_selected is not None and candidate != auto_prev_selected:
                        threshold = float(max(0.0, auto_switch_threshold))
                        prev_score = ema_linear if auto_prev_selected == "linear" else ema_oklab
                        cand_score = ema_oklab if auto_prev_selected == "linear" else ema_linear
                        if cand_score + threshold >= prev_score:
                            chosen = auto_prev_selected
                    corrected_t = linear_t if chosen == "linear" else oklab_t
                    selected_score = score_linear if chosen == "linear" else score_oklab
                    fallback_selected = None
                    fallback_score = None
                    fallback_meta = None
                    if auto_fallback_enabled and selected_score > auto_fallback_threshold_f:
                        fallback_t, fallback_meta = _run_auto_fallback_single(
                            img_t, ref_t, mm_t, auto_fallback_method_s
                        )
                        fallback_metrics = _auto_optimal_candidate_metrics(
                            fallback_t, ref_t, auto_optimal_metric
                        )
                        fallback_score = _auto_optimal_score(fallback_metrics, auto_optimal_metric)
                        if fallback_score + auto_fallback_margin_f < selected_score:
                            corrected_t = fallback_t
                            fallback_selected = auto_fallback_method_s
                    auto_prev_selected = chosen
                    mode = f"auto_optimal:{chosen}{grid_suffix}"
                    if fallback_selected is not None:
                        mode = f"{mode}:fallback:{fallback_selected}"
                    deep_params = {
                        "auto_optimal": {
                            "strategy": auto_optimal_metric,
                            "linear": linear_metrics,
                            "oklab_cdf": oklab_metrics,
                            "score_linear": round(float(score_linear), 6),
                            "score_oklab_cdf": round(float(score_oklab), 6),
                            "ema_score_linear": round(float(ema_linear), 6),
                            "ema_score_oklab_cdf": round(float(ema_oklab), 6),
                            "switch_threshold": float(auto_switch_threshold),
                            "temporal_stability": bool(auto_temporal_stability),
                            "spatial_grid": int(spatial_grid_int if spatial_grid_applied else 1),
                            "fallback_enabled": bool(auto_fallback_enabled),
                            "fallback_method": auto_fallback_method_s,
                            "fallback_threshold": auto_fallback_threshold_f,
                            "fallback_margin": auto_fallback_margin_f,
                            "fallback_score": None if fallback_score is None else round(float(fallback_score), 6),
                            "fallback_applied": bool(fallback_selected is not None),
                            "fallback_meta": fallback_meta,
                            "selected": chosen,
                        }
                    }
            elif corrected_batch is not None:
                corrected_t = corrected_batch[idx]
                mode = f"{preset}{grid_suffix}" if spatial_grid_applied else preset
            elif preset == "perceptual_vgg_fast":
                corrected_t, deep_params = _perceptual_vgg_fast(img_t, ref_t, 5, 0.05)
                mode = "perceptual_vgg_fast"
            else:
                corrected_t = img_t
                mode = "none"

            if strength < 1.0:
                corrected_t = img_t * (1.0 - strength) + corrected_t * strength
            corrected_for_lut = corrected_t.clone()

            if match_valid:
                scale_t, offset_t = scale_batch[idx], offset_batch[idx]
            else:
                scale_t = torch.ones_like(scale_batch[idx])
                offset_t = torch.zeros_like(offset_batch[idx])
            resolve_params = {
                "scale": [round(float(s), 5) for s in scale_t],
                "offset": [round(float(o), 5) for o in offset_t],
                "gamma": [1.0, 1.0, 1.0],
            }
            fusion_params = {
                "gain": resolve_params["scale"],
                "lift": resolve_params["offset"],
                "gamma": resolve_params["gamma"],
            }

            if am_t is not None:
                mask_apply = am_t[..., None]
                corrected_t = corrected_t * mask_apply + img_t * (1.0 - mask_apply)
            skin_mask_mean = None
            if skin_tone_protection:
                corrected_t, skin_mask_mean = _apply_skin_tone_protection(
                    corrected_t, img_t, float(skin_protection_strength)
                )

            corrected_t = torch.clamp(corrected_t, 0.0, 1.0)
            if effective_quality_mode == "full":
                metrics_before = _quality_metrics(img_t, ref_t)
                metrics_after = _quality_metrics(corrected_t, ref_t)
                improvement = _improvement_pct(metrics_before, metrics_after)
            elif effective_quality_mode == "fast":
                metrics_before = _quality_metrics_fast(img_t, ref_t)
                metrics_after = _quality_metrics_fast(corrected_t, ref_t)
                improvement = _improvement_pct(metrics_before, metrics_after)
            else:
                metrics_before = _empty_quality_metrics()
                metrics_after = _empty_quality_metrics()
                improvement = _empty_quality_metrics()
            matched_t = corrected_t
            if alpha_batch is not None and preserve_alpha:
                matched_t = torch.cat([matched_t, alpha_batch[idx]], dim=-1)

            stats = {
                "ref_mean": [round(float(x), 4) for x in ref_t.reshape(-1, 3).mean(dim=0)],
                "img_mean": [round(float(x), 4) for x in img_t.reshape(-1, 3).mean(dim=0)],
                "ref_std": [round(float(x), 4) for x in ref_t.reshape(-1, 3).std(dim=0)],
                "img_std": [round(float(x), 4) for x in img_t.reshape(-1, 3).std(dim=0)],
                "mask_used": mm_t is not None,
                "quality_mode": effective_quality_mode,
                "skin_tone_protection": bool(skin_tone_protection),
                "skin_mask_mean": None if skin_mask_mean is None else round(float(skin_mask_mean), 6),
                "spatial_grid": int(spatial_grid_int if spatial_grid_applied else 1),
                "spatial_grid_applied": bool(spatial_grid_applied),
            }
            payload = {
                "status": "ok",
                "preset": preset,
                "mode": mode,
                "resolve": resolve_params,
                "fusion": fusion_params,
                "linear": {
                    "scale": resolve_params["scale"],
                    "offset": resolve_params["offset"],
                },
                "deep": deep_params,
                "stats": stats,
                "quality": {
                    "before": metrics_before,
                    "after": metrics_after,
                    "improvement_pct": improvement,
                },
            }

            if export_lut:
                lut_payload = {"exported": False, "path": None, "size": int(lut_size_int), "method": None, "error": None}
                try:
                    lut_colors = _lut_grid_colors(lut_size_int, img_t.device, img_t.dtype)
                    if not match_valid:
                        lut_out = lut_colors
                        lut_method = "identity_empty_match_mask"
                    else:
                        lut_method = "baked_poly2"
                        exact_scale = None
                        exact_offset = None
                        if mode == "linear" or mode == "auto_optimal:linear":
                            exact_scale = scale_t
                            exact_offset = offset_t
                        elif mode == "mean_std" or mode == "adain":
                            exact_scale = mean_std_scale_batch[idx]
                            exact_offset = mean_std_offset_batch[idx]
                        if exact_scale is not None and exact_offset is not None:
                            mix_scale = (1.0 - float(strength)) + float(strength) * exact_scale
                            mix_offset = float(strength) * exact_offset
                            lut_out = torch.clamp(lut_colors * mix_scale[None, :] + mix_offset[None, :], 0.0, 1.0)
                            lut_method = "exact_linear"
                        else:
                            beta = _fit_poly2_color_map(img_t, corrected_for_lut, mm_t)
                            lut_out = _apply_poly2_color_map(lut_colors, beta)
                    file_name = f"{lut_base}.cube" if batch_size == 1 else f"{lut_base}_{idx:04d}.cube"
                    lut_path = (lut_dir / file_name).resolve()
                    _write_cube_file(lut_path, lut_out, lut_size_int, f"{lut_base}_{idx:04d}")
                    lut_payload["exported"] = True
                    lut_payload["path"] = str(lut_path)
                    lut_payload["method"] = lut_method
                except Exception as exc:
                    lut_payload["error"] = str(exc)
                payload["lut"] = lut_payload

            json_list.append(json.dumps(payload, ensure_ascii=True))
            matched_list.append(matched_t.cpu())

        return (
            torch.stack(matched_list, dim=0),
            json_list,
        )
