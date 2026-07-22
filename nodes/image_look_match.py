"""
Module: nodes/image_look_match.py
Author: AlexZ1967
Last updated: 2026-02-18

Description:
    Look Match nodes for strong-reference scenarios.

Purpose:
    Provides Resolve-style look transfer (Phase B MVP) and Nuke-style
    Build/Apply contract nodes (Phase A baseline) with stable JSON schemas.
"""

from __future__ import annotations

import json

import torch
from tqdm import tqdm

from ..utils import color_match_utils
from ..utils.interrupt import check_interrupt

_LOOK_RESOLVE_SCHEMA = "alexz.look_match.resolve"
_LOOK_MODEL_SCHEMA = "alexz.look_model.nuke_build"
_LOOK_APPLY_SCHEMA = "alexz.look_apply.nuke_apply"
_SCHEMA_VERSION = 1
_EPS = 1e-6
_LONG_SIDE_TARGET = {
    "as_is": None,
    "1440p": 1440,
    "1080p": 1080,
    "720p": 720,
}


from .image_look_match_ops import (
    _blend_stage,
    _compose_fit_mask,
    _downscale_hwc_long_side,
    _fit_exposure_gain,
    _pad_batch_last,
    _prepare_optional_mask_batch,
    _resize_hwc_to,
    _resize_mask_hw,
    _resolve_compute_device,
    _split_alpha,
    _stage_alpha,
)
from .image_look_match_resolve_ops import (
    _apply_resolve_pipeline_to_rgb,
    _apply_tone_model,
    _build_resolve_cube_text,
    _estimate_skin_mask,
    _fit_palette_affine,
    _fit_tone_params,
    _rank_candidate_for_look,
    _soften_candidate_tone,
)
from .image_look_match_contract_ops import (
    _build_nuke_apply_input_types,
    _build_nuke_build_input_types,
    _build_resolve_input_types,
    _identity_cube_text,
    _safe_json_loads,
)

class ImageLookMatchResolve:
    """Resolve-style monolithic look-match node (Phase B MVP)."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES for resolve-style look transfer."""
        return _build_resolve_input_types()

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("matched_image", "look_json", "cube_text")
    FUNCTION = "match"
    CATEGORY = "image/color"

    def match(
        self,
        reference,
        image,
        strength=1.0,
        compute_device="auto",
        working_space="oklab",
        downscale_long_side="1080p",
        tone_model="monotonic_spline",
        palette_model="lut3d",
        lut_size=25,
        w_exposure=1.0,
        w_tone=1.0,
        w_chroma=1.0,
        skin_protection=True,
        skin_protection_strength=0.6,
        auto_fallback_cdf=True,
        subject_mask=None,
        sky_mask=None,
        ground_mask=None,
        export_lut_cube=False,
    ):
        """Run Resolve-style look transfer with staged fit/apply pipeline."""
        batch_size = max(reference.shape[0], image.shape[0])
        ref_batch = _pad_batch_last(reference, batch_size)
        img_batch = _pad_batch_last(image, batch_size)

        ref_rgb, _ = _split_alpha(ref_batch)
        img_rgb, img_alpha = _split_alpha(img_batch)
        img_h, img_w = img_rgb.shape[1], img_rgb.shape[2]
        subject_batch = _prepare_optional_mask_batch(subject_mask, batch_size, img_h, img_w)
        sky_batch = _prepare_optional_mask_batch(sky_mask, batch_size, img_h, img_w)
        ground_batch = _prepare_optional_mask_batch(ground_mask, batch_size, img_h, img_w)

        device, warning = _resolve_compute_device(compute_device)
        s = float(max(0.0, min(1.0, strength)))
        exposure_alpha = _stage_alpha(float(w_exposure))
        tone_alpha = _stage_alpha(float(w_tone))
        palette_alpha = _stage_alpha(float(w_chroma))
        skin_alpha = float(max(0.0, min(1.0, float(skin_protection_strength))))
        out_list = []
        json_list = []
        cube_list = []

        iterator = tqdm(range(batch_size), desc="LookMatch[resolve]", unit="img")
        for idx in iterator:
            check_interrupt()
            ref_t = ref_rgb[idx].detach().to(device=device).clone()
            img_t = img_rgb[idx].detach().to(device=device).clone()
            subject_t = subject_batch[idx].to(device=device) if subject_batch is not None else None
            sky_t = sky_batch[idx].to(device=device) if sky_batch is not None else None
            ground_t = ground_batch[idx].to(device=device) if ground_batch is not None else None

            fit_mask_full = _compose_fit_mask(subject_t, sky_t, ground_t)
            img_fit, ds_info = _downscale_hwc_long_side(img_t, downscale_long_side)
            ref_fit, _ = _downscale_hwc_long_side(ref_t, downscale_long_side)
            if ref_fit.shape[0] != img_fit.shape[0] or ref_fit.shape[1] != img_fit.shape[1]:
                ref_fit = _resize_hwc_to(ref_fit, img_fit.shape[0], img_fit.shape[1])
            fit_mask = _resize_mask_hw(fit_mask_full, img_fit.shape[0], img_fit.shape[1]) if fit_mask_full is not None else None

            exposure_gain = _fit_exposure_gain(img_fit, ref_fit, fit_mask)
            tone_fit_src = _blend_stage(
                img_fit,
                (img_fit * exposure_gain.view(1, 1, 3)).clamp(0.0, 1.0),
                exposure_alpha,
            )
            tone_params = _fit_tone_params(tone_fit_src, ref_fit, tone_model)
            tone_fit_img = _apply_tone_model(tone_fit_src, tone_params)
            palette_fit_src = _blend_stage(tone_fit_src, tone_fit_img, tone_alpha)
            palette_scale, palette_offset = _fit_palette_affine(palette_fit_src, ref_fit, fit_mask, palette_model)

            corrected = _apply_resolve_pipeline_to_rgb(
                img_t,
                exposure_gain=exposure_gain,
                exposure_alpha=exposure_alpha,
                tone_params=tone_params,
                tone_alpha=tone_alpha,
                palette_scale=palette_scale,
                palette_offset=palette_offset,
                palette_alpha=palette_alpha,
            )
            if bool(skin_protection) and skin_alpha > 0.0:
                skin_mask_soft = _estimate_skin_mask(img_t).unsqueeze(-1)
                corrected = corrected * (1.0 - skin_mask_soft * skin_alpha) + img_t * (skin_mask_soft * skin_alpha)
            if s < 1.0:
                corrected = img_t * (1.0 - s) + corrected * s
            corrected = corrected.clamp(0.0, 1.0)

            ref_eval = ref_t
            if ref_eval.shape[0] != img_t.shape[0] or ref_eval.shape[1] != img_t.shape[1]:
                ref_eval = _resize_hwc_to(ref_eval, img_t.shape[0], img_t.shape[1])
            mse_before = float((img_t - ref_eval).square().mean().item())
            mse_after = float((corrected - ref_eval).square().mean().item())
            rank_before, look_before, contrast_before, clip_before = _rank_candidate_for_look(img_t, ref_eval, img_t)
            rank_after, look_after, contrast_after, clip_after = _rank_candidate_for_look(corrected, ref_eval, img_t)
            fallback_used = False
            fallback_mode = ""
            fallback_mse = None
            fallback_look = None
            fallback_rank = None
            fallback_contrast = None
            fallback_clip = None
            fallback_source = ""
            if bool(auto_fallback_cdf):
                # If resolve-stage fit is weak, try robust alternatives and rank by look+tone quality.
                improve_ratio = (mse_before - mse_after) / max(mse_before, 1e-8)
                look_improve_ratio = (look_before - look_after) / max(look_before, 1e-8)
                if improve_ratio < 0.05 or mse_after > 0.03 or look_improve_ratio < 0.04 or rank_after > 0.22:
                    cdf_mode = "oklab_cdf" if str(working_space) == "oklab" else "lab_cdf"
                    candidates = [cdf_mode, "linear"]
                    palette_tag = str(palette_model).lower()
                    if palette_tag == "rbf":
                        # RBF path: fallback from resolve base to keep its tonal character.
                        fallback_sources = [("resolve_base", corrected, False)]
                    else:
                        # LUT/linear path: fallback from original image for stronger global transfer.
                        fallback_sources = [("source", img_t, True)]
                    best_out = corrected
                    best_mse = mse_after
                    best_look = look_after
                    best_rank = rank_after
                    best_contrast = contrast_after
                    best_clip = clip_after
                    zero_mask = torch.zeros(
                        (1, img_t.shape[0], img_t.shape[1]),
                        device=img_t.device,
                        dtype=img_t.dtype,
                    )
                    for source_name, source_tensor, reapply_stages in fallback_sources:
                        for cand_mode in candidates:
                            cand_raw = color_match_utils.apply_color_match(
                                source_tensor.unsqueeze(0),
                                ref_eval.unsqueeze(0),
                                zero_mask,
                                mode=cand_mode,
                                mask_white_is_keep=False,
                            )[0].to(device=img_t.device, dtype=img_t.dtype)
                            if reapply_stages and bool(skin_protection) and skin_alpha > 0.0:
                                skin_mask_soft = _estimate_skin_mask(img_t).unsqueeze(-1)
                                cand_raw = cand_raw * (1.0 - skin_mask_soft * skin_alpha) + img_t * (skin_mask_soft * skin_alpha)
                            if reapply_stages and s < 1.0:
                                cand_raw = img_t * (1.0 - s) + cand_raw * s
                            cand_raw = cand_raw.clamp(0.0, 1.0)

                            cand_soft, _soft_info = _soften_candidate_tone(cand_raw, img_t)
                            for cand_label, cand_out in ((cand_mode, cand_raw), (f"{cand_mode}_soft", cand_soft)):
                                cand_out = cand_out.clamp(0.0, 1.0)
                                cand_mse = float((cand_out - ref_eval).square().mean().item())
                                if cand_mse > (mse_before * 2.5 + 1e-8):
                                    continue
                                cand_rank, cand_look, cand_contrast, cand_clip = _rank_candidate_for_look(cand_out, ref_eval, img_t)
                                better_rank = cand_rank < (best_rank - 1e-6)
                                near_rank_better_mse = cand_rank <= (best_rank + 0.01) and cand_mse < best_mse
                                if better_rank or near_rank_better_mse:
                                    best_out = cand_out
                                    best_mse = cand_mse
                                    best_look = cand_look
                                    best_rank = cand_rank
                                    best_contrast = cand_contrast
                                    best_clip = cand_clip
                                    fallback_mode = cand_label
                                    fallback_source = source_name
                    if fallback_mode:
                        rank_gain = best_rank < (rank_after - 0.01)
                        mse_gain = best_mse < (mse_after * 0.9)
                        if not (rank_gain or mse_gain):
                            fallback_mode = ""
                    if fallback_mode:
                        corrected = best_out
                        mse_after = best_mse
                        look_after = best_look
                        rank_after = best_rank
                        contrast_after = best_contrast
                        clip_after = best_clip
                        fallback_used = True
                        fallback_mse = best_mse
                        fallback_look = best_look
                        fallback_rank = best_rank
                        fallback_contrast = best_contrast
                        fallback_clip = best_clip

            out_t = corrected.to(device=img_rgb.device)
            if img_alpha is not None:
                out_t = torch.cat([out_t, img_alpha[idx]], dim=-1)
            out_list.append(out_t.cpu())

            payload = {
                "status": "ok",
                "schema_name": _LOOK_RESOLVE_SCHEMA,
                "schema_version": _SCHEMA_VERSION,
                "mode": f"look_match_resolve:{working_space}",
                "phase": "B_resolve_mvp",
                "contracts": {
                    "tone_model": str(tone_model),
                    "palette_model": str(palette_model),
                    "downscale_long_side": str(downscale_long_side),
                },
                "fit": {
                    "optimized_size": ds_info.get("optimized_size"),
                    "exposure_alpha": round(float(exposure_alpha), 6),
                    "tone_alpha": round(float(tone_alpha), 6),
                    "palette_alpha": round(float(palette_alpha), 6),
                },
                "transform": {
                    "exposure_gain": [round(float(v), 6) for v in exposure_gain.detach().cpu().tolist()],
                    "palette_scale": [round(float(v), 6) for v in palette_scale.detach().cpu().tolist()],
                    "palette_offset": [round(float(v), 6) for v in palette_offset.detach().cpu().tolist()],
                    "tone_type": str(tone_params.get("type", "unknown")),
                },
                "optimization": {
                    "compute_device_requested": str(compute_device).lower(),
                    "compute_device_effective": str(device),
                    "device_warning": warning,
                    "auto_fallback_cdf": bool(auto_fallback_cdf),
                    "weights": {
                        "w_exposure": float(w_exposure),
                        "w_tone": float(w_tone),
                        "w_chroma": float(w_chroma),
                        "skin_protection": bool(skin_protection),
                        "skin_protection_strength": float(skin_protection_strength),
                    },
                },
                "masks": {
                    "subject_mask": bool(subject_mask is not None),
                    "sky_mask": bool(sky_mask is not None),
                    "ground_mask": bool(ground_mask is not None),
                },
                "quality": {
                    "before": {"mse": round(mse_before, 8)},
                    "after": {"mse": round(mse_after, 8)},
                    "look_score": {
                        "before": round(float(look_before), 8),
                        "after": round(float(look_after), 8),
                    },
                    "rank_score": {
                        "before": round(float(rank_before), 8),
                        "after": round(float(rank_after), 8),
                    },
                    "tone_profile": {
                        "before": {
                            "contrast_ratio_to_source": round(float(contrast_before), 8),
                            "clip_ratio": round(float(clip_before), 8),
                        },
                        "after": {
                            "contrast_ratio_to_source": round(float(contrast_after), 8),
                            "clip_ratio": round(float(clip_after), 8),
                        },
                    },
                    "fallback": {
                        "used": bool(fallback_used),
                        "mode": str(fallback_mode),
                        "mse": round(float(fallback_mse), 8) if fallback_mse is not None else None,
                        "look_score": round(float(fallback_look), 8) if fallback_look is not None else None,
                        "rank_score": round(float(fallback_rank), 8) if fallback_rank is not None else None,
                        "contrast_ratio_to_source": round(float(fallback_contrast), 8) if fallback_contrast is not None else None,
                        "clip_ratio": round(float(fallback_clip), 8) if fallback_clip is not None else None,
                        "source": str(fallback_source),
                    },
                    "improvement_pct": {
                        "mse": round((mse_before - mse_after) / max(mse_before, 1e-8) * 100.0, 3),
                        "look_score": round((look_before - look_after) / max(look_before, 1e-8) * 100.0, 3),
                        "rank_score": round((rank_before - rank_after) / max(rank_before, 1e-8) * 100.0, 3),
                    },
                },
            }
            json_list.append(json.dumps(payload, ensure_ascii=True))
            cube_list.append(
                _build_resolve_cube_text(
                    lut_size,
                    exposure_gain=exposure_gain,
                    exposure_alpha=exposure_alpha,
                    tone_params=tone_params,
                    tone_alpha=tone_alpha,
                    palette_scale=palette_scale,
                    palette_offset=palette_offset,
                    palette_alpha=palette_alpha,
                )
                if export_lut_cube
                else ""
            )

        return (torch.stack(out_list, dim=0), json_list, cube_list)


class ImageLookMatchNukeBuild:
    """Nuke-style look-model builder (Phase A schema-first implementation)."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES for look-model builder node."""
        return _build_nuke_build_input_types()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("look_model_json", "cube_text")
    FUNCTION = "build"
    CATEGORY = "image/color"

    def build(
        self,
        reference,
        source,
        compute_device="auto",
        working_space="oklab",
        downscale_long_side="1080p",
        fit_global=True,
        fit_tone=True,
        fit_hue_sectors=True,
        fit_local_regions=True,
        skin_mask=None,
        sky_mask=None,
        ground_mask=None,
        subject_mask=None,
        export_lut_cube=False,
        lut_size=25,
    ):
        """Build and return Phase A look-model JSON contract."""
        check_interrupt()
        ref = reference[0]
        src = source[0]
        device, warning = _resolve_compute_device(compute_device)

        model = {
            "status": "ok",
            "schema_name": _LOOK_MODEL_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "mode": f"look_model_build:{working_space}",
            "phase": "A_contract_baseline",
            "build_context": {
                "compute_device_requested": str(compute_device).lower(),
                "compute_device_effective": str(device),
                "device_warning": warning,
                "downscale_long_side": str(downscale_long_side),
                "source_size": [int(src.shape[0]), int(src.shape[1])],
                "reference_size": [int(ref.shape[0]), int(ref.shape[1])],
            },
            "fit_flags": {
                "fit_global": bool(fit_global),
                "fit_tone": bool(fit_tone),
                "fit_hue_sectors": bool(fit_hue_sectors),
                "fit_local_regions": bool(fit_local_regions),
            },
            "mask_flags": {
                "skin_mask": bool(skin_mask is not None),
                "sky_mask": bool(sky_mask is not None),
                "ground_mask": bool(ground_mask is not None),
                "subject_mask": bool(subject_mask is not None),
            },
            "transform": {
                "global_affine": {
                    "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "bias": [0.0, 0.0, 0.0],
                },
                "tone_profile": {"type": "identity", "params": {}},
                "hue_sectors": [],
                "local_regions": [],
            },
        }
        cube_text = _identity_cube_text(lut_size) if export_lut_cube else ""
        return (json.dumps(model, ensure_ascii=True), cube_text)


class ImageLookMatchNukeApply:
    """Nuke-style look-model applier (Phase A identity-safe behavior)."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES for look-model applier node."""
        return _build_nuke_apply_input_types()

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "apply_json")
    FUNCTION = "apply"
    CATEGORY = "image/color"

    def apply(
        self,
        image,
        look_model_json,
        strength=1.0,
        compute_device="auto",
        temporal_stabilization=False,
        temporal_alpha=0.8,
        shot_change_threshold=0.2,
    ):
        """Apply look-model JSON to input image batch."""
        model = _safe_json_loads(look_model_json)
        model_valid = model.get("schema_name") == _LOOK_MODEL_SCHEMA and int(model.get("schema_version", -1)) == _SCHEMA_VERSION

        batch_size = image.shape[0]
        img_rgb, img_alpha = _split_alpha(image)
        device, warning = _resolve_compute_device(compute_device)
        s = float(max(0.0, min(1.0, strength)))

        matrix = torch.eye(3, dtype=torch.float32, device=device)
        bias = torch.zeros(3, dtype=torch.float32, device=device)
        if model_valid:
            affine = model.get("transform", {}).get("global_affine", {})
            mat_list = affine.get("matrix")
            bias_list = affine.get("bias")
            if isinstance(mat_list, list) and len(mat_list) == 3:
                try:
                    matrix = torch.tensor(mat_list, dtype=torch.float32, device=device).reshape(3, 3)
                except Exception:
                    matrix = torch.eye(3, dtype=torch.float32, device=device)
            if isinstance(bias_list, list) and len(bias_list) == 3:
                try:
                    bias = torch.tensor(bias_list, dtype=torch.float32, device=device).reshape(3)
                except Exception:
                    bias = torch.zeros(3, dtype=torch.float32, device=device)

        out_list = []
        json_list = []
        iterator = tqdm(range(batch_size), desc="LookMatch[nuke_apply]", unit="img")
        for idx in iterator:
            check_interrupt()
            src = img_rgb[idx].detach().to(device=device).clone()
            corrected = torch.einsum("hwc,dc->hwd", src, matrix) + bias.view(1, 1, 3)
            corrected = corrected.clamp(0.0, 1.0)
            if s < 1.0:
                corrected = src * (1.0 - s) + corrected * s
            out_t = corrected.to(device=img_rgb.device)
            if img_alpha is not None:
                out_t = torch.cat([out_t, img_alpha[idx]], dim=-1)
            out_list.append(out_t.cpu())

            payload = {
                "status": "ok" if model_valid else "warning",
                "schema_name": _LOOK_APPLY_SCHEMA,
                "schema_version": _SCHEMA_VERSION,
                "mode": "look_model_apply",
                "phase": "A_contract_baseline",
                "model_loaded": bool(model_valid),
                "warning": None if model_valid else "invalid_or_missing_look_model_schema",
                "apply_context": {
                    "compute_device_requested": str(compute_device).lower(),
                    "compute_device_effective": str(device),
                    "device_warning": warning,
                    "temporal_stabilization": bool(temporal_stabilization),
                    "temporal_alpha": float(temporal_alpha),
                    "shot_change_threshold": float(shot_change_threshold),
                },
            }
            json_list.append(json.dumps(payload, ensure_ascii=True))

        return (torch.stack(out_list, dim=0), json_list)
