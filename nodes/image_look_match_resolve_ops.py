"""Resolve-style look fitting, scoring, and application helpers."""

from __future__ import annotations

import math

import torch

from ..utils import color_match_utils
from .image_look_match_ops import (
    _blend_stage,
    _gradient_energy,
    _luma,
    _piecewise_linear_map,
)

_EPS = 1e-6

def _look_distance_score(candidate_hwc: torch.Tensor, reference_hwc: torch.Tensor, source_hwc: torch.Tensor) -> float:
    """Compute spatially-invariant look distance with source-structure regularization."""
    q = torch.linspace(0.0, 1.0, 21, device=candidate_hwc.device, dtype=candidate_hwc.dtype)
    cand_flat = candidate_hwc.reshape(-1, 3)
    ref_flat = reference_hwc.reshape(-1, 3)
    cand_l = _luma(candidate_hwc).reshape(-1)
    ref_l = _luma(reference_hwc).reshape(-1)
    src_l = _luma(source_hwc)
    cand_l_hw = _luma(candidate_hwc)

    cand_q = torch.quantile(cand_flat, q, dim=0)
    ref_q = torch.quantile(ref_flat, q, dim=0)
    cand_l_q = torch.quantile(cand_l, q)
    ref_l_q = torch.quantile(ref_l, q)

    cand_sat = (candidate_hwc.max(dim=-1).values - candidate_hwc.min(dim=-1).values).reshape(-1)
    ref_sat = (reference_hwc.max(dim=-1).values - reference_hwc.min(dim=-1).values).reshape(-1)
    cand_sat_q = torch.quantile(cand_sat, q)
    ref_sat_q = torch.quantile(ref_sat, q)

    dist_rgb_q = float((cand_q - ref_q).abs().mean().item())
    dist_l_q = float((cand_l_q - ref_l_q).abs().mean().item())
    dist_sat_q = float((cand_sat_q - ref_sat_q).abs().mean().item())

    cand_mean = cand_flat.mean(dim=0)
    ref_mean = ref_flat.mean(dim=0)
    cand_std = cand_flat.std(dim=0, unbiased=False).clamp_min(_EPS)
    ref_std = ref_flat.std(dim=0, unbiased=False).clamp_min(_EPS)
    dist_mean = float((cand_mean - ref_mean).abs().mean().item())
    dist_std = float((cand_std - ref_std).abs().mean().item())

    src_l_std = float(src_l.std(unbiased=False).item())
    cand_l_std = float(cand_l_hw.std(unbiased=False).item())
    std_pen = abs(math.log((cand_l_std + _EPS) / (src_l_std + _EPS)))
    src_g = _gradient_energy(src_l)
    cand_g = _gradient_energy(cand_l_hw)
    grad_pen = abs(math.log((cand_g + _EPS) / (src_g + _EPS)))

    return (
        dist_l_q * 0.80
        + dist_rgb_q * 0.60
        + dist_sat_q * 0.35
        + dist_mean * 0.45
        + dist_std * 0.35
        + std_pen * 0.12
        + grad_pen * 0.08
    )


def _clip_ratio(rgb_hwc: torch.Tensor, low: float = 0.005, high: float = 0.995) -> float:
    """Estimate clipped pixel ratio in RGB domain."""
    clipped = ((rgb_hwc <= low) | (rgb_hwc >= high)).to(dtype=rgb_hwc.dtype)
    return float(clipped.mean().item())


def _contrast_ratio_to_source(candidate_hwc: torch.Tensor, source_hwc: torch.Tensor) -> float:
    """Return luma contrast ratio candidate/source."""
    src_std = float(_luma(source_hwc).std(unbiased=False).item())
    cand_std = float(_luma(candidate_hwc).std(unbiased=False).item())
    return cand_std / max(src_std, _EPS)


def _soften_candidate_tone(candidate_hwc: torch.Tensor, source_hwc: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalize candidate tone toward source to reduce harsh/solarized look."""
    src_l = _luma(source_hwc).clamp(0.0, 1.0)
    cand_l = _luma(candidate_hwc).clamp(0.0, 1.0)
    src_std = float(src_l.std(unbiased=False).item())
    cand_std = float(cand_l.std(unbiased=False).item())
    ratio = cand_std / max(src_std, _EPS)

    target_ratio = ratio
    if ratio > 1.08:
        target_ratio = 1.00
    elif ratio > 1.02:
        target_ratio = 1.02
    elif ratio < 0.78:
        target_ratio = 0.90
    elif ratio < 0.92:
        target_ratio = 0.92

    if abs(target_ratio - ratio) < 1e-6:
        return candidate_hwc, {"contrast_ratio": float(ratio), "blend": 0.0}

    target_std = src_std * target_ratio
    gain = target_std / max(cand_std, _EPS)
    cand_mean = float(cand_l.mean().item())
    adjusted_l = ((cand_l - cand_mean) * gain + cand_mean).clamp(0.0, 1.0)
    src_bias = float(max(0.0, min(0.45, (abs(math.log(max(ratio, _EPS))) - 0.03) / 0.20)))
    if src_bias > 0.0:
        adjusted_l = (adjusted_l * (1.0 - src_bias) + src_l * src_bias).clamp(0.0, 1.0)
    adjusted_rgb = (candidate_hwc * (adjusted_l / cand_l.clamp_min(_EPS)).unsqueeze(-1)).clamp(0.0, 1.0)

    # More aggressive when tone deviates strongly or has visible clipping.
    dev = abs(math.log(max(ratio, _EPS)))
    clip = _clip_ratio(candidate_hwc)
    blend = float(max(0.0, min(0.92, (dev - 0.03) / 0.14)))
    blend = max(blend, min(0.92, clip * 22.0))
    if ratio > 1.15:
        blend = max(blend, 0.85)
    softened = (candidate_hwc * (1.0 - blend) + adjusted_rgb * blend).clamp(0.0, 1.0)
    return softened, {"contrast_ratio": float(ratio), "blend": float(blend)}


def _rank_candidate_for_look(candidate_hwc: torch.Tensor, reference_hwc: torch.Tensor, source_hwc: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute ranking score: look match + penalties for harshness/flatness/clipping."""
    look = _look_distance_score(candidate_hwc, reference_hwc, source_hwc)
    contrast_ratio = _contrast_ratio_to_source(candidate_hwc, source_hwc)
    clip = _clip_ratio(candidate_hwc)
    harsh_pen = max(0.0, contrast_ratio - 1.04) * 1.8
    flat_pen = max(0.0, 0.84 - contrast_ratio) * 1.2
    clip_pen = clip * 10.0
    score = float(look + harsh_pen + flat_pen + clip_pen)
    return score, float(look), float(contrast_ratio), float(clip)


def _fit_tone_params(src_hwc: torch.Tensor, ref_hwc: torch.Tensor, tone_model: str) -> dict:
    """Fit tone model parameters from source/reference pair."""
    src_l = _luma(src_hwc).clamp(0.0, 1.0)
    ref_l = _luma(ref_hwc).clamp(0.0, 1.0)
    model = str(tone_model)
    if model == "gamma_gain_lift":
        src_m = src_l.mean().clamp_min(_EPS)
        ref_m = ref_l.mean().clamp_min(_EPS)
        gamma = (torch.log(ref_m) / torch.log(src_m)).clamp(0.35, 3.0)
        mapped = src_l.clamp_min(_EPS).pow(gamma)
        mapped_m = mapped.mean()
        mapped_std = mapped.std(unbiased=False).clamp_min(_EPS)
        ref_std = ref_l.std(unbiased=False).clamp_min(_EPS)
        gain = (ref_std / mapped_std).clamp(0.25, 4.0)
        lift = (ref_m - mapped_m * gain).clamp(-0.5, 0.5)
        return {
            "type": "gamma_gain_lift",
            "gamma": float(gamma.item()),
            "gain": float(gain.item()),
            "lift": float(lift.item()),
        }
    q = torch.linspace(0.0, 1.0, 33, device=src_hwc.device, dtype=src_hwc.dtype)
    src_q = torch.quantile(src_l.reshape(-1), q)
    ref_q = torch.quantile(ref_l.reshape(-1), q)
    return {
        "type": "monotonic_spline",
        "xk": src_q.detach(),
        "yk": ref_q.detach(),
    }


def _apply_tone_model(src_hwc: torch.Tensor, tone_params: dict) -> torch.Tensor:
    """Apply fitted tone model while preserving chroma relation."""
    src_l = _luma(src_hwc).clamp(0.0, 1.0)
    ttype = str(tone_params.get("type", "monotonic_spline"))
    if ttype == "gamma_gain_lift":
        gamma = float(tone_params.get("gamma", 1.0))
        gain = float(tone_params.get("gain", 1.0))
        lift = float(tone_params.get("lift", 0.0))
        mapped_l = (src_l.clamp_min(_EPS).pow(gamma) * gain + lift).clamp(0.0, 1.0)
    else:
        xk = tone_params.get("xk")
        yk = tone_params.get("yk")
        if not isinstance(xk, torch.Tensor) or not isinstance(yk, torch.Tensor):
            return src_hwc
        mapped_l = _piecewise_linear_map(src_l, xk.to(src_hwc.device), yk.to(src_hwc.device))
    ratio = (mapped_l / src_l.clamp_min(_EPS)).clamp(0.0, 4.0).unsqueeze(-1)
    return (src_hwc * ratio).clamp(0.0, 1.0)


def _fit_palette_affine(src_hwc: torch.Tensor, ref_hwc: torch.Tensor, fit_mask_hw: torch.Tensor | None, palette_model: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-channel affine palette transform."""
    src_b = src_hwc.unsqueeze(0)
    ref_b = ref_hwc.unsqueeze(0)
    mask_b = fit_mask_hw.unsqueeze(0) if fit_mask_hw is not None else None
    if str(palette_model) == "rbf":
        scale, offset = color_match_utils.mean_std_fit_torch_batch(src_b, ref_b, mask_b)
    else:
        scale, offset = color_match_utils.linear_fit_torch_batch(src_b, ref_b, mask_b)
    return scale[0].clamp(0.2, 5.0), offset[0].clamp(-1.0, 1.0)


def _estimate_skin_mask(rgb_hwc: torch.Tensor) -> torch.Tensor:
    """Estimate soft skin mask from RGB heuristics."""
    r = rgb_hwc[..., 0]
    g = rgb_hwc[..., 1]
    b = rgb_hwc[..., 2]
    cond = (
        (r > 0.35)
        & (g > 0.20)
        & (b > 0.10)
        & (r > g)
        & (r > b)
        & ((r - g) > 0.02)
    )
    return cond.to(dtype=rgb_hwc.dtype)


def _apply_resolve_pipeline_to_rgb(
    src_hwc: torch.Tensor,
    exposure_gain: torch.Tensor,
    exposure_alpha: float,
    tone_params: dict,
    tone_alpha: float,
    palette_scale: torch.Tensor,
    palette_offset: torch.Tensor,
    palette_alpha: float,
) -> torch.Tensor:
    """Apply resolve stages to one RGB image tensor."""
    exp_stage = (src_hwc * exposure_gain.view(1, 1, 3)).clamp(0.0, 1.0)
    cur = _blend_stage(src_hwc, exp_stage, exposure_alpha).clamp(0.0, 1.0)

    tone_stage = _apply_tone_model(cur, tone_params)
    cur = _blend_stage(cur, tone_stage, tone_alpha).clamp(0.0, 1.0)

    pal_stage = (cur * palette_scale.view(1, 1, 3) + palette_offset.view(1, 1, 3)).clamp(0.0, 1.0)
    cur = _blend_stage(cur, pal_stage, palette_alpha).clamp(0.0, 1.0)
    return cur


def _build_resolve_cube_text(
    size: int,
    exposure_gain: torch.Tensor,
    exposure_alpha: float,
    tone_params: dict,
    tone_alpha: float,
    palette_scale: torch.Tensor,
    palette_offset: torch.Tensor,
    palette_alpha: float,
) -> str:
    """Bake current resolve parameters into .cube text."""
    n = max(2, int(size))
    device = exposure_gain.device
    vals = torch.linspace(0.0, 1.0, n, device=device, dtype=exposure_gain.dtype)
    bb, gg, rr = torch.meshgrid(vals, vals, vals, indexing="ij")
    rgb = torch.stack([rr, gg, bb], dim=-1).reshape(-1, 3)
    rgb_hwc = rgb.reshape(-1, 1, 3)
    corrected = _apply_resolve_pipeline_to_rgb(
        rgb_hwc,
        exposure_gain=exposure_gain,
        exposure_alpha=exposure_alpha,
        tone_params=tone_params,
        tone_alpha=tone_alpha,
        palette_scale=palette_scale,
        palette_offset=palette_offset,
        palette_alpha=palette_alpha,
    ).reshape(-1, 3)

    lines = [
        "# Generated by ALEXZ_tools Look Match Resolve Phase B",
        f"LUT_3D_SIZE {n}",
    ]
    for row in corrected.detach().cpu().tolist():
        lines.append(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    return "\n".join(lines) + "\n"
