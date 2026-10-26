"""Model a calibrated monochrome negative/positive pair in optical-density space."""

import math

import torch
from .film_optics import halation_exposure

MODEL_ID = "alexz.neutral_bw_v1"
TRANSFERS = ("srgb", "linear")
BLACK_FLOOR = 1e-4
NEGATIVE_GAMMA = 0.65
PRINT_GAMMA = 1.0 / NEGATIVE_GAMMA
NEGATIVE_BASE = 0.1
PRINT_BASE = 0.02
NEGATIVE_WHITE = NEGATIVE_BASE - NEGATIVE_GAMMA * math.log10(BLACK_FLOOR)
_GRAY_LOG = math.log10(BLACK_FLOOR + (1 - BLACK_FLOOR) * 0.18)
_NEGATIVE_GRAY = NEGATIVE_WHITE + NEGATIVE_GAMMA * _GRAY_LOG
_PRINT_GRAY = PRINT_BASE - _GRAY_LOG
_SHADOW_LOG = math.log10(BLACK_FLOOR + (1 - BLACK_FLOOR) * 0.03)
_HIGHLIGHT_LOG = math.log10(BLACK_FLOOR + (1 - BLACK_FLOOR) * 0.5)


def decode_transfer(value, transfer):
    """Decode normalized display values into the declared linear-light space."""
    if transfer == "linear":
        return value
    if transfer != "srgb":
        raise ValueError(f"Unsupported film transfer: {transfer}.")
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def encode_transfer(value, transfer):
    """Encode a normalized linear-light positive for the ordinary IMAGE output."""
    if transfer == "linear":
        return value
    if transfer != "srgb":
        raise ValueError(f"Unsupported film transfer: {transfer}.")
    return torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.pow(1 / 2.4) - 0.055)


def scene_luma(rgb, transfer):
    """Convert source RGB to monochrome in linear light, preserving neutral grays."""
    linear = decode_transfer(rgb[..., :3], transfer)
    return linear[..., :1] * 0.2126 + linear[..., 1:2] * 0.7152 + linear[..., 2:3] * 0.0722


def _soft_limits(density, floor, ceiling, low_knee, high_knee, low_strength, high_strength):
    """Blend C1 exponential toe and shoulder curves with the straight-line response."""
    if low_strength:
        span = low_knee - floor
        delta = (density - low_knee).clamp_max(0)
        curved = low_knee + span * torch.expm1(delta / span)
        density = torch.where(density < low_knee, density + low_strength * (curved - density), density)
    if high_strength:
        span = ceiling - high_knee
        delta = (density - high_knee).clamp_min(0)
        curved = high_knee - span * torch.expm1(-delta / span)
        density = torch.where(density > high_knee, density + high_strength * (curved - density), density)
    return density


def negative_density(linear, exposure_ev=0, development_contrast=1, shadow_softness=0,
                     highlight_softness=0, fog_strength=0, halation_strength=0, halation_size=0.15):
    """Expose and develop negative stock around a fixed middle-gray density pivot."""
    if halation_strength:
        linear = halation_exposure(linear, halation_strength, halation_size)
    log_exposure = torch.log10(BLACK_FLOOR + (1 - BLACK_FLOOR) * linear) + exposure_ev * math.log10(2)
    density = _NEGATIVE_GRAY + NEGATIVE_GAMMA * development_contrast * (log_exposure - _GRAY_LOG)
    density = _soft_limits(
        density, NEGATIVE_BASE, NEGATIVE_WHITE,
        NEGATIVE_WHITE + NEGATIVE_GAMMA * _SHADOW_LOG,
        NEGATIVE_WHITE + NEGATIVE_GAMMA * _HIGHLIGHT_LOG,
        shadow_softness, highlight_softness)
    return density + 0.25 * fog_strength


def transmittance(density):
    """Convert physical density into transmittance with numerical safety headroom."""
    # Ограничения вне рабочего диапазона защищают float32; прежнего потолка D=4 нет.
    return torch.pow(10.0, -density.clamp(0, 12))


def print_density(transmission, print_exposure_ev=0, print_contrast=1, shadow_softness=0,
                  highlight_softness=0, fog_strength=0, halation_strength=0, halation_size=0.15):
    """Expose positive stock through the actual negative and develop its density."""
    if halation_strength:
        transmission = halation_exposure(transmission, halation_strength, halation_size)
    log_exposure = torch.log10(transmission.clamp_min(1e-12)) + print_exposure_ev * math.log10(2)
    nominal_density = PRINT_BASE + PRINT_GAMMA * (log_exposure + NEGATIVE_WHITE)
    density = _PRINT_GRAY + print_contrast * (nominal_density - _PRINT_GRAY)
    # У позитива тени сцены находятся в плотных участках, света — в прозрачных.
    density = _soft_limits(density, PRINT_BASE, PRINT_BASE - math.log10(BLACK_FLOOR),
                           PRINT_BASE - _HIGHLIGHT_LOG, PRINT_BASE - _SHADOW_LOG,
                           highlight_softness, shadow_softness)
    return density + 0.25 * fog_strength


def view_positive(density, transfer):
    """Apply fixed reference viewing calibration without per-image normalization."""
    linear = (transmittance(density) * 10 ** PRINT_BASE - BLACK_FLOOR) / (1 - BLACK_FLOOR)
    return encode_transfer(linear.clamp(0, 1), transfer).clamp(0, 1)


def reference_positive(transmission, transfer):
    """View an actual negative through the fixed neutral reference printing process."""
    return view_positive(print_density(transmission), transfer)
