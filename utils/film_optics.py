"""Scatter a bounded fraction of linear projection light with a normalized PSF."""

import math

import torch


def _diffuse_axis(values, sigma, axis):
    """Apply conservative discrete Gaussian diffusion with symmetric boundaries."""
    count = values.size(axis)
    if count == 1:
        return values
    # Симметрия с повторением крайнего пикселя сохраняет и ровное поле, и сумму света.
    mirrored = torch.cat((values, values.flip((axis,))), dim=axis)
    frequencies = torch.fft.rfftfreq(2 * count, device=values.device, dtype=values.dtype)
    response = torch.exp(-2 * (sigma * torch.sin(math.pi * frequencies)).square())
    shape = [1] * values.ndim
    shape[axis] = response.numel()
    spectrum = torch.fft.rfft(mirrored, dim=axis)
    blurred = torch.fft.irfft(spectrum * response.view(shape), n=2 * count, dim=axis)
    return blurred.narrow(axis, 0, count).contiguous()


def projection_glow(linear_rgb, strength, size_percent):
    """Redistribute light into a width-relative halo while retaining the sharp core."""
    if strength == 0:
        return linear_rgb
    sigma = linear_rgb.size(2) * size_percent / 100.0
    # Две одномерные FFT ограничивают рабочую память и не требуют огромных ядер.
    blurred = _diffuse_axis(linear_rgb, sigma, 2)
    blurred = _diffuse_axis(blurred, sigma, 1)
    return linear_rgb * (1 - strength) + blurred * strength


def halation_exposure(incident, strength, size_percent):
    """Add calibrated substrate return light before development, preserving flat fields."""
    if strength == 0:
        return incident
    returned = projection_glow(incident, 1, size_percent)
    # Постоянная калибровка чувствительности материала, без анализа гистограммы.
    return ((incident + strength * returned) / (1 + strength)).clamp_min(0)
