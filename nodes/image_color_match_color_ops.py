"""Tensor color-space transforms and matrix helpers for color matching."""

from __future__ import annotations

import torch

def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB values [0,1] to linear RGB."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, torch.pow((rgb + 0.055) / 1.055, 2.4))


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB values to sRGB [0,1]."""
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * torch.pow(torch.clamp(rgb, min=0.0), 1.0 / 2.4) - 0.055,
    )


def _rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor (...,3) in [0,1] to CIE Lab."""
    rgb_lin = _srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
    m_rgb_xyz = rgb_lin.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.matmul(rgb_lin, m_rgb_xyz.t())
    white = xyz.new_tensor([0.95047, 1.0, 1.08883])
    xyz_n = xyz / white
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    f = torch.where(
        xyz_n > delta3,
        torch.pow(torch.clamp(xyz_n, min=0.0), 1.0 / 3.0),
        xyz_n / (3.0 * delta * delta) + (4.0 / 29.0),
    )
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([l, a, b], dim=-1)


def _lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """Convert CIE Lab tensor (...,3) back to RGB in [0,1]."""
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def invf(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, t ** 3, 3.0 * (delta ** 2) * (t - 4.0 / 29.0))

    white = lab.new_tensor([0.95047, 1.0, 1.08883])
    xyz = torch.stack([invf(fx), invf(fy), invf(fz)], dim=-1) * white
    m_xyz_rgb = lab.new_tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb_lin = torch.matmul(xyz, m_xyz_rgb.t())
    rgb = _linear_to_srgb(rgb_lin)
    return torch.clamp(rgb, 0.0, 1.0)


def _rgb_to_oklab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (...,3) in [0,1] to Oklab."""
    rgb_lin = _srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
    m_rgb_lms = rgb.new_tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = torch.matmul(rgb_lin, m_rgb_lms.t())
    lms_cbrt = torch.pow(torch.clamp(lms, min=0.0), 1.0 / 3.0)
    m_lms_oklab = rgb.new_tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return torch.matmul(lms_cbrt, m_lms_oklab.t())


def _oklab_to_rgb_torch(oklab: torch.Tensor) -> torch.Tensor:
    """Convert Oklab tensor (...,3) to RGB in [0,1]."""
    m_oklab_lms = oklab.new_tensor(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    )
    lms_cbrt = torch.matmul(oklab, m_oklab_lms.t())
    lms = lms_cbrt ** 3
    m_lms_rgb = oklab.new_tensor(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    )
    rgb_lin = torch.matmul(lms, m_lms_rgb.t())
    rgb = _linear_to_srgb(rgb_lin)
    return torch.clamp(rgb, 0.0, 1.0)


def _interp1d_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1D linear interpolation on tensors with monotonic xp."""
    idx = torch.searchsorted(xp, x, right=False)
    max_idx = xp.shape[0] - 1
    idx0 = torch.clamp(idx - 1, 0, max_idx)
    idx1 = torch.clamp(idx, 0, max_idx)
    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]
    denom = torch.clamp(x1 - x0, min=1e-8)
    w = (x - x0) / denom
    y = y0 + w * (y1 - y0)
    y = torch.where(idx <= 0, fp[0], y)
    y = torch.where(idx >= xp.shape[0], fp[-1], y)
    return y


def _hist_match_channel_torch(
    src: torch.Tensor,
    ref: torch.Tensor,
    keep: torch.Tensor,
    bins: int,
    value_range: tuple[float, float],
) -> torch.Tensor:
    """Histogram CDF matching for one channel using torch ops only."""
    src_vals = src[keep]
    ref_vals = ref[keep]
    if src_vals.numel() < 10 or ref_vals.numel() < 10:
        return src
    vmin, vmax = value_range
    src_hist = torch.histc(src_vals.float(), bins=bins, min=vmin, max=vmax)
    ref_hist = torch.histc(ref_vals.float(), bins=bins, min=vmin, max=vmax)
    src_cdf = torch.cumsum(src_hist, dim=0)
    ref_cdf = torch.cumsum(ref_hist, dim=0)
    src_cdf = src_cdf / torch.clamp(src_cdf[-1], min=1e-8)
    ref_cdf = ref_cdf / torch.clamp(ref_cdf[-1], min=1e-8)
    bin_edges = torch.linspace(vmin, vmax, steps=bins + 1, device=src.device, dtype=torch.float32)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    interp_values = _interp1d_torch(src_cdf, ref_cdf, centers)
    src_flat = src.contiguous().view(-1).float()
    indices = torch.bucketize(src_flat, bin_edges, right=True) - 1
    indices = torch.clamp(indices, 0, bins - 1)
    matched = interp_values[indices].reshape_as(src).to(dtype=src.dtype)
    return matched


def _matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable square-root for a symmetric positive-semidefinite matrix."""
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.t()


def _matrix_invsqrt_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable inverse square-root for a symmetric positive-semidefinite matrix."""
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.t()


def _mkl_transfer_matrix(src_cov: torch.Tensor, ref_cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute MKL-like covariance transfer matrix."""
    src_sqrt = _matrix_sqrt_psd(src_cov, eps=eps)
    src_invsqrt = _matrix_invsqrt_psd(src_cov, eps=eps)
    middle = src_sqrt @ ref_cov @ src_sqrt
    middle_sqrt = _matrix_sqrt_psd(middle, eps=eps)
    return src_invsqrt @ middle_sqrt @ src_invsqrt


def _fit_mvgd_affine(src_vals: torch.Tensor, ref_vals: torch.Tensor, ridge: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a full 3x3+bias affine map from source to reference samples."""
    ones = torch.ones((src_vals.shape[0], 1), device=src_vals.device, dtype=src_vals.dtype)
    x_aug = torch.cat([src_vals, ones], dim=1)
    xtx = x_aug.t() @ x_aug
    xty = x_aug.t() @ ref_vals
    reg = torch.eye(4, device=src_vals.device, dtype=src_vals.dtype) * ridge
    reg[-1, -1] = ridge * 0.1
    w = torch.linalg.solve(xtx + reg, xty)  # [4, 3]
    a = w[:3, :].t().contiguous()
    b = w[3, :].contiguous()
    return a, b
