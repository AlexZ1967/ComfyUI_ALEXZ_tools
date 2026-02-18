"""
Module: utils/color_match_utils.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Color matching helper functions.

Purpose:
    Provides reusable color-transfer algorithms and mask/resize helpers for color-processing nodes.
"""

import numpy as np

import torch
import torch.nn.functional as torch_nn_func

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Normalize mask shape to BCHW layout and float range [0, 1]."""
    mask = mask.float()
    max_val = float(mask.max()) if mask.numel() else 0.0
    if max_val > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def ensure_mask_batch(mask: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Ensure mask has a batch dimension matching image batches."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.size(0) == 1 and frame_count > 1:
        mask = mask.repeat(frame_count, 1, 1)
    return mask


def resize_mask_to_output(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize mask to output tensor spatial resolution."""
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(1)
    resized = torch_nn_func.interpolate(mask, size=(height, width), mode="nearest")
    return resized.squeeze(1)


def resize_images_to_size(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize source and reference images to a shared target resolution."""
    if images.shape[1] == height and images.shape[2] == width:
        return images
    images_bchw = images.permute(0, 3, 1, 2)
    resized = torch_nn_func.interpolate(
        images_bchw,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1)


def _match_mean_std_channel(src: np.ndarray, ref: np.ndarray, keep: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Match one channel to reference mean/std values."""
    src_keep = src[keep]
    ref_keep = ref[keep]
    if src_keep.size < 10 or ref_keep.size < 10:
        return src
    src_mean = float(src_keep.mean())
    ref_mean = float(ref_keep.mean())
    src_std = float(src_keep.std())
    ref_std = float(ref_keep.std())
    scale = ref_std / max(src_std, eps)
    return (src - src_mean) * scale + ref_mean


def _match_linear_channel(src: np.ndarray, ref: np.ndarray, keep: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Match one channel using robust linear fit against reference values."""
    src_keep = src[keep]
    ref_keep = ref[keep]
    if src_keep.size < 10 or ref_keep.size < 10:
        return src
    mean_x = float(src_keep.mean())
    mean_y = float(ref_keep.mean())
    var_x = float(((src_keep - mean_x) ** 2).mean())
    if var_x < eps:
        a = 1.0
        b = mean_y - mean_x
    else:
        cov_xy = float(((src_keep - mean_x) * (ref_keep - mean_y)).mean())
        a = cov_xy / var_x
        b = mean_y - a * mean_x
    return src * a + b


def _match_histogram_channel(
    src: np.ndarray,
    ref: np.ndarray,
    keep: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    """Match one channel using histogram CDF transfer."""
    src_keep = src[keep]
    ref_keep = ref[keep]
    if src_keep.size < 10 or ref_keep.size < 10:
        return src
    src_hist, bin_edges = np.histogram(src_keep, bins=bins, range=value_range, density=True)
    ref_hist, _ = np.histogram(ref_keep, bins=bins, range=value_range, density=True)
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)
    if src_cdf[-1] > 0:
        src_cdf = src_cdf / src_cdf[-1]
    if ref_cdf[-1] > 0:
        ref_cdf = ref_cdf / ref_cdf[-1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    interp_values = np.interp(src_cdf, ref_cdf, bin_centers)
    indices = np.searchsorted(bin_edges, src.ravel(), side="right") - 1
    indices = np.clip(indices, 0, bins - 1)
    matched = interp_values[indices].reshape(src.shape)
    return matched


def _match_mean_std_rgb(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Apply per-channel mean/std matching in RGB space."""
    for c in range(3):
        out_np[..., c] = _match_mean_std_channel(out_np[..., c], ref_np[..., c], keep)
    return out_np


def _match_linear_rgb(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Apply per-channel linear matching in RGB space."""
    for c in range(3):
        out_np[..., c] = _match_linear_channel(out_np[..., c], ref_np[..., c], keep)
    return out_np


def _match_hist_rgb(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Apply per-channel histogram matching in RGB space."""
    for c in range(3):
        out_np[..., c] = _match_histogram_channel(
            out_np[..., c], ref_np[..., c], keep, bins=256, value_range=(0.0, 1.0)
        )
    return out_np


def _pca_cov_transfer(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Transfer source color covariance to reference covariance via PCA transform."""
    src = out_np.reshape(-1, 3)
    tar = ref_np.reshape(-1, 3)
    if keep is not None and keep.shape[:2] == out_np.shape[:2]:
        mask_flat = keep.reshape(-1)
        src = src[mask_flat]
        tar = tar[mask_flat]
    if src.shape[0] < 10 or tar.shape[0] < 10:
        return out_np

    src_mean = src.mean(axis=0)
    tar_mean = tar.mean(axis=0)
    src_c = src - src_mean
    tar_c = tar - tar_mean

    cov_src = np.cov(src_c, rowvar=False) + np.eye(3) * 1e-6
    cov_tar = np.cov(tar_c, rowvar=False) + np.eye(3) * 1e-6

    eig_src, E_src = np.linalg.eigh(cov_src)
    eig_tar, E_tar = np.linalg.eigh(cov_tar)

    sqrt_tar = (E_tar @ np.diag(np.sqrt(eig_tar)) @ E_tar.T)
    inv_sqrt_src = (E_src @ np.diag(1.0 / np.sqrt(eig_src)) @ E_src.T)

    A = sqrt_tar @ inv_sqrt_src

    flat = out_np.reshape(-1, 3)
    transformed = (A @ (flat - src_mean).T).T + tar_mean
    transformed = transformed.reshape(out_np.shape)
    return np.clip(transformed, 0.0, 1.0)


def _match_lab_l(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, use_cdf: bool) -> np.ndarray:
    """Match only luminance channel in Lab space while preserving chroma."""
    if cv2 is None:
        return out_np

    out_lab = []
    ref_lab = []
    for idx in range(out_np.shape[0]):
        out_lab.append(cv2.cvtColor(out_np[idx], cv2.COLOR_RGB2LAB))
        ref_lab.append(cv2.cvtColor(ref_np[idx], cv2.COLOR_RGB2LAB))
    out_lab = np.stack(out_lab, axis=0)
    ref_lab = np.stack(ref_lab, axis=0)

    out_l = out_lab[..., 0]
    ref_l = ref_lab[..., 0]
    if use_cdf:
        out_l = _match_histogram_channel(out_l, ref_l, keep, bins=256, value_range=(0.0, 100.0))
    else:
        out_l = _match_mean_std_channel(out_l, ref_l, keep)
    out_lab[..., 0] = np.clip(out_l, 0.0, 100.0)

    out_rgb = []
    for idx in range(out_lab.shape[0]):
        out_rgb.append(cv2.cvtColor(out_lab[idx], cv2.COLOR_LAB2RGB))
    out_rgb = np.stack(out_rgb, axis=0)
    return np.clip(out_rgb, 0.0, 1.0)


def _match_lab_full(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, use_cdf: bool) -> np.ndarray:
    """Match full Lab distribution and convert back to RGB."""
    if cv2 is None:
        return out_np

    out_lab = []
    ref_lab = []
    for idx in range(out_np.shape[0]):
        out_lab.append(cv2.cvtColor(out_np[idx], cv2.COLOR_RGB2LAB))
        ref_lab.append(cv2.cvtColor(ref_np[idx], cv2.COLOR_RGB2LAB))
    out_lab = np.stack(out_lab, axis=0)
    ref_lab = np.stack(ref_lab, axis=0)

    ranges = [(0.0, 100.0), (-127.0, 127.0), (-127.0, 127.0)]
    for ch in range(3):
        if use_cdf:
            out_lab[..., ch] = _match_histogram_channel(
                out_lab[..., ch], ref_lab[..., ch], keep, bins=256, value_range=ranges[ch]
            )
        else:
            out_lab[..., ch] = _match_mean_std_channel(
                out_lab[..., ch], ref_lab[..., ch], keep
            )

    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
    out_lab[..., 1] = np.clip(out_lab[..., 1], -127.0, 127.0)
    out_lab[..., 2] = np.clip(out_lab[..., 2], -127.0, 127.0)

    out_rgb = []
    for idx in range(out_lab.shape[0]):
        out_rgb.append(cv2.cvtColor(out_lab[idx], cv2.COLOR_LAB2RGB))
    out_rgb = np.stack(out_rgb, axis=0)
    return np.clip(out_rgb, 0.0, 1.0)


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0, 1] to Oklab color space. Supports batches (B, H, W, 3) and single images (H, W, 3)."""
    is_batch = rgb.ndim == 4
    if is_batch:
        batch_size = rgb.shape[0]
        h, w = rgb.shape[1:3]
        results = []
        for i in range(batch_size):
            results.append(_rgb_to_oklab(rgb[i]))
        return np.stack(results, axis=0)
    
    # Linear RGB
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4)
    )
    
    # Linear RGB to LMS
    l_mat = np.array([
        [0.3, 0.622, 0.078],
        [0.23, 0.692, 0.078],
        [0.24342268924547819, 0.20476744424496821, 0.55314985651939574]
    ])
    lms = np.dot(linear, l_mat.T)
    
    # LMS to Oklab
    lms_cubic = np.cbrt(lms)
    oklab_mat = np.array([
        [0.210454255534087, 0.791121460798369, -0.0040587710699851425],
        [1.9779984951406756, -2.4285922050660405, 0.4505937099516859],
        [0.029727982640443518, 0.78956734665305050, -0.81917763894369968]
    ])
    oklab = np.dot(lms_cubic, oklab_mat.T)
    
    return oklab


def _oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """Convert Oklab color space to RGB [0, 1]. Supports batches (B, H, W, 3) and single images (H, W, 3)."""
    is_batch = oklab.ndim == 4
    if is_batch:
        batch_size = oklab.shape[0]
        results = []
        for i in range(batch_size):
            results.append(_oklab_to_rgb(oklab[i]))
        return np.stack(results, axis=0)
    
    # Oklab to LMS cubic
    oklab_inv_mat = np.array([
        [1.00246414, 0.39570516, 0.21269317],
        [1.00249668, -0.10571463, -0.06311604],
        [1.00263953, -0.08753328, -1.27385243]
    ])
    lms_cubic = np.dot(oklab, oklab_inv_mat.T)
    
    # LMS cubic to LMS (cube)
    lms = lms_cubic ** 3
    
    # LMS to linear RGB
    lms_inv_mat = np.array([
        [11.03076189, -9.86634701, -0.16419485],
        [-3.2549524, 4.41936727, -0.16419485],
        [-3.64933556, 2.70586743, 1.94086738]
    ])
    linear = np.dot(lms, lms_inv_mat.T)
    
    # Linear RGB to RGB
    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.clip(linear, 0.0, None), 1.0 / 2.4) - 0.055
    )
    
    return rgb


def _match_oklab_l(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, use_cdf: bool) -> np.ndarray:
    """Match only luminance channel in Oklab space while preserving chroma."""
    out_oklab = _rgb_to_oklab(out_np)
    ref_oklab = _rgb_to_oklab(ref_np)
    
    out_l = out_oklab[..., 0]
    ref_l = ref_oklab[..., 0]
    
    if use_cdf:
        out_l = _match_histogram_channel(out_l, ref_l, keep, bins=256, value_range=(0.0, 1.0))
    else:
        out_l = _match_mean_std_channel(out_l, ref_l, keep)
    
    out_oklab[..., 0] = np.clip(out_l, 0.0, 1.0)
    out_rgb = _oklab_to_rgb(out_oklab)
    
    return np.clip(out_rgb, 0.0, 1.0)


def _match_oklab_full(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, use_cdf: bool) -> np.ndarray:
    """Match full Oklab distribution and convert back to RGB."""
    out_oklab = _rgb_to_oklab(out_np)
    ref_oklab = _rgb_to_oklab(ref_np)
    
    ranges = [(0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]
    for ch in range(3):
        if use_cdf:
            out_oklab[..., ch] = _match_histogram_channel(
                out_oklab[..., ch], ref_oklab[..., ch], keep, bins=256, value_range=ranges[ch]
            )
        else:
            out_oklab[..., ch] = _match_mean_std_channel(
                out_oklab[..., ch], ref_oklab[..., ch], keep
            )
    
    out_oklab[..., 0] = np.clip(out_oklab[..., 0], 0.0, 1.0)
    out_oklab[..., 1] = np.clip(out_oklab[..., 1], -0.5, 0.5)
    out_oklab[..., 2] = np.clip(out_oklab[..., 2], -0.5, 0.5)
    
    out_rgb = _oklab_to_rgb(out_oklab)
    return np.clip(out_rgb, 0.0, 1.0)


def _match_tone_curve(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, num_points: int = 5) -> np.ndarray:
    """Match luminance using tone curve (quantile-based points)."""
    if cv2 is None:
        return out_np
    
    # Convert to grayscale for luminance extraction
    out_gray = []
    ref_gray = []
    for idx in range(out_np.shape[0]):
        out_gray.append(cv2.cvtColor(out_np[idx], cv2.COLOR_RGB2GRAY))
        ref_gray.append(cv2.cvtColor(ref_np[idx], cv2.COLOR_RGB2GRAY))
    out_gray = np.stack(out_gray, axis=0)
    ref_gray = np.stack(ref_gray, axis=0)
    
    # Extract quantile points for tone curve
    quantiles = np.linspace(0.05, 0.95, num_points)
    
    # Compute tone mapping for each batch
    out_result = out_np.copy()
    for batch_idx in range(out_np.shape[0]):
        src_vals = out_gray[batch_idx][keep[batch_idx]]
        ref_vals = ref_gray[batch_idx][keep[batch_idx]]
        
        if src_vals.size < 10 or ref_vals.size < 10:
            continue
        
        src_quantiles = np.quantile(src_vals, quantiles)
        ref_quantiles = np.quantile(ref_vals, quantiles)
        
        # Build tone curve mapping
        src_points = np.concatenate([[0.0], src_quantiles, [1.0]])
        ref_points = np.concatenate([[0.0], ref_quantiles, [1.0]])
        
        # Create luminance-based scale map per pixel
        lum = out_gray[batch_idx]
        tone_scale = np.interp(lum, src_points, ref_points / (src_points + 1e-6))
        tone_scale = np.clip(tone_scale, 0.5, 2.0)
        
        # Apply tone scale to RGB channels
        for c in range(3):
            out_result[batch_idx, :, :, c] = out_np[batch_idx, :, :, c] * tone_scale
    
    return np.clip(out_result, 0.0, 1.0)


def _match_adain(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Adaptive Instance Normalization: match mean and variance of ref to out per-channel."""
    # Process per batch
    out_result = out_np.copy()
    for batch_idx in range(out_np.shape[0]):
        mask_b = keep[batch_idx]
        for c in range(3):
            src_ch = out_np[batch_idx, :, :, c]
            ref_ch = ref_np[batch_idx, :, :, c]
            
            src_vals = src_ch[mask_b]
            ref_vals = ref_ch[mask_b]
            
            if src_vals.size < 10 or ref_vals.size < 10:
                continue
            
            ref_mean = float(ref_vals.mean())
            ref_std = float(ref_vals.std())
            src_mean = float(src_vals.mean())
            src_std = float(src_vals.std())
            
            src_std = max(src_std, 1e-6)
            
            # Normalize then scale: (x - mean) / std * ref_std + ref_mean
            normalized = (src_ch - src_mean) / src_std * ref_std + ref_mean
            out_result[batch_idx, :, :, c] = np.clip(normalized, 0.0, 1.0)
    
    return out_result


def _match_optimal_transport(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """
    Оптимальный транспорт (Wasserstein distance) для подгонки цвета.
    
    Для каждого RGB канала отдельно решает 1D задачу оптимального транспорта:
    - Сортирует пиксели источника и эталона
    - Матчит их монотонно (в отсортированном порядке)
    - Восстанавливает исходный порядок
    
    Это более математически обосновано чем tone_curve с квантилями.
    """
    out_result = out_np.copy()
    
    for batch_idx in range(out_np.shape[0]):
        mask_b = keep[batch_idx]
        
        for c in range(3):
            src_ch = out_np[batch_idx, :, :, c]
            ref_ch = ref_np[batch_idx, :, :, c]
            
            # Получить пиксели из маски
            src_vals = src_ch[mask_b]
            ref_vals = ref_ch[mask_b]
            
            if src_vals.size < 10 or ref_vals.size < 10:
                continue
            
            # Отсортировать оба распределения
            src_indices = np.argsort(src_vals)
            ref_indices = np.argsort(ref_vals)
            
            src_sorted = src_vals[src_indices]
            ref_sorted = ref_vals[ref_indices]
            
            # Интерполировать эталонные значения для каждого отсортированного источника
            # (монотонное отображение - основа 1D OT)
            ref_as_cdf = np.interp(
                src_sorted,
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted
            )
            
            # Восстановить исходный порядок пикселей
            out_ch = np.empty_like(src_ch)
            out_ch[mask_b] = ref_as_cdf[np.argsort(src_indices)]
            
            out_result[batch_idx, :, :, c] = np.clip(out_ch, 0.0, 1.0)
    
    return out_result


def apply_color_match(
    output_images: torch.Tensor,
    reference_images: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    mask_white_is_keep: bool = False,
) -> torch.Tensor:
    """Apply selected color-match mode and return corrected image tensor."""
    if mode == "none":
        return output_images

    mask = normalize_mask(mask)
    mask = ensure_mask_batch(mask, output_images.size(dim=0))
    mask = resize_mask_to_output(mask, output_images.shape[1], output_images.shape[2])
    keep_t = mask > 0.5 if mask_white_is_keep else mask < 0.5
    keep = keep_t.detach().cpu().numpy()
    if keep.sum() < 10:
        return output_images

    if reference_images.shape[1:3] != output_images.shape[1:3]:
        reference_images = resize_images_to_size(
            reference_images, output_images.shape[1], output_images.shape[2]
        )

    out_np = output_images.detach().cpu().numpy().astype(np.float32)
    ref_np = reference_images.detach().cpu().numpy().astype(np.float32)

    if mode == "mean_std":
        out_np = _match_mean_std_rgb(out_np, ref_np, keep)
    elif mode == "linear":
        out_np = _match_linear_rgb(out_np, ref_np, keep)
    elif mode == "hist":
        out_np = _match_hist_rgb(out_np, ref_np, keep)
    elif mode == "pca_cov":
        out_np = _pca_cov_transfer(out_np, ref_np, keep)
    elif mode == "tone_curve":
        out_np = _match_tone_curve(out_np, ref_np, keep)
    elif mode == "adain":
        out_np = _match_adain(out_np, ref_np, keep)
    elif mode == "optimal_transport":
        out_np = _match_optimal_transport(out_np, ref_np, keep)
    elif mode == "lab_l":
        out_np = _match_lab_l(out_np, ref_np, keep, use_cdf=False)
    elif mode == "lab_l_cdf":
        out_np = _match_lab_l(out_np, ref_np, keep, use_cdf=True)
    elif mode == "lab_full":
        out_np = _match_lab_full(out_np, ref_np, keep, use_cdf=False)
    elif mode == "lab_cdf":
        out_np = _match_lab_full(out_np, ref_np, keep, use_cdf=True)
    elif mode == "oklab_l":
        out_np = _match_oklab_l(out_np, ref_np, keep, use_cdf=False)
    elif mode == "oklab_cdf":
        out_np = _match_oklab_full(out_np, ref_np, keep, use_cdf=True)
    else:
        return output_images

    out_np = np.clip(out_np, 0.0, 1.0)
    return torch.from_numpy(out_np).to(device=output_images.device, dtype=output_images.dtype)
