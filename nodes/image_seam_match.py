"""
Module: nodes/image_seam_match.py
Author: AlexZ1967
Last updated: 2026-02-18

Description:
    Seam-focused color matching node for minimizing visible cuts.

Purpose:
    Optimizes a compact color transform (3x3 + bias) against a reference frame
    to reduce frame-to-frame seam visibility.
"""

from __future__ import annotations

import json
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        """Fallback tqdm wrapper when tqdm is unavailable."""
        return iterable if iterable is not None else []

from ..utils import color_match_utils

_SSIM_WINDOW_CACHE = {}


def _pad_batch_last(batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Compatibility wrapper around shared batch-padding helper."""
    return color_match_utils.pad_batch_last(batch, batch_size)


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


def _ssim_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5):
    """Create a Gaussian convolution window for SSIM computation."""
    key = (channels, device.type, str(dtype), size, sigma)
    if key in _SSIM_WINDOW_CACHE:
        return _SSIM_WINDOW_CACHE[key]
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel = kernel_2d.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    _SSIM_WINDOW_CACHE[key] = kernel
    return kernel


def _ssim_similarity_tensor(a_bchw: torch.Tensor, b_bchw: torch.Tensor) -> torch.Tensor:
    """Compute SSIM similarity tensor for BCHW RGB inputs."""
    channels = a_bchw.shape[1]
    window = _ssim_window(channels, a_bchw.device, a_bchw.dtype)
    mu1 = F.conv2d(a_bchw, window, padding=window.shape[-1] // 2, groups=channels)
    mu2 = F.conv2d(b_bchw, window, padding=window.shape[-1] // 2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(a_bchw * a_bchw, window, padding=window.shape[-1] // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(b_bchw * b_bchw, window, padding=window.shape[-1] // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(a_bchw * b_bchw, window, padding=window.shape[-1] // 2, groups=channels) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().clamp(0.0, 1.0)


def _gradient_consistency_loss(a_hwc: torch.Tensor, b_hwc: torch.Tensor) -> torch.Tensor:
    """Compute simple luma-gradient consistency loss."""
    a_l = 0.299 * a_hwc[..., 0] + 0.587 * a_hwc[..., 1] + 0.114 * a_hwc[..., 2]
    b_l = 0.299 * b_hwc[..., 0] + 0.587 * b_hwc[..., 1] + 0.114 * b_hwc[..., 2]
    gx_a = a_l[:, 1:] - a_l[:, :-1]
    gx_b = b_l[:, 1:] - b_l[:, :-1]
    gy_a = a_l[1:, :] - a_l[:-1, :]
    gy_b = b_l[1:, :] - b_l[:-1, :]
    return (gx_a - gx_b).abs().mean() + (gy_a - gy_b).abs().mean()


def _robust_charbonnier(err: torch.Tensor, delta: float) -> torch.Tensor:
    """Robust error that is less sensitive to outliers than pure MSE."""
    d = max(1e-6, float(delta))
    return torch.sqrt(err * err + d * d) - d


def _downscale_hwc_long_side(img_hwc: torch.Tensor, mode: str) -> tuple[torch.Tensor, dict]:
    """Downscale HWC tensor by long side mode without upscaling."""
    target_map = {"as_is": None, "1080p": 1080, "720p": 720, "480p": 480}
    target = target_map.get(mode, None)
    h, w = img_hwc.shape[:2]
    if target is None:
        return img_hwc, {"mode": "as_is", "optimized_size": [int(h), int(w)]}
    long_side = max(h, w)
    if long_side <= target:
        return img_hwc, {"mode": mode, "optimized_size": [int(h), int(w)]}
    scale = float(target) / float(long_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    out = F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)
    return out, {"mode": mode, "optimized_size": [int(new_h), int(new_w)]}


def _apply_transform(img_hwc: torch.Tensor, A: torch.Tensor, b: torch.Tensor, color_space: str) -> torch.Tensor:
    """Apply affine color transform in RGB or Oklab space."""
    if color_space == "oklab":
        lab = _rgb_to_oklab_torch(img_hwc)
        out_lab = torch.matmul(lab, A.t()) + b
        return torch.clamp(_oklab_to_rgb_torch(out_lab), 0.0, 1.0)
    out = torch.matmul(img_hwc, A.t()) + b
    return torch.clamp(out, 0.0, 1.0)


def _fit_linear_init(src_hwc: torch.Tensor, ref_hwc: torch.Tensor, color_space: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get linear per-channel initialization as diagonal matrix+bias."""
    if color_space == "oklab":
        src = _rgb_to_oklab_torch(src_hwc).unsqueeze(0)
        ref = _rgb_to_oklab_torch(ref_hwc).unsqueeze(0)
    else:
        src = src_hwc.unsqueeze(0)
        ref = ref_hwc.unsqueeze(0)
    scale, offset = color_match_utils.linear_fit_torch_batch(src, ref, None)
    A = torch.diag(scale[0].float())
    b = offset[0].float()
    return A, b


def _optimize_seam_transform(
    img_opt: torch.Tensor,
    ref_opt: torch.Tensor,
    color_space: str,
    steps: int,
    lr: float,
    w_mse: float,
    w_ssim: float,
    w_grad: float,
    reg_weight: float,
    robust_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Optimize compact color transform for seam matching."""
    inf_ctx = torch.inference_mode(False) if torch.is_inference_mode_enabled() else nullcontext()
    with inf_ctx:
        # ComfyUI may call nodes under global inference mode; convert to normal
        # tensors before using autograd-tracked operations.
        img_work = img_opt.detach().clone()
        ref_work = ref_opt.detach().clone()
        device = img_work.device
        dtype = img_work.dtype
        A0, b0 = _fit_linear_init(img_work, ref_work, color_space)
        A = A0.to(device=device, dtype=torch.float32).clone().requires_grad_(True)
        b = b0.to(device=device, dtype=torch.float32).clone().requires_grad_(True)
        opt = torch.optim.Adam([A, b], lr=float(lr))
        eye = torch.eye(3, device=device, dtype=torch.float32)
        steps_int = max(1, int(steps))
        final_loss = None

        with torch.enable_grad():
            for _ in range(steps_int):
                opt.zero_grad(set_to_none=True)
                pred = _apply_transform(img_work, A.to(dtype=dtype), b.to(dtype=dtype), color_space)
                robust = _robust_charbonnier(pred - ref_work, robust_delta).mean()
                pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
                ref_bchw = ref_work.permute(2, 0, 1).unsqueeze(0)
                ssim_loss = 1.0 - _ssim_similarity_tensor(pred_bchw, ref_bchw)
                grad_loss = _gradient_consistency_loss(pred, ref_work)
                reg_loss = ((A - eye) ** 2).mean() + (b ** 2).mean()
                loss = (
                    float(w_mse) * robust
                    + float(w_ssim) * ssim_loss
                    + float(w_grad) * grad_loss
                    + float(reg_weight) * reg_loss
                )
                loss.backward()
                opt.step()
                final_loss = float(loss.detach().item())

    if final_loss is None:
        final_loss = 0.0
    return A.detach().to(dtype=dtype), b.detach().to(dtype=dtype), final_loss


def _metric_mse(a_hwc: torch.Tensor, b_hwc: torch.Tensor) -> float:
    """Compute MSE metric."""
    return float(torch.mean((a_hwc - b_hwc) ** 2).item())


def _metric_ssim(a_hwc: torch.Tensor, b_hwc: torch.Tensor) -> float:
    """Compute SSIM metric."""
    aa = a_hwc.permute(2, 0, 1).unsqueeze(0)
    bb = b_hwc.permute(2, 0, 1).unsqueeze(0)
    return float(_ssim_similarity_tensor(aa, bb).item())


class ImageSeamMatchToReference:
    """ComfyUI node that minimizes seam-visible color differences to a reference."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Эталонное изображение (куда подгоняем)."}),
                "image": ("IMAGE", {"tooltip": "Изображение для подгонки под reference."}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Сила применения итоговой коррекции (0..1)."},
                ),
            },
            "optional": {
                "preserve_alpha": ("BOOLEAN", {"default": True, "tooltip": "Если вход RGBA, сохранить альфа-канал исходника."}),
                "color_space": (
                    ["rgb", "oklab"],
                    {"default": "oklab", "tooltip": "Пространство оптимизации: oklab обычно лучше для перцептивного seam-match."},
                ),
                "downscale_long_side": (
                    ["as_is", "1080p", "720p", "480p"],
                    {"default": "720p", "tooltip": "Размер для оптимизации: as_is, 1080p, 720p, 480p (без апскейла)."},
                ),
                "steps": ("INT", {"default": 40, "min": 1, "max": 200, "step": 1, "tooltip": "Количество шагов оптимизации."}),
                "lr": ("FLOAT", {"default": 0.05, "min": 0.0005, "max": 0.5, "step": 0.0005, "tooltip": "Скорость обучения оптимизатора."}),
                "w_mse": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Вес robust MSE-терма."}),
                "w_ssim": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Вес SSIM-терма (структурное сходство)."}),
                "w_grad": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Вес градиентного терма (снижение заметности шва)."}),
                "reg_weight": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0005, "tooltip": "Регуляризация: удерживает transform ближе к identity."}),
                "robust_delta": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.2, "step": 0.0005, "tooltip": "Порог robust-loss (больше = мягче к выбросам)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "seam_json")
    FUNCTION = "match"
    CATEGORY = "image/color"

    def match(
        self,
        reference,
        image,
        strength=1.0,
        preserve_alpha=True,
        color_space="oklab",
        downscale_long_side="720p",
        steps=40,
        lr=0.05,
        w_mse=1.0,
        w_ssim=0.2,
        w_grad=0.1,
        reg_weight=0.001,
        robust_delta=0.01,
    ):
        """Execute seam-matching transform and return processed outputs for ComfyUI."""
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

        out_list = []
        json_list = []
        iterator = tqdm(range(batch_size), desc="SeamMatch", unit="img")
        for idx in iterator:
            ref_t = reference_rgb[idx]
            img_t = image_rgb[idx]
            img_opt, ds_info = _downscale_hwc_long_side(img_t, downscale_long_side)
            ref_opt, _ = _downscale_hwc_long_side(ref_t, downscale_long_side)

            A, b, final_loss = _optimize_seam_transform(
                img_opt=img_opt,
                ref_opt=ref_opt,
                color_space=str(color_space),
                steps=int(steps),
                lr=float(lr),
                w_mse=float(w_mse),
                w_ssim=float(w_ssim),
                w_grad=float(w_grad),
                reg_weight=float(reg_weight),
                robust_delta=float(robust_delta),
            )

            corrected = _apply_transform(img_t, A, b, str(color_space))
            if float(strength) < 1.0:
                s = float(max(0.0, min(1.0, strength)))
                corrected = img_t * (1.0 - s) + corrected * s
            corrected = torch.clamp(corrected, 0.0, 1.0)

            mse_before = _metric_mse(img_t, ref_t)
            mse_after = _metric_mse(corrected, ref_t)
            ssim_before = _metric_ssim(img_t, ref_t)
            ssim_after = _metric_ssim(corrected, ref_t)

            out_t = corrected
            if alpha_batch is not None and preserve_alpha:
                out_t = torch.cat([out_t, alpha_batch[idx]], dim=-1)
            out_list.append(out_t.cpu())

            payload = {
                "status": "ok",
                "mode": f"seam_match:{color_space}",
                "optimization": {
                    "downscale_long_side": str(downscale_long_side),
                    "optimized_size": ds_info.get("optimized_size"),
                    "steps": int(steps),
                    "lr": float(lr),
                    "loss_final": round(float(final_loss), 8),
                    "weights": {
                        "w_mse": float(w_mse),
                        "w_ssim": float(w_ssim),
                        "w_grad": float(w_grad),
                        "reg_weight": float(reg_weight),
                        "robust_delta": float(robust_delta),
                    },
                },
                "transform": {
                    "matrix": [[round(float(v), 6) for v in row] for row in A.detach().cpu().tolist()],
                    "bias": [round(float(v), 6) for v in b.detach().cpu().tolist()],
                },
                "quality": {
                    "before": {"mse": round(mse_before, 8), "ssim": round(ssim_before, 8)},
                    "after": {"mse": round(mse_after, 8), "ssim": round(ssim_after, 8)},
                    "improvement_pct": {
                        "mse": round((mse_before - mse_after) / max(mse_before, 1e-8) * 100.0, 3),
                        "ssim": round((ssim_after - ssim_before) / max(ssim_before, 1e-8) * 100.0, 3),
                    },
                },
            }
            json_list.append(json.dumps(payload, ensure_ascii=True))

        return (torch.stack(out_list, dim=0), json_list)
