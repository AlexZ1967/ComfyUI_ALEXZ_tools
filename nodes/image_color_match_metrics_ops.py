"""Quality and perceptual metrics for color matching."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from ..utils import color_match_utils
from ..utils.interrupt import check_interrupt
from .image_color_match_color_ops import _rgb_to_lab_torch

_LOGGER = logging.getLogger("ImageColorMatchToReference")
_SSIM_WINDOW_CACHE = {}
_LPIPS_CACHE = {}
_VGG_CACHE = {}

def _vgg19_features(device: torch.device):
    """Load and cache VGG19 feature extractor used by perceptual matching."""
    key = ("vgg19_f12", device.type, device.index if device.type == "cuda" else -1)
    if key in _VGG_CACHE:
        return _VGG_CACHE[key]
    try:
        from torchvision.models import VGG19_Weights, vgg19
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "torchvision is required for preset=perceptual. "
            "Use ComfyUI default environment or install torchvision."
        ) from exc
    _LOGGER.info("Loading VGG19 (perceptual_vgg)...")
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:12].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    _VGG_CACHE[key] = vgg
    return vgg


def _perceptual_vgg(img: torch.Tensor, ref: torch.Tensor, steps: int, lr: float):
    """Optimize image parameters against VGG perceptual features for high-quality matching."""
    device = img.device
    inf_ctx = torch.inference_mode(False) if torch.is_inference_mode_enabled() else nullcontext()

    with inf_ctx:
        vgg = _vgg19_features(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

        def prep(x):
            """Normalize and resize an image tensor for VGG feature extraction."""
            x = x.float().permute(2, 0, 1).unsqueeze(0)
            return (x - mean) / std

        with torch.no_grad():
            feat_ref = vgg(prep(ref.detach().clone()))

        W = torch.eye(3, device=device, dtype=img.dtype).requires_grad_(True)
        b = torch.zeros(3, device=device, dtype=img.dtype).requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=lr)

        steps_int = max(1, int(steps))
        img_work = img.detach().clone()
        with torch.enable_grad():
            iterator = tqdm(range(steps_int), desc="perceptual_vgg", leave=False)
            for _ in iterator:
                check_interrupt()
                opt.zero_grad(set_to_none=True)
                x = torch.clamp(torch.einsum("hwc,dc->hwd", img_work, W) + b, 0.0, 1.0)
                feat_x = vgg(prep(x))
                loss = torch.mean((feat_x - feat_ref) ** 2)
                loss.backward()
                opt.step()

    corrected = torch.clamp(torch.einsum("hwc,dc->hwd", img, W.detach()) + b.detach(), 0.0, 1.0)
    params = {
        "matrix": W.detach().cpu().tolist(),
        "bias": b.detach().cpu().tolist(),
        "loss_final": float(loss.detach().cpu()),
        "steps": int(steps),
        "lr": float(lr),
    }
    return corrected, params


def _perceptual_vgg_fast(img: torch.Tensor, ref: torch.Tensor, steps: int, lr: float, max_side: int = 256):
    """Run a lightweight perceptual optimization pass tuned for speed."""
    h, w, _ = img.shape
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        img_small = F.interpolate(
            img.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        ref_small = F.interpolate(
            ref.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
    else:
        img_small, ref_small = img, ref

    fast_steps = max(1, min(int(steps), 5))
    _, params = _perceptual_vgg(img_small, ref_small, fast_steps, lr)

    W = torch.tensor(params["matrix"], device=img.device, dtype=img.dtype)
    b = torch.tensor(params["bias"], device=img.device, dtype=img.dtype)
    corrected_full = torch.clamp(torch.einsum("hwc,dc->hwd", img, W) + b, 0.0, 1.0)
    params["mode"] = "perceptual_vgg_fast"
    params["used_scale"] = scale
    params["used_steps"] = fast_steps
    return corrected_full, params


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


def _ssim_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute SSIM similarity score for two image tensors."""
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    channels = a.shape[1]
    window = _ssim_window(channels, a.device, a.dtype)
    mu1 = F.conv2d(a, window, padding=window.shape[-1] // 2, groups=channels)
    mu2 = F.conv2d(b, window, padding=window.shape[-1] // 2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(a * a, window, padding=window.shape[-1] // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=window.shape[-1] // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=window.shape[-1] // 2, groups=channels) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().clamp(0.0, 1.0).item())


def _downscale_max_side(img: torch.Tensor, max_side: int = 256) -> torch.Tensor:
    """Downscale an image tensor so its longest side does not exceed the given limit."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = float(max_side) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    out = F.interpolate(
        img.permute(2, 0, 1).unsqueeze(0),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def _lpips_model(device: torch.device):
    """Load and cache LPIPS model weights for perceptual scoring."""
    key = ("alex", device.type)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    model = lpips.LPIPS(net="alex").to(device).eval()
    _LPIPS_CACHE[key] = model
    return model


def _lpips_alex_distance(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """Compute LPIPS AlexNet perceptual distance between image tensors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _lpips_model(device)
    if model is None:
        return None
    aa = _downscale_max_side(a, 256).permute(2, 0, 1).unsqueeze(0).to(device)
    bb = _downscale_max_side(b, 256).permute(2, 0, 1).unsqueeze(0).to(device)
    aa = aa * 2.0 - 1.0
    bb = bb * 2.0 - 1.0
    with torch.inference_mode():
        dist = model(aa, bb)
    return float(dist.item())


def _delta_e76_mean(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """Compute mean DeltaE76 color distance in Lab space (torch-first)."""
    try:
        a_lab = _rgb_to_lab_torch(torch.clamp(a, 0.0, 1.0).float())
        b_lab = _rgb_to_lab_torch(torch.clamp(b, 0.0, 1.0).float())
        delta = torch.linalg.norm(a_lab - b_lab, dim=-1)
        return float(delta.mean().item())
    except Exception:
        if color_match_utils.cv2 is None:
            return None
        a_np = np.clip(a.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
        b_np = np.clip(b.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
        a_lab = color_match_utils.cv2.cvtColor(a_np, color_match_utils.cv2.COLOR_RGB2LAB)
        b_lab = color_match_utils.cv2.cvtColor(b_np, color_match_utils.cv2.COLOR_RGB2LAB)
        delta = np.linalg.norm(a_lab - b_lab, axis=2)
        return float(np.mean(delta))


def _quality_metrics(
    img: torch.Tensor,
    ref: torch.Tensor,
    lpips_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Optional[float]]] = None,
):
    """Compute quality metrics used for before/after color-match reporting."""
    mse = float(torch.mean((img - ref) ** 2).item())
    ssim = _ssim_similarity(img, ref)
    de = _delta_e76_mean(img, ref)
    lpips_a = (lpips_fn or _lpips_alex_distance)(img, ref)
    return {
        "mse": mse,
        "ssim": ssim,
        "delta_e76": de,
        "lpips_alex": lpips_a,
    }


def _quality_metrics_fast(img: torch.Tensor, ref: torch.Tensor):
    """Compute fast quality metrics (MSE + SSIM only)."""
    mse = float(torch.mean((img - ref) ** 2).item())
    ssim = _ssim_similarity(img, ref)
    return {
        "mse": mse,
        "ssim": ssim,
        "delta_e76": None,
        "lpips_alex": None,
    }


def _auto_optimal_candidate_metrics(
    candidate: torch.Tensor,
    ref: torch.Tensor,
    strategy: str,
    lpips_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Optional[float]]] = None,
) -> dict:
    """Compute candidate metrics used to pick auto_optimal method."""
    mse = float(torch.mean((candidate - ref) ** 2).item())
    ssim = None
    lpips_a = None
    if strategy in ("mse_ssim", "mse_ssim_lpips"):
        cand_s = _downscale_max_side(candidate, 256)
        ref_s = _downscale_max_side(ref, 256)
        ssim = _ssim_similarity(cand_s, ref_s)
    if strategy == "mse_ssim_lpips":
        lpips_a = (lpips_fn or _lpips_alex_distance)(candidate, ref)
    return {
        "mse": mse,
        "ssim": ssim,
        "lpips_alex": lpips_a,
    }


def _auto_optimal_score(metrics: dict, strategy: str) -> float:
    """Compose scalar auto_optimal score from candidate metrics."""
    score = float(metrics["mse"])
    if strategy in ("mse_ssim", "mse_ssim_lpips") and metrics.get("ssim") is not None:
        score += 0.05 * (1.0 - float(metrics["ssim"]))
    if strategy == "mse_ssim_lpips" and metrics.get("lpips_alex") is not None:
        score += 0.05 * float(metrics["lpips_alex"])
    return score


def _empty_quality_metrics() -> dict:
    """Return empty quality metrics payload when metric evaluation is disabled."""
    return {
        "mse": None,
        "ssim": None,
        "delta_e76": None,
        "lpips_alex": None,
    }


def _skin_tone_mask_soft(img: torch.Tensor) -> torch.Tensor:
    """Estimate a soft skin-likelihood mask from RGB image in [0,1]."""
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y) + 0.5
    cr = 0.713 * (r - y) + 0.5
    cb_w = (1.0 - torch.abs(cb - 0.40) / 0.15).clamp(0.0, 1.0)
    cr_w = (1.0 - torch.abs(cr - 0.60) / 0.18).clamp(0.0, 1.0)
    sat = (img.max(dim=-1).values - img.min(dim=-1).values).clamp(0.0, 1.0)
    sat_w = (sat / 0.45).clamp(0.0, 1.0)
    lum_w = ((y - 0.08) / 0.62).clamp(0.0, 1.0)
    mask = cb_w * cr_w * sat_w * lum_w
    return mask.clamp(0.0, 1.0)


def _apply_skin_tone_protection(
    corrected: torch.Tensor, original: torch.Tensor, strength: float
) -> tuple[torch.Tensor, float]:
    """Blend corrected image back toward original in skin-tone areas."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return corrected, 0.0
    skin = _skin_tone_mask_soft(original)
    w = (skin * s).unsqueeze(-1)
    out = corrected * (1.0 - w) + original * w
    return torch.clamp(out, 0.0, 1.0), float(skin.mean().item())


def _improvement_pct(before: dict, after: dict) -> dict:
    """Convert metric pairs into percentage-improvement values."""
    res = {}
    for k in ("mse", "delta_e76", "lpips_alex"):
        bv = before.get(k)
        av = after.get(k)
        if bv is None or av is None:
            res[k] = None
        else:
            denom = max(float(bv), 1e-6)
            res[k] = round((float(bv) - float(av)) / denom * 100.0, 3)
    bv = before.get("ssim")
    av = after.get("ssim")
    if bv is None or av is None:
        res["ssim"] = None
    else:
        denom = max(float(bv), 1e-6)
        res["ssim"] = round((float(av) - float(bv)) / denom * 100.0, 3)
    return res
