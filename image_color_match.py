import json
import logging
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from . import color_match_utils
from .color_match_utils import normalize_mask, resize_mask_to_output
from .utils import select_batch_item

_LOGGER = logging.getLogger("ImageColorMatchToReference")
_EPS = 1e-6
_SSIM_WINDOW_CACHE = {}
_LPIPS_CACHE = {}


def _resize_image(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    out = F.interpolate(
        img.permute(2, 0, 1).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def _linear_fit_torch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is not None:
        m = mask > 0.5
        img_sel = img[m]
        ref_sel = ref[m]
    else:
        img_sel = img.reshape(-1, 3)
        ref_sel = ref.reshape(-1, 3)
    mean_img = img_sel.mean(dim=0)
    mean_ref = ref_sel.mean(dim=0)
    var_img = ((img_sel - mean_img) ** 2).mean(dim=0)
    cov = ((img_sel - mean_img) * (ref_sel - mean_ref)).mean(dim=0)
    scale = torch.where(var_img > _EPS, cov / torch.clamp(var_img, min=_EPS), torch.ones_like(var_img))
    offset = mean_ref - scale * mean_img
    return scale, offset


def _mean_std_match(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is not None:
        m = mask > 0.5
        img_sel = img[m]
        ref_sel = ref[m]
    else:
        img_sel = img.reshape(-1, 3)
        ref_sel = ref.reshape(-1, 3)
    mean_img = img_sel.mean(dim=0)
    mean_ref = ref_sel.mean(dim=0)
    std_img = torch.clamp(img_sel.std(dim=0), min=_EPS)
    std_ref = ref_sel.std(dim=0)
    out = (img - mean_img) * (std_ref / std_img) + mean_ref
    return torch.clamp(out, 0.0, 1.0)


def _linear_match(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]):
    scale, offset = _linear_fit_torch(img, ref, mask)
    return torch.clamp(img * scale + offset, 0.0, 1.0)


def _lab_match_torch(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor], mode: str):
    mask_t = torch.zeros((img.shape[0], img.shape[1]), dtype=img.dtype, device=img.device)
    if mask is not None:
        mask_t = normalize_mask(mask)
    out = color_match_utils.apply_color_match(
        img.unsqueeze(0),
        ref.unsqueeze(0),
        mask_t,
        mode,
    )
    return out[0]


def _perceptual_vgg(img: torch.Tensor, ref: torch.Tensor, steps: int, lr: float):
    try:
        from torchvision.models import VGG19_Weights, vgg19
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "torchvision is required for preset=perceptual. "
            "Use ComfyUI default environment or install torchvision."
        ) from exc

    device = img.device
    _LOGGER.info("Loading VGG19 (perceptual_vgg)...")
    inf_ctx = torch.inference_mode(False) if torch.is_inference_mode_enabled() else nullcontext()

    with inf_ctx:
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:12].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        def prep(x):
            x = x.permute(2, 0, 1).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
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
    if color_match_utils.cv2 is None:
        return None
    a_np = np.clip(a.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    b_np = np.clip(b.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    a_lab = color_match_utils.cv2.cvtColor(a_np, color_match_utils.cv2.COLOR_RGB2LAB)
    b_lab = color_match_utils.cv2.cvtColor(b_np, color_match_utils.cv2.COLOR_RGB2LAB)
    delta = np.linalg.norm(a_lab - b_lab, axis=2)
    return float(np.mean(delta))


def _quality_metrics(img: torch.Tensor, ref: torch.Tensor):
    mse = float(torch.mean((img - ref) ** 2).item())
    ssim = _ssim_similarity(img, ref)
    de = _delta_e76_mean(img, ref)
    lpips_a = _lpips_alex_distance(img, ref)
    return {
        "mse": mse,
        "ssim": ssim,
        "delta_e76": de,
        "lpips_alex": lpips_a,
    }


def _improvement_pct(before: dict, after: dict) -> dict:
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


class ImageColorMatchToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Базовое изображение (образец)."}),
                "image": ("IMAGE", {"tooltip": "Изображение, которое нужно подогнать по цвету."}),
                "preset": (
                    ["fast", "balanced", "quality", "perceptual"],
                    {
                        "default": "balanced",
                        "tooltip": "Пресет: fast=самый быстрый, balanced=быстрый/стабильный, quality=медленнее/точнее, perceptual=самый медленный.",
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
        strength=1.0,
    ):
        batch_size = max(reference.shape[0], image.shape[0])
        matched_list = []
        json_list = []

        for idx in range(batch_size):
            ref_t = select_batch_item(reference, idx)
            img_t = select_batch_item(image, idx)
            ref_h, ref_w = ref_t.shape[0], ref_t.shape[1]

            alpha_channel = None
            if ref_t.shape[2] > 3:
                alpha_channel = ref_t[..., 3:4]
                ref_t = ref_t[..., :3]
            if img_t.shape[2] > 3:
                alpha_channel = img_t[..., 3:4] if alpha_channel is None else alpha_channel
                img_t = img_t[..., :3]

            if img_t.shape[0] != ref_h or img_t.shape[1] != ref_w:
                img_t = _resize_image(img_t, ref_h, ref_w)

            mm_t = select_batch_item(match_mask, idx) if match_mask is not None else None
            am_t = select_batch_item(apply_mask, idx) if apply_mask is not None else None
            if mm_t is not None:
                mm_t = resize_mask_to_output(normalize_mask(mm_t), ref_h, ref_w)
            if am_t is not None:
                am_t = resize_mask_to_output(normalize_mask(am_t), ref_h, ref_w)

            deep_params = None
            if preset == "fast":
                corrected_t = _mean_std_match(img_t, ref_t, mm_t)
                mode = "mean_std"
            elif preset == "balanced":
                corrected_t = _linear_match(img_t, ref_t, mm_t)
                mode = "linear"
            elif preset == "quality":
                corrected_t = _lab_match_torch(img_t, ref_t, mm_t, "lab_cdf")
                mode = "lab_cdf"
            elif preset == "perceptual":
                corrected_t, deep_params = _perceptual_vgg_fast(img_t, ref_t, 5, 0.05)
                mode = "perceptual_vgg_fast"
            else:
                corrected_t = img_t
                mode = "none"

            if strength < 1.0:
                corrected_t = img_t * (1.0 - strength) + corrected_t * strength

            scale_t, offset_t = _linear_fit_torch(img_t, ref_t, mm_t)
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

            corrected_t = torch.clamp(corrected_t, 0.0, 1.0)
            metrics_before = _quality_metrics(img_t, ref_t)
            metrics_after = _quality_metrics(corrected_t, ref_t)
            improvement = _improvement_pct(metrics_before, metrics_after)
            matched_t = corrected_t
            if alpha_channel is not None and preserve_alpha:
                matched_t = torch.cat([matched_t, alpha_channel], dim=-1)

            stats = {
                "ref_mean": [round(float(x), 4) for x in ref_t.reshape(-1, 3).mean(dim=0)],
                "img_mean": [round(float(x), 4) for x in img_t.reshape(-1, 3).mean(dim=0)],
                "ref_std": [round(float(x), 4) for x in ref_t.reshape(-1, 3).std(dim=0)],
                "img_std": [round(float(x), 4) for x in img_t.reshape(-1, 3).std(dim=0)],
                "mask_used": mm_t is not None,
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

            json_list.append(json.dumps(payload, ensure_ascii=True))
            matched_list.append(matched_t.cpu())

        return (
            torch.stack(matched_list, dim=0),
            json_list,
        )


_LOGGER.warning("Loaded ImageColorMatchToReference. NODE_CLASS_MAPPINGS=%s", ["ImageColorMatchToReference"])
