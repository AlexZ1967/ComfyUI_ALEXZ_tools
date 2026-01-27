import json
import logging
import math
from typing import Optional

import os
import urllib.request
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from .color_match_utils import normalize_mask, resize_mask_to_output
from .utils import select_batch_item

_LOGGER = logging.getLogger("ImageColorMatchToReference")
_EPS = 1e-6
_WAVEFORM_HEIGHT = 256


def _resize_image(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    bchw = img.permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(bchw, size=(h, w), mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0)


def _torch_percentiles(values: torch.Tensor, mask: Optional[torch.Tensor], p: float):
    if mask is not None:
        values = values[mask > 0.5]
    if values.numel() < 10:
        return None
    return (
        torch.quantile(values, p / 100.0),
        torch.quantile(values, 1.0 - p / 100.0),
        torch.quantile(values, 0.5),
    )


def _apply_levels(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor], p: float):
    params = {"r": None, "g": None, "b": None}
    out = img.clone()
    for ch, key in enumerate(("r", "g", "b")):
        b_in = _torch_percentiles(img[..., ch], mask, p)
        b_ref = _torch_percentiles(ref[..., ch], mask, p)
        if b_in is None or b_ref is None:
            return img, params, "error: not enough pixels to estimate levels"
        black_in, white_in, med_in = b_in
        black_out, white_out, med_out = b_ref
        denom = torch.clamp(white_in - black_in, min=_EPS)
        norm = torch.clamp((img[..., ch] - black_in) / denom, 0.0, 1.0)
        norm_med_in = torch.clamp((med_in - black_in) / denom, _EPS, 1.0 - _EPS)
        norm_med_out = torch.clamp((med_out - black_out) / torch.clamp(white_out - black_out, min=_EPS), _EPS, 1.0 - _EPS)
        gamma = torch.log(norm_med_out) / torch.log(norm_med_in) if 0 < norm_med_in < 1 else torch.tensor(1.0, device=img.device)
        corr = torch.pow(norm, gamma)
        corr = corr * (white_out - black_out) + black_out
        out[..., ch] = corr
        params[key] = {
            "black": int(round(float(black_in * 255.0))),
            "white": int(round(float(white_in * 255.0))),
            "gamma": round(float(gamma), 4),
            "black_out": int(round(float(black_out * 255.0))),
            "white_out": int(round(float(white_out * 255.0))),
        }
    return out, params, "ok"


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


def _hist_match_channel(src: torch.Tensor, tar: torch.Tensor):
    src_sort, src_idx = torch.sort(src)
    positions = torch.linspace(0, 1, src_sort.numel(), device=src.device, dtype=src.dtype)
    target_vals = torch.quantile(tar, positions.clamp(0, 1))
    out = torch.empty_like(src_sort)
    out[src_idx] = target_vals
    return out


def _hist_match(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]):
    flat_img = img.reshape(-1, 3)
    flat_ref = ref.reshape(-1, 3)
    if mask is not None:
        m = mask.view(-1) > 0.5
        ref_sel = flat_ref[m]
    else:
        ref_sel = flat_ref
    if ref_sel.shape[0] < 10:
        return img
    out = flat_img.clone()
    for c in range(3):
        out[:, c] = _hist_match_channel(flat_img[:, c], ref_sel[:, c])
    return torch.clamp(out.view_as(img), 0.0, 1.0)


def _pca_cov(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]):
    src = img.reshape(-1, 3)
    tar = ref.reshape(-1, 3)
    if mask is not None:
        m = mask.view(-1) > 0.5
        src = src[m]
        tar = tar[m]
    if src.shape[0] < 10 or tar.shape[0] < 10:
        return img
    src_mean = src.mean(dim=0)
    tar_mean = tar.mean(dim=0)
    src_c = src - src_mean
    tar_c = tar - tar_mean
    cov_src = src_c.t() @ src_c / src_c.shape[0] + torch.eye(3, device=img.device, dtype=img.dtype) * 1e-6
    cov_tar = tar_c.t() @ tar_c / tar_c.shape[0] + torch.eye(3, device=img.device, dtype=img.dtype) * 1e-6
    eig_src, E_src = torch.linalg.eigh(cov_src)
    eig_tar, E_tar = torch.linalg.eigh(cov_tar)
    sqrt_tar = E_tar @ torch.diag(torch.sqrt(torch.clamp(eig_tar, min=1e-6))) @ E_tar.t()
    inv_sqrt_src = E_src @ torch.diag(1.0 / torch.sqrt(torch.clamp(eig_src, min=1e-6))) @ E_src.t()
    A = sqrt_tar @ inv_sqrt_src
    flat = img.reshape(-1, 3)
    transformed = (A @ (flat - src_mean).t()).t() + tar_mean
    return torch.clamp(transformed.view_as(img), 0.0, 1.0)


def _rgb_to_hsv(rgb: torch.Tensor):
    r, g, b = rgb.unbind(-1)
    maxc, _ = rgb.max(dim=-1)
    minc, _ = rgb.min(dim=-1)
    v = maxc
    deltac = maxc - minc
    s = torch.where(maxc == 0, torch.zeros_like(maxc), deltac / torch.clamp(maxc, min=_EPS))
    hc = torch.zeros_like(maxc)
    mask = deltac > _EPS
    rc = ((maxc - r) / torch.clamp(deltac, min=_EPS))
    gc = ((maxc - g) / torch.clamp(deltac, min=_EPS))
    bc = ((maxc - b) / torch.clamp(deltac, min=_EPS))
    hc = torch.where((maxc == r) & mask, bc - gc, hc)
    hc = torch.where((maxc == g) & mask, 2.0 + rc - bc, hc)
    hc = torch.where((maxc == b) & mask, 4.0 + gc - rc, hc)
    h = (hc / 6.0) % 1.0
    return torch.stack((h, s, v), dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor):
    h, s, v = hsv.unbind(-1)
    h6 = h * 6.0
    i = torch.floor(h6).long()
    f = h6 - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)
    i_mod = i % 6
    r = torch.where(i_mod == 0, v, r)
    g = torch.where(i_mod == 0, t, g)
    b = torch.where(i_mod == 0, p, b)
    r = torch.where(i_mod == 1, q, r)
    g = torch.where(i_mod == 1, v, g)
    b = torch.where(i_mod == 1, p, b)
    r = torch.where(i_mod == 2, p, r)
    g = torch.where(i_mod == 2, v, g)
    b = torch.where(i_mod == 2, t, b)
    r = torch.where(i_mod == 3, p, r)
    g = torch.where(i_mod == 3, q, g)
    b = torch.where(i_mod == 3, v, b)
    r = torch.where(i_mod == 4, t, r)
    g = torch.where(i_mod == 4, p, g)
    b = torch.where(i_mod == 4, v, b)
    r = torch.where(i_mod == 5, v, r)
    g = torch.where(i_mod == 5, p, g)
    b = torch.where(i_mod == 5, q, b)
    return torch.stack((r, g, b), dim=-1)


def _hsv_shift(img: torch.Tensor, ref: torch.Tensor, mask: Optional[torch.Tensor]):
    hsv_img = _rgb_to_hsv(img)
    hsv_ref = _rgb_to_hsv(ref)
    h_img, s_img, v_img = hsv_img.unbind(-1)
    h_ref, s_ref, v_ref = hsv_ref.unbind(-1)
    if mask is not None:
        m = mask > 0.5
        h_diff = h_ref[m] - h_img[m]
        s_ratio = s_ref[m].mean() / torch.clamp(s_img[m].mean(), min=_EPS)
        v_ratio = v_ref[m].mean() / torch.clamp(v_img[m].mean(), min=_EPS)
    else:
        h_diff = h_ref.reshape(-1) - h_img.reshape(-1)
        s_ratio = s_ref.mean() / torch.clamp(s_img.mean(), min=_EPS)
        v_ratio = v_ref.mean() / torch.clamp(v_img.mean(), min=_EPS)
    hue_shift = torch.atan2(torch.sin(h_diff * 2 * math.pi).mean(), torch.cos(h_diff * 2 * math.pi).mean()) / (2 * math.pi)
    hsv_img_shifted = hsv_img.clone()
    hsv_img_shifted[..., 0] = (hsv_img_shifted[..., 0] + hue_shift) % 1.0
    hsv_img_shifted[..., 1] = torch.clamp(hsv_img_shifted[..., 1] * s_ratio, 0.0, 1.0)
    hsv_img_shifted[..., 2] = torch.clamp(hsv_img_shifted[..., 2] * v_ratio, 0.0, 1.0)
    corrected = torch.clamp(_hsv_to_rgb(hsv_img_shifted), 0.0, 1.0)
    params = {
        "hue_shift_deg": round(float(hue_shift * 360.0), 3),
        "saturation_mul": round(float(s_ratio), 4),
        "value_mul": round(float(v_ratio), 4),
    }
    return corrected, params, "ok"


def _rgb_to_lab(rgb: torch.Tensor):
    mask = rgb > 0.04045
    rgb_lin = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    M = rgb.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.matmul(rgb_lin, M.t())
    xyz_ref = rgb.new_tensor([0.95047, 1.0, 1.08883])
    xyz = xyz / xyz_ref
    delta = 6 / 29

    def f(t):
        return torch.where(t > delta ** 3, torch.pow(t, 1/3), t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def _delta_e76(lab1: torch.Tensor, lab2: torch.Tensor):
    diff = lab1 - lab2
    return torch.sqrt(torch.clamp((diff ** 2).sum(dim=-1), min=0.0))


def _delta_e_stats(de: torch.Tensor):
    flat = de.reshape(-1)
    if flat.numel() == 0:
        return {}
    return {
        "mean": round(float(flat.mean()), 4),
        "median": round(float(flat.median()), 4),
        "p95": round(float(torch.quantile(flat, 0.95)), 4),
        "under2": round(float((flat < 2.0).float().mean()), 4),
        "under5": round(float((flat < 5.0).float().mean()), 4),
        "max": round(float(flat.max()), 4),
    }


def _heatmap(de: torch.Tensor, vmax: float = 20.0):
    norm = torch.clamp(de / max(vmax, _EPS), 0.0, 1.0)
    r = norm
    g = 1.0 - torch.abs(norm - 0.5) * 2.0
    b = 1.0 - norm
    return torch.clamp(torch.stack([r, g, b], dim=-1), 0.0, 1.0)


def _perceptual_vgg(img: torch.Tensor, ref: torch.Tensor, steps: int, lr: float):
    # optimize 3x3 color matrix + bias to minimize VGG relu3_1 MSE
    from torchvision.models import vgg19, VGG19_Weights

    device = img.device
    _LOGGER.info("Loading VGG19 (perceptual_vgg)...")
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:12].to(device).eval()  # up to relu3_1
    for p in vgg.parameters():
        p.requires_grad = False

    def prep(x):
        # x: HWC 0..1
        x = x.permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        return (x - mean) / std

    with torch.inference_mode(False), torch.no_grad():
        feat_ref = vgg(prep(ref))

    W = torch.eye(3, device=device, dtype=img.dtype).requires_grad_(True)
    b = torch.zeros(3, device=device, dtype=img.dtype).requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=lr)

    # Exit inference_mode if enabled and recreate tensors to avoid "Inference tensors cannot be saved for backward"
    steps_int = max(1, int(steps))
    with torch.inference_mode(False):
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


def _waveform(img: torch.Tensor, mode: str, width: int, gain: float, use_log: bool):
    img = torch.clamp(img, 0.0, 1.0)
    H_in, W_in, _ = img.shape
    if W_in != width:
        img = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=(H_in, width), mode="nearest").squeeze(0).permute(1, 2, 0)
    H = _WAVEFORM_HEIGHT
    if mode == "parade":
        chans = [img[..., i] for i in range(3)]
        colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=img.device, dtype=img.dtype)
    else:
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        chans = [luma]
        colors = torch.tensor([[1, 1, 1]], device=img.device, dtype=img.dtype)
    canvas = torch.zeros((H, width, 3), device=img.device, dtype=img.dtype)
    levels = torch.clamp(torch.stack(chans, dim=-1) * (H - 1), 0, H - 1).long()
    for ci in range(len(chans)):
        bins = levels[..., ci]
        counts = torch.zeros((H, width), device=img.device, dtype=img.dtype)
        for x in range(width):
            hist = torch.bincount(bins[:, x], minlength=H).float()
            counts[:, x] = hist
        if use_log:
            counts = torch.log1p(counts)
        counts = counts * gain
        max_per_col = torch.clamp(counts.max(dim=0, keepdim=True).values, min=_EPS)
        counts = counts / max_per_col
        canvas[..., 0] += counts * colors[ci, 0]
        canvas[..., 1] += counts * colors[ci, 1]
        canvas[..., 2] += counts * colors[ci, 2]
    canvas = torch.flip(torch.clamp(canvas, 0.0, 1.0), dims=[0])
    return canvas


def _build_1d_cube_lut(resolve_params: dict | None, size: int) -> str | None:
    if not resolve_params:
        return None
    scale = resolve_params.get("scale")
    offset = resolve_params.get("offset")
    gamma = resolve_params.get("gamma")
    if not scale or not offset or not gamma:
        return None
    size = max(16, int(size))
    lines = ["# LUT generated by ImageColorMatchToReference", f"LUT_1D_SIZE {size}"]
    for i in range(size):
        x = i / (size - 1)
        row = []
        for c in range(3):
            y = (x * scale[c]) + offset[c]
            y = pow(max(y, 0.0), gamma[c]) if gamma[c] not in (None, 0) else y
            row.append(str(max(0.0, min(y, 1.0))))
        lines.append(" ".join(row))
    return "\n".join(lines)


# --- AdaIN (encoder/decoder) ---


def _adain_style_transfer(content: torch.Tensor, style: torch.Tensor, encoder, decoder):
    def encode(x):
        return encoder(x)

    def calc_mean_std(feat, eps=1e-5):
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    with torch.no_grad():
        f_c = encode(content)
        f_s = encode(style)
        mean_c, std_c = calc_mean_std(f_c)
        mean_s, std_s = calc_mean_std(f_s)
        norm = (f_c - mean_c) / std_c
        t = norm * std_s + mean_s
        out = decoder(t)
    return torch.clamp(out, 0.0, 1.0)


def _load_adain_weights(device):
    base = os.path.join(os.path.dirname(__file__), "models", "color_match", "adain")
    os.makedirs(base, exist_ok=True)
    vgg_path = os.path.join(base, "vgg_normalised.pth")
    dec_path = os.path.join(base, "decoder.pth")
    # auto-download if missing
    if not os.path.exists(vgg_path):
        _LOGGER.info("Downloading AdaIN encoder weights...")
        urllib.request.urlretrieve(
            "https://github.com/naoto0804/pytorch-AdaIN/raw/master/models/vgg_normalised.pth",
            vgg_path,
        )
    if not os.path.exists(dec_path):
        _LOGGER.info("Downloading AdaIN decoder weights...")
        urllib.request.urlretrieve(
            "https://github.com/naoto0804/pytorch-AdaIN/raw/master/models/decoder.pth",
            dec_path,
        )

    from torchvision.models import vgg19

    vgg = vgg19(weights=None).features.to(device).eval()
    vgg.load_state_dict(torch.load(vgg_path, map_location=device))
    encoder = torch.nn.Sequential(*list(vgg.children())[:21]).eval()  # up to relu4_1
    for p in encoder.parameters():
        p.requires_grad_(False)

    decoder = torch.nn.Sequential(
        torch.nn.Conv2d(512, 256, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(scale_factor=2, mode="nearest"),
        torch.nn.Conv2d(256, 256, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(256, 256, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(256, 256, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(256, 128, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(scale_factor=2, mode="nearest"),
        torch.nn.Conv2d(128, 128, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(128, 64, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Upsample(scale_factor=2, mode="nearest"),
        torch.nn.Conv2d(64, 64, 3, 1, 1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 3, 3, 1, 1),
    ).to(device).eval()
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    for p in decoder.parameters():
        p.requires_grad_(False)
    return encoder, decoder


class ImageColorMatchToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Базовое изображение (образец)."}),
                "image": ("IMAGE", {"tooltip": "Изображение, которое нужно подогнать по цвету."}),
                "mode": ([
                    "levels",
                    "mean_std",
                    "linear",
                    "hist",
                    "pca_cov",
                    "lab_l",
                    "lab_full",
                    "lab_l_cdf",
                    "lab_cdf",
                    "hsv_shift",
                    "perceptual_vgg",
                    "perceptual_adain",
                    "perceptual_ltct",
                    "perceptual_lut3d",
                    "perceptual_unet",
                ], {"default": "levels", "tooltip": "Метод коррекции."}),
                "percentile": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Процентиль для levels (обрезка хвостов)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Сила применения коррекции (0..1)."}),
                "clip": ("BOOLEAN", {"default": True, "tooltip": "Обрезать результат в диапазоне 0..1."}),
            },
            "optional": {
                "match_mask": ("MASK", {"tooltip": "Где считать статистику (белое=учитывать)."}),
                "apply_mask": ("MASK", {"tooltip": "Где применять коррекцию (белое=применить, чёрное=оставить исходное)."}),
                "preserve_alpha": ("BOOLEAN", {"default": True, "tooltip": "Если вход RGBA — сохранить альфу из исходника."}),
                "export_lut": ("BOOLEAN", {"default": False, "tooltip": "Сгенерировать 1D LUT (.cube) из linear/levels параметров."}),
                "lut_size": ("INT", {"default": 256, "min": 16, "max": 1024, "tooltip": "Размер 1D LUT."}),
                "waveform_enabled": ("BOOLEAN", {"default": False, "tooltip": "Строить waveform/parade для контроля."}),
                "waveform_mode": (["luma", "parade"], {"default": "parade", "tooltip": "Режим waveform: лума или RGB parade."}),
                "waveform_width": ("INT", {"default": 512, "min": 128, "max": 2048, "tooltip": "Ширина waveform."}),
                "waveform_gain": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Усиление яркости точек waveform."}),
                "waveform_log": ("BOOLEAN", {"default": True, "tooltip": "Логарифмическая шкала интенсивностей waveform."}),
                "deltae_heatmap": ("BOOLEAN", {"default": False, "tooltip": "Вывести heatmap ΔE как IMAGE."}),
                "perceptual_steps": ("INT", {"default": 30, "min": 1, "max": 200, "tooltip": "Итерации оптимизации для perceptual_vgg."}),
                "perceptual_lr": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.01, "tooltip": "LR для perceptual_vgg (оптимизация 3x3+bias)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "difference", "deltae_heatmap", "waveform_ref", "waveform_matched", "match_json")
    FUNCTION = "match"
    CATEGORY = "image/color"

    def match(
        self,
        reference,
        image,
        mode,
        percentile,
        clip,
        match_mask=None,
        apply_mask=None,
        preserve_alpha=True,
        export_lut=False,
        lut_size=256,
        strength=1.0,
        waveform_enabled=False,
        waveform_mode="parade",
        waveform_width=512,
        waveform_gain=1.0,
        waveform_log=True,
        deltae_heatmap=False,
        perceptual_steps=30,
        perceptual_lr=0.05,
    ):
        batch_size = max(reference.shape[0], image.shape[0])
        matched_list = []
        diff_list = []
        deltae_list = []
        wave_ref_list = []
        wave_match_list = []
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

            status = "ok"
            gimp_levels = None
            gimp_hsv = None
            deep_params = None

            if mode == "levels":
                corrected_t, gimp_levels, status = _apply_levels(img_t, ref_t, mm_t, float(percentile))
            elif mode == "hsv_shift":
                corrected_t, gimp_hsv, status = _hsv_shift(img_t, ref_t, mm_t)
            elif mode == "mean_std":
                corrected_t = _mean_std_match(img_t, ref_t, mm_t)
            elif mode == "linear":
                corrected_t = _linear_match(img_t, ref_t, mm_t)
            elif mode == "hist":
                corrected_t = _hist_match(img_t, ref_t, mm_t)
            elif mode == "pca_cov":
                corrected_t = _pca_cov(img_t, ref_t, mm_t)
            elif mode in ("lab_l", "lab_full", "lab_l_cdf", "lab_cdf"):
                corrected_t = _lab_match_torch(img_t, ref_t, mm_t, mode)
            elif mode == "perceptual_vgg":
                corrected_t, deep_params = _perceptual_vgg(img_t, ref_t, perceptual_steps, perceptual_lr)
            elif mode == "perceptual_adain":
                encoder, decoder = _load_adain_weights(img_t.device)
                # HWC -> NCHW
                c_bchw = img_t.permute(2, 0, 1).unsqueeze(0)
                s_bchw = ref_t.permute(2, 0, 1).unsqueeze(0)
                corrected_bchw = _adain_style_transfer(c_bchw, s_bchw, encoder, decoder)
                corrected_t = torch.clamp(corrected_bchw.squeeze(0).permute(1, 2, 0), 0.0, 1.0)
                deep_params = {"mode": "adain", "weights": "naoto0804/pytorch-AdaIN"}
            elif mode == "perceptual_ltct":
                _not_implemented_mode("perceptual_ltct")
            elif mode == "perceptual_lut3d":
                _not_implemented_mode("perceptual_lut3d")
            elif mode == "perceptual_unet":
                _not_implemented_mode("perceptual_unet")
            else:
                corrected_t = img_t

            if status.startswith("error"):
                corrected_t = img_t

            if strength < 1.0:
                corrected_t = img_t * (1.0 - strength) + corrected_t * strength

            scale_t, offset_t = _linear_fit_torch(img_t, ref_t, mm_t)
            resolve_params = {
                "scale": [round(float(s), 5) for s in scale_t],
                "offset": [round(float(o), 5) for o in offset_t],
                "gamma": [
                    gimp_levels[k]["gamma"] if gimp_levels else 1.0
                    for k in ("r", "g", "b")
                ],
            }
            fusion_params = {
                "gain": resolve_params["scale"],
                "lift": resolve_params["offset"],
                "gamma": resolve_params["gamma"],
            }

            if am_t is not None:
                mask_apply = am_t[..., None]
                corrected_t = corrected_t * mask_apply + img_t * (1.0 - mask_apply)

            if clip:
                corrected_t = torch.clamp(corrected_t, 0.0, 1.0)

            matched_t = corrected_t
            if alpha_channel is not None and preserve_alpha:
                matched_t = torch.cat([matched_t, alpha_channel], dim=-1)

            diff = torch.abs(matched_t[..., :3] - ref_t)
            delta_e = _delta_e76(_rgb_to_lab(ref_t), _rgb_to_lab(corrected_t))
            delta_stats = _delta_e_stats(delta_e)

            if deltae_heatmap:
                heat = _heatmap(delta_e)
            else:
                heat = torch.zeros((1, 1, 3), dtype=matched_t.dtype, device=matched_t.device)

            if waveform_enabled:
                wave_ref = _waveform(ref_t, waveform_mode, int(waveform_width), float(waveform_gain), bool(waveform_log))
                wave_match = _waveform(corrected_t, waveform_mode, int(waveform_width), float(waveform_gain), bool(waveform_log))
            else:
                wave_ref = torch.zeros((1, 1, 3), dtype=matched_t.dtype, device=matched_t.device)
                wave_match = torch.zeros((1, 1, 3), dtype=matched_t.dtype, device=matched_t.device)

            stats = {
                "ref_mean": [round(float(x), 4) for x in ref_t.reshape(-1, 3).mean(dim=0)],
                "img_mean": [round(float(x), 4) for x in img_t.reshape(-1, 3).mean(dim=0)],
                "ref_std": [round(float(x), 4) for x in ref_t.reshape(-1, 3).std(dim=0)],
                "img_std": [round(float(x), 4) for x in img_t.reshape(-1, 3).std(dim=0)],
                "mask_used": mm_t is not None,
                "delta_e": delta_stats,
            }
            presets = {
                "gimp": {
                    "levels": gimp_levels,
                    "hue_saturation": gimp_hsv,
                    "hint": "Colors -> Levels; Colors -> Hue-Saturation.",
                },
                "resolve": {
                    "color_wheels": {
                        "gain": resolve_params["scale"] if resolve_params else None,
                        "lift": resolve_params["offset"] if resolve_params else None,
                        "gamma": resolve_params["gamma"] if resolve_params else None,
                    },
                    "hint": "Primaries gain/lift/gamma per channel.",
                },
                "fusion": {
                    "color_corrector": {
                        "gain": fusion_params["gain"] if fusion_params else None,
                        "lift": fusion_params["lift"] if fusion_params else None,
                        "gamma": fusion_params["gamma"] if fusion_params else None,
                    },
                    "hint": "Fusion ColorCorrector gain/lift/gamma.",
                },
            }
            payload = {
                "status": status,
                "mode": mode,
                "gimp_levels": gimp_levels,
                "gimp_hsv": gimp_hsv,
                "resolve": resolve_params,
                "fusion": fusion_params,
                "linear": {
                    "scale": resolve_params["scale"] if resolve_params else None,
                    "offset": resolve_params["offset"] if resolve_params else None,
                },
                "deep": deep_params,
                "presets": presets,
                "stats": stats,
            }
            if export_lut:
                lut_text = _build_1d_cube_lut(resolve_params, int(lut_size))
                payload["lut_1d_cube"] = lut_text
                payload["lut_size"] = int(lut_size)

            json_list.append(json.dumps(payload, ensure_ascii=True))
            matched_list.append(matched_t.cpu())
            diff_list.append(diff.cpu())
            deltae_list.append(heat.cpu())
            wave_ref_list.append(wave_ref.cpu())
            wave_match_list.append(wave_match.cpu())

        return (
            torch.stack(matched_list, dim=0),
            torch.stack(diff_list, dim=0),
            torch.stack(deltae_list, dim=0),
            torch.stack(wave_ref_list, dim=0),
            torch.stack(wave_match_list, dim=0),
            json_list,
        )


_LOGGER.warning("Loaded ImageColorMatchToReference. NODE_CLASS_MAPPINGS=%s", ["ImageColorMatchToReference"])
def _not_implemented_mode(name: str):
    raise RuntimeError(
        f"{name} mode requested but weights/model not provided. "
        "Please place weights under models/color_match/ and extend loader accordingly."
    )
