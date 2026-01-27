import json
import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as torch_nn_func

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None

from .color_match_utils import (
    apply_color_match,
    ensure_mask_batch,
    normalize_mask,
    resize_images_to_size,
    resize_mask_to_output,
)
from .utils import select_batch_item


_LOGGER = logging.getLogger("ImageColorMatchToReference")
_EPS = 1e-6
_WAVEFORM_HEIGHT = 256


def _torch_resize_image(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    img_bchw = img.permute(2, 0, 1).unsqueeze(0)
    resized = torch_nn_func.interpolate(img_bchw, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0)


def _get_percentiles(arr: np.ndarray, mask: Optional[np.ndarray], p: float) -> tuple[float, float, float] | None:
    vals = arr.reshape(-1) if mask is None else arr[mask]
    if vals.size < 10:
        return None
    lo = float(np.percentile(vals, p))
    hi = float(np.percentile(vals, 100.0 - p))
    med = float(np.percentile(vals, 50.0))
    return lo, hi, med


def _apply_levels(
    img: np.ndarray,
    ref: np.ndarray,
    mask: Optional[np.ndarray],
    percentile: float,
) -> tuple[np.ndarray, dict, str]:
    params = {"r": None, "g": None, "b": None}
    out = img.copy()
    for ch, key in enumerate(("r", "g", "b")):
        bounds_img = _get_percentiles(img[..., ch], mask, percentile)
        bounds_ref = _get_percentiles(ref[..., ch], mask, percentile)
        if bounds_img is None or bounds_ref is None:
            return img, params, "error: not enough pixels to estimate levels"

        black_in, white_in, med_in = bounds_img
        black_out, white_out, med_out = bounds_ref
        denom = max(white_in - black_in, _EPS)
        norm_in = np.clip((img[..., ch] - black_in) / denom, 0.0, 1.0)

        norm_med_in = np.clip((med_in - black_in) / denom, _EPS, 1.0 - _EPS)
        norm_med_out = np.clip((med_out - black_out) / max(white_out - black_out, _EPS), _EPS, 1.0 - _EPS)
        gamma = math.log(norm_med_out) / math.log(norm_med_in) if norm_med_in not in (0.0, 1.0) else 1.0

        corrected = np.power(norm_in, gamma)
        corrected = corrected * (white_out - black_out) + black_out
        out[..., ch] = corrected
        params[key] = {
            "black": int(round(black_in * 255.0)),
            "white": int(round(white_in * 255.0)),
            "gamma": round(float(gamma), 4),
            "black_out": int(round(black_out * 255.0)),
            "white_out": int(round(white_out * 255.0)),
        }
    return out, params, "ok"


def _linear_fit(img: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if mask is not None:
        keep = mask
        img_sel = img[keep]
        ref_sel = ref[keep]
    else:
        img_sel = img.reshape(-1, 3)
        ref_sel = ref.reshape(-1, 3)
    mean_img = img_sel.mean(axis=0)
    mean_ref = ref_sel.mean(axis=0)
    std_img = img_sel.std(axis=0)
    std_ref = ref_sel.std(axis=0)
    scale = np.where(std_img > _EPS, std_ref / std_img, 1.0)
    offset = mean_ref - scale * mean_img
    return scale, offset


def _apply_hsv_shift(img: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray]) -> tuple[np.ndarray, dict, str]:
    if cv2 is None:
        return img, {}, "error: opencv-python is required for hsv_shift"

    img_hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    ref_hsv = cv2.cvtColor((ref * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    hue_img = img_hsv[..., 0]
    hue_ref = ref_hsv[..., 0]
    sat_img = img_hsv[..., 1]
    sat_ref = ref_hsv[..., 1]
    val_img = img_hsv[..., 2]
    val_ref = ref_hsv[..., 2]

    if mask is not None:
        m = mask
        hue_diff = hue_ref[m] - hue_img[m]
        sat_ratio = sat_ref[m].mean() / max(sat_img[m].mean(), _EPS)
        val_ratio = val_ref[m].mean() / max(val_img[m].mean(), _EPS)
    else:
        hue_diff = hue_ref.reshape(-1) - hue_img.reshape(-1)
        sat_ratio = sat_ref.mean() / max(sat_img.mean(), _EPS)
        val_ratio = val_ref.mean() / max(val_img.mean(), _EPS)

    # average hue difference with wrap-around (OpenCV hue range 0..180)
    hue_diff_rad = hue_diff * (np.pi / 90.0)
    mean_sin = np.sin(hue_diff_rad).mean()
    mean_cos = np.cos(hue_diff_rad).mean()
    avg_diff_rad = math.atan2(mean_sin, mean_cos)
    hue_shift_deg = avg_diff_rad * (90.0 / np.pi)

    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift_deg) % 180.0
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * sat_ratio, 0.0, 255.0)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * val_ratio, 0.0, 255.0)

    corrected_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    params = {
        "hue_shift_deg": round(float(hue_shift_deg), 3),
        "saturation_mul": round(float(sat_ratio), 4),
        "value_mul": round(float(val_ratio), 4),
    }
    return corrected_rgb, params, "ok"


def _build_1d_cube_lut(resolve_params: dict | None, size: int) -> str | None:
    if not resolve_params:
        return None
    scale = resolve_params.get("scale")
    offset = resolve_params.get("offset")
    gamma = resolve_params.get("gamma")
    if not scale or not offset or not gamma:
        return None
    size = max(16, int(size))
    lines = ["# LUT generated by ImageColorMatchToReference", "LUT_1D_SIZE {}".format(size)]
    for i in range(size):
        x = i / (size - 1)
        row = []
        for c in range(3):
            y = (x * scale[c]) + offset[c]
            y = pow(max(y, 0.0), gamma[c]) if gamma[c] not in (None, 0) else y
            row.append(str(max(0.0, min(y, 1.0))))
        lines.append(" ".join(row))
    return "\n".join(lines)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
        # OpenCV Lab: L 0..255, a/b 0..255 -> bring to common float scale
        lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
        lab[..., 1:] = lab[..., 1:] - 128.0
        return lab
    # Fallback manual conversion (approx) to XYZ then Lab
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = rgb_lin @ M.T
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / xyz_ref

    def f(t):
        delta = 6 / 29
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    diff = lab1 - lab2
    return np.sqrt((diff ** 2).sum(axis=-1))


def _delta_e_stats(delta_e: np.ndarray) -> dict:
    flat = delta_e.reshape(-1)
    if flat.size == 0:
        return {}
    return {
        "mean": round(float(np.mean(flat)), 4),
        "median": round(float(np.median(flat)), 4),
        "p95": round(float(np.percentile(flat, 95)), 4),
        "under2": round(float((flat < 2.0).mean()), 4),
        "under5": round(float((flat < 5.0).mean()), 4),
        "max": round(float(np.max(flat)), 4),
    }


def _build_heatmap(delta_e: np.ndarray, vmax: float = 20.0) -> np.ndarray:
    norm = np.clip(delta_e / max(vmax, _EPS), 0.0, 1.0)
    if cv2 is not None:
        norm_u8 = (norm * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # fallback simple colormap: blue->green->red
    r = norm
    g = 1.0 - np.abs(norm - 0.5) * 2.0
    b = 1.0 - norm
    heat = np.stack([r, g, b], axis=-1)
    return np.clip(heat, 0.0, 1.0)


def _downsample_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w, _ = img.shape
    if w == target_w:
        return img
    if cv2 is not None:
        return cv2.resize(img, (target_w, h), interpolation=cv2.INTER_AREA)
    # simple block average
    scale = w / target_w
    idx = (np.arange(target_w) * scale).astype(int)
    idx = np.clip(idx, 0, w - 1)
    return img[:, idx, :]


def _build_waveform(img: np.ndarray, mode: str, width: int, gain: float, use_log: bool) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    img_ds = _downsample_width(img, width)
    h, w, _ = img_ds.shape
    H = _WAVEFORM_HEIGHT

    if mode == "parade":
        chans = [img_ds[..., i] for i in range(3)]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    else:
        luma = 0.2126 * img_ds[..., 0] + 0.7152 * img_ds[..., 1] + 0.0722 * img_ds[..., 2]
        chans = [luma]
        colors = [(1, 1, 1)]

    canvas = np.zeros((H, w, 3), dtype=np.float32)
    for chan, color in zip(chans, colors):
        # hist per column
        bins = (np.clip(chan, 0.0, 1.0) * (H - 1)).astype(int)
        for x in range(w):
            col_bins, counts = np.unique(bins[:, x], return_counts=True)
            if use_log:
                counts = np.log1p(counts)
            counts = counts * gain
            if counts.max() > 0:
                counts = counts / counts.max()
            canvas[col_bins, x, 0] += counts * color[0]
            canvas[col_bins, x, 1] += counts * color[1]
            canvas[col_bins, x, 2] += counts * color[2]

    canvas = np.clip(canvas, 0.0, 1.0)
    canvas = np.flipud(canvas)  # 0 at bottom
    return canvas


class ImageColorMatchToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Базовое изображение (образец)."}),
                "image": ("IMAGE", {"tooltip": "Изображение, которое нужно подогнать по цвету."}),
                "mode": (["levels", "mean_std", "linear", "hist", "pca_cov", "lab_l", "lab_full", "lab_l_cdf", "lab_cdf", "hsv_shift"], {"default": "levels", "tooltip": "Метод коррекции."}),
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
                "waveform_width": ("INT", {"default": 512, "min": 128, "max": 2048, "tooltip": "Ширина waveform в пикселях."}),
                "waveform_gain": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Усиление яркости точек waveform."}),
                "waveform_log": ("BOOLEAN", {"default": True, "tooltip": "Логарифмическая шкала интенсивностей waveform."}),
                "deltae_heatmap": ("BOOLEAN", {"default": False, "tooltip": "Вывести heatmap ΔE как IMAGE."}),
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
                img_t = _torch_resize_image(img_t, ref_h, ref_w)

            mm_t = select_batch_item(match_mask, idx) if match_mask is not None else None
            am_t = select_batch_item(apply_mask, idx) if apply_mask is not None else None
            if mm_t is not None:
                mm_t = resize_mask_to_output(normalize_mask(mm_t), ref_h, ref_w)
                match_mask_np = (mm_t.detach().cpu().numpy() > 0.5)
            else:
                match_mask_np = None
            if am_t is not None:
                am_t = resize_mask_to_output(normalize_mask(am_t), ref_h, ref_w)

            img_np = img_t.detach().cpu().numpy().astype(np.float32)
            ref_np = ref_t.detach().cpu().numpy().astype(np.float32)

            status = "ok"
            gimp_levels = None
            gimp_hsv = None
            resolve_params = None
            fusion_params = None
            linear_scale = None
            linear_offset = None

            if mode == "levels":
                corrected, gimp_levels, status = _apply_levels(img_np, ref_np, match_mask_np, float(percentile))
            elif mode == "hsv_shift":
                corrected, gimp_hsv, status = _apply_hsv_shift(img_np, ref_np, match_mask_np)
            else:
                if match_mask_np is not None:
                    mask_for_stats = torch.from_numpy(1.0 - match_mask_np.astype(np.float32))
                else:
                    mask_for_stats = torch.zeros((1, 1))
                corrected_t = apply_color_match(
                    output_images=torch.from_numpy(img_np).unsqueeze(0),
                    reference_images=torch.from_numpy(ref_np).unsqueeze(0),
                    mask=mask_for_stats if match_mask_np is not None else torch.zeros((1, 1)),
                    mode=mode,
                ).squeeze(0)
                corrected = corrected_t.detach().cpu().numpy().astype(np.float32)

            if status.startswith("error"):
                corrected = img_np

            if strength < 1.0:
                corrected = img_np * (1.0 - strength) + corrected * strength

            linear_scale, linear_offset = _linear_fit(img_np, ref_np, match_mask_np)
            resolve_params = {
                "scale": [round(float(s), 5) for s in linear_scale],
                "offset": [round(float(o), 5) for o in linear_offset],
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
                mask_apply = am_t.detach().cpu().numpy()[..., None]
                corrected = corrected * mask_apply + img_np * (1.0 - mask_apply)

            if clip:
                corrected = np.clip(corrected, 0.0, 1.0)

            matched_t = torch.from_numpy(corrected.astype(np.float32))
            if alpha_channel is not None and preserve_alpha:
                matched_t = torch.cat([matched_t, alpha_channel], dim=-1)

            diff = torch.abs(matched_t[..., :3] - ref_t)
            lab_ref = _rgb_to_lab(ref_np)
            lab_matched = _rgb_to_lab(corrected)
            delta_e = _delta_e76(lab_ref, lab_matched)
            delta_stats = _delta_e_stats(delta_e)

            if deltae_heatmap:
                heat = torch.from_numpy(_build_heatmap(delta_e)).float()
            else:
                heat = torch.zeros((1, 1, 3), dtype=torch.float32)

            if waveform_enabled:
                wave_ref = torch.from_numpy(_build_waveform(ref_np, waveform_mode, int(waveform_width), float(waveform_gain), bool(waveform_log)))
                wave_match = torch.from_numpy(_build_waveform(corrected, waveform_mode, int(waveform_width), float(waveform_gain), bool(waveform_log)))
            else:
                wave_ref = torch.zeros((1, 1, 3), dtype=torch.float32)
                wave_match = torch.zeros((1, 1, 3), dtype=torch.float32)

            stats = {
                "ref_mean": [round(float(x), 4) for x in ref_np.reshape(-1, 3).mean(axis=0)],
                "img_mean": [round(float(x), 4) for x in img_np.reshape(-1, 3).mean(axis=0)],
                "ref_std": [round(float(x), 4) for x in ref_np.reshape(-1, 3).std(axis=0)],
                "img_std": [round(float(x), 4) for x in img_np.reshape(-1, 3).std(axis=0)],
                "mask_used": match_mask_np is not None,
                "delta_e": delta_stats,
            }
            presets = {
                "gimp": {
                    "levels": gimp_levels,
                    "hue_saturation": gimp_hsv,
                    "hint": "Colors -> Levels (set per-channel input black/white/gamma); Colors -> Hue-Saturation for hue/sat/value.",
                },
                "resolve": {
                    "color_wheels": {
                        "gain": resolve_params["scale"] if resolve_params else None,
                        "lift": resolve_params["offset"] if resolve_params else None,
                        "gamma": resolve_params["gamma"] if resolve_params else None,
                    },
                    "hint": "Apply gain/offset per channel in Primaries (Log/Wheels). Gamma ~= per-channel power.",
                },
                "fusion": {
                    "color_corrector": {
                        "gain": fusion_params["gain"] if fusion_params else None,
                        "lift": fusion_params["lift"] if fusion_params else None,
                        "gamma": fusion_params["gamma"] if fusion_params else None,
                    },
                    "hint": "In CC node: set Gain RGB, Lift RGB, Gamma RGB.",
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
                    "scale": [round(float(s), 5) for s in linear_scale] if linear_scale is not None else None,
                    "offset": [round(float(o), 5) for o in linear_offset] if linear_offset is not None else None,
                },
                "presets": presets,
                "stats": stats,
            }
            if export_lut:
                lut_text = _build_1d_cube_lut(resolve_params, int(lut_size))
                payload["lut_1d_cube"] = lut_text
                payload["lut_size"] = int(lut_size)
            json_list.append(json.dumps(payload, ensure_ascii=True))
            matched_list.append(matched_t)
            diff_list.append(diff)
            deltae_list.append(heat)
            wave_ref_list.append(wave_ref)
            wave_match_list.append(wave_match)

        return (
            torch.stack(matched_list, dim=0),
            torch.stack(diff_list, dim=0),
            torch.stack(deltae_list, dim=0),
            torch.stack(wave_ref_list, dim=0),
            torch.stack(wave_match_list, dim=0),
            json_list,
        )


_LOGGER.warning(
    "Loaded ImageColorMatchToReference. NODE_CLASS_MAPPINGS=%s",
    ["ImageColorMatchToReference"],
)
