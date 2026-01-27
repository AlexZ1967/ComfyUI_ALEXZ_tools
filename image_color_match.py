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
        }
    return out, params, "ok"


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


class ImageColorMatchToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "Базовое изображение (образец)."}),
                "image": ("IMAGE", {"tooltip": "Изображение, которое нужно подогнать по цвету."}),
                "mode": (["levels", "mean_std", "linear", "hist", "lab_l", "lab_full", "lab_l_cdf", "lab_cdf", "hsv_shift"], {"default": "levels", "tooltip": "Метод коррекции."}),
                "percentile": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Процентиль для levels (обрезка хвостов)."}),
                "clip": ("BOOLEAN", {"default": True, "tooltip": "Обрезать результат в диапазоне 0..1."}),
            },
            "optional": {
                "match_mask": ("MASK", {"tooltip": "Где считать статистику (белое=учитывать)."}),
                "apply_mask": ("MASK", {"tooltip": "Где применять коррекцию (белое=применить, чёрное=оставить исходное)."}),
                "preserve_alpha": ("BOOLEAN", {"default": True, "tooltip": "Если вход RGBA — сохранить альфу из исходника."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "difference", "match_json")
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
    ):
        batch_size = max(reference.shape[0], image.shape[0])
        matched_list = []
        diff_list = []
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

            if am_t is not None:
                mask_apply = am_t.detach().cpu().numpy()[..., None]
                corrected = corrected * mask_apply + img_np * (1.0 - mask_apply)

            if clip:
                corrected = np.clip(corrected, 0.0, 1.0)

            matched_t = torch.from_numpy(corrected.astype(np.float32))
            if alpha_channel is not None and preserve_alpha:
                matched_t = torch.cat([matched_t, alpha_channel], dim=-1)

            diff = torch.abs(matched_t[..., :3] - ref_t)

            stats = {
                "ref_mean": [round(float(x), 4) for x in ref_np.reshape(-1, 3).mean(axis=0)],
                "img_mean": [round(float(x), 4) for x in img_np.reshape(-1, 3).mean(axis=0)],
                "mask_used": match_mask_np is not None,
            }
            payload = {
                "status": status,
                "mode": mode,
                "gimp_levels": gimp_levels,
                "gimp_hsv": gimp_hsv,
                "stats": stats,
            }
            json_list.append(json.dumps(payload, ensure_ascii=True))
            matched_list.append(matched_t)
            diff_list.append(diff)

        return (
            torch.stack(matched_list, dim=0),
            torch.stack(diff_list, dim=0),
            json_list,
        )


_LOGGER.warning(
    "Loaded ImageColorMatchToReference. NODE_CLASS_MAPPINGS=%s",
    ["ImageColorMatchToReference"],
)
