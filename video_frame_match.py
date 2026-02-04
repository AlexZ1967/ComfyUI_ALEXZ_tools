import json
import logging
import os
from collections import deque
from typing import Optional, Tuple

import cv2
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from .utils import ensure_hwc, image_difference, normalize_to_reference, resize_to_hw

_LOGGER = logging.getLogger("VideoFrameMatch")

def _list_videos():
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    return folder_paths.filter_files_content_types(files, ["video"])


def _to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
    return frame

def _resize_to_match(frame: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = target_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    frame_bchw = frame.permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        frame_bchw, size=(h, w), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).permute(1, 2, 0)
def _mse_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2).item())


def _get_total_frames(cap: cv2.VideoCapture) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return 0
    return total


_SSIM_WINDOW_CACHE = {}


def _get_ssim_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5):
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


def _ssim_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    channels = a.shape[1]
    device = a.device
    dtype = a.dtype
    window = _get_ssim_window(channels, device, dtype)

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
    ssim_val = ssim_map.mean().clamp(0.0, 1.0)
    return float((1.0 - ssim_val).item())


_LPIPS_CACHE = {}


def _get_lpips_model(net: str, device: torch.device):
    key = (net, device.type)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lpips is required for metric=lpips. Install: pip install lpips") from exc
    model = lpips.LPIPS(net=net).to(device).eval()
    _LPIPS_CACHE[key] = model
    return model


def _lpips_distance(a: torch.Tensor, b: torch.Tensor, net: str, device: torch.device) -> float:
    model = _get_lpips_model(net, device)
    a = a.permute(2, 0, 1).unsqueeze(0).to(device)
    b = b.permute(2, 0, 1).unsqueeze(0).to(device)
    a = a * 2.0 - 1.0
    b = b * 2.0 - 1.0
    with torch.inference_mode():
        dist = model(a, b)
    return float(dist.item())


_CLIP_CACHE = {}
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _get_clip_model(model_name: str, pretrained: str, device: torch.device):
    key = (model_name, pretrained, device.type)
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    try:
        import open_clip  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "open-clip-torch is required for metric=clip. Install: pip install open-clip-torch"
        ) from exc
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    image_size = getattr(model.visual, "image_size", 224)
    if isinstance(image_size, tuple):
        image_size = image_size[0]
    _CLIP_CACHE[key] = (model, image_size)
    return model, image_size


def _clip_preprocess(img: torch.Tensor, size: int, device: torch.device) -> torch.Tensor:
    x = img.permute(2, 0, 1).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _clip_encode(img: torch.Tensor, model, size: int, device: torch.device) -> torch.Tensor:
    x = _clip_preprocess(img, size, device)
    with torch.inference_mode():
        feat = model.encode_image(x)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    return feat


def _compute_score(
    metric: str,
    frame: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    lpips_net: str,
    clip_model: str,
    clip_pretrained: str,
    clip_ctx: Optional[Tuple[torch.nn.Module, int, torch.Tensor]] = None,
) -> float:
    if metric == "mse":
        return _mse_score(frame, target)
    if metric == "ssim":
        return _ssim_distance(frame.to(device), target.to(device))
    if metric == "lpips":
        return _lpips_distance(frame, target, lpips_net, device)
    if metric == "clip":
        if clip_ctx is None:
            model, size = _get_clip_model(clip_model, clip_pretrained, device)
            target_feat = _clip_encode(target, model, size, device)
        else:
            model, size, target_feat = clip_ctx
        feat = _clip_encode(frame, model, size, device)
        sim = (feat * target_feat).sum(dim=-1)
        return float((1.0 - sim).item())
    raise ValueError(f"Unknown metric: {metric}")


class VideoFrameMatch:
    @classmethod
    def INPUT_TYPES(cls):
        videos = sorted(_list_videos())
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Целевой кадр для поиска в видео."}),
                "video": (videos, {"video_upload": True, "tooltip": "Видео из папки input/."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Количество последних кадров для анализа (0 = все)."}),
                "metric": (["mse", "ssim", "lpips", "clip"], {"default": "mse", "tooltip": "Метрика сходства кадра и картинки."}),
                "normalize": (["none", "mean_std", "linear", "hist"], {"default": "none", "tooltip": "Нормализация кадра к референсу перед сравнением."}),
                "metric_size": ("INT", {"default": 0, "min": 0, "max": 2048, "tooltip": "Если >0, ресайз обеих картинок к размеру метрики (квадрат)."}),
                "lpips_net": (["vgg", "alex"], {"default": "vgg", "tooltip": "Бэкбон LPIPS (vgg качественнее, alex быстрее)."}),
                "clip_model": ("STRING", {"default": "ViT-B-32", "tooltip": "Имя модели open_clip."}),
                "clip_pretrained": ("STRING", {"default": "laion2b_s34b_b79k", "tooltip": "pretrained тег open_clip."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("best_frame", "difference", "best_index", "best_score", "scores_json")
    FUNCTION = "match"
    CATEGORY = "video/utils"

    def match(self, image, video, max_frames, metric, normalize, metric_size, lpips_net, clip_model, clip_pretrained):
        target = image[0] if isinstance(image, list) else image
        target = torch.clamp(ensure_hwc(target), 0.0, 1.0).float()
        h_t, w_t = target.shape[:2]

        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        best_score: Optional[float] = None
        best_index: int = -1
        best_frame_tensor: Optional[torch.Tensor] = None
        scores = []

        use_gpu = torch.cuda.is_available() and metric in {"ssim", "lpips", "clip"}
        metric_device = torch.device("cuda" if use_gpu else "cpu")
        metric_size_eff = 0 if metric == "clip" else metric_size
        target_metric = resize_to_hw(target, (metric_size_eff, metric_size_eff)) if metric_size_eff else target
        clip_ctx = None
        if metric == "clip":
            model, size = _get_clip_model(clip_model, clip_pretrained, metric_device)
            target_feat = _clip_encode(target_metric, model, size, metric_device)
            clip_ctx = (model, size, target_feat)

        total_frames = _get_total_frames(cap)
        use_tail_only = bool(max_frames)
        start_idx = 0
        seek_ok = False

        if use_tail_only and total_frames > 0:
            start_idx = max(0, total_frames - max_frames)
            if start_idx == 0:
                seek_ok = True
            else:
                seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        try:
            if use_tail_only and (total_frames == 0 or not seek_ok):
                if total_frames > 0 and not seek_ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                queue = deque(maxlen=max_frames)
                idx = 0
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_t = ensure_hwc(_to_tensor(frame_bgr))
                    frame_t = _resize_to_match(frame_t, (h_t, w_t))
                    frame_metric = normalize_to_reference(frame_t, target, normalize) if normalize != "none" else frame_t
                    if metric_size_eff:
                        frame_metric = resize_to_hw(frame_metric, (metric_size_eff, metric_size_eff))
                    score = _compute_score(
                        metric,
                        frame_metric,
                        target_metric,
                        metric_device,
                        lpips_net,
                        clip_model,
                        clip_pretrained,
                        clip_ctx,
                    )
                    queue.append((idx, frame_t, score))
                    idx += 1
                if not queue:
                    raise RuntimeError("No frames processed from video.")
                best_index, best_frame_tensor, best_score = min(queue, key=lambda x: x[2])
                scores = [{"index": item[0], "score": item[2]} for item in queue]
            else:
                idx = start_idx
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_t = ensure_hwc(_to_tensor(frame_bgr))
                    frame_t = _resize_to_match(frame_t, (h_t, w_t))
                    frame_metric = normalize_to_reference(frame_t, target, normalize) if normalize != "none" else frame_t
                    if metric_size_eff:
                        frame_metric = resize_to_hw(frame_metric, (metric_size_eff, metric_size_eff))
                    score = _compute_score(
                        metric,
                        frame_metric,
                        target_metric,
                        metric_device,
                        lpips_net,
                        clip_model,
                        clip_pretrained,
                        clip_ctx,
                    )
                    scores.append({"index": idx, "score": score})
                    if best_score is None or score < best_score:
                        best_score = score
                        best_index = idx
                        best_frame_tensor = frame_t
                    idx += 1
        finally:
            cap.release()

        if best_frame_tensor is None:
            raise RuntimeError("No frames processed from video.")

        scores_json = json.dumps(
            {"metric": metric, "normalize": normalize, "scores": scores[:500]},
            ensure_ascii=True,
        )

        return (
            best_frame_tensor.unsqueeze(0),
            image_difference(best_frame_tensor, target).unsqueeze(0),
            best_index,
            float(best_score),
            scores_json,
        )


_LOGGER.warning(
    "Loaded VideoFrameMatch. NODE_CLASS_MAPPINGS=%s",
    ["VideoFrameMatch"],
)
