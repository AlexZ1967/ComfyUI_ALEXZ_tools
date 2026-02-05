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
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional
    tqdm = None
from .utils import ensure_hwc, normalize_to_reference

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


def _compute_score(
    metric: str,
    frame: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    lpips_net: str,
) -> float:
    if metric == "mse":
        return _mse_score(frame, target)
    if metric == "ssim":
        return _ssim_distance(frame.to(device), target.to(device))
    if metric == "lpips":
        return _lpips_distance(frame, target, lpips_net, device)
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
                "metric": (["mse", "ssim", "lpips"], {"default": "mse", "tooltip": "Метрика сходства кадра и картинки."}),
                "normalize": (["none", "mean_std", "linear", "hist"], {"default": "none", "tooltip": "Нормализация кадра к референсу перед сравнением."}),
                "lpips_net": (["vgg", "alex"], {"default": "vgg", "tooltip": "Бэкбон LPIPS (vgg качественнее, alex быстрее)."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("best_frame", "best_frame_number", "scores_json")
    FUNCTION = "match"
    CATEGORY = "video/utils"

    def match(self, image, video, max_frames, metric, normalize, lpips_net):
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

        use_gpu = torch.cuda.is_available() and metric in {"ssim", "lpips"}
        metric_device = torch.device("cuda" if use_gpu else "cpu")
        target_metric = target

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

        def _score_frame(frame_bgr, frame_index):
            frame_t = ensure_hwc(_to_tensor(frame_bgr))
            frame_t = _resize_to_match(frame_t, (h_t, w_t))
            frame_metric = normalize_to_reference(frame_t, target, normalize) if normalize != "none" else frame_t
            score_val = _compute_score(
                metric,
                frame_metric,
                target_metric,
                metric_device,
                lpips_net,
            )
            return frame_t, score_val, {"index": frame_index, "score": score_val}

        pbar = None
        if tqdm is not None and total_frames > 0:
            if use_tail_only:
                pbar_total = min(max_frames, total_frames) if max_frames else total_frames
            else:
                pbar_total = total_frames
            pbar = tqdm(total=pbar_total, desc="VideoFrameMatch", unit="frame")

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
                    frame_t, score_val, score_item = _score_frame(frame_bgr, idx)
                    queue.append((idx, frame_t, score_val, score_item))
                    if pbar is not None:
                        pbar.update(1)
                    idx += 1
                if not queue:
                    raise RuntimeError("No frames processed from video.")
                best_index, best_frame_tensor, best_score, _ = min(queue, key=lambda x: x[2])
                scores = [item[3] for item in queue]
            else:
                idx = start_idx
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_t, score_val, score_item = _score_frame(frame_bgr, idx)
                    scores.append(score_item)
                    if best_score is None or score_val < best_score:
                        best_score = score_val
                        best_index = idx
                        best_frame_tensor = frame_t
                    if pbar is not None:
                        pbar.update(1)
                    idx += 1
                if best_frame_tensor is None and use_tail_only:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    queue = deque(maxlen=max_frames)
                    idx = 0
                    while True:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        frame_t, score_val, score_item = _score_frame(frame_bgr, idx)
                        queue.append((idx, frame_t, score_val, score_item))
                        if pbar is not None:
                            pbar.update(1)
                        idx += 1
                    if queue:
                        best_index, best_frame_tensor, best_score, _ = min(queue, key=lambda x: x[2])
                        scores = [item[3] for item in queue]
        finally:
            if pbar is not None:
                pbar.close()
            cap.release()

        if best_frame_tensor is None:
            raise RuntimeError("No frames processed from video.")

        scores_json = json.dumps(
            {"metric": metric, "normalize": normalize, "scores": scores[:500]},
            ensure_ascii=True,
        )

        return (
            best_frame_tensor.unsqueeze(0),
            best_index,
            scores_json,
        )


_LOGGER.warning(
    "Loaded VideoFrameMatch. NODE_CLASS_MAPPINGS=%s",
    ["VideoFrameMatch"],
)
