import json
import logging
import os
import subprocess
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


def _to_tensor_rgb(frame_rgb: np.ndarray) -> torch.Tensor:
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


def _ffprobe_frames(video_path: str) -> int:
    probes = [
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nk=1:nw=1",
            video_path,
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nk=1:nw=1",
            video_path,
        ],
    ]
    for cmd in probes:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return 0
        if proc.returncode != 0:
            continue
        out = (proc.stdout or "").strip()
        if out.isdigit():
            val = int(out)
            if val > 0:
                return val
    return 0


def _ffprobe_stream_info(video_path: str) -> Tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "default=nk=1:nw=1",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 0, 0, 0.0
    if proc.returncode != 0:
        return 0, 0, 0.0
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0, 0, 0.0
    try:
        width = int(lines[0])
        height = int(lines[1])
        rate = lines[2]
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(rate)
        return width, height, fps
    except Exception:
        return 0, 0, 0.0


def _iter_ffmpeg_tail_frames(video_path: str, start_time: float, max_frames: int, width: int, height: int):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        video_path,
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it and ensure it is in PATH. "
            "Linux: sudo apt install ffmpeg | Windows: choco install ffmpeg | macOS: brew install ffmpeg"
        )
    if proc.stdout is None:
        return
    frame_bytes = width * height * 3
    while True:
        chunk = proc.stdout.read(frame_bytes)
        if not chunk or len(chunk) < frame_bytes:
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
        yield frame
    proc.stdout.close()
    proc.wait()


def _get_total_frames(cap: cv2.VideoCapture, video_path: str) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        return total
    return _ffprobe_frames(video_path)


def _ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 0.0
    if proc.returncode != 0:
        return 0.0
    out = (proc.stdout or "").strip()
    try:
        val = float(out)
        return val if val > 0 else 0.0
    except Exception:
        return 0.0


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
) -> float:
    if metric == "mse":
        return _mse_score(frame, target)
    if metric == "ssim":
        return _ssim_distance(frame.to(device), target.to(device))
    if metric == "lpips_alex":
        return _lpips_distance(frame, target, "alex", device)
    if metric == "lpips_vgg":
        return _lpips_distance(frame, target, "vgg", device)
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
                "metric": (["mse", "ssim", "lpips_alex", "lpips_vgg"], {"default": "mse", "tooltip": "Метрика сходства кадра и картинки."}),
                "normalize": (["none", "mean_std", "linear", "hist"], {"default": "none", "tooltip": "Нормализация кадра к референсу перед сравнением."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("best_frame", "best_frame_number", "scores_json")
    FUNCTION = "match"
    CATEGORY = "video/utils"

    def match(self, image, video, max_frames, metric, normalize):
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

        use_gpu = torch.cuda.is_available() and metric in {"ssim", "lpips_alex", "lpips_vgg"}
        metric_device = torch.device("cuda" if use_gpu else "cpu")
        target_metric = target

        total_frames = _get_total_frames(cap, video_path)
        use_tail_only = bool(max_frames)
        start_idx = 0
        seek_ok = False

        if use_tail_only and total_frames > 0:
            start_idx = max(0, total_frames - max_frames)
        _LOGGER.info(
            "VideoFrameMatch: total_frames=%s, use_tail_only=%s, start_idx=%s, seek_ok=%s",
            total_frames,
            use_tail_only,
            start_idx,
            seek_ok,
        )

        def _score_frame(frame_bgr, frame_index):
            frame_t = ensure_hwc(_to_tensor(frame_bgr))
            frame_t = _resize_to_match(frame_t, (h_t, w_t))
            frame_metric = normalize_to_reference(frame_t, target, normalize) if normalize != "none" else frame_t
            score_val = _compute_score(
                metric,
                frame_metric,
                target_metric,
                metric_device,
            )
            return frame_t, score_val, {"index": frame_index, "score": score_val}

        pbar = None
        if tqdm is not None and total_frames > 0:
            if use_tail_only:
                pbar_total = max_frames if max_frames else total_frames
            else:
                pbar_total = total_frames
            pbar = tqdm(total=pbar_total, desc="VideoFrameMatch", unit="frame")

        try:
            if use_tail_only:
                if max_frames <= 0:
                    raise RuntimeError("max_frames must be > 0 for tail search.")
                width, height, fps = _ffprobe_stream_info(video_path)
                if width <= 0 or height <= 0 or fps <= 0:
                    raise RuntimeError("ffprobe failed to read video stream info (width/height/fps).")
                duration = _ffprobe_duration(video_path)
                if duration <= 0 and total_frames > 0:
                    duration = total_frames / fps
                if duration <= 0:
                    raise RuntimeError("ffprobe failed to read video duration.")
                start_time = max(0.0, duration - (max_frames / fps))
                if total_frames > 0:
                    start_idx = max(0, total_frames - max_frames)
                _LOGGER.info(
                    "VideoFrameMatch: ffmpeg tail read (start_time=%.3fs, frames=%s)",
                    start_time,
                    max_frames,
                )
                idx = start_idx
                processed = 0
                for frame_rgb in _iter_ffmpeg_tail_frames(video_path, start_time, max_frames, width, height):
                    frame_t = ensure_hwc(_to_tensor_rgb(frame_rgb))
                    frame_t = _resize_to_match(frame_t, (h_t, w_t))
                    frame_metric = (
                        normalize_to_reference(frame_t, target, normalize) if normalize != "none" else frame_t
                    )
                    score_val = _compute_score(
                        metric,
                        frame_metric,
                        target_metric,
                        metric_device,
                    )
                    scores.append({"index": idx, "score": score_val})
                    if best_score is None or score_val < best_score:
                        best_score = score_val
                        best_index = idx
                        best_frame_tensor = frame_t
                    if pbar is not None:
                        pbar.update(1)
                    idx += 1
                    processed += 1
                    if processed >= max_frames:
                        break
            else:
                idx = start_idx
                processed_tail = 0
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
                    if use_tail_only and max_frames:
                        processed_tail += 1
                        if processed_tail >= max_frames:
                            break
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
