"""Match Video Cut Point node implementation.

Finds the best frame pair between the tail of one video and the head of another
video to help build seamless continuation edits.
"""

import json

import logging
import os
import subprocess
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

from ..utils.utils import ensure_hwc, normalize_to_reference

_LOGGER = logging.getLogger("VideoCutMatch")
_SSIM_WINDOW_CACHE = {}
_LPIPS_CACHE = {}


def _list_videos():
    """Return available video filenames from the ComfyUI input directory."""
    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    return folder_paths.filter_files_content_types(files, ["video"])


def _to_tensor_rgb(frame_rgb: np.ndarray) -> torch.Tensor:
    """Convert BGR NumPy frame to normalized RGB torch tensor (BCHW)."""
    return torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)


def _resize_to_match(frame: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize source tensor to match reference spatial resolution."""
    h, w = target_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    out = F.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def _downscale_max_side(frame: torch.Tensor, max_side: int) -> torch.Tensor:
    """Downscale an image tensor so its longest side does not exceed the given limit."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = float(max_side) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    out = F.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def _mse_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean-squared error distance between two tensors."""
    return float(torch.mean((a - b) ** 2).item())


def _get_ssim_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5):
    """Create a cached Gaussian window tensor used by SSIM."""
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
    """Compute SSIM-based distance where lower values are better matches."""
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    channels = a.shape[1]
    window = _get_ssim_window(channels, a.device, a.dtype)
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


def _lpips_model(net: str, device: torch.device):
    """Load and cache LPIPS model weights for perceptual scoring."""
    key = (net, device.type)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "lpips is required for metric=lpips_alex/lpips_vgg. Install: pip install lpips"
        ) from exc
    model = lpips.LPIPS(net=net).to(device).eval()
    _LPIPS_CACHE[key] = model
    return model


def _lpips_distance(a: torch.Tensor, b: torch.Tensor, net: str, device: torch.device) -> float:
    """Compute LPIPS perceptual distance for two images."""
    model = _lpips_model(net, device)
    aa = a.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    bb = b.permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.inference_mode():
        dist = model(aa, bb)
    return float(dist.item())


def _compute_score(metric: str, a: torch.Tensor, b: torch.Tensor, device: torch.device) -> float:
    """Compute similarity score for the selected metric (MSE, SSIM, or LPIPS variants)."""
    if metric == "mse":
        return _mse_score(a, b)
    if metric == "ssim":
        return _ssim_distance(a.to(device), b.to(device))
    if metric == "lpips_alex":
        return _lpips_distance(a, b, "alex", device)
    if metric == "lpips_vgg":
        return _lpips_distance(a, b, "vgg", device)
    raise ValueError(f"Unknown metric: {metric}")


def _update_top_pairs(top_pairs, item, limit: int):
    """Insert candidate pair into fixed-size sorted list of best cut matches."""
    if len(top_pairs) < limit:
        top_pairs.append(item)
        top_pairs.sort(key=lambda x: x["score"])
        return
    if item["score"] < top_pairs[-1]["score"]:
        top_pairs[-1] = item
        top_pairs.sort(key=lambda x: x["score"])


def _confidence_from_top_pairs(top_pairs) -> float:
    """Estimate confidence from the spread between the best and next-best pair scores."""
    if len(top_pairs) < 2:
        return 1.0 if len(top_pairs) == 1 else 0.0
    best = float(top_pairs[0]["score"])
    second = float(top_pairs[1]["score"])
    scale = max(abs(second), abs(best), 1e-6)
    conf = (second - best) / scale
    return float(max(0.0, min(1.0, conf)))


def _blend_window_from_confidence(confidence: float) -> int:
    """Map confidence value to a recommended blend-window size."""
    if confidence >= 0.25:
        return 4
    if confidence >= 0.12:
        return 8
    return 12


def _ffprobe_frames(video_path: str) -> int:
    """Read frame count metadata from ffprobe output."""
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
    """Read video stream width/height/fps metadata using ffprobe."""
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


def _ffprobe_duration(video_path: str) -> float:
    """Read video duration metadata using ffprobe."""
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


def _iter_ffmpeg_tail_frames(video_path: str, start_time: float, max_frames: int, width: int, height: int):
    """Stream only tail frames from a video via ffmpeg stdout."""
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


def _load_head_frames(video_path: str, max_frames: int):
    """Load first N frames from video and convert them to RGB tensors."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    idx = 0
    pbar = tqdm(desc="VideoCutMatch B head", unit="frame") if tqdm is not None else None
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append({"index": idx, "rgb": frame_rgb})
            idx += 1
            if pbar is not None:
                pbar.update(1)
            if max_frames > 0 and idx >= max_frames:
                break
    finally:
        if pbar is not None:
            pbar.close()
        cap.release()
    return frames


def _load_tail_frames(video_path: str, max_frames: int):
    """Load last N frames from video and convert them to RGB tensors."""
    if max_frames <= 0:
        return _load_head_frames(video_path, 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        total = _ffprobe_frames(video_path)

    width, height, fps = _ffprobe_stream_info(video_path)
    duration = _ffprobe_duration(video_path)
    if duration <= 0 and total > 0 and fps > 0:
        duration = total / fps

    if total > 0 and width > 0 and height > 0 and fps > 0 and duration > 0:
        start_idx = max(0, total - max_frames)
        start_time = max(0.0, duration - (max_frames / fps))
        frames = []
        idx = start_idx
        pbar = tqdm(desc="VideoCutMatch A tail", total=max_frames, unit="frame") if tqdm is not None else None
        try:
            for frame_rgb in _iter_ffmpeg_tail_frames(video_path, start_time, max_frames, width, height):
                frames.append({"index": idx, "rgb": frame_rgb})
                idx += 1
                if pbar is not None:
                    pbar.update(1)
                if len(frames) >= max_frames:
                    break
        finally:
            if pbar is not None:
                pbar.close()
        return frames

    _LOGGER.warning("VideoCutMatch: fallback to full scan for A tail (metadata unavailable)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    queue = []
    idx = 0
    pbar = tqdm(desc="VideoCutMatch A tail fallback", unit="frame") if tqdm is not None else None
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            queue.append({"index": idx, "rgb": frame_rgb})
            idx += 1
            if len(queue) > max_frames:
                queue.pop(0)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        cap.release()
    return queue


class VideoCutMatch:
    """ComfyUI node that finds the best cut point between two videos."""
    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        videos = sorted(_list_videos())
        return {
            "required": {
                "video_a": (
                    videos,
                    {"tooltip": "Первое видео (оригинал). Поиск идёт в его хвосте."},
                ),
                "video_b": (
                    videos,
                    {"tooltip": "Второе видео (продолжение). Поиск идёт в его начале."},
                ),
                "search_tail_a": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 200000,
                        "tooltip": "Сколько последних кадров брать из video_a (0 = всё видео).",
                    },
                ),
                "search_head_b": (
                    "INT",
                    {
                        "default": 120,
                        "min": 0,
                        "max": 200000,
                        "tooltip": "Сколько первых кадров брать из video_b (0 = всё видео).",
                    },
                ),
                "metric": (
                    ["mse", "ssim", "lpips_alex", "lpips_vgg"],
                    {
                        "default": "mse",
                        "tooltip": "Метрика сходства: mse быстрее, ssim устойчивее к свету, lpips_* качественнее и медленнее.",
                    },
                ),
                "normalize": (
                    ["none", "mean_std", "linear", "hist"],
                    {
                        "default": "none",
                        "tooltip": "Нормализация кадра A к кадру B перед сравнением. none быстрее, hist медленнее.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 20,
                        "tooltip": "Сколько лучших пар кадров вернуть в match_json.",
                    },
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "best_frame_a",
        "best_frame_b",
        "best_frame_a_number",
        "best_frame_b_number",
        "match_json",
    )
    FUNCTION = "match"
    CATEGORY = "video/utils"

    def match(self, video_a, video_b, search_tail_a, search_head_b, metric, normalize, top_k):
        """Execute the node and return processed outputs for ComfyUI."""
        path_a = folder_paths.get_annotated_filepath(video_a)
        path_b = folder_paths.get_annotated_filepath(video_b)
        if not os.path.exists(path_a):
            raise ValueError(f"Video A not found: {video_a}")
        if not os.path.exists(path_b):
            raise ValueError(f"Video B not found: {video_b}")

        frames_a = _load_tail_frames(path_a, int(search_tail_a))
        frames_b = _load_head_frames(path_b, int(search_head_b))
        if not frames_a:
            raise RuntimeError("No frames loaded from video_a.")
        if not frames_b:
            raise RuntimeError("No frames loaded from video_b.")

        target_hw = (frames_a[0]["rgb"].shape[0], frames_a[0]["rgb"].shape[1])
        a_metric = []
        b_metric = []
        for item in frames_a:
            t = ensure_hwc(_to_tensor_rgb(item["rgb"]))
            a_metric.append(_resize_to_match(t, target_hw))
        for item in frames_b:
            t = ensure_hwc(_to_tensor_rgb(item["rgb"]))
            b_metric.append(_resize_to_match(t, target_hw))

        pairs_count = len(frames_a) * len(frames_b)
        if pairs_count > 1_000_000:
            raise RuntimeError(
                f"Too many frame pairs ({pairs_count}). Reduce search_tail_a/search_head_b for practical runtime."
            )

        device = torch.device("cuda" if torch.cuda.is_available() and metric in {"ssim", "lpips_alex", "lpips_vgg"} else "cpu")
        lpips_mode = metric in {"lpips_alex", "lpips_vgg"}
        candidate_limit = max(24, int(top_k) * 4)
        top_pairs = []

        if lpips_mode and pairs_count > 300_000:
            raise RuntimeError(
                f"LPIPS mode is too expensive for {pairs_count} pairs. Reduce search ranges."
            )

        strategy = "single_pass"
        if lpips_mode:
            strategy = "two_pass_lpips"
            coarse_a = [_downscale_max_side(x, 192) for x in a_metric]
            coarse_b = [_downscale_max_side(x, 192) for x in b_metric]
            coarse_top = []
            pbar = tqdm(total=pairs_count, desc="VideoCutMatch coarse", unit="pair") if tqdm is not None else None
            try:
                for i, fa in enumerate(coarse_a):
                    for j, fb in enumerate(coarse_b):
                        score = _mse_score(fa, fb)
                        _update_top_pairs(
                            coarse_top,
                            {"i": i, "j": j, "score": score},
                            candidate_limit,
                        )
                        if pbar is not None:
                            pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()

            pbar = tqdm(total=len(coarse_top), desc="VideoCutMatch refine", unit="pair") if tqdm is not None else None
            refined_scores = []
            try:
                for item in coarse_top:
                    i = item["i"]
                    j = item["j"]
                    fa = a_metric[i]
                    fb = b_metric[j]
                    if normalize != "none":
                        fa = normalize_to_reference(fa, fb, normalize)
                    score = _compute_score(metric, fa, fb, device)
                    pair = {
                        "frame_a_number": int(frames_a[i]["index"]),
                        "frame_b_number": int(frames_b[j]["index"]),
                        "score": float(score),
                        "coarse_score": float(item["score"]),
                    }
                    refined_scores.append(pair)
                    _update_top_pairs(top_pairs, pair, int(top_k))
                    if pbar is not None:
                        pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()
        else:
            pbar = tqdm(total=pairs_count, desc="VideoCutMatch", unit="pair") if tqdm is not None else None
            try:
                for i, fa_raw in enumerate(a_metric):
                    for j, fb in enumerate(b_metric):
                        fa = normalize_to_reference(fa_raw, fb, normalize) if normalize != "none" else fa_raw
                        score = _compute_score(metric, fa, fb, device)
                        pair = {
                            "frame_a_number": int(frames_a[i]["index"]),
                            "frame_b_number": int(frames_b[j]["index"]),
                            "score": float(score),
                        }
                        _update_top_pairs(top_pairs, pair, int(top_k))
                        if pbar is not None:
                            pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()
            refined_scores = []

        if not top_pairs:
            raise RuntimeError("Failed to match frames between videos.")

        best = top_pairs[0]
        best_a_idx = next(i for i, it in enumerate(frames_a) if it["index"] == best["frame_a_number"])
        best_b_idx = next(i for i, it in enumerate(frames_b) if it["index"] == best["frame_b_number"])
        best_a = _to_tensor_rgb(frames_a[best_a_idx]["rgb"]).unsqueeze(0)
        best_b = _to_tensor_rgb(frames_b[best_b_idx]["rgb"]).unsqueeze(0)

        confidence = _confidence_from_top_pairs(top_pairs)
        blend_window = _blend_window_from_confidence(confidence)
        payload = {
            "status": "ok",
            "metric": metric,
            "normalize": normalize,
            "strategy": strategy,
            "frames_a_considered": len(frames_a),
            "frames_b_considered": len(frames_b),
            "pairs_evaluated": pairs_count,
            "best": {
                "frame_a_number": int(best["frame_a_number"]),
                "frame_b_number": int(best["frame_b_number"]),
                "score": float(best["score"]),
                "confidence": float(confidence),
            },
            "top_k": top_pairs,
            "cut_hint": {
                "cut_at_a": int(best["frame_a_number"]),
                "cut_from_b": int(best["frame_b_number"]),
                "blend_window_frames": int(blend_window),
            },
            "search": {
                "tail_a": int(search_tail_a),
                "head_b": int(search_head_b),
            },
        }
        if lpips_mode:
            payload["coarse_metric"] = "mse"
            payload["coarse_max_side"] = 192
            payload["refine_candidates"] = len(refined_scores)
            payload["refined_scores"] = refined_scores[: candidate_limit]

        return (
            best_a,
            best_b,
            int(best["frame_a_number"]),
            int(best["frame_b_number"]),
            json.dumps(payload, ensure_ascii=True),
        )
