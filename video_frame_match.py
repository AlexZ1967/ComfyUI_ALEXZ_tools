import json
import logging
import os
from collections import deque
from typing import Optional, Tuple

import cv2
import folder_paths
import numpy as np
import torch
from .utils import ensure_hwc, image_difference

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


class VideoFrameMatch:
    @classmethod
    def INPUT_TYPES(cls):
        videos = sorted(_list_videos())
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Целевой кадр для поиска в видео."}),
                "video": (videos, {"video_upload": True, "tooltip": "Видео из папки input/."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Количество последних кадров для анализа (0 = все)."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("best_frame", "difference", "best_index", "best_score", "scores_json")
    FUNCTION = "match"
    CATEGORY = "video/utils"

    def match(self, image, video, max_frames):
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
                    score = _mse_score(frame_t, target)
                    queue.append((idx, frame_t, score))
                    idx += 1
                if not queue:
                    raise RuntimeError("No frames processed from video.")
                best_index, best_frame_tensor, best_score = min(queue, key=lambda x: x[2])
                scores = [{"index": item[0], "mse": item[2]} for item in queue]
            else:
                idx = start_idx
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_t = ensure_hwc(_to_tensor(frame_bgr))
                    frame_t = _resize_to_match(frame_t, (h_t, w_t))
                    score = _mse_score(frame_t, target)
                    scores.append({"index": idx, "mse": score})
                    if best_score is None or score < best_score:
                        best_score = score
                        best_index = idx
                        best_frame_tensor = frame_t
                    idx += 1
        finally:
            cap.release()

        if best_frame_tensor is None:
            raise RuntimeError("No frames processed from video.")

        scores_json = json.dumps(scores[:500], ensure_ascii=True)  # cap to avoid huge UI

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
