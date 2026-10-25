"""Select an entire video batch, one frame, or an inclusive frame range."""

import json

from .video_silent_film_finish import _validate_video_batch


def selection_bounds(total, selection_mode, frame_index=0, start_frame=0, end_frame=-1):
    """Resolve the shared zero-based inclusive selection used by loader and selector."""
    if total < 1:
        raise ValueError("Видео не содержит кадров.")
    if selection_mode not in ("full_video", "single_frame", "frame_range"):
        raise ValueError("Unknown selection_mode.")

    def index(name, value):
        """Reject fractional and out-of-range frame indices."""
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < total:
            raise ValueError(f"{name} must be an integer between 0 and {total - 1}.")
        return value

    if selection_mode == "full_video":
        return 0, total - 1
    if selection_mode == "single_frame":
        frame = index("frame_index", frame_index)
        return frame, frame
    start = index("start_frame", start_frame)
    end = total - 1 if end_frame == -1 and type(end_frame) is int else index("end_frame", end_frame)
    if end < start:
        raise ValueError("end_frame must be greater than or equal to start_frame.")
    return start, end


class VideoFrameSelect:
    """Select source frames before processing and report source/selection counts."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose zero-based selection controls independently of film processing."""
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Весь декодированный ролик из Video Load. Нода выбирает кадры до Camera; верхняя кнопка Load and calculate frame загружает и считает кадры без запуска Camera и остальных этапов."}),
            "selection_mode": (["full_video", "single_frame", "frame_range"], {"default": "full_video", "tooltip": "full_video — весь входной батч; single_frame — только frame_index; frame_range — от start_frame до end_frame включительно. Этапы Silent Film обрабатывают только выбранные кадры."}),
            "frame_index": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Номер отдельного кадра с нуля, только для single_frame. При 241 кадре доступны 0–240; средний — 120. Camera автоматически обработает один кадр статически, без временного размытия и движения."}),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Первый кадр диапазона, с нуля и включительно. Учитывается только в frame_range. Например, 24–47 — это 24 кадра, а не 23."}),
            "end_frame": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "Последний кадр диапазона включительно. -1 означает до конца ролика. Должен быть не меньше start_frame и находиться внутри батча. Одинаковые начало и конец выбирают один кадр."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "total_frames", "selected_frames", "selection_json")
    OUTPUT_NODE = True
    FUNCTION = "select"
    CATEGORY = "video/edit"

    def select(self, image, selection_mode="full_video", frame_index=0, start_frame=0, end_frame=-1):
        """Reuse full batches or copy smaller selections without resampling or color changes."""
        images = _validate_video_batch(image, normalize=False)
        total = len(images)
        start, end = selection_bounds(total, selection_mode, frame_index, start_frame, end_frame)
        selected = images if start == 0 and end == total - 1 else images[start:end + 1].clone()
        count = len(selected)
        data = {"schema_name": "alexz.video_frame_selection", "schema_version": 1, "selection_mode": selection_mode,
                "total_frames": total, "selected_frames": count, "start_frame": start, "end_frame": end,
                "middle_source_frame": start + count // 2, "numbering": "zero_based_inclusive"}
        text = f"В ролике: {total} кадров (0–{total - 1}).\nВыбрано: {start}–{end}, кадров: {count}.\nСередина выбранного отрезка: {data['middle_source_frame']}."
        return {"ui": {"text": [text], "video_info": [{"total_frames": total, "width": images.size(2), "height": images.size(1)}]}, "result": (selected, total, count, json.dumps(data, ensure_ascii=False))}
