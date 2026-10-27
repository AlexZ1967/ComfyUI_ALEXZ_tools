"""Load selected source frames and save final videos using FFmpeg including ProRes."""

import json
import logging
import math
import os
from pathlib import Path
import tempfile

from .video_frame_select import VideoFrameSelect, selection_bounds
from .video_silent_film_finish import _validate_video_batch
from ..utils.video_io import VIDEO_FORMATS, check_interrupt, decode_frames, encode_frames, make_preview, probe_video, resolve_video_path
from ..utils.video_memory_plan import estimate_graph, check_resources
from ..utils.video_cache import enabled as selective_cache_enabled

_LOGGER = logging.getLogger("ALEXZ.VideoIO")


class VideoLoadSource:
    """Read high-bit-depth local video with integrated inclusive frame selection."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose a local file path and the existing frame-selection controls."""
        selection = VideoFrameSelect.INPUT_TYPES()["required"]
        return {"required": {
            "video_path": ("STRING", {"default": "", "tooltip": "Абсолютный путь к локальному видео или имя в ComfyUI/input. Поддерживаются MOV/ProRes, MP4, MXF и другие форматы FFmpeg. Кнопка выбора загружает файл в input; для большого ProRes удобнее указать существующий путь без копирования."}),
            **{name: selection[name] for name in ("selection_mode", "frame_index", "start_frame", "end_frame")},
        }, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "fps", "total_frames", "selected_frames", "video_json", "preview")
    FUNCTION = "load"
    CATEGORY = "video/io"

    @classmethod
    def IS_CHANGED(cls, video_path, **kwargs):
        """Invalidate cached frames when the source file is replaced or modified."""
        path = resolve_video_path(video_path)
        stat = path.stat()
        return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"

    def load(self, video_path, selection_mode="full_video", frame_index=0, start_frame=0, end_frame=-1, prompt=None, unique_id=None):
        """Probe the source then decode only the requested output frames into IMAGE storage."""
        path = resolve_video_path(video_path)
        _LOGGER.info("[Load Source Video] Этап 1/2: информация о файле %s.", path.name)
        info = probe_video(path)
        start, end = selection_bounds(info["total_frames"], selection_mode, frame_index, start_frame, end_frame)
        memory_plan = check_resources(estimate_graph(info, end - start + 1, prompt, unique_id, selective=selective_cache_enabled()))
        _LOGGER.info("[Load Source Video] %s, %s, %dx%d, %.6f fps; всего %d, выбрано %d–%d (%d кадров).", info["codec"], info["pixel_format"], info["width"], info["height"], info["fps"], info["total_frames"], start, end, end - start + 1)
        _LOGGER.info("[Load Source Video] Этап 2/2: потоковое декодирование выбранных кадров в 16-битный RGB/RGBA → float32.")
        image = decode_frames(path, info, start, end)
        count = len(image)
        metadata = {**info, "schema_name": "alexz.video_source", "schema_version": 1,
                    "selection_mode": selection_mode, "start_frame": start, "end_frame": end,
                    "selected_frames": count, "middle_source_frame": start + count // 2,
                    "numbering": "zero_based_inclusive", "timing": "frames_at_source_average_fps", "audio_loaded": False,
                    "memory_plan": memory_plan}
        preview = image[count // 2:count // 2 + 1].clone()
        return {"ui": {"video_info": [metadata]}, "result": (image, info["fps"], info["total_frames"], count, json.dumps(metadata, ensure_ascii=False), preview)}


def _reserve_output(prefix, width, height, extension):
    """Reserve a numbered output filename without overwriting previous exports."""
    import folder_paths
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory(), width, height)
    while True:
        target = Path(folder) / f"{name}_{counter:05d}_.{extension}"
        try:
            with target.open("xb"):
                pass
            return target, subfolder
        except FileExistsError:
            counter += 1


class VideoSaveFinal:
    """Stream a final IMAGE batch to a ProRes MOV or H.264 MP4 with an optional proxy."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose output codec, playback rate, and safe numbered output naming."""
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Готовые кадры из Projector или другой IMAGE-ноды. Читаются по одному, без полной копии GPU/дискового батча в RAM."}),
            "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.1, "max": 240.0, "step": 0.01, "tooltip": "Частота записи готовых кадров. Соедините с Projector.fps; больше — быстрее и короче. Интерполяция и автоматическая компенсация source_fps не выполняются."}),
            "filename_prefix": ("STRING", {"default": "ALEXZ/SilentFilm", "tooltip": "Имя и необязательная подпапка внутри ComfyUI/output. Добавляется уникальный номер; существующие файлы не перезаписываются."}),
            "format": (VIDEO_FORMATS, {"default": "prores_422_hq", "tooltip": "ProRes сохраняется в MOV: Proxy/LT/422/HQ или 4444/XQ. Только 4444 сохраняет альфу; прочие профили и H.264 накладывают RGBA на чёрный. Установленный prores_ks кодирует цвет с точностью 10 бит. H.264 MP4 — компактная 8-битная копия."}),
            "crf": ("INT", {"default": 18, "min": 0, "max": 51, "tooltip": "Только для H.264: меньше — выше качество и размер. На CPU применяется CRF libx264; для CUDA-входа с доступным NVENC — близкий по назначению Constant Quality, шкалы не идентичны. На ProRes не влияет."}),
            "create_preview": ("BOOLEAN", {"default": True, "tooltip": "Для ProRes создаёт отдельную MP4-копию до 960 px для проигрывания в браузере. Мастер MOV не меняется. Выключите, если нужен только мастер. H.264 можно смотреть непосредственно."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "video_json")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "video/io"

    def save(self, image, frame_rate=25.0, filename_prefix="ALEXZ/SilentFilm", format="prores_422_hq", crf=18, create_preview=True):
        """Encode into a temporary file, publish atomically, and expose a playable preview."""
        images = _validate_video_batch(image, normalize=False)
        frame_rate = float(frame_rate)
        if not math.isfinite(frame_rate) or not 0.1 <= frame_rate <= 240:
            raise ValueError("frame_rate должен быть от 0.1 до 240.")
        if format not in VIDEO_FORMATS or type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError("Проверьте format и crf.")
        extension = "mp4" if format == "h264_mp4" else "mov"
        target, subfolder = _reserve_output(filename_prefix, images.size(2), images.size(1), extension)
        reserved_inode = target.stat().st_ino
        temporary = None
        committed = False
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".alexz-video-", suffix="." + extension, delete=False) as file:
                temporary = Path(file.name)
            _LOGGER.info("[Save Final Video] Этап 1/2: %d кадров, %dx%d, %.6f fps, %s; %s.", len(images), images.size(2), images.size(1), frame_rate, format, target)
            encode_frames(images, temporary, frame_rate, format, crf)
            if target.stat().st_ino != reserved_inode or target.stat().st_size != 0:
                raise FileExistsError("Выходной файл изменён другим приложением; перезапись остановлена.")
            os.replace(temporary, target)
            committed = True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not committed and target.exists() and target.stat().st_ino == reserved_inode and target.stat().st_size == 0:
                target.unlink()

        preview = target if extension == "mp4" else None
        preview_error = None
        if extension == "mov" and create_preview:
            _LOGGER.info("[Save Final Video] Этап 2/2: MP4-превью для браузера.")
            preview = target.with_name(target.stem + "_preview.mp4")
            proxy_temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".alexz-preview-", suffix=".mp4", delete=False) as file:
                    proxy_temp = Path(file.name)
                make_preview(target, proxy_temp)
                if preview.exists():
                    raise FileExistsError("Файл MP4-превью уже существует.")
                os.link(proxy_temp, preview)
            except Exception as exc:
                check_interrupt()
                preview, preview_error = None, str(exc)
                _LOGGER.warning("[Save Final Video] Мастер сохранён; MP4-превью не создано: %s", exc)
            finally:
                if proxy_temp is not None:
                    proxy_temp.unlink(missing_ok=True)
        else:
            _LOGGER.info("[Save Final Video] Этап 2/2: завершение без отдельной MP4-копии.")
        metadata = {"schema_name": "alexz.video_export", "schema_version": 1, "path": str(target),
                    "format": format, "fps": frame_rate, "frame_count": len(images), "duration_seconds": len(images) / frame_rate,
                    "width": images.size(2), "height": images.size(1), "preview_path": str(preview) if preview else None, "preview_error": preview_error, "audio_saved": False}
        videos = [{"filename": preview.name, "subfolder": subfolder, "type": "output"}] if preview else []
        files = [{"filename": target.name, "subfolder": subfolder, "type": "output"}]
        return {"ui": {"alexz_videos": videos, "alexz_files": files, "text": [str(target)]}, "result": (str(target), json.dumps(metadata, ensure_ascii=False))}
