"""Stream high-bit-depth video through the locally installed FFmpeg tools."""

from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
from tqdm import tqdm

from .video_memory import allocate_video, release_video_pages

PRORES_PROFILES = {"prores_422_proxy": 0, "prores_422_lt": 1, "prores_422": 2,
                   "prores_422_hq": 3, "prores_4444": 4, "prores_4444_xq": 5}
VIDEO_FORMATS = ["prores_422_hq", "prores_422", "prores_422_lt", "prores_422_proxy", "prores_4444", "prores_4444_xq", "h264_mp4"]
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".mxf", ".mpg", ".mpeg", ".ts"}


def tool(name):
    """Find the executable in the active environment without installing dependencies."""
    value = shutil.which(name)
    if not value:
        raise RuntimeError(f"Не найден {name}. Установите FFmpeg с ffprobe в рабочее окружение ComfyUI.")
    return value


def resolve_video_path(value):
    """Accept a local absolute file or a ComfyUI input-relative annotated path."""
    import folder_paths
    value = str(value).strip().strip('"')
    if not value:
        raise ValueError("Выберите видео или укажите путь к файлу.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(folder_paths.get_annotated_filepath(str(path)))
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Укажите локальный видеофайл, например MOV/ProRes, MP4 или MXF.")
    return path


def _rate(value):
    """Parse a positive finite frame rate without treating N/A as valid timing."""
    try:
        result = float(Fraction(str(value)))
        return result if math.isfinite(result) and result > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe_command(path, count=False):
    """Read stream metadata, optionally counting decoded frames without storing images."""
    command = [tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe,crypto,data", "-select_streams", "v:0"]
    if count:
        command += ["-count_frames"]
    command += ["-show_entries", "stream=width,height,codec_name,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration,color_space,color_transfer,color_primaries,color_range,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation:format=duration", "-of", "json", str(path)]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError("ffprobe: " + result.stderr.decode("utf-8", "replace")[-4000:])
    return json.loads(result.stdout)


@lru_cache(maxsize=32)
def _probe_cached(path, size, modified_ns, exact):
    """Cache metadata by file identity rather than by a stale workflow filename."""
    data = _probe_command(path)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("В файле не найден видеопоток.")
    stream = streams[0]
    try:
        count = int(stream.get("nb_frames", 0))
    except (TypeError, ValueError):
        count = 0
    count_source = "header"
    if not count and exact:
        counted = _probe_command(path, count=True)["streams"][0]
        try:
            count = int(counted.get("nb_read_frames", 0))
        except (TypeError, ValueError):
            count = 0
        count_source = "decoded_count"
        if count < 1:
            raise ValueError("Не удалось точно посчитать кадры. Проверьте целостность файла и поддержку декодера FFmpeg.")
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError("Не удалось определить частоту кадров видео.")
    duration = _rate(stream.get("duration")) or _rate(data.get("format", {}).get("duration"))
    if not count and duration:
        count, count_source = round(duration * fps), "estimated"
    if count < 1:
        raise ValueError("Не удалось определить число кадров видео.")
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    if not math.isclose(rotation % 90, 0, abs_tol=0.01):
        raise ValueError("Поддерживается поворот видео на 0, 90, 180 или 270 градусов.")
    width, height = int(stream["width"]), int(stream["height"])
    if width < 1 or height < 1:
        raise ValueError("Некорректный размер видеокадра.")
    if round(rotation / 90) % 2:
        width, height = height, width
    pixel_format = stream.get("pix_fmt", "")
    alpha = pixel_format.startswith(("yuva", "gbrap", "rgba", "bgra", "argb", "abgr", "ya")) or pixel_format == "pal8"
    return {"path": str(path), "total_frames": count, "count_source": count_source,
            "count_is_exact": count_source != "estimated", "fps": fps, "duration_seconds": duration,
            "width": width, "height": height, "channels": 4 if alpha else 3,
            "codec": stream.get("codec_name", ""), "pixel_format": pixel_format,
            "rotation": rotation, "color_space": stream.get("color_space", "unknown"),
            "color_range": stream.get("color_range", "unknown"),
            "color_transfer": stream.get("color_transfer", "unknown"), "sample_aspect_ratio": stream.get("sample_aspect_ratio", "1:1")}


def probe_video(path, exact=True):
    """Return independent metadata so callers cannot mutate the cached probe."""
    path = Path(path)
    stat = path.stat()
    return dict(_probe_cached(str(path), stat.st_size, stat.st_mtime_ns, bool(exact)))


@contextmanager
def ffmpeg_process(arguments, *, writing=False):
    """Drain diagnostics to a file and always terminate children on errors or cancellation."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
                                   stdin=subprocess.PIPE if writing else subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL if writing else subprocess.PIPE, stderr=errors)
        try:
            yield process
            if writing:
                process.stdin.close()
            deadline = time.monotonic() + 300
            while True:
                check_interrupt()
                try:
                    code = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() > deadline:
                        raise TimeoutError("FFmpeg не завершился за 300 секунд.")
            if code:
                errors.seek(0)
                raise RuntimeError("FFmpeg: " + errors.read().decode("utf-8", "replace")[-4000:])
        except BaseException as exc:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if isinstance(exc, (BrokenPipeError, EOFError)):
                errors.seek(0)
                raise RuntimeError("FFmpeg: " + (errors.read().decode("utf-8", "replace")[-4000:] or str(exc))) from exc
            raise
        finally:
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


def _read_frame(pipe, size):
    """Read exactly one raw frame even when pipe reads return partial buffers."""
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = pipe.readinto(view[offset:])
        if not count:
            raise EOFError("Видео закончилось раньше выбранного кадра.")
        offset += count
    return data


def check_interrupt():
    """Respect the installed ComfyUI interrupt hook when running inside its executor."""
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def decode_frames(path, info, start, end):
    """Decode only the selected output frames into a bounded ordinary IMAGE destination."""
    count, height, width, channels = end - start + 1, info["height"], info["width"], info["channels"]
    output = allocate_video((count, height, width, channels), label="Load Source Video")
    pixel_format = "rgba64le" if channels == 4 else "rgb48le"
    matrix = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020", "smpte240m": "smpte240m"}.get(info.get("color_space"))
    conversion = "scale=out_range=full"
    if matrix:
        conversion += f":in_color_matrix={matrix}"
    if info.get("color_range") in ("pc", "tv"):
        conversion += f":in_range={info['color_range']}"
    arguments = ["-threads", "2", "-protocol_whitelist", "file,pipe,crypto,data", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                 "-vf", f"trim=start_frame={start}:end_frame={end + 1},setpts=PTS-STARTPTS,{conversion}", "-fps_mode", "passthrough",
                 "-frames:v", str(count), "-pix_fmt", pixel_format, "-threads", "2", "-f", "rawvideo", "pipe:1"]
    with ffmpeg_process(arguments) as process:
        for index in tqdm(range(count), desc="[Load Source Video] Декодирование", unit="кадр", dynamic_ncols=True):
            check_interrupt()
            raw = _read_frame(process.stdout, height * width * channels * 2)
            array = np.frombuffer(raw, dtype="<u2").reshape(height, width, channels).astype(np.float32)
            output[index].copy_(torch.from_numpy(array).mul_(1.0 / 65535.0))
            release_video_pages(output, index, index + 1, written=True)
    return output


def encode_frames(images, path, fps, video_format, crf=18):
    """Write one CPU frame at a time without copying an entire CUDA or mapped batch."""
    if video_format not in VIDEO_FORMATS:
        raise ValueError("Неизвестный формат видео.")
    height, width, channels = images.shape[1:]
    if video_format == "h264_mp4" and (height % 2 or width % 2):
        raise ValueError("Для H.264 4:2:0 нужны чётные ширина и высота. Используйте ProRes или измените размер кадра.")
    preserve_alpha = video_format in ("prores_4444", "prores_4444_xq") and channels == 4
    pixel_format = "rgba64le" if preserve_alpha else "rgb48le"
    rate = str(Fraction(str(float(fps))).limit_denominator(1001000))
    arguments = ["-y", "-f", "rawvideo", "-pixel_format", pixel_format, "-video_size", f"{width}x{height}", "-framerate", rate, "-i", "pipe:0", "-an"]
    if video_format == "h264_mp4":
        arguments += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-f", "mp4"]
    else:
        arguments += ["-c:v", "prores_ks", "-profile:v", str(PRORES_PROFILES[video_format]), "-pix_fmt", "yuva444p10le" if preserve_alpha else "yuv444p10le" if PRORES_PROFILES[video_format] >= 4 else "yuv422p10le", "-vendor", "apl0", "-f", "mov"]
    arguments += ["-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709", "-threads", "4", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", str(path)]
    with ffmpeg_process(arguments, writing=True) as process:
        for index in tqdm(range(len(images)), desc="[Save Final Video] Кодирование", unit="кадр", dynamic_ncols=True):
            check_interrupt()
            frame = images[index].detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(frame).all():
                raise ValueError(f"Кадр {index} содержит NaN или Infinity.")
            frame = frame.clamp(0, 1)
            if channels == 4 and not preserve_alpha:
                frame = frame[..., :3] * frame[..., 3:4]
            pixels = np.rint(frame.numpy() * 65535.0).astype("<u2")
            process.stdin.write(pixels.tobytes())
            release_video_pages(images, index, index + 1)


def make_preview(path, target):
    """Create a browser-playable MP4 proxy without changing the ProRes master file."""
    arguments = ["-y", "-threads", "2", "-i", str(path), "-map", "0:v:0", "-an", "-vf", "scale=trunc(min(960\\,iw)/2)*2:-2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(target)]
    with ffmpeg_process(arguments, writing=True):
        pass
