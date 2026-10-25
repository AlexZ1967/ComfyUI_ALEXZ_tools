"""Expose local video probing without queueing or decoding the processing graph."""

import asyncio
import ipaddress
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit

from .video_io import VIDEO_SUFFIXES, probe_video, resolve_video_path


def _local_request(request):
    """Restrict file inspection and upload to same-origin loopback browser requests."""
    try:
        if urlsplit("//" + request.host).hostname not in {"localhost", "127.0.0.1", "::1"} or not ipaddress.ip_address(request.remote).is_loopback:
            return False
        if request.headers.get("Origin", f"{request.scheme}://{request.host}") != f"{request.scheme}://{request.host}":
            return False
        return request.method == "GET" or (request.content_type == "application/octet-stream" and request.headers.get("X-Alexz-Video") == "1")
    except (TypeError, ValueError):
        return False


def register_routes(PromptServer, web):
    """Register a same-origin loopback endpoint for file metadata and frame counts."""
    @PromptServer.instance.routes.get("/alexz/video/probe")
    async def video_probe(request):
        """Probe a local video off the server loop, without executing any workflow node."""
        if not _local_request(request):
            return web.json_response({"error": "Информация о видео доступна локальному same-origin клиенту."}, status=403)
        try:
            def read():
                """Resolve and inspect the local source inside the worker thread."""
                return probe_video(resolve_video_path(request.query.get("path", "")))
            return web.json_response(await asyncio.to_thread(read))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/alexz/video/upload")
    async def video_upload(request):
        """Stream a large browser file into input with bounded RAM and atomic publication."""
        import folder_paths
        if not _local_request(request):
            return web.json_response({"error": "Загрузка разрешена локальному same-origin клиенту."}, status=403)
        temporary = None
        try:
            name = request.query.get("filename", "")
            if not name or name.startswith(".") or any(c in name for c in "/\\:") or any(ord(c) < 32 for c in name) or len(name.encode("utf-8")) > 240 or Path(name).suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("Некорректное имя видеофайла.")
            directory = Path(folder_paths.get_input_directory()) / "alexz_video"
            if directory.is_symlink():
                raise ValueError("Папка загрузки не должна быть символической ссылкой.")
            directory.mkdir(parents=True, exist_ok=True)
            reserve = 512 * 1024**2
            if shutil.disk_usage(directory).free < (request.content_length or 0) + reserve:
                raise ValueError("Недостаточно места на диске для загрузки видео.")
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".upload-", suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                size = 0
                async for chunk in request.content.iter_chunked(1024**2):
                    await asyncio.to_thread(file.write, chunk)
                    size += len(chunk)
                    if size % (64 * 1024**2) < len(chunk) and shutil.disk_usage(directory).free < reserve:
                        raise ValueError("На диске закончился запас места для загрузки.")
                if not size or (request.content_length is not None and size != request.content_length):
                    raise ValueError("Видео загружено не полностью.")
                await asyncio.to_thread(file.flush)
                await asyncio.to_thread(os.fsync, file.fileno())
            suffix, stem, counter = Path(name).suffix, Path(name).stem, 0
            while True:
                filename = name if counter == 0 else f"{stem}_{counter}{suffix}"
                try:
                    os.link(temporary, directory / filename)
                    break
                except FileExistsError:
                    counter += 1
            return web.json_response({"name": filename, "subfolder": "alexz_video", "type": "input"})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
