"""Allocate cached IMAGE tensors across VRAM, RAM and disk without changing tensor contracts."""

import logging
import errno
import math
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import weakref

import torch

_LOGGER = logging.getLogger("SilentFilmMemory")
DISK_THRESHOLD = 256 * 1024**2
RAM_RESERVE = 512 * 1024**2
HOST_RESERVE = 4 * 1024**3
CACHE_DIRECTORY = Path(os.environ.get("ALEXZ_VIDEO_CACHE_DIR", str(
    (Path("/mnt/work") if Path("/mnt/work").is_dir() else Path(tempfile.gettempdir())) / "alexz-video-cache"))).expanduser()
_MAPPINGS = {}


def available_ram():
    """Read cgroup-aware available RAM using the installed ComfyUI when possible."""
    try:
        from comfy.system_memory import virtual_memory_available
    except ImportError:
        import psutil
        return int(psutil.virtual_memory().available)
    return int(virtual_memory_available())


def video_bytes(shape, dtype=torch.float32):
    """Return the exact uncompressed tensor payload size."""
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def use_disk(shape, dtype=torch.float32, ram=None, work_bytes=0):
    """Use host RAM while preserving system headroom and the planned working portion."""
    size = video_bytes(shape, dtype)
    return size + max(HOST_RESERVE, work_bytes + RAM_RESERVE) > (available_ram() if ram is None else ram)


def available_vram(device):
    """Include unused PyTorch allocator blocks in the reusable CUDA budget."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def _forget_mapping(address, reference):
    """Avoid retaining mappings after the last tensor or storage alias is released."""
    if _MAPPINGS.get(address) is reference:
        _MAPPINGS.pop(address, None)


def _reserve_file(file, size):
    """Reserve non-sparse disk blocks, using bounded writes if fallocate is unavailable."""
    reserve = getattr(os, "posix_fallocate", None)
    if reserve is not None:
        try:
            reserve(file.fileno(), 0, size)
            return
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
                raise
    block = bytes(min(size, 8 * 1024**2))
    remaining = size
    while remaining:
        remaining -= file.write(block[:min(remaining, len(block))])
    file.flush()


def allocate_video(shape, *, dtype=torch.float32, device="cpu", label="Silent Film", work_device=None, work_bytes=0):
    """Prefer spare VRAM, then available host RAM, reserving disk only for the remainder."""
    device = torch.device(device)
    size = video_bytes(shape, dtype)
    target = torch.device(work_device) if work_device is not None else device
    if target.type == "cuda" and (size >= DISK_THRESHOLD or device.type == "cuda"):
        if available_vram(target) >= size + work_bytes + RAM_RESERVE:
            try:
                output = torch.empty(shape, dtype=dtype, device=target)
                _LOGGER.info("[%s] Выход: %.2f ГиБ в VRAM; рабочий запас %.2f ГиБ сохранён.", label, size / 1024**3, (work_bytes + RAM_RESERVE) / 1024**3)
                return output
            except torch.OutOfMemoryError:
                _LOGGER.warning("[%s] VRAM изменилась при выделении выхода: проверяется RAM и дисковый кэш.", label)
    if not use_disk(shape, dtype, work_bytes=work_bytes):
        output = torch.empty(shape, dtype=dtype, device="cpu")
        if size >= DISK_THRESHOLD:
            _LOGGER.info("[%s] Выход: %.2f ГиБ в RAM; запас для системы и расчёта %.2f ГиБ.",
                         label, size / 1024**3, max(HOST_RESERVE, work_bytes + RAM_RESERVE) / 1024**3)
        return output
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(CACHE_DIRECTORY).free
    if free < size + RAM_RESERVE:
        raise MemoryError(f"{label}: дисковому кэшу нужно {size / 1024**3:.2f} ГиБ плюс 0.5 ГиБ запаса; свободно {free / 1024**3:.2f} ГиБ в {CACHE_DIRECTORY}.")
    try:
        with tempfile.TemporaryFile(dir=CACHE_DIRECTORY, prefix="frames-") as file:
            # Резервируем блоки заранее: поздний ENOSPC при записи mmap может дать SIGBUS.
            _reserve_file(file, size)
            mapping = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_WRITE)
        output = torch.frombuffer(mapping, dtype=dtype).reshape(tuple(shape))
    except OSError as exc:
        raise MemoryError(f"{label}: не удалось зарезервировать дисковый кэш ({size / 1024**3:.2f} ГиБ): {exc}") from exc
    address = output.untyped_storage().data_ptr()
    _MAPPINGS[address] = weakref.ref(mapping, lambda ref: _forget_mapping(address, ref))
    _LOGGER.info("[%s] Выход: %.2f ГиБ, дисковый IMAGE-кэш в %s; без полного батча в RAM.", label, size / 1024**3, CACHE_DIRECTORY)
    # TemporaryFile не оставляет имени; место освобождается с последним storage alias.
    return output


def disk_backed(images):
    """Recognize shared mapped storage even after detach, slicing, or view creation."""
    if images.device.type != "cpu":
        return False
    reference = _MAPPINGS.get(images.untyped_storage().data_ptr())
    return reference is not None and reference() is not None


def release_video_pages(images, start=0, end=None, *, written=False):
    """Flush written frames and let Linux reclaim file-backed pages between chunks."""
    if not disk_backed(images):
        return
    mapping = _MAPPINGS[images.untyped_storage().data_ptr()]()
    if mapping is None or not images.is_contiguous():
        return
    end = len(images) if end is None else end
    frame_bytes = video_bytes(images.shape[1:], images.dtype)
    first = images.storage_offset() * images.element_size() + start * frame_bytes
    last = min(len(mapping), images.storage_offset() * images.element_size() + end * frame_bytes)
    offset = first // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    if last <= offset:
        return
    if written:
        mapping.flush(offset, last - offset)
    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        mapping.madvise(mmap.MADV_DONTNEED, offset, last - offset)
