"""
Module: nodes/image_download_dzi_tiles.py
Author: AlexZ1967
Last updated: 2026-02-26

Description:
    Download and assemble Deep Zoom (DZI) image tiles into a single image tensor.

Purpose:
    Provides a ComfyUI node that fetches tile images from a Deep Zoom endpoint
    and stitches them into one output IMAGE tensor.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
_DEFAULT_REFERER = "https://www.npg.org.uk/"


def _build_zoom_base_url(base_url: str, mw: str) -> str:
    """Compose normalized zoom base URL from site root and image id."""
    base = str(base_url or "").strip().rstrip("/")
    module_id = str(mw or "").strip()
    if not base:
        raise ValueError("`base_url` must not be empty.")
    if not module_id:
        raise ValueError("`mw` must not be empty.")
    lower_base = base.lower()
    lower_mw = module_id.lower()
    if lower_base.endswith(f"/zoom/{lower_mw}"):
        return base
    if lower_base.endswith("/zoom"):
        return f"{base}/{module_id}"
    if lower_base.endswith(f"/{lower_mw}"):
        prefix = base[: -(len(module_id) + 1)].rstrip("/")
        if prefix.lower().endswith("/zoom"):
            return base
        return f"{prefix}/zoom/{module_id}" if prefix else f"{base}/zoom/{module_id}"
    return f"{base}/zoom/{module_id}"


def _new_session() -> requests.Session:
    """Create HTTP session with browser-like headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": _DEFAULT_UA,
            "Referer": _DEFAULT_REFERER,
        }
    )
    return session


def _tile_url(tiles_base: str, x: int, y: int) -> str:
    """Build tile URL for one tile coordinate."""
    return f"{tiles_base}/{int(x)}_{int(y)}.jpg"


def _http_status(session: requests.Session, url: str, timeout: float) -> int:
    """Return HTTP status code with small retry for transient network errors."""
    for _ in range(3):
        try:
            response = session.get(url, timeout=timeout)
            return int(response.status_code)
        except requests.RequestException:
            continue
    return 0


def _download_tile(session: requests.Session, url: str, timeout: float) -> Image.Image | None:
    """Download one JPEG tile and decode it as PIL image."""
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.load()
        return image
    except Exception:
        return None


def _probe_axis_count(
    session: requests.Session,
    tiles_base: str,
    *,
    axis: str,
    timeout: float,
    max_tiles: int = 4096,
) -> int:
    """Probe tile count on one axis using robust status checks."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be `x` or `y`")
    last_success = -1
    misses_after_success = 0
    for i in range(max_tiles):
        x = i if axis == "x" else 0
        y = i if axis == "y" else 0
        status = _http_status(session, _tile_url(tiles_base, x, y), timeout)
        if status == 200:
            last_success = i
            misses_after_success = 0
            continue
        if last_success < 0:
            # No successful tile yet; avoid very long scan when endpoint is invalid.
            if i >= 7:
                return 0
            continue
        misses_after_success += 1
        if misses_after_success >= 6:
            return last_success + 1
    return (last_success + 1) if last_success >= 0 else 0


def _parse_dzi(session: requests.Session, dzi_url: str, timeout: float) -> dict[str, Any] | None:
    """Try to parse DZI metadata (tile size and nominal dimensions)."""
    try:
        response = session.get(dzi_url, timeout=timeout)
        if response.status_code != 200:
            return None
        root = ET.fromstring(response.text)
        tile_size = int(root.attrib.get("TileSize", "256"))
        overlap = int(root.attrib.get("Overlap", "0"))
        image_format = str(root.attrib.get("Format", "jpg"))

        size_el = None
        for el in root.iter():
            if str(el.tag).lower().endswith("size"):
                size_el = el
                break
        if size_el is None:
            return None

        width = int(size_el.attrib["Width"])
        height = int(size_el.attrib["Height"])
        return {
            "tile_size": tile_size,
            "overlap": overlap,
            "format": image_format,
            "width": width,
            "height": height,
        }
    except Exception:
        return None


def _compute_level_geometry_from_dzi(dzi_info: dict[str, Any], level: int) -> tuple[int, int, int, int]:
    """Compute level-specific output size and tile grid from DeepZoom metadata."""
    tile_size = max(1, int(dzi_info["tile_size"]))
    full_width = max(1, int(dzi_info["width"]))
    full_height = max(1, int(dzi_info["height"]))
    max_dim = max(full_width, full_height)
    max_level = int(math.ceil(math.log2(float(max_dim)))) if max_dim > 1 else 0
    level_i = int(level)
    scale_div = float(2 ** max(0, max_level - level_i))
    level_width = max(1, int(math.ceil(float(full_width) / scale_div)))
    level_height = max(1, int(math.ceil(float(full_height) / scale_div)))
    tiles_x = max(1, int(math.ceil(float(level_width) / float(tile_size))))
    tiles_y = max(1, int(math.ceil(float(level_height) / float(tile_size))))
    return level_width, level_height, tiles_x, tiles_y


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to Comfy IMAGE tensor format [1,H,W,3], float32."""
    np_image = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)


class ImageDownloadDZITiles:
    """ComfyUI node that downloads and assembles DZI tile images."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": "https://collectionimages.npg.org.uk",
                        "multiline": False,
                        "tooltip": "Базовый URL сайта без /zoom (например https://collectionimages.npg.org.uk). Суффикс /zoom добавляется автоматически.",
                    },
                ),
                "mw": (
                    "STRING",
                    {
                        "default": "mw207134",
                        "multiline": False,
                        "tooltip": "Идентификатор изображения (например mw207134). Полный путь формируется как <base_url>/zoom/<mw>/...",
                    },
                ),
                "level": (
                    "INT",
                    {
                        "default": 11,
                        "min": 0,
                        "max": 32,
                        "tooltip": "Уровень DZI-тайлов (папка .../zoomXML_files/<level>/). Чем выше уровень, тем выше итоговое разрешение.",
                    },
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "download"
    CATEGORY = "image/io"

    def download(self, base_url: str, mw: str, level: int):
        """Download DZI tiles for selected level and return assembled image tensor."""
        zoom_base = _build_zoom_base_url(base_url, mw)
        dzi_url = f"{zoom_base}/zoomXML.dzi"
        tiles_base = f"{zoom_base}/zoomXML_files/{int(level)}"
        timeout = 20.0

        session = _new_session()
        first_tile = _download_tile(session, _tile_url(tiles_base, 0, 0), timeout)
        if first_tile is None:
            raise RuntimeError(
                f"First tile is unavailable at `{_tile_url(tiles_base, 0, 0)}`. "
                "Check `base_url`, `mw`, and `level`."
            )

        dzi_info = _parse_dzi(session, dzi_url, timeout)
        tile_size = int(dzi_info["tile_size"]) if isinstance(dzi_info, dict) else int(first_tile.size[0])
        if isinstance(dzi_info, dict):
            width, height, tiles_x, tiles_y = _compute_level_geometry_from_dzi(dzi_info, int(level))
        else:
            tiles_x_probe = _probe_axis_count(session, tiles_base, axis="x", timeout=timeout)
            tiles_y_probe = _probe_axis_count(session, tiles_base, axis="y", timeout=timeout)
            if tiles_x_probe <= 0 or tiles_y_probe <= 0:
                raise RuntimeError("Could not probe tile grid (x/y tile counts are zero).")
            tiles_x = int(tiles_x_probe)
            tiles_y = int(tiles_y_probe)
            last_x_tile = _download_tile(session, _tile_url(tiles_base, tiles_x - 1, 0), timeout)
            last_y_tile = _download_tile(session, _tile_url(tiles_base, 0, tiles_y - 1), timeout)
            width = (tiles_x - 1) * tile_size + (last_x_tile.size[0] if last_x_tile else tile_size)
            height = (tiles_y - 1) * tile_size + (last_y_tile.size[1] if last_y_tile else tile_size)

        canvas = Image.new("RGB", (int(width), int(height)))
        canvas.paste(first_tile, (0, 0))
        for y in range(tiles_y):
            for x in range(tiles_x):
                if x == 0 and y == 0:
                    continue
                tile = _download_tile(session, _tile_url(tiles_base, x, y), timeout)
                if tile is None:
                    continue
                canvas.paste(tile, (x * tile_size, y * tile_size))

        return (_image_to_tensor(canvas),)
