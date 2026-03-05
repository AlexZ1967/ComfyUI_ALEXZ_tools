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
import traceback
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image
try:
    from tqdm.auto import tqdm
except Exception:
    class _NoopTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def update(self, n=1):
            return None

        def set_postfix_str(self, s, refresh=True):
            return None

        def close(self):
            return None

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

    def tqdm(iterable=None, **kwargs):
        return _NoopTqdm(iterable=iterable, **kwargs)


_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
_DEFAULT_REFERER = "https://www.npg.org.uk/"


def _log(message: str) -> None:
    """Emit node logs to ComfyUI console."""
    print(f"[DZI] {message}")


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


def _origin_from_url(url_text: str) -> str:
    """Extract URL origin (`scheme://host[:port]`) for request headers."""
    try:
        parsed = urlsplit(str(url_text or "").strip())
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return ""


def _new_session(*, referer: str | None = None, origin: str | None = None, trust_env: bool = True) -> requests.Session:
    """Create HTTP session with browser-like headers."""
    session = requests.Session()
    session.trust_env = bool(trust_env)
    ref = str(referer or _DEFAULT_REFERER).strip() or _DEFAULT_REFERER
    org = str(origin or _origin_from_url(ref)).strip()
    session.headers.update(
        {
            "User-Agent": _DEFAULT_UA,
            "Referer": ref,
            "Origin": org,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-site",
        }
    )
    return session


def _tile_url(tiles_base: str, x: int, y: int, tile_ext: str = "jpg") -> str:
    """Build tile URL for one tile coordinate."""
    ext = str(tile_ext or "jpg").strip().lower().lstrip(".") or "jpg"
    return f"{tiles_base}/{int(x)}_{int(y)}.{ext}"


def _candidate_tile_exts(dzi_info: dict[str, Any] | None) -> list[str]:
    """Return ordered candidate tile extensions with DZI format priority."""
    candidates: list[str] = []
    if isinstance(dzi_info, dict):
        fmt = str(dzi_info.get("format") or "").strip().lower().lstrip(".")
        if fmt:
            candidates.append("jpg" if fmt == "jpeg" else fmt)
    for fallback in ("jpg", "jpeg", "png", "webp"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _http_status(session: requests.Session, url: str, timeout: float) -> int:
    """Return HTTP status code with small retry for transient network errors."""
    for attempt in range(3):
        try:
            response = session.get(url, timeout=timeout)
            return int(response.status_code)
        except requests.RequestException as exc:
            if attempt >= 2:
                _log(f"HTTP status check failed: {url} ({type(exc).__name__}: {exc})")
            continue
    return 0


def _download_tile(session: requests.Session, url: str, timeout: float) -> Image.Image | None:
    """Download one JPEG tile and decode it as PIL image."""
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code != 200:
            _log(f"Tile unavailable: {url} (status={int(response.status_code)})")
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        _log(f"Tile download/decode error: {url} ({type(exc).__name__}: {exc})")
        return None


def _probe_axis_count(
    session: requests.Session,
    tiles_base: str,
    tile_ext: str,
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
        status = _http_status(session, _tile_url(tiles_base, x, y, tile_ext), timeout)
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
            _log(f"DZI metadata unavailable: {dzi_url} (status={int(response.status_code)})")
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
    except Exception as exc:
        _log(f"DZI parse error: {dzi_url} ({type(exc).__name__}: {exc})")
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
        try:
            zoom_base = _build_zoom_base_url(base_url, mw)
            dzi_url = f"{zoom_base}/zoomXML.dzi"
            tiles_base = f"{zoom_base}/zoomXML_files/{int(level)}"
            referer_root = _origin_from_url(zoom_base) or _DEFAULT_REFERER.rstrip("/")
            timeout = 20.0
            _log(f"Start download: mw={mw}, level={int(level)}")
            _log(f"Base: {zoom_base}")
            _log(f"DZI: {dzi_url}")
            _log(f"Tiles: {tiles_base}")

            session = _new_session(referer=f"{referer_root}/", origin=referer_root, trust_env=True)
            dzi_info = _parse_dzi(session, dzi_url, timeout)
            tile_ext_candidates = _candidate_tile_exts(dzi_info)
            first_tile = None
            tile_ext = ""
            first_tile_statuses: dict[str, int] = {}
            for ext in tile_ext_candidates:
                probe_url = _tile_url(tiles_base, 0, 0, ext)
                status = _http_status(session, probe_url, timeout)
                first_tile_statuses[ext] = int(status)
                if status != 200:
                    continue
                first_tile = _download_tile(session, probe_url, timeout)
                if first_tile is not None:
                    tile_ext = ext
                    break

            if first_tile is None and any(code == 403 for code in first_tile_statuses.values()):
                # Some environments inject HTTP(S)_PROXY rules that can return 403 for CDN image hosts.
                _log("First tile returned 403; retrying with proxy-bypass session.")
                session = _new_session(referer=f"{referer_root}/", origin=referer_root, trust_env=False)
                first_tile_statuses.clear()
                for ext in tile_ext_candidates:
                    probe_url = _tile_url(tiles_base, 0, 0, ext)
                    status = _http_status(session, probe_url, timeout)
                    first_tile_statuses[ext] = int(status)
                    if status != 200:
                        continue
                    first_tile = _download_tile(session, probe_url, timeout)
                    if first_tile is not None:
                        tile_ext = ext
                        break

            if first_tile is None:
                status_hint = ", ".join(f"{ext}:{code}" for ext, code in first_tile_statuses.items()) or "n/a"
                raise RuntimeError(
                    f"First tile is unavailable at `{tiles_base}`. "
                    f"Tried extensions [{', '.join(tile_ext_candidates)}], statuses [{status_hint}]. "
                    "Check `base_url`, `mw`, and `level`."
                )
            _log(f"Tile extension selected: .{tile_ext}")
            tile_size = int(dzi_info["tile_size"]) if isinstance(dzi_info, dict) else int(first_tile.size[0])
            if isinstance(dzi_info, dict):
                width, height, tiles_x, tiles_y = _compute_level_geometry_from_dzi(dzi_info, int(level))
                _log(
                    f"Geometry source=DZI, tile_size={tile_size}, "
                    f"canvas={int(width)}x{int(height)}, grid={int(tiles_x)}x{int(tiles_y)}"
                )
            else:
                tiles_x_probe = _probe_axis_count(session, tiles_base, tile_ext=tile_ext, axis="x", timeout=timeout)
                tiles_y_probe = _probe_axis_count(session, tiles_base, tile_ext=tile_ext, axis="y", timeout=timeout)
                if tiles_x_probe <= 0 or tiles_y_probe <= 0:
                    raise RuntimeError("Could not probe tile grid (x/y tile counts are zero).")
                tiles_x = int(tiles_x_probe)
                tiles_y = int(tiles_y_probe)
                last_x_tile = _download_tile(session, _tile_url(tiles_base, tiles_x - 1, 0, tile_ext), timeout)
                last_y_tile = _download_tile(session, _tile_url(tiles_base, 0, tiles_y - 1, tile_ext), timeout)
                width = (tiles_x - 1) * tile_size + (last_x_tile.size[0] if last_x_tile else tile_size)
                height = (tiles_y - 1) * tile_size + (last_y_tile.size[1] if last_y_tile else tile_size)
                _log(
                    f"Geometry source=probe, tile_size={tile_size}, "
                    f"canvas={int(width)}x{int(height)}, grid={int(tiles_x)}x{int(tiles_y)}"
                )

            canvas = Image.new("RGB", (int(width), int(height)))
            canvas.paste(first_tile, (0, 0))

            total_tiles = max(1, int(tiles_x) * int(tiles_y))
            missing_tiles = 0
            downloaded_tiles = 1
            bar = tqdm(total=total_tiles, desc="DZI Tiles", unit="tile")
            try:
                bar.update(1)
                for y in range(tiles_y):
                    for x in range(tiles_x):
                        if x == 0 and y == 0:
                            continue
                        tile = _download_tile(session, _tile_url(tiles_base, x, y, tile_ext), timeout)
                        if tile is None:
                            missing_tiles += 1
                        else:
                            canvas.paste(tile, (x * tile_size, y * tile_size))
                            downloaded_tiles += 1
                        bar.update(1)
                        bar.set_postfix_str(
                            f"ok={downloaded_tiles}/{total_tiles}, miss={missing_tiles}", refresh=False
                        )
            finally:
                bar.close()

            _log(
                f"Done: canvas={int(width)}x{int(height)}, "
                f"tiles_total={total_tiles}, tiles_ok={downloaded_tiles}, tiles_missing={missing_tiles}"
            )
            return (_image_to_tensor(canvas),)
        except Exception as exc:
            _log(f"Node failed: {type(exc).__name__}: {exc}")
            _log(traceback.format_exc().rstrip())
            raise
