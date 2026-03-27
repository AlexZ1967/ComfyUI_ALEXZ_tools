"""
Module: nodes/image_download_iiif.py
Author: AlexZ1967
Last updated: 2026-03-27

Description:
    Download a single image from IIIF Image API services.

Purpose:
    Provides a ComfyUI node for IIIF-compatible sources such as London Museum,
    resolving either an object page or a direct IIIF service URL into one
    output IMAGE tensor.
"""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import re
import shutil
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests
import torch
from PIL import Image

from ..utils.interrupt import check_interrupt, is_interrupt_exception
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
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
_IIIF_TILE_CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache" / "iiif_tiles"
_RETRYABLE_HTTP_EXCEPTIONS = (
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
)


def _log(message: str) -> None:
    """Emit node logs to ComfyUI console."""
    print(f"[IIIF] {message}")


def _new_http_session() -> requests.Session:
    """Create reusable HTTP session for IIIF requests."""
    session = requests.Session()
    session.headers.update({"User-Agent": _DEFAULT_UA})
    return session


def _http_get(
    url: str,
    *,
    timeout: float,
    session: requests.Session | None = None,
    retries: int = 3,
    retry_backoff: float = 0.75,
) -> requests.Response:
    """HTTP GET wrapper with interrupt support, session reuse, and retry for transient network errors."""
    active_session = session or requests.Session()
    attempts = max(1, int(retries))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            check_interrupt()
            response = active_session.get(url, timeout=timeout)
            check_interrupt()
            return response
        except Exception as exc:
            if is_interrupt_exception(exc):
                raise
            last_exc = exc
            if not isinstance(exc, _RETRYABLE_HTTP_EXCEPTIONS) or attempt >= attempts:
                raise
            _log(
                f"HTTP retry {attempt}/{attempts - 1}: {url} "
                f"({type(exc).__name__}: {exc})"
            )
            time.sleep(float(retry_backoff) * float(attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"HTTP request failed without response: {url}")


def _normalize_iiif_service_url(source_url: str) -> str:
    """Normalize direct IIIF service URL, info.json URL, or image request URL."""
    text = str(source_url or "").strip()
    if not text:
        return ""
    if text.endswith("/info.json"):
        return text[: -len("/info.json")].rstrip("/")
    match = re.match(r"^(https?://.+?)/full/[^/]+/[^/]+/default\.[A-Za-z0-9]+/?$", text)
    if match:
        return str(match.group(1)).rstrip("/")
    return text.rstrip("/")


def _extract_first_london_museum_service_url(html: str) -> str:
    """Extract first IIIF service URL from London Museum object page HTML."""
    patterns = (
        r'data-src="(https://collections\.londonmuseum\.net/iiif/3/[^"]+)"',
        r"(https://collections\.londonmuseum\.net/iiif/3/[^\s\"'<>]+/info\.json)",
        r"(https://collections\.londonmuseum\.net/iiif/3/[^\s\"'<>]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(html or ""), flags=re.I)
        if match:
            return _normalize_iiif_service_url(str(match.group(1)))
    return ""


def _extract_first_generic_iiif_service_url(html: str) -> str:
    """Best-effort extraction of IIIF service URL from arbitrary HTML."""
    patterns = (
        r'data-src="(https?://[^"]+/iiif/[^"]+)"',
        r'src="(https?://[^"]+/iiif/[^"]+/info\.json)"',
        r"(https?://[^\s\"'<>]+/iiif/[^\s\"'<>]+/info\.json)",
        r"(https?://[^\s\"'<>]+/iiif/[^\s\"'<>]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(html or ""), flags=re.I)
        if match:
            return _normalize_iiif_service_url(str(match.group(1)))
    return ""


def _resolve_iiif_service_url(
    site: str,
    source_url: str,
    *,
    timeout: float,
    session: requests.Session | None = None,
) -> str:
    """Resolve effective IIIF service URL from page URL or direct service URL."""
    site_name = str(site or "").strip()
    source_text = str(source_url or "").strip()
    if not source_text:
        raise ValueError("`source_url` must not be empty.")

    if "/iiif/" in source_text:
        service_url = _normalize_iiif_service_url(source_text)
        if service_url:
            return service_url

    response = _http_get(source_text, timeout=timeout, session=session)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Source page unavailable: {source_text} (status={int(response.status_code)})")
    html = response.text or ""

    if site_name == "London Museum Object Page":
        service_url = _extract_first_london_museum_service_url(html)
    else:
        service_url = _extract_first_generic_iiif_service_url(html)
    if service_url:
        return service_url

    raise RuntimeError(f"Could not extract IIIF service URL from `{source_text}`.")


def _fetch_iiif_info(
    service_url: str,
    *,
    timeout: float,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch IIIF info.json metadata."""
    info_url = f"{service_url.rstrip('/')}/info.json"
    response = _http_get(info_url, timeout=timeout, session=session)
    if int(response.status_code) != 200:
        raise RuntimeError(f"IIIF info.json unavailable: {info_url} (status={int(response.status_code)})")
    try:
        return dict(response.json())
    except Exception as exc:
        raise RuntimeError(f"Could not parse IIIF info.json: {info_url} ({type(exc).__name__}: {exc})") from exc


def _build_iiif_size_spec(size_mode: str, requested_width: int) -> str:
    """Build IIIF Image API size segment."""
    mode = str(size_mode or "max").strip().lower()
    if mode == "max":
        return "max"
    width = max(1, int(requested_width))
    return f"{width},"


def _download_iiif_image_bytes(
    service_url: str,
    *,
    size_spec: str,
    output_format: str,
    timeout: float,
    session: requests.Session | None = None,
) -> tuple[str, bytes, str]:
    """Download one IIIF image response, with jpg fallback when requested format fails."""
    requested_format = str(output_format or "jpg").strip().lower().lstrip(".") or "jpg"
    formats_to_try = [requested_format]
    if requested_format != "jpg":
        formats_to_try.append("jpg")
    last_status = 0
    for fmt in formats_to_try:
        check_interrupt()
        image_url = f"{service_url.rstrip('/')}/full/{size_spec}/0/default.{fmt}"
        response = _http_get(image_url, timeout=timeout, session=session)
        last_status = int(response.status_code)
        if last_status == 200:
            return image_url, bytes(response.content or b""), fmt
    raise RuntimeError(
        f"IIIF image request failed for `{service_url}` with size `{size_spec}` "
        f"and formats {formats_to_try} (last_status={last_status})."
    )


def _iiif_source_dimensions(info: dict[str, Any]) -> tuple[int, int]:
    """Return declared source width/height from IIIF info.json."""
    return max(1, int(info.get("width") or 1)), max(1, int(info.get("height") or 1))


def _iiif_limit_from_max_area(info: dict[str, Any]) -> dict[str, Any] | None:
    """Compute expected single-request limit when service exposes maxArea."""
    max_area = int(info.get("maxArea") or 0)
    if max_area <= 0:
        return None
    source_width, source_height = _iiif_source_dimensions(info)
    area = float(source_width) * float(source_height)
    if area <= 0:
        return None
    scale = math.sqrt(float(max_area) / area)
    limited_width = max(1, int(math.floor(float(source_width) * scale)))
    limited_height = max(1, int(math.floor(float(source_height) * scale)))
    return {
        "max_area": max_area,
        "predicted_max_width": limited_width,
        "predicted_max_height": limited_height,
        "scale": scale,
    }


def _iiif_tile_profile(info: dict[str, Any]) -> dict[str, int]:
    """Extract tile profile needed for full-res assembly."""
    tiles = info.get("tiles") or []
    if not isinstance(tiles, list) or not tiles:
        raise RuntimeError("IIIF info.json does not expose `tiles`, full tile assembly is unavailable.")
    tile = tiles[0] if isinstance(tiles[0], dict) else {}
    scale_factors = tile.get("scaleFactors") or []
    if 1 not in scale_factors:
        raise RuntimeError("IIIF service does not expose scaleFactor=1, full-res tile assembly is unavailable.")
    tile_width = max(1, int(tile.get("width") or 512))
    tile_height = max(1, int(tile.get("height") or tile_width))
    return {
        "tile_width": tile_width,
        "tile_height": tile_height,
    }


def _resolve_iiif_cache_dir(cache_dir: str | None) -> Path:
    """Resolve persistent cache directory for IIIF tile assembly."""
    cache_text = str(cache_dir or "").strip()
    if cache_text:
        return Path(os.path.abspath(os.path.expanduser(cache_text)))
    return _IIIF_TILE_CACHE_ROOT


def _resolve_iiif_tile_cache_scope(cache_dir: str | None, service_url: str, info: dict[str, Any]) -> Path:
    """Resolve per-assembly cache scope so completed runs can be cleaned safely."""
    cache_base = _resolve_iiif_cache_dir(cache_dir)
    source_width, source_height = _iiif_source_dimensions(info)
    scope_key = hashlib.sha1(
        f"{service_url.rstrip('/')}|{source_width}|{source_height}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return cache_base / scope_key


def _clear_iiif_cache_dir(cache_root: Path | str | None) -> bool:
    """Remove one assembly cache directory after successful completion."""
    if cache_root is None:
        return False
    path = Path(str(cache_root))
    try:
        if path.exists():
            shutil.rmtree(path)
        return True
    except Exception:
        return False


def _iiif_tile_cache_path(cache_root: Path, image_url: str) -> Path:
    """Map tile URL to stable cache file path."""
    digest = hashlib.sha1(str(image_url or "").encode("utf-8"), usedforsecurity=False).hexdigest()
    split = urlsplit(str(image_url or "").strip())
    ext = Path(split.path).suffix or ".bin"
    return cache_root / digest[:2] / f"{digest}{ext}"


def _load_iiif_tile_from_cache(cache_root: Path, image_url: str) -> bytes | None:
    """Load cached tile bytes if present."""
    path = _iiif_tile_cache_path(cache_root, image_url)
    try:
        if path.exists():
            return path.read_bytes()
    except Exception:
        return None
    return None


def _store_iiif_tile_in_cache(cache_root: Path, image_url: str, content: bytes) -> None:
    """Persist downloaded tile bytes to cache."""
    path = _iiif_tile_cache_path(cache_root, image_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(bytes(content or b""))
    tmp_path.replace(path)


def _download_iiif_tile_bytes(
    service_url: str,
    *,
    region: str,
    output_format: str,
    timeout: float,
    session: requests.Session | None = None,
    cache_dir: str | None = None,
    cache_stats: dict[str, int] | None = None,
) -> tuple[str, bytes, str]:
    """Download one IIIF tile region, with jpg fallback when requested format fails."""
    requested_format = str(output_format or "jpg").strip().lower().lstrip(".") or "jpg"
    formats_to_try = [requested_format]
    if requested_format != "jpg":
        formats_to_try.append("jpg")
    last_status = 0
    cache_root = _resolve_iiif_cache_dir(cache_dir)
    for fmt in formats_to_try:
        check_interrupt()
        image_url = f"{service_url.rstrip('/')}/{region}/max/0/default.{fmt}"
        cached = _load_iiif_tile_from_cache(cache_root, image_url)
        if cached is not None:
            if cache_stats is not None:
                cache_stats["hits"] = int(cache_stats.get("hits", 0)) + 1
            return image_url, cached, fmt
        response = _http_get(image_url, timeout=timeout, session=session)
        last_status = int(response.status_code)
        if last_status == 200:
            content = bytes(response.content or b"")
            _store_iiif_tile_in_cache(cache_root, image_url, content)
            if cache_stats is not None:
                cache_stats["misses"] = int(cache_stats.get("misses", 0)) + 1
                cache_stats["stores"] = int(cache_stats.get("stores", 0)) + 1
            return image_url, content, fmt
    raise RuntimeError(
        f"IIIF tile request failed for `{service_url}` region `{region}` "
        f"and formats {formats_to_try} (last_status={last_status})."
    )


def _assemble_iiif_full_image(
    service_url: str,
    info: dict[str, Any],
    *,
    output_format: str,
    timeout: float,
    session: requests.Session | None = None,
    cache_dir: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Assemble full-resolution image from IIIF tiles at scaleFactor=1."""
    source_width, source_height = _iiif_source_dimensions(info)
    tile_profile = _iiif_tile_profile(info)
    tile_width = int(tile_profile["tile_width"])
    tile_height = int(tile_profile["tile_height"])
    tiles_x = int(math.ceil(float(source_width) / float(tile_width)))
    tiles_y = int(math.ceil(float(source_height) / float(tile_height)))
    total_tiles = max(1, tiles_x * tiles_y)
    canvas = Image.new("RGB", (source_width, source_height))
    selected_format = ""
    last_tile_url = ""
    downloaded_tiles = 0
    cache_root = _resolve_iiif_tile_cache_scope(cache_dir, service_url, info)
    cache_stats = {"hits": 0, "misses": 0, "stores": 0}

    bar = tqdm(total=total_tiles, desc="IIIF Tiles", unit="tile")
    try:
        for ty in range(tiles_y):
            check_interrupt()
            for tx in range(tiles_x):
                check_interrupt()
                x = int(tx * tile_width)
                y = int(ty * tile_height)
                w = int(min(tile_width, source_width - x))
                h = int(min(tile_height, source_height - y))
                region = f"{x},{y},{w},{h}"
                tile_url, content, tile_format = _download_iiif_tile_bytes(
                    service_url,
                    region=region,
                    output_format=output_format,
                    timeout=timeout,
                    session=session,
                    cache_dir=str(cache_root),
                    cache_stats=cache_stats,
                )
                tile = _decode_image(content, tile_url)
                if tile.size != (w, h):
                    tile = tile.resize((w, h), Image.Resampling.LANCZOS)
                canvas.paste(tile, (x, y))
                selected_format = selected_format or tile_format
                last_tile_url = tile_url
                downloaded_tiles += 1
                bar.update(1)
                bar.set_postfix_str(
                    f"ok={downloaded_tiles}/{total_tiles}, cache={cache_stats['hits']}/{downloaded_tiles}",
                    refresh=False,
                )
    finally:
        bar.close()

    metadata = {
        "mode": "tile_assemble_full",
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_x": tiles_x,
        "tiles_y": tiles_y,
        "tiles_total": total_tiles,
        "tiles_downloaded": downloaded_tiles,
        "selected_format": selected_format or str(output_format or "jpg"),
        "last_tile_url": last_tile_url,
        "cache_dir": str(cache_root),
        "cache_hits": int(cache_stats["hits"]),
        "cache_misses": int(cache_stats["misses"]),
        "cache_stores": int(cache_stats["stores"]),
        "cache_cleared": False,
    }
    return canvas, metadata


def _decode_image(content: bytes, image_url: str) -> Image.Image:
    """Decode response bytes into RGB PIL image."""
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        raise RuntimeError(f"Could not decode image bytes from `{image_url}` ({type(exc).__name__}: {exc})") from exc


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to Comfy IMAGE tensor format [1,H,W,3], float32."""
    np_image = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)


def _sanitize_filename_component(text: str) -> str:
    """Normalize user-facing filename fragment into a safe portable stem."""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(text or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "iiif_image"


def _extract_html_title(html_text: str) -> str:
    """Extract best available human-readable title from HTML."""
    text = str(html_text or "")
    patterns = [
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+name=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+property=["\']twitter:title["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+name=["\']twitter:title["\'][^>]+content=["\'](.*?)["\']',
        r"<title[^>]*>(.*?)</title>",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        value = html.unescape(str(match.group(1) or "")).strip()
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            return value
    return ""


def _extract_iiif_stable_id(source_url: str, service_url: str = "") -> str:
    """Extract stable object/service identifier for filename disambiguation."""
    combined = " | ".join([str(source_url or "").strip(), str(service_url or "").strip()])
    patterns = (
        r"\b(object-\d+)\b",
        r"\b(nla\.obj-\d+)\b",
        r"\b(mw\d+)\b",
        r"/([A-Za-z]\d{3,}(?:\.ptif)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\.ptif$", "", str(match.group(1) or ""), flags=re.IGNORECASE)
            value = _sanitize_filename_component(value)
            if value:
                return value
    return ""


def _append_stable_id_to_stem(stem: str, stable_id: str) -> str:
    """Append stable id to stem unless it is already present."""
    base = _sanitize_filename_component(stem)
    stable = _sanitize_filename_component(stable_id)
    if not stable:
        return base or "iiif_image"
    lowered_base = base.lower()
    lowered_stable = stable.lower()
    if lowered_stable in lowered_base:
        return base or stable
    if not base:
        return stable
    return f"{base}_{stable}"


def _derive_output_stem_from_source_url(source_url: str, service_url: str = "") -> str:
    """Derive filename stem from URL slug plus stable object/service id when available."""
    path = str(urlsplit(str(source_url or "").strip()).path or "").strip("/")
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return _append_stable_id_to_stem("iiif_image", _extract_iiif_stable_id(source_url, service_url))
    candidate = segments[-1]
    lowered = candidate.lower()
    if lowered in {"info.json"} and len(segments) >= 2:
        candidate = segments[-2]
    elif lowered.startswith("default.") and len(segments) >= 2:
        candidate = segments[-2]
    candidate = re.sub(r"\.[A-Za-z0-9]+$", "", candidate)
    return _append_stable_id_to_stem(candidate, _extract_iiif_stable_id(source_url, service_url))


def _derive_output_stem_from_source_title_or_url(
    source_url: str,
    *,
    timeout: float,
    session: requests.Session | None = None,
    service_url: str = "",
) -> str:
    """Prefer object page title for filename; fall back to URL slug."""
    stable_id = _extract_iiif_stable_id(source_url, service_url)
    fallback = _derive_output_stem_from_source_url(source_url, service_url)
    url = str(source_url or "").strip()
    if not url:
        return fallback
    try:
        response = _http_get(url, timeout=timeout, session=session)
        if int(response.status_code) != 200:
            return fallback
        title = _extract_html_title(str(response.text or ""))
        return _append_stable_id_to_stem(title, stable_id) if title else fallback
    except Exception:
        return fallback


def _save_pil_image(image: Image.Image, output_path: str, output_format: str) -> None:
    """Save PIL image to disk using requested output format."""
    fmt = str(output_format or "jpg").strip().lower().lstrip(".") or "jpg"
    image_rgb = image.convert("RGB")
    if fmt == "png":
        image_rgb.save(output_path, format="PNG", optimize=True)
        return
    if fmt in {"jpg", "jpeg"}:
        image_rgb.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
        return
    if fmt == "webp":
        image_rgb.save(output_path, format="WEBP", quality=95, method=6)
        return
    if fmt == "tif":
        image_rgb.save(output_path, format="TIFF")
        return
    if fmt == "gif":
        image_rgb.save(output_path, format="GIF")
        return
    raise ValueError("Unsupported save format. Allowed: jpg, jpeg, png, webp, tif, gif.")


def _resolve_unique_output_path(output_dir: str, stem: str, ext: str) -> str:
    """Return non-destructive output path by adding numeric suffix when needed."""
    normalized_ext = str(ext or "jpg").strip().lower().lstrip(".") or "jpg"
    base_path = os.path.join(output_dir, f"{stem}.{normalized_ext}")
    if not os.path.exists(base_path):
        return base_path
    index = 2
    while True:
        candidate = os.path.join(output_dir, f"{stem}_{index}.{normalized_ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


class ImageDownloadIIIFImage:
    """ComfyUI node for downloading one image from IIIF Image API sources."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema for IIIF image download."""
        return {
            "required": {
                "site": (
                    ["London Museum Object Page", "Generic IIIF Service URL"],
                    {
                        "default": "London Museum Object Page",
                        "tooltip": "Источник IIIF. London Museum умеет принимать object page URL. Generic ожидает IIIF service URL, info.json URL или HTML-страницу с встраиваемым IIIF viewer.",
                    },
                ),
                "source_url": (
                    "STRING",
                    {
                        "default": "https://www.londonmuseum.org.uk/collections/v/object-443296/early-portrait-of-anna-pavlova/",
                        "multiline": False,
                        "tooltip": "London Museum: URL object page. Generic: IIIF service URL, info.json URL или HTML-страница, из которой можно извлечь IIIF service URL.",
                    },
                ),
            },
            "optional": {
                "size_mode": (
                    ["max", "width"],
                    {
                        "default": "max",
                        "tooltip": "max = запросить максимально доступный размер. width = запросить изображение заданной ширины.",
                    },
                ),
                "requested_width": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 32,
                        "max": 20000,
                        "tooltip": "Ширина для size_mode=width. При size_mode=max игнорируется.",
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Если указана директория, итоговая картинка будет сохранена на диск. Имя файла берется из последнего meaningful сегмента входного source_url.",
                    },
                ),
                "cache_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Папка persistent-кеша для IIIF tile assembly. Пусто = использовать встроенную папку cache/iiif_tiles в модуле. Уже загруженные тайлы будут переиспользованы при повторных запусках.",
                    },
                ),
                "filename_mode": (
                    ["source_url_slug", "title_or_slug"],
                    {
                        "default": "source_url_slug",
                        "tooltip": "source_url_slug = имя файла из URL. title_or_slug = попытаться взять title страницы source_url, иначе fallback на slug из URL.",
                    },
                ),
                "delivery_mode": (
                    ["single_request", "tile_assemble_full"],
                    {
                        "default": "single_request",
                        "tooltip": "single_request = один IIIF image request. tile_assemble_full = собрать full-resolution изображение из IIIF tiles (если сервис поддерживает scaleFactor=1).",
                    },
                ),
                "output_format": (
                    ["jpg", "png", "webp", "tif", "gif"],
                    {
                        "default": "jpg",
                        "tooltip": "Предпочитаемый формат IIIF image request. Если формат не поддерживается, нода автоматически попробует jpg.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info_json")
    FUNCTION = "download"
    CATEGORY = "image/io"

    def download(
        self,
        site: str,
        source_url: str,
        size_mode: str = "max",
        requested_width: int = 2000,
        output_dir: str = "",
        cache_dir: str = "",
        filename_mode: str = "source_url_slug",
        delivery_mode: str = "single_request",
        output_format: str = "jpg",
    ):
        """Resolve IIIF service URL, fetch image, and return IMAGE tensor with JSON metadata."""
        timeout = 30.0
        try:
            check_interrupt()
            session = _new_http_session()
            service_url = _resolve_iiif_service_url(site, source_url, timeout=timeout, session=session)
            _log(f"Site: {site}")
            _log(f"Source: {source_url}")
            _log(f"Service: {service_url}")

            info = _fetch_iiif_info(service_url, timeout=timeout, session=session)
            source_width, source_height = _iiif_source_dimensions(info)
            limit_info = _iiif_limit_from_max_area(info)
            mode = str(delivery_mode or "single_request").strip().lower()

            if mode == "tile_assemble_full":
                image, delivery_meta = _assemble_iiif_full_image(
                    service_url,
                    info,
                    output_format=output_format,
                    timeout=timeout,
                    session=session,
                    cache_dir=cache_dir,
                )
                width, height = image.size
                image_url = str(delivery_meta.get("last_tile_url") or "")
                selected_format = str(delivery_meta.get("selected_format") or output_format)
                size_spec = "tile_assemble_full"
                _log(
                    f"Tile assembly: {delivery_meta['tiles_x']}x{delivery_meta['tiles_y']} "
                    f"tiles ({delivery_meta['tiles_total']} total, cache_hits={delivery_meta.get('cache_hits', 0)})"
                )
            else:
                size_spec = _build_iiif_size_spec(size_mode, requested_width)
                image_url, content, selected_format = _download_iiif_image_bytes(
                    service_url,
                    size_spec=size_spec,
                    output_format=output_format,
                    timeout=timeout,
                    session=session,
                )
                image = _decode_image(content, image_url)
                width, height = image.size
                delivery_meta = {
                    "mode": "single_request",
                    "selected_format": selected_format,
                    "request_size_spec": size_spec,
                }
                _log(f"Image request: {image_url}")

            limited_by_service = bool((width < source_width) or (height < source_height))
            _log(f"Done: {width}x{height}, format={selected_format}")
            saved_path = ""
            output_dir_text = str(output_dir or "").strip()
            if output_dir_text:
                check_interrupt()
                output_dir_abs = os.path.abspath(os.path.expanduser(output_dir_text))
                os.makedirs(output_dir_abs, exist_ok=True)
                filename_mode_text = str(filename_mode or "source_url_slug").strip().lower()
                if filename_mode_text == "title_or_slug":
                    filename_stem = _derive_output_stem_from_source_title_or_url(
                        source_url,
                        timeout=timeout,
                        session=session,
                        service_url=service_url,
                    )
                else:
                    filename_stem = _derive_output_stem_from_source_url(source_url, service_url)
                save_ext = str(selected_format or output_format or "jpg").strip().lower().lstrip(".") or "jpg"
                saved_path = _resolve_unique_output_path(output_dir_abs, filename_stem, save_ext)
                _save_pil_image(image, saved_path, save_ext)
                _log(f"Saved: {saved_path}")

            if mode == "tile_assemble_full":
                cache_scope = delivery_meta.get("cache_dir")
                cache_cleared = _clear_iiif_cache_dir(cache_scope)
                delivery_meta["cache_cleared"] = bool(cache_cleared)
                if cache_cleared:
                    _log(f"Tile cache cleared: {cache_scope}")

            payload = {
                "site": str(site or "").strip(),
                "source_url": str(source_url or "").strip(),
                "service_url": service_url,
                "info_url": f"{service_url}/info.json",
                "image_url": image_url,
                "size_mode": str(size_mode or "max").strip(),
                "requested_width": int(requested_width),
                "delivery_mode": mode,
                "selected_format": selected_format,
                "iiif": {
                    "id": info.get("id"),
                    "type": info.get("type"),
                    "profile": info.get("profile"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "maxArea": info.get("maxArea"),
                    "maxAllowedSize": info.get("maxAllowedSize"),
                    "tiles": info.get("tiles"),
                    "sizes": info.get("sizes"),
                },
                "source": {
                    "width": int(source_width),
                    "height": int(source_height),
                },
                "downloaded": {
                    "width": int(width),
                    "height": int(height),
                },
                "saved_path": saved_path,
                "delivery": delivery_meta,
                "limits": {
                    "limited_by_service": limited_by_service,
                    "reason": "service_max_area_or_policy" if limited_by_service else "none",
                    "max_area_constraint": limit_info,
                    "requested_mode": mode,
                },
            }
            return (_image_to_tensor(image), json.dumps(payload, ensure_ascii=True, indent=2))
        except Exception as exc:
            if is_interrupt_exception(exc):
                _log("Node interrupted by ComfyUI.")
                raise
            _log(f"Node failed: {type(exc).__name__}: {exc}")
            _log(traceback.format_exc().rstrip())
            raise
