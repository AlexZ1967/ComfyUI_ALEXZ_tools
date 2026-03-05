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
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

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
_FETCH_ERROR_SEEN: set[str] = set()


def _log(message: str) -> None:
    """Emit node logs to ComfyUI console."""
    print(f"[DZI] {message}")


def _log_fetch_error(transport: str, url: str, exc: Exception) -> None:
    """Emit deduplicated transport error details for network diagnostics."""
    key = f"{transport}:{type(exc).__name__}:{str(exc)}"
    if key in _FETCH_ERROR_SEEN:
        return
    _FETCH_ERROR_SEEN.add(key)
    _log(f"Fetch error [{transport}]: {url} ({type(exc).__name__}: {exc})")


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


def _new_session(
    *,
    referer: str | None = None,
    origin: str | None = None,
    trust_env: bool = True,
    cookie: str | None = None,
    proxy_url: str | None = None,
) -> requests.Session:
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
    cookie_text = str(cookie or "").strip()
    if cookie_text:
        session.headers["Cookie"] = cookie_text
    proxy_text = str(proxy_url or "").strip()
    if proxy_text:
        session.proxies.update({"http": proxy_text, "https": proxy_text})
    setattr(session, "_alexz_proxy_url", proxy_text)
    return session


def _make_session(
    *,
    referer: str | None,
    origin: str | None,
    trust_env: bool,
    cookie: str | None = None,
    proxy_url: str | None = None,
) -> requests.Session:
    """Create session with backward-compatible fallback for monkeypatched test stubs."""
    try:
        return _new_session(
            referer=referer,
            origin=origin,
            trust_env=trust_env,
            cookie=cookie,
            proxy_url=proxy_url,
        )
    except TypeError:
        # Compatibility with older tests that monkeypatch `_new_session` as `lambda: ...`.
        session = _new_session()  # type: ignore[call-arg]
        try:
            session.trust_env = bool(trust_env)
        except Exception:
            pass
        try:
            if hasattr(session, "headers") and isinstance(getattr(session, "headers"), dict):
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
        except Exception:
            pass
        try:
            cookie_text = str(cookie or "").strip()
            if cookie_text and hasattr(session, "headers") and isinstance(getattr(session, "headers"), dict):
                session.headers["Cookie"] = cookie_text
        except Exception:
            pass
        try:
            proxy_text = str(proxy_url or "").strip()
            if proxy_text and hasattr(session, "proxies") and isinstance(getattr(session, "proxies"), dict):
                session.proxies.update({"http": proxy_text, "https": proxy_text})
            setattr(session, "_alexz_proxy_url", proxy_text)
        except Exception:
            pass
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


def _fetch_bytes_requests(session: requests.Session, url: str, timeout: float) -> tuple[int, bytes | None]:
    """Fetch URL bytes via requests transport."""
    try:
        response = session.get(url, timeout=timeout)
        return int(response.status_code), bytes(response.content or b"")
    except Exception as exc:
        _log_fetch_error("requests", url, exc)
        return 0, None


def _fetch_bytes_urllib(session: requests.Session, url: str, timeout: float) -> tuple[int, bytes | None]:
    """Fetch URL bytes via urllib transport."""
    req = Request(url, headers={k: str(v) for k, v in session.headers.items()})
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 0) or resp.getcode() or 0)
            body = resp.read()
            return status, bytes(body or b"")
    except HTTPError as exc:
        body = None
        try:
            body = exc.read()
        except Exception:
            body = None
        return int(exc.code or 0), (bytes(body) if body is not None else None)
    except URLError as exc:
        # Keep concise but visible diagnostics for proxy/DNS/connectivity failures.
        _log_fetch_error("urllib", url, exc)
        return 0, None
    except Exception as exc:
        _log_fetch_error("urllib", url, exc)
        return 0, None


def _fetch_bytes_curl(session: requests.Session, url: str, timeout: float) -> tuple[int, bytes | None]:
    """Fetch URL bytes via curl transport if available."""
    if not shutil.which("curl"):
        return 0, None

    marker = b"\n__ALEXZ_HTTP_STATUS__:"
    timeout_s = str(max(1, int(math.ceil(float(timeout)))))
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        timeout_s,
        "-A",
        str(session.headers.get("User-Agent") or _DEFAULT_UA),
        "-e",
        str(session.headers.get("Referer") or _DEFAULT_REFERER),
        "-H",
        f"Origin: {str(session.headers.get('Origin') or _origin_from_url(url))}",
        "-H",
        "Accept: image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "-H",
        "Accept-Language: en-US,en;q=0.9",
        "-H",
        "Sec-Fetch-Dest: image",
        "-H",
        "Sec-Fetch-Mode: no-cors",
        "-H",
        "Sec-Fetch-Site: same-site",
        "-w",
        "__ALEXZ_HTTP_STATUS__:%{http_code}",
        "-o",
        "-",
        url,
    ]
    proxy_url = str(getattr(session, "_alexz_proxy_url", "") or "").strip()
    if not proxy_url:
        try:
            proxy_url = str((getattr(session, "proxies", {}) or {}).get("https") or "").strip()
        except Exception:
            proxy_url = ""
    if proxy_url:
        cmd[1:1] = ["--proxy", proxy_url]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        out = bytes(proc.stdout or b"")
        idx = out.rfind(b"__ALEXZ_HTTP_STATUS__:")
        if idx < 0:
            return 0, out if out else None
        status_raw = out[idx + len(b"__ALEXZ_HTTP_STATUS__:") : idx + len(b"__ALEXZ_HTTP_STATUS__:") + 3]
        try:
            status = int(status_raw.decode("ascii", errors="ignore"))
        except Exception:
            status = 0
        body = out[:idx]
        return status, body
    except Exception as exc:
        _log_fetch_error("curl", url, exc)
        return 0, None


def _fetch_bytes_cloudscraper(session: requests.Session, url: str, timeout: float) -> tuple[int, bytes | None]:
    """Fetch URL bytes via cloudscraper transport when available."""
    try:
        import cloudscraper  # type: ignore
    except Exception as exc:
        _log_fetch_error("cloudscraper", url, exc)
        return 0, None

    try:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
        try:
            scraper.trust_env = bool(getattr(session, "trust_env", True))
        except Exception:
            pass
        try:
            headers_obj = getattr(session, "headers", None)
            if headers_obj and hasattr(scraper, "headers"):
                scraper.headers.update({k: str(v) for k, v in headers_obj.items()})
        except Exception:
            pass
        try:
            proxy_url = str(getattr(session, "_alexz_proxy_url", "") or "").strip()
            if not proxy_url:
                proxy_url = str((getattr(session, "proxies", {}) or {}).get("https") or "").strip()
            if proxy_url and hasattr(scraper, "proxies"):
                scraper.proxies.update({"http": proxy_url, "https": proxy_url})
        except Exception:
            pass
        response = scraper.get(url, timeout=timeout)
        return int(response.status_code), bytes(response.content or b"")
    except Exception:
        return 0, None


def _fetch_bytes(
    session: requests.Session,
    url: str,
    timeout: float,
    *,
    transport: str = "requests",
) -> tuple[int, bytes | None]:
    """Fetch URL bytes with selected transport backend."""
    mode = str(transport or "requests").strip().lower()
    if mode == "requests":
        return _fetch_bytes_requests(session, url, timeout)
    if mode == "cloudscraper":
        return _fetch_bytes_cloudscraper(session, url, timeout)
    if mode == "urllib":
        return _fetch_bytes_urllib(session, url, timeout)
    if mode == "curl":
        return _fetch_bytes_curl(session, url, timeout)
    return 0, None


def _http_status(session: requests.Session, url: str, timeout: float, *, transport: str = "requests") -> int:
    """Return HTTP status code with small retry for transient network errors."""
    for attempt in range(3):
        try:
            status, _ = _fetch_bytes(session, url, timeout, transport=transport)
            return int(status)
        except Exception as exc:
            if attempt >= 2:
                _log(f"HTTP status check failed: {url} ({type(exc).__name__}: {exc})")
            continue
    return 0


def _decode_tile_image(content: bytes | None, url: str) -> Image.Image | None:
    """Decode image bytes to RGB PIL image."""
    try:
        if not content:
            return None
        image = Image.open(BytesIO(content)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        _log(f"Tile decode error: {url} ({type(exc).__name__}: {exc})")
        return None


def _download_tile(
    session: requests.Session,
    url: str,
    timeout: float,
    *,
    transport: str = "requests",
) -> Image.Image | None:
    """Download one JPEG tile and decode it as PIL image."""
    try:
        status, content = _fetch_bytes(session, url, timeout, transport=transport)
        if int(status) != 200:
            _log(f"Tile unavailable: {url} (status={int(status)})")
            return None
        return _decode_tile_image(content, url)
    except Exception as exc:
        _log(f"Tile download/decode error: {url} ({type(exc).__name__}: {exc})")
        return None


def _download_tile_compat(
    session: requests.Session,
    url: str,
    timeout: float,
    *,
    transport: str,
) -> Image.Image | None:
    """Call tile downloader with transport kwarg and keep compatibility with legacy monkeypatches."""
    try:
        return _download_tile(session, url, timeout, transport=transport)
    except TypeError:
        return _download_tile(session, url, timeout)  # type: ignore[call-arg]


def _probe_axis_count(
    session: requests.Session,
    tiles_base: str,
    tile_ext: str,
    transport: str,
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
        status = _http_status(session, _tile_url(tiles_base, x, y, tile_ext), timeout, transport=transport)
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


def _probe_axis_count_compat(
    session: requests.Session,
    tiles_base: str,
    *,
    tile_ext: str,
    transport: str,
    axis: str,
    timeout: float,
    max_tiles: int = 4096,
) -> int:
    """Call probe helper with new kwargs and keep compatibility with legacy monkeypatch signatures."""
    try:
        return _probe_axis_count(
            session,
            tiles_base,
            tile_ext=tile_ext,
            transport=transport,
            axis=axis,
            timeout=timeout,
            max_tiles=max_tiles,
        )
    except TypeError:
        return _probe_axis_count(  # type: ignore[call-arg]
            session,
            tiles_base,
            axis=axis,
            timeout=timeout,
            max_tiles=max_tiles,
        )


def _parse_dzi(
    session: requests.Session,
    dzi_url: str,
    timeout: float,
    *,
    transport: str = "requests",
) -> dict[str, Any] | None:
    """Try to parse DZI metadata (tile size and nominal dimensions)."""
    try:
        status, content = _fetch_bytes(session, dzi_url, timeout, transport=transport)
        if int(status) != 200:
            _log(f"DZI metadata unavailable: {dzi_url} (status={int(status)})")
            return None
        text = (content or b"").decode("utf-8", errors="replace")
        root = ET.fromstring(text)
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
            "optional": {
                "transport": (
                    ["auto", "requests", "cloudscraper", "urllib", "curl"],
                    {
                        "default": "auto",
                        "tooltip": "Транспорт HTTP. auto = перебор requests/cloudscraper/urllib/curl.",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Явный HTTP(S) proxy (например http://127.0.0.1:7890). Пусто = системные env-прокси.",
                    },
                ),
                "cookie": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Дополнительный Cookie заголовок (например cf_clearance=...; other=...).",
                    },
                ),
                "disable_env_proxy": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Игнорировать HTTP(S)_PROXY из окружения для requests/cloudscraper.",
                    },
                ),
                "referer": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Явный Referer (пусто = авто-перебор).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "download"
    CATEGORY = "image/io"

    def download(
        self,
        base_url: str,
        mw: str,
        level: int,
        transport: str = "auto",
        proxy_url: str = "",
        cookie: str = "",
        disable_env_proxy: bool = False,
        referer: str = "",
    ):
        """Download DZI tiles for selected level and return assembled image tensor."""
        try:
            zoom_base = _build_zoom_base_url(base_url, mw)
            dzi_url = f"{zoom_base}/zoomXML.dzi"
            tiles_base = f"{zoom_base}/zoomXML_files/{int(level)}"
            referer_root = _origin_from_url(zoom_base) or _DEFAULT_REFERER.rstrip("/")
            referer_candidates = [
                f"{zoom_base.rstrip('/')}/",
                f"{referer_root.rstrip('/')}/",
                _DEFAULT_REFERER,
            ]
            referer_override = str(referer or "").strip()
            if referer_override:
                referer_candidates.insert(0, referer_override)
            dedup_referers = []
            seen_refs = set()
            for ref in referer_candidates:
                ref_norm = str(ref or "").strip()
                if not ref_norm or ref_norm in seen_refs:
                    continue
                seen_refs.add(ref_norm)
                dedup_referers.append(ref_norm)
            referer_candidates = dedup_referers
            timeout = 20.0
            _log(f"Start download: mw={mw}, level={int(level)}")
            _log(f"Base: {zoom_base}")
            _log(f"DZI: {dzi_url}")
            _log(f"Tiles: {tiles_base}")

            # Do not query DZI metadata before first tile probe: some hosts can
            # deny `.dzi` while still allowing tile JPEGs.
            tile_ext_candidates = _candidate_tile_exts(None)
            selected_transport = str(transport or "auto").strip().lower()
            if selected_transport in {"requests", "cloudscraper", "urllib", "curl"}:
                transport_candidates = [selected_transport]
            else:
                transport_candidates = ["requests", "cloudscraper", "urllib", "curl"]
            proxy_text = str(proxy_url or "").strip()
            cookie_text = str(cookie or "").strip()
            trust_env_primary = not bool(disable_env_proxy)
            first_tile = None
            tile_ext = ""
            chosen_transport = ""
            first_tile_statuses: dict[str, int] = {}
            session = None
            for ref_idx, ref in enumerate(referer_candidates):
                trial_session = _make_session(
                    referer=ref,
                    origin=referer_root,
                    trust_env=trust_env_primary,
                    cookie=cookie_text,
                    proxy_url=proxy_text,
                )
                for ext in tile_ext_candidates:
                    probe_url = _tile_url(tiles_base, 0, 0, ext)
                    for transport in transport_candidates:
                        status, content = _fetch_bytes(trial_session, probe_url, timeout, transport=transport)
                        first_tile_statuses[f"{ext}@{transport}#r{ref_idx+1}"] = int(status)
                        if int(status) == 200:
                            first_tile = _decode_tile_image(content, probe_url)
                        else:
                            # Compatibility fallback for mocked/legacy paths where
                            # status probes are stubbed but `_download_tile` returns data.
                            first_tile = _download_tile_compat(
                                trial_session,
                                probe_url,
                                timeout,
                                transport=transport,
                            )
                        if first_tile is not None:
                            tile_ext = ext
                            chosen_transport = transport
                            session = trial_session
                            break
                    if first_tile is not None:
                        break
                if first_tile is not None:
                    _log(f"Referer selected: {ref}")
                    break

            if first_tile is None and any(code == 403 for code in first_tile_statuses.values()):
                # Some environments inject HTTP(S)_PROXY rules that can return 403 for CDN image hosts.
                _log("First tile returned 403; retrying with proxy-bypass session.")
                first_tile_statuses.clear()
                for ref_idx, ref in enumerate(referer_candidates):
                    trial_session = _make_session(
                        referer=ref,
                        origin=referer_root,
                        trust_env=False,
                        cookie=cookie_text,
                        proxy_url=proxy_text,
                    )
                    for ext in tile_ext_candidates:
                        probe_url = _tile_url(tiles_base, 0, 0, ext)
                        for transport in transport_candidates:
                            status, content = _fetch_bytes(trial_session, probe_url, timeout, transport=transport)
                            first_tile_statuses[f"{ext}@{transport}#r{ref_idx+1}"] = int(status)
                            if int(status) == 200:
                                first_tile = _decode_tile_image(content, probe_url)
                            else:
                                first_tile = _download_tile_compat(
                                    trial_session,
                                    probe_url,
                                    timeout,
                                    transport=transport,
                                )
                            if first_tile is not None:
                                tile_ext = ext
                                chosen_transport = transport
                                session = trial_session
                                break
                        if first_tile is not None:
                            break
                    if first_tile is not None:
                        _log(f"Referer selected (proxy-bypass): {ref}")
                        break

            if first_tile is None:
                status_hint = ", ".join(f"{ext}:{code}" for ext, code in first_tile_statuses.items()) or "n/a"
                proxy_hint = ""
                if proxy_text:
                    proxy_hint = (
                        f" Proxy configured: `{proxy_text}`."
                        " If statuses are 0, check proxy reachability from Comfy runtime"
                        " (e.g. docker/local namespace mismatch)."
                    )
                raise RuntimeError(
                    f"First tile is unavailable at `{tiles_base}`. "
                    f"Tried extensions [{', '.join(tile_ext_candidates)}], statuses [{status_hint}]. "
                    f"Check `base_url`, `mw`, and `level`.{proxy_hint}"
                )
            _log(f"Transport selected: {chosen_transport}")
            _log(f"Tile extension selected: .{tile_ext}")
            dzi_info = _parse_dzi(session, dzi_url, timeout, transport=chosen_transport)
            if dzi_info is None:
                for alt_transport in transport_candidates:
                    if alt_transport == chosen_transport:
                        continue
                    dzi_info = _parse_dzi(session, dzi_url, timeout, transport=alt_transport)
                    if isinstance(dzi_info, dict):
                        _log(f"DZI transport fallback selected: {alt_transport}")
                        break

            tile_size = int(dzi_info["tile_size"]) if isinstance(dzi_info, dict) else int(first_tile.size[0])
            if isinstance(dzi_info, dict):
                width, height, tiles_x, tiles_y = _compute_level_geometry_from_dzi(dzi_info, int(level))
                _log(
                    f"Geometry source=DZI, tile_size={tile_size}, "
                    f"canvas={int(width)}x{int(height)}, grid={int(tiles_x)}x{int(tiles_y)}"
                )
            else:
                tiles_x_probe = _probe_axis_count_compat(
                    session,
                    tiles_base,
                    tile_ext=tile_ext,
                    transport=chosen_transport,
                    axis="x",
                    timeout=timeout,
                )
                tiles_y_probe = _probe_axis_count_compat(
                    session,
                    tiles_base,
                    tile_ext=tile_ext,
                    transport=chosen_transport,
                    axis="y",
                    timeout=timeout,
                )
                if tiles_x_probe <= 0 or tiles_y_probe <= 0:
                    raise RuntimeError("Could not probe tile grid (x/y tile counts are zero).")
                tiles_x = int(tiles_x_probe)
                tiles_y = int(tiles_y_probe)
                last_x_tile = _download_tile_compat(
                    session,
                    _tile_url(tiles_base, tiles_x - 1, 0, tile_ext),
                    timeout,
                    transport=chosen_transport,
                )
                last_y_tile = _download_tile_compat(
                    session,
                    _tile_url(tiles_base, 0, tiles_y - 1, tile_ext),
                    timeout,
                    transport=chosen_transport,
                )
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
                        tile = _download_tile_compat(
                            session,
                            _tile_url(tiles_base, x, y, tile_ext),
                            timeout,
                            transport=chosen_transport,
                        )
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
