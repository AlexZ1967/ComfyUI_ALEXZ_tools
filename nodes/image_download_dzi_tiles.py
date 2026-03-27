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

import html
import json
import math
import os
import re
import socket
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import numpy as np
import requests
import torch
from PIL import Image
from ..utils.interrupt import check_interrupt, is_interrupt_exception
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
_PAC_WARNED_ONCE = False
_COMMON_LOCAL_PROXY_URLS = (
    "http://127.0.0.1:10808",
    "http://127.0.0.1:7890",
    "http://127.0.0.1:7897",
    "http://127.0.0.1:20170",
    "http://127.0.0.1:8889",
)
_DZI_SITE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "dzi_sites.json"
_DZI_SITE_CONFIG_CACHE: dict[str, Any] | None = None


def _fallback_dzi_site_config() -> dict[str, Any]:
    """Return built-in DZI site config when external JSON is unavailable."""
    return {
        "default_site": "National Portrait Gallery UK",
        "sites": [
            {
                "key": "npg",
                "name": "National Portrait Gallery UK",
                "base_url": "https://collectionimages.npg.org.uk",
                "provider": "npg",
                "mw_prefix": "mw",
                "default_mw": "mw207134",
                "default_level": 11,
                "mw_format": "mw<digits>",
                "object_url_template": "{base_url}/zoom/{mw}",
                "dzi_url_template": "{base_url}/zoom/{mw}/zoomXML.dzi",
                "tile_url_template": "{base_url}/zoom/{mw}/zoomXML_files/{level}/{x}_{y}.{ext}",
                "url_scheme": "{base_url}/zoom/{mw}/zoomXML_files/{level}/{x}_{y}.{ext}",
            },
            {
                "key": "nla",
                "name": "National Library of Australia",
                "base_url": "https://nla.gov.au",
                "provider": "nla",
                "mw_prefix": "nla.obj-",
                "default_mw": "nla.obj-138204672",
                "default_level": 11,
                "mw_format": "nla.obj-<digits>",
                "object_url_template": "{base_url}/{mw}",
                "dzi_url_template": "{base_url}/{mw}/dzi?tile=",
                "tile_url_template": "{base_url}/{mw}/dzi?tile={level}/{x}_{y}.{ext}",
                "url_scheme": "{base_url}/{mw}/dzi?tile={level}/{x}_{y}.{ext}",
            },
        ],
    }


def _normalize_provider(provider: str | None) -> str:
    """Normalize provider selector into supported values."""
    value = str(provider or "auto").strip().lower()
    if value in {"npg", "nla"}:
        return value
    return "auto"


def _log(message: str) -> None:
    """Emit node logs to ComfyUI console."""
    print(f"[DZI] {message}")


def _load_dzi_site_config() -> dict[str, Any]:
    """Load DZI site catalog from JSON config with fallback defaults."""
    global _DZI_SITE_CONFIG_CACHE
    if _DZI_SITE_CONFIG_CACHE is not None:
        return _DZI_SITE_CONFIG_CACHE

    fallback = _fallback_dzi_site_config()
    try:
        payload = json.loads(_DZI_SITE_CONFIG_PATH.read_text(encoding="utf-8"))
        sites = payload.get("sites")
        if not isinstance(sites, list) or not sites:
            raise ValueError("`sites` must be a non-empty list")
        normalized_sites = []
        for raw_site in sites:
            if not isinstance(raw_site, dict):
                continue
            name = str(raw_site.get("name") or "").strip()
            base_url = str(raw_site.get("base_url") or "").strip().rstrip("/")
            provider = str(raw_site.get("provider") or "").strip().lower()
            if not name or not base_url or not provider:
                continue
            normalized_sites.append(
                {
                    "key": str(raw_site.get("key") or provider).strip().lower(),
                    "name": name,
                    "base_url": base_url,
                    "provider": provider,
                    "mw_prefix": str(raw_site.get("mw_prefix") or "").strip(),
                    "default_mw": str(raw_site.get("default_mw") or "").strip(),
                    "default_level": int(raw_site.get("default_level") or 11),
                    "mw_format": str(raw_site.get("mw_format") or "").strip(),
                    "object_url_template": str(raw_site.get("object_url_template") or "").strip(),
                    "dzi_url_template": str(raw_site.get("dzi_url_template") or "").strip(),
                    "tile_url_template": str(raw_site.get("tile_url_template") or "").strip(),
                    "url_scheme": str(raw_site.get("url_scheme") or "").strip(),
                }
            )
        if not normalized_sites:
            raise ValueError("no valid site entries found")
        default_site = str(payload.get("default_site") or normalized_sites[0]["name"]).strip()
        _DZI_SITE_CONFIG_CACHE = {
            "default_site": default_site,
            "sites": normalized_sites,
        }
    except Exception as exc:
        _log(
            f"Site config fallback: {_DZI_SITE_CONFIG_PATH} "
            f"({type(exc).__name__}: {exc})"
        )
        _DZI_SITE_CONFIG_CACHE = fallback
    return _DZI_SITE_CONFIG_CACHE


def _get_dzi_sites() -> list[dict[str, Any]]:
    """Return normalized list of configured DZI sites."""
    return list(_load_dzi_site_config().get("sites") or [])


def _get_dzi_site_choice_names() -> list[str]:
    """Return UI dropdown labels for configured DZI sites."""
    sites = _get_dzi_sites()
    names = [str(site.get("name") or "").strip() for site in sites]
    return [name for name in names if name] or ["National Portrait Gallery UK"]


def _get_default_dzi_site_name() -> str:
    """Return configured default site name for INPUT_TYPES."""
    payload = _load_dzi_site_config()
    default_name = str(payload.get("default_site") or "").strip()
    names = _get_dzi_site_choice_names()
    if default_name in names:
        return default_name
    return names[0]


def _resolve_dzi_site(site: str | None, mw: str | None = None) -> dict[str, Any]:
    """Resolve a configured DZI site from dropdown label, key, URL, or legacy base URL."""
    site_text = str(site or "").strip()
    mw_text = str(mw or "").strip()
    sites = _get_dzi_sites()

    if site_text:
        lowered = site_text.lower()
        for candidate in sites:
            if lowered in {
                str(candidate.get("name") or "").strip().lower(),
                str(candidate.get("key") or "").strip().lower(),
                str(candidate.get("base_url") or "").strip().rstrip("/").lower(),
            }:
                return dict(candidate)
        if "://" in site_text:
            detected_provider = _detect_dzi_provider(site_text, mw_text, None)
            return {
                "key": detected_provider,
                "name": site_text,
                "base_url": site_text.rstrip("/"),
                "provider": detected_provider,
                "mw_prefix": "",
                "default_mw": mw_text,
                "default_level": 11,
                "mw_format": "",
                "object_url_template": "",
                "dzi_url_template": "",
                "tile_url_template": "",
                "url_scheme": "",
            }

    detected_provider = _detect_dzi_provider(site_text, mw_text, None)
    for candidate in sites:
        if str(candidate.get("provider") or "").strip().lower() == detected_provider:
            return dict(candidate)

    return dict(sites[0]) if sites else dict(_fallback_dzi_site_config()["sites"][0])


def _log_fetch_error(transport: str, url: str, exc: Exception) -> None:
    """Emit deduplicated transport error details for network diagnostics."""
    key = f"{transport}:{type(exc).__name__}:{str(exc)}"
    if key in _FETCH_ERROR_SEEN:
        return
    _FETCH_ERROR_SEEN.add(key)
    _log(f"Fetch error [{transport}]: {url} ({type(exc).__name__}: {exc})")


def _normalize_site_mw(mw: str | None, site_config: dict[str, Any]) -> str:
    """Normalize site object id: digits-only input gets site prefix, full ids pass through."""
    raw_mw = str(mw or "").strip()
    if not raw_mw:
        return str(site_config.get("default_mw") or "").strip()
    if not raw_mw.isdigit():
        return raw_mw

    prefix = str(site_config.get("mw_prefix") or "").strip()
    if prefix:
        return f"{prefix}{raw_mw}"

    default_mw = str(site_config.get("default_mw") or "").strip()
    match = re.match(r"^(.*?)(\d+)$", default_mw)
    if match:
        return f"{match.group(1)}{raw_mw}"
    return raw_mw


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


def _format_dzi_template(
    template: str,
    *,
    base_url: str,
    mw: str,
    level: int | None = None,
    x: int | None = None,
    y: int | None = None,
    ext: str | None = None,
) -> str:
    """Render URL template for configured DZI site."""
    text = str(template or "").strip()
    if not text:
        raise ValueError("DZI URL template must not be empty.")
    data = {
        "base_url": str(base_url or "").strip().rstrip("/"),
        "mw": str(mw or "").strip(),
        "level": "" if level is None else int(level),
        "x": "" if x is None else int(x),
        "y": "" if y is None else int(y),
        "ext": str(ext or "").strip().lstrip("."),
    }
    try:
        return text.format(**data)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in DZI URL template: {exc}") from exc


def _detect_dzi_provider(base_url: str, mw: str, provider: str | None = None) -> str:
    """Detect supported DZI provider from explicit selection or URL/identifier hints."""
    normalized = _normalize_provider(provider)
    if normalized != "auto":
        return normalized
    base = str(base_url or "").strip().lower()
    module_id = str(mw or "").strip().lower()
    if "nla.gov.au" in base or module_id.startswith("nla.obj-"):
        return "nla"
    return "npg"


def _build_dzi_source_urls(
    base_url: str,
    mw: str,
    level: int,
    provider: str | None = None,
    site_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized DZI/tile URL scheme from config templates or legacy providers."""
    module_id = str(mw or "").strip()
    if not module_id:
        raise ValueError("`mw` must not be empty.")
    cfg = dict(site_config or {})
    base = str(base_url or cfg.get("base_url") or "").strip().rstrip("/")
    if not base:
        raise ValueError("`base_url` must not be empty.")

    object_template = str(cfg.get("object_url_template") or "").strip()
    dzi_template = str(cfg.get("dzi_url_template") or "").strip()
    tile_template = str(cfg.get("tile_url_template") or "").strip()
    provider_name = str(cfg.get("provider") or "").strip().lower() or _detect_dzi_provider(base, module_id, provider)

    if dzi_template and tile_template:
        zoom_base = _format_dzi_template(object_template or "{base_url}/{mw}", base_url=base, mw=module_id)
        dzi_url = _format_dzi_template(dzi_template, base_url=base, mw=module_id, level=int(level))
        first_tile_url = _format_dzi_template(
            tile_template,
            base_url=base,
            mw=module_id,
            level=int(level),
            x=0,
            y=0,
            ext="jpg",
        )
        tiles_base = str(tile_template)
        tile_url_mode = "template"
        return {
            "provider": provider_name or "custom",
            "zoom_base": zoom_base,
            "dzi_url": dzi_url,
            "tiles_base": tiles_base,
            "tile_url_mode": tile_url_mode,
            "tile_url_template": tile_template,
            "tile_example_url": first_tile_url,
            "referer_root": _origin_from_url(base) or _DEFAULT_REFERER.rstrip("/"),
        }

    if provider_name == "nla":
        dzi_base = f"{base}/{module_id}/dzi?tile="
        return {
            "provider": "nla",
            "zoom_base": f"{base}/{module_id}",
            "dzi_url": dzi_base,
            "tiles_base": dzi_base,
            "tile_url_mode": "query",
            "referer_root": _origin_from_url(base) or _DEFAULT_REFERER.rstrip("/"),
        }

    zoom_base = _build_zoom_base_url(base, module_id)
    return {
        "provider": "npg",
        "zoom_base": zoom_base,
        "dzi_url": f"{zoom_base}/zoomXML.dzi",
        "tiles_base": f"{zoom_base}/zoomXML_files/{int(level)}",
        "tile_url_mode": "path",
        "referer_root": _origin_from_url(zoom_base) or _DEFAULT_REFERER.rstrip("/"),
    }


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


def _tile_url(
    tiles_base: str,
    x: int,
    y: int,
    tile_ext: str = "jpg",
    *,
    level: int | None = None,
    mode: str = "path",
    base_url: str | None = None,
    mw: str | None = None,
) -> str:
    """Build tile URL for one tile coordinate."""
    ext = str(tile_ext or "jpg").strip().lower().lstrip(".") or "jpg"
    normalized_mode = str(mode or "path").strip().lower()
    if normalized_mode == "template":
        return _format_dzi_template(
            tiles_base,
            base_url=str(base_url or "").strip(),
            mw=str(mw or "").strip(),
            level=level,
            x=int(x),
            y=int(y),
            ext=ext,
        )
    if normalized_mode == "query":
        if level is None:
            raise ValueError("`level` is required for query tile mode.")
        return f"{tiles_base}{int(level)}/{int(x)}_{int(y)}.{ext}"
    return f"{tiles_base}/{int(x)}_{int(y)}.{ext}"


def _normalize_proxy_url(proxy_url: str) -> str:
    """Normalize proxy URL into scheme://host:port form when possible."""
    text = str(proxy_url or "").strip()
    if not text:
        return ""
    if text.upper() == "DIRECT":
        return ""
    if "://" not in text:
        text = f"http://{text}"
    return text


def _proxy_host_port(proxy_url: str) -> tuple[str, int] | None:
    """Extract host/port from proxy URL."""
    proxy_text = _normalize_proxy_url(proxy_url)
    if not proxy_text:
        return None
    try:
        parsed = urlsplit(proxy_text)
        host = str(parsed.hostname or "").strip()
        port = int(parsed.port or 0)
        if not host or port <= 0:
            return None
        return host, port
    except Exception:
        return None


def _is_proxy_reachable(proxy_url: str, timeout: float = 0.2) -> bool:
    """Check that proxy endpoint is reachable from current runtime."""
    host_port = _proxy_host_port(proxy_url)
    if host_port is None:
        return False
    host, port = host_port
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            return True
    except Exception:
        return False


def _env_proxy_urls(*, include_env: bool) -> list[str]:
    """Collect proxy URLs from common environment variables."""
    if not include_env:
        return []
    keys = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
    found: list[str] = []
    for key in keys:
        value = str(os.environ.get(key, "")).strip()
        if value:
            found.append(_normalize_proxy_url(value))
    dedup: list[str] = []
    seen: set[str] = set()
    for value in found:
        if value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return dedup


def _read_cmd_stdout(cmd: list[str], timeout: float = 1.5) -> str:
    """Run command and return stdout text, suppressing errors/timeouts."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            timeout=float(timeout),
        )
        return str(proc.stdout or "")
    except Exception:
        return ""


def _parse_windows_proxy_server(value: str) -> list[str]:
    """Parse WinINET ProxyServer string into proxy URLs."""
    text = str(value or "").strip()
    if not text:
        return []
    out: list[str] = []
    chunks = [part.strip() for part in text.split(";") if str(part).strip()]
    if not chunks:
        return []
    for chunk in chunks:
        if "=" in chunk:
            proto, addr = [p.strip() for p in chunk.split("=", 1)]
        else:
            proto, addr = "http", chunk.strip()
        proto_l = str(proto).lower()
        addr = str(addr).strip()
        if not addr:
            continue
        if proto_l in {"socks", "socks4", "socks5"}:
            out.append(f"socks5h://{addr}")
        else:
            out.append(_normalize_proxy_url(addr))
    return out


def _linux_system_proxy() -> tuple[list[str], list[str]]:
    """Read Linux desktop proxy configuration (GNOME gsettings)."""
    if not sys.platform.startswith("linux"):
        return [], []
    if not shutil.which("gsettings"):
        return [], []
    mode_raw = _read_cmd_stdout(["gsettings", "get", "org.gnome.system.proxy", "mode"])
    mode = str(mode_raw or "").strip().strip("'\"").lower()
    proxies: list[str] = []
    pac_urls: list[str] = []
    if mode == "manual":
        for proto in ("https", "http"):
            host_raw = _read_cmd_stdout(["gsettings", "get", f"org.gnome.system.proxy.{proto}", "host"])
            port_raw = _read_cmd_stdout(["gsettings", "get", f"org.gnome.system.proxy.{proto}", "port"])
            host = str(host_raw or "").strip().strip("'\"")
            try:
                port = int(str(port_raw or "").strip().strip("'\"") or "0")
            except Exception:
                port = 0
            if host and port > 0:
                proxies.append(_normalize_proxy_url(f"{host}:{port}"))
    elif mode == "auto":
        pac_raw = _read_cmd_stdout(["gsettings", "get", "org.gnome.system.proxy", "autoconfig-url"])
        pac_url = str(pac_raw or "").strip().strip("'\"")
        if pac_url:
            pac_urls.append(pac_url)
    return proxies, pac_urls


def _windows_system_proxy() -> tuple[list[str], list[str]]:
    """Read Windows proxy settings (WinINET + WinHTTP)."""
    if os.name != "nt":
        return [], []
    proxies: list[str] = []
    pac_urls: list[str] = []
    try:
        import winreg  # type: ignore

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            proxy_enable = int(winreg.QueryValueEx(key, "ProxyEnable")[0] or 0)
            if proxy_enable:
                proxy_server = str(winreg.QueryValueEx(key, "ProxyServer")[0] or "")
                proxies.extend(_parse_windows_proxy_server(proxy_server))
            try:
                pac_url = str(winreg.QueryValueEx(key, "AutoConfigURL")[0] or "").strip()
                if pac_url:
                    pac_urls.append(pac_url)
            except Exception:
                pass
    except Exception:
        pass

    netsh_out = _read_cmd_stdout(["netsh", "winhttp", "show", "proxy"])
    for line in str(netsh_out or "").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_l = key.lower()
        if "proxy server" in key_l:
            proxies.extend(_parse_windows_proxy_server(value))
        elif "auto-config url" in key_l:
            pac_url = str(value or "").strip()
            if pac_url:
                pac_urls.append(pac_url)
    return proxies, pac_urls


def _mac_system_proxy() -> tuple[list[str], list[str]]:
    """Read macOS proxy settings (scutil --proxy)."""
    if sys.platform != "darwin":
        return [], []
    out = _read_cmd_stdout(["scutil", "--proxy"])
    fields: dict[str, str] = {}
    for line in str(out or "").splitlines():
        if ":" not in line:
            continue
        k, v = [part.strip() for part in line.split(":", 1)]
        fields[k] = v
    proxies: list[str] = []
    pac_urls: list[str] = []
    if fields.get("HTTPEnable") == "1":
        host = str(fields.get("HTTPProxy", "")).strip()
        port = str(fields.get("HTTPPort", "")).strip()
        if host and port:
            proxies.append(_normalize_proxy_url(f"{host}:{port}"))
    if fields.get("HTTPSEnable") == "1":
        host = str(fields.get("HTTPSProxy", "")).strip()
        port = str(fields.get("HTTPSPort", "")).strip()
        if host and port:
            proxies.append(_normalize_proxy_url(f"{host}:{port}"))
    pac_url = str(fields.get("ProxyAutoConfigURLString", "")).strip()
    if pac_url:
        pac_urls.append(pac_url)
    return proxies, pac_urls


def _system_proxy_urls_and_pac() -> tuple[list[str], list[str]]:
    """Collect system proxy URLs and PAC URLs across supported OSes."""
    proxies: list[str] = []
    pac_urls: list[str] = []
    for proxy_list, pac_list in (_linux_system_proxy(), _windows_system_proxy(), _mac_system_proxy()):
        proxies.extend(proxy_list)
        pac_urls.extend(pac_list)
    return proxies, pac_urls


def _pac_proxy_urls_for_target(*, pac_url: str, target_url: str, timeout: float = 4.0) -> list[str]:
    """Resolve proxies for target URL via PAC script if pacparser is available."""
    global _PAC_WARNED_ONCE
    pac = str(pac_url or "").strip()
    target = str(target_url or "").strip()
    if not pac or not target:
        return []
    try:
        import pacparser  # type: ignore
    except Exception:
        if not _PAC_WARNED_ONCE:
            _PAC_WARNED_ONCE = True
            _log("PAC URL detected, but `pacparser` is not installed. PAC will be skipped.")
        return []
    try:
        pac_script = ""
        if pac.lower().startswith(("http://", "https://")):
            response = requests.get(pac, timeout=float(timeout))
            if int(response.status_code) != 200:
                return []
            pac_script = str(response.text or "")
        else:
            # File path or file:// URL.
            if pac.lower().startswith("file://"):
                pac = pac[7:]
            with open(pac, "r", encoding="utf-8", errors="ignore") as f:
                pac_script = f.read()
        if not pac_script:
            return []
        pacparser.init()
        try:
            pacparser.parse_pac_string(pac_script)
            found = str(pacparser.find_proxy(target) or "")
        finally:
            pacparser.cleanup()
        out: list[str] = []
        for chunk in [part.strip() for part in found.split(";") if str(part).strip()]:
            chunk_u = chunk.upper()
            if chunk_u == "DIRECT":
                continue
            if re.match(r"^(PROXY|HTTP|HTTPS)\s+", chunk_u):
                addr = chunk.split(None, 1)[1].strip() if " " in chunk else ""
                if addr:
                    out.append(_normalize_proxy_url(addr))
                continue
            if re.match(r"^(SOCKS|SOCKS4|SOCKS5)\s+", chunk_u):
                addr = chunk.split(None, 1)[1].strip() if " " in chunk else ""
                if addr:
                    out.append(f"socks5h://{addr}")
        return out
    except Exception:
        return []


def _auto_proxy_candidates(*, include_env: bool, target_url: str) -> list[str]:
    """Return proxy candidates for automatic fallback."""
    candidates: list[str] = []
    candidates.extend(_env_proxy_urls(include_env=include_env))
    system_urls, pac_urls = _system_proxy_urls_and_pac()
    candidates.extend([_normalize_proxy_url(p) for p in system_urls if str(p or "").strip()])
    if include_env:
        for key in ("PROXY_PAC_URL", "proxy_pac_url", "AUTO_PROXY_URL", "auto_proxy_url"):
            pac_env = str(os.environ.get(key, "")).strip()
            if pac_env:
                pac_urls.append(pac_env)
    pac_seen: set[str] = set()
    for pac_url in pac_urls:
        pac = str(pac_url or "").strip()
        if not pac or pac in pac_seen:
            continue
        pac_seen.add(pac)
        candidates.extend(_pac_proxy_urls_for_target(pac_url=pac, target_url=target_url))
    for proxy_url in _COMMON_LOCAL_PROXY_URLS:
        if _is_proxy_reachable(proxy_url):
            candidates.append(_normalize_proxy_url(proxy_url))
    dedup: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        text = _normalize_proxy_url(str(value or "").strip())
        if not text or text in seen:
            continue
        seen.add(text)
        dedup.append(text)
    return dedup


def _build_proxy_profiles(
    *,
    explicit_proxy: str,
    trust_env_primary: bool,
    target_url: str,
) -> list[dict[str, Any]]:
    """Build ordered connection profiles for proxy/direct attempts."""
    proxy_text = _normalize_proxy_url(explicit_proxy)
    profiles: list[dict[str, Any]] = []
    if proxy_text:
        profiles.append({"name": "explicit_proxy", "proxy_url": proxy_text, "trust_env": trust_env_primary})
        if trust_env_primary:
            profiles.append({"name": "explicit_proxy_no_env", "proxy_url": proxy_text, "trust_env": False})
    else:
        # Prefer explicit env/direct first, then concrete auto-proxy candidates, then strict direct.
        if trust_env_primary:
            profiles.append({"name": "env_or_direct", "proxy_url": "", "trust_env": True})
        auto_proxy_candidates = _auto_proxy_candidates(
            include_env=trust_env_primary,
            target_url=target_url,
        )
        for idx, detected_proxy in enumerate(auto_proxy_candidates, start=1):
            profiles.append(
                {
                    "name": f"auto_proxy_{idx}",
                    "proxy_url": str(detected_proxy),
                    "trust_env": False,
                }
            )
        profiles.append({"name": "direct_no_env", "proxy_url": "", "trust_env": False})

    dedup: list[dict[str, Any]] = []
    seen: set[tuple[str, bool]] = set()
    for profile in profiles:
        key = (str(profile.get("proxy_url") or "").strip(), bool(profile.get("trust_env")))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(profile)
    return dedup


def _fetch_bytes_requests(session: requests.Session, url: str, timeout: float) -> tuple[int, bytes | None]:
    """Fetch URL bytes via requests transport."""
    try:
        check_interrupt()
        response = session.get(url, timeout=timeout)
        check_interrupt()
        return int(response.status_code), bytes(response.content or b"")
    except Exception as exc:
        _log_fetch_error("requests", url, exc)
        return 0, None


def _fetch_bytes_urllib(session: requests.Session, url: str, timeout: float) -> tuple[int, bytes | None]:
    """Fetch URL bytes via urllib transport."""
    req = Request(url, headers={k: str(v) for k, v in session.headers.items()})
    proxy_url = str(getattr(session, "_alexz_proxy_url", "") or "").strip()
    try:
        check_interrupt()
        if proxy_url:
            opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
            resp_obj = opener.open(req, timeout=timeout)
        else:
            resp_obj = urlopen(req, timeout=timeout)
        with resp_obj as resp:
            status = int(getattr(resp, "status", 0) or resp.getcode() or 0)
            body = resp.read()
            check_interrupt()
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
    check_interrupt()
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
        check_interrupt()
        proc = subprocess.run(cmd, capture_output=True, check=False)
        check_interrupt()
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
        check_interrupt()
        response = scraper.get(url, timeout=timeout)
        check_interrupt()
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
            check_interrupt()
            status, _ = _fetch_bytes(session, url, timeout, transport=transport)
            return int(status)
        except Exception as exc:
            if is_interrupt_exception(exc):
                raise
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
        check_interrupt()
        status, content = _fetch_bytes(session, url, timeout, transport=transport)
        if int(status) != 200:
            _log(f"Tile unavailable: {url} (status={int(status)})")
            return None
        return _decode_tile_image(content, url)
    except Exception as exc:
        if is_interrupt_exception(exc):
            raise
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
    level: int | None = None,
    tile_url_mode: str = "path",
    base_url: str | None = None,
    mw: str | None = None,
) -> int:
    """Probe tile count on one axis using robust status checks."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be `x` or `y`")
    last_success = -1
    misses_after_success = 0
    for i in range(max_tiles):
        check_interrupt()
        x = i if axis == "x" else 0
        y = i if axis == "y" else 0
        status = _http_status(
            session,
            _tile_url(
                tiles_base,
                x,
                y,
                tile_ext,
                level=level,
                mode=tile_url_mode,
                base_url=base_url,
                mw=mw,
            ),
            timeout,
            transport=transport,
        )
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
    level: int | None = None,
    tile_url_mode: str = "path",
    base_url: str | None = None,
    mw: str | None = None,
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
            level=level,
            tile_url_mode=tile_url_mode,
            base_url=base_url,
            mw=mw,
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
        check_interrupt()
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
            check_interrupt()
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


def _resolve_dzi_request_context(site: str, mw: str, level: int) -> dict[str, Any]:
    """Resolve effective site/object request context for DZI download."""
    site_config = _resolve_dzi_site(site, mw)
    base_url = str(site_config.get("base_url") or "").strip()
    provider_name = str(site_config.get("provider") or "npg").strip().lower()
    effective_mw = _normalize_site_mw(mw, site_config)
    if not effective_mw:
        raise ValueError("`mw` is empty and selected site has no `default_mw` in config/dzi_sites.json.")
    effective_level = int(level)
    if effective_level < 0:
        effective_level = int(site_config.get("default_level") or 11)
    return {
        "site_config": site_config,
        "base_url": base_url,
        "provider_name": provider_name,
        "effective_mw": effective_mw,
        "effective_level": effective_level,
    }


def _parse_dzi_ids_text(ids_text: str) -> list[str]:
    """Parse multiline/comma-separated DZI ids, skipping blanks and comments."""
    values: list[str] = []
    for raw_line in str(ids_text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in re.split(r"[;,]+", line):
            token = str(part or "").strip()
            if token:
                values.append(token)
    return values


def _sanitize_filename_component(text: str) -> str:
    """Normalize user-facing filename fragment into safe portable text."""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(text or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "item"


def _extract_html_title(html_text: str) -> str:
    """Extract human-readable title from HTML metadata with safe fallback."""
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


def _fetch_dzi_object_title(
    session: requests.Session,
    object_url: str,
    *,
    timeout: float,
    transport: str,
    fallback_stem: str,
) -> str:
    """Fetch object page and extract title for filename use, falling back silently."""
    url = str(object_url or "").strip()
    if not url:
        return fallback_stem
    try:
        check_interrupt()
        mode = str(transport or "requests").strip().lower()
        if mode not in {"requests", "cloudscraper", "urllib", "curl"}:
            mode = "requests"
        status, content = _fetch_bytes(session, url, timeout, transport=mode)
        if int(status) != 200 or not content:
            return fallback_stem
        title = _extract_html_title(content.decode("utf-8", errors="ignore"))
        return _sanitize_filename_component(title) if title else fallback_stem
    except Exception:
        return fallback_stem


def _render_dzi_filename(
    filename_template: str,
    *,
    index: int,
    raw_id: str,
    effective_mw: str,
    site_config: dict[str, Any],
    effective_level: int,
    title_stem: str | None = None,
) -> str:
    """Render output filename stem for one batch item."""
    template = str(filename_template or "{mw}").strip() or "{mw}"
    site_name = str(site_config.get("name") or "").strip()
    site_key = str(site_config.get("key") or site_name).strip()
    title_value = _sanitize_filename_component(title_stem or effective_mw)
    data = {
        "index": int(index),
        "raw_id": _sanitize_filename_component(raw_id),
        "mw": _sanitize_filename_component(effective_mw),
        "id": _sanitize_filename_component(effective_mw),
        "title": title_value,
        "site": _sanitize_filename_component(site_name),
        "site_key": _sanitize_filename_component(site_key),
        "level": int(effective_level),
    }
    try:
        rendered = template.format(**data)
    except Exception:
        rendered = data["title"] if "{title" in template else data["mw"]
    return _sanitize_filename_component(rendered)


def _resolve_unique_output_path(output_dir: str, stem: str, ext: str, overwrite_mode: str) -> tuple[str, str]:
    """Resolve final output path according to overwrite strategy."""
    normalized_ext = str(ext or "png").strip().lower().lstrip(".") or "png"
    base_path = os.path.join(output_dir, f"{stem}.{normalized_ext}")
    mode = str(overwrite_mode or "skip").strip().lower()
    if mode == "overwrite":
        return base_path, "overwrite"
    if mode == "unique":
        if not os.path.exists(base_path):
            return base_path, "unique_new"
        index = 2
        while True:
            candidate = os.path.join(output_dir, f"{stem}_{index}.{normalized_ext}")
            if not os.path.exists(candidate):
                return candidate, "unique_suffix"
            index += 1
    return base_path, "skip"


def _tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert Comfy IMAGE tensor [1,H,W,3] to PIL RGB image."""
    if image_tensor.ndim != 4 or int(image_tensor.shape[0]) < 1:
        raise ValueError("Expected IMAGE tensor with shape [1,H,W,3].")
    image = image_tensor[0].detach().cpu().clamp(0.0, 1.0).numpy()
    return Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _save_pil_image(image: Image.Image, output_path: str, output_extension: str) -> None:
    """Save PIL image to disk using requested output extension."""
    ext = str(output_extension or "png").strip().lower().lstrip(".") or "png"
    image_rgb = image.convert("RGB")
    if ext == "png":
        image_rgb.save(output_path, format="PNG", optimize=True)
        return
    if ext in {"jpg", "jpeg"}:
        image_rgb.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
        return
    if ext == "webp":
        image_rgb.save(output_path, format="WEBP", quality=95, method=6)
        return
    raise ValueError("Unsupported output_extension. Allowed: png, jpg, jpeg, webp.")


class ImageDownloadDZITiles:
    """ComfyUI node that downloads and assembles DZI tile images."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "site": (
                    _get_dzi_site_choice_names(),
                    {
                        "default": _get_default_dzi_site_name(),
                        "tooltip": "Сайт-источник DZI. Список и настройки сайтов берутся из config/dzi_sites.json.",
                    },
                ),
                "mw": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Идентификатор изображения. Можно ввести только цифры: нода сама добавит префикс сайта. Пусто = использовать default_mw выбранного сайта из config/dzi_sites.json.",
                    },
                ),
                "level": (
                    "INT",
                    {
                        "default": 11,
                        "min": -1,
                        "max": 32,
                        "tooltip": "Уровень DZI-тайлов. Значение -1 = использовать default_level выбранного сайта из config/dzi_sites.json.",
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
                        "tooltip": "Явный HTTP(S) proxy (например http://127.0.0.1:7890). Пусто = автоопределение (env + локальные порты).",
                    },
                ),
                "tile_extension": (
                    ["jpg", "jpeg", "png", "webp"],
                    {
                        "default": "jpg",
                        "tooltip": "Расширение тайлов на сервере. Используется только выбранный формат, без перебора остальных.",
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Папка для сохранения итоговой собранной картинки. Пусто = не сохранять на диск, только вернуть IMAGE.",
                    },
                ),
                "output_extension": (
                    ["png", "jpg", "jpeg", "webp"],
                    {
                        "default": "png",
                        "tooltip": "Формат записи итоговой собранной картинки на диск. Используется только если output_dir не пустой.",
                    },
                ),
                "filename_mode": (
                    ["mw", "title_or_mw"],
                    {
                        "default": "mw",
                        "tooltip": "mw = имя файла по идентификатору. title_or_mw = попытаться взять осмысленный title со страницы объекта, иначе fallback на mw.",
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
        site: str,
        mw: str,
        level: int,
        transport: str = "auto",
        proxy_url: str = "",
        tile_extension: str = "jpg",
        output_dir: str = "",
        output_extension: str = "png",
        filename_mode: str = "mw",
    ):
        """Download DZI tiles for selected level and return assembled image tensor."""
        try:
            check_interrupt()
            request_ctx = _resolve_dzi_request_context(site, mw, level)
            site_config = dict(request_ctx["site_config"])
            base_url = str(request_ctx["base_url"])
            provider_name = str(request_ctx["provider_name"])
            effective_mw = str(request_ctx["effective_mw"])
            effective_level = int(request_ctx["effective_level"])

            source_urls = _build_dzi_source_urls(
                base_url,
                effective_mw,
                effective_level,
                provider_name,
                site_config=site_config,
            )
            provider_name = str(source_urls["provider"])
            zoom_base = str(source_urls["zoom_base"])
            dzi_url = str(source_urls["dzi_url"])
            tiles_base = str(source_urls["tiles_base"])
            tile_url_mode = str(source_urls.get("tile_url_mode") or "path")
            tile_example_url = str(source_urls.get("tile_example_url") or "")
            referer_root = str(source_urls.get("referer_root") or _origin_from_url(zoom_base) or _DEFAULT_REFERER.rstrip("/"))
            referer_candidates = [
                f"{zoom_base.rstrip('/')}/",
                f"{referer_root.rstrip('/')}/",
                _DEFAULT_REFERER,
            ]
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
            _log(f"Start download: mw={effective_mw}, level={effective_level}")
            _log(f"Site: {site_config.get('name')}")
            _log(f"Provider: {provider_name}")
            _log(f"Base: {zoom_base}")
            _log(f"DZI: {dzi_url}")
            _log(f"Tiles: {tile_example_url or tiles_base}")
            selected_tile_ext = str(tile_extension or "jpg").strip().lower().lstrip(".")
            if selected_tile_ext not in {"jpg", "jpeg", "png", "webp"}:
                raise ValueError(
                    f"Unsupported tile_extension `{tile_extension}`. "
                    "Allowed: jpg, jpeg, png, webp."
                )
            # Do not query DZI metadata before first tile probe: some hosts can
            # deny `.dzi` while still allowing tile images.
            selected_transport = str(transport or "auto").strip().lower()
            if selected_transport in {"requests", "cloudscraper", "urllib", "curl"}:
                transport_candidates = [selected_transport]
            else:
                transport_candidates = ["requests", "cloudscraper", "urllib", "curl"]
            proxy_text = str(proxy_url or "").strip()
            trust_env_primary = True
            first_tile = None
            tile_ext = ""
            chosen_transport = ""
            first_tile_statuses: dict[str, int] = {}
            session = None
            preflight_timeout = min(8.0, float(timeout))
            preflight_url = _tile_url(
                tiles_base,
                0,
                0,
                selected_tile_ext,
                level=effective_level,
                mode=tile_url_mode,
                base_url=base_url,
                mw=effective_mw,
            )
            proxy_profiles = _build_proxy_profiles(
                explicit_proxy=proxy_text,
                trust_env_primary=trust_env_primary,
                target_url=preflight_url,
            )

            chosen_proxy_url = ""
            chosen_profile_name = ""
            # Fast proxy preflight: detect working network route on canonical tile
            # and avoid costly ext/transport/profile cartesian probing.
            preflight_referer = referer_candidates[0] if referer_candidates else _DEFAULT_REFERER
            preflight_attempts: list[dict[str, Any]] = []
            for profile in proxy_profiles:
                preflight_attempts.append(profile)
            # If auto-proxy is requested, try local reachable proxies first to avoid
            # long 403 loops in env/direct profile.
            if not proxy_text:
                local_first = [
                    p
                    for p in preflight_attempts
                    if str(p.get("proxy_url") or "").strip().startswith("http://127.0.0.1:")
                ]
                if local_first:
                    remaining = [p for p in preflight_attempts if p not in local_first]
                    preflight_attempts = local_first + remaining
            for profile in preflight_attempts:
                check_interrupt()
                profile_proxy = str(profile.get("proxy_url") or "").strip()
                profile_trust_env = bool(profile.get("trust_env"))
                profile_name = str(profile.get("name") or "profile")
                trial_session = _make_session(
                    referer=preflight_referer,
                    origin=referer_root,
                    trust_env=profile_trust_env,
                    proxy_url=profile_proxy,
                )
                for candidate_transport in transport_candidates:
                    check_interrupt()
                    status, content = _fetch_bytes(
                        trial_session,
                        preflight_url,
                        preflight_timeout,
                        transport=candidate_transport,
                    )
                    first_tile_statuses[
                        f"preflight:{selected_tile_ext}@{candidate_transport}|{profile_name}:{profile_proxy or '-'}"
                    ] = int(status)
                    if int(status) == 200:
                        maybe_tile = _decode_tile_image(content, preflight_url)
                    else:
                        maybe_tile = None
                        # Compatibility fallback for mocked tests where status path is stubbed.
                        if int(status) <= 0:
                            maybe_tile = _download_tile_compat(
                                trial_session,
                                preflight_url,
                                preflight_timeout,
                                transport=candidate_transport,
                            )
                    if maybe_tile is None:
                        continue
                    first_tile = maybe_tile
                    tile_ext = selected_tile_ext
                    chosen_transport = candidate_transport
                    session = trial_session
                    chosen_proxy_url = profile_proxy
                    chosen_profile_name = profile_name
                    _log(f"Referer selected: {preflight_referer}")
                    _log(f"Proxy preflight selected: {profile_name} ({profile_proxy or 'direct'})")
                    break
                if first_tile is not None:
                    break

            # Full fallback probing only if preflight failed.
            for profile in proxy_profiles:
                if first_tile is not None:
                    break
                check_interrupt()
                profile_proxy = str(profile.get("proxy_url") or "").strip()
                profile_trust_env = bool(profile.get("trust_env"))
                profile_name = str(profile.get("name") or "profile")
                for ref_idx, ref in enumerate(referer_candidates):
                    check_interrupt()
                    trial_session = _make_session(
                        referer=ref,
                        origin=referer_root,
                        trust_env=profile_trust_env,
                        proxy_url=profile_proxy,
                    )
                    probe_url = _tile_url(
                        tiles_base,
                        0,
                        0,
                        selected_tile_ext,
                        level=effective_level,
                        mode=tile_url_mode,
                        base_url=base_url,
                        mw=effective_mw,
                    )
                    for transport in transport_candidates:
                        check_interrupt()
                        status, content = _fetch_bytes(trial_session, probe_url, timeout, transport=transport)
                        first_tile_statuses[
                            f"{selected_tile_ext}@{transport}#r{ref_idx+1}|{profile_name}:{profile_proxy or '-'}"
                        ] = int(status)
                        if int(status) == 200:
                            first_tile = _decode_tile_image(content, probe_url)
                        else:
                            first_tile = None
                            # Compatibility fallback for mocked/legacy paths where
                            # status probes are stubbed but `_download_tile` returns data.
                            if int(status) <= 0:
                                first_tile = _download_tile_compat(
                                    trial_session,
                                    probe_url,
                                    timeout,
                                    transport=transport,
                                )
                        if first_tile is not None:
                            tile_ext = selected_tile_ext
                            chosen_transport = transport
                            session = trial_session
                            chosen_proxy_url = profile_proxy
                            chosen_profile_name = profile_name
                            _log(f"Referer selected: {ref}")
                            break
                    if first_tile is not None:
                        break
                if first_tile is not None:
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
                    f"Tried extension [{selected_tile_ext}], statuses [{status_hint}]. "
                    f"Check `site`, `mw`, `level`, and `tile_extension`.{proxy_hint}"
                )
            _log(f"Transport selected: {chosen_transport}")
            if chosen_proxy_url:
                _log(f"Proxy selected: {chosen_proxy_url} ({chosen_profile_name})")
            else:
                _log(f"Proxy selected: direct ({chosen_profile_name or 'direct'})")
            _log(f"Tile extension selected: .{tile_ext}")
            dzi_info = _parse_dzi(session, dzi_url, timeout, transport=chosen_transport)
            if dzi_info is None:
                for alt_transport in transport_candidates:
                    check_interrupt()
                    if alt_transport == chosen_transport:
                        continue
                    dzi_info = _parse_dzi(session, dzi_url, timeout, transport=alt_transport)
                    if isinstance(dzi_info, dict):
                        _log(f"DZI transport fallback selected: {alt_transport}")
                        break

            tile_size = int(dzi_info["tile_size"]) if isinstance(dzi_info, dict) else int(first_tile.size[0])
            if isinstance(dzi_info, dict):
                width, height, tiles_x, tiles_y = _compute_level_geometry_from_dzi(dzi_info, effective_level)
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
                    level=effective_level,
                    tile_url_mode=tile_url_mode,
                    base_url=base_url,
                    mw=effective_mw,
                )
                tiles_y_probe = _probe_axis_count_compat(
                    session,
                    tiles_base,
                    tile_ext=tile_ext,
                    transport=chosen_transport,
                    axis="y",
                    timeout=timeout,
                    level=effective_level,
                    tile_url_mode=tile_url_mode,
                    base_url=base_url,
                    mw=effective_mw,
                )
                if tiles_x_probe <= 0 or tiles_y_probe <= 0:
                    raise RuntimeError("Could not probe tile grid (x/y tile counts are zero).")
                tiles_x = int(tiles_x_probe)
                tiles_y = int(tiles_y_probe)
                last_x_tile = _download_tile_compat(
                    session,
                    _tile_url(
                        tiles_base,
                        tiles_x - 1,
                        0,
                        tile_ext,
                        level=effective_level,
                        mode=tile_url_mode,
                        base_url=base_url,
                        mw=effective_mw,
                    ),
                    timeout,
                    transport=chosen_transport,
                )
                last_y_tile = _download_tile_compat(
                    session,
                    _tile_url(
                        tiles_base,
                        0,
                        tiles_y - 1,
                        tile_ext,
                        level=effective_level,
                        mode=tile_url_mode,
                        base_url=base_url,
                        mw=effective_mw,
                    ),
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
                    check_interrupt()
                    for x in range(tiles_x):
                        check_interrupt()
                        if x == 0 and y == 0:
                            continue
                        tile = _download_tile_compat(
                            session,
                            _tile_url(
                                tiles_base,
                                x,
                                y,
                                tile_ext,
                                level=effective_level,
                                mode=tile_url_mode,
                                base_url=base_url,
                                mw=effective_mw,
                            ),
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
            output_dir_text = str(output_dir or "").strip()
            if output_dir_text:
                output_dir_path = os.path.abspath(os.path.expanduser(output_dir_text))
                os.makedirs(output_dir_path, exist_ok=True)
                filename_mode_text = str(filename_mode or "mw").strip().lower()
                filename_stem = _sanitize_filename_component(effective_mw)
                if filename_mode_text == "title_or_mw":
                    filename_stem = _fetch_dzi_object_title(
                        session,
                        zoom_base,
                        timeout=timeout,
                        transport=chosen_transport,
                        fallback_stem=filename_stem,
                    )
                filename_stem = _render_dzi_filename(
                    "{title}" if filename_mode_text == "title_or_mw" else "{mw}",
                    index=0,
                    raw_id=effective_mw,
                    effective_mw=effective_mw,
                    site_config=site_config,
                    effective_level=effective_level,
                    title_stem=filename_stem,
                )
                output_path, _ = _resolve_unique_output_path(
                    output_dir_path,
                    filename_stem,
                    output_extension,
                    "unique",
                )
                _save_pil_image(canvas, output_path, output_extension)
                _log(f"Saved assembled image: {output_path}")
            return (_image_to_tensor(canvas),)
        except Exception as exc:
            if is_interrupt_exception(exc):
                _log("Node interrupted by ComfyUI.")
                raise
            _log(f"Node failed: {type(exc).__name__}: {exc}")
            _log(traceback.format_exc().rstrip())
            raise


class ImageDownloadDZITilesBatchSave:
    """ComfyUI node that downloads multiple DZI images and saves them to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema for batch DZI download/save."""
        return {
            "required": {
                "site": (
                    _get_dzi_site_choice_names(),
                    {
                        "default": _get_default_dzi_site_name(),
                        "tooltip": "Сайт-источник DZI. Список и настройки сайтов берутся из config/dzi_sites.json.",
                    },
                ),
                "ids_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Список ID, по одному на строку. Допустимы также разделители ',' и ';'. Пустые строки и строки, начинающиеся с '#', игнорируются.",
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Папка для сохранения итоговых изображений.",
                    },
                ),
                "level": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 32,
                        "tooltip": "Уровень DZI-тайлов. Значение -1 = использовать default_level выбранного сайта.",
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
                        "tooltip": "Явный HTTP(S) proxy. Пусто = автоопределение (env + локальные порты).",
                    },
                ),
                "tile_extension": (
                    ["jpg", "jpeg", "png", "webp"],
                    {
                        "default": "jpg",
                        "tooltip": "Расширение тайлов на сервере.",
                    },
                ),
                "output_extension": (
                    ["png", "jpg", "jpeg", "webp"],
                    {
                        "default": "png",
                        "tooltip": "Формат сохранения итоговой картинки на диск.",
                    },
                ),
                "filename_template": (
                    "STRING",
                    {
                        "default": "{mw}",
                        "multiline": False,
                        "tooltip": "Шаблон имени файла без расширения. Поддерживаются: {index}, {raw_id}, {mw}, {id}, {title}, {site}, {site_key}, {level}. {title} пытается взять title со страницы объекта, иначе fallback на mw.",
                    },
                ),
                "overwrite_mode": (
                    ["skip", "overwrite", "unique"],
                    {
                        "default": "skip",
                        "tooltip": "Поведение при существующем файле: skip / overwrite / unique.",
                    },
                ),
                "continue_on_error": (
                    ["true", "false"],
                    {
                        "default": "true",
                        "tooltip": "Продолжать обработку после ошибки отдельного ID.",
                    },
                ),
                "save_mode": (
                    ["save_only", "save_and_manifest"],
                    {
                        "default": "save_and_manifest",
                        "tooltip": "save_only = только изображения. save_and_manifest = дополнительно записать dzi_batch_manifest.json.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("manifest_json", "saved_paths_json", "count_ok", "count_failed")
    FUNCTION = "download_batch"
    CATEGORY = "image/io"
    OUTPUT_NODE = True

    def download_batch(
        self,
        site: str,
        ids_text: str,
        output_dir: str,
        level: int,
        transport: str = "auto",
        proxy_url: str = "",
        tile_extension: str = "jpg",
        output_extension: str = "png",
        filename_template: str = "{mw}",
        overwrite_mode: str = "skip",
        continue_on_error: str = "true",
        save_mode: str = "save_and_manifest",
    ):
        """Download multiple DZI images, save them to disk, and return manifest data."""
        check_interrupt()
        output_dir_raw = str(output_dir or "").strip()
        if not output_dir_raw:
            raise ValueError("`output_dir` must not be empty.")
        output_dir_text = os.path.abspath(os.path.expanduser(output_dir_raw))
        os.makedirs(output_dir_text, exist_ok=True)

        parsed_ids = _parse_dzi_ids_text(ids_text)
        if not parsed_ids:
            raise ValueError("`ids_text` does not contain any valid IDs.")

        continue_flag = str(continue_on_error or "true").strip().lower() == "true"
        batch_manifest: dict[str, Any] = {
            "site": str(site or "").strip(),
            "output_dir": output_dir_text,
            "level": int(level),
            "tile_extension": str(tile_extension or "jpg").strip().lower(),
            "output_extension": str(output_extension or "png").strip().lower(),
            "filename_template": str(filename_template or "{mw}"),
            "overwrite_mode": str(overwrite_mode or "skip"),
            "save_mode": str(save_mode or "save_only"),
            "items": [],
        }
        saved_paths: list[str] = []
        count_ok = 0
        count_failed = 0
        single_node = ImageDownloadDZITiles()

        for index, raw_id in enumerate(parsed_ids, start=1):
            check_interrupt()
            request_ctx = _resolve_dzi_request_context(site, raw_id, level)
            site_config = dict(request_ctx["site_config"])
            base_url = str(request_ctx["base_url"])
            provider_name = str(request_ctx["provider_name"])
            effective_mw = str(request_ctx["effective_mw"])
            effective_level = int(request_ctx["effective_level"])
            title_stem = _sanitize_filename_component(effective_mw)
            if "{title" in str(filename_template or ""):
                source_urls = _build_dzi_source_urls(
                    base_url,
                    effective_mw,
                    effective_level,
                    provider_name,
                    site_config=site_config,
                )
                zoom_base = str(source_urls["zoom_base"])
                referer_root = str(source_urls.get("referer_root") or _origin_from_url(zoom_base) or _DEFAULT_REFERER.rstrip("/"))
                title_session = _make_session(
                    referer=f"{zoom_base.rstrip('/')}/",
                    origin=referer_root,
                    trust_env=True,
                    proxy_url=str(proxy_url or "").strip(),
                )
                title_stem = _fetch_dzi_object_title(
                    title_session,
                    zoom_base,
                    timeout=20.0,
                    transport=str(transport or "auto").strip().lower(),
                    fallback_stem=title_stem,
                )
            filename_stem = _render_dzi_filename(
                filename_template,
                index=index,
                raw_id=raw_id,
                effective_mw=effective_mw,
                site_config=site_config,
                effective_level=effective_level,
                title_stem=title_stem,
            )
            output_path, output_policy = _resolve_unique_output_path(
                output_dir_text,
                filename_stem,
                output_extension,
                overwrite_mode,
            )
            item_record = {
                "index": index,
                "raw_id": raw_id,
                "mw": effective_mw,
                "level": effective_level,
                "path": output_path,
                "status": "pending",
            }

            try:
                check_interrupt()
                if output_policy == "skip" and os.path.exists(output_path):
                    item_record["status"] = "skipped_existing"
                    batch_manifest["items"].append(item_record)
                    _log(f"Batch skip existing: {output_path}")
                    continue

                image_tensor, = single_node.download(
                    site,
                    raw_id,
                    effective_level,
                    transport=transport,
                    proxy_url=proxy_url,
                    tile_extension=tile_extension,
                )
                check_interrupt()
                image = _tensor_image_to_pil(image_tensor)
                check_interrupt()
                _save_pil_image(image, output_path, output_extension)
                saved_paths.append(output_path)
                count_ok += 1
                item_record["status"] = "saved"
                batch_manifest["items"].append(item_record)
                _log(f"Batch saved [{index}/{len(parsed_ids)}]: {output_path}")
            except Exception as exc:
                if is_interrupt_exception(exc):
                    _log("Batch interrupted by ComfyUI.")
                    raise
                count_failed += 1
                item_record["status"] = "failed"
                item_record["error"] = f"{type(exc).__name__}: {exc}"
                batch_manifest["items"].append(item_record)
                _log(f"Batch failed [{index}/{len(parsed_ids)}]: {raw_id} ({type(exc).__name__}: {exc})")
                if not continue_flag:
                    batch_manifest["aborted"] = True
                    batch_manifest["abort_reason"] = item_record["error"]
                    break

        batch_manifest["count_total"] = len(parsed_ids)
        batch_manifest["count_ok"] = count_ok
        batch_manifest["count_failed"] = count_failed
        batch_manifest["count_skipped"] = sum(1 for item in batch_manifest["items"] if item["status"] == "skipped_existing")
        manifest_json = json.dumps(batch_manifest, ensure_ascii=True, indent=2)
        saved_paths_json = json.dumps(saved_paths, ensure_ascii=True, indent=2)

        if str(save_mode or "save_only").strip().lower() == "save_and_manifest":
            check_interrupt()
            manifest_path, _manifest_policy = _resolve_unique_output_path(
                output_dir_text,
                "dzi_batch_manifest",
                "json",
                "overwrite" if str(overwrite_mode or "skip").strip().lower() == "overwrite" else "unique",
            )
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(manifest_json)
            _log(f"Batch manifest saved: {manifest_path}")

        return (manifest_json, saved_paths_json, int(count_ok), int(count_failed))
