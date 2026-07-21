"""
Module: nodes/image_download_dzi_tiles_ops.py
Author: AlexZ1967
Last updated: 2026-07-21

Description:
    Pure helper functions for DZI site config, URL building, and filename policy.

Purpose:
    Separates deterministic Deep Zoom parsing and naming logic from the ComfyUI
    node adapter in `image_download_dzi_tiles.py` so the heavy node can be
    decomposed incrementally during Phase 4 without changing public node types.
"""

from __future__ import annotations

import html
import math
import os
import re
from typing import Any, Callable
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


def fallback_dzi_site_config() -> dict[str, Any]:
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


def normalize_provider(provider: str | None) -> str:
    """Normalize provider selector into supported values."""
    value = str(provider or "auto").strip().lower()
    if value in {"npg", "nla"}:
        return value
    return "auto"


def normalize_dzi_site_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON-backed DZI site config and validate required fields."""
    sites = payload.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("`sites` must be a non-empty list")

    normalized_sites: list[dict[str, Any]] = []
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
    return {
        "default_site": default_site,
        "sites": normalized_sites,
    }


def get_dzi_site_choice_names(sites: list[dict[str, Any]]) -> list[str]:
    """Return UI dropdown labels for configured DZI sites."""
    names = [str(site.get("name") or "").strip() for site in sites]
    return [name for name in names if name] or ["National Portrait Gallery UK"]


def get_default_dzi_site_name(payload: dict[str, Any]) -> str:
    """Return configured default site name for Comfy INPUT_TYPES."""
    default_name = str(payload.get("default_site") or "").strip()
    names = get_dzi_site_choice_names(list(payload.get("sites") or []))
    if default_name in names:
        return default_name
    return names[0]


def resolve_dzi_site(
    site: str | None,
    mw: str | None,
    *,
    sites: list[dict[str, Any]],
    detect_provider_fn: Callable[[str, str, str | None], str],
    fallback_site: dict[str, Any],
) -> dict[str, Any]:
    """Resolve configured DZI site from dropdown label, key, URL, or provider hints."""
    site_text = str(site or "").strip()
    mw_text = str(mw or "").strip()

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
            detected_provider = detect_provider_fn(site_text, mw_text, None)
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

    detected_provider = detect_provider_fn(site_text, mw_text, None)
    for candidate in sites:
        if str(candidate.get("provider") or "").strip().lower() == detected_provider:
            return dict(candidate)

    return dict(sites[0]) if sites else dict(fallback_site)


def normalize_site_mw(mw: str | None, site_config: dict[str, Any]) -> str:
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


def build_zoom_base_url(base_url: str, mw: str) -> str:
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


def format_dzi_template(
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


def detect_dzi_provider(base_url: str, mw: str, provider: str | None = None) -> str:
    """Detect supported DZI provider from explicit selection or URL/identifier hints."""
    normalized = normalize_provider(provider)
    if normalized != "auto":
        return normalized
    base = str(base_url or "").strip().lower()
    module_id = str(mw or "").strip().lower()
    if "nla.gov.au" in base or module_id.startswith("nla.obj-"):
        return "nla"
    return "npg"


def origin_from_url(url_text: str) -> str:
    """Extract URL origin (`scheme://host[:port]`) for request headers."""
    try:
        parsed = urlsplit(str(url_text or "").strip())
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def normalize_proxy_url(proxy_url: str) -> str:
    """Normalize a proxy address into a usable URL, treating ``DIRECT`` as empty."""
    text = str(proxy_url or "").strip()
    if not text or text.upper() == "DIRECT":
        return ""
    return text if "://" in text else f"http://{text}"


def proxy_host_port(proxy_url: str) -> tuple[str, int] | None:
    """Extract host and positive port from a normalized proxy URL."""
    proxy_text = normalize_proxy_url(proxy_url)
    if not proxy_text:
        return None
    try:
        parsed = urlsplit(proxy_text)
        host = str(parsed.hostname or "").strip()
        port = int(parsed.port or 0)
        return (host, port) if host and port > 0 else None
    except ValueError:
        return None


def env_proxy_urls(*, include_env: bool, environ: dict[str, str] | None = None) -> list[str]:
    """Collect and deduplicate proxy URLs from standard environment variables."""
    if not include_env:
        return []
    source = os.environ if environ is None else environ
    keys = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
    found = [normalize_proxy_url(str(source.get(key, "")).strip()) for key in keys]
    return list(dict.fromkeys(value for value in found if value))


def parse_windows_proxy_server(value: str) -> list[str]:
    """Parse a WinINET ``ProxyServer`` value into normalized proxy URLs."""
    text = str(value or "").strip()
    if not text:
        return []
    proxies: list[str] = []
    for chunk in (part.strip() for part in text.split(";") if part.strip()):
        proto, separator, address = chunk.partition("=")
        address = address.strip() if separator else proto.strip()
        proto = proto.strip().lower() if separator else "http"
        if not address:
            continue
        if proto in {"socks", "socks4", "socks5"}:
            proxies.append(f"socks5h://{address}")
        else:
            proxies.append(normalize_proxy_url(address))
    return proxies


def build_proxy_profiles(
    *,
    explicit_proxy: str,
    trust_env_primary: bool,
    auto_proxy_candidates: list[str],
) -> list[dict[str, Any]]:
    """Build stable, deduplicated proxy/direct connection profiles."""
    proxy_text = normalize_proxy_url(explicit_proxy)
    profiles: list[dict[str, Any]] = []
    if proxy_text:
        profiles.append({"name": "explicit_proxy", "proxy_url": proxy_text, "trust_env": trust_env_primary})
        if trust_env_primary:
            profiles.append({"name": "explicit_proxy_no_env", "proxy_url": proxy_text, "trust_env": False})
    else:
        if trust_env_primary:
            profiles.append({"name": "env_or_direct", "proxy_url": "", "trust_env": True})
        profiles.extend(
            {
                "name": f"auto_proxy_{index}",
                "proxy_url": normalize_proxy_url(proxy_url),
                "trust_env": False,
            }
            for index, proxy_url in enumerate(auto_proxy_candidates, start=1)
        )
        profiles.append({"name": "direct_no_env", "proxy_url": "", "trust_env": False})

    seen: set[tuple[str, bool]] = set()
    deduplicated: list[dict[str, Any]] = []
    for profile in profiles:
        key = (str(profile["proxy_url"]), bool(profile["trust_env"]))
        if key not in seen:
            seen.add(key)
            deduplicated.append(profile)
    return deduplicated


def build_dzi_source_urls(
    base_url: str,
    mw: str,
    level: int,
    provider: str | None = None,
    *,
    site_config: dict[str, Any] | None = None,
    default_referer: str,
) -> dict[str, Any]:
    """Build normalized DZI/tile URL scheme from config templates or provider heuristics."""
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
    provider_name = str(cfg.get("provider") or "").strip().lower() or detect_dzi_provider(base, module_id, provider)

    if dzi_template and tile_template:
        zoom_base = format_dzi_template(object_template or "{base_url}/{mw}", base_url=base, mw=module_id)
        dzi_url = format_dzi_template(dzi_template, base_url=base, mw=module_id, level=int(level))
        first_tile_url = format_dzi_template(
            tile_template,
            base_url=base,
            mw=module_id,
            level=int(level),
            x=0,
            y=0,
            ext="jpg",
        )
        return {
            "provider": provider_name or "custom",
            "zoom_base": zoom_base,
            "dzi_url": dzi_url,
            "tiles_base": tile_template,
            "tile_url_mode": "template",
            "tile_url_template": tile_template,
            "tile_example_url": first_tile_url,
            "referer_root": origin_from_url(base) or str(default_referer).rstrip("/"),
        }

    if provider_name == "nla":
        dzi_base = f"{base}/{module_id}/dzi?tile="
        return {
            "provider": "nla",
            "zoom_base": f"{base}/{module_id}",
            "dzi_url": dzi_base,
            "tiles_base": dzi_base,
            "tile_url_mode": "query",
            "referer_root": origin_from_url(base) or str(default_referer).rstrip("/"),
        }

    zoom_base = build_zoom_base_url(base, module_id)
    return {
        "provider": "npg",
        "zoom_base": zoom_base,
        "dzi_url": f"{zoom_base}/zoomXML.dzi",
        "tiles_base": f"{zoom_base}/zoomXML_files/{int(level)}",
        "tile_url_mode": "path",
        "referer_root": origin_from_url(zoom_base) or str(default_referer).rstrip("/"),
    }


def build_dzi_tile_url(
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
    """Build one tile URL for path, query, or template-backed DZI sources."""
    ext = str(tile_ext or "jpg").strip().lower().lstrip(".") or "jpg"
    normalized_mode = str(mode or "path").strip().lower()
    if normalized_mode == "template":
        return format_dzi_template(
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


def parse_dzi_metadata(content: bytes | None) -> dict[str, Any] | None:
    """Parse DZI XML bytes into tile dimensions, returning ``None`` when invalid."""
    if not content:
        return None
    try:
        root = ET.fromstring(content)
        tile_size = int(root.attrib.get("TileSize", "256"))
        overlap = int(root.attrib.get("Overlap", "0"))
        image_format = str(root.attrib.get("Format", "jpg"))
        size_el = next((el for el in root.iter() if str(el.tag).lower().endswith("size")), None)
        if size_el is None:
            return None
        return {
            "tile_size": tile_size,
            "overlap": overlap,
            "format": image_format,
            "width": int(size_el.attrib["Width"]),
            "height": int(size_el.attrib["Height"]),
        }
    except (ET.ParseError, KeyError, TypeError, ValueError):
        return None


def compute_dzi_level_geometry(dzi_info: dict[str, Any], level: int) -> tuple[int, int, int, int]:
    """Compute level-specific output size and tile grid from DZI metadata."""
    tile_size = max(1, int(dzi_info["tile_size"]))
    full_width = max(1, int(dzi_info["width"]))
    full_height = max(1, int(dzi_info["height"]))
    max_dim = max(full_width, full_height)
    max_level = int(math.ceil(math.log2(float(max_dim)))) if max_dim > 1 else 0
    scale_div = float(2 ** max(0, max_level - int(level)))
    level_width = max(1, int(math.ceil(float(full_width) / scale_div)))
    level_height = max(1, int(math.ceil(float(full_height) / scale_div)))
    tiles_x = max(1, int(math.ceil(float(level_width) / float(tile_size))))
    tiles_y = max(1, int(math.ceil(float(level_height) / float(tile_size))))
    return level_width, level_height, tiles_x, tiles_y


def resolve_dzi_request_context(
    site: str,
    mw: str,
    level: int,
    *,
    resolve_site_fn: Callable[[str | None, str | None], dict[str, Any]],
    normalize_site_mw_fn: Callable[[str | None, dict[str, Any]], str],
) -> dict[str, Any]:
    """Resolve effective site/object request context for DZI download."""
    site_config = resolve_site_fn(site, mw)
    base_url = str(site_config.get("base_url") or "").strip()
    provider_name = str(site_config.get("provider") or "npg").strip().lower()
    effective_mw = normalize_site_mw_fn(mw, site_config)
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


def parse_dzi_ids_text(ids_text: str) -> list[str]:
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


def sanitize_filename_component(text: str) -> str:
    """Normalize user-facing filename fragment into safe portable text."""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(text or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "item"


def extract_html_title(html_text: str) -> str:
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


def render_dzi_filename(
    filename_template: str,
    *,
    index: int,
    raw_id: str,
    effective_mw: str,
    site_config: dict[str, Any],
    effective_level: int,
    title_stem: str | None = None,
) -> str:
    """Render output filename stem for one single/batch DZI item."""
    template = str(filename_template or "{mw}").strip() or "{mw}"
    site_name = str(site_config.get("name") or "").strip()
    site_key = str(site_config.get("key") or site_name).strip()
    title_value = sanitize_filename_component(title_stem or effective_mw)
    data = {
        "index": int(index),
        "raw_id": sanitize_filename_component(raw_id),
        "mw": sanitize_filename_component(effective_mw),
        "id": sanitize_filename_component(effective_mw),
        "title": title_value,
        "site": sanitize_filename_component(site_name),
        "site_key": sanitize_filename_component(site_key),
        "level": int(effective_level),
    }
    try:
        rendered = template.format(**data)
    except (IndexError, KeyError, ValueError):
        rendered = data["title"] if "{title" in template else data["mw"]
    return sanitize_filename_component(rendered)


def append_dzi_stable_id_to_stem(stem: str, effective_mw: str) -> str:
    """Append stable object id to human-readable DZI stem unless already present."""
    base = sanitize_filename_component(stem)
    stable = sanitize_filename_component(effective_mw)
    if not stable:
        return base
    if not base:
        return stable
    if stable.lower() in base.lower():
        return base
    return f"{base}_{stable}"


def resolve_unique_output_path(
    output_dir: str,
    stem: str,
    ext: str,
    overwrite_mode: str,
    *,
    exists_fn: Callable[[str], bool] = os.path.exists,
) -> tuple[str, str]:
    """Resolve final output path according to overwrite strategy."""
    normalized_ext = str(ext or "png").strip().lower().lstrip(".") or "png"
    base_path = os.path.join(output_dir, f"{stem}.{normalized_ext}")
    mode = str(overwrite_mode or "skip").strip().lower()
    if mode == "overwrite":
        return base_path, "overwrite"
    if mode == "unique":
        if not exists_fn(base_path):
            return base_path, "unique_new"
        index = 2
        while True:
            candidate = os.path.join(output_dir, f"{stem}_{index}.{normalized_ext}")
            if not exists_fn(candidate):
                return candidate, "unique_suffix"
            index += 1
    return base_path, "skip"
