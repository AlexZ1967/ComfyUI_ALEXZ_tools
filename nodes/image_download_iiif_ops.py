"""Pure URL and identifier helpers for the IIIF download node."""

from __future__ import annotations

import math
import hashlib
import html
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlsplit, urlunsplit


NYPL_IMAGE_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_iiif_service_url(source_url: str) -> str:
    """Normalize direct IIIF service, info.json, or default image URLs."""
    text = str(source_url or "").strip()
    if not text:
        return ""
    if text.endswith("/info.json"):
        return text[: -len("/info.json")].rstrip("/")
    match = re.match(r"^(https?://.+?)/full/[^/]+/[^/]+/default\.[A-Za-z0-9]+/?$", text)
    return str(match.group(1)).rstrip("/") if match else text.rstrip("/")


def extract_first_london_museum_service_url(html: str) -> str:
    """Extract the first London Museum IIIF service URL from HTML."""
    patterns = (
        r'data-src="(https://collections\.londonmuseum\.net/iiif/3/[^"]+)"',
        r"(https://collections\.londonmuseum\.net/iiif/3/[^\s\"'<>]+/info\.json)",
        r"(https://collections\.londonmuseum\.net/iiif/3/[^\s\"'<>]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(html or ""), flags=re.I)
        if match:
            return normalize_iiif_service_url(str(match.group(1)))
    return ""


def extract_first_generic_iiif_service_url(html: str) -> str:
    """Best-effort extraction of a IIIF service URL from arbitrary HTML."""
    patterns = (
        r'data-src="(https?://[^"]+/iiif/[^"]+)"',
        r'src="(https?://[^"]+/iiif/[^"]+/info\.json)"',
        r"(https?://[^\s\"'<>]+/iiif/[^\s\"'<>]+/info\.json)",
        r"(https?://[^\s\"'<>]+/iiif/[^\s\"'<>]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(html or ""), flags=re.I)
        if match:
            return normalize_iiif_service_url(str(match.group(1)))
    return ""


def extract_nypl_image_id_from_html(html: str) -> str:
    """Extract an NYPL image-id token from page HTML."""
    patterns = (
        r'id\s*=\s*["\']image-id["\'][^>]*>\s*([A-Za-z0-9_-]+)\s*<',
        r'aria-label\s*=\s*["\']Image ID["\'][^>]*>\s*([A-Za-z0-9_-]+)\s*<',
        r"https://iiif\.nypl\.org/iiif/3/([A-Za-z0-9_-]+)(?:/info\.json)?",
        r'"imageId"\s*:\s*"?([A-Za-z0-9_-]+)"?',
        r'"image_id"\s*:\s*"?([A-Za-z0-9_-]+)"?',
        r"Image\s*ID[\s\S]{0,2048}?([A-Za-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, str(html or ""), flags=re.IGNORECASE)
        if match:
            value = str(match.group(1) or "").strip()
            if NYPL_IMAGE_ID_TOKEN_RE.match(value):
                return value
    return ""


def extract_nypl_image_ids_from_text(text: str) -> list[str]:
    """Extract unique numeric NYPL image IDs from text, JSON, or XML."""
    patterns = (
        r'"imageID"\s*:\s*"?(\d+)"?', r'"imageId"\s*:\s*"?(\d+)"?',
        r"<imageID>\s*(\d+)\s*</imageID>", r"https://iiif\.nypl\.org/iiif/3/(\d+)(?:/info\.json)?",
    )
    found: list[str] = []
    for pattern in patterns:
        for value in re.findall(pattern, str(text or ""), flags=re.IGNORECASE):
            value = str(value or "").strip()
            if value and value not in found:
                found.append(value)
    return found


def extract_nypl_image_ids_from_json_payload(payload: Any) -> list[str]:
    """Walk JSON-like data and collect unique numeric imageID values."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key or "").strip().lower() in {"imageid", "image_id"}:
                    values = child if isinstance(child, (list, tuple)) else [child]
                    for item in values:
                        item_text = str(item or "").strip()
                        if item_text.isdigit() and item_text not in found:
                            found.append(item_text)
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(payload)
    return found


def iter_nypl_item_page_candidates(source_url: str) -> list[str]:
    """Return primary and alternate NYPL item page URLs."""
    text = str(source_url or "").strip()
    if not text:
        return []
    split = urlsplit(text)
    candidates = [text]
    if str(split.netloc or "").lower() == "digitalcollections.nypl.org":
        alternate = f"https://rp-digitalcollections.nypl.org{split.path}"
        if split.query:
            alternate = f"{alternate}?{split.query}"
        if alternate not in candidates:
            candidates.append(alternate)
    return candidates


def extract_forced_nypl_image_id_from_source_url(source_url: str) -> str:
    """Read an explicit NYPL image ID override from query or fragment."""
    split = urlsplit(str(source_url or "").strip())
    keys = ("image_id", "imageid", "nypl_image_id", "iiif_id")
    for mapping in (parse_qs(split.query), parse_qs(split.fragment)):
        for key in keys:
            for value in mapping.get(key) or mapping.get(key.upper()) or []:
                value = str(value or "").strip()
                if NYPL_IMAGE_ID_TOKEN_RE.match(value):
                    return value
    return ""


def inject_nypl_image_id_into_source_url(source_url: str, nypl_image_id: str) -> str:
    """Append or replace the explicit NYPL image ID query parameter."""
    source = str(source_url or "").strip()
    image_id = str(nypl_image_id or "").strip()
    if not source or not NYPL_IMAGE_ID_TOKEN_RE.match(image_id):
        return source
    split = urlsplit(source)
    blocked = {"image_id", "imageid", "nypl_image_id", "iiif_id"}
    pairs = [(key, value) for key, value in parse_qsl(split.query, keep_blank_values=True) if key.lower() not in blocked]
    pairs.append(("image_id", image_id))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(pairs), split.fragment))


def extract_gallica_service_url_from_source_url(source_url: str) -> str:
    """Build a Gallica IIIF service URL from an ARK/object page URL."""
    split = urlsplit(str(source_url or "").strip())
    match = re.search(r"/ark:/12148/([A-Za-z0-9]+)", split.path or "", flags=re.IGNORECASE)
    if not match:
        return ""
    tail = (split.path or "")[match.end():]
    page_match = re.search(r"/(f\d+)(?:[./][^/?#]*)?", tail, flags=re.IGNORECASE)
    page = str(page_match.group(1) or "") if page_match else "f1"
    return f"https://gallica.bnf.fr/iiif/ark:/12148/{match.group(1)}/{page}"


def build_iiif_size_spec(size_mode: str, requested_width: int) -> str:
    """Build the IIIF Image API size segment."""
    if str(size_mode or "max").strip().lower() == "max":
        return "max"
    return f"{max(1, int(requested_width))},"


def iiif_source_dimensions(info: dict[str, Any]) -> tuple[int, int]:
    """Return positive source dimensions from IIIF info.json."""
    return max(1, int(info.get("width") or 1)), max(1, int(info.get("height") or 1))


def iiif_limit_from_max_area(info: dict[str, Any]) -> dict[str, Any] | None:
    """Compute the estimated one-request dimensions for a maxArea service."""
    max_area = int(info.get("maxArea") or 0)
    if max_area <= 0:
        return None
    width, height = iiif_source_dimensions(info)
    scale = math.sqrt(float(max_area) / (float(width) * float(height)))
    return {
        "max_area": max_area,
        "predicted_max_width": max(1, int(math.floor(float(width) * scale))),
        "predicted_max_height": max(1, int(math.floor(float(height) * scale))),
        "scale": scale,
    }


def iiif_tile_profile(info: dict[str, Any], *, service_url: str = "", nypl_safe_tile_max: int = 512) -> dict[str, int]:
    """Extract a full-resolution tile profile and cap NYPL tile dimensions."""
    tiles = info.get("tiles") or []
    if not isinstance(tiles, list) or not tiles:
        raise RuntimeError("IIIF info.json does not expose `tiles`, full tile assembly is unavailable.")
    tile = tiles[0] if isinstance(tiles[0], dict) else {}
    if 1 not in (tile.get("scaleFactors") or []):
        raise RuntimeError("IIIF service does not expose scaleFactor=1, full-res tile assembly is unavailable.")
    width = max(1, int(tile.get("width") or 512))
    height = max(1, int(tile.get("height") or width))
    if "iiif.nypl.org/iiif/3/" in str(service_url or "").lower():
        width, height = min(width, nypl_safe_tile_max), min(height, nypl_safe_tile_max)
    return {"tile_width": width, "tile_height": height}


def resolve_iiif_cache_dir(cache_dir: str | None, *, default_root: Path) -> Path:
    """Resolve a user-supplied cache path or the node's default cache root."""
    cache_text = str(cache_dir or "").strip()
    if cache_text:
        return Path(os.path.abspath(os.path.expanduser(cache_text)))
    return default_root


def resolve_iiif_tile_cache_scope(
    cache_dir: str | None,
    service_url: str,
    info: dict[str, Any],
    *,
    default_root: Path,
) -> Path:
    """Build the stable per-assembly cache directory for one IIIF source."""
    cache_base = resolve_iiif_cache_dir(cache_dir, default_root=default_root)
    source_width, source_height = iiif_source_dimensions(info)
    scope_key = hashlib.sha1(
        f"{service_url.rstrip('/')}|{source_width}|{source_height}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return cache_base / scope_key


def iiif_tile_cache_path(cache_root: Path, image_url: str) -> Path:
    """Map a tile URL to its stable cache-file path."""
    digest = hashlib.sha1(str(image_url or "").encode("utf-8"), usedforsecurity=False).hexdigest()
    split = urlsplit(str(image_url or "").strip())
    ext = Path(split.path).suffix or ".bin"
    return cache_root / digest[:2] / f"{digest}{ext}"


def looks_like_raster_image_bytes(content: bytes) -> bool:
    """Check common raster signatures before decoding or caching tile bytes."""
    blob = bytes(content or b"")
    if len(blob) < 8:
        return False
    if blob.startswith(b"\xff\xd8\xff"):
        return True
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if blob.startswith((b"GIF87a", b"GIF89a")):
        return True
    if blob.startswith((b"II*\x00", b"MM\x00*")):
        return True
    return blob[:4] == b"RIFF" and blob[8:12] == b"WEBP"


def sanitize_filename_component(text: str) -> str:
    """Normalize a user-facing filename fragment into a portable stem."""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(text or "").strip())
    cleaned = cleaned.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return re.sub(r"_+", "_", cleaned).strip("._") or "iiif_image"


def normalize_extracted_title(text: str, source_url: str = "") -> str:
    """Clean a human-readable title before filename sanitization."""
    value = html.unescape(str(text or "")).strip()
    if "gallica.bnf.fr" in str(source_url or "").lower():
        value = re.sub(r"\s+[|:-]\s*Gallica.*$", "", value, flags=re.IGNORECASE)
    value = value.replace("/", " ").replace("\\", " ")
    value = value.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    return re.sub(r"\s+", " ", value).strip()


def is_generic_source_title(title: str, source_url: str = "") -> bool:
    """Return whether an extracted title is merely a generic site label."""
    value = str(title or "").strip().lower()
    return not value or ("gallica.bnf.fr" in str(source_url or "").lower() and value in {"gallica", "bnf gallica"})


def extract_gallica_query_title(source_url: str) -> str:
    """Extract a human-readable fallback title from Gallica's `.r=` path segment."""
    text = str(source_url or "").strip()
    if "gallica.bnf.fr" not in text.lower():
        return ""
    match = re.search(r"\.r=([^/?#]+)", unquote(str(urlsplit(text).path or "")), flags=re.IGNORECASE)
    return normalize_extracted_title(str(match.group(1) or ""), source_url) if match else ""


def extract_iiif_label_text(value: Any) -> str:
    """Extract human-readable text from IIIF v2/v3 label values."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("none", "fr", "en", "@value", "value"):
            text = extract_iiif_label_text(value.get(key))
            if text:
                return text
    if isinstance(value, list):
        for item in value:
            text = extract_iiif_label_text(item)
            if text:
                return text
    return ""


def extract_html_title(html_text: str, *, gallica: bool = False) -> str:
    """Extract the most relevant object or document title from HTML."""
    patterns = ([r'<span[^>]+class=["\'][^"\']*\btitle\b[^"\']*["\'][^>]*>(.*?)</span>', r'<div[^>]+class=["\'][^"\']*\btitle\b[^"\']*["\'][^>]*>(.*?)</div>', r'<h1[^>]*>(.*?)</h1>'] if gallica else []) + [r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']', r'<meta[^>]+name=["\']og:title["\'][^>]+content=["\'](.*?)["\']', r'<title[^>]*>(.*?)</title>']
    for pattern in patterns:
        match = re.search(pattern, str(html_text or ""), flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = html.unescape(re.sub(r"<[^>]+>", " ", str(match.group(1) or "")))
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                return value
    return ""


def extract_iiif_stable_id(source_url: str, service_url: str = "") -> str:
    """Extract a stable object/service identifier for filename disambiguation."""
    patterns = (r"ark:/12148/([A-Za-z0-9]+)\b", r"\b(btv[0-9a-z]+)\b", r"\b(object-\d+)\b", r"\b(nla\.obj-\d+)\b", r"\b(mw\d+)\b", r"/([A-Za-z]\d{3,}(?:\.ptif)?)\b")
    for pattern in patterns:
        match = re.search(pattern, " | ".join([str(source_url or "").strip(), str(service_url or "").strip()]), flags=re.IGNORECASE)
        if match:
            return sanitize_filename_component(re.sub(r"\.ptif$", "", str(match.group(1) or ""), flags=re.IGNORECASE))
    return ""


def append_stable_id_to_stem(stem: str, stable_id: str) -> str:
    """Append a stable ID unless the filename stem already contains it."""
    base, stable = sanitize_filename_component(stem), sanitize_filename_component(stable_id)
    if not stable or stable == "iiif_image":
        return base
    return base if stable.lower() in base.lower() else f"{base}_{stable}"


def derive_output_stem_from_source_url(source_url: str, service_url: str = "") -> str:
    """Derive a deterministic output stem from source URL and stable identifier."""
    path = str(urlsplit(str(source_url or "").strip()).path or "").strip("/")
    segments, stable_id = [part for part in path.split("/") if part], extract_iiif_stable_id(source_url, service_url)
    if "gallica.bnf.fr" in str(source_url or "").lower() and stable_id:
        page_match = re.search(r"/(f\d+)(?:[./][^/?#]*)?", f"/{path}", flags=re.IGNORECASE)
        page = str(page_match.group(1) or "") if page_match else ""
        return append_stable_id_to_stem(stable_id if not page or page.lower() == "f1" else f"{stable_id}_{page}", stable_id)
    candidate = unquote(segments[-1]) if segments else "iiif_image"
    if candidate.lower() == "info.json" or candidate.lower().startswith("default."):
        candidate = unquote(segments[-2]) if len(segments) >= 2 else candidate
    return append_stable_id_to_stem(re.sub(r"\.[A-Za-z0-9]+$", "", candidate), stable_id)
