"""Pure URL and identifier helpers for the IIIF download node."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit


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
