"""Pure helpers for the Trove image-id search node."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any


TROVE_API_KEY_ENV = "TROVE_API_KEY"
TROVE_API_RESULT_URL = "https://api.trove.nla.gov.au/v3/result"
TROVE_WEB_CATEGORY_URL = "https://trove.nla.gov.au/search/category/{category}?keyword={query}"


def normalize_trove_ui_category(category: str | None) -> str:
    """Normalize the node/browser category name."""
    text = str(category or "images").strip().lower()
    return text or "images"


def normalize_trove_api_category(category: str | None) -> str:
    """Normalize UI/browser category names into Trove API v3 category names."""
    text = normalize_trove_ui_category(category)
    if text in {"images", "pictures", "picture"}:
        return "image"
    return text


def resolve_trove_api_key(explicit_api_key: str | None, environ: dict[str, str] | None = None) -> tuple[str, str]:
    """Resolve Trove API key from node input first, then environment."""
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit, "input"
    env = environ if environ is not None else os.environ
    value = str(env.get(TROVE_API_KEY_ENV, "") or "").strip()
    if value:
        return value, "env"
    return "", "missing"


def build_trove_api_params(
    query: str,
    *,
    category: str = "images",
    max_results: int = 50,
    include_online_only: bool = True,
) -> dict[str, Any]:
    """Build Trove API v3 `/result` query params for image-id discovery."""
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("`query` must not be empty.")

    limit = max(1, min(100, int(max_results or 1)))
    params: dict[str, Any] = {
        "q": query_text,
        "category": normalize_trove_api_category(category),
        "encoding": "json",
        "n": limit,
    }
    if include_online_only:
        params["l-availability"] = "y/f"
    return params


def extract_nla_obj_ids(text: str) -> list[str]:
    """Extract unique `nla.obj-...` identifiers while preserving order."""
    ids: list[str] = []
    seen: set[str] = set()
    for value in re.findall(r"nla\.obj-\d+", str(text or ""), flags=re.IGNORECASE):
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ids.append(normalized)
    return ids


def iter_string_values(payload: Any) -> Iterable[str]:
    """Yield every string-ish value from nested JSON-compatible data."""
    if isinstance(payload, str):
        yield payload
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from iter_string_values(value)
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            yield from iter_string_values(value)
        return
    if payload is not None and isinstance(payload, (int, float)):
        yield str(payload)


def extract_nla_obj_ids_from_api_payload(payload: Any) -> list[str]:
    """Extract NLA object IDs from a Trove API JSON payload."""
    return extract_nla_obj_ids("\n".join(iter_string_values(payload)))


def limit_ids(ids: Iterable[str], max_results: int) -> list[str]:
    """Deduplicate and clamp IDs to the requested maximum."""
    unique: list[str] = []
    seen: set[str] = set()
    limit = max(1, int(max_results or 1))
    for value in ids:
        normalized = str(value or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return unique


def sanitize_trove_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a diagnostic payload that never exposes the raw API key."""
    sanitized = dict(result or {})
    sanitized.pop("api_key", None)
    headers = sanitized.get("request_headers")
    if isinstance(headers, dict):
        sanitized["request_headers"] = {
            str(key): ("***" if str(key).lower() in {"x-api-key", "authorization"} else str(value))
            for key, value in headers.items()
        }
    return sanitized
