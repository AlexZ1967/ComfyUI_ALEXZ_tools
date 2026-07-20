"""
Module: utils/module_browser_api/request_parsing.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Shared request-parsing helpers for module browser API routes.

Purpose:
    Keeps payload/query normalization out of route glue so malformed request
    bodies do not require broad inline exception handling in each endpoint.
"""

from __future__ import annotations

from typing import Any


def coerce_bool_flag(value: Any, *, default: bool) -> bool:
    """Normalize common query/payload boolean aliases to a strict bool."""
    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    return text not in {"0", "false", "no", "off"}


async def load_json_payload(request: Any) -> dict[str, Any]:
    """Read JSON body from request, returning empty payload on parse errors."""
    try:
        payload = await request.json()
    except (AttributeError, TypeError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}
