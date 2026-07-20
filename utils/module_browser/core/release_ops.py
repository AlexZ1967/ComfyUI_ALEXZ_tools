"""
Module: utils/module_browser/release_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    GitHub release metadata helpers for Module Node Picker backend.

Purpose:
    Isolates network/release parsing logic from API facade while preserving
    current behavior and output schema.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable


def github_latest_release(
    owner: str,
    repo: str,
    *,
    timeout: float = 8.0,
    request_factory: Callable[..., Any] = urllib.request.Request,
    urlopen_fn: Callable[..., Any] = urllib.request.urlopen,
    json_loads: Callable[[str], Any] = json.loads,
) -> dict[str, Any]:
    """Fetch latest GitHub release metadata for repository."""
    owner_text = str(owner or "").strip()
    repo_text = str(repo or "").strip()
    if not owner_text or not repo_text:
        return {}
    url = f"https://api.github.com/repos/{owner_text}/{repo_text}/releases/latest"
    request = request_factory(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ALEXZ_tools-module-picker",
        },
    )
    try:
        with urlopen_fn(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload = json_loads(body)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 429}:
            return {}
        return {}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return {}
    return {
        "tag_name": tag,
        "published_at": str(payload.get("published_at") or "").strip(),
        "created_at": str(payload.get("created_at") or "").strip(),
        "name": str(payload.get("name") or "").strip(),
        "html_url": str(payload.get("html_url") or "").strip(),
    }
