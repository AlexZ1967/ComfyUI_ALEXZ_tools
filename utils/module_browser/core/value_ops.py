"""
Module: utils/module_browser/value_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Pure value/date/repository helper functions for Module Node Picker backend.

Purpose:
    Centralizes deterministic parsing/normalization helpers so API facade code
    stays focused on route orchestration during Phase 3 decomposition.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def short_commit(commit: str | None) -> str:
    """Return short 8-character representation of git commit hash."""
    value = str(commit or "").strip()
    if not value:
        return "unknown"
    return value[:8]


def normalize_repo_url(url: str | None) -> str | None:
    """Normalize repository URL by stripping `.git` suffix and trailing slash."""
    if not isinstance(url, str):
        return None
    value = url.strip()
    if not value:
        return None
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value[len("git@github.com:") :]
    elif value.startswith("git://github.com/"):
        value = "https://github.com/" + value[len("git://github.com/") :]
    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/") or None


def github_id(url: str | None, *, github_re: re.Pattern[str]) -> str | None:
    """Extract canonical `owner/repo` identifier from GitHub URL."""
    normalized = normalize_repo_url(url)
    if not normalized:
        return None
    match = github_re.search(normalized)
    if not match:
        return None
    owner = str(match.group(1) or "").strip()
    repo = str(match.group(2) or "").strip()
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def repo_name(url: str | None, *, github_id_fn) -> str | None:
    """Extract repository name from normalized `owner/repo` GitHub id."""
    gid = github_id_fn(url)
    if not gid or "/" not in gid:
        return None
    return gid.split("/", 1)[1]


def pick_repo_url(entry: dict[str, Any], *, normalize_repo_url_fn) -> str | None:
    """Pick repository URL from manager entry fields using stable priority."""
    if not isinstance(entry, dict):
        return None
    candidates: list[str] = []
    for key in ("repository", "reference"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
    files = entry.get("files")
    if isinstance(files, list):
        candidates.extend(x for x in files if isinstance(x, str))
    for candidate in candidates:
        norm = normalize_repo_url_fn(candidate)
        if norm and "github.com/" in norm.lower():
            return norm
    return normalize_repo_url_fn(candidates[0]) if candidates else None


def parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime text into timezone-aware datetime object."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    formats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y, %H:%M:%S")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def to_iso(dt: datetime | None) -> str | None:
    """Convert datetime to normalized UTC ISO-8601 string."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_comfyui_mode(value: str | None) -> str:
    """Normalize ComfyUI remote-check mode to `releases` or `commits`."""
    text = str(value or "").strip().lower()
    if text in {"commit", "commits", "branch", "branches", "git"}:
        return "commits"
    return "releases"
