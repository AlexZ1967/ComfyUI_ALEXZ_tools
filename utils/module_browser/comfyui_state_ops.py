"""
Module: utils/module_browser/comfyui_state_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    ComfyUI status payload/state helper functions for Module Node Picker backend.

Purpose:
    Extracts repeated ComfyUI status template/cache/state merge logic from
    API facade while preserving payload keys and semantics.
"""

from __future__ import annotations

from typing import Any, Callable


def comfyui_status_template(mode_norm: str) -> dict[str, Any]:
    """Create default ComfyUI status payload for selected check mode."""
    return {
        "path": "",
        "repository": "https://github.com/comfyanonymous/ComfyUI",
        "check_mode": mode_norm,
        "remote_name": "",
        "remote_ref": "",
        "branch": "",
        "upstream": "",
        "installed_commit": "",
        "installed_commit_short": "",
        "installed_updated_at": "",
        "remote_commit": "",
        "remote_commit_short": "",
        "remote_updated_at": "",
        "release_tag": "",
        "release_name": "",
        "release_url": "",
        "ahead": None,
        "behind": None,
        "update_available": None,
        "update_status": "unknown",
        "requirements_update_pending": False,
        "requirements_pending_before_commit": "",
        "requirements_pending_after_commit": "",
        "requirements_pending_updated_at": "",
    }


def resolve_cached_status(state: dict[str, Any], mode_norm: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve cached ComfyUI entry and mode-specific status from state cache."""
    cached_entry = state.get("__comfyui__") if isinstance(state, dict) else None
    status_by_mode = cached_entry.get("status_by_mode") if isinstance(cached_entry, dict) else None
    cached_status: dict[str, Any] | None = None
    if isinstance(status_by_mode, dict):
        candidate = status_by_mode.get(mode_norm)
        if isinstance(candidate, dict):
            cached_status = candidate
    if cached_status is None and isinstance(cached_entry, dict):
        candidate = cached_entry.get("status")
        if isinstance(candidate, dict):
            cached_status = candidate
    return (cached_entry if isinstance(cached_entry, dict) else None, cached_status)


def apply_cached_pending_fields(
    result: dict[str, Any],
    cached_entry: dict[str, Any] | None,
    *,
    short_commit: Callable[[str | None], str],
) -> dict[str, Any]:
    """Attach startup/requirements pending markers from persisted state entry."""
    if not isinstance(cached_entry, dict):
        return result
    pending_prev = (cached_entry.get("pending_prev_commit") or cached_entry.get("startup_prev_commit") or "").strip()
    pending_new = (cached_entry.get("pending_new_commit") or cached_entry.get("startup_new_commit") or "").strip()
    pending_at = (cached_entry.get("pending_update_at") or cached_entry.get("startup_update_at") or "").strip()
    result["updated_between_runs"] = bool(pending_prev and pending_new)
    result["startup_prev_commit_short"] = short_commit(pending_prev) if pending_prev else ""
    result["startup_new_commit_short"] = short_commit(pending_new) if pending_new else ""
    result["startup_update_at"] = pending_at
    result["requirements_update_pending"] = bool(cached_entry.get("pending_requirements_update"))
    result["requirements_pending_before_commit"] = str(cached_entry.get("pending_requirements_before_commit") or "")
    result["requirements_pending_after_commit"] = str(cached_entry.get("pending_requirements_after_commit") or "")
    result["requirements_pending_updated_at"] = str(cached_entry.get("pending_requirements_updated_at") or "")
    return result


def persist_comfyui_status(
    state: dict[str, Any],
    *,
    mode_norm: str,
    result: dict[str, Any],
    now_iso: Callable[[], str],
) -> dict[str, Any]:
    """Persist ComfyUI status into module state payload and return updated state."""
    if not isinstance(state, dict):
        return state
    prev_entry = state.get("__comfyui__")
    entry = dict(prev_entry) if isinstance(prev_entry, dict) else {}
    by_mode = dict(entry.get("status_by_mode")) if isinstance(entry.get("status_by_mode"), dict) else {}
    by_mode[mode_norm] = dict(result)
    entry["status_by_mode"] = by_mode
    entry["status"] = dict(result)
    entry["updated_at"] = now_iso()
    if result.get("installed_commit"):
        entry["installed_commit"] = result.get("installed_commit")
    if result.get("installed_updated_at"):
        entry["installed_updated_at"] = result.get("installed_updated_at")
    state["__comfyui__"] = entry
    return state
