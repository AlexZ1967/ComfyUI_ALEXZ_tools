"""
Module: utils/module_browser/widget_mode_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    UI-mode and log-mode helper functions for Module Node Picker backend.

Purpose:
    Isolates deterministic helper logic for info-only responses, custom-update
    visibility gates, and log-mode normalization as part of Phase 3 split.
"""

from __future__ import annotations

from typing import Any, Callable


def custom_update_checked_flag(state: dict[str, Any] | None) -> bool:
    """Return whether custom remote-update check was explicitly acknowledged."""
    cache = state if isinstance(state, dict) else {}
    meta = cache.get("__meta__") if isinstance(cache, dict) else None
    if not isinstance(meta, dict):
        return False
    return bool(meta.get("custom_update_checked"))


def info_only_rejection_payload(feature: str) -> dict[str, Any]:
    """Build a consistent rejection payload for disabled mutate operations."""
    return {
        "status": "disabled",
        "feature": feature,
        "message": "This widget runs in info-only mode. Use ComfyUI-Manager for install/update actions.",
    }


def set_custom_update_checked(
    *,
    checked: bool,
    load_state_fn: Callable[[], dict[str, Any]],
    save_state_fn: Callable[[dict[str, Any]], None],
    now_iso_fn: Callable[[], str],
    on_changed: Callable[[], None] | None = None,
) -> bool:
    """Persist custom-update gate flag and return true when state changed."""
    state = load_state_fn()
    if not isinstance(state, dict):
        return False
    meta_raw = state.get("__meta__")
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    value = bool(checked)
    if bool(meta.get("custom_update_checked")) == value:
        return False
    meta["custom_update_checked"] = value
    meta["custom_update_checked_at"] = now_iso_fn()
    state["__meta__"] = meta
    save_state_fn(state)
    if on_changed is not None:
        on_changed()
    return True


def normalize_log_mode(value: str | None) -> str:
    """Normalize console log mode to either `summary` or `verbose`."""
    text = str(value or "").strip().lower()
    return "verbose" if text in {"verbose", "debug", "full", "detailed"} else "summary"

