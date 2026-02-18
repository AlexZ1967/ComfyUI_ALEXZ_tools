"""
Module: utils/module_browser/requirements_pending_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Helpers for persisting pending requirements-install markers in module state.

Purpose:
    Moves deterministic state mutation logic for requirements follow-up flags
    out of API facade as part of Phase 3 backend decomposition.
"""

from __future__ import annotations

from typing import Any, Callable


def set_comfyui_requirements_pending(
    *,
    pending: bool,
    before_commit: str,
    after_commit: str,
    load_state_fn: Callable[[], dict[str, Any]],
    save_state_fn: Callable[[dict[str, Any]], None],
    now_iso_fn: Callable[[], str],
    on_state_changed: Callable[[], None] | None = None,
) -> bool:
    """Persist pending ComfyUI requirements marker and return change flag."""
    state = load_state_fn()
    if not isinstance(state, dict):
        return False
    entry_raw = state.get("__comfyui__")
    before_entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    if pending:
        entry["pending_requirements_update"] = True
        if before_commit:
            entry["pending_requirements_before_commit"] = before_commit
        if after_commit:
            entry["pending_requirements_after_commit"] = after_commit
        entry["pending_requirements_updated_at"] = now_iso_fn()
    else:
        entry.pop("pending_requirements_update", None)
        entry.pop("pending_requirements_before_commit", None)
        entry.pop("pending_requirements_after_commit", None)
        entry.pop("pending_requirements_updated_at", None)

    if entry == before_entry:
        return False

    state["__comfyui__"] = entry
    if on_state_changed is not None:
        on_state_changed()
    save_state_fn(state)
    return True


def set_module_requirements_pending(
    *,
    module_name: str,
    pending: bool,
    before_commit: str,
    after_commit: str,
    canonical_custom_module_name_fn: Callable[[str], str],
    load_state_fn: Callable[[], dict[str, Any]],
    save_state_fn: Callable[[dict[str, Any]], None],
    now_iso_fn: Callable[[], str],
    on_state_changed: Callable[[], None] | None = None,
) -> bool:
    """Persist pending requirements marker for one custom module."""
    module = canonical_custom_module_name_fn(module_name)
    if not module or module == "unknown":
        return False

    state = load_state_fn()
    if not isinstance(state, dict):
        return False

    entry_raw = state.get(module)
    before_entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    if pending:
        entry["pending_requirements_update"] = True
        if before_commit:
            entry["pending_requirements_before_commit"] = before_commit
        if after_commit:
            entry["pending_requirements_after_commit"] = after_commit
        entry["pending_requirements_updated_at"] = now_iso_fn()
    else:
        entry.pop("pending_requirements_update", None)
        entry.pop("pending_requirements_before_commit", None)
        entry.pop("pending_requirements_after_commit", None)
        entry.pop("pending_requirements_updated_at", None)

    if entry == before_entry:
        return False

    state[module] = entry
    if on_state_changed is not None:
        on_state_changed()
    save_state_fn(state)
    return True

