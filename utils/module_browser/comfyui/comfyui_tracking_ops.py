"""
Module: utils/module_browser/comfyui_tracking_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    ComfyUI local-change tracking helpers for Module Node Picker backend.

Purpose:
    Extracts ComfyUI startup novelty tracking and acknowledge logic from the
    API facade to keep behavior stable and easier to unit-test in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def track_comfyui_local_update(
    *,
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    comfyui_root: Callable[[], Path | None],
    run_git: Callable[[list[str], float], str | None],
    now_iso: Callable[[], str],
    short_commit: Callable[[str | None], str],
    clear_comfyui_status_cache: Callable[[], None],
) -> None:
    """Track local ComfyUI commit changes between restarts without upstream sync."""
    state = load_module_state()
    if not isinstance(state, dict):
        return
    root = comfyui_root()
    if root is None:
        return
    is_git = run_git(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], 2.0)
    if is_git != "true":
        return

    current_commit = run_git(["git", "-C", str(root), "rev-parse", "HEAD"], 2.0) or ""
    current_updated_at = run_git(["git", "-C", str(root), "log", "-1", "--format=%cI"], 2.0) or ""
    if not current_commit:
        return

    entry_raw = state.get("__comfyui__")
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    status_raw = entry.get("status")
    status = dict(status_raw) if isinstance(status_raw, dict) else {}
    status_by_mode_raw = entry.get("status_by_mode")
    status_by_mode = dict(status_by_mode_raw) if isinstance(status_by_mode_raw, dict) else {}
    prev_commit = (
        (entry.get("installed_commit") or status.get("installed_commit") or "").strip()
    )
    now = now_iso()
    changed = False

    if prev_commit and prev_commit != current_commit:
        entry["pending_prev_commit"] = prev_commit
        entry["pending_new_commit"] = current_commit
        entry["pending_update_at"] = now
        entry["startup_prev_commit"] = prev_commit
        entry["startup_new_commit"] = current_commit
        entry["startup_update_at"] = now
        changed = True

    if entry.get("installed_commit") != current_commit:
        entry["installed_commit"] = current_commit
        changed = True
    if entry.get("installed_updated_at") != current_updated_at:
        entry["installed_updated_at"] = current_updated_at
        changed = True

    status.setdefault("repository", "https://github.com/comfyanonymous/ComfyUI")
    status["path"] = str(root)
    status["installed_commit"] = current_commit
    status["installed_commit_short"] = short_commit(current_commit)
    status["installed_updated_at"] = current_updated_at
    status.setdefault("update_status", "unknown")
    entry["status"] = status
    for mode_name, mode_status_raw in status_by_mode.items():
        if not isinstance(mode_status_raw, dict):
            continue
        mode_status = dict(mode_status_raw)
        mode_status.setdefault("repository", "https://github.com/comfyanonymous/ComfyUI")
        mode_status["path"] = str(root)
        mode_status["installed_commit"] = current_commit
        mode_status["installed_commit_short"] = short_commit(current_commit)
        mode_status["installed_updated_at"] = current_updated_at
        mode_status.setdefault("update_status", "unknown")
        status_by_mode[mode_name] = mode_status
    if status_by_mode:
        entry["status_by_mode"] = status_by_mode
    entry["updated_at"] = now
    state["__comfyui__"] = entry

    if changed:
        clear_comfyui_status_cache()
        save_module_state(state)


def acknowledge_comfyui_novelty(
    *,
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    clear_comfyui_status_cache: Callable[[], None],
) -> dict[str, Any]:
    """Clear pending ComfyUI novelty markers after explicit user refresh action."""
    state = load_module_state()
    if not isinstance(state, dict):
        return {"status": "ok", "changed": False}
    entry_raw = state.get("__comfyui__")
    if not isinstance(entry_raw, dict):
        return {"status": "ok", "changed": False}

    entry = dict(entry_raw)
    before = dict(entry)
    for key in (
        "pending_prev_commit",
        "pending_new_commit",
        "pending_update_at",
        "startup_prev_commit",
        "startup_new_commit",
        "startup_update_at",
    ):
        entry.pop(key, None)

    changed = entry != before
    if changed:
        clear_comfyui_status_cache()
        state["__comfyui__"] = entry
        save_module_state(state)
    return {"status": "ok", "changed": changed}
