"""
Module: utils/module_browser/module_update_state_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Helpers for update-status evaluation and counters.

Purpose:
    Extracts deterministic module/comfy update-state logic from API facade,
    keeping route behavior unchanged while improving testability.
"""

from __future__ import annotations

from typing import Any, Callable


def module_needs_update_now(
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    load_module_state: Callable[[], dict[str, Any]],
    module_git_state_fn: Callable[[str], dict[str, Any]],
    manager_meta_for_module_fn: Callable[[str, str | None], dict[str, Any] | None],
    infer_update_from_manager_stats_fn: Callable[[str | None, str | None], tuple[bool | None, str]],
) -> bool:
    """Check whether local module commit differs from tracked remote state."""
    module = canonical_custom_module_name(module_name)
    state = load_module_state()
    entry = state.get(module) if isinstance(state, dict) else None
    cached_update: bool | None = None
    if isinstance(entry, dict):
        value = entry.get("update_available")
        if isinstance(value, bool):
            cached_update = value

    git_state = module_git_state_fn(module)
    if not git_state:
        repository = ""
        installed_updated_at = ""
        if isinstance(entry, dict):
            repository = str(entry.get("repository") or "")
            installed_updated_at = str(entry.get("installed_updated_at") or "")
        if not repository:
            meta = manager_meta_for_module_fn(module, repository)
            if isinstance(meta, dict):
                repository = str(meta.get("repository") or "")
        inferred, _ = infer_update_from_manager_stats_fn(repository, installed_updated_at)
        if isinstance(inferred, bool):
            return inferred
        return bool(cached_update)

    behind = git_state.get("behind")
    if isinstance(behind, int):
        return behind > 0
    remote_head = str(git_state.get("remote_head") or "").strip()
    installed = str(git_state.get("installed_commit") or "").strip()
    if bool(git_state.get("has_upstream") and remote_head and installed):
        return remote_head != installed

    repository = str(git_state.get("repository") or "")
    installed_updated_at = str(git_state.get("installed_updated_at") or "")
    if not repository and isinstance(entry, dict):
        repository = str(entry.get("repository") or "")
    if not installed_updated_at and isinstance(entry, dict):
        installed_updated_at = str(entry.get("installed_updated_at") or "")
    if not repository:
        meta = manager_meta_for_module_fn(module, repository)
        if isinstance(meta, dict):
            repository = str(meta.get("repository") or "")
    inferred, _ = infer_update_from_manager_stats_fn(repository, installed_updated_at)
    if isinstance(inferred, bool):
        return inferred
    return bool(cached_update)


def count_custom_modules_need_update(
    *,
    load_module_state: Callable[[], dict[str, Any]],
    discover_custom_modules: Callable[[], list[str]],
    canonical_custom_module_name: Callable[[str], str],
) -> int:
    """Count custom modules that currently report available updates."""
    state = load_module_state()
    if not isinstance(state, dict):
        return 0
    count = 0
    for module_name in discover_custom_modules():
        entry = state.get(canonical_custom_module_name(module_name))
        if isinstance(entry, dict) and bool(entry.get("update_available")):
            count += 1
    return count


def count_custom_modules_unknown_update(
    *,
    load_module_state: Callable[[], dict[str, Any]],
    discover_custom_modules: Callable[[], list[str]],
    canonical_custom_module_name: Callable[[str], str],
) -> int:
    """Count custom modules whose remote update status is unknown/uncheckable."""
    state = load_module_state()
    if not isinstance(state, dict):
        return 0
    count = 0
    for module_name in discover_custom_modules():
        entry = state.get(canonical_custom_module_name(module_name))
        if not isinstance(entry, dict):
            count += 1
            continue
        if not isinstance(entry.get("update_available"), bool):
            count += 1
    return count


def comfyui_needs_update_now(*, comfyui_git_status_fn: Callable[..., dict[str, Any]]) -> bool:
    """Check whether local ComfyUI commit is behind remote tracking commit."""
    status = comfyui_git_status_fn(force_refresh=True, mode="releases")
    behind = status.get("behind")
    if isinstance(behind, int):
        return behind > 0
    return bool(status.get("update_status") == "can_update")

