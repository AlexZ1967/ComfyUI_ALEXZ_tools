"""
Module: utils/module_browser_api/module_info_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Module-info orchestration helpers for module browser API facade.

Purpose:
    Keeps module-info cache orchestration and small adapter wrappers out of
    `utils/module_node_browser_api.py` so the facade stays focused on wiring.
"""

from __future__ import annotations

from typing import Any, Callable


def module_local_readme_summary(
    module_name: str,
    *,
    module_local_readme_summary_impl: Callable[..., str | None],
    custom_nodes_roots: Callable[[], Any],
) -> str | None:
    """Read and extract short description snippet from module README file."""
    return module_local_readme_summary_impl(
        module_name=module_name,
        custom_nodes_roots=custom_nodes_roots,
    )


def sanitize_module_description(
    text: str,
    *,
    sanitize_module_description_impl: Callable[[str, Any], str],
    html_tag_re: Any,
) -> str:
    """Normalize module description text for UI card rendering."""
    return sanitize_module_description_impl(text, html_tag_re)


def remember_module_state(
    module_name: str,
    result: dict[str, Any],
    *,
    remember_module_state_impl: Callable[..., None],
    canonical_custom_module_name: Callable[[str], str],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    now_iso: Callable[[], str],
    short_commit: Callable[[str | None], str],
) -> None:
    """Capture current module/node snapshot as baseline for next ComfyUI start."""
    remember_module_state_impl(
        module_name,
        result,
        canonical_custom_module_name=canonical_custom_module_name,
        load_module_state=load_module_state,
        save_module_state=save_module_state,
        now_iso=now_iso,
        short_commit=short_commit,
    )


def apply_node_change_info(
    result: dict[str, Any],
    group: str,
    module_name: str,
    *,
    apply_node_change_info_impl: Callable[..., None],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
) -> None:
    """Attach node-level change markers to module info payload for UI rendering."""
    apply_node_change_info_impl(
        result,
        group,
        module_name,
        load_module_state=load_module_state,
    )


def resolve_module_info_cached(
    *,
    group: str,
    module_name: str,
    force_refresh: bool,
    sync_upstream: bool,
    cache_only: bool,
    now_ts: float,
    module_info_cache: dict[tuple[str, str, bool], tuple[float, dict[str, Any]]],
    ttl_sec: float,
    canonical_custom_module_name: Callable[[str], str],
    resolve_module_info_uncached: Callable[..., dict[str, Any]],
    apply_node_change_info_fn: Callable[[dict[str, Any], str, str], None],
    sync_module_upstream: Callable[[str], Any],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    custom_update_checked_flag: Callable[[dict[str, Any] | None], bool],
    module_git_state: Callable[[str], dict[str, Any]],
    module_repo_url: Callable[[str], str | None],
    manager_meta_for_module: Callable[[str, str | None], dict[str, Any] | None],
    module_local_readme_summary_fn: Callable[[str], str | None],
    sanitize_module_description_fn: Callable[[str], str],
    github_id: Callable[[str | None], str | None],
    infer_update_from_manager_stats: Callable[[str | None, str | None], tuple[bool | None, str]],
    short_commit: Callable[[str | None], str],
    remember_module_state_fn: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    """Build and cache complete module info payload with metadata and git state."""
    group_norm = (group or "").strip().lower()
    module_norm = (module_name or "").strip()
    if group_norm == "custom":
        module_norm = canonical_custom_module_name(module_norm)

    key = (group_norm or "", module_norm or "", bool(cache_only))
    if force_refresh:
        module_info_cache.pop(key, None)

    cached = module_info_cache.get(key)
    if cached is not None and (now_ts - cached[0]) < ttl_sec:
        return dict(cached[1])

    result = resolve_module_info_uncached(
        group=group_norm,
        module_name=module_norm,
        sync_upstream=sync_upstream,
        cache_only=cache_only,
        canonical_custom_module_name=canonical_custom_module_name,
        apply_node_change_info=apply_node_change_info_fn,
        sync_module_upstream=sync_module_upstream,
        load_module_state=load_module_state,
        custom_update_checked_flag=custom_update_checked_flag,
        module_git_state=module_git_state,
        module_repo_url=module_repo_url,
        manager_meta_for_module=manager_meta_for_module,
        module_local_readme_summary=module_local_readme_summary_fn,
        sanitize_module_description=sanitize_module_description_fn,
        github_id=github_id,
        infer_update_from_manager_stats=infer_update_from_manager_stats,
        short_commit=short_commit,
        remember_module_state=remember_module_state_fn,
    )
    module_info_cache[key] = (now_ts, dict(result))
    return result
