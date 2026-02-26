"""
Module: utils/module_browser/runtime_refresh_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Runtime refresh orchestration helpers for Module Node Picker backend.

Purpose:
    Extracts phase-based refresh orchestration from API facade while preserving
    existing progress/log semantics and return payload contract.
"""

from __future__ import annotations

from typing import Any, Callable


def refresh_module_runtime_state(
    *,
    sync_upstreams: bool,
    progress_cb: Callable[..., None] | None,
    module_info_cache_clear: Callable[[], None],
    reset_custom_alias_cache: Callable[[], None],
    clear_comfyui_status_cache: Callable[[], None],
    refresh_console_log: Callable[[str, str], None],
    get_update_console_log_mode: Callable[[], str],
    discover_custom_modules: Callable[[], list[str]],
    sync_module_upstream: Callable[[str], bool],
    announce_tracked_module_updates: Callable[[], dict[str, Any]],
    comfyui_git_status: Callable[[], dict[str, Any]],
    short_commit: Callable[[str | None], str],
    set_custom_update_checked: Callable[[bool], None],
    now_iso: Callable[[], str],
    perf_counter: Callable[[], float],
) -> dict[str, Any]:
    """Recompute module snapshots and runtime tracking state with 3 refresh phases."""
    module_info_cache_clear()
    reset_custom_alias_cache()
    clear_comfyui_status_cache()
    refresh_started = perf_counter()
    refresh_console_log(
        "runtime refresh started (sync_upstreams={sync}, log_mode={mode})".format(
            sync="on" if sync_upstreams else "off",
            mode=get_update_console_log_mode(),
        ),
        "summary",
    )
    if progress_cb is None:
        progress_cb = lambda **_kwargs: None
    if sync_upstreams:
        module_names = discover_custom_modules()
        total = len(module_names)
        refresh_console_log(f"phase 1/3: upstream sync enabled for {total} custom module(s)", "summary")
        progress_cb(phase="sync", current=0, total=total, remaining=total, message="sync_upstreams")
        for idx, module_name in enumerate(module_names, start=1):
            sync_started = perf_counter()
            synced = sync_module_upstream(module_name)
            elapsed = perf_counter() - sync_started
            status = "synced" if synced else "skip"
            progress_cb(
                phase="sync",
                current=idx,
                total=total,
                remaining=total - idx,
                module=module_name,
                message=f"{status} ({elapsed:.2f}s)",
            )
    else:
        refresh_console_log("phase 1/3: upstream sync skipped (fast mode)", "summary")
        progress_cb(phase="sync", current=0, total=0, remaining=0, message="fast_mode")

    scan_started = perf_counter()
    refresh_console_log("phase 2/3: recomputing module snapshots and local git deltas...", "summary")
    progress_cb(phase="snapshots", current=0, total=0, remaining=0, message="recompute_snapshots")
    announce_summary = announce_tracked_module_updates()
    modules_need_update = 0
    modules_unknown_update = 0
    if isinstance(announce_summary, dict):
        modules_need_update = max(0, int(announce_summary.get("modules_need_update", 0)))
        modules_unknown_update = max(0, int(announce_summary.get("modules_unknown_update", 0)))
    modules_checked = max(0, int((announce_summary or {}).get("modules_checked", 0)))
    commit_changed = list((announce_summary or {}).get("commit_change_modules") or [])
    local_changed = list((announce_summary or {}).get("local_change_modules") or [])
    node_changed = list((announce_summary or {}).get("node_changed_modules") or [])
    manager_override_modules = list((announce_summary or {}).get("manager_override_modules") or [])
    new_modules_map = (announce_summary or {}).get("new_modules_between_runs") or {}
    new_modules_count = 0
    if isinstance(new_modules_map, dict):
        for value in new_modules_map.values():
            if isinstance(value, list):
                new_modules_count += len(value)
    scan_elapsed = perf_counter() - scan_started
    refresh_console_log(
        "phase 2/3 done in {elapsed:.2f}s: checked={checked}, need_update={need}, unknown_update={unknown}, "
        "commit_changed={commit}, local_changed={local}, node_changed={node}, manager_override={manager_override}, new_modules={new}".format(
            elapsed=scan_elapsed,
            checked=modules_checked,
            need=modules_need_update,
            unknown=modules_unknown_update,
            commit=len(commit_changed),
            local=len(local_changed),
            node=len(node_changed),
            manager_override=len(manager_override_modules),
            new=new_modules_count,
        ),
        "summary",
    )
    if commit_changed:
        refresh_console_log(f"commit changed modules: {', '.join(commit_changed)}", "verbose")
    if local_changed:
        refresh_console_log(f"locally changed modules: {', '.join(local_changed)}", "verbose")
    if node_changed:
        refresh_console_log(f"node changed modules: {', '.join(node_changed)}", "verbose")
    if manager_override_modules:
        refresh_console_log(
            f"manager-reported update modules: {', '.join(manager_override_modules)}",
            "verbose",
        )
    update_available_modules = list((announce_summary or {}).get("update_available_modules") or [])
    if update_available_modules:
        refresh_console_log(f"update available modules: {', '.join(update_available_modules)}", "verbose")
    unknown_update_modules = list((announce_summary or {}).get("unknown_update_modules") or [])
    if unknown_update_modules:
        refresh_console_log(f"unknown update status modules: {', '.join(unknown_update_modules)}", "summary")

    refresh_console_log("phase 3/3: checking ComfyUI status...", "summary")
    comfy_started = perf_counter()
    comfyui = comfyui_git_status()
    comfy_elapsed = perf_counter() - comfy_started
    refresh_console_log(
        "phase 3/3 done in {elapsed:.2f}s: ComfyUI status={status}, behind={behind}, ahead={ahead}, "
        "local={local}, remote={remote}".format(
            elapsed=comfy_elapsed,
            status=str(comfyui.get("update_status") or "unknown"),
            behind=str(comfyui.get("behind") if comfyui.get("behind") is not None else "-"),
            ahead=str(comfyui.get("ahead") if comfyui.get("ahead") is not None else "-"),
            local=short_commit(str(comfyui.get("installed_commit") or "")),
            remote=short_commit(str(comfyui.get("remote_commit") or "")),
        ),
        "summary",
    )
    progress_cb(
        phase="done",
        current=0,
        total=0,
        remaining=0,
        modules_need_update=modules_need_update,
        modules_unknown_update=modules_unknown_update,
        message="done",
    )
    total_elapsed = perf_counter() - refresh_started
    refresh_console_log(
        "runtime refresh finished in {elapsed:.2f}s (modules_need_update={need}, modules_unknown_update={unknown})".format(
            elapsed=total_elapsed,
            need=modules_need_update,
            unknown=modules_unknown_update,
        ),
        "summary",
    )
    set_custom_update_checked(True)
    return {
        "status": "ok",
        "refreshed_at": now_iso(),
        "comfyui": comfyui,
        "sync_upstreams": sync_upstreams,
        "modules_need_update": modules_need_update,
        "modules_unknown_update": modules_unknown_update,
    }
