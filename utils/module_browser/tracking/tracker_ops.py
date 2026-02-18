"""
Module: utils/module_browser/tracker_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Runtime tracker and novelty-marker helpers for Module Node Picker backend.

Purpose:
    Extracts module/node change tracking and novelty acknowledge operations from
    the monolithic API module while preserving existing behavior via facades.
"""

from __future__ import annotations

from typing import Any, Callable


def remember_module_state(
    module_name: str,
    result: dict[str, Any],
    *,
    canonical_custom_module_name: Callable[[str], str],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    now_iso: Callable[[], str],
    short_commit: Callable[[str | None], str],
) -> None:
    """Capture current module snapshot as baseline for next ComfyUI start."""
    module_name = canonical_custom_module_name(module_name)
    state = load_module_state()
    now = now_iso()
    entry = state.get(module_name, {})
    prev_commit = entry.get("installed_commit")
    current_commit = result.get("installed_commit")
    if not entry.get("first_seen_at"):
        entry["first_seen_at"] = now
    if current_commit and current_commit != prev_commit:
        entry["last_local_change_at"] = now
    entry["last_checked_at"] = now
    entry["installed_commit"] = current_commit
    entry["installed_updated_at"] = result.get("installed_updated_at")
    entry["remote_updated_at"] = result.get("remote_updated_at")
    entry["update_available"] = result.get("update_available")
    entry["update_status"] = result.get("update_status")
    entry["module_path"] = result.get("module_path")
    entry["repository"] = result.get("repository")
    state[module_name] = entry
    result["last_checked_at"] = entry.get("last_checked_at")
    result["last_local_change_at"] = entry.get("last_local_change_at")
    startup_prev = (entry.get("pending_prev_commit") or entry.get("startup_prev_commit") or "").strip()
    startup_new = (entry.get("pending_new_commit") or entry.get("startup_new_commit") or "").strip()
    result["updated_between_runs"] = (
        bool(startup_prev and startup_new)
        or bool(entry.get("pending_commit_change"))
        or bool(entry.get("pending_local_change"))
    )
    result["startup_prev_commit_short"] = short_commit(startup_prev) if startup_prev else ""
    result["startup_new_commit_short"] = short_commit(startup_new) if startup_new else ""
    result["startup_update_at"] = entry.get("pending_update_at") or entry.get("startup_update_at") or ""
    save_module_state(state)


def apply_node_change_info(
    result: dict[str, Any],
    group: str,
    module_name: str,
    *,
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
) -> None:
    """Attach node-level startup/pending change markers to module info payload."""
    state = load_module_state()
    tracker = state.get("__node_tracker__")
    if not isinstance(tracker, dict):
        return
    startup_changes = tracker.get("pending_changes") or tracker.get("startup_changes")
    if isinstance(startup_changes, dict):
        group_changes = startup_changes.get(group)
        if isinstance(group_changes, dict):
            entry = group_changes.get(module_name)
            if isinstance(entry, dict):
                new_nodes = entry.get("new_nodes")
                upd_nodes = entry.get("updated_nodes")
                result["new_nodes_between_runs"] = new_nodes if isinstance(new_nodes, list) else []
                result["updated_nodes_between_runs"] = upd_nodes if isinstance(upd_nodes, list) else []
                result["startup_node_update_at"] = entry.get("at") or ""
                if result["new_nodes_between_runs"] or result["updated_nodes_between_runs"]:
                    result["updated_between_runs"] = True

    startup_new_modules = tracker.get("pending_new_modules") or tracker.get("startup_new_modules")
    if isinstance(startup_new_modules, dict):
        group_new = startup_new_modules.get(group)
        if isinstance(group_new, list) and module_name in group_new:
            result["new_module_between_runs"] = True
            result["updated_between_runs"] = True


def acknowledge_module_novelty(
    group: str,
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    clear_module_info_cache: Callable[[], None],
) -> None:
    """Clear pending novelty markers for one module after explicit user refresh."""
    group_name = (group or "").strip().lower()
    module = (module_name or "").strip()
    if not module:
        return
    if group_name == "custom":
        module = canonical_custom_module_name(module)

    state = load_module_state()
    if not isinstance(state, dict):
        return

    changed = False
    entry = state.get(module)
    if isinstance(entry, dict):
        for key in (
            "pending_prev_commit",
            "pending_new_commit",
            "pending_update_at",
            "pending_commit_change",
            "pending_local_change",
            "startup_prev_commit",
            "startup_new_commit",
            "startup_update_at",
        ):
            if key in entry:
                entry.pop(key, None)
                changed = True
        state[module] = entry

    tracker = state.get("__node_tracker__")
    if isinstance(tracker, dict):
        for pending_key, legacy_key in (("pending_changes", "startup_changes"), ("pending_new_modules", "startup_new_modules")):
            bucket = tracker.get(pending_key)
            if not isinstance(bucket, dict):
                bucket = tracker.get(legacy_key)
            if not isinstance(bucket, dict):
                continue
            group_bucket = bucket.get(group_name)
            if isinstance(group_bucket, dict) and module in group_bucket:
                group_bucket.pop(module, None)
                changed = True
            elif isinstance(group_bucket, list) and module in group_bucket:
                bucket[group_name] = [x for x in group_bucket if x != module]
                changed = True
            updated_group_bucket = bucket.get(group_name)
            if isinstance(updated_group_bucket, dict) and not updated_group_bucket:
                bucket.pop(group_name, None)
            if isinstance(updated_group_bucket, list) and not updated_group_bucket:
                bucket.pop(group_name, None)
            tracker[pending_key] = bucket
        state["__node_tracker__"] = tracker

    if changed:
        clear_module_info_cache()
        save_module_state(state)


def acknowledge_all_novelty(
    *,
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    clear_module_info_cache: Callable[[], None],
) -> dict[str, Any]:
    """Clear pending novelty markers for all modules after global refresh."""
    state = load_module_state()
    if not isinstance(state, dict):
        return {"status": "ok", "changed": False}

    changed = False
    cleared_modules = 0
    for module_name, entry in list(state.items()):
        if str(module_name).startswith("__") or not isinstance(entry, dict):
            continue
        before = dict(entry)
        for key in (
            "pending_prev_commit",
            "pending_new_commit",
            "pending_update_at",
            "pending_commit_change",
            "pending_local_change",
            "startup_prev_commit",
            "startup_new_commit",
            "startup_update_at",
        ):
            entry.pop(key, None)
        if entry != before:
            state[module_name] = entry
            cleared_modules += 1
            changed = True

    tracker = state.get("__node_tracker__")
    if isinstance(tracker, dict):
        for key in ("pending_changes", "pending_new_modules", "startup_changes", "startup_new_modules"):
            value = tracker.get(key)
            if isinstance(value, dict) and value:
                tracker[key] = {}
                changed = True
        state["__node_tracker__"] = tracker

    if changed:
        clear_module_info_cache()
        save_module_state(state)
    return {"status": "ok", "changed": changed, "cleared_modules": cleared_modules}


def announce_tracked_module_updates(
    *,
    local_only: bool = False,
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    now_iso: Callable[[], str],
    discover_custom_modules: Callable[[], list[str]],
    canonical_custom_module_name: Callable[[str], str],
    module_git_state: Callable[[str], dict[str, Any]],
    manager_meta_for_module: Callable[[str, str | None], dict[str, Any] | None],
    infer_update_from_manager_stats: Callable[[str | None, str | None], tuple[bool | None, str | None]],
    module_worktree_signature: Callable[[str], str],
    build_node_snapshots: Callable[[], dict[str, dict[str, dict[str, dict[str, str]]]]],
) -> dict[str, Any]:
    """Build per-module/node change info by comparing saved and current snapshots."""
    state = load_module_state()
    if not isinstance(state, dict):
        return {"modules_need_update": 0}

    now = now_iso()
    changed = False
    modules_need_update = 0
    modules_unknown_update = 0
    modules_checked = 0
    update_available_modules: list[str] = []
    unknown_update_modules: list[str] = []
    local_change_modules: list[str] = []
    commit_change_modules: list[str] = []

    known_modules = set(discover_custom_modules())
    for key in list(state.keys()):
        if key.startswith("__"):
            continue
        canonical = canonical_custom_module_name(key)
        known_modules.add(canonical)
        if canonical != key:
            src = state.get(key)
            dst = state.get(canonical)
            if isinstance(src, dict):
                if not isinstance(dst, dict):
                    state[canonical] = dict(src)
                else:
                    merged = dict(dst)
                    for mk, mv in src.items():
                        if mk not in merged or not merged.get(mk):
                            merged[mk] = mv
                    state[canonical] = merged
                state.pop(key, None)
                changed = True

    for module_name in sorted(known_modules, key=str.lower):
        modules_checked += 1
        entry = state.get(module_name, {})
        if not isinstance(entry, dict):
            entry = {}
        prev_commit = (entry.get("installed_commit") or "").strip()
        git_state = module_git_state(module_name)
        current_commit = (git_state.get("installed_commit") or "").strip()
        before = dict(entry)
        manager_meta = manager_meta_for_module(module_name, entry.get("repository"))
        manager_repo = ""
        if isinstance(manager_meta, dict):
            manager_repo = str(manager_meta.get("repository") or "").strip()

        entry["last_checked_at"] = now
        needs_update: bool | None = None
        had_worktree_signature = "worktree_signature" in entry
        prev_worktree_sig = str(entry.get("worktree_signature") or "")
        curr_worktree_sig = module_worktree_signature(module_name)
        if curr_worktree_sig != prev_worktree_sig:
            if curr_worktree_sig and had_worktree_signature:
                entry["pending_local_change"] = True
                entry["pending_update_at"] = now
            elif not curr_worktree_sig and had_worktree_signature:
                entry.pop("pending_local_change", None)
                if not (
                    entry.get("pending_commit_change")
                    or entry.get("pending_prev_commit")
                    or entry.get("pending_new_commit")
                ):
                    entry.pop("pending_update_at", None)
            entry["worktree_signature"] = curr_worktree_sig
        elif not curr_worktree_sig and had_worktree_signature and entry.get("pending_local_change"):
            # If worktree returned to clean state, drop stale local-change marker.
            entry.pop("pending_local_change", None)
            if not (
                entry.get("pending_commit_change")
                or entry.get("pending_prev_commit")
                or entry.get("pending_new_commit")
            ):
                entry.pop("pending_update_at", None)
        if git_state:
            entry["module_path"] = git_state.get("module_path") or entry.get("module_path")
            entry["repository"] = git_state.get("repository") or entry.get("repository")
            entry["installed_updated_at"] = git_state.get("installed_updated_at") or entry.get("installed_updated_at")
            if not local_only:
                entry["remote_updated_at"] = git_state.get("remote_updated_at") or entry.get("remote_updated_at")
                if bool(git_state.get("manager_cnr_nightly")):
                    needs_update = False
                else:
                    behind = git_state.get("behind")
                    remote_head = (git_state.get("remote_head") or "").strip()
                    if isinstance(behind, int):
                        needs_update = behind > 0
                    elif git_state.get("has_upstream") and remote_head and current_commit:
                        needs_update = remote_head != current_commit
        if manager_repo and not entry.get("repository"):
            entry["repository"] = manager_repo
        if not local_only:
            inferred_update, inferred_remote_updated_at = infer_update_from_manager_stats(
                entry.get("repository"),
                entry.get("installed_updated_at"),
            )
            if inferred_remote_updated_at and not entry.get("remote_updated_at"):
                entry["remote_updated_at"] = inferred_remote_updated_at
            if not isinstance(needs_update, bool) and isinstance(inferred_update, bool):
                needs_update = inferred_update

        if isinstance(needs_update, bool):
            entry["update_available"] = needs_update
            entry["update_status"] = "can_update" if needs_update else "up_to_date"
        else:
            entry["update_available"] = None
            entry["update_status"] = "unknown"

        if current_commit:
            if prev_commit and current_commit != prev_commit:
                entry["installed_commit"] = current_commit
                entry["last_local_change_at"] = now
                entry["pending_prev_commit"] = prev_commit
                entry["pending_new_commit"] = current_commit
                entry["pending_update_at"] = now
                entry["pending_commit_change"] = True
                entry.pop("pending_local_change", None)
                entry["startup_prev_commit"] = prev_commit
                entry["startup_new_commit"] = current_commit
                entry["startup_update_at"] = now
                commit_change_modules.append(module_name)
            else:
                entry["installed_commit"] = current_commit

        if bool(entry.get("pending_local_change")):
            local_change_modules.append(module_name)

        if isinstance(needs_update, bool) and needs_update:
            modules_need_update += 1
            update_available_modules.append(module_name)
        elif not isinstance(needs_update, bool):
            modules_unknown_update += 1
            unknown_update_modules.append(module_name)

        state[module_name] = entry
        if entry != before:
            changed = True

    tracker = state.get("__node_tracker__")
    if not isinstance(tracker, dict):
        tracker = {}
    prev_snapshots_raw = tracker.get("snapshots")
    prev_snapshots = prev_snapshots_raw if isinstance(prev_snapshots_raw, dict) else {}
    prev_module_sets_raw = tracker.get("module_sets")
    prev_module_sets = prev_module_sets_raw if isinstance(prev_module_sets_raw, dict) else {}
    current_snapshots = build_node_snapshots()
    startup_changes: dict[str, dict[str, dict[str, Any]]] = {}
    startup_new_modules: dict[str, list[str]] = {}
    pending_changes_raw = tracker.get("pending_changes")
    pending_changes: dict[str, dict[str, dict[str, Any]]] = (
        pending_changes_raw if isinstance(pending_changes_raw, dict) else {}
    )
    pending_new_modules_raw = tracker.get("pending_new_modules")
    pending_new_modules: dict[str, list[str]] = (
        pending_new_modules_raw if isinstance(pending_new_modules_raw, dict) else {}
    )

    current_module_sets: dict[str, list[str]] = {}
    for group_name, modules in current_snapshots.items():
        if isinstance(modules, dict):
            current_module_sets[group_name] = sorted(modules.keys(), key=str.lower)
    custom_from_fs = discover_custom_modules()
    if custom_from_fs:
        existing = set(current_module_sets.get("custom", []))
        current_module_sets["custom"] = sorted(existing.union(custom_from_fs), key=str.lower)

    for group_name, modules in current_snapshots.items():
        if not isinstance(modules, dict):
            continue
        group_prev = prev_snapshots.get(group_name)
        group_prev = group_prev if isinstance(group_prev, dict) else {}
        for module_name, current_snapshot in modules.items():
            if not isinstance(current_snapshot, dict):
                continue
            prev_snapshot_raw = group_prev.get(module_name)
            prev_snapshot = prev_snapshot_raw if isinstance(prev_snapshot_raw, dict) else {}
            prev_names = {k for k in prev_snapshot if isinstance(k, str)}
            curr_names = {k for k in current_snapshot if isinstance(k, str)}

            new_nodes: list[str] = []
            updated_nodes: list[str] = []
            if prev_snapshot:
                new_nodes = sorted(curr_names - prev_names)
                for node_name in sorted(curr_names & prev_names):
                    prev_node = prev_snapshot.get(node_name, {})
                    prev_sig = prev_node.get("sig") if isinstance(prev_node, dict) else None
                    curr_sig = current_snapshot.get(node_name, {}).get("sig")
                    if prev_sig != curr_sig:
                        updated_nodes.append(node_name)

            if new_nodes or updated_nodes:
                startup_changes.setdefault(group_name, {})[module_name] = {
                    "new_nodes": new_nodes,
                    "updated_nodes": updated_nodes,
                    "at": now,
                }
                existing_entry = pending_changes.setdefault(group_name, {}).get(module_name, {})
                prev_new = existing_entry.get("new_nodes") if isinstance(existing_entry, dict) else []
                prev_updated = existing_entry.get("updated_nodes") if isinstance(existing_entry, dict) else []
                merged_new = sorted(set(prev_new if isinstance(prev_new, list) else []).union(new_nodes))
                merged_updated = sorted(
                    set(prev_updated if isinstance(prev_updated, list) else []).union(updated_nodes)
                )
                pending_changes.setdefault(group_name, {})[module_name] = {
                    "new_nodes": merged_new,
                    "updated_nodes": merged_updated,
                    "at": now,
                }

    for group_name, current_list in current_module_sets.items():
        prev_list_raw = prev_module_sets.get(group_name)
        if not isinstance(prev_list_raw, list):
            continue
        prev_set = {x for x in prev_list_raw if isinstance(x, str)}
        curr_set = {x for x in current_list if isinstance(x, str)}
        new_modules = sorted(curr_set - prev_set, key=str.lower)
        if new_modules:
            startup_new_modules[group_name] = new_modules
            existing = pending_new_modules.get(group_name)
            existing_list = existing if isinstance(existing, list) else []
            pending_new_modules[group_name] = sorted(set(existing_list).union(new_modules), key=str.lower)

    if prev_snapshots != current_snapshots:
        changed = True
    if prev_module_sets != current_module_sets:
        changed = True
    tracker["snapshots"] = current_snapshots
    tracker["startup_changes"] = startup_changes
    tracker["module_sets"] = current_module_sets
    tracker["startup_new_modules"] = startup_new_modules
    tracker["pending_changes"] = pending_changes
    tracker["pending_new_modules"] = pending_new_modules
    tracker["updated_at"] = now
    state["__node_tracker__"] = tracker

    if changed:
        save_module_state(state)
    node_changed_modules = sorted(
        {
            module_name
            for modules in startup_changes.values()
            if isinstance(modules, dict)
            for module_name in modules.keys()
            if isinstance(module_name, str)
        },
        key=str.lower,
    )
    return {
        "modules_need_update": modules_need_update,
        "modules_unknown_update": modules_unknown_update,
        "modules_checked": modules_checked,
        "update_available_modules": sorted(update_available_modules, key=str.lower),
        "unknown_update_modules": sorted(unknown_update_modules, key=str.lower),
        "local_change_modules": sorted(set(local_change_modules), key=str.lower),
        "commit_change_modules": sorted(set(commit_change_modules), key=str.lower),
        "node_changed_modules": node_changed_modules,
        "new_modules_between_runs": startup_new_modules,
    }
