"""
Module: utils/module_browser/module_info.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Module-info payload builders for Module Node Picker backend.

Purpose:
    Extracts module-info and cached-badge assembly from the monolithic API
    file so behavior stays stable while internals become easier to test.
"""

from __future__ import annotations

from typing import Any, Callable


def cached_module_flags(
    *,
    group_name: str,
    module_name: str,
    state: dict[str, Any] | None,
    canonical_custom_module_name: Callable[[str], str],
    custom_update_checked_flag: Callable[[dict[str, Any] | None], bool],
) -> dict[str, Any]:
    """Return cached dropdown badge flags for one module entry."""
    group_norm = (group_name or "").strip().lower()
    module = (module_name or "").strip()
    if not module:
        return {
            "updated_between_runs": False,
            "new_module_between_runs": False,
            "update_available": False,
            "update_status": "",
        }

    state_cache = state if isinstance(state, dict) else {}
    canonical = canonical_custom_module_name(module) if group_norm == "custom" else module
    entry = state_cache.get(canonical) if isinstance(state_cache, dict) else None

    updated_between_runs = False
    new_module_between_runs = False
    update_available = False
    update_status = ""
    custom_update_checked = custom_update_checked_flag(state_cache) if group_norm == "custom" else False

    if isinstance(entry, dict):
        startup_prev = (entry.get("pending_prev_commit") or entry.get("startup_prev_commit") or "").strip()
        startup_new = (entry.get("pending_new_commit") or entry.get("startup_new_commit") or "").strip()
        updated_between_runs = (
            bool(startup_prev and startup_new)
            or bool(entry.get("pending_commit_change"))
            or bool(entry.get("pending_local_change"))
            or bool(entry.get("worktree_signature"))
        )
        if group_norm == "custom":
            update_available = bool(entry.get("update_available"))
            if custom_update_checked:
                cached_status = str(entry.get("update_status") or "").strip().lower()
                if cached_status in {"can_update", "up_to_date", "unknown"}:
                    update_status = cached_status

    tracker = state_cache.get("__node_tracker__") if isinstance(state_cache, dict) else None
    if isinstance(tracker, dict):
        startup_changes = tracker.get("pending_changes") or tracker.get("startup_changes")
        if isinstance(startup_changes, dict):
            group_changes = startup_changes.get(group_norm)
            if isinstance(group_changes, dict) and canonical in group_changes:
                updated_between_runs = True

        startup_new_modules = tracker.get("pending_new_modules") or tracker.get("startup_new_modules")
        if isinstance(startup_new_modules, dict):
            group_new = startup_new_modules.get(group_norm)
            if isinstance(group_new, list) and canonical in group_new:
                new_module_between_runs = True
                updated_between_runs = True

    return {
        "updated_between_runs": updated_between_runs,
        "new_module_between_runs": new_module_between_runs,
        "update_available": update_available,
        "update_status": update_status,
    }


def resolve_module_info_uncached(
    *,
    group: str,
    module_name: str,
    sync_upstream: bool,
    cache_only: bool,
    canonical_custom_module_name: Callable[[str], str],
    apply_node_change_info: Callable[[dict[str, Any], str, str], None],
    sync_module_upstream: Callable[[str], Any],
    load_module_state: Callable[[], dict[str, Any] | None],
    custom_update_checked_flag: Callable[[dict[str, Any] | None], bool],
    module_git_state: Callable[[str], dict[str, Any]],
    module_repo_url: Callable[[str], str | None],
    manager_meta_for_module: Callable[[str, str | None], dict[str, Any] | None],
    module_local_readme_summary: Callable[[str], str | None],
    sanitize_module_description: Callable[[str], str],
    github_id: Callable[[str | None], str],
    infer_update_from_manager_stats: Callable[[str | None, str | None], tuple[bool | None, str | None]],
    short_commit: Callable[[str], str],
    remember_module_state: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    """Build one module-info payload without external caching side-effects."""
    group_norm = (group or "").strip().lower()
    module = (module_name or "").strip()
    if group_norm == "custom":
        module = canonical_custom_module_name(module)

    result: dict[str, Any] = {
        "module": module,
        "group": group_norm,
        "title": module,
        "author": "",
        "description": "",
        "repository": "",
        "owner_url": "",
        "module_path": "",
        "installed_commit": "",
        "installed_commit_short": "",
        "installed_updated_at": "",
        "remote_updated_at": "",
        "update_available": None,
        "update_status": "unknown",
        "git_has_upstream": False,
        "git_ahead": None,
        "git_behind": None,
        "last_checked_at": "",
        "last_local_change_at": "",
        "updated_between_runs": False,
        "startup_prev_commit_short": "",
        "startup_new_commit_short": "",
        "startup_update_at": "",
        "new_nodes_between_runs": [],
        "updated_nodes_between_runs": [],
        "startup_node_update_at": "",
        "new_module_between_runs": False,
        "requirements_update_pending": False,
        "requirements_pending_before_commit": "",
        "requirements_pending_after_commit": "",
        "requirements_pending_updated_at": "",
        "source": "none",
    }

    if group_norm != "custom":
        if group_norm in {"core", "core_extras", "api"}:
            result["author"] = "ComfyUI"
            result["repository"] = "https://github.com/comfyanonymous/ComfyUI"
            result["owner_url"] = result["repository"]
            result["description"] = {
                "core": "Built-in ComfyUI nodes.",
                "core_extras": "Built-in ComfyUI extras module.",
                "api": "Built-in ComfyUI API nodes module.",
            }.get(group_norm, "")
            result["source"] = "builtin"
            result["update_status"] = ""
            result["update_available"] = False
        apply_node_change_info(result, group_norm, module)
        return result

    if sync_upstream and not cache_only:
        sync_module_upstream(module)

    state_cache = load_module_state() or {}
    cache_entry = state_cache.get(module) if isinstance(state_cache, dict) else None

    git_state: dict[str, Any] = {}
    if not cache_only:
        git_state = module_git_state(module) or {}

    repo_url = None
    if isinstance(cache_entry, dict):
        repo_url = cache_entry.get("repository")
    if not repo_url:
        repo_url = git_state.get("repository") if git_state else None
    if not repo_url and not cache_only:
        repo_url = module_repo_url(module)
    meta = manager_meta_for_module(module, repo_url)
    if isinstance(meta, dict) and not repo_url:
        repo_url = meta.get("repository")
    repo_gid = github_id(repo_url)

    if meta is not None:
        result["title"] = meta.get("title") or module
        result["author"] = meta.get("author") or ""
        result["description"] = sanitize_module_description(meta.get("description") or "")
        result["repository"] = meta.get("repository") or repo_url or ""
        result["source"] = "comfyui-manager"
    else:
        result["repository"] = repo_url or ""
        result["description"] = sanitize_module_description(module_local_readme_summary(module) or "")
        result["source"] = "local"

    if not result["author"] and repo_gid:
        result["author"] = repo_gid.split("/", 1)[0]
    if not result["description"]:
        result["description"] = "No description found."
    if result["repository"]:
        result["owner_url"] = result["repository"]

    custom_update_checked = custom_update_checked_flag(state_cache) if cache_only else False
    if cache_only and isinstance(cache_entry, dict):
        result["module_path"] = cache_entry.get("module_path") or ""
        result["installed_commit"] = cache_entry.get("installed_commit") or ""
        result["installed_commit_short"] = (result["installed_commit"] or "")[:8]
        result["installed_updated_at"] = cache_entry.get("installed_updated_at") or ""
        result["remote_updated_at"] = cache_entry.get("remote_updated_at") or ""
        startup_prev = (cache_entry.get("pending_prev_commit") or cache_entry.get("startup_prev_commit") or "").strip()
        startup_new = (cache_entry.get("pending_new_commit") or cache_entry.get("startup_new_commit") or "").strip()
        result["updated_between_runs"] = (
            bool(startup_prev and startup_new)
            or bool(cache_entry.get("pending_commit_change"))
            or bool(cache_entry.get("pending_local_change"))
            or bool(cache_entry.get("worktree_signature"))
        )
        result["startup_prev_commit_short"] = short_commit(startup_prev) if startup_prev else ""
        result["startup_new_commit_short"] = short_commit(startup_new) if startup_new else ""
        result["startup_update_at"] = cache_entry.get("pending_update_at") or cache_entry.get("startup_update_at") or ""
        update_available = cache_entry.get("update_available")
        if isinstance(update_available, bool):
            result["update_available"] = update_available
            result["update_status"] = "can_update" if update_available else "up_to_date"
        elif not custom_update_checked:
            result["update_available"] = False
            result["update_status"] = "up_to_date"
        elif isinstance(cache_entry.get("update_status"), str):
            result["update_status"] = str(cache_entry.get("update_status") or "unknown")
        result["last_checked_at"] = cache_entry.get("last_checked_at") or ""
        result["last_local_change_at"] = cache_entry.get("last_local_change_at") or ""
    elif cache_only and not custom_update_checked:
        result["update_available"] = False
        result["update_status"] = "up_to_date"
    elif git_state:
        result["module_path"] = git_state.get("module_path") or ""
        result["installed_commit"] = git_state.get("installed_commit") or ""
        result["installed_commit_short"] = (result["installed_commit"] or "")[:8]
        result["installed_updated_at"] = git_state.get("installed_updated_at") or ""
        result["remote_updated_at"] = git_state.get("remote_updated_at") or ""
        result["git_has_upstream"] = bool(git_state.get("has_upstream"))
        result["git_ahead"] = git_state.get("ahead")
        result["git_behind"] = git_state.get("behind")
        behind = git_state.get("behind")
        remote_head = git_state.get("remote_head")
        if isinstance(behind, int):
            result["update_available"] = behind > 0
            result["update_status"] = "can_update" if behind > 0 else "up_to_date"
        elif result["git_has_upstream"] and remote_head and result["installed_commit"]:
            if remote_head == result["installed_commit"]:
                result["update_available"] = False
                result["update_status"] = "up_to_date"
            else:
                result["update_available"] = True
                result["update_status"] = "can_update"

    inferred_update, inferred_remote_updated_at = infer_update_from_manager_stats(
        result.get("repository"),
        result.get("installed_updated_at"),
    )
    if inferred_remote_updated_at and not result.get("remote_updated_at"):
        result["remote_updated_at"] = inferred_remote_updated_at
    if not isinstance(result.get("update_available"), bool) and isinstance(inferred_update, bool):
        result["update_available"] = inferred_update
        result["update_status"] = "can_update" if inferred_update else "up_to_date"

    if not cache_only:
        remember_module_state(module, result)

    if isinstance(cache_entry, dict):
        result["requirements_update_pending"] = bool(cache_entry.get("pending_requirements_update"))
        result["requirements_pending_before_commit"] = str(cache_entry.get("pending_requirements_before_commit") or "")
        result["requirements_pending_after_commit"] = str(cache_entry.get("pending_requirements_after_commit") or "")
        result["requirements_pending_updated_at"] = str(cache_entry.get("pending_requirements_updated_at") or "")

    apply_node_change_info(result, group_norm, module)
    return result

