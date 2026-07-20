"""
Module: utils/module_browser/comfyui_git_status_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    ComfyUI git-status orchestration helper for Module Node Picker backend.

Purpose:
    Extracts ComfyUI local/remote status collection logic from API facade while
    preserving payload and cache behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def collect_comfyui_git_status(
    *,
    force_refresh: bool,
    mode: str,
    now_ts: float,
    cache: dict[str, tuple[float, dict[str, Any]]],
    ttl_sec: float,
    normalize_comfyui_mode: Callable[[str], str],
    comfyui_status_template: Callable[[str], dict[str, Any]],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    resolve_cached_status: Callable[[dict[str, Any], str], tuple[dict[str, Any] | None, dict[str, Any] | None]],
    apply_cached_pending_fields: Callable[..., dict[str, Any]],
    short_commit: Callable[[str | None], str],
    comfyui_root: Callable[[], Path | None],
    run_git: Callable[[list[str], float], str | None],
    git_pick_remote: Callable[[Path, str | None], str | None],
    github_latest_release: Callable[[str, str], dict[str, Any]],
    resolve_release_ref: Callable[[Path, str, str], tuple[str | None, str]],
    parse_datetime: Callable[[str], Any],
    to_iso: Callable[[Any], str | None],
    git_resolve_remote_ref: Callable[[Path, str, str | None, str | None], tuple[str | None, str | None]],
    persist_comfyui_status: Callable[..., dict[str, Any]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    now_iso: Callable[[], str],
) -> dict[str, Any]:
    """Collect local/remote git status summary for ComfyUI repository."""
    mode_norm = normalize_comfyui_mode(mode)
    cached_mode = cache.get(mode_norm)
    if (
        not force_refresh
        and cached_mode is not None
        and (now_ts - cached_mode[0]) < ttl_sec
    ):
        return dict(cached_mode[1])

    result: dict[str, Any] = comfyui_status_template(mode_norm)

    if not force_refresh:
        state = load_module_state()
        cached_entry, cached_status = resolve_cached_status(state, mode_norm)
        if isinstance(cached_status, dict) and cached_status:
            merged = dict(cached_status)
            merged["check_mode"] = str(merged.get("check_mode") or mode_norm)
            merged = apply_cached_pending_fields(merged, cached_entry, short_commit=short_commit)
            cache[mode_norm] = (now_ts, dict(merged))
            return merged
        cache[mode_norm] = (now_ts, dict(result))
        return result

    root = comfyui_root()
    if root is None:
        cache[mode_norm] = (now_ts, dict(result))
        state = load_module_state()
        if isinstance(state, dict):
            state = persist_comfyui_status(state, mode_norm=mode_norm, result=result, now_iso=now_iso)
            save_module_state(state)
        return result

    result["path"] = str(root)
    is_git = run_git(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], 2.0)
    if is_git != "true":
        cache[mode_norm] = (now_ts, dict(result))
        state = load_module_state()
        if isinstance(state, dict):
            state = persist_comfyui_status(state, mode_norm=mode_norm, result=result, now_iso=now_iso)
            save_module_state(state)
        return result

    result["branch"] = run_git(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], 2.0) or ""
    result["installed_commit"] = run_git(["git", "-C", str(root), "rev-parse", "HEAD"], 2.0) or ""
    result["installed_commit_short"] = short_commit(result["installed_commit"]) if result["installed_commit"] else ""
    result["installed_updated_at"] = run_git(["git", "-C", str(root), "log", "-1", "--format=%cI"], 2.0) or ""

    upstream = run_git(
        ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        2.0,
    )
    result["upstream"] = upstream or ""
    remote_name = git_pick_remote(root, upstream)
    result["remote_name"] = remote_name or ""
    if not remote_name:
        cache[mode_norm] = (now_ts, dict(result))
        return result

    # Keep remote refs fresh to reflect actual GitHub state.
    run_git(["git", "-C", str(root), "fetch", "--quiet", remote_name], 20.0)

    remote_ref = ""
    if mode_norm == "releases":
        release = github_latest_release("comfyanonymous", "ComfyUI")
        tag_name = str(release.get("tag_name") or "").strip()
        result["release_tag"] = tag_name
        result["release_name"] = str(release.get("name") or "").strip()
        result["release_url"] = str(release.get("html_url") or "").strip()
        published = parse_datetime(str(release.get("published_at") or release.get("created_at") or ""))
        if published is not None:
            result["remote_updated_at"] = to_iso(published) or ""
        if not tag_name:
            # Fall back to freshest locally known tag when GitHub latest-release
            # metadata is temporarily unavailable.
            run_git(["git", "-C", str(root), "fetch", "--quiet", remote_name, "--tags"], 25.0)
            tag_name = (
                run_git(
                    [
                        "git",
                        "-C",
                        str(root),
                        "for-each-ref",
                        "--sort=-version:refname",
                        "--count=1",
                        "--format=%(refname:strip=2)",
                        "refs/tags",
                    ],
                    2.0,
                )
                or ""
            ).strip()
            result["release_tag"] = tag_name
            if not tag_name:
                result["release_check_degraded"] = True
                result["release_check_reason"] = "github_release_unavailable"
        tag_ref, release_commit = resolve_release_ref(root, remote_name, tag_name)
        if tag_ref and release_commit:
            remote_ref = tag_ref
            result["remote_ref"] = tag_ref
            result["remote_commit"] = release_commit
            result["remote_commit_short"] = short_commit(release_commit)
            if not result.get("remote_updated_at"):
                result["remote_updated_at"] = run_git(
                    ["git", "-C", str(root), "log", "-1", "--format=%cI", tag_ref],
                    2.0,
                ) or ""
        elif tag_name:
            result["release_check_degraded"] = True
            result["release_check_reason"] = "release_tag_not_resolved"

    if mode_norm == "commits":
        remote_ref, _remote_branch = git_resolve_remote_ref(root, remote_name, result["branch"], upstream)
        result["remote_ref"] = remote_ref or ""
        if remote_ref:
            result["remote_commit"] = run_git(["git", "-C", str(root), "rev-parse", remote_ref], 2.0) or ""
            result["remote_commit_short"] = short_commit(result["remote_commit"]) if result["remote_commit"] else ""
            result["remote_updated_at"] = run_git(
                ["git", "-C", str(root), "log", "-1", "--format=%cI", remote_ref],
                2.0,
            ) or ""

    if result.get("remote_ref") and result.get("remote_commit"):
        counts = run_git(
            ["git", "-C", str(root), "rev-list", "--left-right", "--count", f"HEAD...{result['remote_ref']}"],
            2.0,
        )
        if counts:
            parts = counts.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                result["ahead"] = int(parts[0])
                result["behind"] = int(parts[1])
                result["update_available"] = result["behind"] > 0
                result["update_status"] = "can_update" if result["behind"] > 0 else "up_to_date"
        elif result["installed_commit"] and result["remote_commit"]:
            if result["installed_commit"] == result["remote_commit"]:
                result["update_available"] = False
                result["update_status"] = "up_to_date"
            else:
                # If exact counters are unavailable, assume remote difference requires update.
                result["update_available"] = True
                result["update_status"] = "can_update"
                result["behind"] = 1

    state = load_module_state()
    cached_entry, _cached_status = resolve_cached_status(state, mode_norm)
    result = apply_cached_pending_fields(result, cached_entry, short_commit=short_commit)

    cache[result["check_mode"]] = (now_ts, dict(result))
    state = load_module_state()
    if isinstance(state, dict):
        state = persist_comfyui_status(state, mode_norm=result["check_mode"], result=result, now_iso=now_iso)
        save_module_state(state)
    return result
