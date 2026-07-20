"""
Module: utils/module_browser/pull_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Git pull/update helper functions for ComfyUI and custom modules.

Purpose:
    Extracts pull orchestration logic from backend API module while keeping
    behavior stable through facade wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def is_git_local_changes_block(text: str | None) -> bool:
    """Return true when git pull failed due to local changes conflict."""
    lower = str(text or "").strip().lower()
    if not lower:
        return False
    markers = (
        "please commit your changes or stash them before you merge",
        "your local changes to the following files would be overwritten by merge",
        "local changes would be overwritten by merge",
        "the following untracked working tree files would be overwritten by merge",
        "please move or remove them before you merge",
        "cannot pull with rebase: you have unstaged changes",
        "cannot pull with rebase: your index contains uncommitted changes",
        "сделайте коммит или спрячьте ваши изменения перед слиянием веток",
        "ваши локальные изменения в указанных файлах будут перезаписаны при слиянии",
        "указанные неотслеживаемые файлы в рабочем каталоге будут перезаписаны при слиянии",
        "переместите эти файлы или удалите их перед переключением веток",
    )
    return any(marker in lower for marker in markers)


def pull_comfyui(
    *,
    comfyui_root: Callable[[], Path | None],
    update_console_log: Callable[[str, str], None],
    run_git: Callable[[list[str], float], str | None],
    git_pick_remote: Callable[[Path, str | None], str | None],
    git_resolve_remote_ref: Callable[[Path, str, str | None, str | None], tuple[str | None, str | None]],
    run_command: Callable[[list[str], float, bool], dict[str, Any]],
    requirements_changed_between: Callable[[Path, str, str], bool],
    set_comfyui_requirements_pending: Callable[[bool, str, str], None],
    perf_counter: Callable[[], float],
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Pull latest ComfyUI changes using ff-only strategy."""
    root = comfyui_root()
    result: dict[str, Any] = {
        "module": "ComfyUI",
        "status": "error",
        "message": "",
        "updated": False,
        "requirements_changed": False,
        "requirements_path": "",
        "before_commit": "",
        "after_commit": "",
    }
    if root is None:
        result["status"] = "not_found"
        result["message"] = "ComfyUI root not found"
        return result
    root_str = str(root)
    result["requirements_path"] = str(root / "requirements.txt")
    update_console_log(f"ComfyUI pull: repo={root_str}", "verbose")
    is_git = run_git(["git", "-C", root_str, "rev-parse", "--is-inside-work-tree"], 2.0)
    if is_git != "true":
        result["status"] = "no_git"
        result["message"] = "ComfyUI is not a git repository"
        return result

    branch = run_git(["git", "-C", root_str, "rev-parse", "--abbrev-ref", "HEAD"], 2.0) or ""
    upstream = run_git(["git", "-C", root_str, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], 2.0)
    remote_name = git_pick_remote(root, upstream)
    if not remote_name:
        result["status"] = "no_remote"
        result["message"] = "ComfyUI remote is not configured"
        return result

    update_console_log(f"ComfyUI pull: fetch {remote_name}...", "verbose")
    run_git(["git", "-C", root_str, "fetch", "--quiet", remote_name], 20.0)
    remote_ref, remote_branch = git_resolve_remote_ref(root, remote_name, branch, upstream)
    if not remote_ref:
        result["status"] = "no_upstream"
        result["message"] = "ComfyUI upstream/default branch is not configured"
        return result

    if branch == "HEAD" and remote_branch:
        checkout = run_command(["git", "-C", root_str, "checkout", remote_branch], timeout, True)
        if not checkout.get("ok"):
            checkout = run_command(["git", "-C", root_str, "checkout", "-B", remote_branch, remote_ref], timeout, True)
        if not checkout.get("ok"):
            result["status"] = "error"
            result["message"] = str(checkout.get("stderr") or checkout.get("stdout") or "git checkout failed")
            return result

    before_commit = run_git(["git", "-C", root_str, "rev-parse", "HEAD"], 2.0) or ""
    result["before_commit"] = before_commit
    if upstream:
        pull_cmd = ["git", "-C", root_str, "pull", "--ff-only"]
    else:
        pull_cmd = ["git", "-C", root_str, "pull", "--ff-only", remote_name]
        if remote_branch:
            pull_cmd.append(remote_branch)
    update_console_log(f"ComfyUI pull: running {' '.join(pull_cmd)}", "verbose")
    pull_started = perf_counter()
    pull = run_command(pull_cmd, timeout, True)
    update_console_log(f"ComfyUI pull: command finished in {perf_counter() - pull_started:.2f}s", "verbose")
    if not pull.get("ok"):
        result["status"] = "error"
        result["message"] = str(pull.get("stderr") or pull.get("stdout") or "git pull failed")
        return result

    after_commit = run_git(["git", "-C", root_str, "rev-parse", "HEAD"], 2.0) or ""
    result["after_commit"] = after_commit
    updated = bool(before_commit and after_commit and before_commit != after_commit)
    result["updated"] = updated
    if updated:
        result["status"] = "updated"
        result["message"] = "ComfyUI updated"
        requirements_changed = requirements_changed_between(root, before_commit, after_commit)
        result["requirements_changed"] = requirements_changed
        if requirements_changed:
            set_comfyui_requirements_pending(True, before_commit, after_commit)
    else:
        result["status"] = "up_to_date"
        result["message"] = "already up to date"
    return result


def pull_custom_module(
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    module_dir_resolver: Callable[[str], Path | None],
    update_console_log: Callable[[str, str], None],
    run_git: Callable[[list[str], float], str | None],
    git_pick_remote: Callable[[Path, str | None], str | None],
    git_resolve_remote_ref: Callable[[Path, str, str | None, str | None], tuple[str | None, str | None]],
    bootstrap_module_remote_from_manager: Callable[[str, Path], bool],
    run_command: Callable[[list[str], float, bool], dict[str, Any]],
    is_git_local_changes_block_fn: Callable[[str | None], bool],
    requirements_changed_between: Callable[[Path, str, str], bool],
    set_module_requirements_pending: Callable[[str, bool, str, str], None],
    perf_counter: Callable[[], float],
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Pull latest changes for one custom module with optional auto-stash retry."""
    module = canonical_custom_module_name(module_name)
    module_dir = module_dir_resolver(module)
    result: dict[str, Any] = {
        "module": module,
        "status": "error",
        "message": "",
        "updated": False,
        "requirements_changed": False,
        "requirements_path": "",
        "stashed_local_changes": False,
        "stash_ref": "",
        "before_commit": "",
        "after_commit": "",
    }
    if module_dir is None:
        result["status"] = "not_found"
        result["message"] = "module directory not found"
        return result

    update_console_log(f"{module}: repo={module_dir}", "verbose")
    result["requirements_path"] = str(module_dir / "requirements.txt")
    is_git = run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"], 2.0)
    if is_git != "true":
        result["status"] = "no_git"
        result["message"] = "not a git repository"
        return result

    update_console_log(f"{module}: resolving upstream...", "verbose")
    branch = run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "HEAD"], 2.0) or ""
    upstream = run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], 2.0)
    remote_name = git_pick_remote(module_dir, upstream)
    if not remote_name and bootstrap_module_remote_from_manager(module, module_dir):
        upstream = run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], 2.0)
        remote_name = git_pick_remote(module_dir, upstream)
    if not remote_name:
        result["status"] = "no_remote"
        result["message"] = "remote is not configured and manager metadata did not provide repository URL"
        return result

    run_git(["git", "-C", str(module_dir), "fetch", "--quiet", remote_name], 20.0)
    remote_ref, remote_branch = git_resolve_remote_ref(module_dir, remote_name, branch, upstream)
    if not remote_ref:
        result["status"] = "no_upstream"
        result["message"] = "upstream/default branch is not configured"
        return result

    if branch == "HEAD" and remote_branch:
        checkout = run_command(["git", "-C", str(module_dir), "checkout", remote_branch], timeout, True)
        if not checkout.get("ok"):
            checkout = run_command(["git", "-C", str(module_dir), "checkout", "-B", remote_branch, remote_ref], timeout, True)
        if not checkout.get("ok"):
            result["status"] = "error"
            result["message"] = str(checkout.get("stderr") or checkout.get("stdout") or "git checkout failed")
            return result

    before_commit = run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"], 2.0) or ""
    result["before_commit"] = before_commit
    if upstream:
        pull_cmd = ["git", "-C", str(module_dir), "pull", "--ff-only"]
    else:
        pull_cmd = ["git", "-C", str(module_dir), "pull", "--ff-only", remote_name]
        if remote_branch:
            pull_cmd.append(remote_branch)
    update_console_log(f"{module}: running {' '.join(pull_cmd)}", "verbose")
    pull_started = perf_counter()
    pull = run_command(pull_cmd, timeout, True)
    update_console_log(f"{module}: pull command finished in {perf_counter() - pull_started:.2f}s", "verbose")

    if not pull.get("ok"):
        error_text = "{stderr}\n{stdout}".format(
            stderr=str(pull.get("stderr") or ""),
            stdout=str(pull.get("stdout") or ""),
        )
        if is_git_local_changes_block_fn(error_text):
            update_console_log(f"{module}: local changes detected, trying auto-stash before update", "summary")
            stash = run_command(
                ["git", "-C", str(module_dir), "stash", "push", "-u", "-m", "ALEXZ_tools auto-stash before module update"],
                60.0,
                True,
            )
            if not stash.get("ok"):
                result["status"] = "error"
                result["message"] = str(stash.get("stderr") or stash.get("stdout") or "git stash failed")
                return result
            stash_out = str(stash.get("stdout") or stash.get("stderr") or "").strip()
            result["stashed_local_changes"] = True
            result["stash_ref"] = stash_out
            update_console_log(f"{module}: auto-stash created; retrying pull", "verbose")
            pull_started = perf_counter()
            pull = run_command(pull_cmd, timeout, True)
            update_console_log(f"{module}: retry pull finished in {perf_counter() - pull_started:.2f}s", "verbose")

    if not pull.get("ok"):
        result["status"] = "error"
        result["message"] = str(pull.get("stderr") or pull.get("stdout") or "git pull failed")
        return result

    after_commit = run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"], 2.0) or ""
    result["after_commit"] = after_commit
    updated = bool(before_commit and after_commit and before_commit != after_commit)
    result["updated"] = updated
    if updated:
        result["status"] = "updated"
        result["message"] = "module updated (local changes were stashed)" if result.get("stashed_local_changes") else "module updated"
        requirements_changed = requirements_changed_between(module_dir, before_commit, after_commit)
        result["requirements_changed"] = requirements_changed
        if requirements_changed:
            set_module_requirements_pending(module, True, before_commit, after_commit)
    else:
        result["status"] = "up_to_date"
        result["message"] = (
            "already up to date (local changes were stashed)"
            if result.get("stashed_local_changes")
            else "already up to date"
        )
    return result
