"""
Module: utils/module_browser/repo_bootstrap_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Repository bootstrap and requirements-path helpers for update workflows.

Purpose:
    Extracts ComfyUI requirements path lookup and custom-module remote bootstrap
    logic from API facade while preserving behavior and diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def comfyui_requirements_path(*, comfyui_root_fn: Callable[[], Path | None]) -> Path | None:
    """Resolve requirements.txt path for the main ComfyUI repository."""
    root = comfyui_root_fn()
    if root is None:
        return None
    req = root / "requirements.txt"
    return req if req.exists() else None


def bootstrap_module_remote_from_manager(
    module_name: str,
    module_dir: Path,
    *,
    git_remote_names_fn: Callable[[Path], list[str]],
    manager_meta_for_module_fn: Callable[[str, str | None], dict[str, Any] | None],
    normalize_repo_url_fn: Callable[[str | None], str | None],
    run_command_fn: Callable[[list[str], float, bool], dict[str, Any]],
    logger_info: Callable[[str, str, str], None] | None = None,
    timeout: float = 20.0,
) -> bool:
    """Configure `origin` remote from manager metadata for repos without remotes."""
    remotes = git_remote_names_fn(module_dir)
    if remotes:
        return True
    meta = manager_meta_for_module_fn(module_name, None)
    repo_url = normalize_repo_url_fn(meta.get("repository")) if isinstance(meta, dict) else None
    if not repo_url:
        return False
    add = run_command_fn(
        ["git", "-C", str(module_dir), "remote", "add", "origin", repo_url],
        timeout,
        True,
    )
    if not bool(add.get("ok")):
        return False
    if logger_info is not None:
        logger_info("Configured origin remote from manager metadata for module %s: %s", module_name, repo_url)
    return True

