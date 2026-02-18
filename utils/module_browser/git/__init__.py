"""
Module: utils/module_browser/git/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Git operations and helpers.
"""

from .git_helpers import (
    git_pick_remote,
    git_ref_exists,
    git_remote_names,
    git_resolve_remote_ref,
    module_git_state,
    module_repo_url,
    module_worktree_signature,
    resolve_release_ref,
    sync_module_upstream,
)
from .pull_ops import (
    is_git_local_changes_block,
    pull_comfyui,
    pull_custom_module,
)
from .command_ops import (
    extract_git_repo_from_args,
    is_git_dubious_ownership_error,
    run_command,
    run_git,
    tail_lines,
    try_mark_git_safe_directory,
)

__all__ = [
    "git_remote_names",
    "git_pick_remote",
    "git_ref_exists",
    "git_resolve_remote_ref",
    "resolve_release_ref",
    "module_repo_url",
    "module_git_state",
    "module_worktree_signature",
    "sync_module_upstream",
    "is_git_local_changes_block",
    "pull_comfyui",
    "pull_custom_module",
    "extract_git_repo_from_args",
    "is_git_dubious_ownership_error",
    "try_mark_git_safe_directory",
    "run_command",
    "run_git",
    "tail_lines",
]
