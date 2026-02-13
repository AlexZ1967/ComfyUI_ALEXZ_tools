"""
Module: utils/module_browser/path_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Filesystem path resolution helpers for Module Node Picker backend.

Purpose:
    Extracts reusable path-resolution logic (custom roots, manager DB paths,
    module dirs, ComfyUI root) from API facade during Phase 3 decomposition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def custom_nodes_roots(*, folder_paths_module: Any | None, fallback_root: Path) -> list[Path]:
    """Return existing custom_nodes root directories."""
    if folder_paths_module is not None and hasattr(folder_paths_module, "get_folder_paths"):
        try:
            roots = [Path(x) for x in folder_paths_module.get_folder_paths("custom_nodes") if x]
            if roots:
                return roots
        except Exception:
            pass
    return [fallback_root]


def manager_custom_db_path(*, custom_nodes_roots_fn: Callable[[], list[Path]]) -> Path | None:
    """Return path to ComfyUI-Manager custom-node database file."""
    for root in custom_nodes_roots_fn():
        db_path = root / "comfyui-manager" / "custom-node-list.json"
        if db_path.exists():
            return db_path
    return None


def manager_github_stats_path(*, custom_nodes_roots_fn: Callable[[], list[Path]]) -> Path | None:
    """Return path to ComfyUI-Manager github-stats cache file."""
    for root in custom_nodes_roots_fn():
        db_path = root / "comfyui-manager" / "github-stats.json"
        if db_path.exists():
            return db_path
    return None


def module_dir(
    module_name: str,
    *,
    canonical_custom_module_name_fn: Callable[[str], str],
    custom_nodes_roots_fn: Callable[[], list[Path]],
) -> Path | None:
    """Resolve filesystem directory for a custom module by canonical name."""
    module = canonical_custom_module_name_fn(str(module_name or "").strip())
    if not module:
        return None
    for root in custom_nodes_roots_fn():
        candidate = root / module
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def comfyui_root(module_file: str | Path) -> Path | None:
    """Resolve root path of currently running ComfyUI installation."""
    base = Path(module_file).resolve()
    for candidate in (base.parents[2], *base.parents):
        try:
            if (candidate / "nodes.py").exists() and (candidate / ".git").exists():
                return candidate
        except Exception:
            continue
    return None

