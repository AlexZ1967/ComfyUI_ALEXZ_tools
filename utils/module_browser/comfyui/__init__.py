"""
Module: utils/module_browser/comfyui/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    ComfyUI-specific operations and state management.
"""

from .comfyui_tracking_ops import (
    acknowledge_comfyui_novelty,
    track_comfyui_local_update,
)
from .comfyui_state_ops import (
    apply_cached_pending_fields,
    comfyui_status_template,
    persist_comfyui_status,
    resolve_cached_status,
)
from .comfyui_git_status_ops import (
    collect_comfyui_git_status,
)
from .manager_data_ops import (
    infer_update_from_manager_stats,
    load_manager_github_stats,
    load_manager_index,
    manager_stats_last_update,
    resolve_manager_meta_for_module,
)

__all__ = [
    "acknowledge_comfyui_novelty",
    "track_comfyui_local_update",
    "comfyui_status_template",
    "resolve_cached_status",
    "apply_cached_pending_fields",
    "persist_comfyui_status",
    "collect_comfyui_git_status",
    "load_manager_github_stats",
    "load_manager_index",
    "resolve_manager_meta_for_module",
    "manager_stats_last_update",
    "infer_update_from_manager_stats",
]
