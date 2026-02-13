"""
Module: utils/module_browser/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Internal helpers package for Module Node Picker backend decomposition.

Purpose:
    Hosts reusable contracts and registry primitives used during Slice 0 / Phase 3
    refactoring, while keeping external API routes unchanged.
"""

from .component_registry import (
    ComponentEntry,
    ComponentRegistry,
    build_default_component_registry,
    build_registry_snapshot,
    compute_snapshot_signature,
)
from .api_manifest import iter_component_api_routes
from .widget_manifest import WidgetSpec, iter_widget_specs
from .contracts import (
    COMPONENT_REGISTRY_SCHEMA_NAME,
    COMPONENT_REGISTRY_SCHEMA_VERSION,
    MODULE_STATE_SCHEMA_VERSION,
    ensure_module_state_schema,
)
from .health import build_component_health_report
from .manifest_check import run_manifest_check
from .catalog import (
    build_catalog,
    build_group_catalog,
    build_group_modules,
    collect_nodes,
    filter_modules,
)
from .module_info_text import (
    module_local_readme_summary,
    sanitize_module_description,
)
from .module_info import (
    cached_module_flags,
    resolve_module_info_uncached,
)
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
from .update_ops import (
    install_comfyui_requirements,
    install_requirements_for_modules,
    install_module_requirements,
    requirements_changed_between,
)
from .pull_ops import (
    is_git_local_changes_block,
    pull_comfyui,
    pull_custom_module,
)
from .state_store import (
    load_state_file,
    save_state_file,
)
from .tracker_ops import (
    acknowledge_all_novelty,
    acknowledge_module_novelty,
    announce_tracked_module_updates,
    apply_node_change_info,
    remember_module_state,
)
from .comfyui_tracking_ops import (
    acknowledge_comfyui_novelty,
    track_comfyui_local_update,
)
from .node_snapshot_ops import (
    build_node_snapshots,
    file_digest,
    node_source_file,
    relative_to_custom_roots,
)
from .runtime_refresh_ops import (
    refresh_module_runtime_state,
)
from .update_job_ops import (
    run_module_update_job,
)
from .refresh_job_ops import (
    run_refresh_job,
)
from .module_identity import (
    build_custom_module_aliases,
    canonical_custom_module_name,
    discover_custom_modules,
    normalize_module_token,
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
from .component_registry_payload_ops import (
    collect_component_registry_payload,
)
from .manager_data_ops import (
    infer_update_from_manager_stats,
    load_manager_github_stats,
    load_manager_index,
    manager_stats_last_update,
    resolve_manager_meta_for_module,
)
from .command_ops import (
    extract_git_repo_from_args,
    is_git_dubious_ownership_error,
    run_command,
    run_git,
    tail_lines,
    try_mark_git_safe_directory,
)
from .catalog_payload_ops import (
    build_group_payload,
    build_module_list_payload,
    build_module_nodes_payload,
)
from .widget_mode_ops import (
    custom_update_checked_flag,
    info_only_rejection_payload,
    normalize_log_mode,
    set_custom_update_checked,
)
from .value_ops import (
    github_id,
    normalize_comfyui_mode,
    normalize_repo_url,
    now_iso,
    parse_datetime,
    pick_repo_url,
    repo_name,
    short_commit,
    to_iso,
)

__all__ = [
    "ComponentEntry",
    "ComponentRegistry",
    "COMPONENT_REGISTRY_SCHEMA_NAME",
    "COMPONENT_REGISTRY_SCHEMA_VERSION",
    "MODULE_STATE_SCHEMA_VERSION",
    "build_default_component_registry",
    "build_component_health_report",
    "run_manifest_check",
    "build_catalog",
    "build_group_catalog",
    "build_group_modules",
    "collect_nodes",
    "filter_modules",
    "module_local_readme_summary",
    "sanitize_module_description",
    "cached_module_flags",
    "resolve_module_info_uncached",
    "git_remote_names",
    "git_pick_remote",
    "git_ref_exists",
    "git_resolve_remote_ref",
    "resolve_release_ref",
    "module_repo_url",
    "module_git_state",
    "module_worktree_signature",
    "sync_module_upstream",
    "requirements_changed_between",
    "install_module_requirements",
    "install_comfyui_requirements",
    "install_requirements_for_modules",
    "is_git_local_changes_block",
    "pull_comfyui",
    "pull_custom_module",
    "load_state_file",
    "save_state_file",
    "remember_module_state",
    "apply_node_change_info",
    "acknowledge_module_novelty",
    "acknowledge_all_novelty",
    "announce_tracked_module_updates",
    "track_comfyui_local_update",
    "acknowledge_comfyui_novelty",
    "node_source_file",
    "relative_to_custom_roots",
    "file_digest",
    "build_node_snapshots",
    "refresh_module_runtime_state",
    "run_module_update_job",
    "run_refresh_job",
    "discover_custom_modules",
    "normalize_module_token",
    "build_custom_module_aliases",
    "canonical_custom_module_name",
    "comfyui_status_template",
    "resolve_cached_status",
    "apply_cached_pending_fields",
    "persist_comfyui_status",
    "collect_comfyui_git_status",
    "collect_component_registry_payload",
    "load_manager_github_stats",
    "load_manager_index",
    "resolve_manager_meta_for_module",
    "manager_stats_last_update",
    "infer_update_from_manager_stats",
    "extract_git_repo_from_args",
    "is_git_dubious_ownership_error",
    "try_mark_git_safe_directory",
    "run_command",
    "run_git",
    "tail_lines",
    "build_group_payload",
    "build_module_list_payload",
    "build_module_nodes_payload",
    "custom_update_checked_flag",
    "info_only_rejection_payload",
    "set_custom_update_checked",
    "normalize_log_mode",
    "short_commit",
    "normalize_repo_url",
    "github_id",
    "repo_name",
    "pick_repo_url",
    "parse_datetime",
    "to_iso",
    "now_iso",
    "normalize_comfyui_mode",
    "ensure_module_state_schema",
    "build_registry_snapshot",
    "compute_snapshot_signature",
    "WidgetSpec",
    "iter_component_api_routes",
    "iter_widget_specs",
]
