"""
Module: utils/module_browser/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Internal helpers package for Module Node Picker backend decomposition.

Purpose:
    Hosts reusable contracts and registry primitives used during Slice 0 / Phase 3
    refactoring, while keeping external API routes unchanged.
    
    Organized into functional submodules for easier maintenance and testing.
"""

# Core contracts and infra
from .contracts import (
    COMPONENT_REGISTRY_SCHEMA_NAME,
    COMPONENT_REGISTRY_SCHEMA_VERSION,
    MODULE_STATE_SCHEMA_VERSION,
    ensure_module_state_schema,
)
from .health import build_component_health_report

# Catalog operations (catalog/, core/)
from .catalog import (
    ComponentEntry,
    ComponentRegistry,
    build_default_component_registry,
    build_registry_snapshot,
    compute_snapshot_signature,
    build_catalog,
    build_group_catalog,
    build_group_modules,
    collect_nodes,
    filter_modules,
    build_group_payload,
    build_module_list_payload,
    build_module_nodes_payload,
    collect_component_registry_payload,
    iter_component_api_routes,
    WidgetSpec,
    iter_widget_specs,
)

# Module operations (module/, core/)
from .module import (
    discover_custom_modules,
    normalize_module_token,
    build_custom_module_aliases,
    canonical_custom_module_name,
    cached_module_flags,
    resolve_module_info_uncached,
    module_local_readme_summary,
    sanitize_module_description,
    module_needs_update_now,
    count_custom_modules_need_update,
    count_custom_modules_unknown_update,
    comfyui_needs_update_now,
    classify_by_source_path,
    classify_by_relative_module,
    fallback_annotation,
    module_root,
    node_source_file,
    relative_to_custom_roots,
    file_digest,
    build_node_snapshots,
)

# Git operations (git/)
from .git import (
    git_remote_names,
    git_pick_remote,
    git_ref_exists,
    git_resolve_remote_ref,
    resolve_release_ref,
    module_repo_url,
    module_git_state,
    module_worktree_signature,
    sync_module_upstream,
    is_git_local_changes_block,
    pull_comfyui,
    pull_custom_module,
    extract_git_repo_from_args,
    is_git_dubious_ownership_error,
    try_mark_git_safe_directory,
    run_command,
    run_git,
    tail_lines,
)

# Jobs operations (jobs/)
from .jobs import (
    emit_refresh_progress,
    format_update_status_line,
    refresh_status_snapshot,
    resolve_update_targets,
    set_refresh_status,
    set_update_status,
    update_status_snapshot,
    run_module_update_job,
    run_refresh_job,
    requirements_changed_between,
    install_module_requirements,
    install_comfyui_requirements,
    install_requirements_for_modules,
)

# State operations (state/)
from .state import (
    load_state_file,
    save_state_file,
    set_comfyui_requirements_pending,
    set_module_requirements_pending,
    refresh_module_runtime_state,
)

# Tracking operations (tracking/)
from .tracking import (
    remember_module_state,
    apply_node_change_info,
    acknowledge_module_novelty,
    acknowledge_all_novelty,
    announce_tracked_module_updates,
)

# ComfyUI operations (comfyui/)
from .comfyui import (
    acknowledge_comfyui_novelty,
    track_comfyui_local_update,
    comfyui_status_template,
    resolve_cached_status,
    apply_cached_pending_fields,
    persist_comfyui_status,
    collect_comfyui_git_status,
    load_manager_github_stats,
    load_manager_index,
    resolve_manager_meta_for_module,
    manager_stats_last_update,
    infer_update_from_manager_stats,
)

# Bootstrap operations (bootstrap/)
from .bootstrap import (
    comfyui_requirements_path,
    bootstrap_module_remote_from_manager,
)

# Core utilities (core/)
from .core import (
    short_commit,
    normalize_repo_url,
    github_id,
    repo_name,
    pick_repo_url,
    parse_datetime,
    to_iso,
    now_iso,
    normalize_comfyui_mode,
    custom_nodes_roots,
    manager_custom_db_path,
    manager_github_stats_path,
    module_dir,
    comfyui_root,
    github_latest_release,
    custom_update_checked_flag,
    info_only_rejection_payload,
    set_custom_update_checked,
    normalize_log_mode,
    run_manifest_check,
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
    "set_comfyui_requirements_pending",
    "set_module_requirements_pending",
    "custom_nodes_roots",
    "manager_custom_db_path",
    "manager_github_stats_path",
    "module_dir",
    "comfyui_root",
    "github_latest_release",
    "module_needs_update_now",
    "count_custom_modules_need_update",
    "count_custom_modules_unknown_update",
    "comfyui_needs_update_now",
    "comfyui_requirements_path",
    "bootstrap_module_remote_from_manager",
    "module_root",
    "classify_by_source_path",
    "classify_by_relative_module",
    "fallback_annotation",
    "ensure_module_state_schema",
    "build_registry_snapshot",
    "compute_snapshot_signature",
    "WidgetSpec",
    "iter_component_api_routes",
    "iter_widget_specs",
]
