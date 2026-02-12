"""
Module: utils/module_browser/__init__.py
Author: AlexZ1967
Last updated: 2026-02-12

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
    "ensure_module_state_schema",
    "build_registry_snapshot",
    "compute_snapshot_signature",
    "WidgetSpec",
    "iter_component_api_routes",
    "iter_widget_specs",
]
