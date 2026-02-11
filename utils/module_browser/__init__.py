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

__all__ = [
    "ComponentEntry",
    "ComponentRegistry",
    "COMPONENT_REGISTRY_SCHEMA_NAME",
    "COMPONENT_REGISTRY_SCHEMA_VERSION",
    "MODULE_STATE_SCHEMA_VERSION",
    "build_default_component_registry",
    "build_component_health_report",
    "run_manifest_check",
    "ensure_module_state_schema",
    "build_registry_snapshot",
    "compute_snapshot_signature",
    "WidgetSpec",
    "iter_component_api_routes",
    "iter_widget_specs",
]
