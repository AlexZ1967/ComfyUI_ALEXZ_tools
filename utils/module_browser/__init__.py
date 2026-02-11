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
)
from .api_manifest import iter_component_api_routes
from .widget_manifest import WidgetSpec, iter_widget_specs
from .contracts import (
    MODULE_STATE_SCHEMA_VERSION,
    ensure_module_state_schema,
)

__all__ = [
    "ComponentEntry",
    "ComponentRegistry",
    "MODULE_STATE_SCHEMA_VERSION",
    "build_default_component_registry",
    "ensure_module_state_schema",
    "WidgetSpec",
    "iter_component_api_routes",
    "iter_widget_specs",
]
