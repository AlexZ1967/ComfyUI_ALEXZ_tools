"""
Module: utils/module_browser/catalog/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Catalog and component registry operations.
"""

from .catalog import (
    build_catalog,
    build_group_catalog,
    build_group_modules,
    collect_nodes,
    filter_modules,
)
from .catalog_payload_ops import (
    build_group_payload,
    build_module_list_payload,
    build_module_nodes_payload,
)
from .component_registry import (
    ComponentEntry,
    ComponentRegistry,
    build_default_component_registry,
    build_registry_snapshot,
    compute_snapshot_signature,
)
from .component_registry_payload_ops import (
    collect_component_registry_payload,
)
from .api_manifest import (
    iter_component_api_routes,
)
from .widget_manifest import (
    WidgetSpec,
    iter_widget_specs,
)

__all__ = [
    "build_catalog",
    "build_group_catalog",
    "build_group_modules",
    "collect_nodes",
    "filter_modules",
    "build_group_payload",
    "build_module_list_payload",
    "build_module_nodes_payload",
    "ComponentEntry",
    "ComponentRegistry",
    "build_default_component_registry",
    "build_registry_snapshot",
    "compute_snapshot_signature",
    "collect_component_registry_payload",
    "iter_component_api_routes",
    "WidgetSpec",
    "iter_widget_specs",
]
