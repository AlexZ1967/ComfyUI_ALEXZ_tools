"""
Module: utils/module_browser/api_manifest.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Canonical API route manifest for Module Node Picker backend endpoints.

Purpose:
    Keeps route paths in one place so route add/remove operations remain
    deterministic and easier to maintain during backend decomposition.
"""

from __future__ import annotations


ROUTE_MODULE_REFRESH = "/alexz_tools/module_refresh"
ROUTE_MODULE_REFRESH_STATUS = "/alexz_tools/module_refresh_status"
ROUTE_MODULE_ACKNOWLEDGE_ALL = "/alexz_tools/module_acknowledge_all"
ROUTE_MODULE_UPDATE = "/alexz_tools/module_update"
ROUTE_MODULE_UPDATE_STATUS = "/alexz_tools/module_update_status"
ROUTE_MODULE_INSTALL_REQUIREMENTS = "/alexz_tools/module_install_requirements"
ROUTE_COMFYUI_INSTALL_REQUIREMENTS = "/alexz_tools/comfyui_install_requirements"
ROUTE_COMPONENT_REGISTRY = "/alexz_tools/component_registry"
ROUTE_NODE_CATALOG = "/alexz_tools/node_catalog"
ROUTE_MODULE_INFO = "/alexz_tools/module_info"
ROUTE_COMFYUI_INFO = "/alexz_tools/comfyui_info"
ROUTE_MODULE_LIST = "/alexz_tools/module_list"
ROUTE_MODULE_NODES = "/alexz_tools/module_nodes"

# Component-facing API routes that should appear in component registry snapshots.
COMPONENT_API_ROUTES: tuple[str, ...] = (
    ROUTE_NODE_CATALOG,
    ROUTE_MODULE_INFO,
    ROUTE_MODULE_LIST,
    ROUTE_MODULE_NODES,
    ROUTE_MODULE_REFRESH,
    ROUTE_MODULE_REFRESH_STATUS,
    ROUTE_COMFYUI_INFO,
    ROUTE_COMPONENT_REGISTRY,
)


def iter_component_api_routes():
    """Yield API routes included in component-registry snapshots."""
    for route in COMPONENT_API_ROUTES:
        yield route

