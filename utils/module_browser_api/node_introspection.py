"""
Module: utils/module_browser_api/node_introspection.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Node-introspection helpers for module browser API facade.

Purpose:
    Isolate NODE_CLASS_MAPPINGS discovery and snapshot assembly from
    `utils/module_node_browser_api.py` during incremental refactoring.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ..module_browser.module.node_snapshot_ops import (
    build_node_snapshots as mb_build_node_snapshots,
)


def node_mappings(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS from ComfyUI."""
    comfy_nodes = import_module("nodes")
    class_map = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    display_map = getattr(comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
    return class_map, display_map


def build_node_snapshots(
    *,
    classifier: Callable[[Any], tuple[str, str]],
    custom_nodes_roots: Callable[[], list[Any]],
    node_mappings_fn: Callable[[], tuple[dict[str, Any], dict[str, str]]] = node_mappings,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Build stable per-node file snapshots grouped by node group/module."""
    class_map, _ = node_mappings_fn()
    return mb_build_node_snapshots(
        class_map=class_map,
        classifier=classifier,
        custom_nodes_roots=custom_nodes_roots,
    )
