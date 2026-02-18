"""
Module: utils/module_browser/catalog_payload_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Payload builder helpers for catalog-related Module Node Picker API routes.

Purpose:
    Keeps route handlers lightweight by moving deterministic response assembly
    into focused pure helpers during Phase 3 backend decomposition.
"""

from __future__ import annotations

from typing import Any, Callable


def build_group_payload(
    *,
    group_order: tuple[tuple[str, str], ...],
    grouped_nodes: dict[str, list[dict[str, Any]]],
    modules_by_group: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build stable ordered group payload for `/node_catalog` response."""
    groups: list[dict[str, Any]] = []
    for group_id, group_title in group_order:
        nodes = grouped_nodes.get(group_id, [])
        modules = modules_by_group.get(group_id, [])
        groups.append(
            {
                "id": group_id,
                "title": group_title,
                "count": len(nodes),
                "nodes": nodes,
                "module_count": len(modules),
                "modules": modules,
            }
        )
    return groups


def build_module_list_payload(
    *,
    catalog: dict[str, list[dict[str, Any]]],
    query: str,
) -> dict[str, Any]:
    """Build module list payload for `/module_list` response."""
    query_l = str(query or "").strip().lower()
    modules: list[dict[str, Any]] = []
    for module_name, nodes in catalog.items():
        if query_l and query_l not in str(module_name).lower():
            continue
        modules.append({"module": module_name, "count": len(nodes)})
    return {"query": query_l, "modules": modules}


def build_module_nodes_payload(
    *,
    catalog: dict[str, list[dict[str, Any]]],
    query: str,
    filter_modules_fn: Callable[[str, list[str]], list[str]],
) -> dict[str, Any]:
    """Build module-nodes payload for `/module_nodes` response."""
    query_raw = str(query or "").strip()
    modules = list(catalog.keys())
    selected_modules = filter_modules_fn(query_raw, modules)
    results: list[dict[str, Any]] = []
    for module_name in selected_modules:
        nodes = catalog.get(module_name, [])
        results.append(
            {
                "module": module_name,
                "count": len(nodes),
                "nodes": nodes,
            }
        )
    return {
        "query": query_raw,
        "module_count": len(results),
        "results": results,
        "hint": "Введите имя python-модуля (например: ComfyUI_ALEXZ_tools).",
    }

