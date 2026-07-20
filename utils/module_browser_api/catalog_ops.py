"""
Module: utils/module_browser_api/catalog_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Catalog/build helpers for module browser API facade.

Purpose:
    Keeps node collection and catalog payload orchestration out of
    `utils/module_node_browser_api.py` so the facade remains a thin wiring
    layer while preserving existing contracts and patch points.
"""

from __future__ import annotations

from typing import Any, Callable


def collect_nodes(
    *,
    node_mappings: Callable[[], tuple[dict[str, Any], dict[str, str]]],
    collect_nodes_impl: Callable[..., list[dict[str, Any]]],
    annotations: dict[str, str],
    fallback_annotation: Callable[[Any], str],
    classify_by_relative_module: Callable[[Any], tuple[str, str]],
) -> list[dict[str, Any]]:
    """Collect node definitions from registered ComfyUI mappings."""
    class_map, display_map = node_mappings()
    return collect_nodes_impl(
        class_map=class_map,
        display_map=display_map,
        annotation_resolver=lambda node_name, node_cls: annotations.get(node_name) or fallback_annotation(node_cls),
        classifier=classify_by_relative_module,
    )


def build_catalog(
    *,
    collect_nodes_fn: Callable[[], list[dict[str, Any]]],
    build_catalog_impl: Callable[[list[dict[str, Any]]], dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """Build cached module-to-node catalog from discovered nodes."""
    return build_catalog_impl(collect_nodes_fn())


def build_group_catalog(
    *,
    collect_nodes_fn: Callable[[], list[dict[str, Any]]],
    build_group_catalog_impl: Callable[[list[dict[str, Any]]], dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """Build grouped node catalog for one category."""
    return build_group_catalog_impl(collect_nodes_fn())


def build_group_modules(
    grouped_nodes: dict[str, list[dict[str, Any]]],
    *,
    build_group_modules_impl: Callable[..., dict[str, list[dict[str, Any]]]],
    discover_custom_modules: Callable[[], list[str]],
    cached_module_flags: Callable[[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build grouped module summaries for one category."""
    return build_group_modules_impl(
        grouped_nodes=grouped_nodes,
        discover_custom_modules=discover_custom_modules,
        cached_module_flags=cached_module_flags,
    )


def filter_modules(
    query: str,
    module_names: list[str],
    *,
    filter_modules_impl: Callable[[str, list[str]], list[str]],
) -> list[str]:
    """Filter module list by case-insensitive text query over module names."""
    return filter_modules_impl(query, module_names)


def build_group_payload(
    grouped_nodes: dict[str, list[dict[str, Any]]],
    modules_by_group: dict[str, list[dict[str, Any]]],
    *,
    build_group_payload_impl: Callable[..., list[dict[str, Any]]],
    group_order: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Build ordered group payload for node-catalog API route."""
    return build_group_payload_impl(
        group_order=group_order,
        grouped_nodes=grouped_nodes,
        modules_by_group=modules_by_group,
    )


def build_module_list_payload(
    catalog: dict[str, list[dict[str, Any]]],
    query: str,
    *,
    build_module_list_payload_impl: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Build module-list payload for module-list API route."""
    return build_module_list_payload_impl(catalog=catalog, query=query)


def build_module_nodes_payload(
    catalog: dict[str, list[dict[str, Any]]],
    query: str,
    *,
    build_module_nodes_payload_impl: Callable[..., dict[str, Any]],
    filter_modules_fn: Callable[[str, list[str]], list[str]],
) -> dict[str, Any]:
    """Build module-nodes payload for module-nodes API route."""
    return build_module_nodes_payload_impl(
        catalog=catalog,
        query=query,
        filter_modules_fn=filter_modules_fn,
    )
