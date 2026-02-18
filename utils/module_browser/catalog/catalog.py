"""
Module: utils/module_browser/catalog.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Pure catalog-building helpers for Module Node Picker backend.

Purpose:
    Separates node/module catalog assembly from API handlers so backend logic
    stays testable and easier to evolve while preserving existing API payloads.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


def collect_nodes(
    *,
    class_map: dict[str, Any],
    display_map: dict[str, str],
    annotation_resolver: Callable[[str, Any], str],
    classifier: Callable[[Any], tuple[str, str]],
) -> list[dict[str, Any]]:
    """Collect normalized node entries from ComfyUI node mappings."""
    items: list[dict[str, Any]] = []
    for node_name, node_cls in class_map.items():
        display_name = display_map.get(node_name, node_name)
        group, module_bucket = classifier(node_cls)
        items.append(
            {
                "node_name": node_name,
                "display_name": display_name,
                "module": module_bucket,
                "group": group,
                "category": getattr(node_cls, "CATEGORY", "") or "",
                "annotation": annotation_resolver(node_name, node_cls),
            }
        )
    return items


def build_catalog(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build module->nodes catalog sorted by module and display name."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        module_name = str(item.get("module") or "unknown")
        grouped[module_name].append(item)
    for module_name in grouped:
        grouped[module_name].sort(key=lambda item: str(item.get("display_name") or "").lower())
    return dict(sorted(grouped.items(), key=lambda kv: kv[0].lower()))


def build_group_catalog(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build group->nodes catalog sorted by display name."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        group_name = str(item.get("group") or "unknown")
        grouped[group_name].append(item)
    for group_name in grouped:
        grouped[group_name].sort(key=lambda item: str(item.get("display_name") or "").lower())
    return grouped


def build_group_modules(
    *,
    grouped_nodes: dict[str, list[dict[str, Any]]],
    discover_custom_modules: Callable[[], list[str]],
    cached_module_flags: Callable[[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build grouped module summaries with count and cached marker flags."""
    module_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for group_name, nodes in grouped_nodes.items():
        for node in nodes:
            module_name = str(node.get("module") or "unknown")
            module_counts[group_name][module_name] += 1

    for module_name in discover_custom_modules():
        module_counts["custom"].setdefault(module_name, 0)

    out: dict[str, list[dict[str, Any]]] = {}
    for group_name, counts in module_counts.items():
        out[group_name] = [
            {
                "module": mod,
                "count": int(cnt),
                **cached_module_flags(group_name, mod),
            }
            for mod, cnt in sorted(counts.items(), key=lambda kv: kv[0].lower())
        ]
    return out


def filter_modules(query: str, module_names: list[str]) -> list[str]:
    """Filter module list by case-insensitive query with exact-match priority."""
    if not query:
        return module_names
    q = query.lower()
    exact = [name for name in module_names if name.lower() == q]
    if exact:
        return exact
    return [name for name in module_names if q in name.lower()]
