"""
Module: utils/module_browser/node_classification_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Node classification and fallback-annotation helper functions.

Purpose:
    Extracts deterministic node grouping/annotation logic from API facade while
    preserving category resolution behavior for Module Node Picker catalogs.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any, Callable


def module_root(node_cls: Any) -> str:
    """Resolve module root directory for a file path inside extension package."""
    module_name = getattr(node_cls, "__module__", "") or ""
    if not module_name:
        return "unknown"
    return module_name.split(".", 1)[0]


def classify_by_source_path(
    node_cls: Any,
    *,
    node_source_file_fn: Callable[[Any], str],
    custom_nodes_roots_fn: Callable[[], list[Path]],
    canonical_custom_module_name_fn: Callable[[str], str],
    module_root_fn: Callable[[Any], str],
) -> tuple[str, str] | None:
    """Classify node into core/extras/api/custom groups from source path."""
    source = node_source_file_fn(node_cls)
    if not source:
        return None

    try:
        src_path = Path(source).resolve()
    except Exception:
        return None

    for root in custom_nodes_roots_fn():
        try:
            rel = src_path.relative_to(root.resolve())
        except Exception:
            continue
        if rel.parts:
            return ("custom", canonical_custom_module_name_fn(rel.parts[0]))

    parts_l = [p.lower() for p in src_path.parts]
    if "comfy_extras" in parts_l:
        idx = parts_l.index("comfy_extras")
        module_name = src_path.parts[idx + 1] if (idx + 1) < len(src_path.parts) else module_root_fn(node_cls)
        return ("core_extras", module_name)
    if "comfy_api_nodes" in parts_l:
        idx = parts_l.index("comfy_api_nodes")
        module_name = src_path.parts[idx + 1] if (idx + 1) < len(src_path.parts) else module_root_fn(node_cls)
        return ("api", module_name)
    return None


def classify_by_relative_module(
    node_cls: Any,
    *,
    canonical_custom_module_name_fn: Callable[[str], str],
    classify_by_source_path_fn: Callable[[Any], tuple[str, str] | None],
    module_root_fn: Callable[[Any], str],
) -> tuple[str, str]:
    """Classify node group and module name using relative-module metadata."""
    rel = getattr(node_cls, "RELATIVE_PYTHON_MODULE", None)
    if isinstance(rel, str) and rel:
        parts = [p for p in rel.split(".") if p]
        if len(parts) >= 2:
            root, module_name = parts[0], parts[1]
        elif len(parts) == 1:
            root, module_name = parts[0], parts[0]
        else:
            root, module_name = "", ""

        if root == "custom_nodes":
            return ("custom", canonical_custom_module_name_fn(module_name))
        if root == "comfy_extras":
            return ("core_extras", module_name)
        if root == "comfy_api_nodes":
            return ("api", module_name)

    source_hit = classify_by_source_path_fn(node_cls)
    if source_hit is not None:
        return source_hit

    module_name = getattr(node_cls, "__module__", "") or ""
    module_l = module_name.lower()
    if module_l.startswith("comfy_extras."):
        parts = module_name.split(".")
        return ("core_extras", parts[1] if len(parts) > 1 else module_name)
    if module_l.startswith("comfy_api_nodes."):
        parts = module_name.split(".")
        return ("api", parts[1] if len(parts) > 1 else module_name)
    return ("core", module_root_fn(node_cls))


def fallback_annotation(node_cls: Any) -> str:
    """Build fallback node annotation from class metadata."""
    category = getattr(node_cls, "CATEGORY", "") or "unknown"
    return_names = getattr(node_cls, "RETURN_NAMES", None)
    if not return_names:
        return_types = getattr(node_cls, "RETURN_TYPES", ())
        return_names = return_types

    if return_names is None:
        output_items = []
    elif isinstance(return_names, (str, bytes)):
        output_items = [str(return_names)]
    else:
        try:
            output_items = [str(x) for x in islice(iter(return_names), 3)]
        except Exception:
            output_items = [str(return_names)]

    outputs = ", ".join(output_items) or "unknown"
    return f"Категория: {category}. Выходы: {outputs}."

