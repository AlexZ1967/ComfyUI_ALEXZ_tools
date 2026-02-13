"""
Module: utils/module_browser/node_snapshot_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Node source/snapshot helper functions for Module Node Picker backend.

Purpose:
    Extracts deterministic node snapshot construction logic from API module
    to keep change-tracker behavior stable and testable in isolation.
"""

from __future__ import annotations

import inspect
import sys
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable


def node_source_file(node_cls: Any) -> str:
    """Resolve absolute source file path for a node class object."""
    source_file = ""
    try:
        source_file = inspect.getsourcefile(node_cls) or ""
    except Exception:
        source_file = ""
    if source_file:
        try:
            return str(Path(source_file).resolve())
        except Exception:
            return source_file

    module_name = getattr(node_cls, "__module__", "") or ""
    module_obj = sys.modules.get(module_name)
    module_file = getattr(module_obj, "__file__", "") if module_obj is not None else ""
    if not module_file:
        return ""
    try:
        return str(Path(module_file).resolve())
    except Exception:
        return module_file


def relative_to_custom_roots(path_text: str, *, custom_nodes_roots: Callable[[], list[Path]]) -> str:
    """Resolve path relative to known custom_nodes roots when possible."""
    if not path_text:
        return ""
    try:
        path_obj = Path(path_text).resolve()
    except Exception:
        return path_text
    for root in custom_nodes_roots():
        try:
            return str(path_obj.relative_to(root.resolve()))
        except Exception:
            continue
    return str(path_obj)


def file_digest(path_text: str) -> str:
    """Compute short SHA1 digest for file content used in node-change tracking."""
    if not path_text:
        return ""
    try:
        data = Path(path_text).read_bytes()
        return sha1(data).hexdigest()[:12]
    except Exception:
        return ""


def build_node_snapshots(
    *,
    class_map: dict[str, Any],
    classifier: Callable[[Any], tuple[str, str]],
    custom_nodes_roots: Callable[[], list[Path]],
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Build stable per-node file snapshots used to detect node additions/changes."""
    snapshots: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    digest_cache: dict[str, str] = {}
    for node_name, node_cls in class_map.items():
        group, module_bucket = classifier(node_cls)
        source_file = node_source_file(node_cls)
        digest = digest_cache.get(source_file)
        if digest is None:
            digest = file_digest(source_file)
            digest_cache[source_file] = digest
        snapshots[group][module_bucket][node_name] = {
            "sig": f"{getattr(node_cls, '__name__', '')}:{digest}",
            "source": relative_to_custom_roots(source_file, custom_nodes_roots=custom_nodes_roots),
        }

    out: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for group, modules in snapshots.items():
        out[group] = {}
        for module_name, nodes in modules.items():
            out[group][module_name] = dict(sorted(nodes.items(), key=lambda kv: kv[0].lower()))
    return out
