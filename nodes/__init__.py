"""
Module: nodes/__init__.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Node registry for ComfyUI_ALEXZ_tools.

Purpose:
    Loads node classes from central registry manifest, builds ComfyUI mapping
    dictionaries, and logs node load status.
"""

import importlib
import logging
import traceback

from .node_registry import NODE_UI_METADATA, iter_node_specs

_LOGGER = logging.getLogger("ALEXZ_tools")

_NODE_SPECS = list(iter_node_specs())
_NODE_UI_METADATA = NODE_UI_METADATA

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
LOAD_RESULTS = {"ok": [], "fail": []}
_LOG_LINES = []


def _load_node(name: str, display: str, module: str, attr: str):
    """Load one node class and store it in ComfyUI mappings."""
    try:
        mod = importlib.import_module(module, __name__)
        cls = getattr(mod, attr)
        NODE_CLASS_MAPPINGS[name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[name] = display
        LOAD_RESULTS["ok"].append(name)
        _LOG_LINES.append(f"✅ {display} loaded")
    except Exception as exc:  # pragma: no cover - diagnostic
        LOAD_RESULTS["fail"].append({"name": name, "reason": str(exc)})
        _LOG_LINES.append(f"❌ {display} failed: {exc}")
        _LOGGER.error("Failed to load node %s: %s\n%s", name, exc, traceback.format_exc())


def _apply_node_ui_metadata() -> None:
    """Attach optional UI metadata used by newer ComfyUI node cards."""
    for node_name, node_cls in NODE_CLASS_MAPPINGS.items():
        meta = _NODE_UI_METADATA.get(node_name)
        if not meta:
            continue
        if not getattr(node_cls, "DESCRIPTION", ""):
            node_cls.DESCRIPTION = meta["description"]
        if not hasattr(node_cls, "OUTPUT_TOOLTIPS"):
            node_cls.OUTPUT_TOOLTIPS = list(meta["output_tooltips"])
        if not hasattr(node_cls, "SEARCH_ALIASES"):
            node_cls.SEARCH_ALIASES = list(meta["search_aliases"])


for _name, _disp, _mod, _attr in _NODE_SPECS:
    _load_node(_name, _disp, _mod, _attr)

_apply_node_ui_metadata()

for line in _LOG_LINES:
    _LOGGER.info(line)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "LOAD_RESULTS"]
