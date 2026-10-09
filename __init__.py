"""
Module: __init__.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    ComfyUI_ALEXZ_tools package entrypoint.

Purpose:
    Initializes extension logging, registers backend APIs, and exports node mappings.
"""

import logging
from pathlib import Path

_LOGGER = logging.getLogger("ALEXZ_tools")
_LOGGER.info("ALEXZ_tools loading...")

from .utils import module_node_browser_api as _module_node_browser_api  # noqa: F401
from .utils.module_updates.service import register_routes as _register_update_routes

if (_module_node_browser_api.PromptServer is not None and _module_node_browser_api.web is not None
        and getattr(_module_node_browser_api.PromptServer, "instance", None) is not None):
    _update_service = _register_update_routes(
        _module_node_browser_api.PromptServer,
        _module_node_browser_api.web,
        _module_node_browser_api._custom_nodes_roots,
        Path(__file__).resolve().parent,
        capture_module=_module_node_browser_api._capture_module_update_tracking,
        finalize_tracking=_module_node_browser_api._finalize_module_update_tracking,
    )
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
