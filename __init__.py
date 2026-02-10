"""Package entrypoint for ComfyUI_ALEXZ_tools.

Initializes package-level logging, registers Module Nodes widget backend, and
exports ComfyUI node mappings.
"""

import logging

_LOGGER = logging.getLogger("ALEXZ_tools")
_LOGGER.info("ALEXZ_tools loading...")

from .utils import module_node_browser_api as _module_node_browser_api  # noqa: F401
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
