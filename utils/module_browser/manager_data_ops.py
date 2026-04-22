"""Compatibility shim for legacy ``utils.module_browser.manager_data_ops`` imports."""

from .comfyui import manager_data_ops as _impl
from .comfyui.manager_data_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
