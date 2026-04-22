"""Compatibility shim for legacy ``utils.module_browser.comfyui_git_status_ops`` imports."""

from .comfyui import comfyui_git_status_ops as _impl
from .comfyui.comfyui_git_status_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
