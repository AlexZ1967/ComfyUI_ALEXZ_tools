"""Compatibility shim for legacy ``utils.module_browser.widget_mode_ops`` imports."""

from .core import widget_mode_ops as _impl
from .core.widget_mode_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
