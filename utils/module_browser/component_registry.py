"""Compatibility shim for legacy ``utils.module_browser.component_registry`` imports."""

from .catalog import component_registry as _impl
from .catalog.component_registry import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
