"""Compatibility shim for legacy ``utils.module_browser.widget_manifest`` imports."""

from .catalog import widget_manifest as _impl
from .catalog.widget_manifest import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
