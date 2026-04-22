"""Compatibility shim for legacy ``utils.module_browser.api_manifest`` imports."""

from .catalog import api_manifest as _impl
from .catalog.api_manifest import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
