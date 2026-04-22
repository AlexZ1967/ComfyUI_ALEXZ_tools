"""Compatibility shim for legacy ``utils.module_browser.path_ops`` imports."""

from .core import path_ops as _impl
from .core.path_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
