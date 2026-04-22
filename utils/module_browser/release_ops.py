"""Compatibility shim for legacy ``utils.module_browser.release_ops`` imports."""

from .core import release_ops as _impl
from .core.release_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
