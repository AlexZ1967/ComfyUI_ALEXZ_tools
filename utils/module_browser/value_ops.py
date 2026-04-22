"""Compatibility shim for legacy ``utils.module_browser.value_ops`` imports."""

from .core import value_ops as _impl
from .core.value_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
