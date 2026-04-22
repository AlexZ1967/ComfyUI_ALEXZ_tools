"""Compatibility shim for legacy ``utils.module_browser.update_ops`` imports."""

from .jobs import update_ops as _impl
from .jobs.update_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
