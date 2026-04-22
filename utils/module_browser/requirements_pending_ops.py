"""Compatibility shim for legacy ``utils.module_browser.requirements_pending_ops`` imports."""

from .state import requirements_pending_ops as _impl
from .state.requirements_pending_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
