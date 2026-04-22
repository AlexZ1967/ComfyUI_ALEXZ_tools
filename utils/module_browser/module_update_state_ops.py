"""Compatibility shim for legacy ``utils.module_browser.module_update_state_ops`` imports."""

from .module import module_update_state_ops as _impl
from .module.module_update_state_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
