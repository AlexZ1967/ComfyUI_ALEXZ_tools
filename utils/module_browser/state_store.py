"""Compatibility shim for legacy ``utils.module_browser.state_store`` imports."""

from .state import state_store as _impl
from .state.state_store import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
