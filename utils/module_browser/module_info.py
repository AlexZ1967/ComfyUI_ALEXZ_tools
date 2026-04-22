"""Compatibility shim for legacy ``utils.module_browser.module_info`` imports."""

from .module import module_info as _impl
from .module.module_info import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
