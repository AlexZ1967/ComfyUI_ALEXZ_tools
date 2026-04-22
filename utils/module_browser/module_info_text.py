"""Compatibility shim for legacy ``utils.module_browser.module_info_text`` imports."""

from .module import module_info_text as _impl
from .module.module_info_text import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
