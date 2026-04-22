"""Compatibility shim for legacy ``utils.module_browser.module_identity`` imports."""

from .module import module_identity as _impl
from .module.module_identity import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
