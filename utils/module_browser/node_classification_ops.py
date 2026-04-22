"""Compatibility shim for legacy ``utils.module_browser.node_classification_ops`` imports."""

from .module import node_classification_ops as _impl
from .module.node_classification_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
