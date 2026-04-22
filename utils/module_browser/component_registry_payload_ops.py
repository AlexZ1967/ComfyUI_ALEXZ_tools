"""Compatibility shim for legacy ``utils.module_browser.component_registry_payload_ops`` imports."""

from .catalog import component_registry_payload_ops as _impl
from .catalog.component_registry_payload_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
