"""Compatibility shim for legacy ``utils.module_browser.catalog_payload_ops`` imports."""

from .catalog import catalog_payload_ops as _impl
from .catalog.catalog_payload_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
