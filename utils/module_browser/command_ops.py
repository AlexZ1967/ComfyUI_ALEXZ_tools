"""Compatibility shim for legacy ``utils.module_browser.command_ops`` imports."""

from .git import command_ops as _impl
from .git.command_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
