"""Compatibility shim for legacy ``utils.module_browser.pull_ops`` imports."""

from .git import pull_ops as _impl
from .git.pull_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
