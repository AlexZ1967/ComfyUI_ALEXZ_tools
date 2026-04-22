"""Compatibility shim for legacy ``utils.module_browser.git_helpers`` imports."""

from .git import git_helpers as _impl
from .git.git_helpers import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
