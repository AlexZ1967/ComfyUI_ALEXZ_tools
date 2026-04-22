"""Compatibility shim for legacy ``utils.module_browser.repo_bootstrap_ops`` imports."""

from .bootstrap import repo_bootstrap_ops as _impl
from .bootstrap.repo_bootstrap_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
