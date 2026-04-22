"""Compatibility shim for legacy ``utils.module_browser.update_job_ops`` imports."""

from .jobs import update_job_ops as _impl
from .jobs.update_job_ops import *  # noqa: F401,F403

__all__ = getattr(_impl, "__all__", [])
