"""
Module: utils/module_browser/jobs/__init__.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Job execution and orchestration.
"""

from .jobs import (
    emit_refresh_progress,
    format_update_status_line,
    refresh_status_snapshot,
    resolve_update_targets,
    set_refresh_status,
    set_update_status,
    update_status_snapshot,
)
from .update_job_ops import (
    run_module_update_job,
)
from .refresh_job_ops import (
    run_refresh_job,
)
from .update_ops import (
    install_comfyui_requirements,
    install_requirements_for_modules,
    install_module_requirements,
    requirements_changed_between,
)

__all__ = [
    "emit_refresh_progress",
    "format_update_status_line",
    "refresh_status_snapshot",
    "resolve_update_targets",
    "set_refresh_status",
    "set_update_status",
    "update_status_snapshot",
    "run_module_update_job",
    "run_refresh_job",
    "requirements_changed_between",
    "install_module_requirements",
    "install_comfyui_requirements",
    "install_requirements_for_modules",
]
