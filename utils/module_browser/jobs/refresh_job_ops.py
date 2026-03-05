"""
Module: utils/module_browser/refresh_job_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Refresh-job execution helpers for Module Node Picker backend.

Purpose:
    Extracts synchronous refresh worker logic from API thread wrapper while
    preserving status and logging behavior.
"""

from __future__ import annotations

from typing import Any, Callable


def run_refresh_job(
    *,
    sync_upstreams: bool,
    get_update_console_log_mode: Callable[[], str],
    refresh_console_log: Callable[[str, str], None],
    refresh_module_runtime_state: Callable[[bool], dict[str, Any]],
    set_refresh_status: Callable[..., None],
) -> dict[str, Any]:
    """Execute refresh job synchronously and update status through callbacks."""
    refresh_console_log(
        "job started (sync_upstreams={sync}, log_mode={mode})".format(
            sync="on" if sync_upstreams else "off",
            mode=get_update_console_log_mode(),
        ),
        "summary",
    )
    result = refresh_module_runtime_state(sync_upstreams)
    set_refresh_status(
        running=False,
        phase="done",
        message="done",
        module="",
        refreshed_at=result.get("refreshed_at", ""),
        modules_need_update=max(0, int(result.get("modules_need_update", 0))),
        modules_unknown_update=max(0, int(result.get("modules_unknown_update", 0))),
        unknown_update_modules=list(result.get("unknown_update_modules") or []),
    )
    refresh_console_log(
        "job finished: modules_need_update={need}, modules_unknown_update={unknown}".format(
            need=max(0, int(result.get("modules_need_update", 0))),
            unknown=max(0, int(result.get("modules_unknown_update", 0))),
        ),
        "summary",
    )
    return result
