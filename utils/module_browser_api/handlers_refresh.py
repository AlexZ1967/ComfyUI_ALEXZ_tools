"""
Module: utils/module_browser_api/handlers_refresh.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Refresh-job handler orchestration helpers for module browser API.

Purpose:
    Isolate refresh start/run lifecycle from route glue in
    `utils/module_node_browser_api.py`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from .state import ModuleBrowserApiState


def start_refresh_job(
    *,
    state: ModuleBrowserApiState,
    sync_upstreams: bool,
    now_iso: Callable[[], str],
    get_update_console_log_mode: Callable[[], str],
    refresh_console_log: Callable[[str, str], None],
    refresh_module_runtime_state: Callable[[bool], dict[str, Any]],
    set_refresh_status: Callable[..., None],
    refresh_status_snapshot: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Start background refresh job, preserving existing thread/lock semantics."""
    with state.refresh_lock:
        thread = state.refresh_thread
        if thread is not None and thread.is_alive():
            return {"status": "running", "refresh": dict(state.refresh_status)}
        state.refresh_console_log_last = ""
        state.refresh_status.update(
            {
                "running": True,
                "phase": "starting",
                "current": 0,
                "total": 0,
                "remaining": 0,
                "modules_need_update": 0,
                "modules_unknown_update": 0,
                "unknown_update_modules": [],
                "module": "",
                "message": "starting",
                "error": "",
                "sync_upstreams": bool(sync_upstreams),
                "started_at": now_iso(),
                "updated_at": now_iso(),
                "refreshed_at": "",
            }
        )

    def _runner() -> None:
        try:
            mode = get_update_console_log_mode()
            refresh_console_log(
                f"job started (sync_upstreams={'on' if sync_upstreams else 'off'}, log_mode={mode})",
                "summary",
            )
            result = refresh_module_runtime_state(bool(sync_upstreams))
            refresh_console_log(
                "job finished (status={status}, need_update={need}, unknown={unknown})".format(
                    status=str((result or {}).get("status") or "ok"),
                    need=int((result or {}).get("modules_need_update") or 0),
                    unknown=int((result or {}).get("modules_unknown_update") or 0),
                ),
                "summary",
            )
            set_refresh_status(
                running=False,
                phase="done",
                message="done",
                error="",
                module="",
                refreshed_at=str((result or {}).get("refreshed_at") or ""),
                modules_need_update=max(0, int((result or {}).get("modules_need_update") or 0)),
                modules_unknown_update=max(0, int((result or {}).get("modules_unknown_update") or 0)),
                unknown_update_modules=list((result or {}).get("unknown_update_modules") or []),
            )
        except Exception as exc:
            refresh_console_log(f"job error: {exc}", "summary")
            set_refresh_status(running=False, phase="error", message="error", error=str(exc), module="")
        finally:
            with state.refresh_lock:
                state.refresh_thread = None

    thread = threading.Thread(target=_runner, name="alexz-module-refresh", daemon=True)
    with state.refresh_lock:
        state.refresh_thread = thread
    thread.start()
    return {"status": "started", "refresh": refresh_status_snapshot()}
