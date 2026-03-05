"""
Module: utils/module_browser_api/handlers_update.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Update-job handler orchestration helpers for module browser API.

Purpose:
    Isolate update start/run lifecycle from route glue in
    `utils/module_node_browser_api.py`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from .state import ModuleBrowserApiState


def start_module_update_job(
    *,
    state: ModuleBrowserApiState,
    scope_norm: str,
    module_name: str,
    normalized_log_mode: str,
    now_iso: Callable[[], str],
    comfyui_root_exists: bool,
    single_module_exists: bool,
    refresh_running: bool,
    update_status_snapshot: Callable[[], dict[str, Any]],
    update_console_log: Callable[[str, str], None],
    run_module_update_job: Callable[..., None],
    resolve_update_targets: Callable[[str, str], list[str]],
    pull_comfyui: Callable[..., dict[str, Any]],
    pull_custom_module: Callable[..., dict[str, Any]],
    refresh_module_runtime_state: Callable[[], dict[str, Any]],
    set_update_status: Callable[..., None],
) -> dict[str, Any]:
    """Start background update job, preserving existing thread/lock semantics."""
    if scope_norm not in {"single", "all", "comfyui"}:
        return {"status": "error", "error": "scope must be 'single', 'all' or 'comfyui'"}
    if scope_norm == "single" and not single_module_exists:
        return {"status": "error", "error": "module not found"}
    if scope_norm == "comfyui" and not comfyui_root_exists:
        return {"status": "error", "error": "ComfyUI root not found"}
    if refresh_running:
        return {"status": "error", "error": "module refresh is running"}

    with state.update_lock:
        thread = state.update_thread
        if thread is not None and thread.is_alive():
            return {"status": "running", "update": dict(state.update_status)}
        state.update_status.update(
            {
                "running": True,
                "phase": "starting",
                "scope": scope_norm,
                "current": 0,
                "total": 0,
                "remaining": 0,
                "module": "",
                "message": "starting",
                "error": "",
                "updated": 0,
                "up_to_date": 0,
                "failed": 0,
                "requirements_changed": False,
                "requirements_modules": [],
                "results": [],
                "log_mode": normalized_log_mode,
                "started_at": now_iso(),
                "updated_at": now_iso(),
                "finished_at": "",
            }
        )

    def _runner() -> None:
        try:
            update_console_log(
                "job started (scope={scope}, module={module}, log_mode={mode})".format(
                    scope=scope_norm,
                    module=module_name or "-",
                    mode=normalized_log_mode,
                ),
                "summary",
            )
            run_module_update_job(
                scope_norm=scope_norm,
                module_name=module_name,
                normalized_log_mode=normalized_log_mode,
                update_console_log=update_console_log,
                set_update_status=set_update_status,
                pull_comfyui=pull_comfyui,
                pull_custom_module=pull_custom_module,
                resolve_update_targets=resolve_update_targets,
                refresh_module_runtime_state=refresh_module_runtime_state,
                now_iso=now_iso,
            )
            update_console_log(
                "job finished (scope={scope})".format(scope=scope_norm),
                "summary",
            )
        except Exception as exc:
            update_console_log(f"job error: {exc}", "summary")
            set_update_status(
                running=False,
                phase="error",
                message="error",
                error=str(exc),
                module="",
                finished_at=now_iso(),
            )
        finally:
            with state.update_lock:
                state.update_thread = None

    thread = threading.Thread(target=_runner, name="alexz-module-update", daemon=True)
    with state.update_lock:
        state.update_thread = thread
    thread.start()
    return {"status": "started", "update": update_status_snapshot()}
