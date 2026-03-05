"""
Module: utils/module_browser_api/state.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Runtime state container for module-node-browser backend.

Purpose:
    Centralizes refresh/update locks, job status snapshots, and related
    short-lived runtime flags so orchestration code does not own globals.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


def _refresh_status_template() -> dict[str, Any]:
    """Return default refresh-job status payload."""
    return {
        "running": False,
        "phase": "idle",
        "current": 0,
        "total": 0,
        "remaining": 0,
        "modules_need_update": 0,
        "modules_unknown_update": 0,
        "unknown_update_modules": [],
        "module": "",
        "message": "",
        "error": "",
        "sync_upstreams": False,
        "started_at": "",
        "updated_at": "",
        "refreshed_at": "",
    }


def _update_status_template() -> dict[str, Any]:
    """Return default update-job status payload."""
    return {
        "running": False,
        "phase": "idle",
        "scope": "",
        "current": 0,
        "total": 0,
        "remaining": 0,
        "module": "",
        "message": "",
        "error": "",
        "updated": 0,
        "up_to_date": 0,
        "failed": 0,
        "requirements_changed": False,
        "requirements_modules": [],
        "results": [],
        "started_at": "",
        "updated_at": "",
        "finished_at": "",
    }


@dataclass
class ModuleBrowserApiState:
    """Mutable runtime state used by module-node-browser backend orchestration."""

    lazy_refresh_done: bool = False
    runtime_warmup_lock: threading.Lock = field(default_factory=threading.Lock)
    runtime_warmup_thread: threading.Thread | None = None

    refresh_lock: threading.Lock = field(default_factory=threading.Lock)
    refresh_thread: threading.Thread | None = None
    refresh_log_last: str = ""
    refresh_console_log_last: str = ""
    refresh_status: dict[str, Any] = field(default_factory=_refresh_status_template)

    update_lock: threading.Lock = field(default_factory=threading.Lock)
    update_thread: threading.Thread | None = None
    update_log_last: str = ""
    update_console_log_mode: str = "summary"
    update_status: dict[str, Any] = field(default_factory=_update_status_template)


_STATE: ModuleBrowserApiState | None = None
_STATE_LOCK = threading.Lock()


def get_state() -> ModuleBrowserApiState:
    """Return singleton runtime state container."""
    global _STATE
    if _STATE is not None:
        return _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = ModuleBrowserApiState()
    return _STATE

