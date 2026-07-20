"""
Module: utils/module_browser_api/logging_ops.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Console-log helpers for module-node-browser refresh/update jobs.

Purpose:
    Provides side-effect logging utilities isolated from API route glue.
"""

from __future__ import annotations

from typing import Callable

from .state import ModuleBrowserApiState


def set_update_console_log_mode(
    state: ModuleBrowserApiState,
    mode: str | None,
    normalize_log_mode: Callable[[str | None], str],
) -> str:
    """Store normalized console log mode in shared update state."""
    normalized = normalize_log_mode(mode)
    with state.update_lock:
        state.update_console_log_mode = normalized
    return normalized


def get_update_console_log_mode(state: ModuleBrowserApiState) -> str:
    """Read active update console log mode from shared state."""
    with state.update_lock:
        return state.update_console_log_mode


def update_console_log(
    state: ModuleBrowserApiState,
    message: str,
    level: str,
    normalize_log_mode: Callable[[str | None], str],
) -> None:
    """Print update-progress line to console according to selected mode."""
    if normalize_log_mode(level) == "verbose" and get_update_console_log_mode(state) != "verbose":
        return
    text = str(message or "").strip()
    if not text:
        return
    try:
        print(f"ALEXZ_tools Module update: {text}", flush=True)
    except (OSError, ValueError):
        pass


def refresh_console_log(
    state: ModuleBrowserApiState,
    message: str,
    level: str,
    normalize_log_mode: Callable[[str | None], str],
) -> None:
    """Print refresh-progress line to console according to selected mode."""
    if normalize_log_mode(level) == "verbose" and get_update_console_log_mode(state) != "verbose":
        return
    text = str(message or "").strip()
    if not text:
        return
    if text == state.refresh_console_log_last:
        return
    state.refresh_console_log_last = text
    try:
        print(f"ALEXZ_tools Module refresh: {text}", flush=True)
    except (OSError, ValueError):
        pass
