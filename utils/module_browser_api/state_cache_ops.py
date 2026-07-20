"""
Module: utils/module_browser_api/state_cache_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    State-cache and runtime-warmup helpers for module browser API facade.

Purpose:
    Keeps module-state persistence and runtime warmup orchestration out of
    `utils/module_node_browser_api.py` so the facade remains thinner.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from .state import ModuleBrowserApiState


def sync_runtime_warmup_from_legacy(
    state: ModuleBrowserApiState,
    *,
    lazy_refresh_done: bool,
    runtime_warmup_thread: threading.Thread | None,
) -> None:
    """Apply legacy warmup globals to extracted runtime state."""
    state.lazy_refresh_done = bool(lazy_refresh_done)
    state.runtime_warmup_thread = runtime_warmup_thread


def sync_runtime_warmup_to_legacy(
    state: ModuleBrowserApiState,
) -> tuple[bool, threading.Thread | None]:
    """Return legacy warmup globals mirrored from extracted runtime state."""
    return (bool(state.lazy_refresh_done), state.runtime_warmup_thread)


def load_module_state_cache(
    cache: dict[str, dict[str, Any]] | None,
    *,
    state_path: Path,
    ensure_schema: Callable[[dict[str, Any]], dict[str, Any]],
    load_state_file: Callable[..., dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Load module state cache lazily from disk and return normalized payload."""
    if cache is not None:
        return cache
    return load_state_file(
        state_path,
        ensure_schema=ensure_schema,
    )


def save_module_state_cache(
    state: dict[str, dict[str, Any]],
    *,
    state_path: Path,
    ensure_schema: Callable[[dict[str, Any]], dict[str, Any]],
    save_state_file: Callable[..., dict[str, dict[str, Any]]],
    logger: Any,
) -> dict[str, dict[str, Any]]:
    """Persist module state cache to disk and return normalized payload."""
    return save_state_file(
        state_path,
        state,
        ensure_schema=ensure_schema,
        logger=logger,
    )


def ensure_runtime_state_ready(
    state: ModuleBrowserApiState,
    *,
    sync_from_legacy: Callable[[], None],
    sync_to_legacy: Callable[[], None],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    set_custom_update_checked: Callable[[bool], bool],
    announce_tracked_module_updates: Callable[..., dict[str, Any]],
    track_comfyui_local_update: Callable[[], None],
) -> None:
    """Ensure startup-time runtime state is initialized once."""
    sync_from_legacy()
    if state.lazy_refresh_done:
        return
    load_module_state()
    set_custom_update_checked(False)
    announce_tracked_module_updates(local_only=True)
    track_comfyui_local_update()
    state.lazy_refresh_done = True
    sync_to_legacy()


def start_runtime_state_warmup(
    state: ModuleBrowserApiState,
    *,
    sync_from_legacy: Callable[[], None],
    sync_to_legacy: Callable[[], None],
    set_custom_update_checked: Callable[[bool], bool],
    ensure_runtime_state_ready_fn: Callable[[], None],
    logger_warning: Callable[[str, Any], None],
) -> bool:
    """Start one background warmup thread for runtime state if needed."""
    sync_from_legacy()
    if state.lazy_refresh_done:
        return False

    with state.runtime_warmup_lock:
        sync_from_legacy()
        if state.lazy_refresh_done:
            return False
        existing = state.runtime_warmup_thread
        if existing is not None and existing.is_alive():
            return False
        set_custom_update_checked(False)

        def _runner() -> None:
            try:
                ensure_runtime_state_ready_fn()
            except Exception as exc:  # pragma: no cover - diagnostic
                logger_warning("Runtime warmup failed: %s", exc, exc_info=True)
            finally:
                with state.runtime_warmup_lock:
                    state.runtime_warmup_thread = None
                    sync_to_legacy()

        state.runtime_warmup_thread = threading.Thread(
            target=_runner,
            name="ALEXZ_tools_RuntimeWarmup",
            daemon=True,
        )
        sync_to_legacy()
        state.runtime_warmup_thread.start()
        return True


def runtime_warmup_status(
    state: ModuleBrowserApiState,
    *,
    sync_from_legacy: Callable[[], None],
) -> dict[str, Any]:
    """Return minimal runtime warmup status for frontend polling."""
    sync_from_legacy()
    with state.runtime_warmup_lock:
        thread = state.runtime_warmup_thread
        running = bool(thread is not None and thread.is_alive())
    return {
        "running": running,
        "done": bool(state.lazy_refresh_done),
    }
