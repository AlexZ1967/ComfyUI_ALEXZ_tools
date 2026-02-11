"""
Module: utils/module_browser/jobs.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Shared backend job helpers for Module Node Picker refresh/update flows.

Purpose:
    Centralizes thread-safe status updates, status snapshot reads, refresh/update
    progress line formatting, and update-target resolution for custom modules.
    This keeps route-level API facade code small while preserving behavior.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


def set_refresh_status(
    *,
    lock: Any,
    status: dict[str, Any],
    now_iso: Callable[[], str],
    **kwargs: Any,
) -> None:
    """Update refresh status dict under lock and stamp `updated_at`."""
    with lock:
        status.update(kwargs)
        status["updated_at"] = now_iso()


def refresh_status_snapshot(*, lock: Any, status: dict[str, Any]) -> dict[str, Any]:
    """Return copy of refresh status dict under lock."""
    with lock:
        return dict(status)


def emit_refresh_progress(
    *,
    lock: Any,
    status: dict[str, Any],
    now_iso: Callable[[], str],
    phase: str,
    current: int = 0,
    total: int = 0,
    remaining: int = 0,
    modules_need_update: int = 0,
    modules_unknown_update: int = 0,
    module: str = "",
    message: str = "",
    last_line: str = "",
    logger_debug: Callable[[str], None] | None = None,
    console_log: Callable[[str, str], None] | None = None,
) -> str:
    """Apply refresh progress counters and return last emitted line."""
    set_refresh_status(
        lock=lock,
        status=status,
        now_iso=now_iso,
        phase=phase,
        current=int(current),
        total=int(total),
        remaining=max(0, int(remaining)),
        modules_need_update=max(0, int(modules_need_update)),
        modules_unknown_update=max(0, int(modules_unknown_update)),
        module=module,
        message=message,
    )
    line = (
        f"phase={phase} current={int(current)}/{int(total)} remaining={max(0, int(remaining))} "
        f"module={module or '-'} message={message or '-'}"
    )
    if line == last_line:
        return last_line

    if callable(logger_debug):
        logger_debug(line)

    if callable(console_log):
        if phase == "sync" and total > 0 and module:
            console_log(
                "sync [{current}/{total}] {module}: {message}".format(
                    current=int(current),
                    total=int(total),
                    module=module,
                    message=message or "-",
                ),
                "verbose",
            )
        elif phase in {"sync", "snapshots", "done"}:
            console_log(
                "phase={phase} message={message} need_update={need}".format(
                    phase=phase,
                    message=message or "-",
                    need=max(0, int(modules_need_update)),
                ),
                "summary",
            )
    return line


def set_update_status(
    *,
    lock: Any,
    status: dict[str, Any],
    now_iso: Callable[[], str],
    **kwargs: Any,
) -> None:
    """Update module-update status dict under lock and stamp `updated_at`."""
    with lock:
        status.update(kwargs)
        status["updated_at"] = now_iso()


def update_status_snapshot(*, lock: Any, status: dict[str, Any]) -> dict[str, Any]:
    """Return copy of update status dict under lock."""
    with lock:
        return dict(status)


def format_update_status_line(status: dict[str, Any]) -> str:
    """Return stable one-line summary of update status for logs."""
    return (
        "scope={scope} phase={phase} current={current}/{total} remaining={remaining} "
        "module={module} updated={updated} up_to_date={up_to_date} failed={failed} message={message}"
    ).format(
        scope=status.get("scope", ""),
        phase=status.get("phase", ""),
        current=int(status.get("current", 0) or 0),
        total=int(status.get("total", 0) or 0),
        remaining=int(status.get("remaining", 0) or 0),
        module=status.get("module", "") or "-",
        updated=int(status.get("updated", 0) or 0),
        up_to_date=int(status.get("up_to_date", 0) or 0),
        failed=int(status.get("failed", 0) or 0),
        message=status.get("message", "") or "-",
    )


def resolve_update_targets(
    *,
    scope: str,
    module_name: str,
    canonical_module_name: Callable[[str], str],
    discover_modules: Callable[[], list[str]],
    sync_module_upstream: Callable[[str], bool],
    module_needs_update: Callable[[str], bool],
    update_console_log: Callable[[str, str], None],
    workers: int,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    """Resolve concrete module names targeted by update request payload."""
    scope_norm = (scope or "").strip().lower()
    if scope_norm == "single":
        canonical = canonical_module_name(module_name)
        if not canonical or canonical == "unknown":
            return []
        return [canonical]

    if scope_norm != "all":
        return []

    discovered = discover_modules()
    total_scan = len(discovered)
    update_console_log(f"target scan started for {total_scan} custom module(s)", "summary")
    if total_scan == 0:
        return []

    def _scan_one(idx: int, mod: str) -> tuple[int, str, bool, bool]:
        sync_ok = sync_module_upstream(mod)
        needs_update = module_needs_update(mod)
        return (idx, mod, sync_ok, needs_update)

    ordered_results: list[tuple[int, str, bool, bool]] = []
    max_workers = max(1, min(int(workers), total_scan))
    if max_workers == 1:
        ordered_results = [_scan_one(idx, mod) for idx, mod in enumerate(discovered, start=1)]
    else:
        result_map: dict[int, tuple[int, str, bool, bool]] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="alexz-update-scan") as executor:
            futures = {executor.submit(_scan_one, idx, mod): idx for idx, mod in enumerate(discovered, start=1)}
            for future in as_completed(futures):
                idx = futures[future]
                mod = discovered[idx - 1]
                try:
                    result_map[idx] = future.result()
                except Exception as exc:  # pragma: no cover - defensive path
                    if callable(warn):
                        warn(f"Update target scan failed for module {mod}: {exc}")
                    result_map[idx] = (idx, mod, False, False)
        ordered_results = [result_map[i] for i in sorted(result_map)]

    targets: list[str] = []
    for idx, mod, sync_ok, needs_update in ordered_results:
        if needs_update:
            targets.append(canonical_module_name(mod))
        update_console_log(
            "scan [{idx}/{total}] {mod}: {state} (sync={sync})".format(
                idx=idx,
                total=total_scan,
                mod=mod,
                state="needs update" if needs_update else "up to date",
                sync="ok" if sync_ok else "skip",
            ),
            "verbose",
        )

    update_console_log(f"target scan finished: {len(targets)} module(s) need update", "summary")
    return list(dict.fromkeys(targets))
