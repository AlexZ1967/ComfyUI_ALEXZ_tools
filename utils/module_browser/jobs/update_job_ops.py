"""
Module: utils/module_browser/update_job_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Module-update job execution helpers for Module Node Picker backend.

Purpose:
    Extracts synchronous update-job execution logic from API worker thread
    wrappers while preserving status/log callback behavior.
"""

from __future__ import annotations

from typing import Any, Callable


def run_module_update_job(
    *,
    scope_norm: str,
    module_name: str,
    normalized_log_mode: str,
    update_console_log: Callable[[str, str], None],
    set_update_status: Callable[..., None],
    pull_comfyui: Callable[[], dict[str, Any]],
    pull_custom_module: Callable[[str], dict[str, Any]],
    resolve_update_targets: Callable[[str, str], list[str]],
    refresh_module_runtime_state: Callable[[], None],
    now_iso: Callable[[], str],
    perf_counter: Callable[[], float],
) -> None:
    """Execute one update job synchronously and report progress via callbacks."""
    update_console_log(
        f"job started (scope={scope_norm}, module={module_name or '-'}, log_mode={normalized_log_mode})",
        "summary",
    )
    if scope_norm == "comfyui":
        update_console_log("ComfyUI update: pull start", "summary")
        set_update_status(phase="update", current=0, total=1, remaining=1, module="ComfyUI", message="pull")
        item = pull_comfyui()
        status = str(item.get("status") or "")
        update_console_log(f"ComfyUI update: pull done (status={status})", "summary")
        updated_count = 1 if status == "updated" else 0
        uptodate_count = 1 if status == "up_to_date" else 0
        failed_count = 1 if status not in {"updated", "up_to_date"} else 0
        requirements_changed = bool(item.get("requirements_changed"))
        requirements_path = str(item.get("requirements_path") or "").strip()
        set_update_status(
            phase="update",
            current=1,
            total=1,
            remaining=0,
            module="ComfyUI",
            message=status or "done",
            updated=updated_count,
            up_to_date=uptodate_count,
            failed=failed_count,
            requirements_changed=requirements_changed,
            requirements_paths=[requirements_path] if requirements_changed and requirements_path else [],
            requirements_modules=[],
            results=[item],
        )
        refresh_module_runtime_state()
        set_update_status(
            running=False,
            phase="done",
            message="done",
            module="",
            requirements_changed=requirements_changed,
            requirements_paths=[requirements_path] if requirements_changed and requirements_path else [],
            requirements_modules=[],
            finished_at=now_iso(),
        )
        update_console_log("job finished (scope=comfyui)", "summary")
        return

    targets = resolve_update_targets(scope_norm, module_name)
    total = len(targets)
    update_console_log(
        "resolved targets: {total} ({mods})".format(
            total=total,
            mods=", ".join(targets) if targets else "-",
        ),
        "summary",
    )
    set_update_status(phase="update", total=total, remaining=total, message="running")
    if total == 0:
        refresh_module_runtime_state()
        set_update_status(
            running=False,
            phase="done",
            message="nothing_to_update",
            results=[],
            requirements_changed=False,
            requirements_modules=[],
            finished_at=now_iso(),
        )
        update_console_log("job finished: nothing to update", "summary")
        return

    updated_count = 0
    uptodate_count = 0
    failed_count = 0
    requirements_modules: list[str] = []
    requirements_paths: list[str] = []
    results: list[dict[str, Any]] = []

    for idx, target in enumerate(targets, start=1):
        set_update_status(
            phase="update",
            current=idx - 1,
            total=total,
            remaining=total - idx + 1,
            module=target,
            message="pull",
        )
        update_console_log(f"[{idx}/{total}] {target}: pull start", "verbose")
        pull_started = perf_counter()
        item = pull_custom_module(target)
        pull_elapsed = perf_counter() - pull_started
        results.append(item)
        status = str(item.get("status") or "")
        update_console_log(
            "[{idx}/{total}] {target}: pull done (status={status}, elapsed={elapsed:.2f}s)".format(
                idx=idx,
                total=total,
                target=target,
                status=status or "unknown",
                elapsed=pull_elapsed,
            ),
            "verbose",
        )
        if status == "updated":
            updated_count += 1
        elif status == "up_to_date":
            uptodate_count += 1
        else:
            failed_count += 1
            update_console_log(
                "[{idx}/{total}] {target}: failed ({msg})".format(
                    idx=idx,
                    total=total,
                    target=target,
                    msg=str(item.get("message") or "unknown error"),
                ),
                "summary",
            )
        if bool(item.get("requirements_changed")):
            requirements_modules.append(target)
            req_path = str(item.get("requirements_path") or "").strip()
            if req_path:
                requirements_paths.append(req_path)
        set_update_status(
            phase="update",
            current=idx,
            total=total,
            remaining=total - idx,
            module=target,
            message=status or "done",
            updated=updated_count,
            up_to_date=uptodate_count,
            failed=failed_count,
            requirements_changed=bool(requirements_modules),
            requirements_paths=list(dict.fromkeys(requirements_paths)),
            requirements_modules=requirements_modules,
            results=results,
        )

    update_console_log("refreshing runtime state after update run...", "summary")
    refresh_module_runtime_state()
    set_update_status(
        running=False,
        phase="done",
        message="done",
        module="",
        requirements_changed=bool(requirements_modules),
        requirements_paths=list(dict.fromkeys(requirements_paths)),
        requirements_modules=requirements_modules,
        finished_at=now_iso(),
    )
    update_console_log(
        "job finished: updated={updated}, up_to_date={uptodate}, failed={failed}".format(
            updated=updated_count,
            uptodate=uptodate_count,
            failed=failed_count,
        ),
        "summary",
    )
