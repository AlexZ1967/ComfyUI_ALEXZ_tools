"""
Module: utils/module_browser/command_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Subprocess and git command execution helpers for Module Node Picker backend.

Purpose:
    Centralizes non-interactive command execution, git safe.directory recovery,
    and log-friendly output normalization as part of Phase 3 backend split.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable


def extract_git_repo_from_args(args: list[str]) -> str | None:
    """Extract normalized repository path from a `git -C <path> ...` command."""
    if not args or str(args[0]).strip() != "git":
        return None
    try:
        idx = args.index("-C")
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    try:
        return str(Path(str(args[idx + 1])).resolve())
    except Exception:
        return str(args[idx + 1])


def is_git_dubious_ownership_error(text: str) -> bool:
    """Return true when output indicates git `safe.directory` protection error."""
    lower = (text or "").strip().lower()
    return "detected dubious ownership in repository" in lower and "safe.directory" in lower


def try_mark_git_safe_directory(
    repo_dir: str,
    env: dict[str, str],
    *,
    timeout: float = 15.0,
    subprocess_run: Callable[..., Any] | None = None,
    logger: Any | None = None,
) -> bool:
    """Try to add repo path to global git `safe.directory` list."""
    repo = str(repo_dir or "").strip()
    if not repo:
        return False
    run = subprocess_run or subprocess.run
    try:
        proc = run(
            ["git", "config", "--global", "--add", "safe.directory", repo],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to add safe.directory for %s: %s", repo, exc)
        return False
    if int(getattr(proc, "returncode", 1)) == 0:
        if logger is not None:
            logger.info("Added git safe.directory: %s", repo)
        return True
    if logger is not None:
        logger.warning(
            "Unable to add safe.directory for %s: %s",
            repo,
            str(getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "unknown error").strip(),
        )
    return False


def run_command(
    args: list[str],
    *,
    timeout: float = 120.0,
    disable_git_prompt: bool = False,
    subprocess_run: Callable[..., Any] | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Run command and return normalized result with git safe.directory retry."""
    env = os.environ.copy()
    if disable_git_prompt:
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_ASKPASS", "echo")

    run = subprocess_run or subprocess.run
    try:
        proc = run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}

    result = {
        "ok": int(getattr(proc, "returncode", 1)) == 0,
        "returncode": int(getattr(proc, "returncode", 1)),
        "stdout": str(getattr(proc, "stdout", "") or "").strip(),
        "stderr": str(getattr(proc, "stderr", "") or "").strip(),
    }
    if result["ok"] or not args or str(args[0]).strip() != "git":
        return result

    repo_dir = extract_git_repo_from_args(args)
    if not repo_dir:
        return result

    error_text = f"{result.get('stderr', '')}\n{result.get('stdout', '')}"
    if not is_git_dubious_ownership_error(error_text):
        return result

    if not try_mark_git_safe_directory(
        repo_dir,
        env,
        timeout=15.0,
        subprocess_run=run,
        logger=logger,
    ):
        return result

    try:
        retry = run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}

    return {
        "ok": int(getattr(retry, "returncode", 1)) == 0,
        "returncode": int(getattr(retry, "returncode", 1)),
        "stdout": str(getattr(retry, "stdout", "") or "").strip(),
        "stderr": str(getattr(retry, "stderr", "") or "").strip(),
    }


def run_git(
    args: list[str],
    *,
    timeout: float = 2.0,
    run_command_fn: Callable[..., dict[str, Any]] = run_command,
) -> str | None:
    """Run git command in non-interactive mode and return trimmed stdout."""
    result = run_command_fn(args, timeout=timeout, disable_git_prompt=True)
    if not bool(result.get("ok")):
        return None
    output = str(result.get("stdout") or "").strip()
    return output or None


def tail_lines(text: str | None, max_lines: int = 80) -> str:
    """Return tail of multiline output to keep diagnostics concise."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["...", *lines[-max_lines:]])

