"""
Module: utils/module_browser/update_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Update/install helpers for Module Node Picker backend update workflows.

Purpose:
    Extracts requirements-diff and requirements-install operations from the
    monolithic API module while preserving route-level behavior through wrappers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def requirements_changed_between(
    module_dir: Path,
    before_commit: str,
    after_commit: str,
    *,
    run_command: Callable[[list[str], float, bool], dict[str, Any]],
) -> bool:
    """Check whether `requirements.txt` changed between two git commits."""
    before = (before_commit or "").strip()
    after = (after_commit or "").strip()
    if not before or not after or before == after:
        return False
    diff = run_command(
        ["git", "-C", str(module_dir), "diff", "--name-only", f"{before}..{after}", "--", "requirements.txt"],
        20.0,
        True,
    )
    if not diff.get("ok"):
        return False
    changed_files = [line.strip().lower() for line in str(diff.get("stdout") or "").splitlines() if line.strip()]
    return "requirements.txt" in changed_files


def install_module_requirements(
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    module_dir_resolver: Callable[[str], Path | None],
    run_command: Callable[[list[str], float, bool], dict[str, Any]],
    python_executable: str,
    tail_lines: Callable[[Any], str],
    set_module_requirements_pending: Callable[[str, bool], None],
    logger: Any,
    timeout: float = 1200.0,
) -> dict[str, Any]:
    """Install module `requirements.txt` with pip in current runtime environment."""
    module = canonical_custom_module_name(module_name)
    module_dir = module_dir_resolver(module)
    result: dict[str, Any] = {
        "module": module,
        "status": "error",
        "message": "",
        "requirements_path": "",
    }
    if module_dir is None:
        result["status"] = "not_found"
        result["message"] = "module directory not found"
        return result

    requirements_path = module_dir / "requirements.txt"
    result["requirements_path"] = str(requirements_path)
    if not requirements_path.exists():
        result["status"] = "missing_requirements"
        result["message"] = "requirements.txt not found"
        return result

    cmd = [python_executable, "-m", "pip", "install", "-r", str(requirements_path)]
    logger.info("Installing requirements for module %s via %s", module, result["requirements_path"])
    run = run_command(cmd, timeout, False)
    run_stdout = tail_lines(run.get("stdout"))
    run_stderr = tail_lines(run.get("stderr"))
    if not run.get("ok"):
        result["status"] = "error"
        result["message"] = str(run.get("stderr") or run.get("stdout") or "pip install failed")
        if run_stdout:
            logger.warning("Requirements pip stdout for module %s:\n%s", module, run_stdout)
        if run_stderr:
            logger.warning("Requirements pip stderr for module %s:\n%s", module, run_stderr)
        logger.error("Requirements install failed for module %s: %s", module, result["message"])
        return result
    if run_stdout:
        logger.info("Requirements pip output for module %s:\n%s", module, run_stdout)
    if run_stderr:
        logger.info("Requirements pip warnings for module %s:\n%s", module, run_stderr)
    result["status"] = "installed"
    result["message"] = "requirements installed"
    set_module_requirements_pending(module, False)
    logger.info("Requirements install completed for module %s", module)
    return result


def install_comfyui_requirements(
    *,
    comfyui_requirements_path: Callable[[], Path | None],
    run_command: Callable[[list[str], float, bool], dict[str, Any]],
    python_executable: str,
    tail_lines: Callable[[Any], str],
    set_comfyui_requirements_pending: Callable[[bool], None],
    logger: Any,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Install ComfyUI `requirements.txt` with pip in current runtime environment."""
    result: dict[str, Any] = {
        "module": "ComfyUI",
        "status": "error",
        "message": "",
        "requirements_path": "",
    }
    req = comfyui_requirements_path()
    if req is None:
        result["status"] = "missing_requirements"
        result["message"] = "ComfyUI requirements.txt not found"
        return result
    result["requirements_path"] = str(req)
    logger.info("Installing ComfyUI requirements via %s", result["requirements_path"])
    run = run_command([python_executable, "-m", "pip", "install", "-r", str(req)], timeout, False)
    run_stdout = tail_lines(run.get("stdout"))
    run_stderr = tail_lines(run.get("stderr"))
    if not run.get("ok"):
        result["status"] = "error"
        result["message"] = str(run.get("stderr") or run.get("stdout") or "pip install failed")
        if run_stdout:
            logger.warning("ComfyUI requirements pip stdout:\n%s", run_stdout)
        if run_stderr:
            logger.warning("ComfyUI requirements pip stderr:\n%s", run_stderr)
        logger.error("ComfyUI requirements install failed: %s", result["message"])
        return result
    if run_stdout:
        logger.info("ComfyUI requirements pip output:\n%s", run_stdout)
    if run_stderr:
        logger.info("ComfyUI requirements pip warnings:\n%s", run_stderr)
    result["status"] = "installed"
    result["message"] = "ComfyUI requirements installed"
    set_comfyui_requirements_pending(False)
    logger.info("ComfyUI requirements install completed")
    return result


def install_requirements_for_modules(
    modules: list[str],
    *,
    canonical_custom_module_name: Callable[[str], str],
    install_module_requirements_fn: Callable[[str], dict[str, Any]],
    logger: Any,
) -> dict[str, Any]:
    """Install requirements for multiple modules and return aggregate summary."""
    if not isinstance(modules, list):
        return {"status": "error", "error": "modules must be a list"}

    canonical = [canonical_custom_module_name(str(x)) for x in modules if str(x).strip()]
    canonical = [x for x in dict.fromkeys(canonical) if x and x != "unknown"]
    if not canonical:
        return {"status": "ok", "count": 0, "installed": 0, "failed": 0, "results": []}

    logger.info("Installing requirements for %d module(s): %s", len(canonical), ", ".join(canonical))
    results: list[dict[str, Any]] = []
    installed = 0
    failed = 0
    for module_name in canonical:
        item = install_module_requirements_fn(module_name)
        results.append(item)
        if str(item.get("status")) == "installed":
            installed += 1
        else:
            failed += 1

    logger.info(
        "Requirements install summary: total=%d installed=%d failed=%d",
        len(canonical),
        installed,
        failed,
    )
    return {
        "status": "ok",
        "count": len(canonical),
        "installed": installed,
        "failed": failed,
        "results": results,
    }
