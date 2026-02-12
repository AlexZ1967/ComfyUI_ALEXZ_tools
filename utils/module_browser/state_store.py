"""
Module: utils/module_browser/state_store.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Persistent state-file helpers for Module Node Picker backend.

Purpose:
    Extracts JSON state load/save operations from API module while keeping
    cache semantics and schema normalization controlled by callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def load_state_file(
    state_path: Path,
    *,
    ensure_schema: Callable[[dict[str, Any]], dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Load state from disk and normalize it with the provided schema function."""
    if not state_path.exists():
        return ensure_schema({})
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return ensure_schema(data if isinstance(data, dict) else {})
    except Exception:
        return ensure_schema({})


def save_state_file(
    state_path: Path,
    state: dict[str, dict[str, Any]],
    *,
    ensure_schema: Callable[[dict[str, Any]], dict[str, dict[str, Any]]],
    logger: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize and persist state to disk, returning normalized value."""
    normalized = ensure_schema(state)
    try:
        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=True, indent=2, sort_keys=True)
    except Exception as exc:
        if logger is not None:
            logger.debug("Failed to save module state cache: %s", exc)
    return normalized
