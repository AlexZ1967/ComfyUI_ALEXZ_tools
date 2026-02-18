"""
Module: utils/interrupt.py
Author: AlexZ1967
Last updated: 2026-02-18

Description:
    Shared ComfyUI interrupt helpers.

Purpose:
    Provides a single, safe way to check user cancellation requests from long
    loops without hard dependency on ComfyUI during unit tests.
"""

from __future__ import annotations

try:
    from comfy import model_management as _model_management
except Exception:  # pragma: no cover - optional in test environments
    _model_management = None


def check_interrupt() -> None:
    """Raise interrupt exception when ComfyUI requests cancellation."""
    if _model_management is not None:
        _model_management.throw_exception_if_processing_interrupted()

