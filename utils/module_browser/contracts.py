"""
Module: utils/module_browser/contracts.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Versioned contract helpers for Module Node Picker backend caches.

Purpose:
    Introduces schema-version metadata for persisted runtime state so future
    internal refactors can evolve cache format safely without breaking startup.
"""

from __future__ import annotations

from typing import Any


MODULE_STATE_SCHEMA_VERSION = 1
COMPONENT_REGISTRY_SCHEMA_NAME = "alexz_component_registry"
COMPONENT_REGISTRY_SCHEMA_VERSION = 1


def ensure_module_state_schema(state: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize persisted module-state object and ensure metadata schema fields."""
    normalized: dict[str, Any] = dict(state) if isinstance(state, dict) else {}
    meta_raw = normalized.get("__meta__")
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    if not meta.get("schema_name"):
        meta["schema_name"] = "alexz_module_state"
    meta["schema_version"] = MODULE_STATE_SCHEMA_VERSION
    normalized["__meta__"] = meta
    return normalized
