#!/usr/bin/env python3
"""
Module: utils/module_browser/manifest_check.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Standalone manifest validator for Module Node Picker component registry.

Purpose:
    Provides a quick command-line check for node/widget/API manifest integrity
    and deterministic registry signature, useful during add/remove refactors.
"""

from __future__ import annotations

import json
from typing import Any

from ..catalog.component_registry import (
    build_default_component_registry,
    build_registry_snapshot,
    compute_snapshot_signature,
)
from ..health import build_component_health_report


def run_manifest_check(strict: bool = True) -> dict[str, Any]:
    """Run component-manifest validation and return structured report."""
    registry = build_default_component_registry()
    snapshot = build_registry_snapshot(registry)
    signature = compute_snapshot_signature(snapshot)
    health = build_component_health_report()
    status = "ok" if bool(health.get("ok")) else "issues"
    report = {
        "status": status,
        "strict": bool(strict),
        "summary": registry.summary(),
        "manifest_signature": signature,
        "health": health,
    }
    if strict and not bool(health.get("ok")):
        report["status"] = "failed"
    return report


def main() -> int:
    """Entry point used when script is executed directly."""
    report = run_manifest_check(strict=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if str(report.get("status")) != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
