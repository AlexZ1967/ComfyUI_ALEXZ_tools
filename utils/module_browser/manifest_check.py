"""Compatibility wrapper for legacy ``utils.module_browser.manifest_check`` imports."""

from __future__ import annotations

import json
from typing import Any

from .catalog.component_registry import (
    build_default_component_registry,
    build_registry_snapshot,
    compute_snapshot_signature,
)
from .health import build_component_health_report


def run_manifest_check(strict: bool = True) -> dict[str, Any]:
    """Run component-manifest validation using legacy module-level dependencies."""
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


__all__ = ["run_manifest_check", "main", "build_component_health_report"]
