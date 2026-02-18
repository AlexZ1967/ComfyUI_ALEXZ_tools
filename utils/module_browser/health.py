"""
Module: utils/module_browser/health.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Manifest health checks for Module Node Picker component registries.

Purpose:
    Validates node/widget/API manifests so add/remove lifecycle changes are
    immediately visible in diagnostics and easier to debug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog.api_manifest import iter_all_api_routes, iter_component_api_routes
from .catalog.widget_manifest import iter_widget_specs


def _iter_node_specs() -> list[tuple[str, str, str, str]]:
    """Load node specs from canonical node registry using safe import fallback."""
    try:
        from ...nodes.node_registry import iter_node_specs  # type: ignore
    except Exception:
        try:
            from nodes.node_registry import iter_node_specs  # type: ignore
        except Exception:
            return []
    return [tuple(item) for item in iter_node_specs()]


def _issue(
    *,
    kind: str,
    component_id: str,
    code: str,
    message: str,
    severity: str = "warning",
) -> dict[str, str]:
    """Create normalized health issue payload entry."""
    return {
        "kind": str(kind),
        "component_id": str(component_id),
        "code": str(code),
        "severity": str(severity),
        "message": str(message),
    }


def build_component_health_report(root: Path | None = None) -> dict[str, Any]:
    """Validate node/widget/API manifests and return structured health report."""
    root_dir = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    issues: list[dict[str, str]] = []

    node_specs = _iter_node_specs()
    node_ids_seen: set[str] = set()
    for spec in node_specs:
        type_name, _display_name, module_import, class_name = spec
        comp_id = f"node:{type_name}"
        if comp_id in node_ids_seen:
            issues.append(
                _issue(
                    kind="node",
                    component_id=comp_id,
                    code="duplicate_component_id",
                    message=f"Duplicate node type in manifest: {type_name}",
                    severity="error",
                )
            )
        node_ids_seen.add(comp_id)
        if not str(module_import or "").strip() or not str(class_name or "").strip():
            issues.append(
                _issue(
                    kind="node",
                    component_id=comp_id,
                    code="invalid_loader_binding",
                    message="Node spec has empty module import or class name.",
                    severity="error",
                )
            )

    widget_specs = list(iter_widget_specs())
    widget_ids_seen: set[str] = set()
    for spec in widget_specs:
        widget_id = str(spec.widget_id or "").strip()
        comp_id = f"widget:{widget_id or 'unknown'}"
        if comp_id in widget_ids_seen:
            issues.append(
                _issue(
                    kind="widget",
                    component_id=comp_id,
                    code="duplicate_component_id",
                    message=f"Duplicate widget_id in manifest: {widget_id}",
                    severity="error",
                )
            )
        widget_ids_seen.add(comp_id)
        entry_rel = str(spec.entrypoint or "").strip()
        if not entry_rel:
            issues.append(
                _issue(
                    kind="widget",
                    component_id=comp_id,
                    code="missing_entrypoint",
                    message="Widget spec entrypoint is empty.",
                    severity="error",
                )
            )
            continue
        entry_abs = (root_dir / entry_rel).resolve()
        if not entry_abs.exists():
            issues.append(
                _issue(
                    kind="widget",
                    component_id=comp_id,
                    code="entrypoint_not_found",
                    message=f"Widget entrypoint not found: {entry_rel}",
                    severity="warning",
                )
            )

    all_api_routes = [str(x) for x in iter_all_api_routes()]
    api_routes = [str(x) for x in iter_component_api_routes()]
    all_route_set = set(all_api_routes)
    route_seen: set[str] = set()
    for route in api_routes:
        comp_id = f"api:{route}"
        if comp_id in route_seen:
            issues.append(
                _issue(
                    kind="api",
                    component_id=comp_id,
                    code="duplicate_component_id",
                    message=f"Duplicate API route in manifest: {route}",
                    severity="error",
                )
            )
        route_seen.add(comp_id)
        if not route.startswith("/"):
            issues.append(
                _issue(
                    kind="api",
                    component_id=comp_id,
                    code="invalid_route",
                    message=f"API route must start with '/': {route}",
                    severity="error",
                )
            )
        if route not in all_route_set:
            issues.append(
                _issue(
                    kind="api",
                    component_id=comp_id,
                    code="component_route_missing_in_all_routes",
                    message=f"Component API route is absent in ALL_API_ROUTES: {route}",
                    severity="error",
                )
            )

    all_seen: set[str] = set()
    for route in all_api_routes:
        if route in all_seen:
            issues.append(
                _issue(
                    kind="api",
                    component_id=f"api:{route}",
                    code="duplicate_route_in_all_routes",
                    message=f"Duplicate route in ALL_API_ROUTES: {route}",
                    severity="error",
                )
            )
        all_seen.add(route)

    error_count = sum(1 for item in issues if item.get("severity") == "error")
    warning_count = sum(1 for item in issues if item.get("severity") == "warning")
    return {
        "ok": len(issues) == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "issue_count": len(issues),
        "issues": issues,
        "checked": {
            "node_specs": len(node_specs),
            "widget_specs": len(widget_specs),
            "all_api_routes": len(all_api_routes),
            "api_routes": len(api_routes),
        },
    }
