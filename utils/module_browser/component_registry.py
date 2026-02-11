"""
Module: utils/module_browser/component_registry.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Lightweight component registry for ALEXZ_tools nodes/widgets/API endpoints.

Purpose:
    Provides a single extensibility surface for add/remove lifecycle of module
    components without coupling this logic to route handlers or loader internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComponentEntry:
    """Immutable descriptor of one registered module component."""

    component_id: str
    kind: str
    name: str
    module: str
    source: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to JSON-friendly dictionary."""
        return asdict(self)


class ComponentRegistry:
    """In-memory registry for nodes/widgets/api components."""

    def __init__(self) -> None:
        self._entries: dict[str, ComponentEntry] = {}

    def register(self, entry: ComponentEntry) -> None:
        """Register or replace a component entry by component_id."""
        self._entries[str(entry.component_id)] = entry

    def unregister(self, component_id: str) -> None:
        """Unregister component by id if present."""
        self._entries.pop(str(component_id), None)

    def list(self, kind: str | None = None) -> list[ComponentEntry]:
        """List registry entries, optionally filtered by component kind."""
        values = self._entries.values()
        if kind is None:
            return sorted(values, key=lambda x: (x.kind, x.module, x.name, x.component_id))
        kind_norm = str(kind).strip().lower()
        return sorted(
            (item for item in values if str(item.kind).strip().lower() == kind_norm),
            key=lambda x: (x.module, x.name, x.component_id),
        )

    def summary(self) -> dict[str, Any]:
        """Return compact summary used by diagnostics/tests."""
        nodes = self.list("node")
        widgets = self.list("widget")
        apis = self.list("api")
        return {
            "node_count": len(nodes),
            "widget_count": len(widgets),
            "api_count": len(apis),
            "total": len(self._entries),
        }


_DEFAULT_API_ROUTES: tuple[str, ...] = (
    "/alexz_tools/node_catalog",
    "/alexz_tools/module_info",
    "/alexz_tools/module_list",
    "/alexz_tools/module_nodes",
    "/alexz_tools/module_refresh",
    "/alexz_tools/module_refresh_status",
    "/alexz_tools/comfyui_info",
)


def _iter_node_specs() -> list[tuple[str, str, str, str]]:
    """Load node specs from central node registry with import-safe fallback."""
    try:
        from ...nodes.node_registry import iter_node_specs  # type: ignore
    except Exception:
        try:
            from nodes.node_registry import iter_node_specs  # type: ignore
        except Exception:
            return []
    return [tuple(item) for item in iter_node_specs()]


def build_default_component_registry() -> ComponentRegistry:
    """Build default registry view for ALEXZ_tools module components."""
    registry = ComponentRegistry()
    root = Path(__file__).resolve().parents[2]

    for type_name, display_name, module_import, class_name in _iter_node_specs():
        registry.register(
            ComponentEntry(
                component_id=f"node:{type_name}",
                kind="node",
                name=display_name,
                module=module_import,
                source=f"{root / 'nodes'}",
                enabled=True,
            )
        )

    registry.register(
        ComponentEntry(
            component_id="widget:module_node_picker",
            kind="widget",
            name="Module Node Picker",
            module="web/module_node_picker.js",
            source=f"{root / 'web' / 'module_node_picker.js'}",
            enabled=True,
        )
    )

    for route in _DEFAULT_API_ROUTES:
        registry.register(
            ComponentEntry(
                component_id=f"api:{route}",
                kind="api",
                name=route,
                module="utils/module_node_browser_api.py",
                source=f"{root / 'utils' / 'module_node_browser_api.py'}",
                enabled=True,
            )
        )

    return registry

