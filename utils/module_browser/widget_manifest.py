"""
Module: utils/module_browser/widget_manifest.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Canonical widget manifest for ComfyUI_ALEXZ_tools frontend entries.

Purpose:
    Provides a stable registry source for widget lifecycle tracking so new
    widgets can be added/removed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WidgetSpec:
    """Declarative descriptor for one frontend widget entrypoint."""

    widget_id: str
    name: str
    entrypoint: str
    enabled: bool = True


WIDGET_SPECS: tuple[WidgetSpec, ...] = (
    WidgetSpec(
        widget_id="module_node_picker",
        name="Module Node Picker",
        entrypoint="web/module_node_picker.js",
        enabled=True,
    ),
)


def iter_widget_specs():
    """Yield declared widget specs from manifest."""
    for spec in WIDGET_SPECS:
        yield spec

