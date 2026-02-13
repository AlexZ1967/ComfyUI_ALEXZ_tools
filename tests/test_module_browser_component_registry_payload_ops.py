"""
Module: tests/test_module_browser_component_registry_payload_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted component-registry payload orchestration helper.

Purpose:
    Ensures Slice-0 registry payload caching and change-tracking behavior stay
    stable after backend helper extraction.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserComponentRegistryPayloadOpsTests(unittest.TestCase):
    """Validate `component_registry_payload_ops` cache and snapshot flows."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import helper modules under test."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import component_registry
        from ComfyUI_ALEXZ_tools.utils.module_browser import component_registry_payload_ops

        self.registry = component_registry
        self.ops = component_registry_payload_ops

    def test_collect_component_registry_payload_returns_cached_under_ttl(self):
        """Cached payload should be returned without rebuilding registry snapshot."""
        cached = (100.0, {"schema_name": "alexz_component_registry", "marker": "cached"})

        def _builder():
            raise AssertionError("registry builder should not run on valid cache hit")

        payload, cache_value = self.ops.collect_component_registry_payload(
            force_refresh=False,
            now_ts=100.1,
            cache_payload=cached,
            ttl_sec=120.0,
            build_default_component_registry=_builder,
            load_module_state=lambda: {},
            save_module_state=lambda _state: None,
            build_registry_snapshot=lambda _registry: {},
            compute_snapshot_signature=lambda _snapshot: "",
            build_component_health_report=lambda: {},
            schema_name="alexz_component_registry",
            schema_version=1,
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
        )
        self.assertEqual(payload.get("marker"), "cached")
        self.assertEqual(cache_value[1].get("marker"), "cached")

    def test_collect_component_registry_payload_tracks_added_removed_components(self):
        """Collector should report added/removed ids and persist tracker state."""
        registry = self.registry.ComponentRegistry()
        registry.register(
            self.registry.ComponentEntry(
                component_id="node:NewNode",
                kind="node",
                name="New Node",
                module="nodes.new",
                source="/tmp/new.py",
            )
        )
        registry.register(
            self.registry.ComponentEntry(
                component_id="widget:module_node_picker",
                kind="widget",
                name="Module Node Picker",
                module="web/module_node_picker.js",
                source="/tmp/module_node_picker.js",
            )
        )
        registry.register(
            self.registry.ComponentEntry(
                component_id="api:/alexz_tools/node_catalog",
                kind="api",
                name="/alexz_tools/node_catalog",
                module="utils/module_node_browser_api.py",
                source="/tmp/module_node_browser_api.py",
            )
        )

        state = {
            "__component_registry__": {
                "snapshot": {
                    "node": ["node:OldNode"],
                    "widget": ["widget:module_node_picker"],
                    "api": ["api:/alexz_tools/old_route"],
                },
                "manifest_signature": "deadbeef0000",
                "updated_at": "2026-02-12T00:00:00+00:00",
            }
        }
        saved_state = []

        payload, _cache_value = self.ops.collect_component_registry_payload(
            force_refresh=True,
            now_ts=101.0,
            cache_payload=None,
            ttl_sec=120.0,
            build_default_component_registry=lambda: registry,
            load_module_state=lambda: state,
            save_module_state=lambda value: saved_state.append(dict(value)),
            build_registry_snapshot=self.registry.build_registry_snapshot,
            compute_snapshot_signature=self.registry.compute_snapshot_signature,
            build_component_health_report=lambda: {"ok": True, "issue_count": 0, "checked": {}},
            schema_name="alexz_component_registry",
            schema_version=1,
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
        )

        self.assertTrue(bool(payload.get("has_changes")))
        self.assertTrue(bool(payload.get("manifest_changed")))
        node_changes = payload.get("changes", {}).get("node", {})
        self.assertIn("node:NewNode", node_changes.get("added", []))
        self.assertIn("node:OldNode", node_changes.get("removed", []))
        self.assertTrue(saved_state)
        tracker = state.get("__component_registry__", {})
        self.assertEqual(tracker.get("schema_name"), "alexz_component_registry")
        self.assertEqual(tracker.get("schema_version"), 1)
        self.assertEqual(str(tracker.get("manifest_signature")), str(payload.get("manifest_signature")))


if __name__ == "__main__":
    unittest.main()

