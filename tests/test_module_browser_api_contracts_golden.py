"""
Module: tests/test_module_browser_api_contracts_golden.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Golden contract tests for module-browser API payloads.

Purpose:
    Freeze stable JSON payload shapes for node catalog, module info, refresh
    status, and update status to prevent accidental API regressions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import types
import unittest


def _install_folder_paths_stub() -> None:
    """Install minimal folder_paths stub for non-Comfy test runtime."""
    if "folder_paths" in sys.modules:
        stub = sys.modules["folder_paths"]
        if not hasattr(stub, "get_folder_paths"):
            stub.get_folder_paths = lambda kind: [os.path.join(os.getcwd(), "custom_nodes")]
        return
    stub = types.SimpleNamespace(
        get_folder_paths=lambda kind: [os.path.join(os.getcwd(), "custom_nodes")]
    )
    sys.modules["folder_paths"] = stub


class ModuleBrowserApiContractsGoldenTests(unittest.TestCase):
    """Validate stable API payload contracts against golden JSON fixtures."""

    @classmethod
    def setUpClass(cls):
        """Prepare package path and lazy import context."""
        repo_root = Path(__file__).resolve().parents[1]
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [str(repo_root)]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()
        cls.repo_root = repo_root
        cls.golden_dir = repo_root / "tests" / "golden" / "module_browser_api_contracts"

    def _load_golden(self, name: str) -> dict:
        """Load one golden payload fixture."""
        payload_path = self.golden_dir / name
        return json.loads(payload_path.read_text(encoding="utf-8"))

    def test_node_catalog_payload_matches_golden(self):
        """Freeze `/alexz_tools/node_catalog` payload contract with deterministic fixture."""
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.handlers_catalog import (
            build_node_catalog_payload,
        )

        payload = build_node_catalog_payload(
            mode="releases",
            start_runtime_state_warmup=lambda: None,
            build_group_catalog=lambda: {
                "core": [
                    {
                        "node_name": "CoreNode",
                        "display_name": "Core Node",
                        "module": "nodes",
                        "group": "core",
                        "category": "image/utils",
                        "annotation": "builtin",
                    }
                ],
                "custom": [
                    {
                        "node_name": "CustomNode",
                        "display_name": "Custom Node",
                        "module": "ComfyUI_ALEXZ_tools",
                        "group": "custom",
                        "category": "ALEXZ/test",
                        "annotation": "custom",
                    }
                ],
                "api": [],
                "core_extras": [],
            },
            build_group_modules=lambda grouped: {
                "core": [{"module": "nodes", "count": 1}],
                "custom": [{"module": "ComfyUI_ALEXZ_tools", "count": 1}],
                "api": [],
                "core_extras": [],
            },
            comfyui_git_status=lambda **kwargs: {
                "check_mode": "releases",
                "update_status": "up_to_date",
                "update_available": False,
            },
            custom_update_checked_flag=lambda: True,
            count_custom_modules_need_update=lambda: 1,
            count_custom_modules_unknown_update=lambda: 0,
            list_custom_modules_unknown_update=lambda: [],
            runtime_warmup_status=lambda: {"running": False, "done": True},
            build_group_payload=lambda grouped, mods: [
                {
                    "id": "core",
                    "title": "Core_Nodes",
                    "count": 1,
                    "nodes": grouped["core"],
                    "module_count": 1,
                    "modules": mods["core"],
                },
                {
                    "id": "custom",
                    "title": "Custom_Nodes",
                    "count": 1,
                    "nodes": grouped["custom"],
                    "module_count": 1,
                    "modules": mods["custom"],
                },
            ],
        )

        self.assertEqual(payload, self._load_golden("node_catalog.json"))

    def test_module_info_payload_matches_golden(self):
        """Freeze `/alexz_tools/module_info` payload contract with deterministic fixture."""
        from ComfyUI_ALEXZ_tools.utils.module_browser.module.module_info import (
            resolve_module_info_uncached,
        )

        payload = resolve_module_info_uncached(
            group="custom",
            module_name="testmod",
            sync_upstream=True,
            cache_only=False,
            canonical_custom_module_name=lambda s: s,
            apply_node_change_info=lambda result, group, module: None,
            sync_module_upstream=lambda module: True,
            load_module_state=lambda: {"__meta__": {"custom_update_checked": True}},
            custom_update_checked_flag=lambda state: bool(
                (state or {}).get("__meta__", {}).get("custom_update_checked")
            ),
            module_git_state=lambda module: {
                "module_path": "/tmp/fake_mod",
                "repository": "https://github.com/alex/testmod",
                "installed_commit": "1234567890abcdef",
                "installed_updated_at": "2026-02-10T00:00:00+00:00",
                "remote_updated_at": "2026-02-10T01:00:00+00:00",
                "has_upstream": True,
                "ahead": 0,
                "behind": 2,
                "remote_head": "fedcba0987654321",
                "update_available": True,
            },
            module_repo_url=lambda module: "https://github.com/alex/testmod",
            manager_meta_for_module=lambda module, repo: {
                "title": "Test Module",
                "description": "From manager",
            },
            module_local_readme_summary=lambda module: "Local summary",
            sanitize_module_description=lambda text: text,
            github_id=lambda repo: "alex/testmod",
            infer_update_from_manager_stats=lambda repo, installed_at: (None, None),
            short_commit=lambda commit: (commit or "")[:8],
            remember_module_state=lambda module, result: None,
        )

        self.assertEqual(payload, self._load_golden("module_info.json"))

    def test_refresh_status_payload_matches_golden(self):
        """Freeze `/alexz_tools/module_refresh_status` payload contract."""
        from ComfyUI_ALEXZ_tools.utils.module_browser.jobs import refresh_status_snapshot

        status = {
            "running": False,
            "phase": "idle",
            "current": 0,
            "total": 0,
            "remaining": 0,
            "modules_need_update": 0,
            "modules_unknown_update": 0,
            "unknown_update_modules": [],
            "module": "",
            "message": "idle",
            "error": "",
            "sync_upstreams": False,
            "started_at": "",
            "updated_at": "",
            "refreshed_at": "",
        }
        payload = refresh_status_snapshot(lock=threading.Lock(), status=status)
        self.assertEqual(payload, self._load_golden("refresh_status.json"))

    def test_update_status_payload_matches_golden(self):
        """Freeze `/alexz_tools/module_update_status` payload contract."""
        from ComfyUI_ALEXZ_tools.utils.module_browser.jobs import update_status_snapshot

        status = {
            "running": False,
            "phase": "idle",
            "scope": "single",
            "current": 0,
            "total": 0,
            "remaining": 0,
            "module": "",
            "message": "idle",
            "error": "",
            "updated": 0,
            "up_to_date": 0,
            "failed": 0,
            "requirements_changed": False,
            "requirements_modules": [],
            "results": [],
            "log_mode": "summary",
            "started_at": "",
            "updated_at": "",
            "finished_at": "",
        }
        payload = update_status_snapshot(lock=threading.Lock(), status=status)
        self.assertEqual(payload, self._load_golden("update_status.json"))


if __name__ == "__main__":
    unittest.main()
