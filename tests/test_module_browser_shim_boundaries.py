"""
Module: tests/test_module_browser_shim_boundaries.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Boundary test for legacy `utils.module_browser.*` shim imports.

Purpose:
    Ensures production code no longer depends on top-level compatibility shims
    as an internal API surface; those wrappers remain for external/backward
    compatibility only.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import types
import unittest


class ModuleBrowserShimBoundariesTests(unittest.TestCase):
    """Verify production modules do not import legacy top-level shim wrappers."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [str(repo_root)]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        cls.repo_root = repo_root
        cls.legacy_shims = {
            "api_manifest",
            "catalog_payload_ops",
            "command_ops",
            "component_registry",
            "component_registry_payload_ops",
            "comfyui_git_status_ops",
            "comfyui_state_ops",
            "comfyui_tracking_ops",
            "git_helpers",
            "manager_data_ops",
            "module_identity",
            "module_info",
            "module_info_text",
            "module_update_state_ops",
            "node_classification_ops",
            "node_snapshot_ops",
            "path_ops",
            "pull_ops",
            "refresh_job_ops",
            "release_ops",
            "repo_bootstrap_ops",
            "requirements_pending_ops",
            "runtime_refresh_ops",
            "state_store",
            "tracker_ops",
            "update_job_ops",
            "update_ops",
            "value_ops",
            "widget_manifest",
            "widget_mode_ops",
        }

    def test_production_code_does_not_import_top_level_module_browser_shims(self):
        """Production modules should target canonical subpackages, not legacy shim wrappers."""
        pattern = re.compile(r"module_browser\.([A-Za-z_][A-Za-z0-9_]*)\b(?!\.)")
        offenders: list[str] = []
        for path in sorted(self.repo_root.rglob("*.py")):
            rel = path.relative_to(self.repo_root)
            if rel.parts[0] == "tests":
                continue
            if rel.parts[:2] == ("utils", "module_browser"):
                # The shim package itself is allowed to reference shim names.
                continue
            text = path.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                if match.group(1) in self.legacy_shims:
                    offenders.append(f"{rel}:{match.group(1)}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
