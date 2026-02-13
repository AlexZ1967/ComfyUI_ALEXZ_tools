"""
Module: tests/test_module_browser_manager_data_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted ComfyUI-Manager metadata/statistics helpers.

Purpose:
    Verifies manager DB/stats cache loaders and fallback update inference remain
    stable after backend helper extraction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone


def _github_id(url: str | None) -> str:
    """Convert GitHub URL to stable `owner/repo` id used in tests."""
    text = str(url or "").strip().replace(".git", "")
    marker = "github.com/"
    idx = text.lower().find(marker)
    if idx < 0:
        return ""
    tail = text[idx + len(marker) :].strip().strip("/")
    parts = [part for part in tail.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0].lower()}/{parts[1].lower()}"


def _repo_name(url: str | None) -> str:
    """Extract repository short name from URL."""
    gid = _github_id(url)
    if not gid:
        return ""
    return gid.split("/", 1)[1]


def _normalize_repo_url(url: str | None) -> str:
    """Normalize URL text for deterministic test lookups."""
    text = str(url or "").strip().replace("htps://", "https://")
    if not text:
        return ""
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/")


def _parse_datetime(text: str | None) -> datetime | None:
    """Parse ISO datetime for stats tests."""
    value = str(text or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


class ModuleBrowserManagerDataOpsTests(unittest.TestCase):
    """Cover manager metadata/stats helpers extracted from API facade."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import manager_data_ops

        self.ops = manager_data_ops

    def test_load_manager_index_builds_alias_maps_and_reuses_cache(self):
        """Index loader should map module metadata by id/github/repo aliases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "custom-node-list.json"
            db_path.write_text(
                json.dumps(
                    {
                        "custom_nodes": [
                            {
                                "id": "ComfyUI_ALEXZ_tools",
                                "title": "ALEXZ Tools",
                                "author": "alexz1967",
                                "description": "Custom nodes",
                                "repository": "https://github.com/alexz1967/ComfyUI_ALEXZ_tools.git",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            warnings = []
            result = self.ops.load_manager_index(
                cache=None,
                manager_custom_db_path=lambda: db_path,
                pick_repo_url=lambda raw: raw.get("repository"),
                github_id=_github_id,
                repo_name=_repo_name,
                logger_warning=lambda msg, exc: warnings.append((msg, str(exc))),
            )
            self.assertIn("comfyui_alexz_tools", result["by_id"])
            self.assertIn("alexz1967/comfyui_alexz_tools", result["by_github"])
            self.assertIn("comfyui_alexz_tools", result["by_repo_name"])
            self.assertFalse(warnings)

            cached = self.ops.load_manager_index(
                cache=result,
                manager_custom_db_path=lambda: Path("/does/not/matter.json"),
                pick_repo_url=lambda raw: raw.get("repository"),
                github_id=_github_id,
                repo_name=_repo_name,
                logger_warning=lambda _msg, _exc: None,
            )
            self.assertIs(cached, result)

    def test_load_manager_github_stats_maps_by_url_and_github_id(self):
        """Stats loader should normalize repository URL keys and github aliases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "github-stats.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "htps://github.com/alexz1967/ComfyUI_ALEXZ_tools": {
                            "last_update": "2026-02-13T01:00:00+00:00"
                        }
                    }
                ),
                encoding="utf-8",
            )
            stats = self.ops.load_manager_github_stats(
                cache=None,
                manager_github_stats_path=lambda: stats_path,
                normalize_repo_url=_normalize_repo_url,
                github_id=_github_id,
                logger_warning=lambda _msg, _exc: None,
            )
            self.assertIn("https://github.com/alexz1967/ComfyUI_ALEXZ_tools", stats["by_url"])
            self.assertIn("alexz1967/comfyui_alexz_tools", stats["by_github"])

    def test_infer_update_from_manager_stats_uses_timestamp_delta(self):
        """Update inference should return True when remote timestamp is newer."""
        needs_update, remote_ts = self.ops.infer_update_from_manager_stats(
            repository_url="https://github.com/alexz1967/ComfyUI_ALEXZ_tools",
            installed_updated_at="2026-02-12T01:00:00+00:00",
            manager_stats_last_update_fn=lambda _repo: "2026-02-13T01:00:00+00:00",
            parse_datetime=_parse_datetime,
        )
        self.assertTrue(needs_update)
        self.assertEqual(remote_ts, "2026-02-13T01:00:00+00:00")

        unknown, _remote = self.ops.infer_update_from_manager_stats(
            repository_url="https://github.com/alexz1967/ComfyUI_ALEXZ_tools",
            installed_updated_at="",
            manager_stats_last_update_fn=lambda _repo: "2026-02-13T01:00:00+00:00",
            parse_datetime=_parse_datetime,
        )
        self.assertIsNone(unknown)


if __name__ == "__main__":
    unittest.main()

