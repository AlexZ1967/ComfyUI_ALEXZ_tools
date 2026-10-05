"""Tests for repository documentation consistency checks."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


class DocsCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("alexz_docs_check", repo_root / "utils" / "docs_check.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load utils/docs_check.py")
        cls.docs_check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.docs_check)

    def test_version_consistency_reports_mismatch(self):
        module = self.docs_check
        old_pyproject = module.PYPROJECT
        old_changelog = module.CHANGELOG
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                module.PYPROJECT = root / "pyproject.toml"
                module.CHANGELOG = root / "CHANGELOG.md"
                module.PYPROJECT.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
                module.CHANGELOG.write_text("# Changelog\n\n## 1.2.2 - test\n", encoding="utf-8")

                issues: list[str] = []
                module._check_version_consistency(issues, "Version: 1.2.3\n")

                self.assertEqual(len(issues), 1)
                self.assertIn("version mismatch", issues[0])
                self.assertIn("CHANGELOG.md=1.2.2", issues[0])
        finally:
            module.PYPROJECT = old_pyproject
            module.CHANGELOG = old_changelog

    def test_local_markdown_link_check_reports_missing_target(self):
        module = self.docs_check
        old_root = module.ROOT
        old_readme = module.README
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                guides = root / "guides"
                guides.mkdir()
                module.ROOT = root
                module.README = root / "README.md"
                module.README.write_text(
                    "[valid](guides/guide.md)\n[missing](guides/missing.md)\n",
                    encoding="utf-8",
                )
                (guides / "guide.md").write_text("[README](../README.md)\n", encoding="utf-8")

                issues: list[str] = []
                module._check_local_markdown_links(issues)

                self.assertEqual(issues, ["README.md: broken local link 'guides/missing.md'"])
        finally:
            module.ROOT = old_root
            module.README = old_readme

    def test_registry_metadata_check_reports_missing_and_orphan_entries(self):
        module = self.docs_check
        old_registry = module.NODE_REGISTRY
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                module.NODE_REGISTRY = Path(tmpdir) / "node_registry.py"
                module.NODE_REGISTRY.write_text(
                    "\n".join(
                        [
                            'NODE_SPECS = (NodeSpec("NodeA", "A", ".a", "A"),)',
                            'NODE_UI_METADATA = {"NodeB": {"description": "orphan"}}',
                        ]
                    ),
                    encoding="utf-8",
                )

                issues: list[str] = []
                module._check_registry_metadata(issues)

                self.assertEqual(
                    issues,
                    [
                        "nodes/node_registry.py: missing NODE_UI_METADATA for `NodeA`",
                        "nodes/node_registry.py: orphan NODE_UI_METADATA for `NodeB`",
                    ],
                )
        finally:
            module.NODE_REGISTRY = old_registry
