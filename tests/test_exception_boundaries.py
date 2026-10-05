"""Guard narrowed exception handling in deterministic helper modules."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class ExceptionBoundaryTests(unittest.TestCase):
    def test_clean_helpers_do_not_reintroduce_broad_exception_handlers(self):
        repo_root = Path(__file__).resolve().parents[1]
        helper_paths = (
            "nodes/image_look_match_contract_ops.py",
            "nodes/json_output.py",
            "utils/module_browser/catalog/component_registry.py",
            "utils/module_browser/comfyui/manager_data_ops.py",
            "utils/module_browser/core/value_ops.py",
            "utils/module_browser/module/module_info_text.py",
            "utils/module_browser/module/node_snapshot_ops.py",
        )
        offenders: list[str] = []
        for relative_path in helper_paths:
            path = repo_root / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                if node.type is None:
                    offenders.append(f"{relative_path}:{node.lineno}:bare")
                elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                    offenders.append(f"{relative_path}:{node.lineno}:{node.type.id}")
        self.assertEqual(offenders, [])
