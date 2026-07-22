"""Unit tests for decomposed Color Match helper modules."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


class ColorMatchOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            package = types.ModuleType("ComfyUI_ALEXZ_tools")
            package.__path__ = [root]
            sys.modules["ComfyUI_ALEXZ_tools"] = package
        if "ComfyUI_ALEXZ_tools.nodes" not in sys.modules:
            nodes = types.ModuleType("ComfyUI_ALEXZ_tools.nodes")
            nodes.__path__ = [os.path.join(root, "nodes")]
            sys.modules["ComfyUI_ALEXZ_tools.nodes"] = nodes

    def setUp(self):
        self.color = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match_color_ops")
        self.match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match_match_ops")
        self.metrics = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match_metrics_ops")
        self.lut = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match_lut_ops")

    def test_lab_and_oklab_roundtrips(self):
        rgb = torch.rand(2, 12, 10, 3)
        lab_rgb = self.color._lab_to_rgb_torch(self.color._rgb_to_lab_torch(rgb))
        oklab_rgb = self.color._oklab_to_rgb_torch(self.color._rgb_to_oklab_torch(rgb))
        self.assertLess(float(torch.max(torch.abs(lab_rgb - rgb))), 2e-4)
        self.assertLess(float(torch.max(torch.abs(oklab_rgb - rgb))), 2e-4)

    def test_matching_algorithms_preserve_shape_and_range(self):
        image = torch.rand(2, 16, 12, 3)
        reference = torch.rand(2, 16, 12, 3)
        for function in (
            self.match._mean_std_match_batch,
            self.match._linear_match_batch,
            self.match._tone_curve_match_batch,
            self.match._oklab_cdf_match_batch,
        ):
            output = function(image, reference, None)
            self.assertEqual(output.shape, image.shape)
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)

    def test_metrics_support_explicit_lpips_dependency(self):
        image = torch.zeros(12, 10, 3)
        reference = torch.ones(12, 10, 3) * 0.25
        metrics = self.metrics._quality_metrics(image, reference, lpips_fn=lambda _a, _b: 0.123)
        self.assertEqual(metrics["lpips_alex"], 0.123)
        candidate = self.metrics._auto_optimal_candidate_metrics(
            image,
            reference,
            "mse_ssim_lpips",
            lpips_fn=lambda _a, _b: 0.456,
        )
        self.assertEqual(candidate["lpips_alex"], 0.456)

    def test_lut_grid_and_cube_export(self):
        colors = self.lut._lut_grid_colors(8, torch.device("cpu"), torch.float32)
        self.assertEqual(colors.shape, (512, 3))
        self.assertEqual(self.lut._sanitize_lut_name("Portrait / Test"), "Portrait___Test")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.cube"
            self.lut._write_cube_file(path, colors, 8, "test")
            text = path.read_text(encoding="utf-8")
            self.assertIn("LUT_3D_SIZE 8", text)
            self.assertEqual(len(text.splitlines()), 517)


if __name__ == "__main__":
    unittest.main()
