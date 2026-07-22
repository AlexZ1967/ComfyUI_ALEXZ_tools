"""Unit tests for decomposed Look Match helper modules."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
import unittest

import torch


class LookMatchOpsTests(unittest.TestCase):
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
        self.ops = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_look_match_ops")
        self.resolve = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_look_match_resolve_ops")
        self.contract = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_look_match_contract_ops")

    def test_batch_alpha_mask_and_resize_helpers(self):
        rgba = torch.rand(1, 20, 30, 4)
        rgb, alpha = self.ops._split_alpha(rgba)
        self.assertEqual(rgb.shape, (1, 20, 30, 3))
        self.assertEqual(alpha.shape, (1, 20, 30, 1))
        mask = self.ops._prepare_optional_mask_batch(torch.ones(1, 10, 15), 2, 20, 30)
        self.assertEqual(mask.shape, (2, 20, 30))
        resized, info = self.ops._downscale_hwc_long_side(rgb[0], "720p")
        self.assertEqual(resized.shape, rgb[0].shape)
        self.assertEqual(info["scale"], 1.0)

    def test_tone_palette_pipeline_is_bounded(self):
        source = torch.rand(24, 20, 3) * 0.5
        reference = torch.clamp(source * 1.25 + 0.1, 0.0, 1.0)
        exposure = self.ops._fit_exposure_gain(source, reference, None)
        tone = self.resolve._fit_tone_params(source, reference, "monotonic_spline")
        scale, offset = self.resolve._fit_palette_affine(source, reference, None, "lut3d")
        output = self.resolve._apply_resolve_pipeline_to_rgb(
            source,
            exposure_gain=exposure,
            exposure_alpha=0.5,
            tone_params=tone,
            tone_alpha=0.5,
            palette_scale=scale,
            palette_offset=offset,
            palette_alpha=0.5,
        )
        self.assertEqual(output.shape, source.shape)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_contract_json_and_cube(self):
        payload = {"schema_name": "test", "schema_version": 1}
        self.assertEqual(self.contract._safe_json_loads(json.dumps(payload)), payload)
        self.assertEqual(self.contract._safe_json_loads("not-json"), {})
        cube = self.contract._identity_cube_text(3)
        self.assertIn("LUT_3D_SIZE 3", cube)
        self.assertEqual(len(cube.splitlines()), 29)
        self.assertIn("required", self.contract._build_resolve_input_types())


if __name__ == "__main__":
    unittest.main()
