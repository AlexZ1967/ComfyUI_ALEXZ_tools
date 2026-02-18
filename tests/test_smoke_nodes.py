"""
Module: tests/test_smoke_nodes.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Smoke tests for node contracts and helper behavior.

Purpose:
    Checks that core node/helper interfaces keep stable output shapes and JSON payload structure.
"""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest

import torch


def _install_folder_paths_stub():
    """Internal helper: `_install_folder_paths_stub`."""
    if "folder_paths" in sys.modules:
        return
    stub = types.SimpleNamespace(
        get_input_directory=lambda: os.getcwd(),
        filter_files_content_types=lambda files, content_types: files,
        get_annotated_filepath=lambda name: name,
    )
    sys.modules["folder_paths"] = stub


class SmokeTests(unittest.TestCase):
    """Run minimal checks against key node helper behavior."""
    @classmethod
    def setUpClass(cls):
        """Execute `setUpClass` routine."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()

    def test_image_difference_autoresize(self):
        """Ensure image difference auto-resizes to the larger image shape."""
        utils_mod = importlib.import_module("ComfyUI_ALEXZ_tools.utils.utils")

        a = torch.rand(64, 64, 3)
        b = torch.rand(32, 48, 3)
        diff = utils_mod.image_difference(a, b)
        self.assertEqual(tuple(diff.shape), (64, 64, 3))

    def test_color_match_json_has_quality_metrics(self):
        """Verify Color Match returns quality metrics in output JSON."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")

        # Avoid LPIPS model download in smoke tests.
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 48, 48, 3)
            img = torch.rand(1, 48, 48, 3)
            matched, payload = node.match(ref, img, "linear", strength=0.8)
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        self.assertEqual(tuple(matched.shape), (1, 48, 48, 3))
        data = json.loads(payload[0])
        self.assertIn("quality", data)
        self.assertIn("before", data["quality"])
        self.assertIn("after", data["quality"])
        self.assertIn("improvement_pct", data["quality"])
        self.assertIn("mse", data["quality"]["before"])
        self.assertIn("ssim", data["quality"]["after"])

    def test_color_match_mask_white_region_used_for_adain(self):
        """Ensure match_mask white area drives stats for presets routed via color_match_utils."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")

        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            img = torch.zeros(1, 8, 8, 3)
            ref = torch.zeros(1, 8, 8, 3)
            img[:, :4, :4, :] = 0.2
            img[:, 4:, 4:, :] = 0.8
            ref[:, :4, :4, :] = 0.9
            ref[:, 4:, 4:, :] = 0.1
            match_mask = torch.zeros(1, 8, 8)
            match_mask[:, :4, :4] = 1.0
            matched, _payload = node.match(ref, img, "adain", match_mask=match_mask, strength=1.0)
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        top_left_mean = float(matched[0, :4, :4, :].mean().item())
        self.assertGreater(top_left_mean, 0.75)

    def test_color_match_batch_repeat_last_item(self):
        """Ensure smaller image batch is padded by repeating its last frame."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(2, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            out, _payload = node.match(ref, img, "linear", strength=1.0)
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        self.assertEqual(tuple(out.shape), (2, 16, 16, 3))

    def test_color_match_can_disable_quality_metrics(self):
        """Ensure quality metrics can be skipped for faster processing."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            _out, payload = node.match(ref, img, "oklab_cdf", compute_quality_metrics=False, strength=1.0)
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        data = json.loads(payload[0])
        self.assertIsNone(data["quality"]["before"]["mse"])
        self.assertIsNone(data["quality"]["after"]["ssim"])
        self.assertIsNone(data["quality"]["improvement_pct"]["delta_e76"])

    def test_color_match_tone_curve_runs(self):
        """Ensure tone_curve preset runs through batch path and keeps output shape."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(2, 16, 16, 3)
            img = torch.rand(2, 16, 16, 3)
            out, _payload = node.match(ref, img, "tone_curve", compute_quality_metrics=False, strength=1.0)
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        self.assertEqual(tuple(out.shape), (2, 16, 16, 3))

    def test_color_match_auto_optimal_runs(self):
        """Ensure auto_optimal preset selects a valid internal mode and returns JSON."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: 0.2
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            out, payload = node.match(
                ref,
                img,
                "auto_optimal",
                compute_quality_metrics=False,
                auto_optimal_metric="mse_ssim_lpips",
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        self.assertEqual(tuple(out.shape), (1, 16, 16, 3))
        data = json.loads(payload[0])
        self.assertTrue(str(data.get("mode", "")).startswith("auto_optimal:"))
        self.assertEqual(data.get("deep", {}).get("auto_optimal", {}).get("strategy"), "mse_ssim_lpips")

    def test_color_match_lut_export_creates_cube(self):
        """Ensure LUT export writes .cube file and path is returned in JSON."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            with tempfile.TemporaryDirectory() as td:
                _out, payload = node.match(
                    ref,
                    img,
                    "linear",
                    compute_quality_metrics=False,
                    export_lut=True,
                    lut_size=8,
                    lut_output_dir=td,
                    lut_name="smoke_lut",
                    strength=1.0,
                )
                data = json.loads(payload[0])
                lut = data.get("lut", {})
                self.assertTrue(bool(lut.get("exported")))
                lut_path = lut.get("path")
                self.assertTrue(isinstance(lut_path, str) and lut_path.endswith(".cube"))
                self.assertTrue(os.path.exists(lut_path))
        finally:
            image_color_match._lpips_alex_distance = old_lpips

    def test_video_frame_topk_helpers(self):
        """Validate top-k and confidence helper math for frame matching."""
        video_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_frame_match")

        top = []
        video_mod._update_top_matches(top, 0, 0.30, limit=3)
        video_mod._update_top_matches(top, 1, 0.10, limit=3)
        video_mod._update_top_matches(top, 2, 0.20, limit=3)
        video_mod._update_top_matches(top, 3, 0.15, limit=3)
        self.assertEqual([x["index"] for x in top], [1, 3, 2])
        conf = video_mod._confidence_from_top(top)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_video_cut_match_helpers(self):
        """Validate top-k pair ranking and blend suggestion helpers."""
        cut_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_cut_match")

        top_pairs = []
        cut_mod._update_top_pairs(top_pairs, {"frame_a_number": 10, "frame_b_number": 0, "score": 0.3}, 3)
        cut_mod._update_top_pairs(top_pairs, {"frame_a_number": 11, "frame_b_number": 1, "score": 0.1}, 3)
        cut_mod._update_top_pairs(top_pairs, {"frame_a_number": 12, "frame_b_number": 2, "score": 0.2}, 3)
        cut_mod._update_top_pairs(top_pairs, {"frame_a_number": 13, "frame_b_number": 3, "score": 0.15}, 3)
        self.assertEqual(top_pairs[0]["score"], 0.1)
        self.assertEqual(len(top_pairs), 3)

        conf = cut_mod._confidence_from_top_pairs(top_pairs)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        blend = cut_mod._blend_window_from_confidence(conf)
        self.assertIn(blend, [4, 8, 12])

    def test_qr_code_generation(self):
        """Verify QR node returns a square image tensor with requested resolution."""
        try:
            import qrcode  # noqa: F401
        except Exception:
            self.skipTest("qrcode is not installed in this environment")

        qr_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.qr_code_generate")
        node = qr_mod.GenerateQRCode()
        out, = node.generate("https://example.com", 256, "M")
        self.assertEqual(tuple(out.shape), (1, 256, 256, 3))

    def test_node_ui_metadata_compat(self):
        """Ensure loaded nodes expose metadata used by newer node-card UI."""
        nodes_pkg = importlib.import_module("ComfyUI_ALEXZ_tools.nodes")
        class_map = getattr(nodes_pkg, "NODE_CLASS_MAPPINGS", {})
        self.assertIn("GenerateQRCode", class_map)
        qr_cls = class_map["GenerateQRCode"]
        self.assertTrue(bool(getattr(qr_cls, "DESCRIPTION", "")))
        self.assertTrue(hasattr(qr_cls, "OUTPUT_TOOLTIPS"))
        self.assertTrue(hasattr(qr_cls, "SEARCH_ALIASES"))


if __name__ == "__main__":
    unittest.main()
