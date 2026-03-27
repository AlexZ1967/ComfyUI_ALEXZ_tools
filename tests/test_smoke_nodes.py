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
import re
import sys
import tempfile
import types
import unittest
from io import BytesIO

import numpy as np
import torch
from PIL import Image


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

    def test_color_match_experimental_presets_run(self):
        """Ensure new experimental color-transfer presets run without breaking output contract."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 20, 20, 3)
            img = torch.rand(1, 20, 20, 3)
            presets = (
                "reinhard_lab_fast",
                "hm",
                "mkl",
                "mvgd",
                "hm-mkl-hm",
                "hm-mvgd-hm",
            )
            for preset in presets:
                out, payload = node.match(
                    ref,
                    img,
                    preset,
                    compute_quality_metrics=False,
                    strength=1.0,
                )
                self.assertEqual(tuple(out.shape), (1, 20, 20, 3))
                data = json.loads(payload[0])
                self.assertEqual(str(data.get("preset", "")), preset)
        finally:
            image_color_match._lpips_alex_distance = old_lpips

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

    def test_color_match_empty_match_mask_returns_original(self):
        """Ensure empty match_mask returns original image for affected frames."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            empty_match = torch.zeros(1, 16, 16)
            out, payload = node.match(
                ref,
                img,
                "auto_optimal",
                match_mask=empty_match,
                compute_quality_metrics=False,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips

        self.assertTrue(torch.allclose(out, img, atol=1e-6))
        data = json.loads(payload[0])
        self.assertIn("empty_match_mask", str(data.get("mode", "")))

    def test_color_match_quality_metrics_mode_fast(self):
        """Ensure fast metrics mode reports MSE/SSIM but skips DeltaE/LPIPS."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: 0.123
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            _out, payload = node.match(
                ref,
                img,
                "linear",
                quality_metrics_mode="fast",
                compute_quality_metrics=True,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips
        data = json.loads(payload[0])
        before = data["quality"]["before"]
        self.assertIsNotNone(before["mse"])
        self.assertIsNotNone(before["ssim"])
        self.assertIsNone(before["delta_e76"])
        self.assertIsNone(before["lpips_alex"])

    def test_color_match_skin_tone_protection_reduces_shift(self):
        """Ensure skin protection keeps output closer to original for skin-like colors."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            img = torch.zeros(1, 24, 24, 3)
            img[:, :, :, 0] = 0.76
            img[:, :, :, 1] = 0.57
            img[:, :, :, 2] = 0.44
            ref = torch.zeros(1, 24, 24, 3)
            ref[:, :, :, 0] = 0.08
            ref[:, :, :, 1] = 0.22
            ref[:, :, :, 2] = 0.88
            out_no, _ = node.match(
                ref,
                img,
                "linear",
                compute_quality_metrics=False,
                skin_tone_protection=False,
                strength=1.0,
            )
            out_yes, _ = node.match(
                ref,
                img,
                "linear",
                compute_quality_metrics=False,
                skin_tone_protection=True,
                skin_protection_strength=1.0,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips
        diff_no = float(torch.mean(torch.abs(out_no - img)).item())
        diff_yes = float(torch.mean(torch.abs(out_yes - img)).item())
        self.assertLess(diff_yes, diff_no)

    def test_auto_optimal_temporal_stability_hysteresis(self):
        """Ensure temporal hysteresis can keep previous mode when score gain is small."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        old_linear = image_color_match._linear_match_batch
        old_oklab = image_color_match._oklab_cdf_match_batch
        old_candidate = image_color_match._auto_optimal_candidate_metrics
        try:
            image_color_match._lpips_alex_distance = lambda a, b: None

            def _fake_linear(img, ref, mask):
                out = img.clone()
                out[0] = 0.10
                out[1] = 0.90
                return out

            def _fake_oklab(img, ref, mask):
                out = img.clone()
                out[0] = 0.90
                out[1] = 0.20
                return out

            def _fake_candidate(candidate, ref, strategy):
                return {"mse": float(candidate.mean().item()), "ssim": None, "lpips_alex": None}

            image_color_match._linear_match_batch = _fake_linear
            image_color_match._oklab_cdf_match_batch = _fake_oklab
            image_color_match._auto_optimal_candidate_metrics = _fake_candidate

            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(2, 16, 16, 3)
            img = torch.rand(2, 16, 16, 3)
            _out, payload = node.match(
                ref,
                img,
                "auto_optimal",
                auto_optimal_metric="mse",
                auto_temporal_stability=True,
                auto_switch_threshold=1.0,
                compute_quality_metrics=False,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips
            image_color_match._linear_match_batch = old_linear
            image_color_match._oklab_cdf_match_batch = old_oklab
            image_color_match._auto_optimal_candidate_metrics = old_candidate

        modes = [json.loads(payload[0])["mode"], json.loads(payload[1])["mode"]]
        self.assertEqual(modes, ["auto_optimal:linear", "auto_optimal:linear"])

    def test_color_match_spatial_grid_mode(self):
        """Ensure spatial grid mode is applied for supported presets."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 18, 18, 3)
            img = torch.rand(1, 18, 18, 3)
            out, payload = node.match(
                ref,
                img,
                "linear",
                spatial_grid=2,
                compute_quality_metrics=False,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips
        self.assertEqual(tuple(out.shape), (1, 18, 18, 3))
        data = json.loads(payload[0])
        self.assertIn("grid2x2", str(data.get("mode", "")))
        self.assertTrue(bool(data.get("stats", {}).get("spatial_grid_applied")))

    def test_auto_optimal_quality_fallback_applies(self):
        """Ensure auto fallback can override selected auto_optimal candidate."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        old_linear = image_color_match._linear_match_batch
        old_oklab = image_color_match._oklab_cdf_match_batch
        old_lab = image_color_match._lab_cdf_match_batch
        old_candidate = image_color_match._auto_optimal_candidate_metrics
        try:
            image_color_match._lpips_alex_distance = lambda a, b: None
            image_color_match._linear_match_batch = lambda img, ref, mask: torch.zeros_like(img) + 0.8
            image_color_match._oklab_cdf_match_batch = lambda img, ref, mask: torch.zeros_like(img) + 0.7
            image_color_match._lab_cdf_match_batch = lambda img, ref, mask: torch.zeros_like(img) + 0.1
            image_color_match._auto_optimal_candidate_metrics = (
                lambda candidate, ref, strategy: {"mse": float(candidate.mean().item()), "ssim": None, "lpips_alex": None}
            )

            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            _out, payload = node.match(
                ref,
                img,
                "auto_optimal",
                auto_optimal_metric="mse",
                auto_quality_fallback=True,
                auto_fallback_method="lab_cdf",
                auto_fallback_threshold=0.0,
                auto_fallback_margin=0.0,
                compute_quality_metrics=False,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips
            image_color_match._linear_match_batch = old_linear
            image_color_match._oklab_cdf_match_batch = old_oklab
            image_color_match._lab_cdf_match_batch = old_lab
            image_color_match._auto_optimal_candidate_metrics = old_candidate

        data = json.loads(payload[0])
        self.assertIn("fallback:lab_cdf", str(data.get("mode", "")))
        self.assertTrue(bool(data.get("deep", {}).get("auto_optimal", {}).get("fallback_applied")))

    def test_color_match_delta_e76_torch_without_cv2(self):
        """Ensure full metrics still report delta_e76 when cv2 is unavailable."""
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_color_match")
        old_lpips = image_color_match._lpips_alex_distance
        old_cv2 = image_color_match.color_match_utils.cv2
        image_color_match._lpips_alex_distance = lambda a, b: None
        image_color_match.color_match_utils.cv2 = None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
            _out, payload = node.match(
                ref,
                img,
                "linear",
                quality_metrics_mode="full",
                compute_quality_metrics=True,
                strength=1.0,
            )
        finally:
            image_color_match._lpips_alex_distance = old_lpips
            image_color_match.color_match_utils.cv2 = old_cv2
        data = json.loads(payload[0])
        self.assertIsNotNone(data["quality"]["before"]["delta_e76"])

    def test_seam_match_node_runs_and_reports_json(self):
        """Ensure seam-match node runs and returns optimization diagnostics."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchToReference()
        ref = torch.rand(1, 20, 20, 3)
        img = torch.rand(1, 20, 20, 3)
        out, payload = node.match(
            ref,
            img,
            strength=1.0,
            color_space="oklab",
            downscale_long_side="720p",
            steps=2,
            lr=0.05,
        )
        self.assertEqual(tuple(out.shape), (1, 20, 20, 3))
        data = json.loads(payload[0])
        self.assertEqual(data.get("mode"), "seam_match:oklab")
        self.assertEqual(data.get("optimization", {}).get("downscale_long_side"), "720p")
        self.assertIn(data.get("optimization", {}).get("seam_model"), ("v1_affine", "v2_tonal", "v3_hybrid", "v4_lut"))
        self.assertIn("matrix", data.get("transform", {}))

    def test_seam_match_downscale_options(self):
        """Ensure all declared downscale options are accepted."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchToReference()
        ref = torch.rand(1, 16, 16, 3)
        img = torch.rand(1, 16, 16, 3)
        for mode in ("as_is", "1080p", "720p", "480p"):
            _out, payload = node.match(
                ref,
                img,
                strength=1.0,
                downscale_long_side=mode,
                steps=1,
                lr=0.03,
            )
            data = json.loads(payload[0])
            self.assertEqual(data.get("optimization", {}).get("downscale_long_side"), mode)

    def test_seam_match_model_options(self):
        """Ensure seam model selector works for v1/v2 modes."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchToReference()
        ref = torch.rand(1, 16, 16, 3)
        img = torch.rand(1, 16, 16, 3)
        for model in ("v1_affine", "v2_tonal", "v3_hybrid", "v4_lut"):
            _out, payload = node.match(
                ref,
                img,
                strength=1.0,
                seam_model=model,
                downscale_long_side="as_is",
                steps=1,
                lr=0.03,
            )
            data = json.loads(payload[0])
            self.assertEqual(data.get("optimization", {}).get("seam_model"), model)
            self.assertIn("matrix", data.get("transform", {}))
            if model == "v2_tonal":
                self.assertIn("tonal_bands", data.get("transform", {}))
            if model == "v3_hybrid":
                self.assertIn("hybrid", data.get("transform", {}))
            if model == "v4_lut":
                self.assertIn("lut", data.get("transform", {}))

    def test_seam_match_variant_nodes_and_inputs(self):
        """Ensure dedicated seam v1/v2/v3/v4 nodes run and expose compact inputs."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        ref = torch.rand(1, 16, 16, 3)
        img = torch.rand(1, 16, 16, 3)
        variants = (
            ("ImageSeamMatchV1AffineToReference", "v1_affine"),
            ("ImageSeamMatchV2TonalToReference", "v2_tonal"),
            ("ImageSeamMatchV3HybridToReference", "v3_hybrid"),
            ("ImageSeamMatchV4LUTToReference", "v4_lut"),
        )
        for class_name, expected_mode in variants:
            node = getattr(seam_mod, class_name)()
            out, payload = node.match(
                ref,
                img,
                strength=1.0,
                downscale_long_side="as_is",
                steps=1,
                lr=0.03,
            )
            self.assertEqual(tuple(out.shape), (1, 16, 16, 3))
            data = json.loads(payload[0])
            self.assertEqual(data.get("optimization", {}).get("seam_model"), expected_mode)

        # preserve_alpha must be removed from all seam nodes.
        all_classes = [
            seam_mod.ImageSeamMatchToReference,
            seam_mod.ImageSeamMatchV1AffineToReference,
            seam_mod.ImageSeamMatchV2TonalToReference,
            seam_mod.ImageSeamMatchV3HybridToReference,
            seam_mod.ImageSeamMatchV4LUTToReference,
        ]
        for cls in all_classes:
            optional = cls.INPUT_TYPES().get("optional", {})
            self.assertNotIn("preserve_alpha", optional)

        # Dedicated variant nodes should not expose seam_model selector.
        self.assertNotIn("seam_model", seam_mod.ImageSeamMatchV1AffineToReference.INPUT_TYPES().get("optional", {}))
        self.assertNotIn("seam_model", seam_mod.ImageSeamMatchV2TonalToReference.INPUT_TYPES().get("optional", {}))
        self.assertNotIn("seam_model", seam_mod.ImageSeamMatchV3HybridToReference.INPUT_TYPES().get("optional", {}))
        self.assertNotIn("seam_model", seam_mod.ImageSeamMatchV4LUTToReference.INPUT_TYPES().get("optional", {}))

    def test_seam_match_always_preserves_alpha_when_present(self):
        """Ensure RGBA alpha channel is preserved automatically."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchV2TonalToReference()
        ref = torch.rand(1, 16, 16, 4)
        img = torch.rand(1, 16, 16, 4)
        out, _payload = node.match(
            ref,
            img,
            strength=1.0,
            downscale_long_side="as_is",
            steps=1,
            lr=0.03,
        )
        self.assertEqual(tuple(out.shape), (1, 16, 16, 4))

    def test_seam_match_compute_device_cpu(self):
        """Ensure explicit CPU compute_device is accepted and reported."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchToReference()
        ref = torch.rand(1, 16, 16, 3)
        img = torch.rand(1, 16, 16, 3)
        _out, payload = node.match(
            ref,
            img,
            compute_device="cpu",
            downscale_long_side="as_is",
            steps=1,
            lr=0.03,
        )
        data = json.loads(payload[0])
        self.assertEqual(data.get("optimization", {}).get("compute_device_effective"), "cpu")

    def test_seam_match_compute_device_cuda_fallback(self):
        """Ensure CUDA request falls back safely when CUDA is unavailable."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchToReference()
        ref = torch.rand(1, 16, 16, 3)
        img = torch.rand(1, 16, 16, 3)
        _out, payload = node.match(
            ref,
            img,
            compute_device="cuda",
            downscale_long_side="as_is",
            steps=1,
            lr=0.03,
        )
        data = json.loads(payload[0])
        effective = data.get("optimization", {}).get("compute_device_effective")
        if torch.cuda.is_available():
            self.assertEqual(effective, "cuda")
        else:
            self.assertEqual(effective, "cpu")
            self.assertEqual(
                data.get("optimization", {}).get("device_warning"),
                "cuda_requested_but_unavailable",
            )

    def test_seam_match_accepts_inference_tensors(self):
        """Ensure seam-match can optimize when inputs come from inference mode."""
        seam_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_seam_match")
        node = seam_mod.ImageSeamMatchToReference()
        with torch.inference_mode():
            ref = torch.rand(1, 16, 16, 3)
            img = torch.rand(1, 16, 16, 3)
        out, payload = node.match(
            ref,
            img,
            strength=1.0,
            downscale_long_side="as_is",
            steps=1,
            lr=0.03,
        )
        self.assertEqual(tuple(out.shape), (1, 16, 16, 3))
        self.assertEqual(json.loads(payload[0]).get("status"), "ok")

    def test_look_match_resolve_contract_and_alpha(self):
        """Ensure resolve look-match node returns contract JSON and preserves RGBA alpha."""
        look_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_look_match")
        node = look_mod.ImageLookMatchResolve()
        ref = torch.rand(1, 20, 20, 3)
        img = torch.rand(1, 20, 20, 4)
        out, look_json, cube_text = node.match(
            ref,
            img,
            strength=1.0,
            compute_device="cpu",
            export_lut_cube=True,
            lut_size=17,
        )
        self.assertEqual(tuple(out.shape), (1, 20, 20, 4))
        self.assertIn("LUT_3D_SIZE 17", cube_text[0])
        data = json.loads(look_json[0])
        self.assertEqual(data.get("schema_name"), "alexz.look_match.resolve")
        self.assertEqual(data.get("schema_version"), 1)
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("phase"), "B_resolve_mvp")

    def test_look_match_resolve_moves_toward_reference(self):
        """Ensure resolve pipeline improves MSE on a simple tonal/color shift case."""
        look_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_look_match")
        node = look_mod.ImageLookMatchResolve()
        ref = torch.full((1, 24, 24, 3), 0.8, dtype=torch.float32)
        img = torch.full((1, 24, 24, 3), 0.2, dtype=torch.float32)
        out, _look_json, _cube_text = node.match(
            ref,
            img,
            strength=1.0,
            compute_device="cpu",
            downscale_long_side="as_is",
            w_exposure=1.0,
            w_tone=1.0,
            w_chroma=1.0,
            export_lut_cube=False,
        )
        mse_before = float(((img - ref) ** 2).mean().item())
        mse_after = float(((out - ref) ** 2).mean().item())
        self.assertLess(mse_after, mse_before)

    def test_look_match_nuke_build_apply_contract(self):
        """Ensure build/apply look-match nodes share stable schema contracts."""
        look_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_look_match")
        build_node = look_mod.ImageLookMatchNukeBuild()
        apply_node = look_mod.ImageLookMatchNukeApply()

        ref = torch.rand(1, 16, 16, 3)
        src = torch.rand(1, 16, 16, 3)
        model_json, cube_text = build_node.build(
            ref,
            src,
            compute_device="cpu",
            export_lut_cube=False,
        )
        self.assertEqual(cube_text, "")
        model_data = json.loads(model_json)
        self.assertEqual(model_data.get("schema_name"), "alexz.look_model.nuke_build")
        self.assertEqual(model_data.get("schema_version"), 1)

        img = torch.rand(1, 16, 16, 4)
        out, apply_json = apply_node.apply(
            img,
            model_json,
            strength=1.0,
            compute_device="cpu",
        )
        self.assertEqual(tuple(out.shape), (1, 16, 16, 4))
        apply_data = json.loads(apply_json[0])
        self.assertEqual(apply_data.get("schema_name"), "alexz.look_apply.nuke_apply")
        self.assertEqual(apply_data.get("schema_version"), 1)
        self.assertEqual(apply_data.get("status"), "ok")

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

    def test_dzi_tiles_download_assembly_mocked(self):
        """Verify DZI tiles node assembles a simple 2x2 tile grid without network."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()

        tile_size = 8
        colors = {
            (0, 0): (255, 0, 0),
            (1, 0): (0, 255, 0),
            (0, 1): (0, 0, 255),
            (1, 1): (255, 255, 0),
        }

        class _DummySession:
            pass

        old_new_session = dzi_mod._new_session
        old_parse_dzi = dzi_mod._parse_dzi
        old_probe_axis = dzi_mod._probe_axis_count
        old_download_tile = dzi_mod._download_tile

        def _parse_xy(url: str):
            match = re.search(r"/(\d+)_(\d+)\.jpg$", str(url))
            if not match:
                return None
            return (int(match.group(1)), int(match.group(2)))

        try:
            from PIL import Image
            import numpy as np

            def _fake_download_tile(_session, url: str, _timeout: float):
                coords = _parse_xy(url)
                if coords is None or coords not in colors:
                    return None
                canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                canvas[:, :, 0] = colors[coords][0]
                canvas[:, :, 1] = colors[coords][1]
                canvas[:, :, 2] = colors[coords][2]
                return Image.fromarray(canvas, mode="RGB")

            dzi_mod._new_session = lambda: _DummySession()
            dzi_mod._parse_dzi = lambda *_args, **_kwargs: None
            dzi_mod._probe_axis_count = (
                lambda _session, _base, *, axis, timeout, max_tiles=4096: 2
                if axis in {"x", "y"}
                else 0
            )
            dzi_mod._download_tile = _fake_download_tile

            out, = node.download("https://example.test/zoom", "mwX", 11)
        finally:
            dzi_mod._new_session = old_new_session
            dzi_mod._parse_dzi = old_parse_dzi
            dzi_mod._probe_axis_count = old_probe_axis
            dzi_mod._download_tile = old_download_tile

        self.assertEqual(tuple(out.shape), (1, tile_size * 2, tile_size * 2, 3))
        self.assertAlmostEqual(float(out[0, 1, 1, 0].item()), 1.0, places=4)  # top-left red
        self.assertAlmostEqual(float(out[0, 1, 1, 1].item()), 0.0, places=4)
        self.assertAlmostEqual(float(out[0, 1, tile_size + 1, 1].item()), 1.0, places=4)  # top-right green
        self.assertAlmostEqual(float(out[0, tile_size + 1, 1, 2].item()), 1.0, places=4)  # bottom-left blue

    def test_dzi_tiles_build_zoom_base_url(self):
        """Verify base URL normalization auto-adds /zoom and keeps backward compatibility."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        self.assertEqual(
            dzi_mod._build_zoom_base_url("https://collectionimages.npg.org.uk", "mw207134"),
            "https://collectionimages.npg.org.uk/zoom/mw207134",
        )
        self.assertEqual(
            dzi_mod._build_zoom_base_url("https://collectionimages.npg.org.uk/zoom", "mw207134"),
            "https://collectionimages.npg.org.uk/zoom/mw207134",
        )
        self.assertEqual(
            dzi_mod._build_zoom_base_url("https://collectionimages.npg.org.uk/zoom/mw207134", "mw207134"),
            "https://collectionimages.npg.org.uk/zoom/mw207134",
        )

    def test_dzi_tiles_build_source_urls_nla(self):
        """Verify NLA provider uses query-based DZI/tile URLs."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        source = dzi_mod._build_dzi_source_urls(
            "https://nla.gov.au",
            "nla.obj-138204672",
            11,
            "auto",
        )
        self.assertEqual(source["provider"], "nla")
        self.assertEqual(source["dzi_url"], "https://nla.gov.au/nla.obj-138204672/dzi?tile=")
        self.assertEqual(source["tiles_base"], "https://nla.gov.au/nla.obj-138204672/dzi?tile=")
        self.assertEqual(
            dzi_mod._tile_url(
                source["tiles_base"],
                3,
                4,
                "jpg",
                level=11,
                mode=source["tile_url_mode"],
            ),
            "https://nla.gov.au/nla.obj-138204672/dzi?tile=11/3_4.jpg",
        )

    def test_dzi_tiles_build_source_urls_from_templates(self):
        """Verify config templates can define provider URLs without hardcoded branches."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        source = dzi_mod._build_dzi_source_urls(
            "https://example.org",
            "obj-42",
            9,
            "my_archive",
            site_config={
                "provider": "my_archive",
                "base_url": "https://example.org",
                "object_url_template": "{base_url}/viewer/{mw}",
                "dzi_url_template": "{base_url}/iiif/{mw}/info.dzi",
                "tile_url_template": "{base_url}/iiif/{mw}/{level}/{x}-{y}.{ext}",
            },
        )
        self.assertEqual(source["provider"], "my_archive")
        self.assertEqual(source["zoom_base"], "https://example.org/viewer/obj-42")
        self.assertEqual(source["dzi_url"], "https://example.org/iiif/obj-42/info.dzi")
        self.assertEqual(
            dzi_mod._tile_url(
                source["tiles_base"],
                3,
                4,
                "png",
                level=9,
                mode=source["tile_url_mode"],
                base_url="https://example.org",
                mw="obj-42",
            ),
            "https://example.org/iiif/obj-42/9/3-4.png",
        )

    def test_dzi_tiles_site_config_dropdown(self):
        """Verify DZI site dropdown is populated from JSON config."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        input_types = dzi_mod.ImageDownloadDZITiles.INPUT_TYPES()
        site_choices = input_types["required"]["site"][0]
        self.assertIn("National Portrait Gallery UK", site_choices)
        self.assertIn("National Library of Australia", site_choices)
        resolved = dzi_mod._resolve_dzi_site("National Library of Australia", "")
        self.assertEqual(resolved["base_url"], "https://nla.gov.au")
        self.assertEqual(resolved["provider"], "nla")
        self.assertEqual(dzi_mod._normalize_site_mw("138204672", resolved), "nla.obj-138204672")
        self.assertEqual(
            dzi_mod._normalize_site_mw("nla.obj-138204672", resolved),
            "nla.obj-138204672",
        )

    def test_dzi_tiles_normalize_numeric_mw_for_npg(self):
        """Verify digits-only mw input is expanded with site prefix."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        resolved = dzi_mod._resolve_dzi_site("National Portrait Gallery UK", "")
        self.assertEqual(dzi_mod._normalize_site_mw("207134", resolved), "mw207134")
        self.assertEqual(dzi_mod._normalize_site_mw("mw207134", resolved), "mw207134")

    def test_dzi_tiles_download_assembly_mocked_nla_provider(self):
        """Verify DZI node supports NLA query tile scheme without network."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()

        tile_size = 8
        requested_urls = []

        class _DummySession:
            pass

        old_new_session = dzi_mod._new_session
        old_fetch_bytes = dzi_mod._fetch_bytes
        old_parse_dzi = dzi_mod._parse_dzi
        old_download_tile = dzi_mod._download_tile

        try:
            from PIL import Image
            import numpy as np

            canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            canvas[:, :, 1] = 255
            image = Image.fromarray(canvas, mode="RGB")
            import io
            encoded = io.BytesIO()
            image.save(encoded, format="JPEG")
            jpeg_bytes = encoded.getvalue()

            def _fake_fetch_bytes(_session, url: str, _timeout: float, *, transport: str = "requests"):
                if "tile=11/0_0.jpg" in str(url):
                    return 200, jpeg_bytes
                return 404, None

            def _fake_download_tile(_session, url: str, _timeout: float):
                requested_urls.append(str(url))
                if "tile=11/0_0.jpg" not in str(url):
                    return None
                return image.copy()

            dzi_mod._new_session = lambda: _DummySession()
            dzi_mod._fetch_bytes = _fake_fetch_bytes
            dzi_mod._parse_dzi = lambda *_args, **_kwargs: {
                "tile_size": tile_size,
                "overlap": 0,
                "format": "jpg",
                "width": tile_size,
                "height": tile_size,
            }
            dzi_mod._download_tile = _fake_download_tile

            out, = node.download(
                "National Library of Australia",
                "138204672",
                11,
            )
        finally:
            dzi_mod._new_session = old_new_session
            dzi_mod._fetch_bytes = old_fetch_bytes
            dzi_mod._parse_dzi = old_parse_dzi
            dzi_mod._download_tile = old_download_tile

        self.assertEqual(tuple(out.shape), (1, tile_size, tile_size, 3))
        self.assertAlmostEqual(float(out[0, 1, 1, 1].item()), 1.0, places=4)

    def test_dzi_tiles_download_prefers_dzi_level_geometry(self):
        """Verify DZI metadata controls grid/size when probe reports misleading larger axes."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()

        tile_size = 8
        requested = []

        class _DummySession:
            pass

        old_new_session = dzi_mod._new_session
        old_parse_dzi = dzi_mod._parse_dzi
        old_probe_axis = dzi_mod._probe_axis_count
        old_download_tile = dzi_mod._download_tile

        def _parse_xy(url: str):
            match = re.search(r"/(\d+)_(\d+)\.jpg$", str(url))
            if not match:
                return None
            return (int(match.group(1)), int(match.group(2)))

        try:
            from PIL import Image
            import numpy as np

            def _fake_download_tile(_session, url: str, _timeout: float):
                coords = _parse_xy(url)
                if coords is not None:
                    requested.append(coords)
                canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                canvas[:, :, 0] = 255
                return Image.fromarray(canvas, mode="RGB")

            dzi_mod._new_session = lambda: _DummySession()
            # Full size 10x16, level 3 => 5x8, grid must be 1x1 (max_level=4).
            dzi_mod._parse_dzi = lambda *_args, **_kwargs: {
                "tile_size": tile_size,
                "overlap": 0,
                "format": "jpg",
                "width": 10,
                "height": 16,
            }
            # Intentionally misleading probe to ensure DZI path is used.
            dzi_mod._probe_axis_count = lambda *_args, **_kwargs: 6
            dzi_mod._download_tile = _fake_download_tile

            out, = node.download("https://example.test/zoom", "mwX", 3)
        finally:
            dzi_mod._new_session = old_new_session
            dzi_mod._parse_dzi = old_parse_dzi
            dzi_mod._probe_axis_count = old_probe_axis
            dzi_mod._download_tile = old_download_tile

        self.assertEqual(tuple(out.shape), (1, 8, 5, 3))
        self.assertEqual(set(requested), {(0, 0)})

    def test_dzi_tiles_single_writes_output_file_when_output_dir_is_set(self):
        """Verify single DZI node optionally saves the assembled image to disk."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()

        tile_size = 8

        class _DummySession:
            pass

        old_new_session = dzi_mod._new_session
        old_parse_dzi = dzi_mod._parse_dzi
        old_download_tile = dzi_mod._download_tile

        try:
            from PIL import Image
            import numpy as np

            def _fake_download_tile(_session, _url: str, _timeout: float):
                canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                canvas[:, :, 2] = 255
                return Image.fromarray(canvas, mode="RGB")

            dzi_mod._new_session = lambda: _DummySession()
            dzi_mod._parse_dzi = lambda *_args, **_kwargs: {
                "tile_size": tile_size,
                "overlap": 0,
                "format": "jpg",
                "width": tile_size,
                "height": tile_size,
            }
            dzi_mod._download_tile = _fake_download_tile

            with tempfile.TemporaryDirectory() as tmpdir:
                out, = node.download(
                    "National Portrait Gallery UK",
                    "207134",
                    11,
                    output_dir=tmpdir,
                    output_extension="png",
                )
                saved_path = os.path.join(tmpdir, "mw207134.png")
                self.assertTrue(os.path.exists(saved_path))
                self.assertEqual(tuple(out.shape), (1, tile_size, tile_size, 3))
        finally:
            dzi_mod._new_session = old_new_session
            dzi_mod._parse_dzi = old_parse_dzi
            dzi_mod._download_tile = old_download_tile

    def test_dzi_tiles_single_title_or_mw_uses_title_when_available(self):
        """Verify single DZI saver can name output file from resolved object title."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()

        tile_size = 8

        class _DummySession:
            pass

        old_new_session = dzi_mod._new_session
        old_parse_dzi = dzi_mod._parse_dzi
        old_download_tile = dzi_mod._download_tile
        old_fetch_title = dzi_mod._fetch_dzi_object_title

        try:
            from PIL import Image
            import numpy as np

            def _fake_download_tile(_session, _url: str, _timeout: float):
                canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                canvas[:, :, 0] = 255
                return Image.fromarray(canvas, mode="RGB")

            dzi_mod._new_session = lambda: _DummySession()
            dzi_mod._parse_dzi = lambda *_args, **_kwargs: {
                "tile_size": tile_size,
                "overlap": 0,
                "format": "jpg",
                "width": tile_size,
                "height": tile_size,
            }
            dzi_mod._download_tile = _fake_download_tile
            dzi_mod._fetch_dzi_object_title = lambda *_args, **_kwargs: "Anna_Pavlova_as_the_Dying_swan_Melbourne_1926"

            with tempfile.TemporaryDirectory() as tmpdir:
                _out, = node.download(
                    "National Library of Australia",
                    "138204672",
                    11,
                    output_dir=tmpdir,
                    output_extension="png",
                    filename_mode="title_or_mw",
                )
                saved_path = os.path.join(tmpdir, "Anna_Pavlova_as_the_Dying_swan_Melbourne_1926_nla.obj-138204672.png")
                self.assertTrue(os.path.exists(saved_path))
        finally:
            dzi_mod._new_session = old_new_session
            dzi_mod._parse_dzi = old_parse_dzi
            dzi_mod._download_tile = old_download_tile
            dzi_mod._fetch_dzi_object_title = old_fetch_title

    def test_dzi_tiles_single_save_adds_numeric_suffix_when_name_exists(self):
        """Verify single DZI saver avoids overwrite by appending numeric suffix."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()

        tile_size = 8

        class _DummySession:
            pass

        old_new_session = dzi_mod._new_session
        old_parse_dzi = dzi_mod._parse_dzi
        old_download_tile = dzi_mod._download_tile

        try:
            from PIL import Image
            import numpy as np

            def _fake_download_tile(_session, _url: str, _timeout: float):
                canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                canvas[:, :, 1] = 255
                return Image.fromarray(canvas, mode="RGB")

            dzi_mod._new_session = lambda: _DummySession()
            dzi_mod._parse_dzi = lambda *_args, **_kwargs: {
                "tile_size": tile_size,
                "overlap": 0,
                "format": "jpg",
                "width": tile_size,
                "height": tile_size,
            }
            dzi_mod._download_tile = _fake_download_tile

            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = os.path.join(tmpdir, "mw207134.png")
                with open(base_path, "wb") as fh:
                    fh.write(b"existing")
                _out, = node.download(
                    "National Portrait Gallery UK",
                    "207134",
                    11,
                    output_dir=tmpdir,
                    output_extension="png",
                )
                suffixed_path = os.path.join(tmpdir, "mw207134_2.png")
                self.assertTrue(os.path.exists(base_path))
                self.assertTrue(os.path.exists(suffixed_path))
        finally:
            dzi_mod._new_session = old_new_session
            dzi_mod._parse_dzi = old_parse_dzi
            dzi_mod._download_tile = old_download_tile

    def test_node_ui_metadata_compat(self):
        """Ensure loaded nodes expose metadata used by newer node-card UI."""
        nodes_pkg = importlib.import_module("ComfyUI_ALEXZ_tools.nodes")
        class_map = getattr(nodes_pkg, "NODE_CLASS_MAPPINGS", {})
        self.assertIn("GenerateQRCode", class_map)
        self.assertIn("ImageDownloadDZITiles", class_map)
        self.assertIn("ImageDownloadDZITilesBatchSave", class_map)
        self.assertIn("ImageDownloadIIIFImage", class_map)
        self.assertIn("ImageDescreenAdaptiveScale", class_map)
        self.assertIn("ImageDescreenApplyPercent", class_map)
        self.assertIn("SearchTroveImageIDs", class_map)
        qr_cls = class_map["GenerateQRCode"]
        self.assertTrue(bool(getattr(qr_cls, "DESCRIPTION", "")))
        self.assertTrue(hasattr(qr_cls, "OUTPUT_TOOLTIPS"))
        self.assertTrue(hasattr(qr_cls, "SEARCH_ALIASES"))

    def test_trove_search_ids_node_contract(self):
        """Verify Trove search node returns newline-separated ids and diagnostic JSON."""
        trove_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.trove_search_ids")
        node = trove_mod.SearchTroveImageIDs()
        old_search = trove_mod._search_trove_ids_via_chrome
        try:
            trove_mod._search_trove_ids_via_chrome = lambda *args, **kwargs: {
                "query": "Pavlova",
                "category": "images",
                "search_url": "https://trove.nla.gov.au/search/category/images?keyword=Pavlova",
                "chrome_path": "/usr/bin/google-chrome",
                "returncode": 0,
                "count": 3,
                "ids": [
                    "nla.obj-138204672",
                    "nla.obj-162204874",
                    "nla.obj-150139367",
                ],
                "warning": "",
                "stdout_excerpt": "",
                "stderr_excerpt": "",
            }
            ids_text, result_json, count = node.search("Pavlova")
        finally:
            trove_mod._search_trove_ids_via_chrome = old_search

        payload = json.loads(result_json)
        self.assertEqual(count, 3)
        self.assertIn("nla.obj-138204672", ids_text)
        self.assertEqual(payload["count"], 3)
        self.assertEqual(payload["category"], "images")

    def test_dzi_tiles_batch_save_writes_files_and_manifest(self):
        """Verify batch DZI saver writes output files and returns manifest JSON."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITilesBatchSave()

        old_download = dzi_mod.ImageDownloadDZITiles.download
        old_fetch_title = dzi_mod._fetch_dzi_object_title

        try:
            def _fake_download(_self, site, mw, level, transport="auto", proxy_url="", tile_extension="jpg"):
                _ = (site, level, transport, proxy_url, tile_extension)
                value = 0.25 if str(mw).endswith("134") else 0.75
                image = torch.full((1, 4, 4, 3), float(value), dtype=torch.float32)
                return (image,)

            dzi_mod.ImageDownloadDZITiles.download = _fake_download
            dzi_mod._fetch_dzi_object_title = lambda *_args, **_kwargs: "Anna_Pavlova"

            with tempfile.TemporaryDirectory() as tmpdir:
                manifest_json, saved_paths_json, count_ok, count_failed = node.download_batch(
                    "National Portrait Gallery UK",
                    "207134\n207135",
                    tmpdir,
                    -1,
                    output_extension="png",
                    filename_template="{title}_{mw}",
                    overwrite_mode="skip",
                    continue_on_error="true",
                    save_mode="save_and_manifest",
                )

                manifest = json.loads(manifest_json)
                saved_paths = json.loads(saved_paths_json)
                self.assertEqual(count_ok, 2)
                self.assertEqual(count_failed, 0)
                self.assertEqual(len(saved_paths), 2)
                self.assertEqual(manifest["count_ok"], 2)
                self.assertTrue(any(path.endswith("Anna_Pavlova_mw207134.png") for path in saved_paths))
                self.assertTrue(any(path.endswith("Anna_Pavlova_mw207135.png") for path in saved_paths))
                for path in saved_paths:
                    self.assertTrue(os.path.exists(path))
                manifest_path = os.path.join(tmpdir, "dzi_batch_manifest_001.json")
                if not os.path.exists(manifest_path):
                    manifest_path = os.path.join(tmpdir, "dzi_batch_manifest.json")
                self.assertTrue(os.path.exists(manifest_path))
        finally:
            dzi_mod.ImageDownloadDZITiles.download = old_download
            dzi_mod._fetch_dzi_object_title = old_fetch_title

    def test_dzi_tiles_single_respects_interrupt(self):
        """Verify single DZI node propagates Comfy interrupt requests."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITiles()
        old_check_interrupt = dzi_mod.check_interrupt

        class InterruptProcessingException(Exception):
            pass

        try:
            dzi_mod.check_interrupt = lambda: (_ for _ in ()).throw(InterruptProcessingException())
            with self.assertRaises(InterruptProcessingException):
                node.download("National Portrait Gallery UK", "207134", -1)
        finally:
            dzi_mod.check_interrupt = old_check_interrupt

    def test_dzi_tiles_batch_propagates_interrupt(self):
        """Verify batch DZI node does not swallow Comfy interrupt as per-item failure."""
        dzi_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_dzi_tiles")
        node = dzi_mod.ImageDownloadDZITilesBatchSave()
        old_download = dzi_mod.ImageDownloadDZITiles.download

        class InterruptProcessingException(Exception):
            pass

        try:
            def _interrupt_download(_self, *args, **kwargs):
                raise InterruptProcessingException()

            dzi_mod.ImageDownloadDZITiles.download = _interrupt_download
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(InterruptProcessingException):
                    node.download_batch(
                        "National Portrait Gallery UK",
                        "207134\n207135",
                        tmpdir,
                        -1,
                        continue_on_error="true",
                    )
        finally:
            dzi_mod.ImageDownloadDZITiles.download = old_download

    def test_iiif_download_node_contract(self):
        """Verify IIIF node returns IMAGE tensor and JSON contract."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_download = iiif_mod._download_iiif_image_bytes

        try:
            def _fake_resolve(site, source_url, timeout=30.0, session=None):
                _ = (site, source_url, timeout, session)
                return "https://collections.example.test/iiif/3/sample.ptif"

            def _fake_info(service_url, timeout=30.0, session=None):
                _ = (service_url, timeout, session)
                return {
                    "id": "https://collections.example.test/iiif/3/sample.ptif",
                    "type": "ImageService3",
                    "profile": "level2",
                    "width": 16,
                    "height": 12,
                    "tiles": [{"width": 512, "height": 512, "scaleFactors": [1, 2, 4]}],
                    "sizes": [{"width": 800, "height": 600}],
                }

            def _fake_download(service_url, size_spec, output_format, timeout=30.0, session=None):
                _ = (service_url, size_spec, output_format, timeout, session)
                image = Image.new("RGB", (16, 12), color=(64, 128, 192))
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                return (
                    "https://collections.example.test/iiif/3/sample.ptif/full/max/0/default.jpg",
                    buffer.getvalue(),
                    "jpg",
                )

            iiif_mod._resolve_iiif_service_url = _fake_resolve
            iiif_mod._fetch_iiif_info = _fake_info
            iiif_mod._download_iiif_image_bytes = _fake_download

            out, info_json = node.download(
                "London Museum Object Page",
                "https://www.londonmuseum.org.uk/collections/v/object-443296/early-portrait-of-anna-pavlova/",
            )
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._download_iiif_image_bytes = old_download

        payload = json.loads(info_json)
        self.assertEqual(tuple(out.shape), (1, 12, 16, 3))
        self.assertEqual(payload["site"], "London Museum Object Page")
        self.assertEqual(payload["iiif"]["type"], "ImageService3")
        self.assertEqual(payload["downloaded"]["width"], 16)
        self.assertEqual(payload["source"]["width"], 16)
        self.assertFalse(payload["limits"]["limited_by_service"])
        self.assertEqual(payload["delivery"]["mode"], "single_request")
        self.assertTrue(str(payload["service_url"]).startswith("https://collections.example.test/iiif/"))

    def test_iiif_download_tile_assembly_contract(self):
        """Verify IIIF node can return full-res tile-assembled image contract."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_assemble = iiif_mod._assemble_iiif_full_image

        try:
            iiif_mod._resolve_iiif_service_url = lambda *args, **kwargs: "https://collections.example.test/iiif/3/sample.ptif"
            iiif_mod._fetch_iiif_info = lambda *args, **kwargs: {
                "id": "https://collections.example.test/iiif/3/sample.ptif",
                "type": "ImageService3",
                "profile": "level2",
                "width": 1600,
                "height": 1200,
                "maxArea": 1000000,
                "tiles": [{"width": 512, "height": 512, "scaleFactors": [1, 2, 4]}],
                "sizes": [{"width": 800, "height": 600}],
            }

            def _fake_assemble(service_url, info, output_format="jpg", timeout=30.0, session=None, cache_dir=None):
                _ = (service_url, info, output_format, timeout, session, cache_dir)
                image = Image.new("RGB", (1600, 1200), color=(32, 64, 96))
                return image, {
                    "mode": "tile_assemble_full",
                    "tile_width": 512,
                    "tile_height": 512,
                    "tiles_x": 4,
                    "tiles_y": 3,
                    "tiles_total": 12,
                    "tiles_downloaded": 12,
                    "selected_format": "jpg",
                    "last_tile_url": "https://collections.example.test/iiif/3/sample.ptif/0,0,512,512/max/0/default.jpg",
                }

            iiif_mod._assemble_iiif_full_image = _fake_assemble
            out, info_json = node.download(
                "Generic IIIF Service URL",
                "https://collections.example.test/iiif/3/sample.ptif/info.json",
                delivery_mode="tile_assemble_full",
            )
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._assemble_iiif_full_image = old_assemble

        payload = json.loads(info_json)
        self.assertEqual(tuple(out.shape), (1, 1200, 1600, 3))
        self.assertEqual(payload["delivery"]["mode"], "tile_assemble_full")
        self.assertEqual(payload["delivery"]["tiles_total"], 12)
        self.assertEqual(payload["downloaded"]["width"], 1600)
        self.assertEqual(payload["source"]["width"], 1600)
        self.assertFalse(payload["limits"]["limited_by_service"])

    def test_iiif_download_saves_using_source_url_slug(self):
        """Verify IIIF node saves final image using source URL slug as filename."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_download = iiif_mod._download_iiif_image_bytes

        try:
            iiif_mod._resolve_iiif_service_url = lambda *args, **kwargs: "https://collections.example.test/iiif/3/sample.ptif"
            iiif_mod._fetch_iiif_info = lambda *args, **kwargs: {
                "id": "https://collections.example.test/iiif/3/sample.ptif",
                "type": "ImageService3",
                "profile": "level2",
                "width": 32,
                "height": 24,
                "tiles": [{"width": 512, "height": 512, "scaleFactors": [1, 2, 4]}],
                "sizes": [{"width": 800, "height": 600}],
            }

            def _fake_download(service_url, size_spec, output_format, timeout=30.0, session=None):
                _ = (service_url, size_spec, output_format, timeout, session)
                image = Image.new("RGB", (32, 24), color=(128, 64, 32))
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                return (
                    "https://collections.example.test/iiif/3/sample.ptif/full/max/0/default.jpg",
                    buffer.getvalue(),
                    "jpg",
                )

            iiif_mod._download_iiif_image_bytes = _fake_download

            source_url = "https://www.londonmuseum.org.uk/collections/v/object-443337/anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house/"
            with tempfile.TemporaryDirectory() as tmpdir:
                _out, info_json = node.download(
                    "London Museum Object Page",
                    source_url,
                    output_dir=tmpdir,
                )
                payload = json.loads(info_json)
                expected_path = os.path.join(
                    tmpdir,
                    "anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house_object-443337.jpg",
                )
                self.assertEqual(payload["saved_path"], expected_path)
                self.assertTrue(os.path.exists(expected_path))
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._download_iiif_image_bytes = old_download

    def test_iiif_download_title_or_slug_uses_page_title_when_available(self):
        """Verify IIIF node can save using resolved page title instead of URL slug."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_download = iiif_mod._download_iiif_image_bytes
        old_title_stem = iiif_mod._derive_output_stem_from_source_title_or_url

        try:
            iiif_mod._resolve_iiif_service_url = lambda *args, **kwargs: "https://collections.example.test/iiif/3/sample.ptif"
            iiif_mod._fetch_iiif_info = lambda *args, **kwargs: {
                "id": "https://collections.example.test/iiif/3/sample.ptif",
                "type": "ImageService3",
                "profile": "level2",
                "width": 32,
                "height": 24,
                "tiles": [{"width": 512, "height": 512, "scaleFactors": [1, 2, 4]}],
                "sizes": [{"width": 800, "height": 600}],
            }

            def _fake_download(service_url, size_spec, output_format, timeout=30.0, session=None):
                _ = (service_url, size_spec, output_format, timeout, session)
                image = Image.new("RGB", (32, 24), color=(64, 64, 160))
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                return (
                    "https://collections.example.test/iiif/3/sample.ptif/full/max/0/default.jpg",
                    buffer.getvalue(),
                    "jpg",
                )

            iiif_mod._download_iiif_image_bytes = _fake_download
            iiif_mod._derive_output_stem_from_source_title_or_url = (
                lambda _source_url, timeout=30.0, session=None, service_url="": "Anna_Pavlova_posed_in_day_dress_by_urn_in_the_garden_of_Ivy_House_object-443337"
            )

            source_url = "https://www.londonmuseum.org.uk/collections/v/object-443337/anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house/"
            with tempfile.TemporaryDirectory() as tmpdir:
                _out, info_json = node.download(
                    "London Museum Object Page",
                    source_url,
                    output_dir=tmpdir,
                    filename_mode="title_or_slug",
                )
                payload = json.loads(info_json)
                expected_path = os.path.join(
                    tmpdir,
                    "Anna_Pavlova_posed_in_day_dress_by_urn_in_the_garden_of_Ivy_House_object-443337.jpg",
                )
                self.assertEqual(payload["saved_path"], expected_path)
                self.assertTrue(os.path.exists(expected_path))
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._download_iiif_image_bytes = old_download
            iiif_mod._derive_output_stem_from_source_title_or_url = old_title_stem

    def test_iiif_download_save_adds_numeric_suffix_when_name_exists(self):
        """Verify IIIF saver avoids overwrite by appending numeric suffix."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_download = iiif_mod._download_iiif_image_bytes

        try:
            iiif_mod._resolve_iiif_service_url = lambda *args, **kwargs: "https://collections.example.test/iiif/3/sample.ptif"
            iiif_mod._fetch_iiif_info = lambda *args, **kwargs: {
                "id": "https://collections.example.test/iiif/3/sample.ptif",
                "type": "ImageService3",
                "profile": "level2",
                "width": 32,
                "height": 24,
            }

            def _fake_download(service_url, size_spec, output_format, timeout=30.0, session=None):
                _ = (service_url, size_spec, output_format, timeout, session)
                image = Image.new("RGB", (32, 24), color=(128, 64, 32))
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                return (
                    "https://collections.example.test/iiif/3/sample.ptif/full/max/0/default.jpg",
                    buffer.getvalue(),
                    "jpg",
                )

            iiif_mod._download_iiif_image_bytes = _fake_download

            source_url = "https://www.londonmuseum.org.uk/collections/v/object-443337/anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house/"
            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = os.path.join(
                    tmpdir,
                    "anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house_object-443337.jpg",
                )
                with open(base_path, "wb") as fh:
                    fh.write(b"existing")
                _out, info_json = node.download(
                    "London Museum Object Page",
                    source_url,
                    output_dir=tmpdir,
                )
                payload = json.loads(info_json)
                suffixed_path = os.path.join(
                    tmpdir,
                    "anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house_object-443337_2.jpg",
                )
                self.assertEqual(payload["saved_path"], suffixed_path)
                self.assertTrue(os.path.exists(base_path))
                self.assertTrue(os.path.exists(suffixed_path))
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._download_iiif_image_bytes = old_download

    def test_iiif_http_get_retries_read_timeout(self):
        """Verify IIIF HTTP wrapper retries transient timeout errors."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        old_sleep = iiif_mod.time.sleep
        import requests

        class _DummySession:
            def __init__(self):
                self.calls = 0

            def get(self, url, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise requests.exceptions.ReadTimeout("timed out")

                class _Response:
                    status_code = 200
                    content = b"ok"
                    text = "ok"

                    def json(self):
                        return {}

                return _Response()

        try:
            iiif_mod.time.sleep = lambda *_args, **_kwargs: None
            session = _DummySession()
            response = iiif_mod._http_get(
                "https://collections.example.test/ping",
                timeout=1.0,
                session=session,
                retries=2,
            )
        finally:
            iiif_mod.time.sleep = old_sleep

        self.assertEqual(int(response.status_code), 200)
        self.assertEqual(session.calls, 2)

    def test_iiif_tile_cache_reuses_saved_tile_bytes(self):
        """Verify IIIF tile cache skips repeated network fetches for the same tile URL."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        old_http_get = iiif_mod._http_get
        calls = {"count": 0}

        class _DummyResponse:
            def __init__(self, content: bytes):
                self.status_code = 200
                self.content = content
                self.text = ""

            def json(self):
                return {}

        try:
            def _fake_http_get(url, *, timeout, session=None, retries=3, retry_backoff=0.75):
                _ = (timeout, session, retries, retry_backoff)
                calls["count"] += 1
                return _DummyResponse(b"tile-bytes")

            iiif_mod._http_get = _fake_http_get
            with tempfile.TemporaryDirectory() as tmpdir:
                tile_url_1, content_1, fmt_1 = iiif_mod._download_iiif_tile_bytes(
                    "https://collections.example.test/iiif/3/sample.ptif",
                    region="0,0,128,128",
                    output_format="jpg",
                    timeout=1.0,
                    cache_dir=tmpdir,
                )
                tile_url_2, content_2, fmt_2 = iiif_mod._download_iiif_tile_bytes(
                    "https://collections.example.test/iiif/3/sample.ptif",
                    region="0,0,128,128",
                    output_format="jpg",
                    timeout=1.0,
                    cache_dir=tmpdir,
                )
        finally:
            iiif_mod._http_get = old_http_get

        self.assertEqual(tile_url_1, tile_url_2)
        self.assertEqual(content_1, b"tile-bytes")
        self.assertEqual(content_2, b"tile-bytes")
        self.assertEqual(fmt_1, "jpg")
        self.assertEqual(fmt_2, "jpg")
        self.assertEqual(calls["count"], 1)

    def test_iiif_tile_assembly_clears_cache_after_success(self):
        """Verify resume cache is removed after successful tile assembly completion."""
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_assemble = iiif_mod._assemble_iiif_full_image

        try:
            iiif_mod._resolve_iiif_service_url = lambda *args, **kwargs: "https://collections.example.test/iiif/3/sample.ptif"
            iiif_mod._fetch_iiif_info = lambda *args, **kwargs: {
                "id": "https://collections.example.test/iiif/3/sample.ptif",
                "type": "ImageService3",
                "profile": "level2",
                "width": 64,
                "height": 48,
                "tiles": [{"width": 32, "height": 32, "scaleFactors": [1, 2, 4]}],
                "sizes": [{"width": 800, "height": 600}],
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                cache_scope = os.path.join(tmpdir, "resume_scope")

                def _fake_assemble(service_url, info, output_format="jpg", timeout=30.0, session=None, cache_dir=None):
                    _ = (service_url, info, output_format, timeout, session, cache_dir)
                    os.makedirs(cache_scope, exist_ok=True)
                    with open(os.path.join(cache_scope, "tile.jpg"), "wb") as f:
                        f.write(b"tile")
                    image = Image.new("RGB", (64, 48), color=(96, 48, 24))
                    return image, {
                        "mode": "tile_assemble_full",
                        "tile_width": 32,
                        "tile_height": 32,
                        "tiles_x": 2,
                        "tiles_y": 2,
                        "tiles_total": 4,
                        "tiles_downloaded": 4,
                        "selected_format": "jpg",
                        "last_tile_url": "https://collections.example.test/iiif/3/sample.ptif/0,0,32,32/max/0/default.jpg",
                        "cache_dir": cache_scope,
                        "cache_hits": 2,
                        "cache_misses": 2,
                        "cache_stores": 2,
                        "cache_cleared": False,
                    }

                iiif_mod._assemble_iiif_full_image = _fake_assemble
                _out, info_json = node.download(
                    "Generic IIIF Service URL",
                    "https://collections.example.test/iiif/3/sample.ptif/info.json",
                    delivery_mode="tile_assemble_full",
                    cache_dir=tmpdir,
                )
                payload = json.loads(info_json)
                self.assertFalse(os.path.exists(cache_scope))
                self.assertTrue(payload["delivery"]["cache_cleared"])
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._assemble_iiif_full_image = old_assemble

    def test_descreen_adaptive_scale_node_contract(self):
        """Verify descreen node returns processed image, preview, floats, and JSON diagnostics."""
        mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_descreen_adaptive")
        node = mod.ImageDescreenAdaptiveScale()

        h, w = 128, 128
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 140.0 + 55.0 * np.exp(-(((xx - 64.0) ** 2 + (yy - 64.0) ** 2) / (2.0 * 24.0 * 24.0)))
        halftone = 14.0 * np.cos(2.0 * np.pi * (xx + yy) / 8.0)
        image = np.clip(base + halftone, 0.0, 255.0).astype(np.uint8)
        rgb = np.stack([image, image, image], axis=-1)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)

        processed, roi_preview, recommended_percent, estimated_period_px, analysis_json = node.descreen(
            tensor,
            roi_mode="full_frame",
            min_scale_percent=8.0,
            max_scale_percent=18.0,
            step_percent=1.0,
            target_screen_px=1.0,
            detail_weight=1.25,
            pre_blur_px=0.0,
        )

        payload = json.loads(analysis_json)
        self.assertEqual(tuple(processed.shape), (1, h, w, 3))
        self.assertEqual(int(roi_preview.shape[0]), 1)
        self.assertGreater(int(roi_preview.shape[2]), w)
        self.assertGreater(float(recommended_percent), 0.0)
        self.assertTrue(8.0 <= float(recommended_percent) <= 18.0)
        self.assertTrue(4.5 <= float(estimated_period_px) <= 12.5)
        self.assertIn("candidates", payload)
        self.assertIn("recommended_percent", payload)

    def test_descreen_apply_percent_node_contract(self):
        """Verify fixed descreen node returns processed image, applied percent, and JSON diagnostics."""
        mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_descreen_adaptive")
        node = mod.ImageDescreenApplyPercent()

        h, w = 96, 96
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 120.0 + 30.0 * np.exp(-(((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / (2.0 * 18.0 * 18.0)))
        halftone = 10.0 * np.cos(2.0 * np.pi * (xx + yy) / 8.0)
        image = np.clip(base + halftone, 0.0, 255.0).astype(np.uint8)
        rgb = np.stack([image, image, image], axis=-1)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)

        processed, applied_percent, analysis_json = node.apply(
            tensor,
            scale_percent=13.0,
            pre_blur_px=0.0,
        )

        payload = json.loads(analysis_json)
        self.assertEqual(tuple(processed.shape), (1, h, w, 3))
        self.assertEqual(float(applied_percent), 13.0)
        self.assertEqual(payload["mode"], "fixed_percent")
        self.assertEqual(float(payload["applied_percent"]), 13.0)
        self.assertEqual(int(payload["batch_size"]), 1)


if __name__ == "__main__":
    unittest.main()
