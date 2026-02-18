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
            export_lut_cube=False,
        )
        self.assertEqual(tuple(out.shape), (1, 20, 20, 4))
        self.assertEqual(cube_text[0], "")
        data = json.loads(look_json[0])
        self.assertEqual(data.get("schema_name"), "alexz.look_match.resolve")
        self.assertEqual(data.get("schema_version"), 1)
        self.assertEqual(data.get("status"), "ok")

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
