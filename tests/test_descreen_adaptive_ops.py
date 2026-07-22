"""Unit tests for pure adaptive-descreen image operations."""

from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np
import torch


class DescreenAdaptiveOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            package = types.ModuleType("ComfyUI_ALEXZ_tools")
            package.__path__ = [root]
            sys.modules["ComfyUI_ALEXZ_tools"] = package

    def setUp(self):
        from ComfyUI_ALEXZ_tools.nodes import image_descreen_adaptive_ops as ops

        self.ops = ops

    def test_tensor_batch_roundtrip_and_validation(self):
        image = torch.full((1, 8, 10, 3), 0.5)
        rgb = self.ops.to_rgb_batch(image)
        self.assertEqual(rgb.shape, (1, 8, 10, 3))
        self.assertEqual(self.ops.to_tensor(rgb[0]).shape, (1, 8, 10, 3))
        with self.assertRaises(ValueError):
            self.ops.to_rgb_batch(torch.zeros((8, 10, 3)))

    def test_roi_selection_and_resampling(self):
        self.assertEqual(self.ops.select_roi_rect(100, 80, roi_mode="center_square", roi_size_percent=50, roi_x=0, roi_y=0, roi_w=0, roi_h=0), (30, 20, 40, 40))
        self.assertEqual(self.ops.clip_roi(-4, 70, 20, 20, image_w=100, image_h=80), (0, 70, 20, 10))
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        self.assertEqual(self.ops.pil_resample_rgb_with_mode(image, 0.5, pre_blur_px=0, resample_mode="lanczos").shape, (20, 40, 3))
        self.assertEqual(self.ops.pil_resample_rgb_with_mode(image, 0.5, pre_blur_px=0, resample_mode="bicubic", restore_size=False).shape, (10, 20, 3))

    def test_fft_metrics_detect_periodic_pattern(self):
        from ComfyUI_ALEXZ_tools.nodes import image_descreen_adaptive_fft_ops as fft_ops

        x = np.arange(64, dtype=np.float32)
        plane = np.tile(127.5 + 100.0 * np.sin(2.0 * np.pi * x / 8.0), (64, 1))
        peak = fft_ops._fft_peak(plane)
        self.assertAlmostEqual(peak["period_px"], 8.0, places=4)
        self.assertEqual((peak["dx"], peak["dy"]), (8, 0))
        self.assertGreater(fft_ops._screen_energy(plane, peak["dx"], peak["dy"]), 0.0)
        self.assertEqual(fft_ops._fft_log_magnitude(plane).shape, plane.shape)

        peaks, _magnitude = fft_ops._find_fft_peaks(
            plane,
            peak_count=2,
            protect_low_freq=0.02,
            min_period_px=2.0,
            max_period_px=32.0,
            nms_radius=3,
        )
        self.assertTrue(peaks)
        expanded = fft_ops._expand_fft_peaks_with_harmonics(peaks[:1], harmonic_count=2)
        mask = fft_ops._build_fft_notch_mask(
            plane.shape,
            expanded,
            notch_radius=2.0,
            notch_strength=0.8,
            notch_tangent_scale=1.5,
        )
        self.assertEqual(mask.shape, plane.shape)
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_tonal_masks_are_bounded_and_shape_stable(self):
        from ComfyUI_ALEXZ_tools.nodes import image_descreen_adaptive_tonal_ops as tonal_ops

        ramp = np.linspace(0, 255, 32, dtype=np.uint8)
        image = np.repeat(ramp[None, :, None], 24, axis=0)
        image = np.repeat(image, 3, axis=2)
        transition = tonal_ops._build_transition_cleanup_mask(image)
        hybrid = tonal_ops._build_tonal_hybrid_mask(
            image,
            cleanup_strength=0.75,
            midtone_weight=0.8,
            transition_weight=0.6,
            shadow_protect=0.5,
            highlight_protect=0.5,
        )
        for mask in (transition, hybrid):
            self.assertEqual(mask.shape, image.shape[:2])
            self.assertGreaterEqual(float(mask.min()), 0.0)
            self.assertLessEqual(float(mask.max()), 1.0)

    def test_preview_and_scale_policy(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        preview = self.ops.build_compare_preview(image, image)
        self.assertEqual(preview.shape, (12, 40, 3))
        sheet = self.ops.build_scale_sheet_preview([{"percent": 50.0, "image": image}])
        self.assertEqual(sheet.shape, (40, 16, 3))
        self.assertEqual(self.ops.predict_descreen_scale_percent(8.0, target_screen_px=1.0), 12.5)


if __name__ == "__main__":
    unittest.main()
