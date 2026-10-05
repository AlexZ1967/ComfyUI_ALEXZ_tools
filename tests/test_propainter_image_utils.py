"""Focused tests for ProPainter image configuration helpers."""

from __future__ import annotations

import os
import sys
import types
import unittest


class ProPainterImageUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            package = types.ModuleType("ComfyUI_ALEXZ_tools")
            package.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = package
        from ComfyUI_ALEXZ_tools.propainter.utils.image_utils import ImageConfig, ImageOutpaintConfig

        cls.ImageConfig = ImageConfig
        cls.ImageOutpaintConfig = ImageOutpaintConfig

    def test_image_config_rounds_process_size_to_eight(self):
        config = self.ImageConfig(
            width=1919,
            height=1079,
            mask_dilates=4,
            flow_mask_dilates=2,
            input_size=(1919, 1079),
            video_length=3,
        )
        self.assertEqual(config.process_size, (1912, 1072))

    def test_outpaint_config_reuses_base_process_size(self):
        config = self.ImageOutpaintConfig(
            width=1919,
            height=1079,
            mask_dilates=4,
            flow_mask_dilates=2,
            input_size=(1919, 1079),
            video_length=3,
            width_scale=1.25,
            height_scale=1.5,
        )
        self.assertEqual(config.process_size, (1912, 1072))
        self.assertEqual(config.outpaint_size, (2392, 1616))
