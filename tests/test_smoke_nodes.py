import json
import os
import sys
import types
import unittest
import importlib

import torch


def _install_folder_paths_stub():
    if "folder_paths" in sys.modules:
        return
    stub = types.SimpleNamespace(
        get_input_directory=lambda: os.getcwd(),
        filter_files_content_types=lambda files, content_types: files,
        get_annotated_filepath=lambda name: name,
    )
    sys.modules["folder_paths"] = stub


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()

    def test_image_difference_autoresize(self):
        utils_mod = importlib.import_module("ComfyUI_ALEXZ_tools.utils")

        a = torch.rand(64, 64, 3)
        b = torch.rand(32, 48, 3)
        diff = utils_mod.image_difference(a, b)
        self.assertEqual(tuple(diff.shape), (64, 64, 3))

    def test_color_match_json_has_quality_metrics(self):
        image_color_match = importlib.import_module("ComfyUI_ALEXZ_tools.image_color_match")

        # Avoid LPIPS model download in smoke tests.
        old_lpips = image_color_match._lpips_alex_distance
        image_color_match._lpips_alex_distance = lambda a, b: None
        try:
            node = image_color_match.ImageColorMatchToReference()
            ref = torch.rand(1, 48, 48, 3)
            img = torch.rand(1, 48, 48, 3)
            matched, payload = node.match(ref, img, "fast", strength=0.8)
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

    def test_video_frame_topk_helpers(self):
        video_mod = importlib.import_module("ComfyUI_ALEXZ_tools.video_frame_match")

        top = []
        video_mod._update_top_matches(top, 0, 0.30, limit=3)
        video_mod._update_top_matches(top, 1, 0.10, limit=3)
        video_mod._update_top_matches(top, 2, 0.20, limit=3)
        video_mod._update_top_matches(top, 3, 0.15, limit=3)
        self.assertEqual([x["index"] for x in top], [1, 3, 2])
        conf = video_mod._confidence_from_top(top)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


if __name__ == "__main__":
    unittest.main()
