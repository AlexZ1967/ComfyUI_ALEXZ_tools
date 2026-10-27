"""Compare Camera edge defocus at equal framing and display size across HD and UHD."""

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def camera():
    """Import Camera without executing package startup or unrelated node registrations."""
    root = Path(__file__).resolve().parents[1]
    for name, path in (("ComfyUI_ALEXZ_tools", root),):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    module = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_pipeline")
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield module.VideoSilentFilmCamera()
    torch.set_num_threads(previous)


def chart(width):
    """Sample the same band-limited scene at each pixel center and frame resolution."""
    height = width * 9 // 16
    x = (torch.arange(width) + 0.5) / width
    y = (torch.arange(height) + 0.5) / width
    gray = 0.5 + 0.25 * torch.sin(x[None, :] * 2 * torch.pi * 80) + 0.2 * torch.cos(y[:, None] * 2 * torch.pi * 60)
    return gray[None, ..., None].expand(1, height, width, 3)


@pytest.mark.parametrize("strength", [0.03, 0.15])
def test_camera_edge_defocus_matches_at_equal_display_size(camera, strength):
    """Keep edge contrast loss stable when UHD output is displayed at Full HD size."""
    outputs, effects = [], []
    for width in (1920, 3840):
        source = chart(width)
        output, _, preview = camera.apply(
            source, softness=0, vignette_strength=0, edge_softness=strength,
            edge_start=0.6, edge_end=1.0, focus_drift_strength=0, exposure_variation=0,
            compute_device="cpu", batch_size=1)
        torch.testing.assert_close(preview, output, atol=0, rtol=0)
        # В центре маска нулевая; Camera только переводит исходник в монохром.
        torch.testing.assert_close(output[:, :, width//2-2:width//2+2],
                                   source[:, :, width//2-2:width//2+2], atol=2e-7, rtol=0)
        if width == 3840:
            output = F.avg_pool2d(output.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
            source = F.avg_pool2d(source.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        # Одинаковый участок у правого края, вдали от отражённой границы фильтра.
        region = (slice(None), slice(270, 810), slice(1740, 1820), slice(None))
        effects.append(float((output[region] - source[region]).abs().mean()))
        outputs.append(output)
    assert float((outputs[0] - outputs[1]).abs().mean()) < 0.002
    assert effects[1] == pytest.approx(effects[0], rel=0.04, abs=0.0005)
    assert effects[0] > 0.001
