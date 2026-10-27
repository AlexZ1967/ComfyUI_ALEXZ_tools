"""Validate material polarity, deterministic timelines and independent defect nodes."""

import importlib
import itertools
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

NAMES = ["CaptureDirt", "NegativeDevelopment", "NegativeDamage", "PrintingDirt",
         "PositiveDevelopment", "PositiveDamage", "ProjectorDirt"]


@pytest.fixture(scope="module")
def defects():
    """Load the extension without starting the ComfyUI server."""
    if "ComfyUI_ALEXZ_tools" not in sys.modules:
        package = types.ModuleType("ComfyUI_ALEXZ_tools")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["ComfyUI_ALEXZ_tools"] = package
    return importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_defects")


def metadata(defects, images, stage="print", ids=None):
    """Make a valid main-stage passport with an optional repeated-frame timeline."""
    data = {"history": [], "cadence": {"output_group_ids": ids} if ids is not None else None}
    return defects.pipeline._write_film_meta(data, images, stage, {"resolved_output_transfer": "srgb"})


@pytest.mark.parametrize("name", NAMES)
def test_disabled_effect_reuses_batch_and_preserves_material(defects, name):
    """Do not allocate another full video or invent a material stage when disabled."""
    node = getattr(defects, "VideoSilentFilm" + name)()
    source = torch.rand((5, 17, 29, 4))
    meta = metadata(defects, source, "camera")
    with patch.object(defects.pipeline, "_run_stage", side_effect=AssertionError("disabled effect allocated video")):
        output = node.apply(source, meta)
    assert output[0].data_ptr() == source.data_ptr()
    assert torch.equal(output[0], source)
    assert output[2].shape == (1, 17, 29, 4)
    assert output[3].shape == (1, 17, 29, 3) and output[3].count_nonzero() == 0
    data = json.loads(output[1])
    assert data["stage"] == "camera" and data["history"] == json.loads(meta)["history"]
    assert data["defect_history"][-1]["enabled"] is False
    assert len(output) == len(node.RETURN_TYPES)
    assert node.apply(source)[1] == ""


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_active_effects_chunk_device_alpha_and_previews(defects, name, device):
    """Keep visual results independent of chunk size and GPU while preserving alpha."""
    node = getattr(defects, "VideoSilentFilm" + name)()
    source = torch.rand((4, 41, 67, 4), generator=torch.Generator().manual_seed(4)) * 0.6 + 0.2
    settings = dict(strength=0.4, size=3, seed=1925)
    expected = node.apply(source, compute_device="cpu", batch_size=1, **settings)
    actual = node.apply(source, compute_device=device, batch_size=3, **settings)
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=3e-6, rtol=0)
    torch.testing.assert_close(actual[0][..., 3].cpu(), source[..., 3], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], actual[0][2:3])
    assert torch.isfinite(actual[0]).all() and actual[0].min() >= 0 and actual[0].max() <= 1
    assert not torch.equal(actual[0].cpu(), source)
    assert actual[3].max() > 0
    assert len(actual) == len(node.RETURN_NAMES)
    if "preview_positive" in node.RETURN_NAMES:
        expected_positive = defects.sensitometry.reference_positive(actual[2][..., :3], "srgb")
        torch.testing.assert_close(actual[4][..., :3], expected_positive)
        torch.testing.assert_close(actual[4][..., 3], actual[2][..., 3])


@pytest.mark.parametrize("name", NAMES)
def test_segment_offset_and_repeated_execution(defects, name):
    """Reproduce a slice of an unchanged physical timeline without stored RNG state."""
    node = getattr(defects, "VideoSilentFilm" + name)()
    source = torch.full((9, 23, 37, 3), 0.45)
    kwargs = dict(strength=0.5, size=2, compute_device="cpu", persistence_frames=5)
    full = node.apply(source, **kwargs)[0]
    segment = node.apply(source[3:7], frame_offset=3, batch_size=1, **kwargs)[0]
    torch.testing.assert_close(segment, full[3:7], atol=0, rtol=0)
    torch.testing.assert_close(node.apply(source, **kwargs)[0], full, atol=0, rtol=0)


def test_physical_frame_identity_and_stationary_projection_shadow(defects):
    """Reuse film marks on repeated frames and let screen dirt follow existing lamp flux."""
    source = torch.full((3, 33, 65, 3), 0.5)
    meta = metadata(defects, source, "negative", [7, 7, 8])
    node = defects.VideoSilentFilmNegativeDamage()
    output = node.apply(source, meta, pattern="dust", size=3, amount=5, strength=0.7, compute_device="cpu")[0]
    assert torch.equal(output[0], output[1])
    assert not torch.equal(output[1], output[2])
    gain = torch.tensor([0.5, 1, 1.5]).view(-1, 1, 1, 1)
    linear = torch.full_like(source, 0.2) * gain
    source = defects.sensitometry.encode_transfer(linear, "srgb")
    output = defects.VideoSilentFilmProjectorDirt().apply(source, strength=0.5, size=3,
        irregularity=0, persistence_frames=10000, softness=0.2, compute_device="cpu")[0]
    actual = defects.sensitometry.decode_transfer(output, "srgb")
    torch.testing.assert_close(actual / gain, actual[1:2].expand_as(actual), atol=2e-7, rtol=0)


@pytest.mark.parametrize("negative", [True, False])
@pytest.mark.parametrize("kind", ["surface", "dirty", "emulsion_loss"])
def test_scratch_material_polarity(defects, negative, kind):
    """Differentiate removal of emulsion from opaque surface damage, including printing."""
    node = (defects.VideoSilentFilmNegativeDamage if negative else defects.VideoSilentFilmPositiveDamage)()
    source = torch.full((1, 65, 97, 3), 0.3)
    out = node.apply(source, pattern="scratches", damage_type=kind, strength=0.8, amount=2,
        size=2, edge_bias=0, compute_device="cpu")[0]
    if kind == "emulsion_loss":
        assert out.max() > source.max() and out.min() >= source.min() - 1e-6
    else:
        assert out.min() < source.min() and out.max() <= source.max() + 1e-6
    if negative:
        before = defects.sensitometry.reference_positive(source, "srgb")
        after = defects.sensitometry.reference_positive(out, "srgb")
        if kind == "emulsion_loss":
            assert after.min() < before.min()
        else:
            assert after.max() > before.max()


def test_development_density_polarity_and_continuity(defects):
    """Positive density darkens projection while negative density lightens the print."""
    source = torch.full((2, 17, 25, 3), 0.4)
    with patch.object(defects.masks, "development_field", return_value=torch.ones((17, 25))):
        negative = defects.VideoSilentFilmNegativeDevelopment().apply(source, strength=0.2, compute_device="cpu")
        positive = defects.VideoSilentFilmPositiveDevelopment().apply(source, strength=0.2, compute_device="cpu")
    assert negative[0].mean() < source.mean() and positive[0].mean() < source.mean()
    assert negative[4].mean() > defects.sensitometry.reference_positive(source[:1], "srgb").mean()
    field = defects.masks.development_field
    a = field(17, 25, 999, 3, "mixed", 10, 1000, 0.4, "cpu")
    b = field(17, 25, 1000, 3, "mixed", 10, 1000, 0.4, "cpu")
    assert (a - b).abs().max() < 1e-4


@pytest.mark.parametrize("omitted", itertools.product((False, True), repeat=7))
def test_all_defect_bypass_combinations_accept_following_main_stage(defects, omitted):
    """Preserve material passports through every combination of optional defect stages."""
    source = torch.full((1, 5, 7, 3), 0.4)
    p = defects.pipeline
    image, meta = source, metadata(defects, source, "camera")
    for index, name in enumerate(NAMES):
        if index == 1:
            image, meta, *_ = p.VideoSilentFilmNegative().apply(image, meta, compute_device="cpu")
        elif index == 4:
            image, meta, *_ = p.VideoSilentFilmPrint().apply(image, meta, compute_device="cpu")
        elif index == 6:
            image, _, meta, _ = p.VideoSilentFilmProjector().apply(image, meta, gate_weave_px=0,
                lamp_flicker=0, lamp_breathing=0, compute_device="cpu")
        if not omitted[index]:
            image, meta, *_ = getattr(defects, "VideoSilentFilm" + name)().apply(image, meta, strength=0.1, compute_device="cpu")
    data = json.loads(meta)
    assert data["stage"] == "projector"
    assert len(data.get("defect_history", [])) == 7 - sum(omitted)
    assert image.shape == source.shape and torch.isfinite(image).all()


@pytest.mark.parametrize("bad", [{"strength": float("nan")}, {"amount": -1}, {"persistence_frames": 0},
    {"size": float("inf")}, {"frame_offset": 1.5}, {"seed": -1}, {"pattern": "unknown"}, {"extra": 1}])
def test_bad_controls_fail_before_processing(defects, bad):
    """Reject malformed public inputs instead of rendering silent fallback results."""
    with pytest.raises(ValueError):
        defects.VideoSilentFilmCaptureDirt().apply(torch.ones((1, 3, 5, 3)), **bad)


def test_metadata_validation_and_registry(defects):
    """Keep malformed metadata visible and expose all seven nodes through the loader."""
    source = torch.ones((2, 3, 5, 3))
    node = defects.VideoSilentFilmCaptureDirt()
    for text in ("bad json", "{}", metadata(defects, source[:1]), metadata(defects, source, ids=[1])):
        with pytest.raises(ValueError):
            node.apply(source, text)
    registry = importlib.import_module("ComfyUI_ALEXZ_tools.nodes")
    for name in NAMES:
        cls = registry.NODE_CLASS_MAPPINGS["VideoSilentFilm" + name]
        assert cls is getattr(defects, "VideoSilentFilm" + name)
        assert len(cls.OUTPUT_TOOLTIPS) == len(cls.RETURN_NAMES)


def test_workflow_contains_ordered_defects_and_preserves_seed_and_preview(defects):
    """Ensure the export processes the complete chain and previews stay one frame."""
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root / "workflows/Видео-в-Хронику-по-этапам.json").read_text())
    nodes = {n["id"]: n for n in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    node = next(n for n in nodes.values() if n["type"] == "VideoSaveFinal")
    reverse_chain = []
    while True:
        reverse_chain.append(node["type"])
        image = next((i for i in node.get("inputs", []) if i["name"] == "image"), None)
        if not image or image.get("link") is None:
            break
        node = nodes[links[image["link"]][1]]
    assert list(reversed(reverse_chain)) == ["VideoLoadSource", "VideoSilentFilmCamera",
        "VideoSilentFilmCaptureDirt", "VideoSilentFilmNegative", "VideoSilentFilmNegativeDevelopment",
        "VideoSilentFilmNegativeDamage", "VideoSilentFilmPrintingDirt", "VideoSilentFilmPrint",
        "VideoSilentFilmPositiveDevelopment", "VideoSilentFilmPositiveDamage", "VideoSilentFilmProjector",
        "VideoSilentFilmProjectorDirt", "VideoSaveFinal"]
    for name in NAMES:
        node = next(n for n in nodes.values() if n["type"] == "VideoSilentFilm" + name)
        assert node["widgets_values_named"]["strength"] == 0
        assert next(i for i in node["inputs"] if i["name"] == "film_json")["link"] is not None
        assert any(link[1] == node["id"] and nodes[link[3]]["type"] == "PreviewImage" for link in links.values())
