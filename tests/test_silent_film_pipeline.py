"""Validate stage semantics, defect insertion, memory fallback, and device consistency."""

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def pipeline():
    """Import the custom-node package without executing ComfyUI startup."""
    if "ComfyUI_ALEXZ_tools" not in sys.modules:
        package = types.ModuleType("ComfyUI_ALEXZ_tools")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["ComfyUI_ALEXZ_tools"] = package
    return importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_pipeline")


def camera(pipeline, image=None, batch_size=2, device="cpu"):
    """Capture a short deterministic source without expensive flow."""
    if image is None:
        image = torch.rand((8, 33, 49, 4), generator=torch.Generator().manual_seed(3))
    return pipeline.VideoSilentFilmCamera().apply(
        image, source_fps=24, target_fps_min=16, target_fps_max=20, blur_mode="simple", blur_samples=3,
        camera_shake_strength=0.1, softness=0, focus_drift_strength=0, exposure_variation=0,
        vignette_strength=0.12, edge_softness=0.2, compute_device=device, batch_size=batch_size)


def test_full_pipeline_contract_and_alpha(pipeline):
    """Preserve frame metadata and alpha through all four physical stages."""
    captured, meta = camera(pipeline)
    assert len(captured) < 8
    negative, neg_meta = pipeline.VideoSilentFilmNegative().apply(captured, meta, compute_device="cpu")
    positive, pos_meta = pipeline.VideoSilentFilmPrint().apply(negative, neg_meta, compute_device="cpu")
    projected, fps, out_meta = pipeline.VideoSilentFilmProjector().apply(positive, pos_meta, gate_weave_px=0, softness=0, lamp_flicker=0, lamp_breathing=0, compute_device="cpu")
    assert fps == 25
    assert projected.shape == captured.shape
    for image in (negative, positive, projected):
        assert torch.isfinite(image).all()
        assert image.min() >= 0 and image.max() <= 1
        torch.testing.assert_close(image[..., 3], captured[..., 3], atol=0, rtol=0)
    data = json.loads(out_meta)
    assert [entry["stage"] for entry in data["history"]] == ["camera", "negative", "print", "projector"]
    assert data["cadence"] == json.loads(meta)["cadence"]
    assert data["image_shape"] == list(projected.shape)


def test_negative_inversion_print_response_and_defects(pipeline):
    """Invert scene values twice and carry a scratch through printing with correct polarity."""
    image = torch.linspace(0.01, 0.99, 49).view(1, 1, 49, 1).expand(4, 33, 49, 3).clone()
    captured, meta = camera(pipeline, image)
    negative, neg_meta = pipeline.VideoSilentFilmNegative().apply(captured, meta, grain_strength=0, development_variation=0, compute_device="cpu")
    assert torch.all(torch.diff(negative[0, 16, :, 0]) <= 0)
    printer = pipeline.VideoSilentFilmPrint()
    positive, _ = printer.apply(negative, neg_meta, grain_strength=0, print_softness=0, compute_device="cpu")
    assert torch.all(torch.diff(positive[0, 16, :, 0]) >= 0)
    scratched = negative.clone()
    scratched[:, :, 24:25, :3] = 1.0
    defect, _ = printer.apply(scratched, neg_meta, grain_strength=0, print_softness=0, compute_device="cpu")
    assert torch.all(defect[:, :, 24, :3] < positive[:, :, 24, :3])
    torch.testing.assert_close(defect[:, :, :24], positive[:, :, :24])
    dense, _ = pipeline.VideoSilentFilmNegative().apply(captured, meta, exposure_ev=1, grain_strength=0, development_variation=0, compute_device="cpu")
    assert dense[..., :3].mean() < negative[..., :3].mean()
    darker_print, _ = printer.apply(negative, neg_meta, print_exposure_ev=1, grain_strength=0, print_softness=0, compute_device="cpu")
    assert darker_print.mean() < positive.mean()


@pytest.mark.parametrize("stage", ["negative", "print", "projector"])
def test_stages_reject_wrong_metadata_and_bad_parameters(pipeline, stage):
    """Prevent silent misuse of stage polarity, malformed metadata, or changed frame counts."""
    captured, meta = camera(pipeline)
    classes = {"negative": pipeline.VideoSilentFilmNegative, "print": pipeline.VideoSilentFilmPrint, "projector": pipeline.VideoSilentFilmProjector}
    node = classes[stage]()
    for text in ("", "[]", "{}", '{"schema_name":"wrong"}'):
        with pytest.raises(ValueError):
            node.apply(captured, text, compute_device="cpu")
    wrong = json.loads(meta)
    wrong["stage"] = stage
    with pytest.raises(ValueError, match="Expected film stage"):
        node.apply(captured, json.dumps(wrong), compute_device="cpu")
    if stage == "negative":
        with pytest.raises(ValueError, match="image_shape"):
            node.apply(captured[:1], meta, compute_device="cpu")
        with pytest.raises(ValueError):
            node.apply(captured, meta, development_contrast=float("nan"), compute_device="cpu")


def test_grain_chunks_and_devices(pipeline):
    """Keep independent grain layers reproducible across chunk boundaries and real CUDA."""
    captured, meta = camera(pipeline)
    negative_node, print_node = pipeline.VideoSilentFilmNegative(), pipeline.VideoSilentFilmPrint()
    negative, neg_meta = negative_node.apply(captured, meta, compute_device="cpu", batch_size=1)
    positive, pos_meta = print_node.apply(negative, neg_meta, compute_device="cpu", batch_size=1)
    projector = pipeline.VideoSilentFilmProjector()
    projected, _, _ = projector.apply(positive, pos_meta, compute_device="cpu", batch_size=1, vignette_strength=0.1, edge_softness=0.2)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for size in (2, 4):
            actual, _ = negative_node.apply(captured, meta, compute_device=device, batch_size=size)
            torch.testing.assert_close(actual, negative, atol=1e-4, rtol=1e-4)
            actual, _ = print_node.apply(negative, neg_meta, compute_device=device, batch_size=size)
            torch.testing.assert_close(actual, positive, atol=1e-4, rtol=1e-4)
            actual, fps, _ = projector.apply(positive, pos_meta, compute_device=device, batch_size=size, vignette_strength=0.1, edge_softness=0.2)
            torch.testing.assert_close(actual, projected, atol=1e-4, rtol=1e-4)
            assert fps == 25
    other_negative, _ = negative_node.apply(captured, meta, seed=11, compute_device="cpu")
    assert (negative - other_negative).abs().mean() > 1e-5
    other_print, _ = print_node.apply(negative, neg_meta, seed=11, compute_device="cpu")
    assert (positive - other_print).abs().mean() > 1e-5


def test_projector_optics_are_stationary(pipeline):
    """Move the print and its defects before applying the fixed projector vignette."""
    captured, meta = camera(pipeline)
    negative, meta = pipeline.VideoSilentFilmNegative().apply(captured, meta, compute_device="cpu")
    positive, meta = pipeline.VideoSilentFilmPrint().apply(negative, meta, compute_device="cpu")
    positive = torch.full_like(positive, 0.6)
    positive[..., 3] = 0.7
    actual, fps, _ = pipeline.VideoSilentFilmProjector().apply(positive, meta, projection_fps=18, gate_weave_px=3, softness=0,
                                                            vignette_strength=0.4, edge_softness=0, lamp_flicker=0, lamp_breathing=0, compute_device="cpu")
    mask = pipeline.finish._lens_edge_mask(33, 49, 0.6, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(actual[..., :3], positive[..., :3] * (1 - 0.4 * mask))
    torch.testing.assert_close(actual[..., 3], positive[..., 3])
    assert fps == 18


def test_stage_oom_retry_and_logs(pipeline):
    """Retry only the failed chunk and avoid a false completion on unrecoverable errors."""
    image = torch.rand((5, 9, 13, 3))
    attempts = []
    real_chunk = pipeline._stage_chunk
    def simulated(images, start, end, device, operation):
        """Force shrinking and CPU fallback without allocating CUDA tensors."""
        attempts.append((start, end, device.type))
        if device.type == "cuda":
            raise torch.OutOfMemoryError("test")
        return real_chunk(images, start, end, device, operation)
    with patch.object(pipeline.finish, "_finish_processing_plan", return_value=(torch.device("cuda"), 4, 2)):
        with patch.object(pipeline, "_stage_chunk", side_effect=simulated):
            result = pipeline._run_stage(image, "Test", "auto", 4, lambda chunk, start, end: chunk)
    torch.testing.assert_close(result, image)
    assert attempts[:4] == [(0, 4, "cuda"), (0, 2, "cuda"), (0, 1, "cuda"), (0, 2, "cpu")]


def test_camera_defaults_and_schema_registration(pipeline):
    """Keep the public four-stage contract registered with per-output UI metadata."""
    registry = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.node_registry")
    specs = {spec.type_name: spec for spec in registry.NODE_SPECS}
    for stage in ("Camera", "Negative", "Print", "Projector"):
        name = f"VideoSilentFilm{stage}"
        cls = getattr(pipeline, name)
        assert name in specs and name in registry.NODE_UI_METADATA
        assert len(registry.NODE_UI_METADATA[name]["output_tooltips"]) == len(cls.RETURN_TYPES)
    camera_schema = pipeline.VideoSilentFilmCamera.INPUT_TYPES()["required"]
    assert "playback_mode" not in camera_schema and "crank_frames_per_turn" not in camera_schema
    assert camera_schema["source_fps"][1]["default"] == 36
    assert camera_schema["vignette_strength"][1]["default"] == 0.12


def test_camera_chunk_device_and_uniform_optics(pipeline):
    """Keep camera results stable across device/chunk choices and avoid dark padding halos."""
    expected, _ = camera(pipeline, batch_size=1)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        actual, _ = camera(pipeline, batch_size=3, device=device)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    uniform = torch.full((2, 33, 49, 3), 0.6)
    softened = pipeline._global_softness(uniform, 1)
    torch.testing.assert_close(softened, uniform)


def test_staged_workflow_contract(pipeline):
    """Validate saved widget order, IMAGE/metadata links and the explicit projector fps link."""
    path = Path(__file__).resolve().parents[1] / "workflows/Видео-в-Хронику-по-этапам.json"
    workflow = json.loads(path.read_text())
    nodes = {n["id"]: n for n in workflow["nodes"]}
    for node in nodes.values():
        if not node["type"].startswith("VideoSilentFilm"):
            continue
        cls = getattr(pipeline, node["type"])
        schema = cls.INPUT_TYPES()["required"]
        assert schema["seed"][1]["control_after_generate"] is True
        assert [i["name"] for i in node["inputs"]] == list(schema)
        widgets = iter(node["widgets_values"])
        for name, spec in schema.items():
            options = spec[1] if len(spec) > 1 else {}
            if spec[0] == "IMAGE" or options.get("forceInput"):
                continue
            value = next(widgets)
            assert value == node["widgets_values_named"][name]
            if isinstance(spec[0], list):
                assert value in spec[0]
            else:
                assert options["min"] <= value <= options["max"]
            if name == "seed":
                assert next(widgets) == "fixed"
                assert node["widgets_values_named"]["control_after_generate"] == "fixed"
        assert list(widgets) == []
    for id, source, output, target, input_slot, kind in workflow["links"]:
        assert nodes[source]["outputs"][output]["type"] == kind
        assert nodes[target]["inputs"][input_slot]["type"] == kind
        assert nodes[target]["inputs"][input_slot]["link"] == id
        assert id in nodes[source]["outputs"][output]["links"]
    assert any(nodes[s]["type"] == "VideoSilentFilmProjector" and nodes[t]["inputs"][i]["name"] == "frame_rate"
               for _, s, _, t, i, _ in workflow["links"])


def test_staged_workflow_preserves_saved_artistic_settings():
    """Transfer the actual user's settings instead of substituting recommended defaults."""
    root = Path(__file__).resolve().parents[1] / "workflows"
    old = json.loads((root / "Видео-в-Хронику.json").read_text())
    staged = json.loads((root / "Видео-в-Хронику-по-этапам.json").read_text())
    previous = {n["type"]: n for n in old["nodes"]}
    current = {n["type"]: n["widgets_values_named"] for n in staged["nodes"] if n["type"].startswith("VideoSilentFilm")}
    cadence = previous["VideoSilentFilmCadence"]["widgets_values_named"]
    finish = previous["VideoSilentFilmFinish"]["widgets_values_named"]
    camera = current["VideoSilentFilmCamera"]
    for key in ("source_fps", "target_fps_min", "target_fps_max", "fps_drift_strength", "crank_irregularity", "shutter_fraction", "motion_blur_strength", "blur_mode", "blur_samples", "camera_shake_strength"):
        assert camera[key] == cadence[key]
    for key in ("softness", "vignette_strength", "edge_softness", "edge_start", "edge_end", "focus_drift_strength"):
        assert camera[key] == finish[key]
    negative, positive, projector = (current["VideoSilentFilm" + stage] for stage in ("Negative", "Print", "Projector"))
    assert negative["grain_strength"] == finish["grain_strength"]
    assert negative["grain_size"] == finish["grain_size"]
    assert positive["tone_mode"] == finish["tone_mode"]
    assert projector["gate_weave_px"] == finish["gate_weave_px"]
    assert projector["lamp_flicker"] == finish["flicker_strength"]
    assert projector["lamp_breathing"] == finish["breathing_strength"]
    assert projector["projection_fps"] == 25
    for values in current.values():
        assert values["seed"] == previous["Seed"]["widgets_values"][0]
        assert values["batch_size"] == finish["batch_size"]
    assert camera["exposure_variation"] == negative["development_variation"] == positive["grain_strength"] == positive["print_softness"] == projector["softness"] == 0


def test_common_seed_uses_separate_effect_streams(pipeline):
    """Ensure actual stages feed distinct seeds to capture, temporal signals and both grains."""
    temporal_seeds, grain_seeds = [], []
    temporal = pipeline.finish._temporal_signal
    grain = pipeline._density_grain
    capture = pipeline.VideoSilentFilmCadence.apply
    capture_seeds = []
    def record_temporal(*args, **kwargs):
        """Record the stream actually consumed by the temporal generator."""
        temporal_seeds.append(args[1])
        return temporal(*args, **kwargs)
    def record_grain(*args, **kwargs):
        """Record the independent material stream, allowing repeated chunk calls."""
        grain_seeds.append(args[3])
        return grain(*args, **kwargs)
    def record_capture(*args, **kwargs):
        """Record the seed consumed by the existing cadence implementation."""
        capture_seeds.append(args[-1])
        return capture(*args, **kwargs)
    with patch.object(pipeline.finish, "_temporal_signal", side_effect=record_temporal), \
         patch.object(pipeline, "_density_grain", side_effect=record_grain), \
         patch.object(pipeline.VideoSilentFilmCadence, "apply", autospec=True, side_effect=record_capture):
        image, meta = camera(pipeline)
        image, meta = pipeline.VideoSilentFilmNegative().apply(image, meta, compute_device="cpu", batch_size=16)
        image, meta = pipeline.VideoSilentFilmPrint().apply(image, meta, compute_device="cpu", batch_size=16)
        _, _, meta = pipeline.VideoSilentFilmProjector().apply(image, meta, compute_device="cpu", batch_size=16)
    assert len(capture_seeds) == 1 and len(temporal_seeds) == 7 and len(grain_seeds) == 2
    used = capture_seeds + temporal_seeds + grain_seeds
    assert len(set(used)) == len(used)
    assert json.loads(meta)["seed_scheme"] == "blake2b_stage_effect_v1"
    assert all(entry["parameters"]["seed"] == 1925 for entry in json.loads(meta)["history"])
    same = pipeline._effect_seed(1925, "camera", "exposure")
    assert same == pipeline._effect_seed(1925, "camera", "exposure")
    assert same != pipeline._effect_seed(1926, "camera", "exposure")
    assert same != pipeline._effect_seed(1925, "projector", "lamp_flicker")
    first = pipeline.finish._temporal_signal(256, same, 1, 0.55, torch.device("cpu"), torch.float32)
    other = pipeline.finish._temporal_signal(256, pipeline._effect_seed(1925, "projector", "lamp_flicker"), 1, 0.55, torch.device("cpu"), torch.float32)
    assert not torch.equal(first, other)
    assert abs(float(torch.mean(first * other))) < 0.4


def test_workflow_has_one_common_fixed_seed():
    """Drive all four stages from one installed ComfyUI core integer primitive."""
    path = Path(__file__).resolve().parents[1] / "workflows/Видео-в-Хронику-по-этапам.json"
    workflow = json.loads(path.read_text())
    seed = next(n for n in workflow["nodes"] if n["type"] == "PrimitiveInt")
    assert seed["widgets_values"] == [1925, "fixed"]
    assert seed["widgets_values_named"]["value"] == 1925
    for node in workflow["nodes"]:
        if node["type"].startswith("VideoSilentFilm"):
            slot = next(i for i, inp in enumerate(node["inputs"]) if inp["name"] == "seed")
            assert [node["inputs"][slot]["link"], seed["id"], 0, node["id"], slot, "INT"] in workflow["links"]
