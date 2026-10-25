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
        vignette_strength=0.12, edge_softness=0.2, compute_device=device, batch_size=batch_size)[:2]


@pytest.mark.parametrize("frame_count", [1, 5])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required"))])
def test_camera_captures_color_as_monochrome(pipeline, frame_count, channels, device):
    """Record color patches as distinct neutral grays in stills and clips, preserving alpha."""
    colors = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).repeat_interleave(3, dim=0)
    source = colors.view(1, 1, 9, 3).expand(frame_count, 7, 9, 3).clone()
    if channels == 4:
        alpha = torch.linspace(0.2, 0.8, 9).view(1, 1, 9, 1).expand(frame_count, 7, 9, 1)
        source = torch.cat((source, alpha), dim=-1)
    original = source.clone()
    output, meta, preview = pipeline.VideoSilentFilmCamera().apply(
        source, source_fps=24, target_fps_min=24, target_fps_max=24,
        motion_blur_strength=0, camera_shake_strength=0, blur_mode="simple",
        softness=0, vignette_strength=0, edge_softness=0,
        focus_drift_strength=0, exposure_variation=0, compute_device=device, batch_size=2)
    expected = torch.tensor([0.299, 0.587, 0.114]).repeat_interleave(3).view(1, 1, 9)
    torch.testing.assert_close(output[..., 0], expected.expand(len(output), 7, 9), atol=1e-6, rtol=0)
    torch.testing.assert_close(output[..., 0], output[..., 1], atol=0, rtol=0)
    torch.testing.assert_close(output[..., 1], output[..., 2], atol=0, rtol=0)
    if channels == 4:
        torch.testing.assert_close(output[..., 3], original[:1, ..., 3].expand(len(output), 7, 9), atol=1e-6, rtol=0)
    torch.testing.assert_close(preview, output[len(output)//2:len(output)//2+1], atol=0, rtol=0)
    torch.testing.assert_close(source, original, atol=0, rtol=0)
    assert json.loads(meta)["history"][-1]["parameters"]["capture_color_mode"] == "monochrome_luma"


def test_full_pipeline_contract_and_alpha(pipeline):
    """Preserve frame metadata and alpha through all four physical stages."""
    captured, meta = camera(pipeline)
    assert len(captured) < 8
    negative, neg_meta, *_preview = pipeline.VideoSilentFilmNegative().apply(captured, meta, compute_device="cpu")
    positive, pos_meta, *_preview = pipeline.VideoSilentFilmPrint().apply(negative, neg_meta, compute_device="cpu")
    projected, fps, out_meta, *_preview = pipeline.VideoSilentFilmProjector().apply(positive, pos_meta, gate_weave_px=0, softness=0, lamp_flicker=0, lamp_breathing=0, compute_device="cpu")
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
    negative, neg_meta, *_preview = pipeline.VideoSilentFilmNegative().apply(captured, meta, grain_strength=0, development_variation=0, compute_device="cpu")
    assert torch.all(torch.diff(negative[0, 16, :, 0]) <= 0)
    printer = pipeline.VideoSilentFilmPrint()
    positive, _, *_preview = printer.apply(negative, neg_meta, grain_strength=0, print_softness=0, compute_device="cpu")
    assert torch.all(torch.diff(positive[0, 16, :, 0]) >= 0)
    scratched = negative.clone()
    scratched[:, :, 24:25, :3] = 1.0
    defect, _, *_preview = printer.apply(scratched, neg_meta, grain_strength=0, print_softness=0, compute_device="cpu")
    assert torch.all(defect[:, :, 24, :3] < positive[:, :, 24, :3])
    torch.testing.assert_close(defect[:, :, :24], positive[:, :, :24])
    dense, _, *_preview = pipeline.VideoSilentFilmNegative().apply(captured, meta, exposure_ev=1, grain_strength=0, development_variation=0, compute_device="cpu")
    assert dense[..., :3].mean() < negative[..., :3].mean()
    darker_print, _, *_preview = printer.apply(negative, neg_meta, print_exposure_ev=1, grain_strength=0, print_softness=0, compute_device="cpu")
    assert darker_print.mean() < positive.mean()


@pytest.mark.parametrize("stage", ["negative", "print", "projector"])
def test_stages_reject_wrong_metadata_and_bad_parameters(pipeline, stage):
    """Prevent silent misuse of stage polarity, malformed metadata, or changed frame counts."""
    captured, meta = camera(pipeline)
    classes = {"negative": pipeline.VideoSilentFilmNegative, "print": pipeline.VideoSilentFilmPrint, "projector": pipeline.VideoSilentFilmProjector}
    node = classes[stage]()
    for text in ("not json", "[]", "{}", '{"schema_name":"wrong"}'):
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


@pytest.mark.parametrize("stage,previous_stage", [("negative", "camera"), ("print", "negative"), ("projector", "print")])
def test_optional_metadata_preserves_stage_effects(pipeline, stage, previous_stage):
    """Run disconnected inputs identically without inventing upstream history or cadence."""
    classes = {"negative": pipeline.VideoSilentFilmNegative, "print": pipeline.VideoSilentFilmPrint, "projector": pipeline.VideoSilentFilmProjector}
    node = classes[stage]()
    specs = node.INPUT_TYPES()
    assert "film_json" not in specs["required"]
    assert specs["optional"]["film_json"][0] == "STRING"
    assert specs["optional"]["film_json"][1]["forceInput"] is True
    image = torch.rand((3, 17, 25, 4), generator=torch.Generator().manual_seed(42))
    supplied = pipeline._write_film_meta({"history": [], "cadence": {"known": True}}, image, previous_stage, {})
    expected = node.apply(image, supplied, compute_device="cpu", batch_size=2)
    for kwargs in ({}, {"film_json": None}, {"film_json": ""}, {"film_json": "  \n"}):
        result = node.apply(image, compute_device="cpu", batch_size=2, **kwargs)
        torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(result[-1], expected[-1], atol=0, rtol=0)
        data = json.loads(result[2 if stage == "projector" else 1])
        assert data["schema_name"] == "alexz.silent_film_pipeline"
        assert data["image_shape"] == list(image.shape)
        assert data["stage"] == stage
        assert [entry["stage"] for entry in data["history"]] == [stage]
        assert data["cadence"] is None
        if stage == "projector":
            assert result[1] == expected[1] == 25


@pytest.mark.parametrize("use_negative,use_print", [(False, False), (True, False), (False, True)])
def test_bypassed_stages_preserve_cadence_and_real_history(pipeline, use_negative, use_print, caplog):
    """Forward bypassed metadata without inventing film effects or missing history entries."""
    image, meta = camera(pipeline)
    original_cadence = json.loads(meta)["cadence"]
    stages = ["camera"]
    with caplog.at_level("INFO", logger="SilentFilmPipeline"):
        if use_negative:
            image, meta, _ = pipeline.VideoSilentFilmNegative().apply(image, meta, compute_device="cpu")
            stages.append("negative")
        if use_print:
            image, meta, _ = pipeline.VideoSilentFilmPrint().apply(image, meta, compute_device="cpu")
            stages.append("print")
        settings = dict(gate_weave_px=0, softness=0, lamp_flicker=0, lamp_breathing=0, compute_device="cpu")
        projected, fps, final_meta, preview = pipeline.VideoSilentFilmProjector().apply(image, meta, **settings)
    torch.testing.assert_close(projected, image, atol=0, rtol=0)
    torch.testing.assert_close(preview, projected[len(projected)//2:len(projected)//2+1])
    data = json.loads(final_meta)
    assert data["cadence"] == original_cadence
    assert [entry["stage"] for entry in data["history"]] == stages + ["projector"]
    assert fps == 25 and "Пропуск этапов" in caplog.text


def test_bypass_metadata_still_rejects_stale_shapes_and_unknown_stages(pipeline):
    """Keep genuine metadata failures visible even when forward-stage skipping is allowed."""
    image, meta = camera(pipeline)
    projector = pipeline.VideoSilentFilmProjector()
    with pytest.raises(ValueError, match="image_shape"):
        projector.apply(image[:1], meta, compute_device="cpu")
    data = json.loads(meta)
    data["stage"] = "unknown"
    with pytest.raises(ValueError, match="Expected film stage"):
        projector.apply(image, json.dumps(data), compute_device="cpu")


@pytest.mark.parametrize("bypass_mask", range(16))
def test_all_stage_bypass_combinations(pipeline, bypass_mask):
    """Run every subset of four film stages, including no Camera and all stages bypassed."""
    source = torch.rand((7, 25, 33, 4), generator=torch.Generator().manual_seed(7))
    image, meta, stages = source, None, []
    if not bypass_mask & 1:
        image, meta = camera(pipeline, source)
        stages.append("camera")
    for bit, cls, stage in ((2, pipeline.VideoSilentFilmNegative, "negative"),
                            (4, pipeline.VideoSilentFilmPrint, "print"),
                            (8, pipeline.VideoSilentFilmProjector, "projector")):
        if bypass_mask & bit:
            continue
        result = cls().apply(image, meta, compute_device="cpu")
        image = result[0]
        meta = result[2 if stage == "projector" else 1]
        stages.append(stage)
        torch.testing.assert_close(result[-1], image[len(image)//2:len(image)//2+1])
    assert torch.isfinite(image).all() and image.shape[1:] == source.shape[1:]
    if stages:
        data = json.loads(meta)
        assert [entry["stage"] for entry in data["history"]] == stages
        assert (data["cadence"] is None) == bool(bypass_mask & 1)
    else:
        assert image is source and meta is None
    torch.testing.assert_close(source, torch.rand(source.shape, generator=torch.Generator().manual_seed(7)), atol=0, rtol=0)


@pytest.mark.parametrize("count", [1, 4, 5])
def test_preview_frame_encodes_one_middle_frame_and_forwards_full_batch(pipeline, count):
    """Keep a preview branch bounded after bypass while retaining core IMAGE pass-through."""
    module = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_preview")
    image = torch.rand((count, 7, 11, 3))
    calls = []
    class CorePreview:
        def save_images(self, images, **kwargs):
            """Record the real core-preview call contract without writing PNG fixtures."""
            calls.append(images)
            return {"ui": {"images": [{"filename": "middle.png", "type": "temp"}]}, "result": (images,)}
    with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(PreviewImage=CorePreview)}):
        result = module.VideoPreviewFrame().save_images(image)
    assert result["result"][0] is image
    assert len(result["ui"]["images"]) == len(calls[0]) == 1
    torch.testing.assert_close(calls[0], image[count//2:count//2+1])


def test_pipeline_can_continue_from_standalone_negative(pipeline):
    """Forward newly generated metadata through later stages after a disconnected start."""
    image = torch.rand((2, 17, 25, 3))
    negative, meta, _ = pipeline.VideoSilentFilmNegative().apply(image, compute_device="cpu")
    positive, meta, _ = pipeline.VideoSilentFilmPrint().apply(negative, meta, compute_device="cpu")
    output, fps, meta, _ = pipeline.VideoSilentFilmProjector().apply(positive, meta, compute_device="cpu")
    data = json.loads(meta)
    assert data["cadence"] is None
    assert [entry["stage"] for entry in data["history"]] == ["negative", "print", "projector"]
    assert output.shape == image.shape and fps == 25


def test_grain_chunks_and_devices(pipeline):
    """Keep independent grain layers reproducible across chunk boundaries and real CUDA."""
    captured, meta = camera(pipeline)
    negative_node, print_node = pipeline.VideoSilentFilmNegative(), pipeline.VideoSilentFilmPrint()
    negative, neg_meta, *_preview = negative_node.apply(captured, meta, compute_device="cpu", batch_size=1)
    positive, pos_meta, *_preview = print_node.apply(negative, neg_meta, compute_device="cpu", batch_size=1)
    projector = pipeline.VideoSilentFilmProjector()
    projected, _, _, *_preview = projector.apply(positive, pos_meta, compute_device="cpu", batch_size=1, vignette_strength=0.1, edge_softness=0.2)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for size in (2, 4):
            actual, _, *_preview = negative_node.apply(captured, meta, compute_device=device, batch_size=size)
            torch.testing.assert_close(actual, negative, atol=1e-4, rtol=1e-4)
            actual, _, *_preview = print_node.apply(negative, neg_meta, compute_device=device, batch_size=size)
            torch.testing.assert_close(actual, positive, atol=1e-4, rtol=1e-4)
            actual, fps, _, *_preview = projector.apply(positive, pos_meta, compute_device=device, batch_size=size, vignette_strength=0.1, edge_softness=0.2)
            torch.testing.assert_close(actual, projected, atol=1e-4, rtol=1e-4)
            assert fps == 25
    other_negative, _, *_preview = negative_node.apply(captured, meta, seed=11, compute_device="cpu")
    assert (negative - other_negative).abs().mean() > 1e-5
    other_print, _, *_preview = print_node.apply(negative, neg_meta, seed=11, compute_device="cpu")
    assert (positive - other_print).abs().mean() > 1e-5


def test_projector_optics_are_stationary(pipeline):
    """Move the print and its defects before applying the fixed projector vignette."""
    captured, meta = camera(pipeline)
    negative, meta, *_preview = pipeline.VideoSilentFilmNegative().apply(captured, meta, compute_device="cpu")
    positive, meta, *_preview = pipeline.VideoSilentFilmPrint().apply(negative, meta, compute_device="cpu")
    positive = torch.full_like(positive, 0.6)
    positive[..., 3] = 0.7
    actual, fps, _, *_preview = pipeline.VideoSilentFilmProjector().apply(positive, meta, projection_fps=18, gate_weave_px=3, softness=0,
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
        inputs = cls.INPUT_TYPES()
        schema = inputs["required"]
        all_inputs = {**schema, **inputs.get("optional", {})}
        assert schema["seed"][1]["control_after_generate"] is True
        # В сохранённом workflow необязательный разъём остаётся на прежнем месте.
        names = [i["name"] for i in node["inputs"]]
        assert len(names) == len(all_inputs) and set(names) == set(all_inputs)
        assert [name for name in names if name in schema] == list(schema)
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
        image, meta, *_preview = pipeline.VideoSilentFilmNegative().apply(image, meta, compute_device="cpu", batch_size=16)
        image, meta, *_preview = pipeline.VideoSilentFilmPrint().apply(image, meta, compute_device="cpu", batch_size=16)
        _, _, meta, *_preview = pipeline.VideoSilentFilmProjector().apply(image, meta, compute_device="cpu", batch_size=16)
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


def test_camera_full_preview_does_not_change_main_outputs(pipeline):
    """Append a detached middle frame and preserve existing output slot indices."""
    image = torch.rand((8, 33, 49, 3))
    output, meta, preview = pipeline.VideoSilentFilmCamera().apply(image, blur_mode="simple", blur_samples=3, compute_device="cpu")
    middle = len(output) // 2
    torch.testing.assert_close(preview, output[middle:middle + 1])
    original = output.clone()
    preview.zero_()
    torch.testing.assert_close(output, original)
    assert pipeline.VideoSilentFilmCamera.RETURN_NAMES[:2] == ("image", "film_json")
    assert pipeline.VideoSilentFilmProjector.RETURN_NAMES[:3] == ("image", "fps", "film_json")


def test_frame_selection_counts_bounds_and_inclusive_range(pipeline):
    """Expose true source counts and copy inclusive ranges without changing pixels."""
    selector = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_frame_select").VideoFrameSelect()
    assert selector.OUTPUT_NODE is True
    image = torch.rand((9, 7, 11, 4))
    for mode, settings, start, end in (
        ("full_video", {}, 0, 8), ("single_frame", {"frame_index": 4}, 4, 4),
        ("frame_range", {"start_frame": 2, "end_frame": 5}, 2, 5),
        ("frame_range", {"start_frame": 6, "end_frame": -1}, 6, 8),
        ("frame_range", {"start_frame": 3, "end_frame": 3}, 3, 3),
    ):
        response = selector.select(image, mode, **settings)
        selected, total, count, meta = response["result"]
        torch.testing.assert_close(selected, image[start:end + 1], atol=0, rtol=0)
        assert (selected.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()) == (start == 0 and end == 8)
        assert total == 9 and count == end - start + 1
        data = json.loads(meta)
        assert data["middle_source_frame"] == start + count // 2
        assert str(total) in response["ui"]["text"][0]
        assert response["ui"]["video_info"] == [{"total_frames": 9, "width": 11, "height": 7}]
    for mode, settings in (
        ("single_frame", {"frame_index": 9}), ("single_frame", {"frame_index": -1}),
        ("single_frame", {"frame_index": 2.5}), ("single_frame", {"frame_index": True}),
        ("frame_range", {"start_frame": 5, "end_frame": 2}),
        ("frame_range", {"end_frame": 9}), ("frame_range", {"end_frame": -1.0}),
        ("unknown", {}),
    ):
        with pytest.raises(ValueError):
            selector.select(image, mode, **settings)


def test_single_selected_frame_runs_entire_pipeline_without_mode(pipeline):
    """Automatically handle one source frame and retain consistent middle previews."""
    selector = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_frame_select").VideoFrameSelect()
    source = torch.rand((11, 33, 49, 4))
    selected = selector.select(source, "single_frame", frame_index=5)["result"][0]
    with patch.object(pipeline.VideoSilentFilmCadence, "apply", side_effect=AssertionError("single frame cannot run cadence")):
        image, meta, preview = pipeline.VideoSilentFilmCamera().apply(selected, compute_device="cpu")
    assert json.loads(meta)["cadence"]["single_frame"] is True
    for cls in (pipeline.VideoSilentFilmNegative, pipeline.VideoSilentFilmPrint, pipeline.VideoSilentFilmProjector):
        result = cls().apply(image, meta, compute_device="cpu")
        image, preview = result[0], result[-1]
        meta = result[2 if cls is pipeline.VideoSilentFilmProjector else 1]
        assert len(image) == len(preview) == 1
        torch.testing.assert_close(preview, image)
        assert "processing_mode" not in cls.INPUT_TYPES()["required"]
    assert "processing_mode" not in pipeline.VideoSilentFilmCamera.INPUT_TYPES()["required"]


def test_selected_segment_midpoint_is_shared_by_stages(pipeline):
    """Process the complete selected clip and preview its middle output exposure at every stage."""
    selector = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_frame_select").VideoFrameSelect()
    source = torch.rand((20, 33, 49, 3))
    selected = selector.select(source, "frame_range", start_frame=4, end_frame=15)["result"][0]
    assert len(selected) == 12
    result = pipeline.VideoSilentFilmCamera().apply(selected, blur_mode="simple", blur_samples=3, compute_device="cpu")
    image, meta, preview = result
    count = len(image)
    torch.testing.assert_close(preview, image[count // 2:count // 2 + 1])
    for cls in (pipeline.VideoSilentFilmNegative, pipeline.VideoSilentFilmPrint, pipeline.VideoSilentFilmProjector):
        result = cls().apply(image, meta, compute_device="cpu")
        image, preview = result[0], result[-1]
        meta = result[2 if cls is pipeline.VideoSilentFilmProjector else 1]
        assert len(image) == count
        torch.testing.assert_close(preview, image[count // 2:count // 2 + 1])
    workflow = json.loads((Path(__file__).resolve().parents[1] / "workflows/Видео-в-Хронику-по-этапам.json").read_text())
    by_id = {n["id"]: n for n in workflow["nodes"]}
    camera_node = next(n for n in workflow["nodes"] if n["type"] == "VideoSilentFilmCamera")
    camera_link = next(l for l in workflow["links"] if l[3] == camera_node["id"] and l[4] == 0)
    assert by_id[camera_link[1]]["type"] == "VideoLoadSource"
