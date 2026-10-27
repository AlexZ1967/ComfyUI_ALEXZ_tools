"""Validate preflight frame counts, fanout, disabled effects and tier budgets."""

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch


@pytest.fixture
def planner(tmp_path, monkeypatch):
    """Load a planner without ComfyUI startup or hardware resource dependencies."""
    name = "ComfyUI_ALEXZ_tools"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        monkeypatch.setitem(sys.modules, name, package)
    module = importlib.import_module(f"{name}.utils.video_memory_plan")
    monkeypatch.setattr(module.memory, "CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr(module.memory, "available_ram", lambda: 24 * 1024**3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return module


def source_info():
    """Describe the user's clip without reading or decoding their video."""
    return dict(width=1660, height=1248, channels=3, total_frames=241, fps=24.)


def test_preflight_counts_camera_exactly_and_deduplicates_fanout(planner):
    """Match real cadence timing while counting shared producers and disabled effects once."""
    prompt = {
        "1": {"class_type": "VideoLoadSource", "inputs": {}},
        "2": {"class_type": "VideoSilentFilmCamera", "inputs": {"image": ["1", 0], "source_fps": ["1", 1], "seed": 1925}},
        "3": {"class_type": "VideoSilentFilmNegative", "inputs": {"image": ["2", 0]}},
        "4": {"class_type": "VideoSilentFilmNegativeDamage", "inputs": {"image": ["3", 0], "strength": 0}},
        "5": {"class_type": "VideoSilentFilmNegativeDamage", "inputs": {"image": ["4", 0], "strength": 0.1}},
        "6": {"class_type": "VideoSilentFilmPrint", "inputs": {"image": ["4", 0]}},
        "7": {"class_type": "VideoSilentFilmNegative", "inputs": {"image": ["2", 2]}},
    }
    result = planner.estimate_graph(source_info(), 241, prompt, "1")
    allocations = {item["node"]: item for item in result["allocations"]}
    assert len(allocations) == 6 and "4" not in allocations
    assert allocations["7"]["shape"][0] == 1
    pipeline = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_pipeline")
    output = pipeline.VideoSilentFilmCamera().apply(torch.full((241, 3, 5, 3), 0.5), source_fps=24,
                                                    seed=1925, motion_blur_strength=0, blur_mode="simple", compute_device="cpu")[0]
    assert allocations["2"]["shape"][0] == len(output)
    assert allocations["2"]["bytes"] == len(output) * 1660 * 1248 * 3 * 4
    assert result["scratch_bytes"] == allocations["2"]["bytes"]
    assert not result["unknown_nodes"]


def test_preflight_single_frame_has_no_cadence_scratch(planner):
    """Treat still-image selection as one frame even when nominal fps differs."""
    prompt = {"2": {"class_type": "VideoSilentFilmCamera", "inputs": {"image": ["1", 0]}}}
    plan = planner.estimate_graph(source_info(), 1, prompt, "1")
    assert plan["scratch_bytes"] == 0
    assert [item["shape"][0] for item in plan["allocations"]] == [1, 1]


@pytest.mark.parametrize("kind,inputs", [
    ("UnknownResize", {"image": ["1", 0]}),
    ("VideoSilentFilmCamera", {"image": ["1", 0], "source_fps": ["opaque", 0]}),
])
def test_preflight_marks_unknown_boundary(planner, kind, inputs):
    """Never invent output dimensions or fps from an unknown producer."""
    prompt = {"2": {"class_type": kind, "inputs": inputs},
              "3": {"class_type": "VideoSilentFilmNegative", "inputs": {"image": ["2", 0]}}}
    plan = planner.estimate_graph(source_info(), 241, prompt, "1")
    assert plan["unknown_nodes"] == ["2"]
    assert len(plan["allocations"]) == 1


def test_preflight_respects_ram_budget_and_reports_low_disk(planner, monkeypatch, caplog):
    """Report overflow on the configured filesystem before any video allocation."""
    prompt = {str(index): {"class_type": "VideoSilentFilmNegative", "inputs": {"image": [str(index-1), 0]}}
              for index in range(2, 8)}
    plan = planner.estimate_graph(source_info(), 241, prompt, "1")
    monkeypatch.setattr(planner.memory, "available_ram", lambda: 12 * 1024**3)
    monkeypatch.setattr(planner.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=1024**3))
    checked = planner.check_resources(plan)
    placements = checked["estimated_placement"]
    assert placements["ram"] > 0 and placements["disk"] > 0 and placements["vram"] == 0
    assert sum(placements.values()) + plan["preview_bytes"] + plan["scratch_bytes"] == plan["peak_bytes"]
    assert "может не хватить диска" in caplog.text


def test_preflight_gpu_storage_preserves_working_budget(planner, monkeypatch):
    """Budget reusable VRAM blocks while keeping source decoding in RAM or on disk."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(get_torch_device=lambda: torch.device("cuda:0"))
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setattr(planner.memory, "available_vram", lambda device: 24 * 1024**3)
    prompt = {str(index): {"class_type": "VideoSilentFilmNegative", "inputs": {"image": [str(index-1), 0]}}
              for index in range(2, 8)}
    plan = planner.estimate_graph(source_info(), 241, prompt, "1")
    checked = planner.check_resources(plan)
    assert 0 < checked["estimated_placement"]["vram"] <= 24 * 1024**3 - plan["work_bytes"] - planner.memory.RAM_RESERVE
    assert checked["estimated_placement"]["ram"] >= plan["allocations"][0]["bytes"]


def test_preflight_only_checkpoints_accumulate_on_long_chain(planner):
    """Bound full transient storage to input/output while preserving both checkpoints."""
    prompt = {"2": {"class_type": "VideoSilentFilmCamera", "inputs": {"image": ["1", 0]}}}
    prompt.update({str(index): {"class_type": "VideoSilentFilmNegativeDamage",
                              "inputs": {"image": [str(index-1), 0], "strength": 0.1}}
                   for index in range(3, 23)})
    plan = planner.estimate_graph(source_info(), 241, prompt, "1")
    assert len(plan["allocations"]) == 22
    assert len(plan["storage_allocations"]) == 4
    assert plan["cached_bytes"] < plan["peak_bytes"] < plan["all_outputs_bytes"]
    assert not plan["branched_estimate"]
    fallback = planner.estimate_graph(source_info(), 241, prompt, "1", selective=False)
    assert len(fallback["storage_allocations"]) == 22


def test_preflight_alias_fanout_uses_conservative_storage_bound(planner):
    """Include branches introduced by a disabled defect that aliases its input storage."""
    prompt = {"2": {"class_type": "VideoSilentFilmNegative", "inputs": {"image": ["1", 0]}},
              "3": {"class_type": "VideoSilentFilmNegativeDamage", "inputs": {"image": ["2", 0], "strength": 0}}}
    prompt.update({str(index): {"class_type": "VideoSilentFilmPrint", "inputs": {"image": ["3", 0]}}
                   for index in range(4, 8)})
    plan = planner.estimate_graph(source_info(), 241, prompt, "1")
    assert plan["branched_estimate"]
    assert len(plan["storage_allocations"]) == len(plan["allocations"]) == 6
