"""Validate disk IMAGE lifecycle, unchanged effects, and separate RAM/VRAM budgets."""

import gc
import errno
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def memory(tmp_path):
    """Use a temporary cache without importing the custom-node startup module."""
    if "ComfyUI_ALEXZ_tools" not in sys.modules:
        package = types.ModuleType("ComfyUI_ALEXZ_tools")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["ComfyUI_ALEXZ_tools"] = package
    module = importlib.import_module("ComfyUI_ALEXZ_tools.utils.video_memory")
    with patch.object(module, "CACHE_DIRECTORY", tmp_path):
        yield module


def test_video_memory_mapping_views_and_cleanup(memory, tmp_path):
    """Keep bytes valid through eviction and aliases, then release the mapping completely."""
    with patch.object(memory, "use_disk", return_value=True):
        output = memory.allocate_video((5, 11, 17, 3))
    address = output.untyped_storage().data_ptr()
    expected = torch.rand(output.shape)
    for start, end in ((0, 2), (2, 5)):
        output[start:end].copy_(expected[start:end])
        memory.release_video_pages(output, start, end, written=True)
    assert memory.disk_backed(output)
    alias = output[1:4].detach()
    del output
    gc.collect()
    assert memory.disk_backed(alias)
    memory.release_video_pages(alias)
    torch.testing.assert_close(alias, expected[1:4], atol=0, rtol=0)
    assert list(tmp_path.iterdir()) == []
    del alias
    gc.collect()
    assert address not in memory._MAPPINGS


def test_video_memory_disk_admission_and_small_ram_output(memory, tmp_path):
    """Reject missing disk space before mapping and keep tiny outputs in ordinary RAM."""
    small = memory.allocate_video((2, 3, 4, 3))
    assert not memory.disk_backed(small)
    assert memory.video_bytes((145, 1440, 1920, 3)) == 4810752000
    assert not memory.use_disk((145, 1440, 1920, 3), ram=32 * 1024**3)
    with patch.object(memory, "use_disk", return_value=True), patch.object(memory.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
        with pytest.raises(MemoryError, match="дисковому кэшу"):
            memory.allocate_video((2, 3, 4, 3))
    assert list(tmp_path.iterdir()) == []


def test_video_memory_reservation_failure_and_portable_fallback(memory, tmp_path):
    """Remove failed temporary files and reserve bounded blocks without fallocate."""
    with patch.object(memory, "use_disk", return_value=True), patch.object(memory.os, "posix_fallocate", side_effect=OSError(errno.ENOSPC, "disk full"), create=True):
        with pytest.raises(MemoryError, match="не удалось зарезервировать"):
            memory.allocate_video((3, 11, 17, 3))
    assert list(tmp_path.iterdir()) == []
    with patch.object(memory, "use_disk", return_value=True), patch.object(memory.os, "posix_fallocate", None, create=True):
        output = memory.allocate_video((3, 11, 17, 3))
        output.fill_(0.5)
        memory.release_video_pages(output, written=True)
        torch.testing.assert_close(output, torch.full(output.shape, 0.5))


@pytest.mark.parametrize("blur_mode", ["simple", "flow_integrated"])
def test_video_memory_pipeline_and_cadence_match_ram(memory, blur_mode):
    """Process the same seeded clip through every stage in RAM and on disk."""
    pipeline = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_pipeline")
    source = torch.rand((7, 25, 33, 4), generator=torch.Generator().manual_seed(1925))

    def run():
        """Capture and process a short RGBA clip, retaining all public outputs."""
        image, meta, preview = pipeline.VideoSilentFilmCamera().apply(source, blur_mode=blur_mode, compute_device="cpu")
        stages = [(image, meta, preview)]
        for cls in (pipeline.VideoSilentFilmNegative, pipeline.VideoSilentFilmPrint, pipeline.VideoSilentFilmProjector):
            result = cls().apply(image, meta, compute_device="cpu", batch_size=2)
            image, preview = result[0], result[-1]
            meta = result[2 if cls is pipeline.VideoSilentFilmProjector else 1]
            stages.append((image, meta, preview))
        return stages

    with patch.object(memory, "DISK_THRESHOLD", 2**60):
        ram = run()
    with patch.object(memory, "use_disk", return_value=True):
        disk = run()
    for expected, actual in zip(ram, disk):
        assert memory.disk_backed(actual[0])
        assert not memory.disk_backed(actual[2])
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
        assert actual[1] == expected[1]
    torch.testing.assert_close(source, torch.rand(source.shape, generator=torch.Generator().manual_seed(1925)), atol=0, rtol=0)


def test_video_memory_finish_matches_ram(memory):
    """Preserve legacy Finish image output when its destination is disk-backed."""
    finish = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_finish")
    image = torch.rand((5, 25, 33, 4))
    parameters = {name: spec[1]["default"] for name, spec in finish.VideoSilentFilmFinish.INPUT_TYPES()["required"].items() if name != "image"}
    parameters.update(compute_device="cpu", batch_size=2)
    expected, _ = finish.VideoSilentFilmFinish().apply(image, **parameters)
    with patch.object(memory, "use_disk", return_value=True):
        actual, _ = finish.VideoSilentFilmFinish().apply(image, **parameters)
    assert memory.disk_backed(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_video_memory_spills_gpu_output_when_storage_budget_is_insufficient(memory):
    """Use CPU mapped storage instead of exhausting CUDA with another full video batch."""
    with patch.object(memory, "use_disk", return_value=True), patch.object(memory, "available_vram", return_value=0):
        output = memory.allocate_video((3, 11, 17, 3), device="cuda:0", work_device="cuda:0", work_bytes=1024**3)
    assert output.device.type == "cpu" and memory.disk_backed(output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_video_memory_uses_real_vram_with_working_headroom(memory):
    """Store a full result on CUDA when space for the working portion remains."""
    with patch.object(memory, "DISK_THRESHOLD", 1):
        output = memory.allocate_video((3, 11, 17, 3), work_device="cuda:0", work_bytes=1024**2)
    assert output.device.type == "cuda" and not memory.disk_backed(output)
    output.fill_(0.25)
    torch.testing.assert_close(output.cpu(), torch.full(output.shape, 0.25))


def test_video_memory_gpu_plan_does_not_require_full_cpu_workspace(memory):
    """Allow CUDA when RAM fits transfer windows but not a full output or CPU effect frame."""
    finish = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_finish")
    image = torch.zeros((1, 1, 1, 3)).expand(145, 1440, 1920, 3)
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(get_torch_device=lambda: torch.device("cuda:0"))
    with patch.dict(sys.modules, {"comfy": comfy}), patch.object(finish, "_available_finish_ram", return_value=768 * 1024**2):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "mem_get_info", return_value=(22 * 1024**3, 24 * 1024**3)):
            with patch.object(torch.cuda, "memory_reserved", return_value=0), patch.object(torch.cuda, "memory_allocated", return_value=0):
                device, size, cpu_size = finish._finish_processing_plan(image, "cuda", 10, label="Silent Film Projector")
        assert device.type == "cuda" and 1 <= size <= 10 and cpu_size == 0
        with pytest.raises(MemoryError, match="Silent Film Projector.*RAM"):
            finish._finish_processing_plan(image, "cpu", 10, label="Silent Film Projector")


def test_video_memory_host_budget_includes_workspace(memory):
    """Keep a large cached batch in RAM unless it would consume reserved working memory."""
    shape = (181, 1248, 1660, 3)
    assert not memory.use_disk(shape, ram=24 * 1024**3, work_bytes=10 * 1024**3)
    assert memory.use_disk(shape, ram=12 * 1024**3, work_bytes=10 * 1024**3)
    assert memory.use_disk(shape, ram=8 * 1024**3)


def test_video_memory_uses_ram_when_vram_is_full(memory, monkeypatch):
    """Use available host RAM before allocating another disk mapping."""
    monkeypatch.setattr(memory, "DISK_THRESHOLD", 1)
    monkeypatch.setattr(memory, "available_vram", lambda device: 0)
    monkeypatch.setattr(memory, "available_ram", lambda: 8 * 1024**3)
    output = memory.allocate_video((3, 11, 17, 3), device="cuda:0", work_device="cuda:0")
    assert output.device.type == "cpu" and not memory.disk_backed(output)
    output.fill_(0.25)
    torch.testing.assert_close(output, torch.full(output.shape, 0.25))


def test_video_memory_cuda_allocation_race_falls_back_to_ram(memory, monkeypatch):
    """Preserve a useful RAM tier if another process consumes the observed VRAM headroom."""
    original_empty = torch.empty
    def empty(*args, **kwargs):
        """Simulate an allocation-time CUDA OOM without requiring a GPU."""
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            raise torch.OutOfMemoryError("simulated allocation race")
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(memory, "DISK_THRESHOLD", 1)
    monkeypatch.setattr(memory, "available_vram", lambda device: 8 * 1024**3)
    monkeypatch.setattr(memory, "available_ram", lambda: 8 * 1024**3)
    monkeypatch.setattr(torch, "empty", empty)
    output = memory.allocate_video((3, 11, 17, 3), work_device="cuda:0")
    assert output.device.type == "cpu" and not memory.disk_backed(output)


def test_video_memory_gpu_computes_when_only_disk_output_fits(memory):
    """Reserve CUDA for working chunks even when cached GPU inputs rule out a GPU output."""
    finish = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_finish")
    image = types.SimpleNamespace(shape=(145, 1440, 1920, 3), device=torch.device("cuda:0"))
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.SimpleNamespace(get_torch_device=lambda: torch.device("cuda:0"))
    with patch.dict(sys.modules, {"comfy": comfy}), patch.object(finish, "_available_finish_ram", return_value=768 * 1024**2):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "mem_get_info", return_value=(4 * 1024**3, 24 * 1024**3)):
            with patch.object(torch.cuda, "memory_reserved", return_value=0), patch.object(torch.cuda, "memory_allocated", return_value=0):
                device, size, cpu_size = finish._finish_processing_plan(image, "cuda", 10)
    assert device.type == "cuda" and size == 2 and cpu_size == 0


def test_video_memory_reports_unavailable_cpu_fallback(memory):
    """Stop safely instead of retrying an empty CPU chunk after a one-frame CUDA OOM."""
    pipeline = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_pipeline")
    image = torch.zeros((2, 11, 17, 3))
    with patch.object(pipeline.finish, "_finish_processing_plan", return_value=(torch.device("cuda:0"), 1, 0)):
        with patch.object(pipeline, "_stage_chunk", side_effect=torch.OutOfMemoryError("test")):
            with pytest.raises(MemoryError, match="Silent Film Projector.*CPU fallback"):
                pipeline._run_stage(image, "Silent Film Projector", "cuda", 10, lambda chunk, start, end: chunk)


def test_video_memory_cadence_normalizes_only_sampled_frames(memory):
    """Clamp before interpolation without duplicating the full input clip."""
    cadence = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_cadence")
    image = torch.tensor([-1.0, 2.0], dtype=torch.float64).view(2, 1, 1, 1).expand(2, 3, 5, 3)
    validated = cadence._validate_video_batch(image)
    assert validated.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()
    torch.testing.assert_close(cadence._lerp_frame(validated, 0.25), torch.full((3, 5, 3), 0.25))
    for dtype in (torch.float32, torch.float64):
        source = image.to(dtype).clone()
        original = source.clone()
        for position, value in ((0.0, 0.0), (1.0, 1.0)):
            sampled = cadence._lerp_frame_flow(source, position, {}, {})
            torch.testing.assert_close(sampled, torch.full((3, 5, 3), value))
            sampled.fill_(0.5)
            torch.testing.assert_close(source, original, atol=0, rtol=0)
