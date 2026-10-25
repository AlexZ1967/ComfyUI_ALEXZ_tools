"""Exercise real ProRes decoding, frame selection, export, proxies, and failure cleanup."""

import asyncio
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import torch


@pytest.fixture
def io(tmp_path, monkeypatch):
    """Bind the nodes to temporary ComfyUI input/output directories."""
    root = Path(__file__).resolve().parents[1]
    if "ComfyUI_ALEXZ_tools" not in sys.modules:
        package = types.ModuleType("ComfyUI_ALEXZ_tools")
        package.__path__ = [str(root)]
        monkeypatch.setitem(sys.modules, "ComfyUI_ALEXZ_tools", package)
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir(); output_dir.mkdir()
    paths = types.ModuleType("folder_paths")
    paths.get_annotated_filepath = lambda value: str(input_dir / value)
    paths.get_input_directory = lambda: str(input_dir)
    paths.get_output_directory = lambda: str(output_dir)
    def save_path(prefix, output, width, height):
        """Provide the inspected core path helper's directory/counter contract."""
        target = (Path(output) / prefix).resolve()
        if not target.is_relative_to(Path(output)):
            raise ValueError("outside output")
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target.parent), target.name, 1, str(target.parent.relative_to(output)).replace(".", ""), prefix
    paths.get_save_image_path = save_path
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    nodes = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_io")
    utils = importlib.import_module("ComfyUI_ALEXZ_tools.utils.video_io")
    memory = importlib.import_module("ComfyUI_ALEXZ_tools.utils.video_memory")
    monkeypatch.setattr(memory, "CACHE_DIRECTORY", tmp_path / "cache")
    return nodes, utils, memory, input_dir, output_dir


@pytest.fixture
def source(io):
    """Generate six real ProRes HQ frames with a Unicode filename."""
    _, utils, _, input_dir, _ = io
    path = input_dir / "Проверка ProRes.mov"
    result = subprocess.run([utils.tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=32x24:rate=24:duration=0.25", "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-threads", "2", str(path)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    return path


def test_video_io_prores_probe_selection_and_disk_cache(io, source, monkeypatch):
    nodes, utils, memory, _, _ = io
    info = utils.probe_video(source)
    assert info["total_frames"] == 6 and info["count_is_exact"]
    assert info["fps"] == 24 and info["codec"] == "prores" and info["pixel_format"] == "yuv422p10le"
    full = nodes.VideoLoadSource().load(source.name)["result"][0]
    assert full.shape == (6, 24, 32, 3) and full.dtype == torch.float32
    assert torch.abs(full * 255 - (full * 255).round()).max() > 0.01
    monkeypatch.setattr(memory, "DISK_THRESHOLD", 1)
    for mode, arguments, start, end in (
        ("frame_range", {"start_frame": 2, "end_frame": 4}, 2, 4),
        ("single_frame", {"frame_index": 5}, 5, 5),
        ("frame_range", {"start_frame": 3, "end_frame": -1}, 3, 5),
    ):
        result = nodes.VideoLoadSource().load(str(source), mode, **arguments)["result"]
        image, fps, total, selected, meta, preview = result
        assert memory.disk_backed(image)
        torch.testing.assert_close(image, full[start:end + 1], atol=0, rtol=0)
        torch.testing.assert_close(preview, image[len(image)//2:len(image)//2+1], atol=0, rtol=0)
        assert (fps, total, selected) == (24, 6, end-start+1)
        assert json.loads(meta)["middle_source_frame"] == start + selected//2
    with pytest.raises(ValueError):
        nodes.VideoLoadSource().load(str(source), "single_frame", frame_index=6)


def test_video_io_counts_frames_without_container_counter(io, source):
    """Count a real Matroska stream when nb_frames is unavailable in its header."""
    nodes, utils, _, _, _ = io
    target = source.with_suffix(".mkv")
    result = subprocess.run([utils.tool("ffmpeg"), "-v", "error", "-y", "-i", str(source), "-c:v", "libx264", "-threads", "2", str(target)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    info = utils.probe_video(target)
    assert info["total_frames"] == 6 and info["count_source"] == "decoded_count" and info["count_is_exact"]
    image = nodes.VideoLoadSource().load(str(target), "frame_range", start_frame=4, end_frame=5)["result"][0]
    assert len(image) == 2


@pytest.mark.parametrize("video_format,codec", [("prores_422_hq", "prores"), ("prores_4444", "prores"), ("h264_mp4", "h264")])
def test_video_io_export_roundtrip_and_numbering(io, source, video_format, codec):
    nodes, utils, _, _, _ = io
    image = nodes.VideoLoadSource().load(str(source))["result"][0]
    first = nodes.VideoSaveFinal().save(image, 25, "Плёнка/Финал", video_format, create_preview=False)
    path = Path(first["result"][0])
    assert path.is_file() and path.stat().st_size > 0
    info = utils.probe_video(path)
    assert info["codec"] == codec and info["total_frames"] == 6 and info["fps"] == 25
    original = path.read_bytes()
    second = nodes.VideoSaveFinal().save(image, 25, "Плёнка/Финал", video_format, create_preview=False)
    assert second["result"][0] != str(path) and path.read_bytes() == original
    metadata = json.loads(first["result"][1])
    assert metadata["duration_seconds"] == 6/25


def test_video_io_alpha_and_browser_proxy(io):
    nodes, utils, _, _, _ = io
    image = torch.rand((4, 24, 32, 4), generator=torch.Generator().manual_seed(4))
    image[..., 3] = torch.linspace(0.2, 0.8, 32)
    result = nodes.VideoSaveFinal().save(image, format="prores_4444", create_preview=True)
    path = Path(result["result"][0])
    assert utils.probe_video(path)["channels"] == 4
    restored = nodes.VideoLoadSource().load(str(path))["result"][0]
    torch.testing.assert_close(restored[..., 3], image[..., 3], atol=0.01, rtol=0)
    video = result["ui"]["alexz_videos"][0]
    preview = path.parent / video["filename"]
    assert preview.exists() and utils.probe_video(preview)["codec"] == "h264"
    assert utils.probe_video(path)["codec"] == "prores"


@pytest.mark.parametrize("video_format", ["prores_422_hq", "h264_mp4"])
def test_video_io_rec709_color_roundtrip(io, video_format):
    """Apply the Rec.709 conversion matrix as well as writing matching color tags."""
    nodes, _, _, _, _ = io
    image = torch.tensor([0.25, 0.55, 0.8]).view(1, 1, 1, 3).expand(2, 24, 32, 3)
    path = nodes.VideoSaveFinal().save(image, format=video_format, create_preview=False)["result"][0]
    decoded = nodes.VideoLoadSource().load(path)["result"][0]
    torch.testing.assert_close(decoded, image, atol=0.012, rtol=0)


def test_video_io_failed_export_leaves_no_partial_files(io):
    nodes, _, _, _, output_dir = io
    image = torch.full((2, 24, 32, 3), float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        nodes.VideoSaveFinal().save(image)
    assert not [p for p in output_dir.rglob("*") if p.is_file()]
    with pytest.raises(ValueError):
        nodes.VideoSaveFinal().save(torch.zeros((2, 25, 33, 3)), format="h264_mp4")
    with pytest.raises(ValueError):
        nodes.VideoSaveFinal().save(torch.zeros((2, 24, 32, 3)), filename_prefix="../outside")


def test_video_io_probe_route_does_not_queue_workflow(io, source):
    api = importlib.import_module("ComfyUI_ALEXZ_tools.utils.video_io_api")
    async def run():
        """Exercise the actual read-only probe endpoint through loopback HTTP."""
        routes = web.RouteTableDef()
        api.register_routes(types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)), web)
        application = web.Application(); application.add_routes(routes)
        async with TestClient(TestServer(application)) as client:
            response = await client.get("/alexz/video/probe", params={"path": str(source)})
            assert response.status == 200 and (await response.json())["total_frames"] == 6
            response = await client.get("/alexz/video/probe", params={"path": str(source)}, headers={"Origin": "https://example.invalid"})
            assert response.status == 403
            headers = {"Content-Type": "application/octet-stream", "X-Alexz-Video": "1"}
            for index in range(2):
                response = await client.post("/alexz/video/upload", params={"filename": source.name}, data=source.read_bytes(), headers=headers)
                assert response.status == 200
                uploaded = await response.json()
                assert (source.parent / uploaded["subfolder"] / uploaded["name"]).read_bytes() == source.read_bytes()
                assert ("_1" in uploaded["name"]) == (index == 1)
            response = await client.post("/alexz/video/upload", params={"filename": "../outside.mov"}, data=b"no", headers=headers)
            assert response.status == 400
            assert not list(source.parent.rglob(".upload-*.tmp"))
    asyncio.run(run())
