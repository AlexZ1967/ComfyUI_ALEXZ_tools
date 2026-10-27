"""Validate the selective cache adapter without importing or changing ComfyUI core."""

import asyncio
import importlib
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def adapter(monkeypatch):
    """Import the project policy against a small cache-protocol fixture."""
    name = "ComfyUI_ALEXZ_tools"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        monkeypatch.setitem(sys.modules, name, package)
    return importlib.import_module(f"{name}.utils.video_cache")


class Cache:
    """Record native cache calls and provide the inspected dynamic-prompt interface."""

    def __init__(self, prompt):
        """Expose an ordinary cache and a read-only graph lookup."""
        self.cache = {}
        self.dynprompt = types.SimpleNamespace(has_node=lambda key: key in prompt, get_node=prompt.__getitem__)

    async def get(self, node_id):
        """Read a cached value."""
        return self.get_local(node_id)

    async def set(self, node_id, value):
        """Write a cached value."""
        self.set_local(node_id, value)

    def get_local(self, node_id):
        """Read without asynchronous provider lookups."""
        return self.cache.get(node_id)

    def set_local(self, node_id, value):
        """Simulate write-back on dependency access."""
        self.cache[node_id] = value

    async def ensure_subcache_for(self, node_id, children_ids):
        """Match the LRU subcache's same-instance behavior."""
        return self


def test_selective_cache_only_filters_fast_stages_and_consumers(adapter):
    """Keep unrelated nodes and checkpoints while excluding fast stages and their sinks."""
    prompt = {
        "source": {"class_type": "VideoLoadSource"},
        "camera": {"class_type": "VideoSilentFilmCamera", "inputs": {"image": ["source", 0]}},
        "negative": {"class_type": "VideoSilentFilmNegative", "inputs": {"image": ["camera", 0]}},
        "preview": {"class_type": "PreviewImage", "inputs": {"images": ["negative", 2]}},
        "save": {"class_type": "OtherSave", "inputs": {"image": ["negative", 0]}},
        "second_camera": {"class_type": "VideoSilentFilmCamera", "inputs": {"image": ["negative", 0]}},
        "unrelated": {"class_type": "OtherNode", "inputs": {}},
    }
    original = Cache(prompt)
    cache = adapter.FilmOutputCache(original)
    expected = {"source", "camera", "second_camera", "unrelated"}
    async def run():
        """Exercise both executor stores and dependency write-back paths."""
        for node_id in prompt:
            await cache.set(node_id, node_id)
            cache.set_local(node_id, node_id)
            assert (await cache.get(node_id) is not None) == (node_id in expected)
            assert (cache.get_local(node_id) is not None) == (node_id in expected)
        assert await cache.ensure_subcache_for("camera", []) is cache
    asyncio.run(run())
    assert set(original.cache) == expected


def test_selective_cache_suppresses_existing_fast_entries(adapter):
    """Do not expose an older fast result even if a native cache already contains it."""
    original = Cache({"fast": {"class_type": "VideoSilentFilmPrint"}})
    original.cache["fast"] = object()
    cache = adapter.FilmOutputCache(original)
    assert cache.get_local("fast") is None
    assert asyncio.run(cache.get("fast")) is None


def test_selective_cache_ephemeral_children_keep_the_filter(adapter):
    """Wrap hierarchical child caches without changing their native graph ownership."""
    original = Cache({})
    child = Cache({"fast": {"class_type": "VideoSilentFilmProjector"}})
    async def ensure(*args):
        """Return a distinct hierarchical cache."""
        return child
    original.ensure_subcache_for = ensure
    cache = asyncio.run(adapter.FilmOutputCache(original).ensure_subcache_for("parent", ["fast"]))
    assert cache.inner is child
    cache.set_local("fast", object())
    assert not child.cache
