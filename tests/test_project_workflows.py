"""Validate bounded project workflow writes, revision conflicts, and HTTP contracts."""

import asyncio
import importlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest


@pytest.fixture
def module():
    """Import the isolated store without starting the custom-node entry point."""
    if "ComfyUI_ALEXZ_tools" not in sys.modules:
        package = types.ModuleType("ComfyUI_ALEXZ_tools")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["ComfyUI_ALEXZ_tools"] = package
    return importlib.import_module("ComfyUI_ALEXZ_tools.utils.project_workflows")


def workflow(value=1):
    """Return a regular editor graph retaining extension-specific fields."""
    return {"id": "sample", "version": 0.4, "nodes": [{"id": 1, "type": "PrimitiveInt", "widgets_values": [value, "fixed"]}], "links": [], "extra": {"ds": {"scale": 1}}, "groups": []}


def test_create_overwrite_and_revision_conflicts(module, tmp_path):
    """Save readable Unicode JSON and refuse both accidental and stale overwrites."""
    store = module.ProjectWorkflows(tmp_path)
    assert store.list()["files"] == []
    assert not (tmp_path / "workflows").exists()
    original = workflow()
    saved = store.save({"filename": "Видео-в-Хронику", "workflow": original})
    assert saved["filename"] == "Видео-в-Хронику.json"
    path = tmp_path / "workflows" / saved["filename"]
    data = json.loads(path.read_text())
    assert data["nodes"] == original["nodes"] and data["extra"]["ds"] == original["extra"]["ds"]
    assert data["extra"]["alexz_project_workflow"] == saved["filename"]
    assert "alexz_project_workflow" not in original["extra"]
    assert store.list()["files"] == [{"filename": saved["filename"], "revision": saved["revision"]}]
    with pytest.raises(module.WorkflowConflict):
        store.save({"filename": saved["filename"], "workflow": workflow(2)})
    changed = store.save({"filename": saved["filename"], "workflow": workflow(2), "expected_revision": saved["revision"]})
    assert changed["revision"] != saved["revision"]
    with pytest.raises(module.WorkflowConflict):
        store.save({"filename": saved["filename"], "workflow": workflow(3), "expected_revision": saved["revision"]})
    assert json.loads(path.read_text())["nodes"][0]["widgets_values"][0] == 2


@pytest.mark.parametrize("name", ["../outside.json", "/tmp/out.json", "C:\\out.json", "a/b.json", "", ".json", "a\n.json", "x" * 241])
def test_filenames_cannot_escape_scope(module, tmp_path, name):
    """Reject arbitrary paths and malformed names before creating files."""
    store = module.ProjectWorkflows(tmp_path)
    with pytest.raises(ValueError):
        store.save({"filename": name, "workflow": workflow()})
    assert not (tmp_path / "outside.json").exists()


def test_symlinks_and_failed_atomic_write(module, tmp_path):
    """Keep external files and previous workflow bytes intact when writes are rejected."""
    store = module.ProjectWorkflows(tmp_path)
    saved = store.save({"filename": "saved", "workflow": workflow()})
    path = tmp_path / "workflows/saved.json"
    before = path.read_bytes()
    with patch.object(module.os, "replace", side_effect=OSError("test failure")):
        with pytest.raises(OSError):
            store.save({"filename": "saved", "workflow": workflow(9), "expected_revision": saved["revision"]})
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".alexz-workflow-*.tmp"))
    outside = tmp_path / "external.json"
    outside.write_text("original")
    (path.parent / "link.json").symlink_to(outside)
    assert [f["filename"] for f in store.list()["files"]] == ["saved.json"]
    with pytest.raises(ValueError):
        store.save({"filename": "link", "workflow": workflow()})
    assert outside.read_text() == "original"
    other = tmp_path / "project2"
    other.mkdir()
    (other / "workflows").symlink_to(path.parent)
    with pytest.raises(ValueError):
        module.ProjectWorkflows(other).save({"filename": "outside", "workflow": workflow()})


def test_http_save_contract_and_origin_checks(module, tmp_path):
    """Exercise actual aiohttp routes without connecting to a running ComfyUI server."""
    async def run():
        """Drive listing, writes and failures through a loopback HTTP test server."""
        routes = web.RouteTableDef()
        server = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
        module.register_routes(server, web, tmp_path)
        application = web.Application()
        application.add_routes(routes)
        async with TestClient(TestServer(application)) as client:
            response = await client.get(module.ROUTE)
            assert response.status == 200 and (await response.json())["files"] == []
            response = await client.post(module.ROUTE, json={"filename": "new", "workflow": workflow()})
            assert response.status == 403
            headers = {"X-Alexz-Workflow": "1"}
            response = await client.post(module.ROUTE, json={"filename": "new", "workflow": workflow()}, headers=headers)
            assert response.status == 200
            saved = await response.json()
            response = await client.post(module.ROUTE, json={"filename": "new", "workflow": workflow(2)}, headers=headers)
            assert response.status == 409
            response = await client.post(module.ROUTE, json={"filename": "new", "workflow": workflow(2), "expected_revision": saved["revision"]}, headers=headers)
            assert response.status == 200
            response = await client.post(module.ROUTE, json={"filename": "bad", "workflow": {}}, headers=headers)
            assert response.status == 400
            response = await client.post(module.ROUTE, json={"filename": "evil", "workflow": workflow()}, headers={**headers, "Origin": "https://example.org"})
            assert response.status == 403
            assert not (tmp_path / "workflows/evil.json").exists()
            with patch.object(module, "MAX_BYTES", 32):
                response = await client.post(module.ROUTE, json={"filename": "big", "workflow": workflow()}, headers=headers)
                assert response.status == 413
    asyncio.run(run())
