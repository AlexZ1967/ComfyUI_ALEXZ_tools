"""Exercise the installed ComfyUI executor without a server or user state changes."""

import gc
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]


def main():
    """Compare cache policies on a real film graph, including preview consumers."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "p313-torch214-cu132":
        raise RuntimeError("Запускайте проверку в Conda p313-torch214-cu132.")
    workflow = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "workflows/Видео-в-Хронику-по-этапам.json"
    data = json.loads(workflow.read_text())
    # Отдельный процесс импортирует установленный core в CPU-режиме, не запускает сервер.
    sys.path.insert(0, str(ROOT.parents[1]))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import execution
    import torch
    package = types.ModuleType("ComfyUI_ALEXZ_tools")
    package.__path__ = [str(ROOT)]
    sys.modules[package.__name__] = package
    pipeline = importlib.import_module(f"{package.__name__}.nodes.video_silent_film_pipeline")
    defects = importlib.import_module(f"{package.__name__}.nodes.video_silent_film_defects")
    memory = importlib.import_module(f"{package.__name__}.utils.video_memory")
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_num_threads(2)
    results = []
    peaks = []
    original_allocate = memory.allocate_video
    original_use_disk = memory.use_disk

    def allocate(*args, **kwargs):
        """Measure live file mappings without extending tensor lifetimes."""
        value = original_allocate(*args, **kwargs)
        peaks.append(len(memory._MAPPINGS))
        return value

    class Source:
        calls = 0
        RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "STRING", "IMAGE")
        FUNCTION = "load"

        @classmethod
        def INPUT_TYPES(cls):
            """Expose a deterministic synthetic source."""
            return {"required": {"version": ("INT", {"default": 0})}}

        def load(self, version=0):
            """Allocate a small mapped clip with the loader's public outputs."""
            Source.calls += 1
            image = allocate((8, 48, 64, 3))
            image.copy_(torch.rand(image.shape, generator=torch.Generator().manual_seed(1925 + version)))
            return image, 24.0, 8, 8, "", image[4:5].clone()

    class Sink:
        RETURN_TYPES = ()
        OUTPUT_NODE = True
        FUNCTION = "save"

        @classmethod
        def INPUT_TYPES(cls):
            """Consume either a final batch or an independent preview branch."""
            return {"required": {"image": ("IMAGE",), "tag": ("STRING",)}}

        def save(self, image, tag):
            """Retain only a tiny copied result, never an upstream tensor alias."""
            results.append((tag, image.clone()))
            return ()

    nodes.NODE_CLASS_MAPPINGS.update({"MemorySource": Source, "MemorySink": Sink})
    prompt = {}
    by_id = {node["id"]: node for node in data["nodes"]}
    links = {link[0]: link for link in data["links"]}
    output_ids = []
    for node in data["nodes"]:
        node_id, kind = str(node["id"]), node["type"]
        if kind == "VideoLoadSource":
            prompt[node_id] = {"class_type": "MemorySource", "inputs": {}}
            continue
        if kind == "VideoSaveFinal":
            link = links[next(i["link"] for i in node["inputs"] if i["name"] == "image")]
            prompt[node_id] = {"class_type": "MemorySink", "inputs": {"image": [str(link[1]), link[2]], "tag": "final"}}
            output_ids.append(node_id)
            continue
        if not kind.startswith("VideoSilentFilm"):
            continue
        if node.get("mode", 0) != 0:
            raise ValueError("Для проверки нужен workflow без bypass основных этапов.")
        cls = getattr(pipeline, kind, None) or getattr(defects, kind)
        nodes.NODE_CLASS_MAPPINGS[kind] = cls
        values = iter(node["widgets_values"])
        inputs = {}
        for name, spec in cls.INPUT_TYPES()["required"].items():
            if spec[0] == "IMAGE":
                continue
            inputs[name] = next(values)
            if len(spec) > 1 and spec[1].get("control_after_generate"):
                next(values)
        assert next(values, None) is None, kind
        for slot in node.get("inputs", []):
            if slot.get("link") is None:
                continue
            link = links[slot["link"]]
            upstream = by_id[link[1]]
            inputs[slot["name"]] = (upstream["widgets_values"][0] if upstream["type"] == "PrimitiveInt"
                                    else [str(link[1]), link[2]])
        inputs.update(compute_device="cpu", batch_size=2)
        if kind == "VideoSilentFilmCamera":
            inputs.update(blur_mode="simple", blur_samples=3)
        prompt[node_id] = {"class_type": kind, "inputs": inputs}
        # Даже отключённые пользовательские превью проверяем как активные ответвления.
        for index, name in enumerate(cls.RETURN_NAMES):
            if "preview" in name:
                preview_id = f"preview-{node_id}-{name}"
                prompt[preview_id] = {"class_type": "MemorySink", "inputs": {"image": [node_id, index], "tag": preview_id}}
                output_ids.append(preview_id)
    server = types.SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                                   send_sync=lambda *args, **kwargs: None)
    reports, reference = {}, None
    with tempfile.TemporaryDirectory(prefix="alexz-memory-") as directory:
        memory.CACHE_DIRECTORY = Path(directory)
        memory.use_disk = lambda *args, **kwargs: True
        pipeline.allocate_video = allocate
        cadence = importlib.import_module(f"{package.__name__}.nodes.video_silent_film_cadence")
        cadence.allocate_video = allocate
        for mode in (execution.CacheType.CLASSIC, execution.CacheType.RAM_PRESSURE, execution.CacheType.NONE):
            peaks.clear()
            results.clear()
            executor = execution.PromptExecutor(server, cache_type=mode, cache_args={"ram": 0, "ram_inactive": 0})
            executor.execute(prompt, mode.name, execute_outputs=output_ids)
            assert executor.success, executor.status_messages
            gc.collect()
            reports[mode.name] = {"peak_mappings": max(peaks), "retained_mappings": len(memory._MAPPINGS), "outputs": len(results)}
            current = dict(results)
            if reference is None:
                reference = current
            else:
                assert current.keys() == reference.keys()
                for key in current:
                    torch.testing.assert_close(current[key], reference[key], atol=0, rtol=0)
            del executor
            gc.collect()
            assert not memory._MAPPINGS, "После удаления исполнителя остались mmap-ссылки."
        selective = importlib.import_module(f"{package.__name__}.utils.video_cache")
        assert selective.install()
        assert selective.install()
        camera_calls = []
        original_camera = pipeline.VideoSilentFilmCamera.apply
        def camera(self, *args, **kwargs):
            """Count expensive camera executions independently of cached preview consumers."""
            camera_calls.append(1)
            return original_camera(self, *args, **kwargs)
        pipeline.VideoSilentFilmCamera.apply = camera
        for storage, policy in (("disk", execution.CacheType.CLASSIC), ("automatic", execution.CacheType.CLASSIC),
                                ("disk", execution.CacheType.RAM_PRESSURE), ("disk", execution.CacheType.LRU)):
            memory.use_disk = (lambda *args, **kwargs: True) if storage == "disk" else original_use_disk
            peaks.clear()
            results.clear()
            camera_calls.clear()
            Source.calls = 0
            executor = execution.PromptExecutor(server, cache_type=policy,
                                               cache_args={"ram": 0, "ram_inactive": 0, "lru": 100})
            executor.execute(prompt, f"selective-{storage}-first", execute_outputs=output_ids)
            assert executor.success and Source.calls == 1 and len(camera_calls) == 1
            for key, value in results:
                torch.testing.assert_close(value, reference[key], atol=0, rtol=0)
            assert len(results) == len(reference)
            initial_outputs = dict(results)
            results.clear()
            executor.execute(prompt, f"selective-{storage}-repeat", execute_outputs=output_ids)
            assert executor.success and Source.calls == 1 and len(camera_calls) == 1 and results
            for key, value in results:
                torch.testing.assert_close(value, initial_outputs[key], atol=0, rtol=0)
            assert any(tag == "final" for tag, _ in results)
            changed_id = next(key for key, node in prompt.items() if node["class_type"] == "VideoSilentFilmNegativeDamage")
            previous_strength = prompt[changed_id]["inputs"]["strength"]
            prompt[changed_id]["inputs"]["strength"] = 0.27
            results.clear()
            executor.execute(prompt, f"selective-{storage}-edited", execute_outputs=output_ids)
            assert executor.success and Source.calls == 1 and len(camera_calls) == 1 and results
            assert any(tag == "final" for tag, _ in results)
            reports[f"SELECTIVE_{policy.name}_{storage}"] = {
                "source_calls": Source.calls, "camera_calls": len(camera_calls),
                "recomputed_outputs": len(results), "peak_mappings": max(peaks),
                "retained_mappings": len(memory._MAPPINGS)}
            if storage == "disk":
                assert len(memory._MAPPINGS) == 2, reports
                assert max(peaks) <= 5, reports
            camera_id = next(key for key, node in prompt.items() if node["class_type"] == "VideoSilentFilmCamera")
            old_seed = prompt[camera_id]["inputs"]["seed"]
            prompt[camera_id]["inputs"]["seed"] = old_seed + 1
            executor.execute(prompt, f"selective-{storage}-camera-change", execute_outputs=output_ids)
            assert executor.success and Source.calls == 1 and len(camera_calls) == 2
            source_id = next(key for key, node in prompt.items() if node["class_type"] == "MemorySource")
            prompt[source_id]["inputs"]["version"] = 1
            executor.execute(prompt, f"selective-{storage}-source-change", execute_outputs=output_ids)
            assert executor.success and Source.calls == 2 and len(camera_calls) == 3
            prompt[source_id]["inputs"].pop("version")
            prompt[camera_id]["inputs"]["seed"] = old_seed
            prompt[changed_id]["inputs"]["strength"] = previous_strength
            del executor
            gc.collect()
            assert not memory._MAPPINGS
    assert reports["NONE"]["retained_mappings"] == 0
    assert reports["NONE"]["peak_mappings"] <= 4, reports
    assert reports["NONE"]["peak_mappings"] < reports["CLASSIC"]["peak_mappings"], reports
    print(json.dumps({"workflow": workflow.name, "result": "PASS", "policies": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
