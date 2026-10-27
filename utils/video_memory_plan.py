"""Estimate cached film graph storage before decoding a source video."""

import logging
import math
import shutil

import torch

from . import video_memory as memory
from .video_cache import PERSISTENT_NODES

_LOGGER = logging.getLogger("SilentFilmMemory")


def estimate_graph(info, count, prompt=None, source_id=None, *, selective=True):
    """Count known IMAGE allocations and cadence frames without rendering pixels."""
    from ..nodes import video_silent_film_pipeline as pipeline
    from ..nodes import video_silent_film_defects as defects
    from ..nodes.video_silent_film_cadence import VideoSilentFilmCadence, _build_capture_intervals

    prompt = prompt or {}
    source_id = str(source_id)
    source_shape = (count, info["height"], info["width"], info["channels"])
    allocations = [{"node": source_id, "kind": "VideoLoadSource", "shape": source_shape,
                    "bytes": memory.video_bytes(source_shape), "gpu": False}]
    shapes = {(source_id, 0): source_shape, (source_id, 5): (1, *source_shape[1:])}
    visiting, done, unknown = set(), {source_id}, set()
    scratch, work = 0, info["height"] * info["width"] * 64
    preview_bytes = memory.video_bytes((1, *source_shape[1:]))
    persistent_previews = preview_bytes

    def scalar(value):
        """Resolve simple literals and known scalar outputs without running nodes."""
        if isinstance(value, list):
            parent, socket = str(value[0]), value[1]
            if parent == source_id:
                return {1: info["fps"], 2: info["total_frames"], 3: count}.get(socket)
            node = prompt.get(parent, {})
            if socket == 0 and node.get("class_type") in {"PrimitiveInt", "PrimitiveFloat"}:
                value = node.get("inputs", {}).get("value")
                return None if isinstance(value, list) else value
            return None
        return value

    def shape_at(link):
        """Visit image ancestors once and preserve preview-specific frame counts."""
        if not isinstance(link, list) or len(link) != 2:
            return None
        parent, socket = str(link[0]), link[1]
        visit(parent)
        return shapes.get((parent, socket))

    def visit(node_id):
        """Propagate shapes through supported stages and identify opaque boundaries."""
        nonlocal scratch, work, preview_bytes, persistent_previews
        if node_id in done or node_id in visiting:
            return
        visiting.add(node_id)
        node = prompt.get(node_id, {})
        kind, inputs = node.get("class_type", ""), node.get("inputs", {})
        image_shape = shape_at(inputs.get("image", inputs.get("images")))
        visiting.remove(node_id)
        done.add(node_id)
        if image_shape is None:
            return
        if kind in {"PreviewImage", "VideoPreviewFrame", "VideoSaveFinal"}:
            return
        cls = getattr(pipeline, kind, None) or getattr(defects, kind, None)
        if kind == "VideoSilentFilmCadence":
            cls = VideoSilentFilmCadence
        if cls is None or not hasattr(cls, "INPUT_TYPES"):
            unknown.add(node_id)
            return
        parameters = {}
        for name, spec in cls.INPUT_TYPES()["required"].items():
            if name == "image":
                continue
            default = spec[1].get("default", spec[0][0] if isinstance(spec[0], list) else None)
            parameters[name] = scalar(inputs.get(name, default))
        output_shape = image_shape
        if kind in {"VideoSilentFilmCamera", "VideoSilentFilmCadence"} and image_shape[0] > 1:
            names = ("source_fps", "target_fps_min", "target_fps_max", "fps_drift_strength", "seed", "crank_irregularity")
            if any(parameters.get(name) is None for name in names):
                unknown.add(node_id)
                return
            source_fps = float(parameters["source_fps"])
            if not math.isfinite(source_fps) or source_fps <= 0:
                unknown.add(node_id)
                return
            is_camera = kind == "VideoSilentFilmCamera"
            seed = pipeline._effect_seed(parameters["seed"], "camera", "capture") if is_camera else parameters["seed"]
            starts, _, _ = _build_capture_intervals(
                image_shape[0] / source_fps, parameters["target_fps_min"], parameters["target_fps_max"],
                parameters["fps_drift_strength"], seed,
                cadence_model="hand_crank" if is_camera else parameters.get("cadence_model", "legacy_random"),
                crank_irregularity=parameters["crank_irregularity"])
            output_count = len(starts) - 1 if is_camera or parameters.get("playback_mode") == "undercrank_projected_25fps" else image_shape[0]
            output_shape = (output_count, *image_shape[1:])
            if is_camera:
                scratch = max(scratch, memory.video_bytes(output_shape))
        # Выключенный дефект возвращает исходный storage, но создаёт маленькие превью.
        bypass = issubclass(cls, defects._DefectNode) and parameters.get("strength") == 0
        if not bypass:
            allocations.append({"node": node_id, "kind": kind, "shape": output_shape,
                                "bytes": memory.video_bytes(output_shape), "gpu": parameters.get("compute_device") != "cpu"})
            work = max(work, image_shape[1] * image_shape[2] * 512 * int(parameters.get("batch_size") or 16))
        for socket, name in enumerate(cls.RETURN_NAMES):
            if cls.RETURN_TYPES[socket] == "IMAGE":
                is_preview = "preview" in name
                shape = (1, *output_shape[1:]) if is_preview else output_shape
                shapes[(node_id, socket)] = shape
                if is_preview:
                    preview_bytes += memory.video_bytes(shape)
                    if kind in PERSISTENT_NODES:
                        persistent_previews += memory.video_bytes(shape)

    for node_id in prompt:
        visit(str(node_id))
    persistent = [item for item in allocations if item["kind"] in PERSISTENT_NODES]
    transient = [item for item in allocations if item["kind"] not in PERSISTENT_NODES]
    consumers = {}
    for node_id, node in prompt.items():
        if node.get("class_type") in {"PreviewImage", "VideoPreviewFrame"}:
            continue
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2:
                consumers.setdefault(str(value[0]), set()).add(node_id)
    transient_ids = {node_id for node_id, _ in shapes if node_id != source_id
                     and prompt.get(node_id, {}).get("class_type") not in PERSISTENT_NODES}
    branched = any(len(consumers.get(node_id, ())) > 1 for node_id in transient_ids)
    # Линейная цепочка держит вход/выход; у разветвлений оставляем верхнюю оценку.
    working = transient if branched else sorted(transient, key=lambda item: item["bytes"], reverse=True)[:2]
    storage_allocations = persistent + working if selective else allocations
    cached_bytes = sum(item["bytes"] for item in persistent) + persistent_previews if selective else sum(item["bytes"] for item in allocations) + preview_bytes
    return {"allocations": allocations, "storage_allocations": storage_allocations,
            "cached_bytes": cached_bytes, "cache_policy": "source_camera" if selective else "all_outputs",
            "all_outputs_bytes": sum(item["bytes"] for item in allocations) + preview_bytes,
            "peak_bytes": sum(item["bytes"] for item in storage_allocations) + preview_bytes + scratch,
            "branched_estimate": branched,
            "preview_bytes": preview_bytes, "scratch_bytes": scratch, "work_bytes": work,
            "unknown_nodes": sorted(unknown), "scope": "known_descendants" if prompt else "source_only"}


def check_resources(plan):
    """Log conservative placement and verify disk fallback before the expensive render."""
    ram_free = memory.available_ram()
    work = plan["work_bytes"]
    ram_budget = max(0, ram_free - max(memory.HOST_RESERVE, work + memory.RAM_RESERVE))
    vram_free = 0
    if torch.cuda.is_available():
        try:
            from comfy import model_management
            device = model_management.get_torch_device()
        except ImportError:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type == "cuda":
            vram_free = memory.available_vram(device)
    vram_budget = max(0, vram_free - work - memory.RAM_RESERVE)
    placements = {"ram": 0, "vram": 0, "disk": 0}
    for item in plan["storage_allocations"]:
        size = item["bytes"]
        if item["gpu"] and size >= memory.DISK_THRESHOLD and size <= vram_budget:
            vram_budget -= size
            placements["vram"] += size
        elif size <= ram_budget:
            ram_budget -= size
            placements["ram"] += size
        else:
            placements["disk"] += size
    # Временный каденс Camera и превью учитываем дополнительно, не обещаем точный пик.
    disk_estimate = placements["disk"] + plan["scratch_bytes"] + plan["preview_bytes"]
    memory.CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(memory.CACHE_DIRECTORY).free
    result = {**plan, "estimated_placement": placements, "disk_estimate_bytes": disk_estimate,
              "disk_fallback_bytes": plan["peak_bytes"],
              "free_ram_bytes": ram_free, "free_vram_bytes": vram_free, "free_disk_bytes": disk_free,
              "cache_directory": str(memory.CACHE_DIRECTORY)}
    gib = 1024**3
    _LOGGER.info("[Silent Film] План памяти (%s): постоянный кэш %.2f ГиБ; хранение вместе с быстрыми этапами/превью до %.2f ГиБ; рабочая порция до %.2f ГиБ. Все результаты при сплошном кэшировании заняли бы %.2f ГиБ.",
                 plan["cache_policy"], plan["cached_bytes"] / gib, plan["peak_bytes"] / gib, work / gib, plan["all_outputs_bytes"] / gib)
    _LOGGER.info("[Silent Film] Оценка размещения: VRAM %.2f, RAM %.2f, диск до %.2f ГиБ; свободно VRAM %.2f, RAM %.2f, диск %.2f ГиБ (%s).",
                 placements["vram"] / gib, placements["ram"] / gib, disk_estimate / gib,
                 vram_free / gib, ram_free / gib, disk_free / gib, memory.CACHE_DIRECTORY)
    if plan["unknown_nodes"]:
        _LOGGER.warning("[Silent Film] Неполная оценка: неизвестные преобразования/параметры в нодах %s; последующие размеры не предполагаются.", ", ".join(plan["unknown_nodes"]))
    if disk_estimate + memory.RAM_RESERVE > disk_free:
        _LOGGER.warning("[Silent Film] Плану может не хватить диска: оценка %.2f ГиБ + запас, доступно %.2f. Уменьшите диапазон или освободите кэш-диск. Кэширование ComfyUI и другие процессы могут изменить фактическую потребность.", disk_estimate / gib, disk_free / gib)
    return result
