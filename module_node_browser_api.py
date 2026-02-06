from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from itertools import islice
from typing import Any

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - non-Comfy environment
    PromptServer = None
    web = None


_LOGGER = logging.getLogger("ALEXZ_tools.ModuleBrowser")
_GROUP_ORDER = (
    ("core", "Core_Nodes"),
    ("core_extras", "Core_Extras_Nodes"),
    ("api", "API_Nodes"),
    ("custom", "Custom_Nodes"),
)

_ALEXZ_ANNOTATIONS = {
    "ImagePrepare_for_QwenEdit_outpaint": "Подготавливает изображение и latent под QwenEdit Outpaint.",
    "ImageAlignOverlayToBackground": "Выравнивает оверлей относительно фона по ключевым точкам.",
    "JsonDisplayAndSave": "Показывает и сохраняет JSON в читаемом виде.",
    "VideoInpaintWatermark": "Удаляет статический вотермарк/объект из видео.",
    "ImageColorMatchToReference": "Подгоняет цвет и тон изображения под референс.",
    "VideoFrameMatch": "Ищет наиболее похожий кадр в видео для входной картинки.",
    "VideoCutMatch": "Подбирает оптимальную пару кадров для склейки двух видео.",
    "ImageDifference": "Строит абсолютную разницу двух изображений.",
    "ImageWaveformScope": "Строит waveform/parade scope для анализа яркости и каналов.",
    "ImageHistogramScope": "Строит RGB/Luma гистограмму изображения.",
}


def _module_root(node_cls: Any) -> str:
    module_name = getattr(node_cls, "__module__", "") or ""
    if not module_name:
        return "unknown"
    return module_name.split(".", 1)[0]


def _classify_by_relative_module(node_cls: Any) -> tuple[str, str]:
    rel = getattr(node_cls, "RELATIVE_PYTHON_MODULE", None)
    if not isinstance(rel, str) or not rel:
        return ("core", _module_root(node_cls))
    parts = [p for p in rel.split(".") if p]
    if len(parts) >= 2:
        root, module_name = parts[0], parts[1]
    elif len(parts) == 1:
        root, module_name = parts[0], parts[0]
    else:
        return ("core", _module_root(node_cls))

    if root == "custom_nodes":
        return ("custom", module_name)
    if root == "comfy_extras":
        return ("core_extras", module_name)
    if root == "comfy_api_nodes":
        return ("api", module_name)
    module_name = getattr(node_cls, "__module__", "") or ""
    module_l = module_name.lower()
    if module_l.startswith("comfy_extras."):
        parts = module_name.split(".")
        return ("core_extras", parts[1] if len(parts) > 1 else module_name)
    if module_l.startswith("comfy_api_nodes."):
        parts = module_name.split(".")
        return ("api", parts[1] if len(parts) > 1 else module_name)
    return ("core", _module_root(node_cls))


def _fallback_annotation(node_cls: Any) -> str:
    category = getattr(node_cls, "CATEGORY", "") or "unknown"
    return_names = getattr(node_cls, "RETURN_NAMES", None)
    if not return_names:
        return_types = getattr(node_cls, "RETURN_TYPES", ())
        return_names = return_types

    if return_names is None:
        output_items = []
    elif isinstance(return_names, (str, bytes)):
        output_items = [str(return_names)]
    else:
        try:
            output_items = [str(x) for x in islice(iter(return_names), 3)]
        except Exception:
            output_items = [str(return_names)]

    outputs = ", ".join(output_items) or "unknown"
    return f"Категория: {category}. Выходы: {outputs}."


def _collect_nodes() -> list[dict[str, Any]]:
    comfy_nodes = importlib.import_module("nodes")
    class_map = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    display_map = getattr(comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

    items: list[dict[str, Any]] = []
    for node_name, node_cls in class_map.items():
        display_name = display_map.get(node_name, node_name)
        annotation = _ALEXZ_ANNOTATIONS.get(node_name) or _fallback_annotation(node_cls)
        group, module_bucket = _classify_by_relative_module(node_cls)
        items.append(
            {
                "node_name": node_name,
                "display_name": display_name,
                "module": module_bucket,
                "group": group,
                "category": getattr(node_cls, "CATEGORY", "") or "",
                "annotation": annotation,
            }
        )
    return items


def _build_catalog() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _collect_nodes():
        module_name = item["module"]
        grouped[module_name].append(item)

    for module_name in grouped:
        grouped[module_name].sort(key=lambda item: item["display_name"].lower())
    return dict(sorted(grouped.items(), key=lambda kv: kv[0].lower()))


def _build_group_catalog() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _collect_nodes():
        grouped[item["group"]].append(item)

    for group_name in grouped:
        grouped[group_name].sort(key=lambda item: item["display_name"].lower())
    return grouped


def _filter_modules(query: str, module_names: list[str]) -> list[str]:
    if not query:
        return module_names
    q = query.lower()
    exact = [name for name in module_names if name.lower() == q]
    if exact:
        return exact
    return [name for name in module_names if q in name.lower()]


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None):
    @PromptServer.instance.routes.get("/alexz_tools/node_catalog")
    async def alexz_tools_node_catalog(request):
        try:
            grouped = _build_group_catalog()
            groups = []
            for group_id, group_title in _GROUP_ORDER:
                nodes = grouped.get(group_id, [])
                groups.append(
                    {
                        "id": group_id,
                        "title": group_title,
                        "count": len(nodes),
                        "nodes": nodes,
                    }
                )
            return web.json_response({"groups": groups})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Node catalog API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_list")
    async def alexz_tools_module_list(request):
        query = (request.query.get("q", "") or "").strip().lower()
        try:
            catalog = _build_catalog()
            modules = []
            for module_name, nodes in catalog.items():
                if query and query not in module_name.lower():
                    continue
                modules.append({"module": module_name, "count": len(nodes)})
            return web.json_response({"query": query, "modules": modules})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module list API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_nodes")
    async def alexz_tools_module_nodes(request):
        query = (request.query.get("module", "") or request.query.get("q", "")).strip()
        try:
            catalog = _build_catalog()
            modules = list(catalog.keys())
            selected_modules = _filter_modules(query, modules)

            results = []
            for module_name in selected_modules:
                nodes = catalog.get(module_name, [])
                results.append(
                    {
                        "module": module_name,
                        "count": len(nodes),
                        "nodes": nodes,
                    }
                )

            return web.json_response(
                {
                    "query": query,
                    "module_count": len(results),
                    "results": results,
                    "hint": "Введите имя python-модуля (например: ComfyUI_ALEXZ_tools).",
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module browser API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
