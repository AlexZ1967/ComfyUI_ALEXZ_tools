"""Insert independent film defects without changing the established four-stage pipeline."""

import copy
import json
import logging
import math

import torch

from . import video_silent_film_pipeline as pipeline
from ..utils import film_defects as masks
from ..utils import film_sensitometry as sensitometry
from ..utils.film_optics import projection_glow

_LOGGER = logging.getLogger("SilentFilmDefects")
_MODEL = "alexz.film_defects_v1"


def _read_metadata(text, images):
    """Preserve the last real material stage while validating optional metadata."""
    if text is None or isinstance(text, str) and not text.strip():
        return None
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("film_json: требуется паспорт Silent Film или пустой вход.") from exc
    if (not isinstance(data, dict) or data.get("schema_name") != "alexz.silent_film_pipeline"
            or data.get("schema_version") != 1 or data.get("stage") not in ("camera", "negative", "print", "projector")
            or not isinstance(data.get("history"), list)):
        raise ValueError("film_json: неподдерживаемый паспорт Silent Film.")
    if data.get("image_shape") != list(images.shape):
        raise ValueError("film_json: форма батча не совпадает с image.")
    if not isinstance(data.get("defect_history", []), list):
        raise ValueError("film_json: defect_history должен быть списком.")
    return data


def _frame_ids(meta, count, offset, screen_space=False):
    """Reuse physical film identities for repeated frames and local indices otherwise."""
    cadence = meta.get("cadence") if meta else None
    ids = cadence.get("output_group_ids") if isinstance(cadence, dict) and not screen_space else None
    if ids is None:
        ids = range(count)
    elif (not isinstance(ids, list) or len(ids) != count
          or any(type(value) is not int or value < 0 for value in ids)):
        raise ValueError("film_json: output_group_ids должны содержать номер каждого физического кадра.")
    return [value + offset for value in ids]


def _validate_parameters(cls, supplied):
    """Resolve schema defaults and reject unsupported, nonfinite or out-of-range inputs."""
    schema = cls.INPUT_TYPES()["required"]
    unknown = set(supplied) - (set(schema) - {"image"})
    if unknown:
        raise ValueError(f"Неизвестные параметры: {', '.join(sorted(unknown))}.")
    result = {}
    for name, spec in schema.items():
        if name == "image":
            continue
        kind, options = spec
        value = supplied.get(name, options.get("default", kind[0] if isinstance(kind, list) else None))
        if isinstance(kind, list):
            if value not in kind:
                raise ValueError(f"{name}: неизвестное значение {value!r}.")
        elif kind == "INT":
            if type(value) is not int or not options["min"] <= value <= options["max"]:
                raise ValueError(f"{name}: требуется целое число от {options['min']} до {options['max']}.")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}: требуется конечное число.")
            if not options["min"] <= value <= options["max"]:
                raise ValueError(f"{name}: значение вне допустимого диапазона.")
        result[name] = value
    return result


class _DefectNode:
    """Share bounded processing and preview contracts for independent image modifiers."""

    KIND = "gate"
    MATERIAL = "camera"
    LABEL = "Silent Film Capture Dirt"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "film_json", "preview", "mask_preview")
    FUNCTION = "apply"
    CATEGORY = "video/stylize/silent_film/defects"

    @classmethod
    def INPUT_TYPES(cls):
        """Expose only controls relevant to this material and defect family."""
        f = pipeline._float
        required = {
            "image": ("IMAGE", {"tooltip": "Кадры на соответствующем этапе. Число, порядок, разрешение и альфа сохраняются. Нода работает самостоятельно и допускает bypass соседних этапов."}),
            "strength": f(0, 0, 1, "Общая сила дефектов. 0 полностью пропускает обработку и выделение выходного батча. Начните с 0.1–0.25 для пыли/царапин, 0.03–0.1 для проявки."),
        }
        if cls.KIND == "development":
            required.update({
                "pattern": (["mottling", "streaks", "mixed"], {"default": "mixed", "tooltip": "mottling — пятнистая плотность; streaks — продольные полосы; mixed — сочетание. Рисунок связан между кадрами, без независимого шума на каждом кадре."}),
                "size": f(8, 0.5, 50, "Масштаб пятен или расстояние между полосами в процентах ширины кадра. 8 — около 154 px при ширине 1920. Начните с 5–15.", step=0.5),
                "fog_strength": f(0.15, 0, 1, "Доля дополнительной положительной плотности в неоднородности. На негативе осветляет последующую печать, на позитиве затемняет. Умножается на общую strength; при strength=0 тоже выключена."),
            })
        else:
            patterns = ["dust", "hair", "mixed"] if cls.KIND == "gate" else ["scratches", "dust", "stains", "mixed"]
            required["pattern"] = (patterns, {"default": "mixed", "tooltip": "dust — пыль; hair — волоски; scratches — царапины; stains — пятна; mixed — сочетание доступных для этой ноды дефектов."})
            if cls.KIND == "damage":
                required["damage_type"] = (["surface", "emulsion_loss", "dirty"], {"default": "surface", "tooltip": "surface — поверхностное повреждение с потерей света; emulsion_loss — снятая эмульсия царапин/пятен (пыль остаётся непрозрачной); dirty — глубокое загрязнение. Негатив и позитив дают разную полярность после печати. Позитив обрабатывается приближённо, по экранному изображению."})
            required.update({
                "amount": f(3, 0, 24, "Среднее число одновременно активных следов, допускает дробь. 0 выключает обработку. Не растёт с разрешением. В mixed следы распределены между типами.", step=0.5),
                "size": f(0.12 if cls.KIND == "gate" else 0.06, 0.01, 5, "Размер в процентах ширины: диаметр пыли/пятна или ширина царапины/волоска. 0.1 = 1.92 px при 1920. В mixed повреждений пыль вдвое крупнее, пятна в 20 раз крупнее базовой ширины. Для отдельных пятен попробуйте 1–3%.", step=0.01),
            })
        required.update({
            "persistence_frames": ("INT", {"default": 48 if cls.KIND != "damage" else 24, "min": 1, "max": 10000, "tooltip": "Длительность события в кадрах плёнки, не исходного ролика: пыль окна держится, затем заменяется; царапина продолжается вдоль плёнки. Для проявки — расстояние между плавно сменяющимися полями. Пыль на самой плёнке меняется с каждым физическим кадром. При показе 25 fps значение 50 соответствует примерно 2 секундам."}),
            "irregularity": f(0.6, 0, 1, "Неровность формы и движения. Царапины гуляют без периодической синусоиды; пыль окна слегка шевелится. Для проявки управляет долей медленно меняющегося поля; 0 даёт неподвижное поле."),
        })
        if cls.KIND != "development":
            required["edge_bias"] = f(0.35, 0, 1, "Доля следов, сосредоточенных возле краёв: 0 — равномерно по кадру, 1 — все у краёв.")
        if cls.MATERIAL == "projector":
            required["softness"] = f(0.05, 0, 1, "Мягкость тени загрязнения окна в процентах ширины. 0 — резкая; 0.05–0.15 — мягче. Эффект добавляется после готовой проекции и не проходит повторно через внутреннюю оптику Projector.", step=0.01)
        required["frame_offset"] = ("INT", {"default": 0, "min": 0, "max": 1000000000, "tooltip": "Смещение номера физического кадра для воспроизводимого фрагмента. Обычно 0. Совпадение с полным роликом требует тех же кадров и того же каденса; выбор нового исходного отрезка перед Camera сам по себе этого не гарантирует."})
        runtime = pipeline._runtime_inputs()
        runtime["seed"][1]["tooltip"] = "Общий seed варианта. Каждая новая нода и каждый тип дефекта получают свой поток. Для второго экземпляра того же эффекта задайте другой seed. Рисунок не зависит от размера порции и CPU/CUDA."
        required.update(runtime)
        return {"required": required, "optional": {
            "film_json": ("STRING", {"forceInput": True, "tooltip": "Необязательный паспорт: передаётся дальше с записью defect_history, без смены основного этапа. Помогает сохранить кодирование и номера физических кадров. Без паспорта нода использует свой тип материала и локальную нумерацию, выход film_json пустой."}),
        }}

    def apply(self, image, film_json=None, **parameters):
        """Process one frame at a time inside the existing bounded CPU/CUDA chunk runner."""
        images = pipeline.finish._validate_video_batch(image, normalize=False)
        if images.size(1) < 1 or images.size(2) < 1:
            raise ValueError("image: ширина и высота должны быть положительными.")
        meta = _read_metadata(film_json, images)
        settings = _validate_parameters(type(self), parameters)
        frames = _frame_ids(meta, len(images), settings["frame_offset"], self.MATERIAL == "projector")
        seed = pipeline._effect_seed(settings["seed"], "defects", type(self).__name__)
        negative = self.MATERIAL in ("negative", "printing")
        transfer = "linear" if negative else pipeline._projection_transfer(meta or {"history": []})
        middle = len(images) // 2
        mask_preview = torch.zeros((1, images.size(1), images.size(2), 1), dtype=torch.float32)
        enabled = settings["strength"] > 0 and settings.get("amount", 1) > 0

        def operation(chunk, start, end):
            """Build only current-frame masks and preserve the original alpha channel."""
            nonlocal mask_preview
            result = torch.empty_like(chunk)
            for local, index in enumerate(range(start, end)):
                rgb = chunk[local:local + 1, ..., :3]
                height, width = rgb.shape[1:3]
                linear = sensitometry.decode_transfer(rgb, transfer)
                if self.KIND == "development":
                    field = masks.development_field(height, width, frames[index], seed, settings["pattern"],
                        settings["size"], settings["persistence_frames"], settings["irregularity"], chunk.device)
                    density = settings["strength"] * (0.35 * field + 0.3 * settings["fog_strength"] * (field + 1) / 2)
                    linear = linear * torch.pow(10.0, -density[None, ..., None])
                    coverage = density.abs().clamp(0, 1)
                else:
                    layers = masks.surface_masks(height, width, frames[index], seed, settings["pattern"],
                        settings["amount"], settings["size"], settings["persistence_frames"],
                        settings["irregularity"], settings["edge_bias"], self.KIND == "gate", chunk.device)
                    combined = torch.stack(tuple(layers.values())).amax(0)
                    coverage = combined * settings["strength"]
                    if self.KIND == "gate":
                        if settings.get("softness", 0):
                            coverage = projection_glow(coverage[None, ..., None], 1, settings["softness"])[0, ..., 0].clamp(0, 1)
                        linear = linear * (1 - coverage[None, ..., None])
                    elif settings["damage_type"] == "emulsion_loss":
                        loss = torch.maximum(layers["scratches"], layers["stains"])[None, ..., None] * settings["strength"]
                        clear = 10 ** -sensitometry.NEGATIVE_BASE if negative else 1.0
                        linear = linear + (linear.clamp_min(clear) - linear) * loss
                        linear = linear * (1 - layers["dust"][None, ..., None] * settings["strength"])
                    else:
                        density_scale = 1.5 if settings["damage_type"] == "dirty" else 0.65
                        linear = linear * torch.pow(10.0, -density_scale * coverage[None, ..., None])
                rgb = sensitometry.encode_transfer(linear.clamp(0, 1), transfer).clamp(0, 1)
                result[local:local + 1] = pipeline._rgb_alpha(rgb, chunk[local:local + 1])
                if index == middle:
                    mask_preview = coverage.detach().to(device="cpu")[None, ..., None].clone()
            return result

        if enabled:
            _LOGGER.info("[%s] План: рисунок дефектов → материал %s → средний кадр; сила %.3f, масштаб %.3f%% ширины.",
                         self.LABEL, self.MATERIAL, settings["strength"], settings["size"])
            output = pipeline._run_stage(images, self.LABEL, settings["compute_device"], settings["batch_size"], operation)
        else:
            _LOGGER.info("[%s] Выключено: передача входного батча без обработки.", self.LABEL)
            output = images
        if meta is not None:
            meta = copy.deepcopy(meta)
            meta.setdefault("defect_history", []).append({"node": type(self).__name__, "model": _MODEL,
                "material": self.MATERIAL, "enabled": enabled, "parameters": settings})
            output_meta = json.dumps(meta, ensure_ascii=False)
        else:
            output_meta = ""
        preview = pipeline._preview_frame(output)
        values = (output, output_meta, preview, mask_preview.expand(-1, -1, -1, 3))
        if negative:
            display_transfer = pipeline._print_transfer("from_negative", meta or {"history": []})
            positive = sensitometry.reference_positive(preview[..., :3].clamp(0, 1), display_transfer)
            values += (pipeline._rgb_alpha(positive, preview),)
        return values


class VideoSilentFilmCaptureDirt(_DefectNode):
    """Shadow capture exposure with persistent camera-gate dust and hairs."""


class _NegativeDefect(_DefectNode):
    """Add a neutral positive preview for modifiers of negative transmittance."""
    MATERIAL = "negative"
    RETURN_TYPES = _DefectNode.RETURN_TYPES + ("IMAGE",)
    RETURN_NAMES = _DefectNode.RETURN_NAMES + ("preview_positive",)


class VideoSilentFilmNegativeDevelopment(_NegativeDefect):
    """Modify developed negative density with coherent mottling and streaks."""
    KIND = "development"
    LABEL = "Silent Film Negative Development"


class VideoSilentFilmNegativeDamage(_NegativeDefect):
    """Damage negative transmittance before it is exposed onto positive stock."""
    KIND = "damage"
    LABEL = "Silent Film Negative Damage"


class VideoSilentFilmPrintingDirt(_NegativeDefect):
    """Attenuate printing exposure with independent printer-gate contamination."""
    MATERIAL = "printing"
    LABEL = "Silent Film Printing Dirt"


class VideoSilentFilmPositiveDevelopment(_DefectNode):
    """Approximate positive development defects in decoded viewing light."""
    KIND = "development"
    MATERIAL = "positive"
    LABEL = "Silent Film Positive Development"


class VideoSilentFilmPositiveDamage(_DefectNode):
    """Approximate print wear while preserving the input viewing color and alpha."""
    KIND = "damage"
    MATERIAL = "positive"
    LABEL = "Silent Film Positive Damage"


class VideoSilentFilmProjectorDirt(_DefectNode):
    """Apply stationary screen-space dirt shadows to already projected linear light."""
    MATERIAL = "projector"
    LABEL = "Silent Film Projector Dirt"
