"""Model silent-film capture, negative development, positive printing, and projection."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from time import perf_counter

import torch
from tqdm import tqdm

from .video_silent_film_cadence import VideoSilentFilmCadence
from . import video_silent_film_finish as finish
from ..utils.video_memory import allocate_video, release_video_pages

_LOGGER = logging.getLogger("SilentFilmPipeline")
_SCHEMA = "alexz.silent_film_pipeline"
_SEED_SCHEME = "blake2b_stage_effect_v1"


def _effect_seed(seed, stage, effect):
    """Derive a stable random stream from the user seed, stage, and effect name."""
    # В отличие от hash(), результат не зависит от процесса Python.
    payload = f"{int(seed)}:{stage}:{effect}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"ALEXZFilmSeedV1").digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _float(default, minimum, maximum, tooltip, step=0.01):
    """Define a bounded floating-point widget."""
    return ("FLOAT", {"default": default, "min": minimum, "max": maximum, "step": step, "tooltip": tooltip})


def _number(name, value, minimum, maximum):
    """Validate finite numeric parameters before allocating video buffers."""
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _runtime_inputs():
    """Share seed and bounded device controls between independent stages."""
    specs = finish.VideoSilentFilmFinish.INPUT_TYPES()["required"]
    required = {key: copy.deepcopy(specs[key]) for key in ("seed", "compute_device", "batch_size")}
    # Явное объявление нужно миграции widgets_values при наличии forceInput.
    required["seed"][1]["control_after_generate"] = True
    required["seed"][1]["tooltip"] = "Общий seed варианта. Этапы и эффекты получают отдельные воспроизводимые последовательности: одинаковое значение во всех четырёх нодах не синхронизирует камеру, зерно и проектор."
    required["compute_device"][1]["tooltip"] = "auto выбирает CPU/CUDA по размеру кадра и доступной памяти. CUDA ускоряет эффекты; Farneback в Camera остаётся на CPU. При OOM порция уменьшается, затем расчёт переходит на CPU. Начните с auto."
    required["batch_size"][1]["tooltip"] = "Максимум кадров в рабочей порции, 1–16. Начните с 4; при дефиците VRAM используйте 1–2. Не меняет художественный результат и зерно. Крупный результат хранится в VRAM при достаточном рабочем запасе, иначе в дисковом IMAGE-кэше; маленький — в обычной памяти. В консоли видны расчёт памяти и выбранное хранение."
    return required


def _optics_inputs():
    """Share optical controls without exposing projection or emulsion effects."""
    specs = finish.VideoSilentFilmFinish.INPUT_TYPES()["required"]
    required = {key: copy.deepcopy(specs[key]) for key in ("softness", "vignette_strength", "edge_softness", "edge_start", "edge_end")}
    tips = {
        "softness": "Равномерная мягкость всего кадра, радиус масштабируется по ширине. 0 выключает; Camera: старт 0.1–0.15; Projector: 0–0.03, чтобы не удваивать смягчение. Съёмочная мягкость записывается на негатив, проекционная размывает уже готовую копию вместе с её зерном.",
        "vignette_strength": "Затемнение RGB там, где общая маска оптики достигает полной силы: 0 выключает, 0.12 даёт до 12% до следующих этапов. Начните с 0.10–0.15; 0.6 — выраженная виньетка. Маску задают edge_start/end. Camera записывает её на плёнку; Projector добавляет неподвижную виньетку показа.",
        "edge_softness": "Максимальный расфокус краёв: 0 выключает, 0.15 — лёгкий, 0.5–0.7 — заметный. Маска задаёт местный радиус размытия, а не прозрачность размытого слоя. При 1 радиус sigma около 42 px для ширины 1664. Чтобы охватить больше кадра, уменьшите edge_start. У Camera и Projector это разные объективы.",
        "edge_start": "Радиус начала виньетки и расфокуса. 1 равен половине ширины кадра, 0.8 означает 40% ширины от центра. До начала центр сохраняется. Меньше — больше область эффекта; ближе к edge_end — уже переход. Начните с 0.6; конец должен быть строго больше начала.",
        "edge_end": "Радиус полной силы виньетки и расфокуса; единица — половина ширины кадра. Разность edge_end − edge_start задаёт толщину перехода: больше — шире и плавнее. Старт 1; пример 0.8 → 1.2. Если конец выходит за углы кадра, максимум внутри кадра не достигается.",
    }
    for key, tooltip in tips.items():
        required[key][1]["tooltip"] = tooltip
    return required


def _grain_inputs(default_strength):
    """Define independent film-stock grain in width-relative units."""
    return {
        "grain_strength": _float(default_strength, 0.0, 0.2, "Неоднородность оптической плотности именно этой эмульсии. 0 выключает зерно. Начните с 0.02–0.04 для Negative и 0–0.015 для Print: зерно негатива уже отпечатано на позитиве. Увеличение усиливает фактуру; это не тот же масштаб силы, что у старого Finish.", step=0.005),
        "grain_size": _float(2.0, 0.1, 16.0, "Размер структуры зерна относительно ширины 1920 px, допускает дробные значения. Старт 2; 1 — тоньше, 3 — грубее. Размер 3 даёт около 6 px при 3840, 1.5 px при 960 и 1 px при 640. Субпиксельное зерно ослабевает. Это масштаб многослойной фактуры, а не диаметр отдельного зерна."),
    }


def _read_film_meta(text, images, expected_stage):
    """Start local history if metadata is absent; otherwise validate stage and image shape."""
    if text is None or (isinstance(text, str) and not text.strip()):
        _LOGGER.info("[Silent Film] film_json не передан: автономная обработка, история начинается с текущего этапа.")
        return {"history": [], "cadence": None}
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("film_json must contain pipeline metadata.") from exc
    if not isinstance(data, dict) or data.get("schema_name") != _SCHEMA or data.get("schema_version") != 1:
        raise ValueError("Unsupported film_json schema.")
    if data.get("stage") != expected_stage:
        raise ValueError(f"Expected film stage {expected_stage}, got {data.get('stage')}.")
    if data.get("image_shape") != list(images.shape):
        raise ValueError("film_json image_shape does not match image; preserve frame count, dimensions and channels between stages.")
    if not isinstance(data.get("history"), list):
        raise ValueError("film_json history must be a list.")
    return data


def _preview_frame(images):
    """Return an independent one-frame batch without retaining the full video storage."""
    index = len(images) // 2
    return images[index:index + 1].clone()


def _write_film_meta(previous, images, stage, parameters):
    """Append a stage record without duplicating tensors or cadence arrays."""
    data = copy.deepcopy(previous)
    data.update(schema_name=_SCHEMA, schema_version=1, seed_scheme=_SEED_SCHEME, stage=stage, image_shape=list(images.shape))
    data.setdefault("history", []).append({"stage": stage, "parameters": parameters})
    return json.dumps(data, ensure_ascii=False)


def _stage_chunk(images, start, end, device, operation):
    """Keep temporary chunk tensors out of the OOM retry scope."""
    chunk = images[start:end].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    return operation(chunk, start, end)


def _run_stage(images, name, compute_device, batch_size, operation):
    """Apply a stage in bounded chunks with CUDA retry and CPU fallback."""
    work_device, size, cpu_size = finish._finish_processing_plan(images, compute_device, batch_size, label=name)
    started = perf_counter()
    _LOGGER.info("[%s] Расчёт: %d кадров, %dx%d; %s, порция %d.", name, len(images), images.size(2), images.size(1), work_device, size)
    output = allocate_video(images.shape, device=images.device, label=name, work_device=work_device,
                            work_bytes=images.size(1) * images.size(2) * 512 * size)
    start = 0
    with tqdm(total=len(images), desc=f"[{name}] Обработка", unit="кадр", dynamic_ncols=True, mininterval=0.5) as progress:
        while start < len(images):
            end = min(len(images), start + size)
            try:
                result = _stage_chunk(images, start, end, work_device, operation)
                output[start:end].copy_(result)
                release_video_pages(output, start, end, written=True)
                release_video_pages(images, start, end)
                del result
            except torch.OutOfMemoryError:
                if work_device.type != "cuda":
                    raise
                if size > 1:
                    size = max(1, size // 2)
                else:
                    if cpu_size < 1:
                        raise MemoryError(f"{name}: после CUDA OOM недостаточно RAM для безопасного CPU fallback.")
                    work_device, size = torch.device("cpu"), cpu_size
                with tqdm.external_write_mode():
                    _LOGGER.warning("[%s] Не хватает VRAM: повтор с кадра %d, устройство %s, порция %d.", name, start + 1, work_device, size)
                continue
            progress.update(end - start)
            start = end
    _LOGGER.info("[%s] Готово за %.1f с.", name, perf_counter() - started)
    return output


def _rgb_alpha(rgb, original):
    """Preserve alpha independently of optical density and tone."""
    return torch.cat((rgb, original[..., 3:4]), dim=-1) if original.size(-1) == 4 else rgb


def _luma(rgb):
    """Compute the monochrome signal from normalized RGB input."""
    return rgb[..., :1] * 0.299 + rgb[..., 1:2] * 0.587 + rgb[..., 2:3] * 0.114


def _density_grain(density, strength, size, seed, offset):
    """Add reproducible per-frame grain in optical-density space for one film stock."""
    if strength == 0:
        return density
    effective_size = size * density.size(2) / 1920.0
    amplitude = strength * min(1.0, effective_size)
    output = torch.empty_like(density)
    for index in range(len(density)):
        frame_seed = (int(seed) + (offset + index) * 1000003) % (2**63 - 1)
        field = finish._build_emulsion_grain_field(1, density.size(1), density.size(2), effective_size, frame_seed, density.device, density.dtype)
        output[index:index + 1] = density[index:index + 1] + field.permute(0, 2, 3, 1) * amplitude
    return output.clamp(0.0, 4.0)


def _global_softness(rgb, strength):
    """Apply width-relative optical softness without zero-padding edge darkening."""
    return finish._lens_gaussian_blur(rgb, max(0.1, rgb.size(2) / 1920.0 * strength * 4.0)) if strength else rgb


def _optics(rgb, settings, masks):
    """Reuse one spatial mask per processing device and apply camera or projector optics."""
    if settings["vignette_strength"] or settings["edge_softness"]:
        key = (rgb.device, rgb.dtype)
        if key not in masks:
            masks[key] = finish._lens_edge_mask(rgb.size(1), rgb.size(2), settings["edge_start"], rgb.device, rgb.dtype, settings["edge_end"])
        rgb = finish._apply_lens_edges(rgb, settings["vignette_strength"], settings["edge_softness"], settings["edge_start"], masks[key])
    return _global_softness(rgb, settings["softness"])


def _validate_optics(settings):
    """Validate radii and effect strengths before processing any frame."""
    for key in ("softness", "vignette_strength", "edge_softness"):
        settings[key] = _number(key, settings[key], 0, 1)
    settings["edge_start"] = _number("edge_start", settings["edge_start"], 0, 4)
    settings["edge_end"] = _number("edge_end", settings["edge_end"], 0.01, 4)
    if settings["edge_end"] <= settings["edge_start"]:
        raise ValueError("edge_end must be greater than edge_start.")


class VideoSilentFilmCamera:
    """Capture a moving source with hand-cranked exposure timing and camera optics."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose capture controls separately from film stock and projection."""
        cadence = VideoSilentFilmCadence.INPUT_TYPES()
        all_inputs = {**cadence["required"], **cadence["optional"]}
        keys = ("image", "source_fps", "target_fps_min", "target_fps_max", "fps_drift_strength", "crank_irregularity", "shutter_fraction", "motion_blur_strength", "blur_mode", "blur_samples", "camera_shake_strength")
        required = {key: copy.deepcopy(all_inputs[key]) for key in keys}
        for key, value in dict(source_fps=36.0, target_fps_min=16.0, target_fps_max=20.0,
                               fps_drift_strength=0.7, crank_irregularity=0.15, shutter_fraction=0.65,
                               motion_blur_strength=0.8, blur_samples=9, camera_shake_strength=0.25).items():
            required[key][1]["default"] = value
        capture_tooltips = {
            'image': 'Исходная движущаяся сцена, RGB/RGBA-батч кадров. Отдельный первый проектор и экран не моделируются. Выход Camera — ещё не плёнка: передайте image и film_json в Negative.',
            'source_fps': 'Частота интерпретации исходных кадров, а не частота выходного файла. Выше — короче виртуальная сцена и меньше снятых кадров. Для обычного видео укажите его fps; для материала, замедленного вдвое, удвойте fps интерпретации. 36 — сохранённая настройка данного workflow, не универсальный стандарт.',
            'target_fps_min': 'Нижняя граница скорости ручной съёмки. Начните с 16 при верхней 20. Вместе с верхней границей и fps_drift_strength задаёт ускорения движения; равные границы дают ровную частоту. Скорость показа задаётся отдельно в Projector.',
            'target_fps_max': 'Верхняя граница частоты ручной съёмки, не ниже target_fps_min. Старт 20 при нижней 16. Выше средняя частота — больше кадров плёнки и длиннее результат при том же projection_fps. Шире диапазон — заметнее неравномерность движения.',
            'fps_drift_strength': 'Амплитуда циклического изменения скорости ручки: 0 — постоянная середина диапазона, 1 — полный размах min/max. Старт 0.6–0.8; 1 для выраженного ритма. Не является силой покачивания камеры.',
            'crank_irregularity': 'Плавная нерегулярность между оборотами поверх цикла ручки. 0 — регулярная скорость, 0.10–0.25 — лёгкая живость, 0.5 — выраженная. Не превращает движение в независимый случайный fps; покачивание камеры остаётся нерегулярным даже при 0.',
            'shutter_fraction': 'Доля периода снимаемого кадра, открытая для экспозиции. Больше — длиннее экспозиционный след движения, меньше — резче отдельные положения. Старт 0.5–0.65. Скорость ролика не меняет; оценивайте вместе с motion_blur_strength.',
            'motion_blur_strength': 'Выраженность усреднения движения внутри экспозиции: 0 отключает временное размытие, 0.5–0.8 — старт. Высокие значения могут скрыть характер кадрового движения. Это не расфокус объектива и не скорость воспроизведения.',
            'blur_mode': 'simple усредняет временные отсчёты без компенсации движения: быстрее, но возможны двойные контуры. flow_integrated использует CPU Farneback и CPU/CUDA деформации. Для первого просмотра simple, для финального — flow_integrated; при ошибках потока сравните оба режима.',
            'blur_samples': 'Число временных отсчётов экспозиции: 3 для быстрой пробы, 7–9 для финального просмотра. Больше — плавнее интегрирование, но дольше расчёт; ошибки optical flow само по себе не исправляет. Покачивание также учитывается внутри экспозиции.',
            'camera_shake_strength': 'Сдвиги, кивки и наклоны съёмочного аппарата от усилий на ручке, с различиями между оборотами. 0 отключает, 0.15–0.30 — умеренно, 0.5 — выраженно. Это движение записывается на плёнку; дрожание кадрового окна Projector настраивается отдельно.',
        }
        for key, tooltip in capture_tooltips.items():
            required[key][1]["tooltip"] = tooltip
        required.update(_optics_inputs())
        for key, value in dict(softness=0.1, vignette_strength=0.12, edge_softness=0.15).items():
            required[key][1]["default"] = value
        required["focus_drift_strength"] = _float(0.03, 0, 0.5, "Медленный дрейф съёмочного фокуса, записываемый на негатив. 0 отключает, 0.02–0.04 — старт. Добавляет небольшое размытие всего кадра поверх softness; не меняет радиус маски edge_start/end. Для проверки движения сначала выключите.", step=0.005)
        required["exposure_variation"] = _float(0.02, 0, 0.3, "Межкадровая неравномерность экспозиции при съёмке. 0 отключает, 0.01–0.02 — деликатный старт. Позже проходит через обе эмульсионные кривые. Не связана со случайным сигналом лампы Projector; для сохранённого примера выключена.", step=0.005)
        required.update(_runtime_inputs())
        return {"required": required}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "film_json", "preview")
    FUNCTION = "apply"
    CATEGORY = "video/stylize/silent_film"

    def apply(self, image, source_fps=36.0, target_fps_min=16.0, target_fps_max=20.0,
              fps_drift_strength=0.7, crank_irregularity=0.15, shutter_fraction=0.65,
              motion_blur_strength=0.8, blur_mode="flow_integrated", blur_samples=9,
              camera_shake_strength=0.25, softness=0.1, vignette_strength=0.12,
              edge_softness=0.15, edge_start=0.6, edge_end=1.0,
              focus_drift_strength=0.03, exposure_variation=0.02,
              seed=1925, compute_device="auto", batch_size=4):
        """Capture one output frame per virtual film exposure and apply acquisition optics."""
        settings = dict(softness=softness, vignette_strength=vignette_strength, edge_softness=edge_softness, edge_start=edge_start, edge_end=edge_end)
        _validate_optics(settings)
        focus_drift_strength = _number("focus_drift_strength", focus_drift_strength, 0, 0.5)
        exposure_variation = _number("exposure_variation", exposure_variation, 0, 0.3)
        source = finish._validate_video_batch(image, normalize=False)
        if len(source) == 1:
            captured = source
            cadence_text = json.dumps({"single_frame": True, "source_frame_count": 1})
            _LOGGER.info("[Silent Film Camera] Один входной кадр: статическая оптика без каденса, optical flow и движения камеры.")
        else:
            # Сохраняем существующий расчёт экспозиций, но не привязываем съёмку к проектору.
            captured, cadence_text = VideoSilentFilmCadence().apply(
                image, source_fps, "undercrank_projected_25fps", target_fps_min, target_fps_max,
                fps_drift_strength, shutter_fraction, motion_blur_strength, blur_mode, blur_samples, _effect_seed(seed, "camera", "capture"),
                cadence_model="hand_crank", crank_irregularity=crank_irregularity,
                camera_shake_strength=camera_shake_strength, compute_device=compute_device)
        focus = torch.tanh(finish._temporal_signal(len(captured), _effect_seed(seed, "camera", "focus"), 1.0, 0.48, torch.device("cpu"), torch.float32))
        exposure = finish._temporal_signal(len(captured), _effect_seed(seed, "camera", "exposure"), exposure_variation, 0.55, torch.device("cpu"), torch.float32)
        masks = {}

        def operation(chunk, start, end):
            """Apply optics and capture instability without monochrome conversion or grain."""
            rgb = _optics(chunk[..., :3], settings, masks)
            if focus_drift_strength and len(captured) > 1:
                focus_mask = ((focus[start:end].to(chunk.device) + 1) * 0.5).view(-1, 1, 1, 1)
                rgb = finish._apply_variable_lens_blur(rgb, focus_mask, rgb.size(2) / 1920.0 * focus_drift_strength * 8.0)
            rgb = (rgb * (1 + exposure[start:end].to(chunk.device).view(-1, 1, 1, 1))).clamp(0, 1)
            return _rgb_alpha(rgb, chunk)

        _LOGGER.info("[Silent Film Camera] План: экспозиции и ручка → съёмочная оптика, фокус и экспозиция.")
        output = _run_stage(captured, "Silent Film Camera", compute_device, batch_size, operation)
        parameters = dict(source_fps=float(source_fps), target_fps_min=float(target_fps_min), target_fps_max=float(target_fps_max),
                          fps_drift_strength=float(fps_drift_strength), crank_irregularity=float(crank_irregularity),
                          shutter_fraction=float(shutter_fraction), motion_blur_strength=float(motion_blur_strength),
                          blur_mode=blur_mode, blur_samples=int(blur_samples), camera_shake_strength=float(camera_shake_strength),
                          focus_drift_strength=focus_drift_strength, exposure_variation=exposure_variation, seed=int(seed), **settings)
        meta = {"history": [], "cadence": json.loads(cadence_text)}
        return output, _write_film_meta(meta, output, "camera", parameters), _preview_frame(output)


class VideoSilentFilmNegative:
    """Expose and develop a monochrome negative represented as film transmittance."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define negative exposure, development, base fog, and independent grain."""
        return {"required": {
            "image": ("IMAGE", {"tooltip": 'Изображение после Camera, до проявки. Пыль съёмочного кадрового окна можно вставить сюда. Выход Negative будет пропусканием плёнки с инвертированной полярностью.'}),
            "exposure_ev": _float(0, -4, 4, "Экспозиция негатива в ступенях. 0 — старт; +1 удваивает условную экспозицию, делает негатив плотнее и обычно осветляет последующий позитив. Для деликатной настройки меняйте по 0.1–0.25. Не путайте с print_exposure_ev, где плюс затемняет копию."),
            "development_contrast": _float(1, 0.2, 3, "Контраст проявленного негатива. 1 — старт; 0.8–0.9 мягче, 1.1–1.2 выразительнее. Слишком высокий контраст теряет детали теней и светов при последующей печати. Сначала настройте exposure_ev, затем этот параметр."),
            "base_fog": _float(0.1, 0, 1, "Плотность подложки и вуали негатива, добавляемая ко всему кадру. Старт 0.1. Увеличение затемняет сам негатив и меняет экспозицию при печати; это не подъём чёрного готового видео. Обычно оставляйте исходное значение и настраивайте экспозицию отдельно."),
            "density_range": _float(2.2, 0.2, 3, "Размах плотности негативной эмульсии; старт 2.2. Увеличение усиливает различие прозрачных и плотных участков, влияет и на контраст, и на яркость последующей копии. Это расширенная настройка материала: сначала подберите exposure_ev и development_contrast."),
            "development_variation": _float(0.01, 0, 0.2, "Медленная межкадровая неравномерность плотности от проявки. 0 — ровная проявка, 0.005–0.01 — деликатно. Меняет негатив, поэтому эффект далее проходит печать. Пространственные пятна и потёки не добавляет; не дублируйте без необходимости exposure_variation и лампу.", step=0.005),
            **_grain_inputs(0.03), **_runtime_inputs(),
        }, "optional": {
            "film_json": ("STRING", {"forceInput": True, "tooltip": "Необязательный паспорт из Camera: каденс, этап, размер и история. Без него Negative работает по image и своим параметрам, начиная новую историю. При подключении этап и форма батча проверяются; после IMAGE-ноды дефектов передайте метаданные напрямую."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("negative", "film_json", "preview")
    FUNCTION = "apply"
    CATEGORY = "video/stylize/silent_film"

    def apply(self, image, film_json=None, exposure_ev=0, development_contrast=1, base_fog=0.1,
              density_range=2.2, development_variation=0.01, grain_strength=0.03,
              grain_size=2, seed=1925, compute_device="auto", batch_size=4):
        """Develop log-exposure into optical density and emit normalized transmittance."""
        images = finish._validate_video_batch(image, normalize=False)
        meta = _read_film_meta(film_json, images, "camera")
        settings = {key: _number(key, value, low, high) for key, value, low, high in (
            ("exposure_ev", exposure_ev, -4, 4), ("development_contrast", development_contrast, 0.2, 3),
            ("base_fog", base_fog, 0, 1), ("density_range", density_range, 0.2, 3),
            ("development_variation", development_variation, 0, 0.2), ("grain_strength", grain_strength, 0, 0.2), ("grain_size", grain_size, 0.1, 16))}
        variation = finish._temporal_signal(len(images), _effect_seed(seed, "negative", "development"), settings["development_variation"], 0.95, torch.device("cpu"), torch.float32)

        def operation(chunk, start, end):
            """Convert scene exposure to negative density and add negative-stock grain."""
            log_exposure = torch.log2(_luma(chunk).clamp_min(1e-6) / 0.18) + settings["exposure_ev"]
            density = settings["base_fog"] + settings["density_range"] * torch.sigmoid(log_exposure * settings["development_contrast"])
            density = density + variation[start:end].to(chunk.device).view(-1, 1, 1, 1)
            density = _density_grain(density, settings["grain_strength"], settings["grain_size"], _effect_seed(seed, "negative", "grain"), start)
            rgb = torch.pow(10.0, -density.clamp(0, 4)).expand(-1, -1, -1, 3)
            return _rgb_alpha(rgb, chunk)

        _LOGGER.info("[Silent Film Negative] План: экспозиция → характеристическая кривая и проявка → зерно негатива.")
        output = _run_stage(images, "Silent Film Negative", compute_device, batch_size, operation)
        settings["seed"] = int(seed)
        return output, _write_film_meta(meta, output, "negative", settings), _preview_frame(output)


class VideoSilentFilmPrint:
    """Print a developed negative onto a separate positive film stock."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define positive print exposure, density response, softness, tint, and grain."""
        return {"required": {
            "image": ("IMAGE", {"tooltip": 'Пропускание негатива из Negative, не его инвертированный preview. Изменения изображения, пыль и царапины действительно переносятся в позитив; скрытая исходная картинка не восстанавливается.'}),
            "print_exposure_ev": _float(0, -4, 4, "Экспозиция при печати позитива. 0 — старт; плюс даёт больше света позитивной эмульсии, плотнее и темнее готовую копию, минус осветляет. Меняйте по 0.1–0.25, оценивая выход Print. Направление противоположно exposure_ev негатива."),
            "print_contrast": _float(1, 0.2, 3, "Контраст позитивной эмульсии, дополнительно к уже проявленному негативу. 1 — старт; ниже мягче, выше выразительнее. Не повышайте одновременно оба контраста: при потере деталей сначала верните оба к 1 и настройте экспозиции."),
            "reference_density": _float(1.2, 0, 4, "Опорная плотность негатива для настройки печати. Старт 1.2 для base_fog=0.1 и density_range=2.2 у Negative. Больше затемняет позитив, меньше осветляет. Не подстраивается автоматически под изменённый материал; обычно сначала настройте print_exposure_ev."),
            "base_fog": _float(0.02, 0, 1, "Плотность прозрачных участков позитивной копии. 0.02 — старт. Больше уменьшает максимальную яркость белого, при избытке делает копию тусклой. Это свойство материала, не яркость лампы; не компенсируйте им слишком тёмную экспозицию печати."),
            "density_range": _float(2.2, 0.2, 3, "Размах плотности позитивной эмульсии. Старт 2.2. Увеличение углубляет тени и меняет тональный рисунок; слишком большое значение скрывает тёмные детали. Для обычной настройки сначала используйте print_contrast и print_exposure_ev."),
            "print_softness": _float(0.05, 0, 1, "Дополнительная мягкость печати: размывает отпечатанное изображение и зерно негатива до добавления зерна позитивной эмульсии. 0 сохраняет резкость; 0.03–0.05 — деликатно. Если Camera уже мягкая, начните с 0."),
            "tone_mode": (["neutral_bw", "warm_print", "sepia_print", "cool_nitrate"], {"default": "neutral_bw", "tooltip": "Оттенок позитивной копии: neutral_bw — нейтральное ЧБ для настройки, warm_print — лёгкая теплота, sepia_print — выраженная сепия, cool_nitrate — холодная художественная тонировка. Не меняет скорость и размер зерна; начните с neutral_bw."}),
            **_grain_inputs(0.015), **_runtime_inputs(),
        }, "optional": {
            "film_json": ("STRING", {"forceInput": True, "tooltip": "Необязательный паспорт из Negative: каденс, этап, размер и история. Без него Print работает по image и своим параметрам, начиная новую историю; image по-прежнему должен быть негативом. При подключении этап и форма батча проверяются; после IMAGE-ноды дефектов передайте метаданные напрямую."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("positive", "film_json", "preview")
    FUNCTION = "apply"
    CATEGORY = "video/stylize/silent_film"

    def apply(self, image, film_json=None, print_exposure_ev=0, print_contrast=1, reference_density=1.2,
              base_fog=0.02, density_range=2.2, print_softness=0.05, tone_mode="neutral_bw",
              grain_strength=0.015, grain_size=2, seed=1925, compute_device="auto", batch_size=4):
        """Expose positive stock through the actual negative, then tone the print."""
        images = finish._validate_video_batch(image, normalize=False)
        meta = _read_film_meta(film_json, images, "negative")
        settings = {key: _number(key, value, low, high) for key, value, low, high in (
            ("print_exposure_ev", print_exposure_ev, -4, 4), ("print_contrast", print_contrast, 0.2, 3),
            ("reference_density", reference_density, 0, 4), ("base_fog", base_fog, 0, 1), ("density_range", density_range, 0.2, 3),
            ("print_softness", print_softness, 0, 1), ("grain_strength", grain_strength, 0, 0.2), ("grain_size", grain_size, 0.1, 16))}
        if tone_mode not in ("neutral_bw", "warm_print", "sepia_print", "cool_nitrate"):
            raise ValueError("Unsupported tone_mode.")

        def operation(chunk, start, end):
            """Let negative density drive print exposure; retain prior-stage defects."""
            transmission = _global_softness(_luma(chunk), settings["print_softness"])
            negative_density = -torch.log10(transmission.clamp_min(1e-4))
            # Две светочувствительные эмульсии дают две инверсии, возвращая позитив.
            exposure = (settings["reference_density"] - negative_density) * (2.2 * settings["print_contrast"]) + (settings["print_exposure_ev"] - 1.0) * math.log(2)
            density = settings["base_fog"] + settings["density_range"] * torch.sigmoid(exposure)
            density = _density_grain(density, settings["grain_strength"], settings["grain_size"], _effect_seed(seed, "print", "grain"), start)
            positive = torch.pow(10.0, -density.clamp(0, 4))
            return _rgb_alpha(finish._tone_gray(positive, tone_mode).clamp(0, 1), chunk)

        _LOGGER.info("[Silent Film Print] План: печать через негатив → позитивная эмульсия → зерно и тонировка копии.")
        output = _run_stage(images, "Silent Film Print", compute_device, batch_size, operation)
        settings.update(seed=int(seed), tone_mode=tone_mode)
        return output, _write_film_meta(meta, output, "print", settings), _preview_frame(output)


class VideoSilentFilmProjector:
    """Project the positive copy with independent gate, lens, lamp, and playback speed."""

    @classmethod
    def INPUT_TYPES(cls):
        """Expose projection controls and an explicit output frame-rate connection."""
        optics = _optics_inputs()
        optics["softness"][1]["default"] = 0.03
        return {"required": {
            "image": ("IMAGE", {"tooltip": 'Позитивная копия из Print. Пыль и царапины на позитиве вставляйте перед этим входом: они должны двигаться вместе с плёнкой и проходить через объектив проектора.'}),
            "projection_fps": _float(25.0, 1, 120, "Частота показа готовой копии. Начните с 25 и соедините выход fps с frame_rate в Video Combine. Больше — быстрее и короче результат. Кадры не интерполируются: длительность = число кадров / fps. Исходный source_fps задаётся только в Camera.", 0.1),
            "gate_weave_px": _float(0.8, 0, 8, "Дрожание плёнки при показе в выходных пикселях: 0 отключает, 0.5–1 — деликатно, 2 — выраженно. Сдвигает весь позитив вместе с зерном и дефектами перед неподвижной оптикой проектора. Не связано с покачиванием съёмочной камеры."),
            **optics,
            "lamp_flicker": _float(0.025, 0, 0.3, "Быстрое межкадровое колебание яркости лампы. 0 отключает, 0.02–0.035 — старт. Не меняет негатив или позитив; работает уже при показе и не связано с ручкой камеры. Это не субкадровое мерцание затвора.", step=0.005),
            "lamp_breathing": _float(0.01, 0, 0.2, "Медленное плавное изменение яркости лампы поверх быстрого lamp_flicker. 0 отключает, 0.01–0.015 — старт. Если яркость слишком нестабильна, сначала выключите его и добавляйте источники колебаний по одному.", step=0.005),
            **_runtime_inputs(),
        }, "optional": {
            "film_json": ("STRING", {"forceInput": True, "tooltip": "Необязательный паспорт из Print: каденс, этап, размер и история. Без него Projector работает по image и своим параметрам, начиная новую историю; image считается позитивом. projection_fps задаётся в самой ноде. При подключении этап и форма батча проверяются."}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "fps", "film_json", "preview")
    FUNCTION = "apply"
    CATEGORY = "video/stylize/silent_film"

    def apply(self, image, film_json=None, projection_fps=25, gate_weave_px=0.8, softness=0.03,
              vignette_strength=0, edge_softness=0, edge_start=0.6, edge_end=1,
              lamp_flicker=0.025, lamp_breathing=0.01, seed=1925, compute_device="auto", batch_size=4):
        """Move the film before stationary projection optics and lamp illumination."""
        images = finish._validate_video_batch(image, normalize=False)
        meta = _read_film_meta(film_json, images, "print")
        settings = dict(softness=softness, vignette_strength=vignette_strength, edge_softness=edge_softness, edge_start=edge_start, edge_end=edge_end)
        _validate_optics(settings)
        projection_fps = _number("projection_fps", projection_fps, 1, 120)
        gate_weave_px = _number("gate_weave_px", gate_weave_px, 0, 8)
        lamp_flicker = _number("lamp_flicker", lamp_flicker, 0, 0.3)
        lamp_breathing = _number("lamp_breathing", lamp_breathing, 0, 0.2)
        def signal(effect, strength, smoothing):
            """Build lamp and transport signals independently of camera cadence."""
            return finish._temporal_signal(len(images), _effect_seed(seed, "projector", effect), strength, smoothing, torch.device("cpu"), torch.float32)
        x, y = signal("gate_x", gate_weave_px, 0.55), signal("gate_y", gate_weave_px * 0.8, 0.6)
        lamp = 1 + signal("lamp_flicker", lamp_flicker, 0.18) + signal("lamp_breathing", lamp_breathing, 0.92)
        masks = {}

        def operation(chunk, start, end):
            """Project the transported print through a stationary optical field."""
            woven = finish._apply_gate_weave(chunk, x[start:end].to(chunk.device), y[start:end].to(chunk.device)) if gate_weave_px else chunk
            rgb = _optics(woven[..., :3], settings, masks)
            rgb = (rgb * lamp[start:end].to(chunk.device).view(-1, 1, 1, 1)).clamp(0, 1)
            return _rgb_alpha(rgb, woven)

        _LOGGER.info("[Silent Film Projector] План: кадровое окно → неподвижная оптика проектора → лампа; показ %.2f fps.", projection_fps)
        output = _run_stage(images, "Silent Film Projector", compute_device, batch_size, operation)
        settings.update(projection_fps=projection_fps, gate_weave_px=gate_weave_px, lamp_flicker=lamp_flicker, lamp_breathing=lamp_breathing, seed=int(seed))
        return output, projection_fps, _write_film_meta(meta, output, "projector", settings), _preview_frame(output)
