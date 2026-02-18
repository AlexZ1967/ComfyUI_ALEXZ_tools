# Look Match (Resolve + Nuke) — Guide

## Назначение
Набор нод для color/look matching в случаях, когда референс и исходник сильно отличаются.  
В `0.22.x`:
- `Look Match Resolve` — Phase B MVP (рабочий staged алгоритм).
- `Look Match Nuke Build/Apply` — Phase A contract baseline.

## Когда использовать
- Нужен быстрый и управляемый auto-look для сложного референса (`Look Match Resolve`).
- Нужна заготовка model-pipeline Build/Apply под дальнейшую эволюцию (`Nuke` ноды).
- Требуется совместимость контрактов для последующих фаз roadmap.

## Минимальный сценарий (3 шага)
1. Для монолитного режима используйте `Look Match Resolve`.
2. Для модельного режима используйте `Look Match Nuke Build` -> `Look Match Nuke Apply`.
3. Анализируйте `look_json` / `look_model_json` / `apply_json` (поля `schema_name`, `schema_version`, `phase`).

## Параметры
Общие:
- `compute_device`: `auto` / `cpu` / `cuda`.
- `working_space`: `oklab` / `lab`.
- `downscale_long_side`: `as_is` / `1440p` / `1080p` / `720p`.

`Look Match Resolve`:
- `reference`, `image`, `strength`
- `tone_model`, `palette_model`, `lut_size`
- `w_exposure`, `w_tone`, `w_chroma`
- `skin_protection`, `skin_protection_strength`
- `subject_mask`, `sky_mask`, `ground_mask`
- `export_lut_cube`

Алгоритм `Look Match Resolve` (Phase B MVP):
1. Fit exposure/WB gain по downscaled версии.
2. Fit tone model (`monotonic_spline` или `gamma_gain_lift`).
3. Fit palette affine (`lut3d`=linear fit, `rbf`=mean/std fallback).
4. Применение stages на full-res с весами.
5. Опциональная защита skin-tones.

`Look Match Nuke Build`:
- `reference`, `source`
- `fit_global`, `fit_tone`, `fit_hue_sectors`, `fit_local_regions`
- `skin_mask`, `sky_mask`, `ground_mask`, `subject_mask`
- `export_lut_cube`, `lut_size`

`Look Match Nuke Apply`:
- `image`, `look_model_json`, `strength`
- `temporal_stabilization`, `temporal_alpha`, `shot_change_threshold`

## Decision helper
- Нужна "одна нода, быстро": `Look Match Resolve`.
- Нужен reusable look для серии кадров: `Look Match Nuke Build` + `Look Match Nuke Apply`.
- Нужна максимальная управляемость по регионам/шотам: пока через ручной грейдинг или будущие фазы Nuke-пайплайна.

## Интерпретация выходов
- `matched_image`: результат применения текущего этапа.
- `look_json`: JSON диагностики resolve-режима.
- `look_model_json`: JSON-модель build-режима.
- `apply_json`: JSON статуса применения модели.
- `cube_text`: текст `.cube` (для Resolve-ноды запекается текущий fitted look).

## Типовые ошибки и решения
- `invalid_or_missing_look_model_schema` в `apply_json`: подайте валидный `look_model_json` из `Look Match Nuke Build`.
- `cuda_requested_but_unavailable`: переключите `compute_device` в `auto` или `cpu`.
- Слишком агрессивный результат в Resolve-нoде: снизьте `strength`, `w_tone`, `w_chroma` и/или включите `skin_protection`.

## Производительность
- `Look Match Resolve` использует fit на downscaled изображении и apply на full-res.
- Для больших батчей используйте `compute_device=auto`.
- Для слабых систем снижайте `downscale_long_side` до `720p`.
- `Nuke Build/Apply` в текущей фазе легковесны и служат контрактной базой.
