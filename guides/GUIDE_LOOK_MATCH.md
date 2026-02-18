# Look Match (Resolve + Nuke) — Guide

## Назначение
Набор нод для color/look matching в случаях, когда референс и исходник сильно отличаются.  
В `0.22.x` это Phase A (contract baseline): зафиксированы интерфейсы, JSON-схемы и безопасное поведение без агрессивной перекраски.

## Когда использовать
- Нужен будущий профессиональный pipeline `Resolve-style` и `Nuke-style`.
- Нужно заранее собрать workflow под стабильные контракты входов/выходов.
- Требуется совместимость для дальнейшего апгрейда Phase B/C/D без пересборки графа.

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
- Нужна максимальная художественная точность уже сейчас: пока используйте ваши текущие ручные пайплайны, т.к. Phase A — контрактный базис.

## Интерпретация выходов
- `matched_image`: результат применения текущего этапа.
- `look_json`: JSON диагностики resolve-режима.
- `look_model_json`: JSON-модель build-режима.
- `apply_json`: JSON статуса применения модели.
- `cube_text`: текст `.cube` (identity в Phase A при включенном экспорте).

## Типовые ошибки и решения
- `invalid_or_missing_look_model_schema` в `apply_json`: подайте валидный `look_model_json` из `Look Match Nuke Build`.
- `cuda_requested_but_unavailable`: переключите `compute_device` в `auto` или `cpu`.
- Неожиданный визуальный эффект: в Phase A intentionally baseline-логика; качественная перекраска будет в следующих фазах roadmap.

## Производительность
- Phase A быстрый и безопасный, т.к. без тяжелой оптимизации.
- Для больших батчей используйте `compute_device=auto`.
- Для слабых систем снижайте `downscale_long_side`.
