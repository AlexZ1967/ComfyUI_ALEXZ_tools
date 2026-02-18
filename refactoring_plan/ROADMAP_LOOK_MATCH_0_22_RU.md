# Roadmap 0.22.x — Look Match (Resolve + Nuke подходы)

Статус: активный рабочий план  
Дата открытия: 2026-02-18  
Линейка версий: `0.22.x`

## 1) Цель

Добавить профессиональный пайплайн цветопереноса для случаев, когда `image` и `reference` сильно отличаются по тону/контрасту/палитре.

План включает два независимых режима:
- `Resolve-style`: монолитная нода "сделай хорошо из коробки".
- `Nuke-style`: модельный подход `Build -> Apply` для повторяемости на шотах.

## 2) Результат по продукту

К концу `0.22.x` в пакете должны быть:
1. `ImageLookMatchResolve`
2. `ImageLookMatchNukeBuild`
3. `ImageLookMatchNukeApply`
4. Обновленные гайды, smoke-тесты, docs-check совместимость.

## 3) Границы и принципы

1. Без breaking changes для существующих `Color Match` и `Seam Match`.
2. Alpha-канал сохраняется автоматически (если был на входе).
3. Базовый runtime: `torch` first, минимум CPU roundtrip.
4. Для долгих циклов обязательна единая проверка interrupt.
5. Новые ноды должны иметь режимы `compute_device`: `auto/cpu/cuda`.

## 4) Этапы

## Phase A — Architecture + Contracts

Цель: зафиксировать контракты входов/выходов и формат look-модели.

Сделать:
1. Специфицировать `INPUT_TYPES`/`RETURN_TYPES` для трех нод.
2. Зафиксировать JSON-форматы:
   - `look_json` (Resolve)
   - `look_model_json` (Nuke Build)
   - `apply_json` (Nuke Apply)
3. Определить общий helper-слой в `utils` для:
   - цветовых пространств (`rgb/lab/oklab`);
   - tone-моделей;
   - LUT сериализации (`.cube` text).

Критерии выхода:
1. Контракты описаны в коде и guide.
2. Smoke-тесты загружают ноды и валидируют базовые поля JSON.

## Phase B — Resolve MVP

Цель: дать рабочий качественный монолит для одиночного кадра.

Сделать:
1. Реализовать `ImageLookMatchResolve` pipeline:
   - экспозиция/белый баланс;
   - tone matching (`monotonic_spline` или `gamma_gain_lift`);
   - palette matching (`lut3d` базовый).
2. Добавить опции защиты кожи и региональные маски (если поданы).
3. Добавить экспорт LUT (`cube_text` output или toggle).

Критерии выхода:
1. На сложных парах визуально лучше текущих глобальных методов.
2. 1080p single frame: приемлемое время на `cuda` и `cpu`.
3. JSON содержит диагностику этапов и веса.

## Phase C — Nuke Build MVP

Цель: выделить построение reusable look-модели.

Сделать:
1. Реализовать `ImageLookMatchNukeBuild`:
   - fit global grade;
   - fit tone;
   - fit hue sectors;
   - fit local regions (по маскам).
2. Сохранить модель в `look_model_json` + опционально LUT.
3. Добавить стабильные идентификаторы версий схемы модели.

Критерии выхода:
1. Модель переносится между кадрами без падений.
2. Есть backward-safe `schema_version` в JSON.

## Phase D — Nuke Apply + Temporal

Цель: корректно применять модель к сериям кадров.

Сделать:
1. Реализовать `ImageLookMatchNukeApply` для batch/sequence сценария.
2. Добавить temporal stabilization:
   - EMA по параметрам/результату;
   - shot-change gate.
3. Добавить debug-диагностику применения.

Критерии выхода:
1. Нет заметного фликера в плавных сценах.
2. На резком cut нет "перетягивания" старого look.

## Phase E — QA + Docs + Release

Цель: стабилизировать и документировать для production.

Сделать:
1. Расширить smoke/contract tests.
2. Добавить guide с пресетами и примерами "когда какой режим".
3. Обновить README/CHANGELOG.

Критерии выхода:
1. `make docs-check` и `make smoke` проходят.
2. Гайды содержат quick-start и рекомендации по производительности.

## 5) Порядок реализации (очередь)

1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E

## 6) Риски и меры

1. Риск: сильный дрейф оттенков кожи.
   Мера: skin protection + отдельный вес skin-loss.
2. Риск: LUT-переобучение на одном кадре.
   Мера: identity/smoothness регуляризация.
3. Риск: медленная работа на CPU.
   Мера: downscale профиль + staged optimization + early stop.
4. Риск: UI перегружен параметрами.
   Мера: отдельные ноды и compact defaults вместо одной "комбайн" ноды.

## 7) Definition of Done для 0.22.x

1. Три новые ноды зарегистрированы и документированы.
2. Есть минимум 2 практических workflow-примера:
   - "single frame look transfer"
   - "build once, apply to sequence"
3. Тесты покрывают:
   - базовый smoke,
   - RGBA auto-preserve,
   - CPU/CUDA ветки,
   - schema-version совместимость JSON.
