# ROADMAP: ComfyUI_ALEXZ_tools

Дата фиксации: 2026-07-19
Горизонт: H2 2026
Статус базы на момент составления:
- `255` тестов проходят
- `docs-check: OK`
- JS behavioral test для `Module Node Picker` проходит

## Цели роадмапа

1. Повысить совместимость с актуальными стандартами ComfyUI Registry и Manager.
2. Снизить риск регрессий в крупных нодах и `Module Node Picker`.
3. Упростить сопровождение за счет декомпозиции крупных файлов и legacy-слоев.
4. Сделать зависимости и сетевое поведение более предсказуемыми и безопасными.
5. Укрепить frontend/backend контракты без ломки существующих workflow.

## Принципы выполнения

- Не ломать существующие type names нод и публичные JSON-контракты без явной миграции.
- Любой заметный рефакторинг выполнять только после фиксации baseline-тестами.
- Сначала устранять архитектурные риски вокруг packaging, dependency flow и API surface.
- Декомпозировать крупные модули по ролям, а не по произвольным кускам.
- Новые проверки добавлять в `Makefile` и прогонять в `p313`.

## Приоритеты

- `P0`: совместимость, безопасность, публикационные блокеры
- `P1`: снижение стоимости сопровождения
- `P2`: UX и качество инструментов разработки
- `P3`: исследовательские и опциональные улучшения

## Phase 1: Registry / Packaging Hardening

Срок: 2026-07-19 -> 2026-08-02
Приоритет: `P0`
Статус: ✅ выполнено (2026-07-19, `0.37.0`)

### Цели

- Привести упаковку к текущим ожиданиям ComfyUI Registry.
- Уменьшить риск будущих проблем при установке через Manager / Registry.

### Задачи

1. Расширить `pyproject.toml`:
   - добавить `repository`
   - добавить `requires-python`
   - добавить `urls`
   - добавить `classifiers`
   - добавить `requires-comfyui`
   - заполнить `Icon` и при необходимости `Banner`
2. Перепроверить `requirements.txt` на излишне широкие или конфликтные зависимости.
3. Явно разделить:
   - обязательные runtime-зависимости
   - опциональные зависимости для отдельных нод
4. Подготовить `.comfyignore`, если в публикацию не должны попадать лишние heavy/dev assets.
5. Актуализировать `README.md` под registry-style install flow.

Результат:
- ✅ Расширили `pyproject.toml` registry-метаданными: `requires-python`, `urls`, `classifiers`, `requires-comfyui`, `optional-dependencies`.
  Выполнено: 2026-07-19 (`0.37.0`), `pyproject.toml`.
- ✅ Подготовили registry assets и publication filters.
  Выполнено: 2026-07-19 (`0.37.0`), `assets/registry/icon.svg`, `assets/registry/banner.svg`, `.comfyignore`.
- ✅ Актуализировали install/update/dependency flow в `README.md` под registry-style публикацию.
  Выполнено: 2026-07-19 (`0.37.0`), `README.md`.

### Артефакты

- обновленный `pyproject.toml`
- уточненный `requirements.txt`
- обновленный `README.md`
- при необходимости `.comfyignore`

### Критерии завершения

- метаданные пакета полны и соответствуют текущему spec
- README описывает install/update/dependency flow без двусмысленностей
- нет скрытой зависимости на ручное знание структуры репозитория

## Phase 2: Dependency Flow Cleanup

Срок: 2026-08-02 -> 2026-08-16
Приоритет: `P0`
Статус: ✅ выполнено (2026-07-19, `0.37.0`)

### Цели

- Убрать наиболее спорный runtime dependency-install путь из UI.
- Сохранить полезную диагностику без запуска `pip install` из API-роутов.

### Задачи

1. Перевести `Module Node Picker` из install-mode в advisory-mode:
   - показывать, что `requirements` изменились
   - показывать рекомендуемую команду
   - не запускать `pip install` автоматически
2. Оставить статус `requirements pending` в state/tracker слое.
3. Сохранить route compatibility только если это нужно фронтенду:
   - либо вернуть `403/info-only`
   - либо отдавать structured payload с инструкцией вместо инсталла
4. Упростить `update_ops.py`, убрав фактическую установку зависимостей из runtime API.
5. Обновить тесты и документацию под новый контракт.

Результат:
- ✅ Перевели `Module Node Picker` из install-mode в advisory/manual mode.
  Выполнено: 2026-07-19 (`0.37.0`), backend routes и frontend follow-up payloads больше не инициируют runtime `pip install`.
- ✅ Сохранили `requirements pending` как tracker/state сигнал с точным путем к `requirements.txt`.
  Выполнено: 2026-07-19 (`0.37.0`), `utils/module_browser/tracker.py`, `utils/module_browser/update_status_payloads.py`.
- ✅ Упростили runtime API и update flow до info-only контракта для dependency changes.
  Выполнено: 2026-07-19 (`0.37.0`), `utils/module_browser_api/routes.py`, `utils/module_browser/jobs/update_ops.py`.
- ✅ Обновили tests/docs под новый контракт и добавили helper для ручной UI-проверки.
  Выполнено: 2026-07-19 (`0.37.0`), API/frontend tests, `scripts/module_picker_requirements_demo.py`, `CHANGELOG.md`.

### Артефакты

- обновленные `utils/module_browser_api/routes.py`
- обновленные `utils/module_browser/jobs/update_ops.py`
- обновленные frontend payload handlers
- обновленные тесты API contracts

### Критерии завершения

- UI больше не инициирует `pip install` как часть обычного runtime flow
- фронтенд корректно объясняет пользователю, что нужно сделать вручную
- contract tests отражают новое поведение

## Phase 3: Module Node Picker Stabilization Pass

Срок: 2026-08-16 -> 2026-09-06
Приоритет: `P1`

### Цели

- Уменьшить размер и связность backend-фасада `module_node_browser_api`.
- Зафиксировать тонкие frontend/backend контракты до следующего цикла развития.

### Задачи

1. Дорезать `utils/module_node_browser_api.py`:
   - state/cache helpers
   - network/release helpers
   - thin facade only
2. Просмотреть compatibility shims в `utils/module_browser/*`:
   - отметить, какие еще реально нужны
   - удалить неиспользуемые
3. Сузить `except Exception` там, где ошибка уже понятна по домену.
4. Дофиксировать contract boundaries:
   - routes
   - payload shape
   - runtime refresh/update jobs
5. Добавить smoke-check на route registration и info-only режимы.

### Артефакты

- уменьшенный `utils/module_node_browser_api.py`
- сокращенный legacy shim слой
- обновленные contract tests

### Критерии завершения

- backend facade перестает быть монолитом общего назначения
- legacy-import слой явно минимизирован
- исключения становятся диагностичнее и уже по типам

## Phase 4: Large Node Decomposition

Срок: 2026-09-06 -> 2026-10-18
Приоритет: `P1`

### Цели

- Снизить стоимость изменений в самых тяжелых нодах.
- Перевести крупные файлы на модульную структуру без ломки публичных классов.

### Приоритетный порядок

1. `nodes/image_download_dzi_tiles.py`
2. `nodes/image_download_iiif.py`
3. `nodes/image_descreen_adaptive.py`
4. `nodes/image_color_match.py`
5. `nodes/image_look_match.py`

### Подход

Для каждого крупного файла выделять подпакеты по ролям:
- `input parsing / validation`
- `transport / fetch`
- `image assembly / transforms`
- `metrics / scoring`
- `json contract / serialization`
- `Comfy node adapter`

### Конкретные задачи

1. Вынести чистые helper-функции в `nodes/<feature>_ops.py` или `utils/<feature>/...`.
2. Сохранить классы нод и `FUNCTION/RETURN_TYPES/CATEGORY` на прежних местах.
3. Покрыть выделенные helper-слои unit-тестами отдельно от smoke-тестов нод.
4. Убрать повторяющуюся логику:
   - slug/title normalization
   - retry/timeout helpers
   - file naming collision policy
   - network diagnostics payloads

### Артефакты

- декомпозированные модули по 1 feature family за итерацию
- новые unit tests
- неизмененные публичные type names нод

### Критерии завершения

- ни один из целевых файлов не остается “single-file subsystem”
- pure logic отделена от Comfy adapter layer
- regressions прикрыты тестами до merge

## Phase 5: Frontend Tooling Upgrade

Срок: 2026-10-18 -> 2026-11-01
Приоритет: `P2`

### Цели

- Сделать JS-проверки частью штатного dev-loop.
- Снизить риск тихих фронтенд-регрессий в `Module Node Picker`.

### Задачи

1. Расширить `Makefile`:
   - добавить `js-test`
   - добавить `js-check-all`
   - добавить общий `test` target
2. Прогонять syntax check не по одному файлу, а по всему `web/`.
3. Включить `tests/js/test_module_node_picker_frontend_behavior.mjs` в стандартный сценарий проверок.
4. При необходимости добавить lightweight lint/format contract для JS.
5. Добавить короткий guide для локального frontend-check workflow.

### Артефакты

- обновленный `Makefile`
- обновленный `README.md` / `AGENTS.md`
- улучшенный JS verification flow

### Критерии завершения

- frontend behavior test запускается штатно
- синтаксис проверяется по всему `web/`
- стандартные команды разработки покрывают и Python, и JS слой

## Phase 6: Trove / Network Strategy Revision

Срок: 2026-11-01 -> 2026-11-22
Приоритет: `P2`

### Цели

- Сделать `SearchTroveImageIDs` устойчивее и ближе к официальному способу интеграции.
- Снизить зависимость от headless browser scraping.

### Задачи

1. Добавить API-first режим для Trove:
   - через официальный API key flow
   - с ясной конфигурацией ключа
2. Оставить browser fallback только как optional advanced mode.
3. Явно маркировать ограничения:
   - anti-bot
   - нестабильность DOM
   - необходимость Chrome
4. Унифицировать сетевую диагностику для IIIF/DZI/Trove family.
5. Обновить guide и node tooltips.

### Артефакты

- улучшенный `nodes/trove_search_ids.py`
- docs по API key setup
- обновленные тесты для API-first ветки

### Критерии завершения

- основной путь не зависит от headless Chrome
- fallback путь явно вторичен и документирован
- пользователь понимает, какой режим надежный, а какой best-effort

## Phase 7: Cleanup / Quality Sweep

Срок: 2026-11-22 -> 2026-12-13
Приоритет: `P2`

### Цели

- Закрыть технические хвосты после основных изменений.
- Подготовить репозиторий к следующему циклу развития.

### Задачи

1. Пересмотреть широкие `except Exception` и сузить их там, где это возможно без потери UX.
2. Удалить устаревшие compatibility shims, если на них больше нет живых импортов.
3. Проверить consistency:
   - docstrings
   - metadata
   - guide links
   - changelog entries
4. Пройтись по TODO/legacy comments в `propainter` и вспомогательных util-файлах.
5. Подготовить release notes для следующей версии.

### Артефакты

- cleanup commits
- обновленный `CHANGELOG.md`
- финальный stabilization pass

### Критерии завершения

- нет явно лишнего legacy слоя
- документация и тесты синхронизированы
- релизный diff читаем и объясним

## Риски

### Высокие

- Ломка frontend/backend контрактов `Module Node Picker`
- Непреднамеренная смена JSON-структур, которые уже используются в workflow
- Регрессии в больших image/network нодах после декомпозиции

### Средние

- Слишком раннее удаление compatibility shim-слоя
- Разъезд документации и реального поведения после смены dependency flow
- Неполное покрытие опциональных зависимостей

### Низкие

- Косметические несоответствия метаданных Registry
- Неидеальная структура dev-команд в `Makefile`

## Минимальный safe order

1. Phase 1
2. Phase 2
3. Phase 5
4. Phase 3
5. Phase 4
6. Phase 6
7. Phase 7

Такой порядок дает сначала packaging и safety improvements, затем tooling, и только потом глубокую декомпозицию.

## Что не делать в этом цикле

- Не переименовывать существующие node type names.
- Не менять массово категории нод без migration note.
- Не совмещать глубокий рефакторинг и feature expansion в одном PR.
- Не трогать ProPainter-ядро без отдельной причины и изолированных тестов.

## Definition of Done для H2 2026

- Репозиторий совместимее с актуальным Registry/Manager flow.
- Runtime dependency-install убран или переведен в безопасный advisory режим.
- `Module Node Picker` имеет более узкую backend surface area.
- Самые крупные node-модули частично или полностью декомпозированы.
- Dev-loop покрывает Python docs/tests и JS checks штатными командами.
- Сетевые ноды документированы и предсказуемы по ограничениям.
