# ComfyUI_ALEXZ_tools — Трекер исполнения плана стабилизации

Статус: active
Связанный план: [PLAN_SAFE_STABILIZATION_RU.md](PLAN_SAFE_STABILIZATION_RU.md)
Дата старта: 2026-04-23
Последнее обновление: 2026-04-23 (`plan completed`, `follow-up baseline + image_prepare smoke`)

## 1) Как используем этот трекер

Этот файл фиксирует исполнение плана по факту.

Правила:
- план в `PLAN_SAFE_STABILIZATION_RU.md` задает scope и порядок фаз;
- этот файл отражает текущее состояние исполнения;
- после каждого завершенного атомарного шага обновляются:
  - текущая фаза;
  - статус шагов;
  - что сделано;
  - чем проверено;
  - что идет следующим;
- если в фазе найден блокер, он фиксируется здесь, а не теряется в переписке.

## 2) Сводка прогресса

Общий прогресс: `6 / 6` фаз завершено
Текущая фаза: `Phase F — Контрольный полный прогон`
Текущий шаг: `План выполнен, follow-up улучшения зафиксированы`
Следующий шаг: `Новых обязательных шагов по этому плану нет`
Статус репозитория: `план выполнен, baseline обновлен`

## 3) Статус фаз

| Фаза | Название | Статус | Прогресс | Примечание |
|---|---|---|---:|---|
| A | Freeze и минимальный safety baseline | completed | 100% | Baseline подтвержден локальными проверками |
| B | Import-совместимость `utils.module_browser` | completed | 100% | Legacy shim layer добавлен и проверен |
| C | Import-time GPU/Comfy side effects | completed | 100% | `image_prepare` и `video_inpaint` переведены на lazy runtime imports |
| D | Локальная стабилизация node loader | completed | 100% | Диагностика partial failures улучшена, registry import path исправлен |
| E | Закрытие backend helper regressions | completed | 100% | `module_browser` backend suite полностью зеленая |
| F | Контрольный полный прогон | completed | 100% | Полный `pytest` зеленый |

Статусы:
- `pending` — еще не начинали
- `in_progress` — сейчас в работе
- `blocked` — есть явный блокер
- `completed` — фаза завершена и проверена

## 4) Атомарные шаги

### Phase A

- [x] Зафиксировать baseline-команды как обязательный стартовый набор
- [x] Подтвердить список текущих зеленых проверок
- [x] Зафиксировать список известных красных зон
- [x] Обновить этот трекер и пометить Phase A как `completed`

### Phase B

- [x] Определить исторические import-path entrypoints, которые должны жить
- [x] Добавить thin compatibility shims
- [x] Прогнать точечные `module_browser` тесты
- [x] Прогнать агрегирующий набор `tests/test_module_browser_*.py -x`
- [x] Обновить этот трекер и пометить Phase B как `completed`

### Phase C

- [x] Найти import-time GPU/Comfy зависимости в целевых нодах
- [x] Перенести тяжелые импорты в lazy/runtime точки
- [x] Сохранить GPU requirement только как runtime check
- [x] Прогнать прицельные smoke/unit тесты
- [x] Обновить этот трекер и пометить Phase C как `completed`

### Phase D

- [x] Разобрать текущий контракт node loader
- [x] Улучшить различимость expected runtime-unavailable vs real import error
- [x] Не менять формат mapping dictionaries
- [x] Прогнать целевые тесты loader/registry/smoke
- [x] Обновить этот трекер и пометить Phase D как `completed`

### Phase E

- [x] Разобрать оставшиеся backend helper regressions по кластерам
- [x] Править код или тесты только после явной классификации причины
- [x] Прогнать `tests/test_module_browser_*.py`
- [x] Обновить этот трекер и пометить Phase E как `completed`

### Phase F

- [x] Прогнать полный `pytest`
- [x] Зафиксировать остаточные падения или green status
- [x] Обновить baseline
- [x] Обновить этот трекер и пометить Phase F как `completed`

## 5) Журнал исполнения

### 2026-04-23

Старт исполнения.

Зафиксировано:
- базовый план принят;
- выбран способ отображения прогресса через отдельный execution tracker;
- текущая работа начинается с `Phase A`.

### 2026-04-23 — Phase A completed

Сделано:
- baseline-команды подтверждены повторным локальным прогоном в `p313`;
- стартовая точка исполнения зафиксирована как валидная;
- Phase A закрыта без изменения runtime-поведения.

Проверки:
- `conda run -n p313 python utils/docs_check.py` → OK
- `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_api_contracts_golden.py` → `20 passed`
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k seam_match` → `8 passed`
- `conda run -n p313 node --check web/widget_visibility_profiles.js` → OK

### 2026-04-23 — Phase B completed

Сделано:
- собран список legacy import-path entrypoints, ожидаемых тестами;
- добавлен thin compatibility shim layer в корне `utils/module_browser/`;
- отдельно исправлен `manifest_check`, чтобы legacy monkeypatch-контракт модуля сохранился;
- агрегирующий `module_browser` прогон стал зеленым.

Проверки:
- `conda run -n p313 pytest -q tests/test_module_browser_value_ops.py tests/test_module_browser_release_ops.py tests/test_module_browser_state_store.py tests/test_module_browser_runtime_refresh_ops.py` → `14 passed`
- `conda run -n p313 pytest -q tests/test_module_browser_manifest_check.py` → `2 passed`
- `conda run -n p313 pytest -q tests/test_module_browser_*.py -x` → `143 passed`

### 2026-04-23 — Phase C started

Сделано:
- локализованы текущие import-time Comfy/GPU кандидаты;
- основными целями для refactor подтверждены:
  - `nodes/image_prepare.py`
  - `nodes/video_inpaint.py`

Дополнительные наблюдения:
- `utils/interrupt.py` уже использует lazy import pattern;
- другие ноды в ряде мест используют `torch.cuda.is_available()` внутри runtime-логики, а не на import-time.

### 2026-04-23 — Phase C completed

Сделано:
- `nodes/image_prepare.py` переведен на lazy import `comfy.model_management` и `comfy.utils` только внутри runtime path;
- `nodes/video_inpaint.py` переведен на lazy runtime access к `model_management` через compatibility proxy;
- в `video_inpaint` сохранен legacy monkeypatch contract для `mod.model_management.get_torch_device` в smoke-тестах;
- `video_inpaint` дополнительно переведен на lazy import `folder_paths`, чтобы прямой headless import модуля не требовал Comfy input subsystem на этапе импорта.

Проверки:
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "video_inpaint or seam_match"` → `13 passed`
- `conda run -n p313 pytest -q tests/test_module_browser_*.py -x` → `143 passed`
- headless import в stubbed package context:
  - `OK image_prepare`
  - `OK video_inpaint`

Ограничение scope:
- у `video_frame_match` и `video_cut_match` по-прежнему есть import-time зависимость от `folder_paths`, но это не было целевым scope текущей GPU/Comfy фазы.

### 2026-04-23 — Phase D started

Сделано:
- зафиксировано, что loader остается tolerant и переживает partial import failures;
- следующий шаг — улучшить диагностичность причин падения без изменения public mappings.

### 2026-04-23 — Phase D completed

Сделано:
- `nodes/__init__.py` обогащен классификацией `failure_kind` для `LOAD_RESULTS["fail"]`;
- добавлено различение `runtime_unavailable` vs `import_error` без изменения public mappings;
- для runtime-недоступности loader теперь пишет более мягкую диагностику;
- исправлен относительный импорт в `utils/module_browser/catalog/component_registry.py`, из-за которого registry в некоторых test-runtime сценариях терял node specs.

Проверки:
- `conda run -n p313 pytest -q tests/test_slice0_registry.py` → `9 passed`
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py` → `69 passed`
- `conda run -n p313 pytest -q tests/test_module_browser_*.py -x` → `143 passed`

### 2026-04-23 — Phase E completed

Сделано:
- после Phase B-D отдельного хвоста backend helper regressions не осталось;
- `module_browser` backend suite целиком зеленая, поэтому Phase E закрыта без отдельной широкой переписи.

Проверки:
- `conda run -n p313 pytest -q tests/test_module_browser_*.py -x` → `143 passed`

### 2026-04-23 — Phase F completed

Сделано:
- выполнен полный контрольный прогон репозитория;
- план стабилизации закрыт как выполненный;
- новый рабочий baseline зафиксирован в этом трекере.

Проверки:
- `conda run -n p313 pytest -q` → `237 passed`
- `conda run -n p313 python utils/docs_check.py` → `OK`
- `conda run -n p313 node --check web/widget_visibility_profiles.js` → `OK`

### 2026-04-23 — Follow-up after plan completion

Сделано:
- добавлен отдельный smoke-тест для `image_prepare`, чтобы Phase C-подобные регрессии ловились прямым тестом;
- добавлен baseline helper `scripts/save_baseline.sh`;
- добавлен `make save-baseline` как удобная точка входа;
- сформирован `baseline.json` с зафиксированными результатами контрольных проверок.

Проверки:
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "image_prepare or video_inpaint or seam_match"` → `14 passed`
- `conda run -n p313 pytest -q` → `238 passed`
- `bash scripts/save_baseline.sh` → `baseline.json` создан

## 6) Последние проверки

Последний подтвержденный baseline:
- `conda run -n p313 pytest -q` → `238 passed`
- `conda run -n p313 python utils/docs_check.py` → OK
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py` → green, зафиксирован в `baseline.json`
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k seam_match` → OK
- `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_api_contracts_golden.py` → OK
- `conda run -n p313 pytest -q tests/test_slice0_registry.py` → `9 passed`
- `conda run -n p313 node --check web/widget_visibility_profiles.js` → OK
- `conda run -n p313 pytest -q tests/test_module_browser_*.py -x` → `143 passed`
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "video_inpaint or seam_match"` → `13 passed`
- `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "image_prepare or video_inpaint or seam_match"` → `14 passed`
- `bash scripts/save_baseline.sh` → `baseline.json` создан

Известная красная зона:
- нет: контрольный полный прогон зеленый

## 7) Блокеры

На текущий момент:
- явных блокеров старта нет
- блокеров для перехода в `Phase C` нет
- блокеров завершения плана нет

## 8) Правило обновления прогресса

После каждого завершенного шага обновляем здесь:
- `Последнее обновление`
- `Сводка прогресса`
- таблицу статуса фаз
- чеклист текущей фазы
- `Журнал исполнения`
- `Последние проверки`

Это и будет основное видимое место прогресса в репозитории.
