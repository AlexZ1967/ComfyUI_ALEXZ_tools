# Предложение: рефакторинг `utils/module_node_browser_api.py`

## Контекст
`utils/module_node_browser_api.py` сейчас совмещает слишком много обязанностей:
- wiring HTTP-роутов (aiohttp/PromptServer)
- оркестрация long-running job (refresh/update) и хранение статуса
- кэши/TTL и persisted state I/O
- glue-код поверх `utils/module_browser/*`

В результате это "god module": сложнее тестировать, сложнее менять безопасно, выше риск регрессий.

## Цели
- Сохранить обратную совместимость всех роутов и JSON payload (ключи, значения по умолчанию).
- Разделить ответственности на небольшие модули, чтобы уменьшить когнитивную нагрузку.
- Сделать состояние/кэши явными и тестируемыми (без неочевидных глобалов).
- Подготовить базу для будущих расширений (доп. поля статуса, сигналы в UI) без роста хаоса.
- Сохранить текущую runtime-модель (threads/locks/TTL/логирование).

## Не цели
- Не менять UX/тексты/дефолты, если это не оговорено отдельно.
- Не переписывать фронтенд-оркестрацию виджета.
- Не делать "большое переименование" публичных функций внутри `utils/module_browser/*`.
- Не менять модель исполнения job на async (threads остаются).

## Целевая структура после рефакторинга
Выделить отдельный пакет:
- `utils/module_browser_api/`

Предлагаемые модули:
- `utils/module_browser_api/state.py`
  - все runtime-глобалы: `_REFRESH_STATUS`, `_UPDATE_STATUS`, кэши, locks, TTL-константы
  - `get_state()` singleton, чтобы контролировать инициализацию и побочные эффекты в тестах
- `utils/module_browser_api/logging_ops.py`
  - `_update_console_log`, `_refresh_console_log`, нормализация log-mode
- `utils/module_browser_api/node_introspection.py`
  - `_node_mappings`, `_build_node_snapshots`, относительные пути/классификация
- `utils/module_browser_api/handlers_refresh.py`
  - логика refresh job, статус, progress callback wiring
- `utils/module_browser_api/handlers_update.py`
  - логика update job, статус, resolve targets, requirements pending
- `utils/module_browser_api/handlers_catalog.py`
  - `/alexz_tools/node_catalog` и сборка payload
- `utils/module_browser_api/routes.py`
  - только регистрация роутов PromptServer/aiohttp
- `utils/module_browser_api/__init__.py`
  - минимальная поверхность ре-экспорта (см. план совместимости)

`utils/module_node_browser_api.py` оставить как тонкий shim:
- импортирует и вызывает `utils.module_browser_api.routes.register_routes(...)`
- переэкспортирует нужные символы (если где-то есть прямые импорты)

## План по фазам (безопасно, небольшими шагами)

### Фаза 0: Инвентаризация и "freeze"
- Короткая заметка для разработчика:
  - список роутов и ключей payload
  - текущие ключи статуса refresh/update
  - где лежит persisted state и какие ожидания по схеме/версии
- Добавить smoke-тест на импорт backend-модуля вне ComfyUI (если такого нет).

Критерий готовности:
- Тесты проходят без изменений.
- Runtime-поведение не меняется.

### Фаза 1: Вынос состояния и консольного логирования
- Перенести все глобалы/locks/TTL/шаблоны статуса в `state.py`.
- Вынести `_update_console_log`/`_refresh_console_log` и log-mode в `logging_ops.py`.
- В текущем коде заменить прямой доступ к глобалам на `state.<...>`.
- Сохранить поведение тестов, которые иногда заменяют кэши на `None` (явные `ensure_*` хелперы).

Критерий готовности:
- Payload ключи не поменялись.
- Проходят:
  - `conda run -n p313 pytest -q tests/test_module_browser_jobs.py`
  - `conda run -n p313 pytest -q tests/test_module_browser_runtime_refresh_ops.py`

### Фаза 2: Вынос refresh/update handler-ов
- Разделить refresh/update в `handlers_refresh.py` и `handlers_update.py`.
- Сохранить внутренние имена функций там, где тесты делают monkeypatch (или дать алиасы).
- Никаких изменений в threading-семантике (locks, имена потоков, callback-и).

Критерий готовности:
- Проходят:
  - `conda run -n p313 pytest -q tests/test_module_browser_refresh_job_ops.py`
  - `conda run -n p313 pytest -q tests/test_module_browser_update_job_ops.py`

### Фаза 3: Вынос каталога и node introspection
- Перенести snapshot/node mapping в `node_introspection.py`.
- Перенести `/alexz_tools/node_catalog` в `handlers_catalog.py`.
- Не менять кэширование и invalidation (TTL, очистка кэшей).

Критерий готовности:
- Проходят:
  - `conda run -n p313 pytest -q tests/test_module_browser_catalog.py`
  - `conda run -n p313 pytest -q tests/test_module_browser_catalog_payload_ops.py`

### Фаза 4: `routes.py` и compatibility shim
- Добавить `routes.py` с явной функцией `register_routes()`.
- `utils/module_node_browser_api.py` превратить в thin shim.
- Проверить, что import-time side effects минимальны и предсказуемы.

Критерий готовности:
- `conda run -n p313 pytest -q` проходит.
- ComfyUI грузит расширение и виджет работает без изменений.

## План совместимости
- Пути роутов не менять (как в `utils/module_browser/catalog/api_manifest.py`).
- `utils.module_node_browser_api` должен оставаться importable.
- Все ключи JSON оставить прежними.
- Сохранить поведение info-only режима (monitor-only).

## Риски и как их снижать
- Риск: тесты/код завязаны на старые локации символов.
  - Решение: алиасы в shim на 1 релиз, затем чистка.
- Риск: регистрация роутов зависит от import side effects.
  - Решение: явная `register_routes()` вызывается только при наличии PromptServer.
- Риск: поведение глобального состояния слегка "поплывет" после переноса.
  - Решение: объект состояния с явной инициализацией и snapshot-хелперами; smoke-тест на инварианты.

## Опционально после рефакторинга
- Сделать путь persisted state конфигурируемым (env var или `folder_paths` temp/output), вместо записи в корень репо.
- Унифицировать `_resolve_compute_device` в общую утилиту и использовать в Color/Seam/Look match.
- Добавить contract-тесты на схемы API payload (минимальные golden JSON).

## Команды проверки (Conda env `p313`)
- `conda run -n p313 python utils/docs_check.py`
- `conda run -n p313 pytest -q tests/test_module_browser_jobs.py`
- `conda run -n p313 pytest -q tests/test_module_browser_refresh_job_ops.py`
- `conda run -n p313 pytest -q tests/test_module_browser_update_job_ops.py`
- `conda run -n p313 pytest -q`

