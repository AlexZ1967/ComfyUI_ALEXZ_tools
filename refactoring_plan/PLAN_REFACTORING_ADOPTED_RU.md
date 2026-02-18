# ComfyUI_ALEXZ_tools — Принятый план рефакторинга

Статус: утвержденный рабочий план поэтапной реализации  
Дата: 2026-02-10  
Источник: `PROPOSAL_REFACTORING.md`, `PROPOSAL_VUE_INTEGRATION_FIX.md`, `PROPOSAL_PHASE_1_IMPLEMENTATION.md` (+ RU версии)

## 1) Цель

Стабилизировать поведение вкладки `Module Node Picker` и снизить риски сопровождения без нарушения совместимости с ComfyUI.

Ключевые ограничения:
- Без breaking changes для текущих роутов и поведения виджета.
- Пошаговая миграция с точками отката после каждой фазы.
- Сохранение стабильного production-поведения при улучшении внутренней архитектуры.

## 2) Что принимаем / откладываем / отклоняем

### Принимаем сейчас (высокая ценность, низкий/средний риск)

1. Постепенно разделяем frontend-ответственности:
- состояние;
- диагностика;
- API-обертки;
- рендер-логика.

2. Вводим централизованное хранилище состояния (`web/state/store.js`) для состояния picker.

3. Вводим отдельный логгер диагностики (`web/diagnostics/logger.js`) и убираем разрозненную debug-логику.

4. Упрощаем синхронизацию вкладок до одного доминирующего механизма и убираем конкурирующие контроллеры видимости.

5. Начинаем декомпозицию `utils/module_node_browser_api.py` на внутренние модули с сохранением текущих HTTP-роутов.

### Откладываем (после стабилизации)

1. Глубокую async-перепись всех backend jobs.
2. Унификацию всех нод через единый base class.
3. Отдельный CLI-пакет для standalone-режима.
4. Полный E2E-стек до завершения базовой стабилизации.

### Отклоняем в текущем виде (требует переработки)

1. Vue-эвристики на конкретных CSS-классах (`vue-entering`, `vue-temp-wrapper`) как основной механизм.
2. Примеры, невалидные для текущего runtime JS (например, TS-синтаксис внутри `.js`).
3. Массовый `asyncio.gather` для git-операций без ограничения конкуренции.

## 3) Риски в исходных предложениях (и как их снижаем)

1. Смешение JS/TS в примерах  
- Риск: runtime-ошибки.  
- Снижение: писать реализацию на plain JS, совместимом с текущим web-runtime ComfyUI.

2. Слишком агрессивная привязка к Vue  
- Риск: хрупкость между разными сборками ComfyUI.  
- Снижение: использовать правила container-ownership и детерминированное attach/detach, а не эвристики классов Vue.

3. Big-bang рефакторинг  
- Риск: регрессии в picker, update и status UI.  
- Снижение: фазовый rollout с тестами после каждой фазы.

4. Агрессивный async fan-out на backend  
- Риск: перегрузка IO и race conditions.  
- Снижение: ограниченные worker-очереди и поэтапная миграция.

## 4) Этапы реализации

## Phase 0 — Базовая фиксация и guardrails

Статус: ✅ выполнено (2026-02-10, commit `50a93f3`)

Результат:
- Фиксируем текущее поведение регрессионными проверками для:
  - загрузки каталога;
  - загрузки информации о модуле;
  - polling статуса обновлений;
  - переходов между вкладками.
- Добавляем/обновляем smoke-проверки критичных API-полей.

Критерии выхода:
- Базовые тесты проходят.
- Текущая known issue воспроизводима и задокументирована.

## Phase 1 — База frontend (Store + Diagnostics)

Статус: ✅ выполнено (2026-02-11)

Результат:
- Добавляем `web/state/store.js`:
  - единый source of truth для UI-состояния picker;
  - API подписки/отписки;
  - сохраняем только минимальные ключи (`selectedGroup`, `selectedModule`, debug flag).
- Добавляем `web/diagnostics/logger.js`:
  - уровни логирования (`info`, `warn`, `error`);
  - ограниченный буфер логов в памяти;
  - opt-in debug режим.
- Интегрируем эти модули в `web/module_node_picker.js` без изменения видимого поведения.
- Вынесены крупные части picker в отдельные модули при сохранении поведения:
  - UI: `web/ui/module_node_picker_renderers.js`, `web/ui/module_node_picker_alerts.js`,
    `web/ui/module_node_picker_process.js`, `web/ui/module_node_picker_catalog.js`
  - orchestration: `web/orchestration/flow/progress/module_node_picker_update_flow.js`,
    `web/orchestration/flow/catalog/module_node_picker_data_flow.js`,
    `web/orchestration/flow/actions/module_node_picker_actions.js`,
    `web/orchestration/core/infra/module_node_picker_bindings.js`

Критерии выхода:
- Нет UX-регрессий.
- Диагностика включается/выключается в рантайме.
- Текущие сценарии picker продолжают работать.

## Phase 2 — Стабилизация tab-sync

Статус: ✅ выполнено (2026-02-12)

Результат:
- Оставляем один основной механизм синхронизации вкладок.
- Убираем или жестко отключаем конкурирующие sync-пути, вызывающие конфликты ownership.
- Обеспечиваем детерминированный attach/detach root picker при смене вкладки.
- Внешний баг `NodesMap` (сторонний виджет) зафиксирован как known issue и вынесен за рамки этого этапа.
- Промежуточный прогресс (2026-02-11):
  - удалены legacy/конкурирующие sync-пути, оставлен один relay-путь,
  - внутренности relay вынесены в отдельные модули:
    - `web/orchestration/module_node_picker_tab_relay_helpers.js`
    - `web/orchestration/module_node_picker_tab_relay_runtime.js`
  - удалена широкая fallback-логика unknown-tab detach при кликах по контенту,
  - уменьшен шум событий relay (единый pointer/mouse путь, без per-button listeners),
  - добавлены debounce синхронизации и явный dispose cleanup при unbind,
  - снижено давление пассивного `relay_tick`, когда вкладка picker не активна,
  - распознавание tab-click в relay ограничено только sidebar-контекстом и tab-controls (меньше ложных срабатываний от кликов по контенту),
  - `relay_keyup` ограничен навигационными/tab-клавишами и исключает target'ы ввода текста,
  - добавлены bind-token guards: устаревшие таймеры/листенеры от предыдущего bind больше не могут менять видимость текущего relay,
  - удален неиспользуемый fallback-helper определения tab-id из relay helpers (снижение лишней эвристики),
  - фиксированный interval relay заменен на адаптивный timeout-loop (ниже фоновое давление синхронизации),
  - матчинг tab-кандидатов в relay сужен до явных tab-like sidebar controls (меньше ложных захватов обычных кнопок).
  - добавлен request-token guard загрузки каталога, чтобы устаревшие async-ответы не перезаписывали актуальный UI-стейт.
  - добавлен render-lifecycle cleanup hook picker и request-token guard для `loadModuleInfo` (устаревшие async-ответы не перезаписывают карточку модуля между рендерами).
  - очищены relay helpers: удален неиспользуемый helper и вынесены selector-константы для tab/sidebar матчей.
  - добавлены early-exit guards для загрузки каталога/инфо модуля на disposed-instance и debounce смены `ComfyUI check` режима.
  - добавлен явный cleanup жизненного цикла UI-событий (unbind callbacks + cleanup таймеров + unsubscribe store при dispose picker).
  - liveness-aware cancellation guards протянуты через refresh/update/install orchestration (исключены stale UI-мутации после dispose).
  - добавлен явный lifecycle dispose для process-controller (прогресс-хост гарантированно очищается/отцепляется между re-render).
  - устранено дублирование lifecycle-guard логики: общий helper вынесен в отдельный orchestration-модуль.
  - исправлена семантика startup-liveness (жизненный цикл вместо `root.isConnected`), устранен баг пустых селектов при первом открытии.
  - добавлен bounded startup-retry загрузки каталога с явной отменой таймера при dispose (для transient пустого backend-состояния при старте).
  - добавлены loading placeholders + disable/enable lifecycle для селектов во время загрузки каталога (без визуально пустых dropdown на старте).
  - добавлены frontend API timeout-границы (AbortController), чтобы зависшие запросы не блокировали UX виджета бесконечно.
  - все API-запросы picker привязаны к lifecycle per-render через `AbortController`; in-flight запросы теперь явно отменяются при dispose/re-render.
  - смена `ComfyUI check` режима отвязана от полной reload каталога (стабильные dropdown без фликера при быстрых переключениях).
  - финализирована семантика `ComfyUI check`: переключение — только сохранение настройки; сетевое обновление — только по явному `Refresh ComfyUI Info`.
  - вынесена orchestration-логика category/group/module селекторов в отдельный модуль `web/orchestration/ui/module_node_picker_selection_controller.js` (пополнение/фильтрация dropdown и sync со store), с сохранением прежнего UI-поведения.
  - вынесена orchestration-логика long-running действий (refresh/update/resume/requirements follow-up + per-module refresh/install) в `web/orchestration/flow/actions/module_node_picker_action_flows.js`, а в `web/module_node_picker.js` оставлены только thin bindings и wiring зависимостей.
  - вынесен token-based polling lifecycle (refresh/update progress loops) в `web/orchestration/module_node_picker_polling_controller.js`; dispose picker теперь инвалидирует poll-контроллер через единый API.
  - вынесен рендер и UI-state панели модуля (module-card + node-list, включая expand/collapse состояние) в `web/orchestration/module_node_picker_module_panel_controller.js`.
  - вынесен lifecycle/dispose controller picker instance в `web/orchestration/module_node_picker_lifecycle.js` (единая очистка токенов, polling, bind/unbind, startup cancel, debug/process/API cleanup).
  - вынесена логика регистрации extension и fallback-монтажа в `web/orchestration/core/infra/module_node_picker_registration.js`.
  - централизованы константы picker (ID, storage-keys, group-labels, marks, defaults) в `web/constants/module_node_picker_constants.js`.
  - вынесены helper-функции создания/позиционирования LiteGraph-нод в `web/ui/module_node_picker_node_factory.js`.
  - полная композиция picker (`renderPicker`) перенесена из `web/module_node_picker.js` в `web/orchestration/core/composition/module_node_picker_composer.js`; основной entrypoint теперь отвечает только за регистрацию extension/fallback wiring.
  - вынесен wiring селекторов/busy/view/status-карточек в `web/orchestration/ui/module_node_picker_ui_controllers.js` для дальнейшего уменьшения плотности композиционного кода.
  - вынесен orchestration-бандл polling/catalog/actions/module-panel в `web/orchestration/module_node_picker_flow_wiring.js` для модульной сборки runtime-пайплайна.
  - вынесен runtime-bootstrap (bind events, восстановление ComfyUI-card, wiring startup coordinator) в `web/orchestration/module_node_picker_runtime_bootstrap.js`.
  - вынесен базовый runtime-setup (runtime context, lifecycle, API client, debug/process controllers) в `web/orchestration/module_node_picker_runtime_setup.js`.
  - вынесена UI-stage сборка в `web/orchestration/ui/module_node_picker_ui_stage.js` (adapter-композиция selector/busy/view/status контроллеров).
  - вынесена flow-stage сборка в `web/orchestration/module_node_picker_flow_stage.js` (adapter-композиция polling/catalog/action/module-panel контроллеров).
  - крупные dependency-map объекты composer вынесены в `web/orchestration/core/composition/module_node_picker_context_builders.js` (контекст-билдеры для runtime-setup/ui-stage/flow-stage/runtime-bootstrap), что уменьшило размер `module_node_picker_composer.js` без изменения поведения.
  - внутренности pending-resume логики разнесены по отдельным модулям (`module_node_picker_resume_custom_refresh.js`, `module_node_picker_resume_module_update.js`, `module_node_picker_resume_comfy_refresh.js`) при сохранении стабильного фасада экспортов в `module_node_picker_resume_flow.js`.
  - orchestration polling/runtime warmup вынесена в `web/orchestration/module_node_picker_warmup_controller.js`; авто-подхват маркеров после первого открытия сохраняется в фоновом режиме.
  - адаптивный relay tick-loop вынесен в `web/orchestration/module_node_picker_tab_relay_tick.js`; bind/runtime слой релея упрощен без изменения поведения.
  - relay tab-intent/event orchestration вынесена в `web/orchestration/module_node_picker_tab_relay_intent.js`; wiring listeners в `module_node_picker_tab_relay.js` упрощен при сохранении поведения.
  - CSS Module Node Picker вынесен в `web/orchestration/styles/module_node_picker_styles.js` с секциями и подробными комментариями, чтобы отделить правки оформления от orchestration-логики.
  - CSS Module Node Picker перемещен в UI-слой (`web/ui/styles/module_node_picker_styles.js`) для корректного разделения ответственности по директориям.
  - wiring deferred-stage из composer вынесен в `web/orchestration/core/composition/module_node_picker_stage_bridge.js`, чтобы централизовать handoff flow-stage и adapter callbacks без изменения поведения.
  - runtime-bootstrap callback bindings вынесены из composer в `web/orchestration/module_node_picker_runtime_bootstrap_bindings.js`, чтобы снизить плотность inline-callback кода в композиции.
  - проекция/распаковка runtime-setup вынесена в `web/orchestration/module_node_picker_runtime_projection.js`, что уменьшило шум flat-mapping полей в composer.
  - устранено зависание индикатора warmup: warmup-poller привязан к reload каталога, добавлены fail-safe ветки сброса индикатора при исчерпании retry-бюджета и ошибках poll.
  - введены семантические подпапки orchestration:
    - `web/orchestration/relay/` для tab-relay внутренних модулей,
    - `web/orchestration/runtime/` для runtime/bootstrap/lifecycle/warmup модулей,
    с обновлением импортов и тестовых путей.
  - введена подпапка `web/orchestration/flow/` для pipeline/data/update/resume orchestration-модулей; связанные файлы вынесены из плоского корня с обновлением импортов и test-маркеров.
  - добавлены дополнительные семантические подпапки orchestration:
    - `web/orchestration/core/` для composition/bootstrap-адаптеров и общих orchestration helper-ов,
    - `web/orchestration/ui/` для UI-контроллеров/представлений/status-card orchestration,
    - `web/orchestration/api/` для lifecycle-bound API client обертки.
  - оставшиеся плоские `module_node_picker_*` orchestration-модули перенесены в группы `core/ui/api`, обновлены зависимые импорты, шапки `Module:` и пути в baseline-контрактных frontend-тестах.
  - исправлены runtime-пути импортов после переноса в `module_node_picker_runtime_setup.js` (импорты `process` UI и runtime-context), чтобы сохранить рабочее runtime-поведение.
  - flow-orchestration дополнительно разделен по ответственности:
    - `web/orchestration/flow/progress/` для refresh/update polling и progress loops,
    - `web/orchestration/flow/resume/` для pending-refresh/update/comfy resume flows.
  - `module_node_picker_update_flow.js` и `module_node_picker_polling_controller.js` перенесены в `flow/progress/`, resume-модули перенесены в `flow/resume/`, затем обновлены зависимые импорты, шапки `Module:` и пути в baseline frontend-тестах.
  - runtime-orchestration дополнительно разделен по ответственности:
    - `web/orchestration/runtime/bootstrap/` для runtime setup/bootstrap/startup/warmup модулей,
    - `web/orchestration/runtime/lifecycle/` для lifecycle guard и dispose-логики picker instance.
  - runtime-модули перенесены в группы `bootstrap/lifecycle`, обновлены зависимые импорты в composer/flow/runtime модулях, шапки `Module:` и пути в baseline frontend-тестах.
  - оставшийся flow-раздел дополнительно разделен по ответственности:
    - `web/orchestration/flow/actions/` для action-handlers и composed action flows,
    - `web/orchestration/flow/catalog/` для catalog/module-info data controllers и loaders.
  - `module_node_picker_actions.js` + `module_node_picker_action_flows.js` перенесены в `flow/actions/`, `module_node_picker_catalog_controller.js` + `module_node_picker_data_flow.js` перенесены в `flow/catalog/`, затем обновлены зависимые импорты, шапки `Module:` и пути в baseline frontend-тестах.
  - оставшиеся плоские flow-файлы разделены по назначению:
    - `web/orchestration/flow/stage/` для stage-адаптеров (`flow_stage`, `flow_wiring`),
    - `web/orchestration/flow/panel/` для контроллера рендера панели модуля.
  - `module_node_picker_flow_stage.js` + `module_node_picker_flow_wiring.js` перенесены в `flow/stage/`, `module_node_picker_module_panel_controller.js` перенесен в `flow/panel/`, затем обновлены зависимые импорты, шапки `Module:` и пути в baseline frontend-тестах.
  - core-слой orchestration дополнительно разделен по ответственности:
    - `web/orchestration/core/composition/` для composition/stage-bridge/context сборки,
    - `web/orchestration/core/infra/` для инфраструктурных модулей bindings/error/registration.
  - `module_node_picker_composer.js` + `module_node_picker_context_builders.js` + `module_node_picker_stage_bridge.js` перенесены в `core/composition/`, `module_node_picker_bindings.js` + `module_node_picker_error_utils.js` + `module_node_picker_registration.js` перенесены в `core/infra/`, затем обновлены зависимые импорты, шапки `Module:`, ссылки в плане и пути в baseline frontend-тестах.
  - убран хрупкий глубокий импорт `scripts/app.js` из composer; теперь `app` передается из entrypoint через dependency injection (`renderModuleNodePicker(container, { appInstance: app })`), что снижает риск поломки инициализации после переносов файлов.
  - реализация relay bind/unbind вынесена в `web/orchestration/relay/module_node_picker_tab_relay_facade.js`, а `web/module_node_picker_tab_relay.js` преобразован в стабильный re-export entrypoint с сохранением текущего import path и уменьшением плотности root-слоя orchestration.
  - доступ к глобальному relay-состоянию централизован в `web/orchestration/relay/module_node_picker_tab_relay_state.js`; relay facade переведен на shared state helper для операций read/write/clear.
  - reason/timing константы relay централизованы в `web/orchestration/relay/module_node_picker_tab_relay_constants.js`, а `runtime/intent/tick/facade` переведены на constants-driven flow.
  - логика bypass debounce для immediate-reason в relay унифицирована через общий helper `isImmediateRelayReason()` из constants-модуля.
  - уменьшена нагрузка relay-диагностики: события relay прокидываются в debug-панель только при включенном debug-режиме.
  - в relay facade добавлены browser-context guards для bind/unbind, чтобы исключить падения в частичных/non-browser сценариях инициализации.
  - добавлен явный `mountHost`-wiring от composer до relay runtime, чтобы восстановление attach/detach могло переаттачить root к актуальному sidebar-host при смене исходного parent-контейнера.
  - нормализован relay bind input (guard на `root` как `Element` в facade) и улучшен host-preference в runtime: подключенный root при дрейфе owner-контейнера принудительно возвращается под preferred mount host.
  - DOM-детекторы relay (tab-candidate/sidebar-context) вынесены в `web/orchestration/relay/module_node_picker_tab_relay_dom.js`, а relay helpers переведены на shared DOM helper-функции.
  - lifecycle relay bind-state (создание state + детерминированный dispose/unbind) вынесен в `web/orchestration/relay/module_node_picker_tab_relay_lifecycle.js`, а facade переведен на shared lifecycle helpers.
  - формирование payload relay-диагностики и dedup-emitter вынесены в `web/orchestration/relay/module_node_picker_tab_relay_diagnostics.js`, runtime переведен на shared diagnostics helpers.
  - DOM ownership relay (attach/detach + восстановление host) вынесен в `web/orchestration/relay/module_node_picker_tab_relay_dom_ownership.js`, runtime visibility переведен на shared ownership controller.
  - relay-bind wiring в composer вынесен в `web/orchestration/core/composition/module_node_picker_relay_bridge.js`, чтобы body composition оставался компактнее и relay-инициализация была изолирована в bridge-модуле.

Критерии выхода:
- Многократные переходы для штатных вкладок стабильны (`Module Nodes -> Workflows/PNG Info -> Module Nodes`).
- Нет дублирующихся root и пустой панели в самом `Module Node Picker`.
- Известная внешняя проблема прямого перехода в `NodesMap` документирована с workaround в `guides/GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md`.

## Срез 0 — Основа расширяемости (nodes/widgets lifecycle)

Статус: ✅ выполнено (2026-02-12)

Результат:
- Вводим registry-first слой для компонентов модуля (`nodes`, `widgets`, `api`) с автообнаружением.
- Выносим стабильные backend-контракты (schema/versioned payload) и версионируемый кэш состояния.
- Фиксируем единые точки подключения/отключения компонентов, чтобы добавление/удаление нод и виджетов не требовало правок в нескольких местах.
- Добавлен health-report манифестов + сигнатура (`manifest_signature`) для быстрого контроля рассинхрона при изменениях компонентов.
- Добавляем минимальные контрактные тесты на сценарии:
  - добавлен новый компонент;
  - удален компонент;
  - API-ключи payload не изменились.

Критерии выхода:
- Добавление/удаление ноды или виджета выполняется через registry без ручного обхода кода.
- API/UX поведение не меняется.
- Базовые тесты проходят.

## Phase 3 — Модульная декомпозиция backend (без API-изменений)

Статус: ✅ выполнено (2026-02-13)

Результат:
- Внутренне делим `utils/module_node_browser_api.py` на блоки:
  - сбор/построение каталога;
  - git state/sync helper-ы;
  - сборка module info;
  - orchestration refresh/update jobs.
- Сохраняем совместимость сигнатур роутов и ключей payload.
- Выполнен шаг 1: вынесены job-helpers (`refresh/update` status + target resolution)
  в `utils/module_browser/jobs.py`, основной backend оставлен как совместимый facade.
- Выполнен шаг 2: вынесена catalog-сборка (`collect/build/filter`) в
  `utils/module_browser/catalog.py` c сохранением совместимых wrapper-функций в API-файле.
- Выполнен шаг 3: вынесена text-часть module-info (README summary + description sanitize)
  в `utils/module_browser/module_info_text.py`.
- Выполнен шаг 4: вынесена сборка payload module-info и cached-флаги бейджей модулей
  в `utils/module_browser/module_info.py`, при этом совместимые facade-wrapper-ы
  в `utils/module_node_browser_api.py` сохранены.
- Выполнен шаг 5: вынесен слой git state/sync helper-ов (выбор/разрешение remote,
  release-tag ref resolve, git-state/sync custom модулей, worktree signature) в
  `utils/module_browser/git_helpers.py` с сохранением facade-wrapper-ов в API-файле.
- Выполнен шаг 6: операции diff/install для requirements вынесены в
  `utils/module_browser/update_ops.py` (`requirements_changed_between`,
  install requirements для module/comfyui) с сохранением facade-wrapper-ов.
- Выполнен шаг 7: helper-ы pull/update вынесены в
  `utils/module_browser/pull_ops.py` (`is_git_local_changes_block`,
  `pull_comfyui`, `pull_custom_module`) с сохранением facade-wrapper-ов.
- Выполнен шаг 8: агрегатор batch-установки requirements вынесен в
  `utils/module_browser/update_ops.py` (`install_requirements_for_modules`)
  с сохранением facade-wrapper-а в API-файле.
- Выполнен шаг 9: операции чтения/записи state-cache вынесены в
  `utils/module_browser/state_store.py` (`load_state_file`,
  `save_state_file`) с сохранением API-cache facade.
- Выполнен шаг 10: tracker/novelty операции вынесены в
  `utils/module_browser/tracker_ops.py` (`remember/apply/acknowledge`,
  `announce_tracked_module_updates`) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 11: helper-ы startup-tracking для ComfyUI вынесены в
  `utils/module_browser/comfyui_tracking_ops.py`
  (`track_comfyui_local_update`, `acknowledge_comfyui_novelty`)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 12: helper-ы node snapshot/path вынесены в
  `utils/module_browser/node_snapshot_ops.py`
  (`node_source_file`, `relative_to_custom_roots`, `file_digest`,
  `build_node_snapshots`) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 13: orchestration runtime-refresh фаз вынесен в
  `utils/module_browser/runtime_refresh_ops.py`
  (`refresh_module_runtime_state`) с сохранением facade-wrapper-а в API.
- Выполнен шаг 14: логика исполнения module-update job вынесена в
  `utils/module_browser/update_job_ops.py`
  (`run_module_update_job`) при сохранении API worker-thread wrapper-а.
- Выполнен шаг 15: логика исполнения refresh job вынесена в
  `utils/module_browser/refresh_job_ops.py`
  (`run_refresh_job`) при сохранении API worker-thread wrapper-а.
- Выполнен шаг 16: helper-ы идентификации custom-модулей вынесены в
  `utils/module_browser/module_identity.py`
  (`discover/normalize/alias/canonical`) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 17: helper-ы ComfyUI status/state merge вынесены в
  `utils/module_browser/comfyui_state_ops.py`
  (template/cache resolve/pending merge/state persist) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 18: orchestration ComfyUI git-status вынесен в
  `utils/module_browser/comfyui_git_status_ops.py`
  (`collect_comfyui_git_status`) с сохранением facade-wrapper-а в API.
- Выполнен шаг 19: orchestration payload component-registry вынесен в
  `utils/module_browser/component_registry_payload_ops.py`
  (`collect_component_registry_payload`) с сохранением facade-wrapper-а в API.
- Выполнен шаг 20: helper-ы metadata/statistics ComfyUI-Manager вынесены в
  `utils/module_browser/manager_data_ops.py`
  (`load_manager_index`, `load_manager_github_stats`, `resolve/infer` helper-ы)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 21: helper-ы исполнения subprocess/git-команд вынесены в
  `utils/module_browser/command_ops.py`
  (`run_command`, `run_git`, `extract_git_repo_from_args`,
  `is_git_dubious_ownership_error`, `try_mark_git_safe_directory`, `tail_lines`)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 22: builder-ы payload для catalog-роутов вынесены в
  `utils/module_browser/catalog_payload_ops.py`
  (`build_group_payload`, `build_module_list_payload`,
  `build_module_nodes_payload`) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 23: helper-ы widget-mode/log-mode вынесены в
  `utils/module_browser/widget_mode_ops.py`
  (`custom_update_checked_flag`, `info_only_rejection_payload`,
  `set_custom_update_checked`, `normalize_log_mode`) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 24: чистые helper-ы значений/дат/репозиториев вынесены в
  `utils/module_browser/value_ops.py`
  (`short_commit`, `normalize_repo_url`, `github_id`, `repo_name`,
  `pick_repo_url`, `parse_datetime`, `to_iso`, `now_iso`,
  `normalize_comfyui_mode`) с сохранением facade-wrapper-ов в API.
- Выполнен шаг 25: helper-ы мутаций pending-state для requirements вынесены в
  `utils/module_browser/requirements_pending_ops.py`
  (`set_comfyui_requirements_pending`, `set_module_requirements_pending`)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 26: helper-ы резолвинга путей вынесены в
  `utils/module_browser/path_ops.py`
  (`custom_nodes_roots`, `manager_custom_db_path`,
  `manager_github_stats_path`, `module_dir`, `comfyui_root`)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 27: сетевой helper latest-release GitHub вынесен в
  `utils/module_browser/release_ops.py`
  (`github_latest_release`) с сохранением facade-wrapper-а в API.
- Выполнен шаг 28: helper-ы решений update-state вынесены в
  `utils/module_browser/module_update_state_ops.py`
  (`module_needs_update_now`, `count_custom_modules_need_update`,
  `count_custom_modules_unknown_update`, `comfyui_needs_update_now`)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 29: helper-ы bootstrap репозитория вынесены в
  `utils/module_browser/repo_bootstrap_ops.py`
  (`comfyui_requirements_path`, `bootstrap_module_remote_from_manager`)
  с сохранением facade-wrapper-ов в API.
- Выполнен шаг 30: helper-ы классификации/аннотации нод вынесены в
  `utils/module_browser/node_classification_ops.py`
  (`module_root`, `classify_by_source_path`, `classify_by_relative_module`,
  `fallback_annotation`) с сохранением facade-wrapper-ов в API.

Критерии выхода:
- Текущий frontend работает без API-изменений.
- Внутренние модули проще тестировать отдельно.

## Phase 4 — Усиление качества и покрытия

### Срез 1 — Структурная реорганизация backend модулей (2026-02-18)

Статус: ✅ выполнено (2026-02-18)

Результат:
- Реорганизован монолитный `utils/module_browser/` (36 файлов в одной папке) на 9 функциональных подпапок:
  - `catalog/` — сборка каталога, компонент-реестр, payload-билдеры;
  - `git/` — git-операции, pull-helpers, subprocess/git команды;
  - `module/` — метаданные модулей, идентичность, классификация;
  - `comfyui/` — ComfyUI-специфичные tracking/state/manager-data;
  - `state/` — storage, pending-state mutations, runtime-refresh;
  - `jobs/` — job execution, status handling, requirements;
  - `tracking/` — module novelty tracking и change detection;
  - `bootstrap/` — repository bootstrap helpers;
  - `core/` — core utilities (paths, values, manifest-checks, widget-mode).
- Создано по `__init__.py` в каждой подпапке с переэкспортом;
- Сохранена полная обратная совместимость публичного API;
- Обновлены внутренние импорты и пути;
- Тесты: все Phase 3 baseline тесты проходят без регрессий.

Рациональ:
- улучшена навигация для contributors (ясная семантика папок);
- проще организовывать тесты по папкам;
- снижена когнитивная нагрузка при сопровождении;
- нет breaking changes API.

Критерии выхода:
- API полностью обратно совместимо.
- Структурные папки логически организованы.
- Импорты работают как в коде, так и в тестах.

### Фаза 4 — остальное (ниже)

Результат:
- Добавляем integration-тесты критичных роутов.
- Добавляем frontend-проверки для tab transitions и progress state.
- Добавляем облегченный CI workflow для Python-тестов и docs check.

Критерии выхода:
- Покрытие заметно улучшено на затронутых модулях.
- Нет блокирующих регрессий в рабочих сценариях picker.

## 5) Инженерные правила для рефакторинга

1. Публичное поведение сохраняем, если нет явного согласования изменений.
2. Один вектор миграции на PR (state, sync, backend split, tests).
3. Не добавляем новые глобальные `window[...]` состояния (кроме временных и документированных).
4. Рискованные изменения — за feature-flag.
5. Откат каждой фазы должен быть независимым и простым.

## 6) Стратегия отката

По фазам:
- Phase 1: переключение назад на legacy local state paths.
- Phase 2: возврат к предыдущему tab relay при регрессии.
- Phase 3: route handlers остаются стабильным facade, возможен fallback на старую внутреннюю реализацию.

Практика репозитория:
- Небольшие коммиты с четкой областью.
- Не смешивать в одном коммите рефакторинг и новый функционал.

## 7) Метрики успеха

Функциональные:
- Стабильные повторные переключения вкладок без пустой панели после 2–3 циклов.
- Без регрессий в refresh/update сценариях.

Сопровождаемость:
- Существенное уменьшение `web/module_node_picker.js` за счет выноса инфраструктурной логики.
- `utils/module_node_browser_api.py` становится orchestration-facade.

Качество:
- Рост автоматических проверок frontend-поведения (минимум smoke) и backend payload-контрактов.

## 8) Следующий шаг

Запустить реализацию **Среза 0**, затем перейти к Phase 3 с сохранением совместимости API и поведения UI.
