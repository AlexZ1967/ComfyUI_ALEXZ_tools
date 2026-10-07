# Browser smoke через Playwright MCP

Сценарий [alexz_tools_smoke.js](../tests/browser/alexz_tools_smoke.js) проверяет
runtime frontend ALEXZ_tools в настоящем браузере на `http://127.0.0.1:8188`.
Он использует `Page`, предоставляемую Playwright MCP, и не требует новых npm
зависимостей проекта. Это отдельная проверка после серьёзных JS/TS изменений,
а не часть `make test`.

## Запуск одним запросом Codex

```text
Выполни browser smoke ALEXZ_tools по guides/GUIDE_BROWSER_SMOKE.md через
Playwright MCP. Запусти tests/browser/alexz_tools_smoke.js инструментом
browser_run_code_unsafe с параметром filename. Не меняй исходники, не выполняй
install/update/remove. Сообщи status, checks, failures и сторонние ошибки.
```

1. ComfyUI должен быть запущен и доступен по указанному адресу; сценарий сам
   сервер не запускает. После изменения runtime TS заранее выполните
   `make ts-check` и `make frontend-build` в `p313-torch214-cu132`.
2. Проверьте наличие инструментов Playwright MCP. Если они отсутствуют,
   перезапустите расширение Codex и остановитесь до появления инструментов.
3. Создайте отдельную пустую вкладку через `browser_tabs` с `action: "new"`
   без URL. Не используйте вкладку с несохранённым пользовательским workflow:
   сценарий выполняет навигацию и меняет только выбор sidebar/модуля в UI.
4. Вызовите `browser_run_code_unsafe` с аргументом:

   ```json
   {"filename": "tests/browser/alexz_tools_smoke.js"}
   ```

   Путь разрешается от рабочего корня MCP; если он отличается от корня проекта,
   используйте абсолютный путь к этому файлу. Файл содержит async-функцию для
   MCP, а не самостоятельный CLI runner. Название инструмента `unsafe` означает
   возможность исполнения кода в процессе сервера; перед запуском читайте
   сценарий и выполняйте только этот проверенный файл.
5. Успех — только возвращённый `status: "PASS"` со всеми девятью `checks: PASS`.
   Ошибка MCP, timeout, отсутствие подключения или `FAIL` не являются успехом.
   При failure сохраните console/network, snapshot и при необходимости screenshot
   средствами MCP в `.playwright-mcp/`. Не исправляйте сторонние extensions
   автоматически. После проверки закройте только созданную тестовую вкладку.

`make browser-smoke` намеренно не добавлен: Makefile не имеет доступа к живому
MCP-сеансу Codex. HTTP или синтаксическая проверка вместо браузера не считается
browser smoke. При необходимости будущего автономного CI runner его следует
добавлять отдельной задачей.

## Проверки и критерии failure

| Проверка | Условие успеха |
|---|---|
| ComfyUI loaded | HTTP 2xx, `document.readyState === "complete"`, готовы Vue app, graph и canvas |
| ALEXZ.Tools.Hello setup | Extension зарегистрирован и получен setup log именно из `generated/hello.js` |
| Module Node Picker opens | Кнопка sidebar видима, после клика панель `.alexz-mod-picker` видима |
| Picker DOM and node list | Есть заголовок, Debug, mode select, три selection select, фильтр, карточка ALEXZ_tools и список нод |
| Update planner controls | Есть единая Refresh Custom Nodes Info и кнопка Подробнее; подробности изначально скрыты, проверка и обновление не запускаются |
| Sidebar tab switching | Два цикла переходов между Module Nodes и Apps; при установленном EasyUse дополнительно NodesMap → Module Nodes → NodesMap → Apps → NodesMap, а при наличии PNG Info — NodesMap → PNG Info → NodesMap. При наличии Doctor проверяются Doctor → Module Nodes и Doctor → Apps с восстановлением inline layout. Активная панель видима, чужие панели скрыты |
| Catalog and module-info API | GET через `api.fetchApi`, HTTP 2xx, JSON без `error`, каталог содержит ноды ALEXZ_tools, info соответствует модулю |
| Extension resources loaded | Все JS URL ALEXZ_tools из `/extensions` наблюдались при загрузке без HTTP errors; Hello и picker присутствуют |
| No ALEXZ errors or unsafe operations | Нет связанных console errors/pageerrors, failed requests или заблокированных опасных запросов |

Сбор console и network начинается **до навигации**, поэтому учитывает ошибки
импорта и регистрации. Console errors проверяются по сообщению, location и
сериализованным аргументам (включая вложенные Error/stack); `pageerror` — по stack.
Связь определяется по `ALEXZ_tools`, namespace `ALEXZ.Tools`/`ALEXZ.ModulePicker`
или `module_node_picker`. Ошибки `web/generated/` определяются по URL/stack
пакета ALEXZ_tools; один общий сегмент `web/generated/` стороннего пакета
не подтверждает такую связь. Network failures и HTTP >= 400
для `/extensions/ComfyUI_ALEXZ_tools/` всегда означают FAIL.

Ошибки других extensions и их 404, например отсутствие пользовательского CSS,
учитываются в `backgroundErrorCount` и первых пяти `backgroundErrors`, но не
проваливают сценарий. Warnings подсчитываются отдельно. HTTP 200 сам по себе
не доказывает выполнение JS — поэтому проверяются Hello setup и работа picker.

## Безопасность и ограничения

- Выполняются только навигация, открытие панели, выбор Custom/ALEXZ_tools и
  раскрытие карточки модуля и переключение sidebar Module Nodes/Apps/NodesMap/PNG Info/Doctor.
  Кнопки refresh, install, update, remove и нод
  не нажимаются; workflow не запускается и не сохраняется.
- Guard перехватывает `/alexz_tools/` и `/api/alexz_tools/`: разрешает только
  GET каталога с `cache_only=1`, GET module info с `cache_only=1` без refresh/sync,
  а также чтение статусов jobs. Остальные запросы блокируются до backend и
  приводят к FAIL, включая попытки возобновить прежние операции.
- Эти GET могут инициировать штатный backend warmup/чтение runtime cache.
  Guard не отменяет jobs, уже запущенные до теста, и не контролирует фоновую
  работу других extensions. Не запускайте smoke одновременно с обновлением.
- Общий бюджет ожиданий UI — 60 секунд, API — 10 секунд; в конце есть окно
  1 секунду для отложенных ошибок. Это smoke, а не полный E2E: вставка нод,
  вычисления, upload и ошибки, появляющиеся значительно позже, не проверяются.
- Проверка зависит от текущих DOM-классов picker и `data-testid` sidebar;
  при намеренном изменении этих контрактов обновите сценарий.
- EasyUse/NodesMap и PNG Info — необязательные сторонние расширения. Проверка сообщает
  их наличие и число циклов в `sidebarSwitching`; при отсутствии NodesMap
  выполняются только переходы Module Nodes ↔ Apps. Отсутствие EasyUse не FAIL,
  но пустая NodesMap после перехода из Module Nodes при наличии расширения — FAIL.
  Одновременное отображение NodesMap и PNG Info при переключении между ними — FAIL.
- Doctor также необязателен: сценарий сообщает `doctorInstalled` и `doctorCycles`.
  Открывается только sidebar, без анализа, отправки сообщений AI или изменения настроек.
  После ухода из Doctor проверяется восстановление `min-width`, `width`, `flex-basis`
  у sidebar content и панели. Оставшиеся принудительные размеры означают FAIL.
- При изменениях общего lifecycle sidebar дополнительно проверьте все направленные
  переходы между установленными панелями в отдельной тестовой вкладке. Для каждой
  пары откройте первую панель, затем вторую; проверьте содержимое, отсутствие чужого
  DOM поверх него и восстановление layout. Переключайтесь только кнопками toolbar,
  не нажимайте элементы вставки нод, загрузки workflow или операций backend.
- `.playwright-mcp/` содержит локальные диагностику и screenshots, игнорируется
  Git и не коммитится. Настройки MCP также не добавляются в исходники проекта.
