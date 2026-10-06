# Known Issues: Module Node Picker

## 1) `Module Nodes -> NodesMap` иногда открывает пустой `NodesMap`
**Статус: исправлено в 0.43.1 (2026-10-06).** В текущей локальной установке
прошли все 72 направленных перехода между 9 sidebar-панелями, включая
Module Nodes, NodesMap, PNG Info и Doctor. Новые конфигурации сторонних extensions
проверяйте отдельно по [browser smoke](GUIDE_BROWSER_SMOKE.md).

До исправления в некоторых конфигурациях ComfyUI при прямом переключении из
`Module Nodes` в `NodesMap` вкладка `NodesMap` могла открыться пустой.

### Влияние
На функциональность нод и самого `Module Nodes` это не влияет.

### Воспроизводимость (baseline)
1. Перейдите в `Module Nodes`.
2. Переключитесь в `NodesMap`.
3. Вернитесь в `Module Nodes`.
4. Повторите цикл `Module Nodes -> NodesMap` еще 1-2 раза.

Если проблема проявилась, в `Module Nodes` diagnostics-блок обычно показывает:
- `diag.reason=relay_tick`,
- `diag.active_tab=alexz-module-nodes`,
- `diag.last_clicked_tab=easyuse_nodes_map` (или `(unknown-other-tab)`),
- `diag.child_nodes_short=ROOT`.

Причина: ComfyUI переиспользует общий mount между custom-панелями. Удаление DOM
соседнего Vue renderer оставляло его state без живого содержимого. PNG Info также
могла накладываться на NodesMap. В 0.43.1 панели получают изолированные mount,
а существующий `destroy` выполняется при уходе с панели; это также восстанавливает
ширину sidebar после Doctor. ComfyUI core и файлы сторонних extensions не изменены.

### Обходной путь для версий до 0.43.1
1. Переключитесь сначала на штатную Vue-панель (например `Apps` или `Workflows`).
2. Затем откройте `NodesMap`.

После такого переключения `NodesMap` обычно отображается корректно.

## 2) Регрессионная проверка
Перед крупными изменениями в tab-sync выполните сценарий из
[GUIDE_BROWSER_SMOKE.md](GUIDE_BROWSER_SMOKE.md) через Playwright MCP.
Для дополнительной ручной проверки:

1. `Module Nodes` открывается и показывает карточку модуля + список нод.
2. `Module Nodes -> NodesMap -> PNG Info -> NodesMap` не оставляет пустых или наложенных панелей.
3. Переключение `Module Nodes -> Workflows -> Module Nodes` работает стабильно.
4. После `Doctor -> Module Nodes` и `Doctor -> Apps` восстанавливается ширина sidebar.

Не нажимайте refresh/install/update/remove или кнопки вставки нод в browser smoke.
