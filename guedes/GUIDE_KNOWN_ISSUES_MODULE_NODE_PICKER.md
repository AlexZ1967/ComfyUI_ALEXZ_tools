# Known Issues: Module Node Picker

## 1) `Module Nodes -> NodesMap` иногда открывает пустой `NodesMap`
В некоторых конфигурациях ComfyUI при прямом переключении из `Module Nodes` в `NodesMap` вкладка `NodesMap` может открыться пустой.

### Влияние
На функциональность нод и самого `Module Nodes` это не влияет.

### Воспроизводимость (baseline)
1. Перейдите в `Module Nodes`.
2. Переключитесь в `NodesMap`.
3. Вернитесь в `Module Nodes`.
4. Повторите цикл `Module Nodes -> NodesMap` еще 1-2 раза.

Если проблема проявилась, в `Module Nodes` diagnostics-блок обычно показывает:
- `diag.reason=relay_tick` (или `relay_unknown_tab_click`),
- `diag.active_tab=alexz-module-nodes`,
- `diag.last_clicked_tab=easyuse_nodes_map` (или `(unknown-other-tab)`),
- `diag.child_nodes_short=ROOT`.

Это означает, что активная вкладка в sidebar и фактический контейнер стороннего виджета разошлись.

### Обходной путь
1. Переключитесь сначала на любой другой виджет (например `Workflows` или `PNG Info`).
2. Затем откройте `NodesMap`.

После такого переключения `NodesMap` обычно отображается корректно.

## 2) Phase 0 baseline checklist (ручной smoke)
Перед крупными изменениями в tab-sync рекомендуется проверить:
1. `Module Nodes` открывается и показывает карточку модуля + список нод.
2. `Обновить информацию о модулях` отрабатывает и завершает статус одной строкой.
3. Переключение `Module Nodes -> Workflows -> Module Nodes` работает стабильно.
4. Проблема из пункта 1 воспроизводится по шагам (или подтверждается, что больше не воспроизводится).
