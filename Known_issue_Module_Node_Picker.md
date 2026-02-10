# Known Issues: Module Node Picker

![KNOWN ISSUE](https://img.shields.io/badge/KNOWN%20ISSUE-red)

## 1) `Module Nodes -> NodesMap` иногда открывает пустой `NodesMap`
В некоторых конфигурациях ComfyUI при прямом переключении из `Module Nodes` в `NodesMap` вкладка `NodesMap` может открыться пустой.

### Влияние
На функциональность нод и самого `Module Nodes` это не влияет.

### Обходной путь
1. Переключитесь сначала на любой другой виджет (например `Workflows` или `PNG Info`).
2. Затем откройте `NodesMap`.

После такого переключения `NodesMap` обычно отображается корректно.

## Почему раньше текст "красным" не работал в README
В GitHub Markdown inline-стили типа `<span style="color:red">...</span>` часто санитизируются и не применяются.
Поэтому для заметных предупреждений лучше использовать:
- badge (как выше),
- либо текстовые маркеры (`⚠`, `🟥`),
- либо отдельный файл `Known Issues`.
