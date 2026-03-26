# GUIDE: Search Trove Image IDs

## Назначение
Нода `Search Trove Image IDs` выполняет best-effort поиск по Trove в категории `Images, Maps & Artefacts` и пытается извлечь `nla.obj-...` идентификаторы для дальнейшей пакетной загрузки через DZI-ноды.

Важно:
- это не официальный Trove API режим;
- нода использует headless Chrome и рендер публичной search-страницы;
- Trove может менять UI-flow, anti-bot защиту или требовать дополнительное взаимодействие.

## Когда использовать
- Когда нужен стартовый список `nla.obj-...` по текстовому запросу, например `Pavlova`.
- Когда нет API key для Trove, но нужен best-effort поиск через публичный web UI.
- Когда вы хотите быстро собрать IDs и затем передать их в `Download DZI Tiles Batch Save`.

## Минимальный сценарий (3 шага)
1. Добавьте ноду `Search Trove Image IDs`.
2. Укажите `query`, например `Pavlova`.
3. Подайте выход `ids_text` в batch DZI workflow или сохраните его через `Show/Save JSON`.

## Параметры
- `query` (`STRING`): поисковый запрос для Trove.
- `category` (`images`): сейчас поддерживается только категория изображений.
- `max_results` (`INT`): максимум возвращаемых `nla.obj-...`.
- `virtual_time_budget_ms` (`INT`): сколько времени дать headless Chrome на рендер страницы перед `dump-dom`.

## Decision helper
- Если нужен production-grade и стабильный поиск: лучше использовать официальный Trove API с личным API key.
- Если нужен быстрый best-effort без ключа: используйте эту ноду.
- Если `count=0`, посмотрите `result_json.warning`:
  - могла сработать anti-bot защита;
  - Trove мог не раскрыть результаты автоматически;
  - запрос мог не дать image results.

## Интерпретация выходов
- `ids_text`: найденные `nla.obj-...`, по одному на строку.
- `result_json`: диагностический JSON:
  - `search_url`
  - `chrome_path`
  - `count`
  - `warning`
  - `stdout_excerpt`
  - `stderr_excerpt`
- `count`: число найденных уникальных IDs.

## Типовые ошибки и решения
- `Chrome/Chromium binary was not found in PATH`:
  - установите Chrome/Chromium или добавьте бинарник в `PATH`.
- `count=0`, `warning=Trove anti-bot challenge...`:
  - Trove заблокировал headless search flow;
  - в таком случае лучше использовать API key или ручной список IDs.
- `count=0`, `warning=results were not auto-expanded`:
  - текущий UI Trove не раскрыл результаты без дополнительного клика/действия.

## Производительность
- Скорость ограничена запуском headless Chrome и рендером SPA.
- Для единичных запросов это приемлемо, для массового поиска хуже API.
- Увеличение `virtual_time_budget_ms` может помочь, но делает ноду медленнее.
