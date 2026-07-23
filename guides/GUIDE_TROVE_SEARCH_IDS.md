# GUIDE: Search Trove Image IDs

## Назначение
Нода `Search Trove Image IDs` выполняет API-first поиск по Trove в категории `Images, Maps & Artefacts` и пытается извлечь `nla.obj-...` идентификаторы для дальнейшей пакетной загрузки через DZI-ноды.

Важно:
- основной режим использует официальный Trove API v3 `/result`;
- без API key нода может попробовать запрос без авторизации, но текущий Trove API v3
  отвечает на него `401 No API key found in request`;
- если API key задан, он передается через HTTP header `X-API-KEY`, а не через URL;
- legacy headless Chrome режим оставлен только как optional advanced fallback;
- Chrome fallback зависит от публичного UI Trove и может ломаться из-за DOM/anti-bot/Chrome runtime.

## Когда использовать
- Когда нужен стартовый список `nla.obj-...` по текстовому запросу, например `Pavlova`.
- Когда нужен API-поиск с ключом или явная диагностика доступности запроса без ключа.
- Когда вы хотите быстро собрать IDs и затем передать их в `Download DZI Tiles Batch Save`.

## API key setup
Для диагностической проверки можно оставить `api_key` пустым и не задавать
`TROVE_API_KEY`: нода отправит запрос без авторизации. На момент проверки
2026-07-23 Trove API v3 требует ключ и возвращает `401`.

Для более стабильных лимитов рекомендуется:

1. Получите Trove API key в аккаунте Trove / National Library of Australia.
2. Перед запуском ComfyUI задайте переменную окружения:

```bash
export TROVE_API_KEY="your_key_here"
```

3. В ноде оставьте `api_key` пустым, чтобы ключ не сохранялся в workflow JSON.

Допустимый, но менее безопасный способ: вставить ключ прямо в поле `api_key`. Это удобно для быстрой проверки, но такой workflow будет содержать ключ.

## Минимальный сценарий (3 шага)
1. Добавьте ноду `Search Trove Image IDs`.
2. Укажите `query`, например `Pavlova`.
3. Подайте выход `ids_text` в batch DZI workflow или сохраните его через `Show/Save JSON`.

## Параметры
- `query` (`STRING`): поисковый запрос для Trove.
- `search_mode` (`api_first|api_only|browser_only`): `api_first` сначала использует официальный API; `api_only` не запускает Chrome; `browser_only` включает legacy headless Chrome scraping публичной страницы.
- `api_key` (`STRING`): опциональный Trove API key. Пусто = использовать `TROVE_API_KEY`, а если переменная не задана — попробовать запрос без авторизации.
- `category` (`images`): UI-совместимое имя категории; для API автоматически нормализуется в `image`.
- `max_results` (`INT`): максимум возвращаемых `nla.obj-...`.
- `include_online_only` (`BOOLEAN`): добавляет API facet `l-availability=y/f`, чтобы предпочитать онлайн-доступные записи.
- `enable_browser_fallback` (`BOOLEAN`): разрешить Chrome fallback, если `api_first` не дал IDs.
- `virtual_time_budget_ms` (`INT`): сколько времени дать headless Chrome на рендер страницы перед `dump-dom`; используется только в browser/fallback режиме.

## Decision helper
- Для проверки поведения без ключа: используйте `api_first` или `api_only` с пустым `api_key`; ожидаемый текущий результат — диагностический `401`.
- Если нужен более стабильный лимит запросов: задайте `TROVE_API_KEY`.
- Если важно исключить Chrome и scraping полностью: используйте `api_only`.
- Если ключ недоступен и вы готовы к нестабильному результату: используйте `browser_only` или включите `enable_browser_fallback`.
- Если `count=0`, посмотрите `result_json.warning`: API мог вернуть 401/403/429/5xx, browser mode мог упереться в anti-bot, или запрос мог не дать image results.

## Интерпретация выходов
- `ids_text`: найденные `nla.obj-...`, по одному на строку.
- `result_json`: диагностический JSON с `mode`, `api_url`, `api_category`, `api_key_source`, `count`, `warning`, `diagnostic`; в browser-only режиме также будут `search_url`, `chrome_path`, `stdout_excerpt`, `stderr_excerpt`.
- `count`: число найденных уникальных IDs.

## Типовые ошибки и решения
- `status=401/403`, `api_key_source=anonymous`: Trove отклонил анонимный доступ; задайте `TROVE_API_KEY` или заполните `api_key`.
- `status=401/403` с ключом: проверьте, что ключ действителен и относится к Trove API; перезапустите ComfyUI после изменения переменной окружения.
- `status=429`: достигнут анонимный лимит или лимит API key; уменьшите частоту запросов и повторите позже.
- `Chrome/Chromium binary was not found in PATH`: установите Chrome/Chromium или добавьте бинарник в `PATH`.
- `count=0`, `warning=Trove anti-bot challenge...`: Trove заблокировал headless search flow; используйте API key или ручной список IDs.
- `count=0`, `warning=results were not auto-expanded`: текущий UI Trove не раскрыл результаты без дополнительного клика/действия.

## Производительность
- API-first режим делает обычный HTTP-запрос и существенно быстрее Chrome.
- Browser-only режим ограничен запуском headless Chrome и рендером SPA.
- Увеличение `virtual_time_budget_ms` может помочь Chrome fallback, но делает ноду медленнее.
