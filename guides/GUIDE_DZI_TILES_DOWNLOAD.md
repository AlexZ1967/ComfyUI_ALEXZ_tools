# GUIDE: Download DZI Tiles Image / Batch Save

## Назначение
Ноды `Download DZI Tiles Image` и `Download DZI Tiles Batch Save` работают с Deep Zoom Image (DZI) источниками.

- `Download DZI Tiles Image`: скачивает один DZI-источник и возвращает `IMAGE`.
- `Download DZI Tiles Image`: также может опционально сохранить итоговую картинку на диск и выбрать имя по `mw` или title страницы объекта.
- `Download DZI Tiles Batch Save`: скачивает список DZI-источников и сразу сохраняет результат на диск, возвращая manifest JSON и статистику.

Сейчас поддерживаются как минимум две схемы:
- `npg`: `collectionimages.npg.org.uk`
- `nla`: `nla.gov.au`

## Когда использовать
- Когда исходник доступен только через DeepZoom/тайлы.
- Когда нужно быстро получить цельную картинку из `site + mw + level`.
- Когда DZI-метаданные частично недоступны: нода умеет строить сетку тайлов probe-методом.
- Когда нужно скачать сразу много изображений и сохранить их в папку без промежуточного выхода `IMAGE`.

## Минимальный сценарий (3 шага)
1. Добавьте ноду `Download DZI Tiles Image`.
2. Выберите `site`, затем укажите `mw` и `level`.
3. При необходимости задайте `output_dir`, если нужно сразу сохранить файл на диск.
4. Подключите выход `image` к `Preview Image` или следующей ноде.

Для batch-варианта:
1. Добавьте ноду `Download DZI Tiles Batch Save`.
2. Вставьте список ID в `ids_text`, задайте `output_dir` и при необходимости `filename_template`.
3. Запустите ноду и используйте `manifest_json` / `saved_paths_json` для контроля результата.

## Параметры
- `site` (`LIST`): сайт-источник из `config/dzi_sites.json`.  
  Примеры: `National Portrait Gallery UK`, `National Library of Australia`
- `mw` (`STRING`): идентификатор изображения.  
  Можно вводить только цифры, и нода сама добавит префикс выбранного сайта.  
  Пример для NPG: `207134` или `mw207134`  
  Пример для NLA: `138204672` или `nla.obj-138204672`  
  Если поле пустое, используется `default_mw` выбранного сайта из `config/dzi_sites.json`.
- `level` (`INT`): уровень тайлов.  
  Если указать `-1`, используется `default_level` выбранного сайта из `config/dzi_sites.json`.
- `transport` (`auto|requests|cloudscraper|urllib|curl`): HTTP-транспорт.
- `proxy_url` (`STRING`): явный прокси URL (опционально).  
  Если пусто, нода использует автоопределение маршрута (env/system proxy + локальные прокси).
- `tile_extension` (`jpg|jpeg|png|webp`): формат тайлов на стороне сервера.  
  Нода использует только выбранный формат, без перебора остальных.
- `output_dir` (`STRING`, single optional / batch required): папка для сохранения итоговых изображений.  
  Для `Download DZI Tiles Image` пустое значение означает: не сохранять файл, только вернуть `IMAGE`.
- `output_extension` (`png|jpg|jpeg|webp`, single optional / batch only previously): формат сохранения итогового файла.
- `filename_mode` (`mw|title_or_mw`, single only): режим имени файла для `Download DZI Tiles Image`.  
  `title_or_mw` пытается взять `og:title`/`title` со страницы объекта и использует `mw` как fallback.  
  В режиме `title_or_mw` к title также автоматически добавляется stable ID объекта.  
  Пример: `Anna_Pavlova_as_the_Dying_swan_Melbourne_1926_nla.obj-138204672.png`
- `ids_text` (`STRING`, batch only): список ID, по одному на строку.  
  Также поддерживаются разделители `,` и `;`. Пустые строки и строки/хвосты после `#` игнорируются.
- `filename_template` (`STRING`, batch only): шаблон имени файла без расширения.  
  Поддерживаются плейсхолдеры: `{index}`, `{raw_id}`, `{mw}`, `{id}`, `{title}`, `{site}`, `{site_key}`, `{level}`.  
  `{title}` пытается взять `og:title`/`title` со страницы объекта и использует `mw` как fallback.
- `overwrite_mode` (`skip|overwrite|unique`, batch only): поведение при существующем файле.  
  В режиме `unique` нода теперь пишет человекочитаемые суффиксы: `_2`, `_3`, ...
- `continue_on_error` (`true|false`, batch only): продолжать ли батч после ошибки отдельного элемента.
- `save_mode` (`save_only|save_and_manifest`, batch only): сохранять только изображения или дополнительно записывать `dzi_batch_manifest*.json`.

## Конфиг сайтов
Файл: `config/dzi_sites.json`

Для каждого сайта в конфиге задаются:
- `name`: отображаемое имя в dropdown.
- `base_url`: корневой URL сайта.
- `provider`: идентификатор схемы/сайта.
- `default_mw`: дефолтный идентификатор изображения.
- `mw_prefix`: префикс, который нода автоматически добавляет, если в `mw` введены только цифры.
- `default_level`: дефолтный DZI level.
- `mw_format`: ожидаемый формат `mw`.
- `object_url_template`: шаблон URL страницы/объекта.
- `dzi_url_template`: шаблон URL DZI metadata.
- `tile_url_template`: шаблон URL отдельного тайла.
- `url_scheme`: справочная строка, как формируется tile URL.

Шаблоны используют плейсхолдеры:
- `{base_url}`
- `{mw}`
- `{level}`
- `{x}`
- `{y}`
- `{ext}`

Это значит, что для большинства новых сайтов достаточно добавить запись в
`config/dzi_sites.json` без правок Python-кода.

## Decision helper
- Для `National Portrait Gallery UK` используйте `mw` вида `mw...`.
- Для `National Library of Australia` используйте `mw` вида `nla.obj-...`.
- Если не уверены в уровне, начните с уровня, который точно существует у источника.
- Если при запуске ошибка `First tile is unavailable`, проверьте корректность `site`, `mw`, `level` и `tile_extension`.
- Если хотите полный размер из конкретного уровня, используйте probe-режим (встроен автоматически).
- Если в браузере URL открывается, а в ноде нет, оставьте `proxy_url` пустым (авто) или задайте явный рабочий прокси.

## Интерпретация выходов
- `image`: собранное изображение в формате ComfyUI `IMAGE` (`[1, H, W, 3]`, float32, `0..1`).
- `Download DZI Tiles Image` при заданном `output_dir` сохраняет файл либо с именем из `mw`, либо с title страницы объекта, если выбран `filename_mode=title_or_mw`.  
  Для `title_or_mw` single-нода дописывает к title stable ID объекта, чтобы разные картинки с похожими названиями не конфликтовали.  
  Если файл с таким именем уже существует, single-нода автоматически сохраняет как `name_2.ext`, `name_3.ext`, ...
- `manifest_json` (batch): JSON с параметрами батча, статусом каждого элемента и итоговыми счётчиками.
- `saved_paths_json` (batch): JSON-массив путей к успешно сохранённым файлам.
- `count_ok` / `count_failed` (batch): агрегированная статистика батча.

## Типовые ошибки и решения
- Ошибка доступа к первому тайлу:
  - Причина: неверный URL/ID/уровень или сервер блокирует запрос.
  - Решение: проверьте путь `.../zoomXML_files/<level>/0_0.<tile_extension>` вручную.
- Неполная картинка:
  - Причина: источник имеет нестандартную схему тайлов.
  - Решение: проверьте доступность тайлов по осям и корректность `level`.
- Batch ничего не сохранил:
  - Причина: пустой `ids_text`, неверный `output_dir` или все элементы были пропущены режимом `skip`.
  - Решение: проверьте `ids_text`, права на запись в `output_dir` и `overwrite_mode`.

## Производительность
- Основное время уходит на сетевые запросы и декодирование JPEG.
- При больших сетках тайлов узким местом будет сеть и латентность сервера.
- Для ускорения держите источник в стабильной сети и избегайте слишком высоких уровней без необходимости.
- В консоли ComfyUI отображается прогресс-бар `DZI Tiles` и подробные логи ошибок/пропусков тайлов.
