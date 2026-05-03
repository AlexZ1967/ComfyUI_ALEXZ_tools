# GUIDE: Download IIIF Image

## Назначение
Нода `Download IIIF Image` скачивает изображение из IIIF Image API сервиса и возвращает его как ComfyUI `IMAGE`.

Текущий фокус:
- `London Museum Object Page`: принимает URL страницы объекта и сама извлекает IIIF service URL.
- `Gallica BnF Object Page`: принимает Gallica ARK/object URL и строит IIIF service URL напрямую.
- `The New York Public Library (NYPL) Digital Collections`: принимает plain NYPL `image_id`, прямой `iiif.nypl.org/.../info.json` или service URL.
- `Generic IIIF Service URL`: принимает прямой IIIF service URL, `info.json` URL или HTML-страницу с встраиваемым IIIF viewer.

## Когда использовать
- Когда сайт использует IIIF Image API вместо DZI.
- Когда нужен не viewer-тайлинг, а готовая картинка в одном `IMAGE`.
- Когда источник похож на London Museum: object page -> embedded IIIF viewer -> `info.json` / `full/max`.
- Когда сервер ограничивает single-request размер и нужен full-resolution через tile assembly.

## Минимальный сценарий (3 шага)
1. Добавьте ноду `Download IIIF Image`.
2. Выберите `site` и вставьте `source_url`.
3. Подключите `image` к `Preview Image` или следующей ноде.

## Параметры
- `site` (`LIST`):
  - `London Museum Object Page`
  - `Gallica BnF Object Page`
  - `The New York Public Library (NYPL) Digital Collections`
  - `Generic IIIF Service URL`
- `source_url` (`STRING`):
  - для London Museum: URL object page, например  
    `https://www.londonmuseum.org.uk/collections/v/object-443296/early-portrait-of-anna-pavlova/`
  - для Gallica: URL object page/ARK, например  
    `https://gallica.bnf.fr/ark:/12148/btv1b10579141s/f1.item.r=L'exposition%20universelle%20de%20Paris.zoom#`
  - для NYPL:
    - прямой `info.json`, например  
      `https://iiif.nypl.org/iiif/3/57538105/info.json`
    - или service URL  
      `https://iiif.nypl.org/iiif/3/57538105`
  - для Generic:
    - IIIF service URL
    - или `info.json` URL
    - или HTML-страница, из которой можно извлечь IIIF service URL
- `size_mode` (`max|width`):
  - `max`: запросить максимально доступный размер
  - `width`: запросить изображение заданной ширины
- `requested_width` (`INT`): ширина для `size_mode=width`
- `output_dir` (`STRING`): если указана директория, нода сохранит итоговую картинку на диск  
  При пустом значении файл не сохраняется, а картинка только идет в выход `image`  
  Если имя уже занято, нода автоматически сохраняет как `name_2.ext`, `name_3.ext`, ...
- `cache_dir` (`STRING`): recovery-кеш для `tile_assemble_full`  
  При пустом значении используется встроенная папка `cache/iiif_tiles` в модуле. Уже загруженные тайлы будут переиспользованы при повторном запуске после сбоя. После полностью успешной сборки кеш именно этой картинки автоматически удаляется.
- `filename_mode` (`source_url_slug|title_or_slug`):
  - `source_url_slug`: имя файла из последнего meaningful сегмента `source_url`
  - `title_or_slug`: попытаться взять `og:title`/`title` страницы `source_url`, иначе fallback на slug из URL  
  В оба режима нода автоматически добавляет стабильный ID объекта/сервиса, если может извлечь его из URL  
  Пример: `anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house_object-443337.jpg`
- `delivery_mode` (`single_request|tile_assemble_full`):
  - `single_request`: один IIIF image request (`full/max/...` или `full/{width},/...`)
  - `tile_assemble_full`: скачать и склеить full-resolution изображение из IIIF tiles
- `output_format` (`jpg|png|webp|tif|gif`): желаемый формат IIIF image request  
  Если формат не сработал, нода пробует fallback на `jpg`.

## Decision helper
- Нужна страница London Museum объекта:
  - `site = London Museum Object Page`
  - `source_url = URL страницы объекта`
- Нужна страница Gallica BnF:
  - `site = Gallica BnF Object Page`
  - `source_url = URL страницы/ARK`
- Есть NYPL `image_id`:
  - `site = The New York Public Library (NYPL) Digital Collections`
  - `source_url = 57538105` или `source_url = NIJINSKY_2032V`
- Уже есть IIIF `info.json` или service URL:
  - `site = Generic IIIF Service URL`
  - `source_url = direct IIIF URL`
- Уже найден прямой NYPL `iiif.nypl.org/.../info.json`:
  - `site = The New York Public Library (NYPL) Digital Collections`
  - `source_url = direct IIIF URL`
- Есть только NYPL item page `digitalcollections.nypl.org/items/...`:
  - при `site = The New York Public Library (NYPL) Digital Collections` нода пытается извлечь NYPL `image_id` из HTML и собрать `https://iiif.nypl.org/iiif/3/<image_id>`;
  - если страница недоступна из-за Imperva/Incapsula, используйте plain `image_id` в `source_url` или прямой `iiif.nypl.org/.../info.json`.
- Нужен максимум качества:
  - `size_mode = max`
- Сервис режет single-request размер:
  - `delivery_mode = tile_assemble_full`
- Для NYPL full-size single request не используется:
  - нода сразу переключается на tile assembly по IIIF region URLs
- Нужен контролируемый размер для скорости:
  - `size_mode = width`
  - задайте `requested_width`

## Интерпретация выходов
- `image`: итоговое изображение в формате ComfyUI `IMAGE` (`[1, H, W, 3]`, float32, `0..1`)
- `info_json`: JSON со служебной информацией:
  - `service_url`
  - `info_url`
  - `image_url`
  - `source.width` / `source.height`
  - `downloaded.width` / `downloaded.height`
  - `saved_path`
  - `iiif.width` / `iiif.height`
  - `iiif.maxArea` / `iiif.maxAllowedSize`
  - `delivery`
  - `limits.limited_by_service`
  - `profile`, `type`, `tiles`, `sizes`

## Типовые ошибки и решения
- `Could not extract IIIF service URL`:
  - Причина: страница не содержит ожидаемый viewer/service URL
  - Решение: передайте прямой IIIF service URL или `info.json`
- `IIIF info.json unavailable`:
  - Причина: неверный service URL или сайт блокирует доступ
  - Решение: проверьте `source_url` и доступность `.../info.json`
- `IIIF image request failed`:
  - Причина: размер/формат не поддерживается сервисом
  - Решение: переключите `size_mode`, уменьшите ширину или оставьте `output_format=jpg`
- Результат меньше, чем `iiif.width` / `iiif.height`:
  - Причина: сервер ограничивает single-request размер через `maxArea` или собственную policy
  - Решение: смотрите `limits` в `info_json`; для full-res используйте `delivery_mode=tile_assemble_full`
- `full-res tile assembly is unavailable`:
  - Причина: сервис не публикует `tiles` или нет `scaleFactor=1`
  - Решение: используйте `single_request` или другой источник

## Производительность
- Эта нода обычно быстрее DZI, потому что сервер сам отдает готовую картинку по IIIF image request.
- `size_mode=width` полезен для больших музейных исходников, если не нужен full-size вывод.
- `tile_assemble_full` медленнее и делает много HTTP-запросов, но позволяет получить полный размер там, где `single_request` ограничен сервером.
- Для нестабильной сети используйте `tile_assemble_full` вместе с `cache_dir`: повторный запуск не будет заново качать уже сохранённые тайлы. После успешного завершения кеш этой сборки очищается автоматически.
