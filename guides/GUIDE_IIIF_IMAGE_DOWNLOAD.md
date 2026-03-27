# GUIDE: Download IIIF Image

## Назначение
Нода `Download IIIF Image` скачивает изображение из IIIF Image API сервиса и возвращает его как ComfyUI `IMAGE`.

Текущий фокус:
- `London Museum Object Page`: принимает URL страницы объекта и сама извлекает IIIF service URL.
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
  - `Generic IIIF Service URL`
- `source_url` (`STRING`):
  - для London Museum: URL object page, например  
    `https://www.londonmuseum.org.uk/collections/v/object-443296/early-portrait-of-anna-pavlova/`
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
- `filename_mode` (`source_url_slug|title_or_slug`):
  - `source_url_slug`: имя файла из последнего meaningful сегмента `source_url`
  - `title_or_slug`: попытаться взять `og:title`/`title` страницы `source_url`, иначе fallback на slug из URL
- `delivery_mode` (`single_request|tile_assemble_full`):
  - `single_request`: один IIIF image request (`full/max/...` или `full/{width},/...`)
  - `tile_assemble_full`: скачать и склеить full-resolution изображение из IIIF tiles
- `output_format` (`jpg|png|webp|tif|gif`): желаемый формат IIIF image request  
  Если формат не сработал, нода пробует fallback на `jpg`.

## Decision helper
- Нужна страница London Museum объекта:
  - `site = London Museum Object Page`
  - `source_url = URL страницы объекта`
- Уже есть IIIF `info.json` или service URL:
  - `site = Generic IIIF Service URL`
  - `source_url = direct IIIF URL`
- Нужен максимум качества:
  - `size_mode = max`
- Сервис режет single-request размер:
  - `delivery_mode = tile_assemble_full`
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
