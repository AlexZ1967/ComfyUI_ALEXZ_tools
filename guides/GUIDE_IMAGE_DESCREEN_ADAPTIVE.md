# GUIDE: Descreen By Adaptive Scale (Legacy)

## Назначение
Нода `Descreen By Adaptive Scale` теперь считается legacy all-in-one вариантом.

Она:
- оценивает шаг растра по ROI через FFT;
- перебирает диапазон scale-кандидатов;
- выбирает scale с лучшим компромиссом между подавлением растра и сохранением структуры;
- строит scale-sheet от расчетного базового процента вверх, чтобы можно было выбрать вариант глазом;
- сразу отдает обработанное изображение.

## Когда использовать
- Когда нужен старый one-shot workflow в одной ноде.
- Когда вы не хотите разделять этапы `estimate -> preview -> apply`.
- Когда скан или репродукция содержит полутоновый печатный растр.
- Когда вы хотите подобрать рабочий downscale автоматически, а не вручную.
- Для новых workflow предпочтительнее связка:
  - `Estimate Raster Period`
  - `Descreen Scale Preview`
  - `Apply Descreen Percent`

## Минимальный сценарий (3 шага)
1. Подайте изображение в `Descreen By Adaptive Scale`.
2. Оставьте `roi_mode=center_square`, если растр в центре хорошо виден.
3. Используйте выход `image`, а `recommended_percent` и `analysis_json` смотрите для контроля.
  Выход `scale_sheet` показывает подборку scale-вариантов с подписями процентов без отдельной карточки исходника.

## Параметры
- `image` (`IMAGE`): входной скан/изображение с видимым растром.
- `roi_mode` (`center_square|full_frame|manual_rect`):
  - `center_square`: анализ по центральному квадрату;
  - `full_frame`: анализ по всему кадру;
  - `manual_rect`: анализ по прямоугольному ROI.
- `roi_size_percent` (`FLOAT`): размер центрального ROI в процентах от меньшей стороны кадра.
- `roi_x`, `roi_y`, `roi_w`, `roi_h` (`INT`): ручной ROI для `manual_rect`.
- `min_scale_percent`, `max_scale_percent` (`FLOAT`): диапазон поиска optimal scale.
- `step_percent` (`FLOAT`): шаг перебора.
- `resample_mode` (`lanczos|bicubic`): ресемплер для scale-descreen.
- `target_screen_px` (`FLOAT`): целевой остаточный размер шага растра после уменьшения.
  Меньше = агрессивнее подавление.
- `detail_weight` (`FLOAT`): штраф за потерю структуры.
  Больше = бережнее к деталям, меньше = агрессивнее к растру.
- `pre_blur_px` (`FLOAT`): blur перед downscale.
  По умолчанию `0.0`, потому что на многих сканах лучший результат получается без blur.
- `sheet_zone_mode` (`analysis_roi|center_square|full_frame|manual_rect`): отдельная зона для `scale_sheet`.
  `analysis_roi` использует тот же ROI, что и анализ. Остальные режимы задают sheet preview независимо.
- `sheet_zone_size_percent` (`FLOAT`): размер центральной зоны для `sheet_zone_mode=center_square`.
- `sheet_zone_x`, `sheet_zone_y`, `sheet_zone_w`, `sheet_zone_h` (`INT`): ручная зона для `sheet_zone_mode=manual_rect`.
- `sheet_range_up_percent` (`FLOAT`): на сколько процентов вверх от расчетного базового scale строить подборочную лестницу в `scale_sheet`.
- `sheet_step_percent` (`FLOAT`): шаг процентов между вариантами в `scale_sheet`.

## Decision helper
- Если не знаете, где выбирать ROI:
  - начните с `center_square`.
- Если в центре мало фактуры, а растр по всему кадру:
  - попробуйте `full_frame`.
- Если нужен точный замер по фрагменту:
  - используйте `manual_rect`.
- Если анализ нужно делать по одной зоне, а подбор вариантов смотреть по другой:
  - настройте `sheet_zone_mode` и `sheet_zone_*` отдельно.
- Если нода предлагает слишком маленький scale и картинка становится мыльной:
  - увеличьте `detail_weight`.
- Если растр еще виден:
  - уменьшите `target_screen_px`
  - или расширьте диапазон вниз (`min_scale_percent`).

## Интерпретация выходов
- `image`: обработанное изображение после adaptive downscale/upscale.
- `scale_sheet`: подборка вариантов `base | base+step | ...` по выбранной зоне preview.
- `recommended_percent`: рекомендованный процент downscale.
- `estimated_period_px`: оцененный шаг растра в пикселях.
- `analysis_json`: JSON с:
  - ROI;
  - estimated period;
  - predicted percent по формуле;
  - базовым scale для sheet;
  - списком `sheet_scales`;
  - таблицей кандидатов и их score.

## Типовые ошибки и решения
- Рекомендуемый scale выглядит слишком маленьким:
  - Причина: нода агрессивно минимизирует остаточный растр.
  - Решение: увеличьте `detail_weight`.
- Растр почти не ушел:
  - Причина: диапазон поиска слишком узкий.
  - Решение: уменьшите `min_scale_percent` и/или `target_screen_px`.
- ROI неудачный:
  - Причина: в области анализа мало периодической структуры или слишком много пятен/грязи.
  - Решение: смените ROI.

## Производительность
- Нода работает быстро, потому что анализ делает только по ROI.
- Чем меньше `step_percent`, тем точнее поиск, но тем больше перебор.
- `scale_sheet` строится отдельно и почти не влияет на скорость, если ROI небольшой.
- Для практики обычно достаточно:
  - диапазон `8..20%`
  - шаг `1%`
  - `sheet_range_up_percent=10`
  - `sheet_step_percent=2`
  - затем при необходимости доуточнение вокруг найденной точки.
