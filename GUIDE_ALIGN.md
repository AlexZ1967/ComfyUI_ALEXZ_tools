# Align Overlay To Background — Guide

## Назначение
Выравнивание оверлея по фону (масштаб/поворот/сдвиг) с экспортом трансформации для Fusion/Resolve. Отдаёт aligned/composite/difference и transform_json.

## Быстрый старт
1) Подайте `background` и `overlay` (желательно одинаковое или близкое содержимое).  
2) Начальные параметры: `matcher_type=orb`, `feature_count=1500`, `good_match_percent=0.2`, `ransac_thresh=5`, `opacity=1.0`, `scale_mode=preserve_aspect`, `allow_rotation=true`.  
3) По желанию маски: `background_mask` / `overlay_mask` (белое=учитывать).  
4) Смотрите `transform_json.status`: ok = успех, иначе причина.

## Основные параметры
- `matcher_type`: orb (быстро), akaze (устойчив к шуму), sift (точно, медленнее).  
- `feature_count`: больше — точнее, но медленнее. Типично 1200–2500.  
- `good_match_percent`: доля лучших матчей (0.1–0.3 обычно).  
- `min_matches` / `min_inliers`: нижние пороги старта и качества решения.  
- `ransac_thresh`: 2–5 точнее, 6–10 устойчивее к шуму.  
- `scale_mode`: preserve_aspect | independent_xy.  
- `allow_rotation`: запретить, если нежелателен поворот.  
- `color_mode`: gray (универсально), lab_l, lab (лучше на цветных текстурах).  
- `opacity`: смешивание aligned + background в composite.

## Рекомендации по настройке
- Если `Not enough matches/inliers`: увеличьте `feature_count`, снизьте `min_matches`/`min_inliers`, повысите `good_match_percent`, попробуйте `sift`.  
- Для крупных цветовых паттернов включите `color_mode=lab` и/или `lab_channels=lab`.  
- Для неравномерных деформаций — `scale_mode=independent_xy`, но это менее стабильно.  
- Маски: выделяйте полезные участки (лого/текстуры), чтобы снизить ложные совпадения.

## Интерпретация outputs
- `aligned_overlay`: оверлей, приведённый к фону.  
- `composite`: фон + aligned с заданной `opacity`.  
- `difference`: |aligned - background| (быстро подсветить рассинхрон).  
- `transform_json`: scale/rotation/position в пикселях и нормированных координатах; готово для Fusion/Resolve.

## Производительность
- Основная нагрузка — OpenCV feature detect + RANSAC; растёт с `feature_count` и `sift`.  
- Для больших батчей/высокого разрешения: уменьшите `feature_count`, оставьте `orb`, понизьте `good_match_percent`.

## Частые проблемы
- `status` не ok: недобор матчей или инлайеров — см. рекомендации выше.  
- Сильный сдвиг/масштаб: попробуйте `sift` и меньший `ransac_thresh`.  
- Прозрачный overlay: альфа учитывается, composite использует альфу + `opacity`; difference — по RGB. 
