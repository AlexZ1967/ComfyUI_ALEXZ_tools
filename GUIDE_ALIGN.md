# Align Overlay To Background — Guide

## Назначение
Геометрическое выравнивание `overlay` к `background` по фичам (ORB/AKAZE/SIFT) с экспортом `transform_json`.

## Когда использовать
- Нужно совместить два почти одинаковых изображения (скан/рендер/кадр до-после).
- Нужно получить трансформацию для Fusion/Resolve.

## Минимальный сценарий (3 шага)
1. Подайте `background` и `overlay`.
2. Стартуйте с `matcher_type=orb`, `feature_count=1500`, `good_match_percent=0.2`, `ransac_thresh=5`.
3. Оцените `composite`, `difference` и `transform_json.status`.

## Параметры
| Параметр | Что делает | Рекомендация |
|---|---|---|
| `matcher_type` | Детектор/дескриптор | `orb` быстро, `akaze` устойчиво, `sift` точнее/медленнее |
| `feature_count` | Кол-во ключевых точек | 1200-2500 типично |
| `good_match_percent` | Доля лучших совпадений | 0.1-0.3 |
| `min_matches` | Минимум матчей до RANSAC | 8-30 |
| `min_inliers` | Минимум inliers после RANSAC | 6-25 |
| `ransac_thresh` | Порог RANSAC (px) | 2-5 точнее, 6-10 устойчивее |
| `scale_mode` | Масштабирование | `preserve_aspect` безопаснее |
| `allow_rotation` | Разрешить поворот | Выкл, если поворот физически невозможен |
| `color_mode` / `lab_channels` | Цветовое пространство для матчинга | `gray` быстрее, `lab` надёжнее на цвете |
| `background_mask` / `overlay_mask` | Область матчинга | Белое = использовать |

## Decision helper
- Лёгкий кейс, нужна скорость -> `orb`, `feature_count=1200`.
- Шум/компрессия -> `akaze`, `feature_count=2000`.
- Точный совмес на сложной текстуре -> `sift`, `feature_count=2500+`.
- Поворота быть не должно -> `allow_rotation=false`.

## Интерпретация выходов
- `aligned_overlay`: оверлей в координатах фона.
- `composite`: наложение на фон с `opacity`.
- `difference`: абсолютная разница `|aligned - background|`.
- `transform_json`: статус, матрица/параметры трансформации и данные для NLE.

## Типовые ошибки и решения
- `status` = not enough matches/inliers: увеличьте `feature_count`, смягчите `min_matches/min_inliers`, попробуйте `sift`.
- Неверный масштаб/сдвиг: уменьшите `ransac_thresh`, проверьте маски.
- Плохой матч на цветных паттернах: `color_mode=lab`.

## Производительность
- Самое тяжёлое: `sift` + большое `feature_count`.
- Для батчей и 4K: снижайте `feature_count`, начинайте с `orb`.
