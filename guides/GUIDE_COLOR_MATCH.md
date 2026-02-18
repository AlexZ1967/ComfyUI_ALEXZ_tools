# Color Match To Reference — Quick Guide

Полный развернутый гайд: [GUIDE_COLOR_MATCH_DETAILED.md](GUIDE_COLOR_MATCH_DETAILED.md).

## Назначение
Быстрый выбор пресета для подгонки цвета `image` к `reference`.

## Когда использовать
- Нужна быстрая настройка без чтения полного гайда.

## Минимальный сценарий (3 шага)
1. Подайте `reference` и `image`.
2. Выберите `preset`.
3. Проверьте `match_json.quality`.

## Параметры
| Параметр | Что делает | Рекомендация |
|---|---|---|
| `preset` | Метод цветокоррекции | `mean_std`, `linear`, `lab_cdf`, `oklab_cdf`, `perceptual_vgg_fast` |
| `strength` | Сила эффекта | 0.6-0.9 |

## Decision helper
- Скорость → `mean_std` (среднее/стд по каналам).
- Базовое качество → `linear` (линейная подгонка, дефолт).
- Высокое качество → `lab_cdf` (Lab гистограмма) или `oklab_cdf` (Oklab, перцептивнее).
- Максимум → `perceptual_vgg_fast` (нейросить, медленнее).

## Интерпретация выходов
- `matched_image`: результат.
- `match_json.quality`: метрики до/после и `improvement_pct`.

## Типовые ошибки и решения
- Перекоррекция → снизить `strength`.
- Слабая коррекция → перейти с `linear` на `lab_cdf`/`oklab_cdf` или `perceptual_vgg_fast`.

## Производительность
- `mean_std` быстрее всех, `perceptual_vgg_fast` медленнее всех.
- `oklab_cdf` немного медленнее `lab_cdf` но перцептивно лучше.
