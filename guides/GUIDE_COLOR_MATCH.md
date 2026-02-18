# Color Match To Reference — Quick Guide

Полный развернутый гайд: [GUIDE_COLOR_MATCH_DETAILED.md](GUIDE_COLOR_MATCH_DETAILED.md).

## Назначение
Быстрый выбор пресета для подгонки цвета `image` к `reference` (8 методов).

## Когда использовать
- Нужна быстрая настройка без чтения полного гайда.

## Минимальный сценарий (3 шага)
1. Подайте `reference` и `image`.
2. Выберите `preset`.
3. Проверьте `match_json.quality`.

## Параметры
| Параметр | Что делает | Рекомендация |
|---|---|---|
| `preset` | Метод цветокоррекции | 9 методов: `mean_std`, `linear`, `tone_curve`, `adain`, `optimal_transport`, `lab_cdf`, `oklab_cdf`, `auto_optimal`, `perceptual_vgg_fast` |
| `strength` | Сила эффекта | 0.6-0.9 |
| `compute_quality_metrics` | Считать quality-метрики | Отключить для ускорения батча |
| `auto_optimal_metric` | Критерий выбора в `auto_optimal` | `mse_ssim` по умолчанию, `mse_ssim_lpips` для финального качества |
| `export_lut` | Сохранять LUT `.cube` | Включить при подготовке grading pipeline |
| `lut_size` | Размер 3D LUT | 17 или 33 для практики |
| `lut_output_dir` | Папка LUT | Пусто = `./output/color_luts` |
| `lut_name` | Имя LUT | Базовое имя без расширения |

## Decision helper
- Скорость → `mean_std` (среднее/стд по каналам).
- Базовое качество → `linear` (линейная подгонка, дефолт).
- Контрастность/тоны → `tone_curve` (кривая тонов по квантилям).
- Перцептивно → `adain` (adaptive normalization).
- Математически строго → `optimal_transport` (Wasserstein distance, монотонное отображение).
- Высокое качество → `lab_cdf` (Lab гистограмма) или `oklab_cdf` (Oklab, перцептивнее).
- Автовыбор (без ручного тюнинга) → `auto_optimal` (выбирает между `linear` и `oklab_cdf` по MSE к референсу).
- Максимум → `perceptual_vgg_fast` (нейросеть, медленнее).

## Сравнение методов

| Метод | Скорость | Качество | Когда использовать |
|-------|----------|----------|-------------------|
| `mean_std` | ⚡⚡⚡ | ⭐ | Максимальная скорость, выравнить средние значения |
| `linear` | ⚡⚡ | ⭐⭐ | Баланс скорость/качество (дефолт) |
| `tone_curve` | ⚡⚡ | ⭐⭐ | Матчинг контраста и кривой тонов |
| `adain` | ⚡⚡ | ⭐⭐⭐ | Перцептивная нормализация |
| `optimal_transport` | ⚡ | ⭐⭐⭐ | Wasserstein расстояние, математически обосновано |
| `lab_cdf` | ⚡ | ⭐⭐⭐ | Хорошее качество, гистограмма |
| `oklab_cdf` | ⚡ | ⭐⭐⭐⭐ | Лучший результат, перцептивно улучшенный |
| `auto_optimal` | ⚡ | ⭐⭐⭐⭐ | Автовыбор между `linear` и `oklab_cdf` |
| `perceptual_vgg_fast` | 🐢 | ⭐⭐⭐⭐⭐ | Максимальное качество через нейросеть |

## Интерпретация выходов
- `matched_image`: результат.
- `match_json.quality`: метрики до/после и `improvement_pct`.

## Типовые ошибки и решения
- Перекоррекция → снизить `strength`.
- Слабая коррекция → перейти с `linear` на `lab_cdf`/`oklab_cdf` или `perceptual_vgg_fast`.

## Производительность
- `mean_std` быстрее всех, `perceptual_vgg_fast` медленнее всех.
- `oklab_cdf` немного медленнее `lab_cdf` но перцептивно лучше.
