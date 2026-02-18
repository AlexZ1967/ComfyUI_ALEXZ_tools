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
| `compute_quality_metrics` | Legacy-флаг метрик | `false` принудительно отключает метрики |
| `quality_metrics_mode` | Режим метрик | `off` / `fast` / `full` |
| `auto_optimal_metric` | Критерий выбора в `auto_optimal` | `mse_ssim` по умолчанию, `mse_ssim_lpips` для финального качества |
| `auto_temporal_stability` | Стабилизация выбора в `auto_optimal` | Включать для видео |
| `auto_temporal_alpha` | EMA сглаживание авто-оценки | 0.7-0.9 |
| `auto_switch_threshold` | Порог переключения метода | 0.005-0.03 |
| `auto_quality_fallback` | Fallback в `auto_optimal` | Включить для сложных сцен |
| `auto_fallback_method` | Метод fallback | `lab_cdf` / `oklab_cdf` / `perceptual_vgg_fast` |
| `auto_fallback_threshold` | Порог включения fallback | 0.02-0.08 |
| `auto_fallback_margin` | Мин. улучшение score для fallback | 0.0005-0.005 |
| `spatial_grid` | Локальный матчинг NxN | 2-4 для неравномерного света |
| `skin_tone_protection` | Защита оттенков кожи | Включать для портретов |
| `skin_protection_strength` | Сила защиты кожи | 0.4-0.8 |
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
- Для видео в `auto_optimal` включайте `auto_temporal_stability` чтобы избежать частых переключений метода.
- Для сложных кейсов включайте `auto_quality_fallback`, чтобы `auto_optimal` мог перейти на более тяжелый метод.
- При градиентах освещения используйте `spatial_grid` для локальной подгонки.
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
- Пустая `match_mask` → нода вернет исходное изображение для кадра и запишет warning в лог.

## Производительность
- `mean_std` быстрее всех, `perceptual_vgg_fast` медленнее всех.
- `oklab_cdf` немного медленнее `lab_cdf` но перцептивно лучше.
