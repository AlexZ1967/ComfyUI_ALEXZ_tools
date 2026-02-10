# Color Match To Reference — Quick Guide

Полный развернутый гайд: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md).

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
| `preset` | Режим коррекции | `fast`/`balanced`/`quality`/`perceptual` |
| `strength` | Сила эффекта | 0.6-0.9 |

## Decision helper
- Скорость -> `fast`.
- Базовое качество -> `balanced`.
- Точность -> `quality`.
- Максимум визуального match -> `perceptual`.

## Интерпретация выходов
- `matched_image`: результат.
- `match_json.quality`: метрики до/после и `improvement_pct`.

## Типовые ошибки и решения
- Перекоррекция -> снизить `strength`.
- Слабая коррекция -> перейти на `quality` или `perceptual`.

## Производительность
- `fast` быстрее всех, `perceptual` медленнее всех.
