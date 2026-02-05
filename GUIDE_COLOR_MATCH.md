# Color Match To Reference — Guide (кратко)

Основной гайд: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md)

## Пресеты
- `fast` — mean/std match (самый быстрый).
- `balanced` — linear match (надёжный базовый).
- `quality` — LAB CDF match (точнее, медленнее).
- `perceptual` — VGG perceptual fast (самый медленный).

## Выходы
- `matched_image` — результат коррекции.
- `match_json` — параметры коррекции и статистика.
