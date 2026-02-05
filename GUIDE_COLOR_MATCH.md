# Color Match To Reference — Guide (кратко)

Основной гайд: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md)

## Пресеты
- `fast` — быстрый mean/std match.
- `balanced` — линейная подгонка (scale/offset).
- `quality` — LAB CDF match (качественнее, но медленнее).
- `perceptual` — VGG perceptual (самый медленный).

## Выходы
- `matched_image` — результат коррекции.
- `difference` — |matched - reference|.
- `match_json` — параметры коррекции и статистика.
