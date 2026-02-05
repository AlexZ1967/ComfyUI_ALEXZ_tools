# Color Match To Reference — Guide (кратко)

Основной гайд: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md)

## Пресеты
- `fast` — mean/std match (самый быстрый).
- `balanced` — linear match (надёжный базовый).
- `quality` — LAB CDF match (точнее, медленнее).
- `perceptual` — VGG perceptual fast (самый медленный).

## Зависимости
- Для `perceptual` нужен `torchvision` (обычно уже есть в стандартной среде ComfyUI).

## Выходы
- `matched_image` — результат коррекции.
- `match_json` — параметры коррекции, статистика и блок `quality` с метриками до/после (`mse`, `ssim`, `delta_e76`, `lpips_alex`) и `improvement_pct`.
