# Color Match To Reference — Practical Guide

## Быстрый старт
1) Подайте `reference` (эталон) и `image` (исправить).  
2) Выберите `preset`:
   - `fast` — mean/std match (самый быстрый),
   - `balanced` — linear match (надёжный дефолт),
   - `quality` — LAB CDF match (точнее, медленнее),
   - `perceptual` — VGG perceptual fast (самый медленный).
3) При необходимости снизьте `strength` до 0.6–0.8.
4) Смотрите `match_json` для контроля; разницу можно вывести отдельной нодой Image Difference.

## Пресеты
-- **fast** → mean/std match.  
- **balanced** → linear match (scale/offset).  
- **quality** → LAB CDF match.  
- **perceptual** → VGG perceptual fast.

## Маски
- `match_mask`: где собирать статистику (белое = учитывать).  
- `apply_mask`: где применять коррекцию (белое = применить).  

## Как оценивать качество
-- **match_json.stats**: mean/std по каналам до/после.

## Советы
- Если результат «перекручен», уменьшите `strength`.  
- Если цвета всё равно не совпадают — попробуйте `quality` или `perceptual`.  
- При сложных сценах используйте `match_mask`, чтобы исключить фон/шум.
