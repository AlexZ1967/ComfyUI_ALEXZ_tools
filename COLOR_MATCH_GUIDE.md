# Color Match To Reference — Practical Guide

## Быстрый старт
1) Подайте `reference` (эталон) и `image` (исправить).  
2) Выберите `preset`:
   - `fast` — быстрый и мягкий,
   - `balanced` — базовый дефолт,
   - `quality` — точнее, но медленнее,
   - `perceptual` — самый «перцептивный», но самый медленный.
3) При необходимости снизьте `strength` до 0.6–0.8.
4) Смотрите `difference` и `match_json` для контроля.

## Пресеты
- **fast** → mean/std match.  
- **balanced** → linear match (scale/offset).  
- **quality** → LAB CDF match.  
- **perceptual** → VGG perceptual (fast‑variant).

## Маски
- `match_mask`: где собирать статистику (белое = учитывать).  
- `apply_mask`: где применять коррекцию (белое = применить).  

## Как оценивать качество
- **difference**: чем темнее, тем ближе.  
- **match_json.stats**: mean/std по каналам до/после.

## Советы
- Если результат «перекручен», уменьшите `strength`.  
- Если цвета всё равно не совпадают — попробуйте `quality` или `perceptual`.  
- При сложных сценах используйте `match_mask`, чтобы исключить фон/шум.
