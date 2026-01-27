# Color Match To Reference — Practical Guide

## Быстрый старт
1. Подайте `reference` (эталон) и `image` (исправить).  
2. Mode: попробуйте `levels` (percentile 0.5–1%).  
3. Смотрите `delta_e` в `match_json.stats`: mean < 2, p95 < 5 — обычно «очень близко».  
4. Если оттенок/контраст не попал — попробуйте `lab_full`/`lab_cdf` или `perceptual_vgg` (качественнее) / `perceptual_adain` (мягко и быстро).  
5. Если есть артефакты — снизьте `strength` (0.6–0.8) или используйте более мягкий режим (`mean_std`, `adain`).  

## Режимы (кратко)
- **mean_std / linear** — быстрые, базовая подгонка.  
- **hist** — поканальная гистограмма, точнее, но шумнее.  
- **pca_cov** — цветовая матрица по ковариации.  
- **lab_l / lab_full / lab_l_cdf / lab_cdf** — работа в LAB, CDF‑варианты точнее для тонов.  
- **hsv_shift** — сдвиг оттенка/сатурации/яркости.  
- **perceptual_vgg** — оптимизирует 3×3+bias по VGG19 (качественно, медленнее).  
- **perceptual_adain** — одношаговая AdaIN, веса грузятся автоматически (быстро, мягко).  
- **perceptual_ltct / perceptual_lut3d / perceptual_unet** — места под глубокие модели; нужны веса (см. README).  

## Как оценивать качество
- **Цифры**: `delta_e` (mean/p95/under2/under5) в `match_json`.  
- **difference**: чем темнее, тем ближе.  
- **deltae_heatmap**: красные зоны — большие отклонения.  
- **waveform/parade**: полосы рефа и результата должны накладываться; разъезд = несоответствие яркости/контраста/каналов.  

## Маски
- `match_mask`: где собирать статистику/перцептуал (белое=учитывать). Полезно исключить шум/фон.  
- `apply_mask`: где применять коррекцию (белое=применить). Правьте только нужную область.  

## Производительность
- Всё на torch и на том же устройстве, что входы. GPU ускоряет все режимы.  
- Дешёвые: mean_std, linear, hsv_shift. Дороже: hist, pca_cov, lab_cdf, perceptual_vgg (самый тяжёлый).  
- Выключайте waveform/deltae_heatmap для максимальной скорости; уменьшайте `waveform_width` при больших картинках.  

## LUT
- Включите `export_lut=true` — LUT 1D (.cube) появится в `match_json.lut_1d_cube` (размер `lut_size`).  

## Где брать веса (deep)
- **VGG19**: подтягивается из torchvision автоматически.  
- **AdaIN**: автозагрузка в `models/color_match/adain/` с GitHub (naoto0804/pytorch-AdaIN).  
- **LTCT / LUT3D / UNet**: пока заглушки — положите веса в `models/color_match/<mode>/` и допишите loader (или дайте ссылку — подключим).  

## Если «не сработало»
- Нет улучшения: попробуйте другой режим (lab_full, perceptual_vgg/adain), увеличьте percentile (levels), снизьте strength.  
- Цвет «ломается»: уменьшите strength; используйте mask; возьмите более мягкий режим.  
- Шум/полосы на waveform: включите log, снизьте gain, уменьшите width.  
