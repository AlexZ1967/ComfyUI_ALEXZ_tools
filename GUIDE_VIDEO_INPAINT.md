# Remove Static Watermark from Video — Guide

## Назначение
Удаление статичных водяных знаков/объектов на видео через инпейнтинг. Встроены ProPainter и E2FGVI (e2fgvi/e2fgvi_hq), веса автозагружаются.

## Быстрый старт
1) `mask`: бинарная маска (1 кадр или batch). Белое = удалить.  
2) `method=propainter`, `mask_dilates=8`, `flow_mask_dilates=8`, `ref_stride=10`, `neighbor_length=10`, `subvideo_length=80`, `raft_iter=20`, `fp16=enable`.  
3) Укажите `cache_dir` и `output_dir`, `output_name` (префикс), `video` (из input/), `preview_frame=0`.  
4) Получите `preview_image` и `transform_json`; файлы патчей пишутся в `output_dir`.

## Ключевые параметры
- `method`: propainter (баланс качество/скорость), e2fgvi (стандарт), e2fgvi_hq (лучше на произвольных разрешениях, дороже).  
- `mask_dilates` / `flow_mask_dilates`: расширение маски (8–12 для логотипов, 10–20 для крупных объектов).  
- `ref_stride` (E2FGVI): 5–15 — шаг опорных кадров.  
- `neighbor_length`: 5–15 — окно соседей.  
- `subvideo_length` (ProPainter): 40–120; уменьшайте при нехватке VRAM.  
- `raft_iter`: 10–30 — больше = точнее поток, медленнее.  
- `fp16`: enable экономит VRAM.  
- `throughput_mode`: enable — быстрее, но меньше очистки VRAM (осторожно с OOM).  
- `crop_padding`: 8–32 — контекст вокруг маски при pre-crop.  
- `color_match_mode`: none/mean_std/linear/hist/lab_l/lab_l_cdf/lab_full/lab_cdf — подгон цвета патча к окружению.
- `write_fullframes`: true, если нужны полноразмерные кадры с патчем.  
- `fullframe_prefix`: префикс для полноразмерных кадров.

## Рекомендации
- Маска: лучше чуть больше реального логотипа и с мягкими краями; используйте `mask_dilates` вместо рисования «толстой» маски.  
- Скорость: уменьшите `subvideo_length` (ProPainter), `neighbor_length`, `raft_iter`; включите `fp16`.  
- Качество: увеличьте `neighbor_length`, уменьшите `ref_stride`, повысите `raft_iter`, используйте `color_match_mode` (lab_full/lab_cdf).  
- Память: отключите `write_fullframes`, держите `throughput_mode=disable`, уменьшайте `subvideo_length`.

## Потоки и файлы
- Работает в стриминге: вход режется по маске (pre-crop) и кэшируется в `cache_dir` (`input_XXXX.png`, `mask_XXXX.png`).  
- Результат патчей (`output_name0000.png` + `transform.json`) — в `output_dir`.  
- `preview_image` — кадр с патчем поверх кэша (если `preview_frame>=0`), иначе 1×1.

## transform_json
Содержит позицию/масштаб обрезки в координатах полного кадра (подставляется для компоновки в NLE).  
status: ok или empty_mask (если маска пустая).

## Частые проблемы
- OOM: уменьшайте `subvideo_length`, `neighbor_length`, `raft_iter`; включите `fp16`; отключите `throughput_mode`.  
- Цветовой дрейф патча: примените `color_match_mode` (начните с `lab_l_cdf` или `lab_full`).  
- Пустой результат / status=empty_mask: проверьте маску (белая область должна быть >0).  
- Сильные артефакты на движении: увеличьте `neighbor_length`, `raft_iter`; попробуйте `method=e2fgvi_hq`. 
