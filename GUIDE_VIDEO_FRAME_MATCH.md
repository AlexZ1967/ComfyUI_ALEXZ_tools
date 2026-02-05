# Find Closest Video Frame — Guide

## Назначение
Поиск кадра в видео, максимально похожего на заданное изображение. Доступны метрики: MSE, SSIM, LPIPS.

## Зависимости
- При `max_frames > 0` нода читает хвост видео через `ffmpeg`.
- `ffmpeg` должен быть доступен в `PATH`.
- Установка:
  Linux: `sudo apt install ffmpeg`
  Windows: `choco install ffmpeg`
  macOS: `brew install ffmpeg`

## Быстрый старт
1) `image` — целевой кадр.  
2) `video` — выберите файл из input/.  
3) `max_frames=0` чтобы пройти всё видео, или ограничьте (напр. 500), чтобы анализировать только последние кадры.  
4) `metric=mse` для быстрого точного сравнения, `ssim` устойчивее к яркости, `lpips_alex/lpips_vgg` — более «перцептивные».  
Получите `best_frame`, `best_frame_number`, `scores_json`.

## Параметры
- `max_frames`: количество последних кадров для анализа (0 = без лимита).  
- `metric`: `mse` / `ssim` / `lpips_alex` / `lpips_vgg`.  
- `normalize`: `none` / `mean_std` / `linear` / `hist` (приводит кадр к цвету референса перед сравнением).  

## Выходы
- `best_frame`: кадр, наиболее похожий на картинку.  
- `best_frame_number`: номер кадра (с нуля).  
- `scores_json`: объект `{metric, normalize, scores}` с первыми 500 оценками `{index, score}`.  
  Для `lpips_alex/lpips_vgg` используется двухпроходный поиск: coarse-pass (`mse`) + refine-pass (`lpips`) по top-k кандидатам; в JSON добавляются поля `search`, `coarse_metric`, `coarse_max_side`, `refine_candidates`, `refined_scores`.

## Рекомендации
- Если видео большое, ограничьте `max_frames`, чтобы ускорить поиск.  
- Референс автоматически приводится к размеру видео (если размеры отличаются).  
- Для различий по яркости/контрасту используйте `ssim` или `normalize=mean_std`.  
- Для сложных случаев используйте `lpips_alex` (быстрее) или `lpips_vgg` (качественнее).
- Для `lpips_*` теперь автоматически включается ускоренный двухпроходный поиск (обычно быстрее полного LPIPS по всем кадрам).
- При `max_frames > 0` без `ffmpeg` нода завершится с ошибкой и подсказкой по установке.
