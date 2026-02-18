# Seam Match To Reference — Guide

## Назначение
Подгонка `image` к `reference` с приоритетом минимального визуального стыка (минимальный diff), а не художественного color-look.

## Когда использовать
- Два очень похожих кадра в монтаже дают заметный "дрыжок" по цвету/тону.
- Нужно свести кадры так, чтобы разница была минимальна на всей картинке.
- Глобальная подгонка важнее, чем стилизация.

## Минимальный сценарий (3 шага)
1. Подайте `reference` и `image`.
2. Выберите подходящий вариант ноды (`v1`/`v2`/`v3`/`v4`) или используйте универсальную legacy-ноду с `seam_model`.
3. Проверьте `seam_json.quality` и `difference` в вашем пайплайне.

Варианты нод:
- `Seam Match v1 Affine` (`ImageSeamMatchV1AffineToReference`) — быстрый глобальный affine.
- `Seam Match v2 Tonal` (`ImageSeamMatchV2TonalToReference`) — тональные диапазоны.
- `Seam Match v3 Hybrid` (`ImageSeamMatchV3HybridToReference`) — global + tonal residual.
- `Seam Match v4 LUT` (`ImageSeamMatchV4LUTToReference`) — 3D LUT, максимальная точность (медленнее).
- `Seam Match To Reference` (`ImageSeamMatchToReference`) — универсальная legacy-нода с `seam_model`.

## Параметры
| Параметр | Что делает | Рекомендация |
|---|---|---|
| `reference` | Эталон для подгонки | Кадр, к которому примыкает стык |
| `image` | Что корректируем | Кадр, который нужно "подтянуть" |
| `strength` | Сила применения [0..1] | 0.8-1.0 |
| `alpha` | Сохраняется автоматически при наличии на входе | Без отдельного параметра |
| `compute_device` | Устройство расчета (`auto`/`cpu`/`cuda`) | `auto` или `cuda` при наличии GPU |
| `color_space` | Пространство оптимизации (`rgb`/`oklab`) | `oklab` по умолчанию |
| `downscale_long_side` | Разрешение оптимизации (`as_is`, `1080p`, `720p`, `480p`) | `720p` |
| `seam_model` | Модель трансформации (`v2_tonal`/`v3_hybrid`/`v4_lut`/`v1_affine`) | Только в legacy-ноде |
| `steps` | Шаги оптимизации | 25-60 |
| `lr` | Скорость обучения | 0.02-0.08 |
| `w_mse` | Вес robust MSE | 1.0 |
| `w_ssim` | Вес SSIM-терма | 0.1-0.3 |
| `w_grad` | Вес градиентного терма | 0.05-0.2 |
| `reg_weight` | Регуляризация transform | 0.0005-0.005 |
| `robust_delta` | Порог robust-loss | 0.005-0.02 |
| `hybrid_residual_strength` | Сила tonal-residual в `v3_hybrid` | 0.8-1.2 |
| `hybrid_residual_reg` | Штраф амплитуды residual в `v3_hybrid` | 0.0005-0.005 |
| `hybrid_coherence_reg` | Штраф расхождения residual-бэндов в `v3_hybrid` | 0.0002-0.003 |
| `lut_size` | Размер куба 3D LUT в `v4_lut` | 25 (17/25/33) |
| `lut_identity_reg` | Штраф отклонения LUT от identity/init в `v4_lut` | 0.005-0.03 |
| `lut_smooth_reg` | Штраф гладкости LUT в `v4_lut` | 0.01-0.05 |
| `lut_lr_scale` | Множитель lr для LUT-ветки в `v4_lut` | 0.2-0.6 |

## Decision helper
- Нужна максимальная скорость: `downscale_long_side=480p`, `steps=20-30`.
- Нужен баланс: `downscale_long_side=720p`, `steps=35-45`.
- Нужна максимальная точность стыка: `downscale_long_side=1080p` или `as_is`, `steps=50-80`.
- Есть GPU: `compute_device=cuda` для ускорения.
- Если плохо ловит нелинейные сдвиги (color balance по диапазонам): `seam_model=v2_tonal`.
- Если в паре есть и глобальный сдвиг, и нелинейный тональный сдвиг: `seam_model=v3_hybrid`.
- Если нужна максимальная точность на сложных нелинейных кейсах: `seam_model=v4_lut` (медленнее).
- Если нужна максимально быстрая и предсказуемая глобальная подгонка: `seam_model=v1_affine`.

## Интерпретация выходов
- `matched_image`: скорректированная картинка.
- `seam_json`: JSON с параметрами оптимизации, матрицей transform и метриками.

Пример:
```json
{
  "mode": "seam_match:oklab",
  "optimization": {"downscale_long_side": "720p", "seam_model": "v3_hybrid", "steps": 40, "loss_final": 0.0042},
  "quality": {
    "before": {"mse": 0.012, "ssim": 0.84},
    "after": {"mse": 0.006, "ssim": 0.91}
  }
}
```

## Типовые ошибки и решения
- Недостаточная подгонка: увеличьте `steps`, `w_mse`, `w_ssim`.
- Переизменение картинки: уменьшите `strength` или увеличьте `reg_weight`.
- Медленно на CPU: используйте `480p`/`720p` и меньше `steps`.

## Производительность
- Самый сильный фактор: `downscale_long_side` + `steps`.
- Для 2K обычно хватает `720p` + `35-45` шагов.
- `as_is` может быть заметно медленнее, особенно на CPU.
