# Silent Film Finish — Guide

## Назначение
Финальный стилистический слой для имитации немой пленочной проекции: ЧБ/тонировка, flicker, gate weave, мягкость, легкое синхронизированное гуляние фокуса и зерно.

## Когда использовать
- Уже есть нужная каденция, но картинка все еще выглядит слишком современной.
- Нужно добавить механическую нестабильность проектора и фотохимическую “материальность”.
- Нужен завершающий B&W/tint pass для 1920s эстетики.

## Минимальный сценарий (3 шага)
1. Подайте батч кадров в `image`, лучше уже после `Silent Film Cadence`.
2. Если используете `Silent Film Cadence`, подключите его `cadence_json` в optional вход finish-ноды.
3. Поставьте `tone_mode=neutral_bw` или `warm_print`.
4. Настройте `flicker_strength`, `gate_weave_px` и `grain_strength`, затем проверьте `finish_json`.

## Параметры
| Параметр | Что делает | Рекомендация |
|---|---|---|
| `image` | Батч кадров в формате THWC | Обычно после каденции |
| `tone_mode` | Режим ЧБ/тонировки | `neutral_bw` для чистой стилизации, `warm_print` для теплой копии |
| `contrast` | Глобальный контраст | `0.9-1.0` обычно убедительнее, чем жесткий modern contrast |
| `midtone_gamma` | Гамма средних тонов | `0.9-1.0` для чуть приподнятых midtones |
| `black_lift` | Подъем черного | `0.02-0.05` для менее цифровой картинки |
| `highlight_rolloff` | Компрессия светов | `0.25-0.45` как старт |
| `softness` | Легкая оптическая мягкость | `0.15-0.35` |
| `focus_drift_strength` | Очень деликатное пульсирующее изменение резкости: часть кадров чуть резче, часть чуть мягче | `0.05-0.12`, чтобы получить старую оптику, а не digital autofocus hunting |
| `flicker_strength` | Быстрый кадр-кадр flicker | `0.03-0.07` обычно достаточно |
| `breathing_strength` | Медленное плавание яркости | `0.015-0.04` |
| `gate_weave_px` | Трансляционный jitter кадра в пикселях | `0.7-1.5` для HD/2K старта |
| `grain_strength` | Сила зерна | `0.02-0.05` |
| `grain_size` | Крупность зерна | `2-3` для умеренного print look |
| `seed` | Seed для weave/flicker/grain | Меняйте для другой механики |

## Decision helper
- Нужна более “реальная проекция” -> увеличьте `flicker_strength` и `gate_weave_px`.
- Нужно, чтобы мерцание, дрожание и гуляние фокуса были связаны с ручной скоростью съемки -> обязательно подключайте `cadence_json` из `Silent Film Cadence`.
- Картинка слишком чистая -> добавьте `grain_strength` и немного `softness`.
- Нужен едва заметный "то в фокусе, то чуть мягче" -> поднимайте `focus_drift_strength`, обычно без выхода выше `0.12`.
- Нужен архивный теплый print -> `tone_mode=warm_print` или `sepia_print`.
- Нужен более холодный nitrate mood -> `tone_mode=cool_nitrate`.

## Интерпретация выходов
- `image`: готовый стилизованный батч.
- `finish_json`: JSON-диагностика с режимом тонировки, jitter и flicker preview.
- `finish_json.sync_mode`: `cadence_locked`, если эффекты успешно синхронизировались по `cadence_json`; иначе причина fallback (`none`, `invalid_json`, `frame_mismatch` и т.д.).
- `finish_json.focus_preview`: первые signed значения focus drift после синхронизации по cadence. Отрицательные кадры чуть резче базового уровня, положительные чуть мягче.
- `finish_json.gate_x_preview` / `finish_json.gate_y_preview`: первые смещения кадра по осям.
- `finish_json.exposure_preview`: первые значения глобальной экспозиции после flicker/breathing.

Пример:
```json
{
  "tone_mode": "neutral_bw",
  "sync_mode": "cadence_locked",
  "focus_preview": [-0.041, 0.018, 0.063],
  "gate_x_preview": [0.21, -0.34, 0.08],
  "gate_y_preview": [-0.11, 0.27, 0.04],
  "exposure_preview": [0.98, 1.03, 0.97]
}
```

## Типовые ошибки и решения
- Эффект почти незаметен: увеличьте `flicker_strength`, `gate_weave_px` и `grain_strength`.
- "Пульс фокуса" не читается: поднимите `focus_drift_strength`, но маленькими шагами по `0.005-0.01`.
- Слишком “грязно” и карикатурно: уменьшите `grain_strength` и `gate_weave_px`.
- Слишком похоже на современный hunting autofocus: уменьшите `focus_drift_strength` и `softness`.
- Слишком цифровые света: увеличьте `highlight_rolloff` и немного `softness`.
- Слишком серо и плоско: поднимите `contrast` до `1.0-1.1` и уменьшите `black_lift`.

## Производительность
- Нода полностью работает на torch и подходит для обычных видео-батчей ComfyUI.
- Самые дорогие части здесь: `gate_weave` через `grid_sample` и крупное зерно на больших кадрах.
- На практике этот finish легче, чем optical-flow подходы, и хорошо подходит как финальный pass после каденции.
