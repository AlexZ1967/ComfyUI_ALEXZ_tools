# Changelog — ALEXZ_tools

## 0.6.4 — 2026-01-27
- Color Match: добавлен режим `perceptual_vgg` (VGG19, без ручной загрузки весов).
- Добавлен режим `perceptual_adain` (автодонлоад весов AdaIN).
- GPU torch-only реализация, подсказки по скорости.
- Color Match Guide вынесен в отдельный файл.
- Добавлены stub-режимы perceptual_ltct/lut3d/unet (требуют весов).
- README упрощён, гайды вынесены в отдельные файлы с линками из описаний нод.
- Логируемая загрузка нод при старте (видно, что загрузилось/почему упало).
- Исправлен perceptual_vgg: принудительное отключение inference_mode для корректного backward (фикc падения "does not require grad").
- Добавлен tqdm‑прогресс в perceptual_vgg и лог скачивания весов AdaIN/VGG (видно процесс загрузки/оптимизации).

## 0.6.3 — 2026-01-27
- Torch-only pipeline (GPU/CPU), perf tips для Color Match.

## 0.6.2 — 2026-01-27
- Waveform/parade выходы, ΔE метрики и heatmap, расширенный JSON.

## 0.6.1 — 2026-01-27
- Режимы PCA/strength, экспорт 1D LUT; обновлена документация.

## 0.6.0 — 2026-01-27
- Новая нода Color Match To Reference.

## 0.5.4–0.2.0 (2026-01-21…2026-01-19)
- Улучшения VideoInpaintWatermark (ProPainter/E2FGVI, кеш, полноразмерные кадры).
- Улучшения ImageAlignOverlayToBackground (LAB режимы, min_matches/min_inliers).
- JsonDisplayAndSave объединён и упрощён.

## 0.3.0–0.2.0 (2026-01-13)
- Пример workflow: восстановление фото + Align Overlay.
- Опции use_color, rotation lock, independent scaling.
