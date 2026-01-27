# Color Match To Reference — Guide (кратко)

Основной гайд: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md)

Доп. выходы:
- `difference` — |matched - reference|
- `raw_difference` — |input - reference| (без коррекции) — помогает сравнить, сколько улучшили.

Быстрые режимы: `mean_std`, `linear`, `hsv_shift`, `perceptual_vgg_fast`, `perceptual_adain`.  
Качественный, но медленнее: `perceptual_vgg`.  
Stub (нужны веса): `perceptual_ltct/lut3d/unet`.
