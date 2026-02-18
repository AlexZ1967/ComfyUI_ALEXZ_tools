# Color Match Examples

This folder contains deterministic before/after visual examples used by:

- `guides/GUIDE_COLOR_MATCH_DETAILED.md`

## Regeneration

Run from repository root.

Generate synthetic baseline (`case01`):

```bash
python guides/assets/color_match_examples/generate_case01.py
```

Generate real pair example (`case02`) from provided files:

```bash
python guides/assets/color_match_examples/generate_case01.py \
  --case case02 \
  --image guides/assets/color_match_examples/before_image.jpg \
  --reference guides/assets/color_match_examples/reference.jpg
```

Outputs are written to:

- `guides/assets/color_match_examples/case01/`
- `guides/assets/color_match_examples/case02/`
