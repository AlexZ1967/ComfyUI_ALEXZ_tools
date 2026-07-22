# AGENTS Instructions

## Runtime Environment
- Always run local checks and tests inside Conda env `p313`.
- Preferred command prefix: `conda run -n p313`.

## Canonical Commands
- Docs check: `conda run -n p313 python utils/docs_check.py`
- Seam smoke tests: `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k seam_match`
- Full smoke tests: `conda run -n p313 pytest -q tests/test_smoke_nodes.py`
- All JS syntax checks: `make js-check-all`
- Module Node Picker behavior test: `make js-test`
- Full Python, JS, and docs validation: `make test`

## Rule
- If a command was run without `p313`, rerun it with `conda run -n p313` before reporting results.
