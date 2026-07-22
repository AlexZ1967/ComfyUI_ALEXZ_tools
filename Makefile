CONDA_ENV ?= p313
CONDA_RUN = conda run -n $(CONDA_ENV)
BASELINE_OUTPUT ?= baseline.json

.PHONY: docs-check seam-smoke smoke python-test js-check js-check-all js-test test save-baseline

docs-check:
	$(CONDA_RUN) python utils/docs_check.py

seam-smoke:
	$(CONDA_RUN) pytest -q tests/test_smoke_nodes.py -k seam_match

smoke:
	$(CONDA_RUN) pytest -q tests/test_smoke_nodes.py

python-test:
	$(CONDA_RUN) pytest -q

js-check: js-check-all

js-check-all:
	$(CONDA_RUN) node scripts/check_js_syntax.mjs web

js-test:
	$(CONDA_RUN) node tests/js/test_module_node_picker_frontend_behavior.mjs

test: docs-check python-test js-check-all js-test

save-baseline:
	bash scripts/save_baseline.sh --output "$(BASELINE_OUTPUT)"
