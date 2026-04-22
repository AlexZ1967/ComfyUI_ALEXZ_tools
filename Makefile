CONDA_ENV ?= p313
CONDA_RUN = conda run -n $(CONDA_ENV)
BASELINE_OUTPUT ?= baseline.json

.PHONY: docs-check seam-smoke smoke js-check save-baseline

docs-check:
	$(CONDA_RUN) python utils/docs_check.py

seam-smoke:
	$(CONDA_RUN) pytest -q tests/test_smoke_nodes.py -k seam_match

smoke:
	$(CONDA_RUN) pytest -q tests/test_smoke_nodes.py

js-check:
	$(CONDA_RUN) node --check web/widget_visibility_profiles.js

save-baseline:
	bash scripts/save_baseline.sh --output "$(BASELINE_OUTPUT)"
