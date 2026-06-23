# AML-typology-detection. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect test clean
help:
	@echo "make detect  - generate synthetic AML graph + run detectors + write figures & metrics.json"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures & metrics.json"

detect: ## the one-shot money target
	$(PY) scripts/detect_aml.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf results/figures/*.png results/metrics.json
