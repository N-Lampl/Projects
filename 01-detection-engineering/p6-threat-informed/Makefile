# Threat-informed detection-as-code. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect test clean
help:
	@echo "make detect  - validate Sigma rules + map to ATT&CK + write heatmap & metrics.json"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures and metrics.json"

detect: ## the one-shot money target
	$(PY) scripts/build_coverage.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
