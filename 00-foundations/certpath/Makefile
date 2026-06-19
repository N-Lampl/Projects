# certpath. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help coverage test clean
help:
	@echo "make coverage - read syllabus.json -> write results/figures + metrics.json"
	@echo "make test     - fast smoke tests (-m 'not slow')"
	@echo "make clean    - remove generated figure & metrics.json"

coverage: ## the one-shot money target
	$(PY) scripts/cert_coverage.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
