# Refusal-direction interpretability. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test clean
help:
	@echo "make run    - run the synthetic extract->ablate->measure pipeline + write figures & metrics.json"
	@echo "make test   - fast smoke tests (-m 'not slow')"
	@echo "make clean  - remove generated figures & metrics.json"

run: ## the one-shot offline analysis target (no weights downloaded)
	$(PY) scripts/run_analysis.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
