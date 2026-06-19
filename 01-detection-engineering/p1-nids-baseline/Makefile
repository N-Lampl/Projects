# NIDS baseline. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect test clean
help:
	@echo "make detect  - train the NIDS baseline + write SOC metrics & figures (synthetic)"
	@echo "make detect ARGS=--real        - benchmark on real NSL-KDD (see data/README.md)"
	@echo "make detect ARGS='--classifier xgb'  - optional xgboost path"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures + metrics.json"

detect: ## the one-shot run target
	$(PY) scripts/run_nids.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
