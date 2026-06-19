# Log-UEBA. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect test clean
help:
	@echo "make detect  - generate synthetic auth events + run UEBA detector + write figures & metrics.json"
	@echo "make detect ARGS=--autoencoder  - also run the optional torch autoencoder"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures + metrics.json"

detect: ## the one-shot money target
	$(PY) scripts/run_ueba.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
