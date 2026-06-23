# Transaction-anomaly. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")
ARGS ?=

.PHONY: help detect test clean
help:
	@echo "make detect  - generate synthetic stream + run detectors + write figures & metrics.json"
	@echo "make detect ARGS=--ae   - also run the optional torch autoencoder (sklearn fallback if no torch)"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures and metrics.json"

detect: ## the one-shot money target
	$(PY) scripts/run_detect.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
