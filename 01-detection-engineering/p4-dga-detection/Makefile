# DGA-detection. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect lstm test clean
help:
	@echo "make detect  - train detector on synthetic domains + write figures & metrics.json"
	@echo "make lstm    - also train the optional torch char-LSTM (detect --lstm)"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures, metrics, and saved models"

detect: ## the one-shot money target
	$(PY) scripts/run_detection.py

lstm:
	$(PY) scripts/run_detection.py --lstm

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json models/*.pkl models/*.pt
