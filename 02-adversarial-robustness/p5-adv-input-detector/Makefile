# adv-input-detector. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect test clean
help:
	@echo "make detect  - train target (if needed) + build FGSM examples + train detector + write figures & metrics.json"
	@echo "make detect ARGS=\"--dataset mnist\"  - optional: use real MNIST (~11MB download)"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove data/, models/ and generated figures"

detect: ## the one-shot money target
	$(PY) scripts/run_detector.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics.json
