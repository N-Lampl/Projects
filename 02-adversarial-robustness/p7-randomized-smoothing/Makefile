# Randomized-smoothing certification. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help certify train test clean
help:
	@echo "make certify - train (if needed) + certify a subset + write figures & metrics.json"
	@echo "make train   - just train the noise-augmented base model and save weights"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove data/, models/ and generated figures"

certify: ## the one-shot money target (default: synthetic data, offline)
	$(PY) scripts/run_certify.py $(ARGS)

train:
	$(PY) scripts/train_base.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics.json
