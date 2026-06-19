# Transfer + black-box on MNIST-style models. Auto-detects uv; falls back to python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack train test clean
help:
	@echo "make attack  - train two models (if needed) + transfer + black-box attacks + figures & metrics.json"
	@echo "make train   - just train the CNN surrogate and MLP target, save weights"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove data/, models/ and generated figures/metrics"

attack: ## the one-shot money target
	$(PY) scripts/run_attacks.py

train:
	$(PY) scripts/train_models.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics.json
