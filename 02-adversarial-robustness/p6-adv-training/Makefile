# PGD adversarial training. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")
ARGS ?=

.PHONY: help run train test clean
help:
	@echo "make run     - train both models (if needed) + PGD sweep + figures & metrics.json"
	@echo "make train   - train the standard + adversarial models and save weights"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove data/, models/ and generated figures"
	@echo "  pass ARGS=--real to use real MNIST (needs torchvision); default is synthetic/offline"

run: ## the one-shot money target
	$(PY) scripts/run_compare.py $(ARGS)

train:
	$(PY) scripts/train_models.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics.json
