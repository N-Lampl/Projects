# FGSM-on-MNIST. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack train test clean
help:
	@echo "make attack  - train (if needed) + run FGSM sweep + write figures & metrics.json"
	@echo "make train   - just train the SmallCNN and save weights"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean    - remove data/, models/ and generated figures"

attack: ## the one-shot money target
	$(PY) scripts/run_fgsm.py

train:
	$(PY) scripts/train_mnist.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics.json
