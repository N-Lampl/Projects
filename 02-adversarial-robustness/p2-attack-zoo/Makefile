# attack-zoo. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack train test clean
help:
	@echo "make attack  - train (if needed) + run PGD/C&W/DeepFool + write figures & metrics.json"
	@echo "make train   - just train the SmallCNN target and save weights"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove data/, models/ and generated figures"
	@echo ""
	@echo "  ARGS examples:  make attack ARGS='--source cifar10'   (optional real-data path)"

attack: ## the one-shot comparison target
	$(PY) scripts/run_attacks.py $(ARGS)

train:
	$(PY) scripts/train_target.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/CIFAR* data/cifar* data/MNIST models/*.pt results/figures/*.png results/metrics.json
