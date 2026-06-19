# Model-extraction. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack train test clean
help:
	@echo "make attack  - train victim (if needed) + run budget sweep + defense + figures/metrics.json"
	@echo "make train   - just train the victim and save weights"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove models/, data/MNIST and generated figures/metrics"

attack: ## the one-shot money target (offline synthetic data by default)
	$(PY) scripts/run_extraction.py $(ARGS)

train:
	$(PY) scripts/train_victim.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics.json
