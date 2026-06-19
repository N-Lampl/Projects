# LiRA membership inference. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack test clean
help:
	@echo "make attack  - run online LiRA (target + warm-started shadows) + write figures & metrics.json"
	@echo "make attack ARGS='--n-shadows 16'   - pass through extra args"
	@echo "make attack ARGS='--dataset fashion_mnist'  - optional torchvision path (downloads data)"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures, metrics.json and downloaded data"

attack: ## the one-shot money target
	$(PY) scripts/run_lira.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/FashionMNIST results/figures/*.png results/metrics.json
