# Model inversion + attribute inference. Auto-detects uv; falls back to python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run invert attribute test clean
help:
	@echo "make run        - run BOTH attacks + write merged results/metrics.json (the money target)"
	@echo "make invert     - model inversion only (reconstruction figure + metrics)"
	@echo "make attribute  - attribute inference only (sweep figure + metrics)"
	@echo "make test       - fast smoke tests (-m 'not slow')"
	@echo "make clean      - remove data/, models/ and generated figures/metrics"

run: ## the one-shot money target: both privacy attacks
	$(PY) scripts/run_all.py

invert:
	$(PY) scripts/run_inversion.py

attribute:
	$(PY) scripts/run_attribute.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/MNIST models/*.pt results/figures/*.png results/metrics*.json
