# Bayesian hierarchical modeling. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test clean
help:
	@echo "make run    - fit the model + calibration study -> results/figures + metrics.json"
	@echo "make test   - fast offline smoke tests (-m 'not slow')"
	@echo "make clean  - remove generated figures + metrics.json"
	@echo "  overrides:  make run ARGS='--groups 12 --n-sims 200'"

run: ## Gibbs sampler + shrinkage + calibration -> figures + metrics.json
	$(PY) scripts/run_analysis.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
