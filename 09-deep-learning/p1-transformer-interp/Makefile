# Transformer interpretability. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test clean
help:
	@echo "make run    - train tiny transformer + 3 interp analyses -> results/figures + metrics.json"
	@echo "make test   - fast offline smoke tests (-m 'not slow')"
	@echo "make clean  - remove generated figures + metrics.json"
	@echo "  overrides:  make run ARGS='--steps 600 --seed 7'"

run: ## train + induction score + logit lens + activation patching -> figures + metrics.json
	$(PY) scripts/run_analysis.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
