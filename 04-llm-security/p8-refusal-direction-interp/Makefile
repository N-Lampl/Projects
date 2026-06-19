# Refusal-direction interpretability. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run abliterate test clean
help:
	@echo "make abliterate - REAL path: abliterate a small open-weight model (downloads weights, CPU-OK)"
	@echo "make run        - offline synthetic extract->ablate->measure (no download; for CI/quick demo)"
	@echo "make test       - fast smoke tests (-m 'not slow')"
	@echo "make clean      - remove generated figures & metrics.json"

# The committed results/ come from THIS target (real Qwen2.5-0.5B-Instruct run).
# Needs the optional extras:  pip install transformers accelerate
abliterate: ## real-model abliteration (the headline result)
	$(PY) scripts/run_real_abliteration.py

run: ## offline synthetic analysis (no weights downloaded) -- overwrites metrics.json with synthetic numbers
	$(PY) scripts/run_analysis.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json results/refusal_direction.json
