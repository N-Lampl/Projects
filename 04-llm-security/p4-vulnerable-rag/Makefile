# Vulnerable-RAG lab target. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test clean
help:
	@echo "make run    - run the RAG target with the offline MOCK provider + write figures & metrics.json"
	@echo "make test   - fast smoke tests (-m 'not slow')"
	@echo "make clean  - remove generated figures + metrics.json"

run: ## the one-shot target: exercise the app + map its attack surface
	$(PY) scripts/run_rag.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf results/figures/*.png results/metrics.json
