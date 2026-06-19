# llm-privacy (canary exposure). Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect test clean
help:
	@echo "make detect  - train tiny char-LM on canary-laced corpus + measure exposure + write figures & metrics.json"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove trained weights and generated figures/metrics"

detect: ## the one-shot money target
	$(PY) scripts/detect_memorization.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf models/*.pt results/figures/*.png results/metrics.json
