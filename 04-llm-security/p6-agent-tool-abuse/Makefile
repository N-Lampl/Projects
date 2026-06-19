# Agent tool-abuse. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack test clean
help:
	@echo "make attack  - run the tool-abuse suite + guardrail eval -> figure & metrics.json"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures + metrics.json"
	@echo ""
	@echo "  default provider is the offline MOCK LLM. Optional real LLM:"
	@echo "  make attack ARGS='--provider anthropic'   (needs anthropic + ANTHROPIC_API_KEY)"

attack: ## the one-shot money target
	$(PY) scripts/run_attack.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf results/figures/*.png results/metrics.json
