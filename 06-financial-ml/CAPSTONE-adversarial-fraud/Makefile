# Adversarial-fraud capstone. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test clean
help:
	@echo "make run   - train fraud model + run evasion attack + harden + write figures, metrics.json, REPORT_CARD.md"
	@echo "make test  - fast smoke tests (-m 'not slow')"
	@echo "make clean - remove generated figures, metrics.json, report card, models"

run: ## the one-shot money target
	$(PY) scripts/run_capstone.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf results/figures/*.png results/metrics.json results/REPORT_CARD.md models/*.pkl
