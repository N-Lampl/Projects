# Adversarial-IDS capstone. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")
ARGS ?=

.PHONY: help attack test clean
help:
	@echo "make attack  - train IDS + craft constrained evasions + harden + write report card"
	@echo "make attack ARGS=--use-art        - use IBM ART if installed (else falls back)"
	@echo "make attack ARGS='--defense ensemble' - harden with a diverse ensemble instead"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated figures, report card, metrics.json, models"

attack: ## the one-shot money target
	$(PY) scripts/run_capstone.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf models/*.pkl results/figures/*.png results/report_card.md results/metrics.json
