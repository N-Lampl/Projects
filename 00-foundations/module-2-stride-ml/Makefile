# STRIDE-ML threat model. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help detect pytm test clean
help:
	@echo "make detect  - run the STRIDE analysis: writes docs/threat-model.md + figure + metrics.json"
	@echo "make pytm    - same, using pytm if installed (falls back to stdlib otherwise)"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove generated docs/figures/metrics"

detect: ## the one-shot money target (stdlib-only, offline)
	$(PY) scripts/threatmodel.py

pytm:
	$(PY) scripts/threatmodel.py --pytm

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f docs/threat-model.md results/figures/*.png results/metrics.json
