# DP-defenses capstone. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")
ARGS ?=

.PHONY: help run overnight test clean
help:
	@echo "make run            - train DP targets + re-run MIA & extraction + write figures/metrics.json"
	@echo "make run ARGS=--full- flagship sweep {inf, 3, 1} on the larger pool"
	@echo "make overnight      - finer epsilon grid {inf,8,3,1,0.5}, more shadows -> results/batches/<ts>/"
	@echo "make test           - fast smoke tests (-m 'not slow')"
	@echo "make clean          - remove generated figures, metrics.json, batches/"

run: ## the one-shot money target (DEFAULT: fast 2-point smoke sweep)
	$(PY) scripts/run_tradeoff.py $(ARGS)

overnight:
	$(PY) scripts/overnight_batch.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf results/figures/*.png results/metrics.json results/batches
	mkdir -p results/figures && touch results/figures/.gitkeep
