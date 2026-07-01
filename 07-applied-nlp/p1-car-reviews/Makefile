# Car-reviews NLP. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run run-full test clean
help:
	@echo "make run       - fast --sample run (nlptown, HF) -> figures + metrics.json"
	@echo "make run-full  - FULL 36,984-review run (the committed artifact; ~30-75 min CPU)"
	@echo "make test      - fast offline smoke tests (stub backend, -m 'not slow')"
	@echo "make clean     - remove generated figures + metrics.json"
	@echo "  overrides:  make run ARGS='--model sst2 --summaries abstractive'"

run: ## fast sampled run (downloads the HF model on first use)
	$(PY) scripts/run_analysis.py $(ARGS)

run-full: ## the committed artifact: every review
	$(PY) scripts/run_analysis.py --full $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
