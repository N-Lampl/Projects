# garak-style LLM scan. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help scan parse test clean
help:
	@echo "make scan              - run offline built-in probes vs the p4 RAG mock -> figures + metrics.json"
	@echo "make parse REPORT=path - parse an EXISTING (e.g. real garak) report.jsonl with the same code"
	@echo "make test              - fast smoke tests (-m 'not slow')"
	@echo "make clean             - remove generated report/figures/metrics"

scan: ## the one-shot money target (offline)
	$(PY) scripts/run_scan.py

parse: ## parse a real garak report: make parse REPORT=/path/to/report.jsonl
	$(PY) scripts/run_scan.py --report $(REPORT)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/report.jsonl results/metrics.json results/figures/*.png
