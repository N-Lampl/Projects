# CAPSTONE-appsec-ci. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run gate gate-remediated dashboard report test clean
help:
	@echo "make run             - run the full pipeline: figures + report + metrics.json"
	@echo "make gate            - run the CI gate vs the VULNERABLE target (exits 1 = FAIL, expected)"
	@echo "make gate-remediated - run the CI gate vs the REMEDIATED target (exits 0 = PASS)"
	@echo "make test            - fast smoke tests (-m 'not slow')"
	@echo "make clean           - remove generated figures / report / metrics / history"

run: ## produce all artifacts (always exits 0)
	$(PY) scripts/run_pipeline.py --full

gate: ## the CI gate against the vulnerable RAG -- FAILS the build by design
	$(PY) scripts/run_pipeline.py --enforce --threshold 0.0

gate-remediated: ## the same gate against the hardened RAG -- PASSES
	$(PY) scripts/run_pipeline.py --remediated --enforce --threshold 0.0 --full

dashboard: run  ## alias: the dashboard figures are produced by `run`

report: run     ## alias: the threat-model report is produced by `run`

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/metrics.json results/threat_model_report.md results/history.jsonl \
	      results/figures/*.png
	touch results/figures/.gitkeep
