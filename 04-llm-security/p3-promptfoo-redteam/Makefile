# promptfoo-redteam. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help redteam promptfoo parse test clean
help:
	@echo "make redteam   - OFFLINE: run the pure-python OWASP-LLM harness vs p4 + write figures & metrics.json"
	@echo "make promptfoo - OPTIONAL: run the real promptfoo owasp:llm redteam (needs node/npx)"
	@echo "make parse     - parse promptfoo/results.json -> metrics.json + chart"
	@echo "make test      - fast smoke tests (-m 'not slow')"
	@echo "make clean     - remove generated figures, metrics.json and promptfoo output"

redteam: ## the one-shot offline money target
	$(PY) scripts/run_redteam.py

promptfoo: ## requires node + npx; runs the eval-as-code path
	cd promptfoo && npx promptfoo@latest redteam run -o results.json

parse:
	$(PY) scripts/parse_promptfoo.py promptfoo/results.json

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json promptfoo/results.json
	touch results/figures/.gitkeep
