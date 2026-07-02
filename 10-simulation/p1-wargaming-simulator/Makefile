# DOW war-gaming simulator. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test sim montecarlo dashboard clean
help:
	@echo "make run        - run the Monte Carlo experiment -> results/figures + metrics.json"
	@echo "make test       - fast offline smoke tests (-m 'not slow')"
	@echo "make sim        - play one battle to the terminal (ASCII board + combat log)"
	@echo "make montecarlo - win probability + CI for a scenario/policy matchup"
	@echo "make dashboard  - launch the Streamlit dashboard"
	@echo "make clean      - remove generated figures + metrics.json"
	@echo "  overrides:  make run ARGS='--n 5000 --seed 7'"

run: ## simulate battles, estimate win probs + CIs, sweep sensitivity -> metrics.json + figures
	$(PY) scripts/run_analysis.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

sim:
	$(PY) scripts/dow.py sim $(ARGS)

montecarlo:
	$(PY) scripts/dow.py montecarlo $(ARGS)

dashboard:
	$(PY) -m streamlit run scripts/dashboard.py

clean:
	rm -f results/figures/*.png results/metrics.json
