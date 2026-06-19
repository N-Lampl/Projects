# Attack Atlas. Auto-detects uv; falls back to system python3. Stdlib-only build.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run navigator test clean
help:
	@echo "make run        - build results/atlas_map.json + metrics.json + coverage figure"
	@echo "make navigator  - regenerate navigator/portfolio_layer.json (ATT&CK Navigator layer)"
	@echo "make test       - fast smoke tests (-m 'not slow')"
	@echo "make clean      - remove generated results + the navigator layer"

run: ## the one-shot money target
	$(PY) scripts/build_atlas_map.py

navigator:
	$(PY) scripts/build_navigator_layer.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/atlas_map.json results/metrics.json results/figures/*.txt navigator/portfolio_layer.json
