# OWASP LLM Top 10 mapping. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help map test clean
help:
	@echo "make map   - build owasp_map.json + coverage figure + metrics.json"
	@echo "make test  - fast smoke tests (-m 'not slow')"
	@echo "make clean - remove generated outputs"

map:
	$(PY) scripts/build_owasp_map.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json results/owasp_map.json
