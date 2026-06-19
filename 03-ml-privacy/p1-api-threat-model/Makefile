# api-threat-model. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help demo serve serve-fastapi test clean
help:
	@echo "make demo          - run the abuse-test suite + write figures & metrics.json"
	@echo "make serve         - run the hardened endpoint (stdlib transport, no fastapi)"
	@echo "make serve-fastapi - run via optional FastAPI/uvicorn (needs: pip install fastapi uvicorn)"
	@echo "make test          - fast smoke tests (-m 'not slow')"
	@echo "make clean         - remove generated figures & metrics.json"

demo: ## the one-shot money target
	$(PY) scripts/run_abuse_tests.py

serve:
	$(PY) scripts/serve.py

serve-fastapi:
	$(PY) scripts/serve.py --fastapi

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
