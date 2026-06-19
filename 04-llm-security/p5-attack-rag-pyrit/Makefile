# Attack the p4 lab RAG. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack test clean
help:
	@echo "make attack - run the injection/exfil suite vs ../p4-vulnerable-rag, write metrics + figures"
	@echo "make test   - fast smoke tests (-m 'not slow')"
	@echo "make clean  - remove generated figures + metrics.json"

attack:
	$(PY) scripts/run_attacks.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
