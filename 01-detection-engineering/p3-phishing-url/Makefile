# Phishing-URL detector. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")
ARGS ?=

.PHONY: help detect test clean
help:
	@echo "make detect              - train detector on SYNTHETIC URLs + write ROC/figures & metrics.json"
	@echo "make detect ARGS=...      - pass flags, e.g. ARGS=\"--model rf\" or ARGS=\"--data phiusiil\""
	@echo "make test                - fast smoke tests (-m 'not slow')"
	@echo "make clean               - remove generated figures + metrics.json"

detect: ## the one-shot money target (offline by default)
	$(PY) scripts/detect.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
