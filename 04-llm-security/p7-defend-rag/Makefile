# defend-rag. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help defend train test clean
help:
	@echo "make defend  - train detector (if needed) + replay attacks + write money plot & metrics.json"
	@echo "make train   - just train the from-scratch injection detector -> models/"
	@echo "make test    - fast smoke tests (-m 'not slow')"
	@echo "make clean   - remove trained detector + generated figures/metrics"

defend: ## the one-shot money target (before/after ASR)
	$(PY) scripts/defend.py

train:
	$(PY) scripts/train_detector.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f models/*.joblib results/figures/*.png results/metrics.json
