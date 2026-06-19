# crypto-lab. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run test clean
help:
	@echo "make run    - run hashing + AES(ECB/GCM) + RSA demos; write figures & metrics.json"
	@echo "make test   - fast smoke tests (-m 'not slow')"
	@echo "make clean  - remove generated figures and metrics.json"

run: ## the one-shot money target
	$(PY) scripts/run_crypto_lab.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json
