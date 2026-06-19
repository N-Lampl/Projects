# secure-ml-pipeline. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run poc detonate gate test clean
help:
	@echo "make run       - train -> safetensors -> scan -> sign/verify gate + figures & metrics.json"
	@echo "make poc       - build the BENIGN pickle PoC at runtime + opcode-scan it (no execution)"
	@echo "make detonate  - run the PoC inside a locked-down Docker sandbox (skips if no docker)"
	@echo "make gate      - run the CI gate locally (scan pickles + verify signature)"
	@echo "make test      - fast smoke tests (-m 'not slow')"
	@echo "make clean     - remove generated artifacts, figures, metrics.json, PoC"

run: ## the one-shot money target (offline-safe)
	$(PY) scripts/secure_pipeline.py

poc:
	$(PY) scripts/build_poc.py

detonate:
	bash scripts/run_in_docker.sh

gate: run
	$(PY) scripts/ci_gate.py --scan-pickles . --fail-on-malicious
	$(PY) scripts/ci_gate.py --verify models/model.safetensors --signature models/model.safetensors.sig

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf models/*.safetensors models/*.sig data/poc results/figures/*.png results/metrics.json
