# Module-3 network labs. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run pcap test clean
help:
	@echo "make run    - generate the synthetic pcap (if needed) + parse it + write figures & metrics.json"
	@echo "make pcap   - just (re)generate data/synthetic.pcap"
	@echo "make test   - fast smoke tests (-m 'not slow')"
	@echo "make clean  - remove the generated pcap, figures and metrics.json"

run: ## the one-shot target: build + analyze the synthetic trace
	$(PY) scripts/analyze.py

pcap:
	$(PY) scripts/make_pcap.py

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f data/*.pcap results/figures/*.png results/metrics.json
