# OWASP Top 10 web-appsec lab. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help lab up down test clean
help:
	@echo "make lab              - run all OWASP probes OFFLINE (mock, no docker) + write findings/metrics/figure"
	@echo "make lab ARGS=--live  - run the probes against a live local Juice Shop (docker compose up first)"
	@echo "make up               - docker compose up the Juice Shop container (localhost:3000)"
	@echo "make down             - docker compose down the container"
	@echo "make test             - fast smoke tests (-m 'not slow')"
	@echo "make clean            - remove generated figures, findings.json, metrics.json"

lab: ## the one-shot target; runs offline by default
	$(PY) scripts/run_lab.py $(ARGS)

up:
	docker compose up -d

down:
	docker compose down

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/findings.json results/metrics.json
