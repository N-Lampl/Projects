# Root task runner for the monorepo.
# Each project also has its own Makefile (data / attack / figures).
#
# Tooling: `uv` is the recommended Python manager but is optional. These targets
# auto-detect it and fall back to plain python3/pip so the repo runs out of the box.

PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")
PIP := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip" || echo "python3 -m pip")

.PHONY: help setup lint fmt test precommit clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dev tooling (ruff, pytest, pre-commit)
	$(PIP) install ruff pytest pre-commit
	@command -v pre-commit >/dev/null 2>&1 && pre-commit install || true

lint: ## Lint all projects with ruff
	ruff check .

fmt: ## Auto-format with ruff
	ruff format .
	ruff check --fix .

test: ## Run fast tests only (skips @pytest.mark.slow); CI never trains
	$(PY) -m pytest -m "not slow" -q

precommit: ## Run all pre-commit hooks across the repo
	pre-commit run --all-files

clean: ## Remove caches and downloaded data/weights (keeps committed figures/metrics)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d \( -name data -o -name models \) -prune -exec sh -c \
		'find "$$0" -type f ! -name ".gitkeep" ! -name "README.md" -delete' {} \;
