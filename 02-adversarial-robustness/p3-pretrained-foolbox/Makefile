# Inference-time attack on a (pretrained) model. Auto-detects uv; falls back to python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help attack pretrained test clean
help:
	@echo "make attack      - OFFLINE: train SmallCNN on synthetic data, run FGSM/PGD sweep + figures + metrics.json"
	@echo "make pretrained  - ONLINE : attack torchvision ResNet-18 on CIFAR (downloads ~45MB weights); auto-falls back offline"
	@echo "make attack ARGS=--pretrained  - same as 'make pretrained'"
	@echo "make test        - fast smoke tests (-m 'not slow')"
	@echo "make clean       - remove data/, models/ and generated figures/metrics"

attack: ## the one-shot money target (offline by default)
	$(PY) scripts/run_attack.py $(ARGS)

pretrained:
	$(PY) scripts/run_attack.py --pretrained

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -rf data/cifar-10-batches-py data/*.tar.gz models/*.pt results/figures/*.png results/metrics.json
