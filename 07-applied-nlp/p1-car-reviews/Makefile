# Car-reviews NLP. Auto-detects uv; falls back to system python3.
PY := $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python3")

.PHONY: help run run-full finetune improve test clean
help:
	@echo "make run       - fast --sample run (nlptown, HF) -> figures + metrics.json"
	@echo "make run-full  - FULL 36,984-review run (the committed artifact; ~30-75 min CPU)"
	@echo "make finetune  - fine-tune DistilBERT on the star labels (~1-2 h CPU; the improvement)"
	@echo "make improve   - baseline vs. calibrated vs. fine-tuned comparison -> improvement.json"
	@echo "make test      - fast offline smoke tests (stub backend, -m 'not slow')"
	@echo "make clean     - remove generated figures + metrics.json + improvement.json"
	@echo "  overrides:  make run ARGS='--model sst2 --summaries abstractive'"

run: ## fast sampled run (downloads the HF model on first use)
	$(PY) scripts/run_analysis.py $(ARGS)

run-full: ## the committed artifact: every review
	$(PY) scripts/run_analysis.py --full $(ARGS)

finetune: ## slow: fine-tune DistilBERT -> models/finetuned (git-ignored)
	$(PY) scripts/train_finetune.py $(ARGS)

improve: ## compare baseline vs calibrated vs fine-tuned on the held-out test set
	$(PY) scripts/improve_model.py $(ARGS)

test:
	$(PY) -m pytest -m "not slow" -q

clean:
	rm -f results/figures/*.png results/metrics.json results/improvement.json
