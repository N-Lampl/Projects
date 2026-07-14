"""Fine-tune DistilBERT on the car-review star ratings (5-class), CPU-friendly.

The baseline models are general-domain; this learns car-review representations
directly from the 1-5 ``Rating`` labels. Weights are saved under ``models/``
(git-ignored) with ``id2label`` set to ``"1".."5"``, so the fine-tuned model plugs
straight into the SAME :class:`~car_reviews.sentiment.HFBackend` (kind ``star5``) -
identical ``predict`` / ``predict_proba`` and the same validation + aggregation code.
"""

from __future__ import annotations

from pathlib import Path

from .improve import build_text

BASE_MODEL = "distilbert-base-uncased"


def _lazy():
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - optional heavy path
        raise SystemExit(
            "fine-tuning needs transformers + datasets + torch:\n"
            "  pip install 'transformers>=4.44' 'torch>=2.2' datasets"
        ) from exc
    return (
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )


def finetune_distilbert(
    train_df,
    val_df=None,
    out_dir: str | Path = "models/finetuned",
    base_model: str = BASE_MODEL,
    epochs: int = 2,
    max_length: int = 256,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Fine-tune ``base_model`` on ``Rating`` (mapped 1-5 -> label 0-4) and save it."""
    (
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    ) = _lazy()
    from .utils import configure_torch_threads, set_seed

    set_seed(seed)
    configure_torch_threads()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model)
    id2label = {i: str(i + 1) for i in range(5)}  # -> HFBackend star5 reuse
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=5, id2label=id2label, label2id=label2id
    )

    def to_ds(df):
        ds = Dataset.from_dict(
            {"text": build_text(df), "labels": (df["Rating"].astype(int) - 1).tolist()}
        )
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True
        )

    train_ds = to_ds(train_df)
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        seed=seed,
        use_cpu=True,
        report_to=[],
        disable_tqdm=not verbose,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tok),
        processing_class=tok,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if verbose:
        print(f"[finetune] saved fine-tuned model to {out_dir}")
    return out_dir


def load_finetuned(
    model_dir: str | Path = "models/finetuned",
    max_length: int = 256,
    batch_size: int = 16,
    verbose: bool = True,
):
    """Return an HFBackend (kind ``star5``) backed by the locally fine-tuned model."""
    from .sentiment import HFBackend, ModelSpec

    spec = ModelSpec(str(model_dir), "1-5", "star5")
    return HFBackend(spec, max_length=max_length, batch_size=batch_size, verbose=verbose)
