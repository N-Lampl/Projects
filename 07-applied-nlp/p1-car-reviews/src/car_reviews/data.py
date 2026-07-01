"""Load the car-review dataset — real HuggingFace data by default, with a
deterministic offline fallback so tests / CI never touch the network.

Default (real) source: ``florentgbelidji/car-reviews`` on the HuggingFace Hub —
36,984 Edmunds reviews, single ``train`` CSV split. Columns we use:
``Vehicle_Title`` (year+make+model+trim), ``Review`` (text), ``Review_Title``,
``Rating`` (1-5, the ground-truth label), ``Review_Date``, ``Author_Name``.

If ``datasets`` is missing or the download fails (offline CI, no network), we
fall back to :func:`car_reviews.synthetic.make_reviews`.
"""

from __future__ import annotations

import pandas as pd

from .synthetic import make_reviews

HF_DATASET = "florentgbelidji/car-reviews"
NEEDED = ["Review_Date", "Author_Name", "Vehicle_Title", "Review_Title", "Review", "Rating"]


def _load_hf() -> pd.DataFrame:
    """Load the real dataset from the Hub as a DataFrame (raises if unavailable)."""
    from datasets import load_dataset  # lazy: optional in offline/CI

    ds = load_dataset(HF_DATASET, split="train")
    return ds.to_pandas()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the needed columns, drop empty reviews, coerce ``Rating`` to 1-5 int."""
    for col in NEEDED:
        if col not in df.columns:
            df[col] = "" if col != "Rating" else 0
    df = df[NEEDED].copy()
    df["Review"] = df["Review"].fillna("").astype(str).str.strip()
    df["Review_Title"] = df["Review_Title"].fillna("").astype(str).str.strip()
    df["Vehicle_Title"] = df["Vehicle_Title"].fillna("").astype(str).str.strip()
    df = df[df["Review"].str.len() > 0]
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df[df["Rating"].between(1, 5)]
    df["Rating"] = df["Rating"].round().astype(int)
    return df.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int, by: str) -> pd.DataFrame:
    """Down-sample to ~``n`` rows keeping the per-``by`` proportions (seeded)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    out = df.groupby(by, group_keys=False, observed=True).sample(frac=frac, random_state=seed)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def load_reviews(
    sample: int | None = 3000,
    full: bool = False,
    seed: int = 42,
    stratify_by: str = "Rating",
) -> pd.DataFrame:
    """Return a cleaned review DataFrame.

    Parameters
    ----------
    sample : cap the number of rows (stratified by ``stratify_by``). Ignored if
        ``full`` is True or ``sample`` is None.
    full : process the entire dataset (the committed-artifact path).
    """
    try:
        df = _clean(_load_hf())
        source = f"huggingface: {HF_DATASET}"
    except Exception as exc:  # offline / no `datasets` / download error
        n = max(sample or 500, 500)
        df = _clean(make_reviews(n=n, seed=seed))
        source = "synthetic (offline fallback)"
        print(f"[data] real dataset unavailable ({type(exc).__name__}); using {source}.")

    n_total = int(len(df))
    if not full and sample is not None and sample < len(df):
        df = _stratified_sample(df, sample, seed, stratify_by)

    df.attrs["source"] = source
    df.attrs["n_total"] = n_total
    return df
