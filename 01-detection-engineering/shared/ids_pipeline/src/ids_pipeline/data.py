"""Data loading for the tabular IDS pipeline.

Two sources, one API:

- ``load_data(synthetic=True)``  -> a deterministic SYNTHETIC network-flow
  generator (the DEFAULT, fully offline, no downloads). Produces a realistic
  mix of benign and attack flows with class imbalance.
- ``load_data(synthetic=False)`` -> the real NSL-KDD dataset, loaded from a
  local CSV you download yourself (see ``data/README.md``). If the files are
  missing we raise a clear error pointing at the download instructions instead
  of silently falling back, so benchmarks are never accidentally synthetic.

Everything returns a :class:`Dataset` of pandas DataFrames so the preprocessing
stage can fit encoders/scalers on TRAIN only (leak-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Columns the synthetic generator produces. A small but representative slice of
# the kinds of features a flow-based NIDS sees (durations, byte counts, rates,
# flag ratios, plus one categorical protocol field).
NUMERIC_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
LABEL_COL = "label"  # 0 = benign (normal), 1 = attack


@dataclass
class Dataset:
    """A train/test split plus the column roles needed for leak-free preprocessing."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    source: str  # "synthetic" or "nsl-kdd"

    @property
    def n_features(self) -> int:
        return len(self.numeric_features) + len(self.categorical_features)


def make_synthetic_flows(
    n_samples: int = 12000,
    attack_fraction: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic, class-imbalanced network-flow table.

    Benign and attack flows are drawn from different (overlapping) distributions
    so the problem is learnable but not trivial -- mirroring the signal a real
    NIDS gets. ``attack_fraction`` controls the imbalance (defaults to a SOC-ish
    ~25% positives so precision@k and ROC-AUC are meaningful).
    """
    rng = np.random.default_rng(seed)
    n_attack = int(round(n_samples * attack_fraction))
    n_benign = n_samples - n_attack

    protocols = np.array(["tcp", "udp", "icmp"])
    services = np.array(["http", "smtp", "dns", "ftp", "ssh", "other"])
    flags = np.array(["SF", "S0", "REJ", "RSTO"])

    # Distributions deliberately OVERLAP so the task is non-trivial (ROC-AUC well
    # below 1.0). A clean, honest baseline -- not a perfect-score toy.
    def _block(n: int, attack: bool) -> pd.DataFrame:
        if attack:
            duration = rng.exponential(2.0, n)
            src_bytes = rng.lognormal(5.7, 1.5, n)
            dst_bytes = rng.lognormal(5.5, 1.6, n)
            count = rng.poisson(34, n).astype(float)
            srv_count = rng.poisson(27, n).astype(float)
            serror_rate = np.clip(rng.beta(2.2, 5, n), 0, 1)
            rerror_rate = np.clip(rng.beta(2, 6, n), 0, 1)
            same_srv_rate = np.clip(rng.beta(4, 3.5, n), 0, 1)
            diff_srv_rate = np.clip(rng.beta(2.5, 4, n), 0, 1)
            dst_host_count = rng.poisson(85, n).astype(float)
            proto = rng.choice(protocols, n, p=[0.68, 0.20, 0.12])
            svc = rng.choice(services, n, p=[0.30, 0.10, 0.16, 0.12, 0.12, 0.20])
            flag = rng.choice(flags, n, p=[0.62, 0.18, 0.12, 0.08])
        else:
            duration = rng.exponential(2.4, n)
            src_bytes = rng.lognormal(5.9, 1.5, n)
            dst_bytes = rng.lognormal(5.9, 1.6, n)
            count = rng.poisson(28, n).astype(float)
            srv_count = rng.poisson(22, n).astype(float)
            serror_rate = np.clip(rng.beta(1.6, 6, n), 0, 1)
            rerror_rate = np.clip(rng.beta(1.6, 7, n), 0, 1)
            same_srv_rate = np.clip(rng.beta(5, 3, n), 0, 1)
            diff_srv_rate = np.clip(rng.beta(1.8, 4.5, n), 0, 1)
            dst_host_count = rng.poisson(68, n).astype(float)
            proto = rng.choice(protocols, n, p=[0.74, 0.18, 0.08])
            svc = rng.choice(services, n, p=[0.38, 0.12, 0.18, 0.10, 0.12, 0.10])
            flag = rng.choice(flags, n, p=[0.74, 0.10, 0.10, 0.06])
        return pd.DataFrame(
            {
                "duration": duration,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "count": count,
                "srv_count": srv_count,
                "serror_rate": serror_rate,
                "rerror_rate": rerror_rate,
                "same_srv_rate": same_srv_rate,
                "diff_srv_rate": diff_srv_rate,
                "dst_host_count": dst_host_count,
                "protocol_type": proto,
                "service": svc,
                "flag": flag,
                LABEL_COL: 1 if attack else 0,
            }
        )

    df = pd.concat([_block(n_benign, False), _block(n_attack, True)], ignore_index=True)
    # shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _train_test_split_df(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def _load_nsl_kdd(data_dir: Path) -> pd.DataFrame:
    """Load the real NSL-KDD train/test CSVs from ``data_dir``.

    NSL-KDD has 41 features + a label + a difficulty column. We map the
    multiclass attack labels to a binary normal/attack target and keep the
    subset of features our synthetic schema also exposes, so downstream code is
    source-agnostic.
    """
    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"NSL-KDD files not found in {data_dir}. "
            "See data/README.md for the exact download command, "
            "or use load_data(synthetic=True) for the offline path."
        )
    # Canonical NSL-KDD column order.
    cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty",
    ]
    keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def _read(path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, names=cols)
        out = raw[keep].copy()
        out[LABEL_COL] = (raw["attack"].str.lower() != "normal").astype(int)
        return out

    train = _read(train_path)
    test = _read(test_path)
    # tag split so we can re-split consistently downstream
    train["__split"] = "train"
    test["__split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def load_data(
    synthetic: bool = True,
    *,
    n_samples: int = 12000,
    attack_fraction: float = 0.25,
    test_size: float = 0.25,
    seed: int = 42,
    data_dir: str | Path | None = None,
) -> Dataset:
    """Load an IDS dataset as a leak-free train/test :class:`Dataset`.

    Parameters
    ----------
    synthetic:
        ``True`` (default) -> offline synthetic flow generator.
        ``False`` -> real NSL-KDD from local CSVs (see ``data/README.md``).
    """
    if synthetic:
        df = make_synthetic_flows(n_samples=n_samples, attack_fraction=attack_fraction, seed=seed)
        X_train, X_test, y_train, y_test = _train_test_split_df(df, test_size, seed)
        source = "synthetic"
    else:
        ddir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data"
        df = _load_nsl_kdd(ddir)
        train_df = df[df["__split"] == "train"].drop(columns="__split").reset_index(drop=True)
        test_df = df[df["__split"] == "test"].drop(columns="__split").reset_index(drop=True)
        X_train = train_df.drop(columns=[LABEL_COL])
        X_test = test_df.drop(columns=[LABEL_COL])
        y_train = train_df[LABEL_COL].astype(int)
        y_test = test_df[LABEL_COL].astype(int)
        source = "nsl-kdd"

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        source=source,
    )
