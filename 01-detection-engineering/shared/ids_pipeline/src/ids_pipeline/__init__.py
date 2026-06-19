"""ids_pipeline: a reusable, leak-free tabular-IDS library.

Shared by ``p1-nids-baseline`` and ``CAPSTONE-adversarial-ids``. The default
path runs fully offline on synthetic network-flow data with scikit-learn only.

Clean API::

    from ids_pipeline import load_data, build_pipeline, train, evaluate

    ds = load_data(synthetic=True)        # offline synthetic flows
    pipe = build_pipeline(ds)             # leak-free preprocess + RandomForest
    train(pipe, ds)                       # fits on TRAIN only
    metrics = evaluate(pipe, ds)          # SOC metrics on TEST
"""

from .data import Dataset, load_data, make_synthetic_flows
from .metrics import evaluate, precision_at_k
from .pipeline import build_pipeline, predict_proba, train
from .utils import get_device, set_seed

__all__ = [
    "Dataset",
    "load_data",
    "make_synthetic_flows",
    "build_pipeline",
    "train",
    "predict_proba",
    "evaluate",
    "precision_at_k",
    "set_seed",
    "get_device",
]
