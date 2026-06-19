"""Train the from-scratch prompt-injection detector and save it to models/.

Default path is fully offline: generates a synthetic injection dataset, fits a
TF-IDF -> LogisticRegression pipeline, prints precision/recall/ROC-AUC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defend_rag import (  # noqa: E402
    InjectionDetector,
    generate_dataset,
    set_seed,
    train_detector,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "injection_detector.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the injection detector.")
    parser.add_argument("--n-per-class", type=int, default=600)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    set_seed(42)
    ds = generate_dataset(n_per_class=args.n_per_class, seed=42)
    n_test = int(len(ds) * args.test_frac)
    train_x, train_y = ds.texts[n_test:], ds.labels[n_test:]
    test_x, test_y = ds.texts[:n_test], ds.labels[:n_test]

    det: InjectionDetector = train_detector(train_x, train_y, threshold=args.threshold)
    report = det.evaluate(test_x, test_y)
    det.save(args.out)

    print(f"trained on {len(train_x)} examples, tested on {len(test_x)}")
    print(f"  precision={report.precision:.3f}  recall={report.recall:.3f}")
    print(f"  f1={report.f1:.3f}  roc_auc={report.roc_auc:.3f}")
    print(f"saved detector -> {args.out}")


if __name__ == "__main__":
    main()
