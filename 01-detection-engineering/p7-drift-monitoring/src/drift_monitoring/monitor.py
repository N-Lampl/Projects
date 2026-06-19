"""The drift monitor: scan a stream window-by-window, score each feature with
PSI + KS, and raise alerts against configurable thresholds.

This is the 'security control' layer. A model whose inputs have drifted is a
model you can no longer trust -- and in an adversarial setting, drift on a
detector's inputs can be the *signature of an evasion attempt*. The monitor
turns raw drift statistics into actionable, thresholded alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import PSI_MAJOR, ks_pvalue, ks_statistic, psi, psi_severity


@dataclass
class AlertThresholds:
    psi: float = PSI_MAJOR        # PSI >= this -> alert (0.25 = major shift)
    ks: float = 0.15              # KS statistic >= this -> alert
    ks_pvalue: float = 0.01       # KS p-value < this -> distributions differ


def score_feature(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> dict:
    """Compute PSI + KS for one feature in one window."""
    p = psi(reference, current, n_bins=n_bins)
    d = ks_statistic(reference, current)
    pv = ks_pvalue(reference, current)
    return {
        "psi": p,
        "psi_severity": psi_severity(p),
        "ks": d,
        "ks_pvalue": pv,
    }


def evaluate_window(
    reference: np.ndarray,
    window: np.ndarray,
    feature_names: list[str],
    thresholds: AlertThresholds,
    n_bins: int = 10,
) -> dict:
    """Score every feature of one window and decide if it alerts.

    A feature alerts when PSI crosses its threshold AND the KS evidence agrees
    (statistic large *and* p-value significant) -- requiring both cuts noise.
    A window alerts if any feature alerts.
    """
    per_feature = {}
    alerting = []
    for j, name in enumerate(feature_names):
        s = score_feature(reference[:, j], window[:, j], n_bins=n_bins)
        ks_fires = s["ks"] >= thresholds.ks and s["ks_pvalue"] < thresholds.ks_pvalue
        s["alert"] = bool(s["psi"] >= thresholds.psi and ks_fires)
        per_feature[name] = s
        if s["alert"]:
            alerting.append(name)
    return {
        "window_alert": len(alerting) > 0,
        "alerting_features": alerting,
        "features": per_feature,
    }


def run_monitor(
    reference: np.ndarray,
    windows: list[np.ndarray],
    feature_names: list[str],
    thresholds: AlertThresholds | None = None,
    n_bins: int = 10,
) -> list[dict]:
    """Evaluate every window in a stream. Returns one report dict per window."""
    thr = thresholds or AlertThresholds()
    reports = []
    for i, w in enumerate(windows):
        report = evaluate_window(reference, w, feature_names, thr, n_bins=n_bins)
        report["window"] = i
        reports.append(report)
    return reports


def first_alert_window(reports: list[dict]) -> int | None:
    """Index of the first window that raised any alert, or None."""
    for r in reports:
        if r["window_alert"]:
            return r["window"]
    return None


def psi_matrix(reports: list[dict], feature_names: list[str]) -> np.ndarray:
    """(n_windows, n_features) matrix of PSI values for plotting."""
    return np.array(
        [[r["features"][f]["psi"] for f in feature_names] for r in reports],
        dtype=float,
    )
