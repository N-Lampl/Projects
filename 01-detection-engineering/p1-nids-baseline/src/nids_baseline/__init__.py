"""nids_baseline: a NIDS baseline built on the shared ``ids_pipeline`` library.

This project is thin by design. The reusable, leak-free preprocessing +
RandomForest pipeline lives in ``../shared/ids_pipeline`` and is imported BY
PATH (see :func:`nids_baseline.utils.ensure_ids_pipeline_on_path`). Here we add
a SOC-facing reporting layer (alert-budget thresholds, daily false-alert load)
and the run/plot entrypoint.

Public API::

    from nids_baseline import set_seed, get_pipeline_api, soc_report

    api = get_pipeline_api()                 # the shared ids_pipeline module
    ds = api.load_data(synthetic=True)
    pipe = api.train(api.build_pipeline(ds), ds)
    scores = api.predict_proba(pipe, ds.X_test)
    rep = soc_report(ds.y_test, scores, threshold=0.5)
"""

from __future__ import annotations

from .soc import soc_report, sweep_operating_points, threshold_for_alert_rate
from .utils import ensure_ids_pipeline_on_path, get_device, set_seed, shared_pipeline_src


def get_pipeline_api():
    """Import and return the shared ``ids_pipeline`` module (loads it by path)."""
    ensure_ids_pipeline_on_path()
    import ids_pipeline  # noqa: PLC0415

    return ids_pipeline


__all__ = [
    "set_seed",
    "get_device",
    "shared_pipeline_src",
    "ensure_ids_pipeline_on_path",
    "get_pipeline_api",
    "soc_report",
    "sweep_operating_points",
    "threshold_for_alert_rate",
]
