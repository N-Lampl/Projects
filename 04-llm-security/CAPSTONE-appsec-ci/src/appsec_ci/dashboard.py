"""Trend ASR per OWASP category to a figure + maintain a run history.

Two figures are produced:

  * ``asr_by_category.png`` -- grouped bars: ASR per OWASP category for the
    vulnerable target vs the remediated target (the before/after that proves the
    fixes work).
  * ``asr_trend.png``       -- a line per OWASP category over the recorded run
    history (``results/history.jsonl``), so a dashboard can show ASR trending
    DOWN as remediations land across CI runs.

History is a small append-only JSONL of ``{run, ts, target, by_owasp}`` records;
each dashboard run appends the current vulnerable + remediated points so the
trend is reproducible offline without a real CI database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from .gate import GateResult  # noqa: E402
from .harness import OWASP_CATEGORIES  # noqa: E402

_CAT_ORDER = list(OWASP_CATEGORIES.values())
_SHORT = {v: k for k, v in OWASP_CATEGORIES.items()}  # full label -> "LLM01"


def _asr_vector(gate: GateResult) -> list[float]:
    return [gate.by_owasp[c].asr if c in gate.by_owasp else 0.0 for c in _CAT_ORDER]


def plot_before_after(vuln: GateResult, remediated: GateResult, out: Path) -> Path:
    """Grouped bar chart: per-OWASP ASR, vulnerable vs remediated."""
    labels = [_SHORT.get(c, c) for c in _CAT_ORDER]
    before = _asr_vector(vuln)
    after = _asr_vector(remediated)
    x = range(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar([i - w / 2 for i in x], before, width=w, label="vulnerable (p4)",
                color="#c0392b")
    b2 = ax.bar([i + w / 2 for i in x], after, width=w, label="remediated (p7)",
                color="#27ae60")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("attack-success rate (ASR)")
    ax.set_ylim(0, 1.1)
    ax.set_title("CI red-team: ASR per OWASP LLM category (before vs after fixes)")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def append_history(history_path: Path, run: int, target: str, gate: GateResult) -> None:
    """Append one run record (per-OWASP ASR) to the JSONL history."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "run": run,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "overall_asr": gate.overall_asr,
        "by_owasp": {_SHORT.get(c, c): (gate.by_owasp[c].asr if c in gate.by_owasp else 0.0)
                     for c in _CAT_ORDER},
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def seed_history(history_path: Path, vuln: GateResult, remediated: GateResult) -> None:
    """(Re)write a deterministic demo history: a fix rollout over 5 runs.

    Runs 1-2 are the vulnerable baseline; runs 3-4 show partial remediation
    (interpolated); run 5 is the fully-remediated target. This makes the trend
    figure show ASR declining without needing real historical CI data.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)
    before = {_SHORT.get(c, c): (vuln.by_owasp[c].asr if c in vuln.by_owasp else 0.0)
              for c in _CAT_ORDER}
    after = {_SHORT.get(c, c): (remediated.by_owasp[c].asr if c in remediated.by_owasp else 0.0)
             for c in _CAT_ORDER}
    short = [_SHORT.get(c, c) for c in _CAT_ORDER]

    records = []
    # A staggered fix rollout: each category is remediated on a slightly
    # different schedule (high-severity first), so the per-category trend lines
    # are visibly distinct rather than overlapping.
    schedules = {
        "LLM01": [1.0, 0.7, 0.3, 0.0, 0.0],  # injection fixed first
        "LLM07": [1.0, 0.8, 0.4, 0.1, 0.0],  # system-prompt leak next
        "LLM02": [1.0, 1.0, 0.6, 0.2, 0.0],  # PII/secret redaction
        "LLM06": [1.0, 1.0, 0.8, 0.4, 0.0],  # tool authz last
    }
    for i in range(1, 6):
        by = {
            k: round(after[k] + schedules.get(k, [1, 1, 0.6, 0.3, 0])[i - 1]
                     * (before[k] - after[k]), 4)
            for k in short
        }
        overall = round(sum(by.values()) / len(by), 4)
        records.append({
            "run": i,
            "ts": f"2026-06-{10 + i:02d}T12:00:00+00:00",
            "target": "vulnerable->remediated rollout",
            "overall_asr": overall,
            "by_owasp": by,
        })
    with history_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def plot_trend(history_path: Path, out: Path) -> Path:
    """Line chart: ASR per OWASP category across the recorded run history."""
    records = []
    with history_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["run"])
    runs = [r["run"] for r in records]
    short = [_SHORT.get(c, c) for c in _CAT_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in short:
        series = [r["by_owasp"].get(cat, 0.0) for r in records]
        ax.plot(runs, series, marker="o", label=cat)
    overall = [r.get("overall_asr", 0.0) for r in records]
    ax.plot(runs, overall, marker="s", linewidth=2.4, color="black", label="overall")
    ax.set_xlabel("CI run #")
    ax.set_ylabel("attack-success rate (ASR)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(runs)
    ax.set_title("ASR trend per OWASP LLM category across CI runs")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


__all__ = [
    "plot_before_after",
    "plot_trend",
    "append_history",
    "seed_history",
]
