#!/usr/bin/env python3
"""Read syllabus.json (the SY0-701 domain -> repo-module map) and produce a domain-coverage
metric, results/metrics.json, and a coverage figure. Pure stdlib + matplotlib -> runs offline.
Run via `make coverage`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
SYLLABUS = PROJECT / "syllabus.json"
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"

COVERED = "#27ae60"   # green: domain has >=1 repo artifact
UNCOVERED = "#c0392b"  # red: domain unmapped


def load_syllabus(path: Path = SYLLABUS) -> dict:
    """Load and lightly validate the domain map."""
    data = json.loads(path.read_text())
    if not data.get("domains"):
        raise ValueError("syllabus.json has no 'domains'")
    return data


def is_covered(domain: dict) -> bool:
    """A domain is covered iff it maps to at least one non-empty repo module."""
    return any(str(m).strip() for m in domain.get("repo_modules", []))


def compute_coverage(data: dict) -> dict:
    """Compute domain coverage and exam-weight-weighted coverage."""
    domains = data["domains"]
    n = len(domains)
    covered = [d for d in domains if is_covered(d)]
    total_weight = sum(float(d.get("weight", 0.0)) for d in domains)
    covered_weight = sum(float(d.get("weight", 0.0)) for d in covered)
    return {
        "n_domains": n,
        "n_covered": len(covered),
        "domain_coverage": round(len(covered) / n, 4) if n else 0.0,
        "total_weight": round(total_weight, 4),
        "weighted_coverage": round(covered_weight, 4),
    }


def _plot(data: dict, cov: dict) -> Path:
    domains = data["domains"]
    labels = [f'{d["id"]} {d["name"]}' for d in domains]
    weights = [float(d.get("weight", 0.0)) * 100 for d in domains]
    colors = [COVERED if is_covered(d) else UNCOVERED for d in domains]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = range(len(domains))
    ax.barh(list(y), weights, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("official exam weight (%)")
    ax.set_title(
        f"Security+ SY0-701 domain coverage: "
        f"{cov['n_covered']}/{cov['n_domains']} domains mapped to repo artifacts "
        f"({cov['domain_coverage'] * 100:.0f}%)",
        fontsize=11,
        pad=10,
    )
    for i, w in enumerate(weights):
        ax.annotate(f"{w:.0f}%", (w, i), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8)
    ax.set_xlim(0, max(weights) * 1.18)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COVERED),
        plt.Rectangle((0, 0), 1, 1, color=UNCOVERED),
    ]
    ax.legend(handles, ["covered by a repo artifact", "not yet mapped"], loc="lower right",
              fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "domain_coverage.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _rel(path: Path) -> str:
    """Path relative to the project root when possible, else just the name."""
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return path.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--syllabus", type=Path, default=SYLLABUS, help="path to syllabus.json")
    args = ap.parse_args()

    data = load_syllabus(args.syllabus)
    cov = compute_coverage(data)

    print(f"exam: {data.get('exam')}")
    for d in data["domains"]:
        mark = "OK " if is_covered(d) else "GAP"
        mods = ", ".join(d.get("repo_modules", [])) or "(none)"
        print(f"  [{mark}] {d['id']} {d['name']:<42} {d['weight'] * 100:4.0f}%  -> {mods}")
    print(
        f"\ndomain coverage: {cov['n_covered']}/{cov['n_domains']} "
        f"({cov['domain_coverage'] * 100:.0f}%)  |  weighted: {cov['weighted_coverage'] * 100:.0f}%"
    )

    fig = _plot(data, cov)

    metrics = {
        "project": "certpath",
        "summary": (
            f"{cov['n_covered']}/{cov['n_domains']} Security+ SY0-701 domains "
            f"({cov['domain_coverage'] * 100:.0f}%) map to a concrete repo artifact; "
            f"weighted coverage {cov['weighted_coverage'] * 100:.0f}% of exam weight."
        ),
        "exam": data.get("exam"),
        "n_domains": cov["n_domains"],
        "n_covered": cov["n_covered"],
        "domain_coverage": cov["domain_coverage"],
        "weighted_coverage": cov["weighted_coverage"],
        "total_weight": cov["total_weight"],
        "domains": [
            {
                "id": d["id"],
                "name": d["name"],
                "weight": d["weight"],
                "covered": is_covered(d),
                "repo_modules": d.get("repo_modules", []),
            }
            for d in data["domains"]
        ],
        "figures": [_rel(fig)],
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {_rel(fig)}")
    print(f"wrote {_rel(METRICS)}")


if __name__ == "__main__":
    main()
