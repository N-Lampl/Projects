#!/usr/bin/env python3
"""Run the prompt-injection / exfiltration suite against the p4 lab target and
write results/metrics.json + figures. Offline + deterministic by default.

Authorized use only: targets ../p4-vulnerable-rag (self-built lab). See ../../ETHICS.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attack_rag import attack_success_rate, run_suite, set_seed  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FIG = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _bar_asr(results) -> Path:
    names = [r.technique for r in results]
    vals = [1 if r.succeeded else 0 for r in results]
    colors = ["#c0392b" if v else "#27ae60" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("attack succeeded (1 = target compromised)")
    ax.set_title("Prompt-injection ASR vs the (undefended) lab RAG")
    fig.tight_layout()
    out = FIG / "attack_success.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _bar_exfil(channel) -> Path:
    summ = channel.summary()
    by_type: dict[str, int] = {}
    for ev in channel.events:
        by_type[ev.channel] = by_type.get(ev.channel, 0) + len(ev.decoded)
    labels = list(by_type) or ["(none)"]
    vals = list(by_type.values()) or [0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color="#8e44ad")
    ax.set_ylabel("secrets smuggled (simulated)")
    ax.set_title(f"Data exfiltration: {summ['distinct_secrets']} distinct secrets captured")
    fig.tight_layout()
    out = FIG / "exfiltration.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    set_seed()
    FIG.mkdir(parents=True, exist_ok=True)

    results, channel, control = run_suite()
    asr = attack_success_rate(results)

    print(f"ASR over {len(results)} attacks: {asr * 100:.0f}%")
    for r in results:
        flag = "HIT " if r.succeeded else "miss"
        leaked = ",".join(r.secrets_leaked) if r.secrets_leaked else "-"
        print(f"  [{flag}] {r.technique:22} leaked={leaked}")
    print(f"  [ctrl] benign_control     leaked={'YES (bad!)' if control.succeeded else 'no (good)'}")

    fig1 = _bar_asr(results)
    fig2 = _bar_exfil(channel)

    metrics = {
        "project": "p5-attack-rag-pyrit",
        "summary": "Prompt-injection + exfiltration attacks vs the p4 lab RAG (offline mock).",
        "attack_success_rate": asr,
        "n_attacks": len(results),
        "by_technique": {
            r.technique: {"owasp": r.owasp, "succeeded": r.succeeded,
                          "leaked_types": list(r.secrets_leaked), "turns": r.turns}
            for r in results
        },
        "benign_control_leaked": control.succeeded,
        "exfiltration": channel.summary(),
        "figures": [str(fig1.relative_to(PROJECT)), str(fig2.relative_to(PROJECT))],
        "note": "Offline deterministic mock target. Real garak/PyRIT + LLM via .env (optional).",
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {METRICS.relative_to(PROJECT)} and 2 figures")


if __name__ == "__main__":
    main()
