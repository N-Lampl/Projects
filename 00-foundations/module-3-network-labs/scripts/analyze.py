#!/usr/bin/env python3
"""Parse the synthetic pcap, print a per-packet dissection + per-protocol writeups,
draw figures, and write results/metrics.json. Auto-generates the pcap if missing.
Run via `make run`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netlabs import generate_pcap, parse_pcap, set_seed, summarize  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
PCAP = PROJECT / "data" / "synthetic.pcap"
FIG_DIR = PROJECT / "results" / "figures"
METRICS = PROJECT / "results" / "metrics.json"


def _one_liner(p: dict) -> str:
    """A tcpdump-ish single line for a parsed packet."""
    if p.get("l3") != "IPv4":
        return f"{p.get('eth_src')} -> {p.get('eth_dst')}  {p.get('l3', '?')}"
    flow = f"{p.get('src_ip')}:{p.get('sport', '-')} -> {p.get('dst_ip')}:{p.get('dport', '-')}"
    l4 = p.get("l4", "?")
    extra = ""
    if l4 == "TCP":
        extra = f"[{p.get('flags')}]"
        if p.get("payload_len"):
            extra += f" len={p['payload_len']}"
    if p.get("l7") == "DNS":
        extra += f" DNS {p.get('dns_qr')} {p.get('qname', '')} {p.get('qtype', '')}"
        if "answer_ip" in p:
            extra += f" -> {p['answer_ip']}"
    elif p.get("l7") == "HTTP":
        if p.get("http_kind") == "request":
            extra += f" HTTP {p.get('http_method')} {p.get('http_path')}"
        else:
            extra += f" HTTP {p.get('http_status')}"
    elif p.get("l7") == "TLS":
        extra += f" TLS {p.get('tls_handshake', 'record')}"
    return f"{l4:4} {flow:42} {extra}"


def _plot_protocol_bars(summary: dict) -> Path:
    layers = [("L3", summary["l3_protocols"]), ("L4", summary["l4_protocols"]),
              ("L7", summary["l7_protocols"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    colors = ["#2980b9", "#27ae60", "#c0392b"]
    for ax, (title, counts), color in zip(axes, layers, colors):
        names = list(counts) or ["(none)"]
        vals = [counts.get(n, 0) for n in names]
        ax.bar(names, vals, color=color)
        ax.set_title(f"{title} protocols", fontsize=11)
        ax.set_ylabel("packets")
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=9)
    fig.suptitle("Synthetic pcap: packet counts by protocol layer", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "protocol_breakdown.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_timeline(packets: list[dict]) -> Path:
    """A flow timeline: each packet as a labelled step, coloured by L4."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    color_map = {"TCP": "#27ae60", "UDP": "#2980b9"}
    for i, p in enumerate(packets):
        c = color_map.get(p.get("l4"), "#7f8c8d")
        ax.scatter(i, 0, s=160, color=c, zorder=3)
        label = p.get("l7") or p.get("l4", "?")
        if p.get("l4") == "TCP" and not p.get("l7"):
            label = p.get("flags", "TCP")
        ax.annotate(label, (i, 0), rotation=55, fontsize=7, ha="left", va="bottom",
                    xytext=(0, 6), textcoords="offset points")
    ax.plot(range(len(packets)), [0] * len(packets), color="#bdc3c7", zorder=1)
    ax.set_yticks([])
    ax.set_xlabel("packet # (capture order)")
    ax.set_title("Conversation timeline: DNS lookup -> TCP+HTTP fetch -> TLS handshake", pad=10)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markersize=10,
                          label=k) for k, v in color_map.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    out = FIG_DIR / "flow_timeline.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


WRITEUPS = {
    "TCP/IP": "Ethernet II frames carry IPv4 datagrams; TCP provides the reliable byte stream. "
              "The capture shows the canonical 3-way handshake (SYN, SYN/ACK, ACK), data with "
              "PSH/ACK, and a FIN/ACK teardown — connection state you can track from flags alone.",
    "DNS": "Name resolution runs first over UDP/53: a recursion-desired A query for the host, "
           "answered with an A record. DNS is unauthenticated and unencrypted here — a classic "
           "spoofing / cache-poisoning surface, and the reason DNSSEC / DoH exist.",
    "HTTP": "The cleartext GET /index.html and its 200 OK ride inside the established TCP stream. "
            "Method, path, Host header and status line are fully visible to any on-path observer "
            "— the motivation for TLS.",
    "TLS": "A second flow to :443 begins the TLS handshake (ClientHello -> ServerHello). The "
           "ClientHello's SNI extension leaks the destination hostname in the clear even though "
           "the payload that follows is encrypted — relevant to traffic-analysis and ECH.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", type=Path, default=PCAP)
    ap.add_argument("--engine", choices=["python", "scapy"], default="python")
    ap.add_argument("--quiet", action="store_true", help="suppress the per-packet dump")
    args = ap.parse_args()

    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.pcap.exists():
        print(f"no pcap at {args.pcap.relative_to(PROJECT)} - generating it...")
        args.pcap.parent.mkdir(parents=True, exist_ok=True)
        generate_pcap(str(args.pcap), engine=args.engine)

    packets = parse_pcap(str(args.pcap), engine=args.engine)
    summary = summarize(packets)

    if not args.quiet:
        print(f"\n=== dissection of {args.pcap.name} ({len(packets)} packets) ===")
        for i, p in enumerate(packets):
            print(f"{i:2}  {_one_liner(p)}")

    print("\n=== per-protocol writeups ===")
    for proto, text in WRITEUPS.items():
        print(f"\n[{proto}]\n{text}")

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    bars = _plot_protocol_bars(summary)
    timeline = _plot_timeline(packets)

    metrics = {
        "project": "module-3-network-labs",
        "summary": (
            f"Parsed a self-generated synthetic pcap of {summary['total_packets']} packets "
            f"spanning {len(summary['l4_protocols'])} L4 and {len(summary['l7_protocols'])} L7 "
            f"protocols; TCP 3-way handshake observed="
            f"{summary['tcp_3way_handshake_observed']}."
        ),
        "seed": 42,
        "engine": args.engine,
        "pcap": str(args.pcap.relative_to(PROJECT)),
        **summary,
        "figures": [str(bars.relative_to(PROJECT)), str(timeline.relative_to(PROJECT))],
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nwrote {bars.relative_to(PROJECT)}")
    print(f"wrote {timeline.relative_to(PROJECT)}")
    print(f"wrote {METRICS.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
