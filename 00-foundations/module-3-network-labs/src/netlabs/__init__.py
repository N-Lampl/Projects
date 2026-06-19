"""netlabs: TCP/IP, HTTP, DNS, TLS hands-on labs over a SYNTHETIC pcap.

Everything is offline: we hand-craft real wire-format frames, write a real .pcap,
and dissect it back — no live sniffing, no network. scapy is an optional engine.

Public API:
    set_seed, get_device        -- reproducibility helpers (stdlib-only)
    protocols (module)          -- wire-format encoders (Ethernet/IP/TCP/UDP/DNS/HTTP/TLS)
    write_pcap, read_pcap       -- classic libpcap I/O
    generate_pcap, build_scenario -- create the synthetic trace
    parse_pcap, parse_frame, summarize -- dissect + aggregate per-protocol counts
"""

from . import protocols
from .generate import build_scenario, generate_pcap
from .parse import parse_frame, parse_pcap, summarize
from .pcap import read_pcap, write_pcap
from .utils import get_device, set_seed

__all__ = [
    "set_seed",
    "get_device",
    "protocols",
    "write_pcap",
    "read_pcap",
    "generate_pcap",
    "build_scenario",
    "parse_pcap",
    "parse_frame",
    "summarize",
]
