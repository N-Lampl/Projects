"""Generate a SYNTHETIC, deterministic .pcap that exercises the whole stack.

No live sniffing — we hand-craft realistic conversations (DNS lookup -> TCP handshake
-> HTTP request/response, plus a TLS handshake) and serialise them to a real pcap.
An optional scapy path (``--engine scapy``) builds the same scenario with scapy when
it is installed; the default pure-python path needs nothing but the stdlib.
"""

from __future__ import annotations

from . import protocols as P
from .pcap import PcapRecord, write_pcap

# A small, self-consistent topology (all RFC-1918 / documentation addresses).
CLIENT_MAC = "02:00:00:00:00:01"
GATEWAY_MAC = "02:00:00:00:00:02"
CLIENT_IP = "10.0.0.10"
DNS_IP = "10.0.0.1"
WEB_IP = "93.184.216.34"  # example.com documentation address
HOST = "example.com"


def _udp_frame(src_ip, dst_ip, sport, dport, payload, *, to_client) -> bytes:
    seg = P.udp_segment(src_ip, dst_ip, sport, dport, payload)
    ip = P.ipv4_packet(src_ip, dst_ip, P.IPPROTO_UDP, seg)
    macs = (CLIENT_MAC, GATEWAY_MAC) if to_client else (GATEWAY_MAC, CLIENT_MAC)
    return P.eth_frame(macs[0], macs[1], ip)


def _tcp_frame(src_ip, dst_ip, sport, dport, seq, ack, flags, payload, *, to_client) -> bytes:
    seg = P.tcp_segment(src_ip, dst_ip, sport, dport, seq, ack, flags, payload)
    ip = P.ipv4_packet(src_ip, dst_ip, P.IPPROTO_TCP, seg)
    macs = (CLIENT_MAC, GATEWAY_MAC) if to_client else (GATEWAY_MAC, CLIENT_MAC)
    return P.eth_frame(macs[0], macs[1], ip)


def build_scenario() -> list[PcapRecord]:
    """Return ordered pcap records for: DNS A lookup, TCP+HTTP fetch, TLS handshake."""
    recs: list[PcapRecord] = []
    t = 0  # microseconds, monotonic deterministic clock

    def add(frame: bytes, dt_us: int = 1500) -> None:
        nonlocal t
        recs.append(PcapRecord(ts_sec=0, ts_usec=t, data=frame))
        t += dt_us

    cport = 49152  # ephemeral client port

    # --- DNS: client asks the resolver for example.com, gets an A record ---
    add(_udp_frame(CLIENT_IP, DNS_IP, cport, 53,
                   P.dns_query(0x1a2b, HOST, qtype=1), to_client=False))
    add(_udp_frame(DNS_IP, CLIENT_IP, 53, cport,
                   P.dns_response(0x1a2b, HOST, WEB_IP, qtype=1), to_client=True))

    # --- TCP 3-way handshake to the web server on :80 ---
    cseq, sseq = 1000, 5000
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport, 80, cseq, 0, P.SYN, b"", to_client=False))
    add(_tcp_frame(WEB_IP, CLIENT_IP, 80, cport, sseq, cseq + 1, P.SYN | P.ACK, b"",
                   to_client=True))
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport, 80, cseq + 1, sseq + 1, P.ACK, b"",
                   to_client=False))

    # --- HTTP request + response carried over the established TCP stream ---
    req = P.http_request("GET", "/index.html", HOST)
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport, 80, cseq + 1, sseq + 1, P.PSH | P.ACK, req,
                   to_client=False))
    body = b"<html><body><h1>Hello from netlabs</h1></body></html>"
    resp = P.http_response(200, "OK", body)
    add(_tcp_frame(WEB_IP, CLIENT_IP, 80, cport, sseq + 1, cseq + 1 + len(req),
                   P.PSH | P.ACK, resp, to_client=True))
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport, 80, cseq + 1 + len(req), sseq + 1 + len(resp),
                   P.ACK, b"", to_client=False))

    # --- TCP teardown (FIN/ACK both ways) ---
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport, 80, cseq + 1 + len(req), sseq + 1 + len(resp),
                   P.FIN | P.ACK, b"", to_client=False))
    add(_tcp_frame(WEB_IP, CLIENT_IP, 80, cport, sseq + 1 + len(resp), cseq + 2 + len(req),
                   P.FIN | P.ACK, b"", to_client=True))

    # --- A second flow: TLS handshake to :443 (ClientHello / ServerHello) ---
    cport2 = 49153
    cseq2, sseq2 = 2000, 9000
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport2, 443, cseq2, 0, P.SYN, b"", to_client=False))
    add(_tcp_frame(WEB_IP, CLIENT_IP, 443, cport2, sseq2, cseq2 + 1, P.SYN | P.ACK, b"",
                   to_client=True))
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport2, 443, cseq2 + 1, sseq2 + 1, P.ACK, b"",
                   to_client=False))
    ch = P.tls_client_hello(HOST)
    add(_tcp_frame(CLIENT_IP, WEB_IP, cport2, 443, cseq2 + 1, sseq2 + 1, P.PSH | P.ACK, ch,
                   to_client=False))
    sh = P.tls_server_hello()
    add(_tcp_frame(WEB_IP, CLIENT_IP, 443, cport2, sseq2 + 1, cseq2 + 1 + len(ch),
                   P.PSH | P.ACK, sh, to_client=True))

    return recs


def generate_pcap(path: str, engine: str = "python") -> int:
    """Write the synthetic trace to ``path``. Returns the number of packets written."""
    if engine == "scapy":
        return _generate_with_scapy(path)
    recs = build_scenario()
    write_pcap(path, recs)
    return len(recs)


def _generate_with_scapy(path: str) -> int:
    """Optional enhanced path: build the same scenario with scapy (lazy import)."""
    try:
        from scapy.all import (  # type: ignore  # noqa: I001
            DNS,
            DNSQR,
            DNSRR,
            IP,
            TCP,
            UDP,
            Ether,
            Raw,
            wrpcap,
        )
    except ImportError as e:  # pragma: no cover - exercised only when scapy is absent
        raise RuntimeError(
            "scapy is not installed; use engine='python' (the default offline path)."
        ) from e

    e_out = Ether(src=GATEWAY_MAC, dst=CLIENT_MAC)
    e_in = Ether(src=CLIENT_MAC, dst=GATEWAY_MAC)
    pkts = []
    pkts.append(e_in / IP(src=CLIENT_IP, dst=DNS_IP) / UDP(sport=49152, dport=53)
                / DNS(rd=1, qd=DNSQR(qname=HOST)))
    pkts.append(e_out / IP(src=DNS_IP, dst=CLIENT_IP) / UDP(sport=53, dport=49152)
                / DNS(qr=1, qd=DNSQR(qname=HOST), an=DNSRR(rrname=HOST, rdata=WEB_IP)))
    pkts.append(e_in / IP(src=CLIENT_IP, dst=WEB_IP) / TCP(sport=49152, dport=80, flags="S"))
    pkts.append(e_out / IP(src=WEB_IP, dst=CLIENT_IP) / TCP(sport=80, dport=49152, flags="SA"))
    pkts.append(e_in / IP(src=CLIENT_IP, dst=WEB_IP) / TCP(sport=49152, dport=80, flags="A"))
    pkts.append(e_in / IP(src=CLIENT_IP, dst=WEB_IP) / TCP(sport=49152, dport=80, flags="PA")
                / Raw(P.http_request("GET", "/index.html", HOST)))
    wrpcap(path, pkts)
    return len(pkts)
