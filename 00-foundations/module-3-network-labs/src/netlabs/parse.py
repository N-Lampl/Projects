"""Pure-stdlib protocol dissector: decode the synthetic pcap layer by layer.

Given the raw Ethernet frames from :mod:`netlabs.pcap`, peel Ethernet -> IPv4 ->
TCP/UDP -> (DNS | HTTP | TLS) and return one dict per packet plus aggregate counts.
An optional ``engine='scapy'`` path delegates to scapy when installed; the default
``engine='python'`` parser needs nothing but the stdlib.
"""

from __future__ import annotations

import socket
import struct
from collections import Counter

from . import protocols as P
from .pcap import read_pcap

TCP_FLAG_NAMES = [("FIN", P.FIN), ("SYN", P.SYN), ("RST", P.RST), ("PSH", P.PSH), ("ACK", P.ACK)]


def _flag_str(flags: int) -> str:
    return "/".join(name for name, bit in TCP_FLAG_NAMES if flags & bit) or "-"


def _decode_dns_name(buf: bytes, offset: int) -> tuple[str, int]:
    """Decode a (possibly compressed) DNS name; return (name, offset_after)."""
    labels = []
    jumped = False
    end_offset = offset
    while True:
        length = buf[offset]
        if length & 0xC0 == 0xC0:  # pointer
            pointer = ((length & 0x3F) << 8) | buf[offset + 1]
            if not jumped:
                end_offset = offset + 2
            offset = pointer
            jumped = True
            continue
        if length == 0:
            offset += 1
            if not jumped:
                end_offset = offset
            break
        labels.append(buf[offset + 1:offset + 1 + length].decode("ascii", "replace"))
        offset += 1 + length
    return ".".join(labels), end_offset


def _parse_dns(payload: bytes) -> dict:
    txn_id, flags, qd, an, _ns, _ar = struct.unpack("!HHHHHH", payload[:12])
    is_response = bool(flags & 0x8000)
    info: dict = {"l7": "DNS", "dns_id": txn_id, "dns_qr": "response" if is_response else "query",
                  "questions": qd, "answers": an}
    off = 12
    if qd:
        qname, off = _decode_dns_name(payload, off)
        qtype, _qclass = struct.unpack("!HH", payload[off:off + 4])
        off += 4
        info["qname"] = qname
        info["qtype"] = {1: "A", 28: "AAAA"}.get(qtype, str(qtype))
    if is_response and an:
        _name, off = _decode_dns_name(payload, off)
        atype, _aclass, _ttl, rdlen = struct.unpack("!HHIH", payload[off:off + 10])
        off += 10
        if atype == 1 and rdlen == 4:
            info["answer_ip"] = socket.inet_ntoa(payload[off:off + 4])
    return info


def _parse_http(payload: bytes) -> dict:
    try:
        head = payload.split(b"\r\n\r\n", 1)[0].decode("ascii", "replace")
    except Exception:
        return {"l7": "HTTP"}
    first = head.split("\r\n", 1)[0]
    info: dict = {"l7": "HTTP"}
    if first.startswith("HTTP/"):
        info["http_kind"] = "response"
        parts = first.split(" ", 2)
        if len(parts) >= 2:
            info["http_status"] = parts[1]
    else:
        info["http_kind"] = "request"
        parts = first.split(" ")
        if len(parts) >= 2:
            info["http_method"], info["http_path"] = parts[0], parts[1]
    return info


def _parse_tls(payload: bytes) -> dict:
    content_type, _ver, _length = struct.unpack("!BHH", payload[:5])
    info: dict = {"l7": "TLS", "tls_content_type": content_type}
    if content_type == P.TLS_HANDSHAKE and len(payload) >= 6:
        hs_type = payload[5]
        info["tls_handshake"] = {P.TLS_CLIENT_HELLO: "ClientHello",
                                 P.TLS_SERVER_HELLO: "ServerHello"}.get(hs_type, f"type{hs_type}")
    return info


def _classify_l7(sport: int, dport: int, payload: bytes) -> dict:
    if not payload:
        return {}
    if 80 in (sport, dport):
        return _parse_http(payload)
    if 443 in (sport, dport) and payload[0] in (P.TLS_HANDSHAKE, 0x14, 0x15, 0x17):
        return _parse_tls(payload)
    return {}


def parse_frame(frame: bytes) -> dict:
    """Dissect one Ethernet frame into a flat dict of decoded fields."""
    eth_dst, eth_src, ethertype = frame[:6], frame[6:12], struct.unpack("!H", frame[12:14])[0]
    pkt: dict = {
        "eth_src": ":".join(f"{b:02x}" for b in eth_src),
        "eth_dst": ":".join(f"{b:02x}" for b in eth_dst),
        "ethertype": ethertype,
    }
    if ethertype != P.ETH_TYPE_IPV4:
        pkt["l3"] = "non-IPv4"
        return pkt

    ip = frame[14:]
    ihl = (ip[0] & 0x0F) * 4
    proto = ip[9]
    src_ip = socket.inet_ntoa(ip[12:16])
    dst_ip = socket.inet_ntoa(ip[16:20])
    pkt.update({"l3": "IPv4", "src_ip": src_ip, "dst_ip": dst_ip, "ttl": ip[8]})
    l4 = ip[ihl:]

    if proto == P.IPPROTO_TCP:
        sport, dport, seq, ack = struct.unpack("!HHII", l4[:12])
        data_offset = (l4[12] >> 4) * 4
        flags = l4[13]
        payload = l4[data_offset:]
        pkt.update({"l4": "TCP", "sport": sport, "dport": dport, "seq": seq, "ack": ack,
                    "flags": _flag_str(flags), "payload_len": len(payload)})
        pkt.update(_classify_l7(sport, dport, payload))
    elif proto == P.IPPROTO_UDP:
        sport, dport, length, _chk = struct.unpack("!HHHH", l4[:8])
        payload = l4[8:length] if length >= 8 else l4[8:]
        pkt.update({"l4": "UDP", "sport": sport, "dport": dport, "payload_len": len(payload)})
        if 53 in (sport, dport) and payload:
            pkt.update(_parse_dns(payload))
    else:
        pkt["l4"] = f"proto-{proto}"
    return pkt


def parse_pcap(path: str, engine: str = "python") -> list[dict]:
    """Parse every frame in a pcap into decoded dicts."""
    if engine == "scapy":
        return _parse_with_scapy(path)
    return [parse_frame(r.data) for r in read_pcap(path)]


def summarize(packets: list[dict]) -> dict:
    """Aggregate per-protocol counts and a few headline facts for metrics.json."""
    l3 = Counter(p.get("l3", "?") for p in packets)
    l4 = Counter(p.get("l4", "?") for p in packets if p.get("l3") == "IPv4")
    l7 = Counter(p["l7"] for p in packets if "l7" in p)
    tcp_flags = Counter(p["flags"] for p in packets if p.get("l4") == "TCP")
    dns_names = sorted({p["qname"] for p in packets if p.get("l7") == "DNS" and "qname" in p})
    http_paths = sorted({p["http_path"] for p in packets if "http_path" in p})
    tls_handshakes = sorted({p["tls_handshake"] for p in packets if "tls_handshake" in p})
    saw_handshake = {"SYN", "SYN/ACK", "ACK"}.issubset(set(tcp_flags))
    return {
        "total_packets": len(packets),
        "l3_protocols": dict(l3),
        "l4_protocols": dict(l4),
        "l7_protocols": dict(l7),
        "tcp_flag_combos": dict(tcp_flags),
        "dns_queried_names": dns_names,
        "http_paths": http_paths,
        "tls_handshake_msgs": tls_handshakes,
        "tcp_3way_handshake_observed": saw_handshake,
    }


def _parse_with_scapy(path: str) -> list[dict]:  # pragma: no cover - optional path
    try:
        from scapy.all import DNS, IP, TCP, UDP, rdpcap  # type: ignore  # noqa: I001
    except ImportError as e:
        raise RuntimeError(
            "scapy is not installed; use engine='python' (the default offline path)."
        ) from e
    out: list[dict] = []
    for pkt in rdpcap(path):
        d: dict = {}
        if pkt.haslayer(IP):
            d.update({"l3": "IPv4", "src_ip": pkt[IP].src, "dst_ip": pkt[IP].dst})
        if pkt.haslayer(TCP):
            d.update({"l4": "TCP", "sport": int(pkt[TCP].sport), "dport": int(pkt[TCP].dport),
                      "flags": str(pkt[TCP].flags)})
        elif pkt.haslayer(UDP):
            d.update({"l4": "UDP", "sport": int(pkt[UDP].sport), "dport": int(pkt[UDP].dport)})
        if pkt.haslayer(DNS):
            d["l7"] = "DNS"
        out.append(d)
    return out
