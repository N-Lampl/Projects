"""Fast smoke tests (run in CI). One slow end-to-end test is marked @slow and
excluded from CI via `-m "not slow"`.
"""

from __future__ import annotations

import socket
import struct

from netlabs import (
    build_scenario,
    generate_pcap,
    parse_frame,
    parse_pcap,
    protocols as P,
    read_pcap,
    set_seed,
    summarize,
    write_pcap,
)


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_ipv4_checksum_is_valid():
    """A correct IPv4 header checksums to zero when re-summed over the header."""
    pkt = P.ipv4_packet("10.0.0.1", "10.0.0.2", P.IPPROTO_TCP, b"payload")
    header = pkt[:20]
    assert P.checksum16(header) == 0


def test_tcp_pseudo_header_checksum_is_valid():
    seg = P.tcp_segment("10.0.0.1", "10.0.0.2", 1234, 80, 1, 0, P.SYN, b"hello")
    pseudo = struct.pack("!4s4sBBH", socket.inet_aton("10.0.0.1"),
                         socket.inet_aton("10.0.0.2"), 0, P.IPPROTO_TCP, len(seg))
    assert P.checksum16(pseudo + seg) == 0


def test_dns_query_roundtrips_through_parser():
    seg = P.udp_segment("10.0.0.10", "10.0.0.1", 5000, 53, P.dns_query(0xABCD, "example.com"))
    ip = P.ipv4_packet("10.0.0.10", "10.0.0.1", P.IPPROTO_UDP, seg)
    frame = P.eth_frame("02:0:0:0:0:2".replace(":0", ":00"), "02:00:00:00:00:01", ip)
    p = parse_frame(frame)
    assert p["l4"] == "UDP" and p["l7"] == "DNS"
    assert p["qname"] == "example.com" and p["dns_qr"] == "query"


def test_dns_response_answer_ip_decoded():
    seg = P.udp_segment("10.0.0.1", "10.0.0.10", 53, 5000,
                        P.dns_response(0xABCD, "example.com", "93.184.216.34"))
    ip = P.ipv4_packet("10.0.0.1", "10.0.0.10", P.IPPROTO_UDP, seg)
    frame = P.eth_frame("02:00:00:00:00:01", "02:00:00:00:00:02", ip)
    p = parse_frame(frame)
    assert p["dns_qr"] == "response" and p["answer_ip"] == "93.184.216.34"


def test_http_request_parsed():
    payload = P.http_request("GET", "/index.html", "example.com")
    seg = P.tcp_segment("10.0.0.10", "93.184.216.34", 49152, 80, 1, 1, P.PSH | P.ACK, payload)
    ip = P.ipv4_packet("10.0.0.10", "93.184.216.34", P.IPPROTO_TCP, seg)
    p = parse_frame(P.eth_frame("02:00:00:00:00:02", "02:00:00:00:00:01", ip))
    assert p["l7"] == "HTTP" and p["http_method"] == "GET" and p["http_path"] == "/index.html"


def test_tls_client_hello_handshake_type():
    payload = P.tls_client_hello("example.com")
    seg = P.tcp_segment("10.0.0.10", "93.184.216.34", 49153, 443, 1, 1, P.PSH | P.ACK, payload)
    ip = P.ipv4_packet("10.0.0.10", "93.184.216.34", P.IPPROTO_TCP, seg)
    p = parse_frame(P.eth_frame("02:00:00:00:00:02", "02:00:00:00:00:01", ip))
    assert p["l7"] == "TLS" and p["tls_handshake"] == "ClientHello"


def test_pcap_write_read_roundtrip(tmp_path):
    path = str(tmp_path / "t.pcap")
    recs = build_scenario()
    write_pcap(path, recs)
    back = read_pcap(path)
    assert len(back) == len(recs)
    assert back[0].data == recs[0].data


def test_scenario_summary_invariants(tmp_path):
    path = str(tmp_path / "s.pcap")
    n = generate_pcap(path)
    packets = parse_pcap(path)
    assert len(packets) == n
    s = summarize(packets)
    # The scenario must contain every layer/protocol the labs cover.
    assert s["l3_protocols"].get("IPv4", 0) == n
    assert {"TCP", "UDP"} <= set(s["l4_protocols"])
    assert {"DNS", "HTTP", "TLS"} <= set(s["l7_protocols"])
    assert s["tcp_3way_handshake_observed"] is True
    assert "example.com" in s["dns_queried_names"]


import pytest  # noqa: E402


@pytest.mark.slow
def test_end_to_end_generate_parse_metrics(tmp_path):
    """Full path: generate a pcap, parse it, and confirm a coherent summary."""
    path = str(tmp_path / "e2e.pcap")
    n = generate_pcap(path)
    packets = parse_pcap(path)
    s = summarize(packets)
    assert s["total_packets"] == n == len(packets)
    assert "ClientHello" in s["tls_handshake_msgs"]
    assert "/index.html" in s["http_paths"]
    # Every IPv4 header we wrote must verify (checksum re-sums to zero).
    for r in read_pcap(path):
        ihl = (r.data[14] & 0x0F) * 4
        assert P.checksum16(r.data[14:14 + ihl]) == 0
