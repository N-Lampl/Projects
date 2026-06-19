"""Hand-rolled, wire-format encoders for the protocols in the labs.

Everything here is pure stdlib (``struct``/``socket``) and produces *real* bytes that
match the on-the-wire layouts in the RFCs, so the bytes our generator writes are the
same bytes a real sniffer would capture and the same bytes our parser (or Wireshark,
or scapy) can decode. No live networking happens — we only build/parse byte strings.

Layers implemented (the Module-3 syllabus: TCP/IP, HTTP, DNS, TLS):
    Ethernet II            -- RFC 894 framing
    IPv4                   -- RFC 791 (with header checksum)
    TCP / UDP              -- RFC 793 / RFC 768 (with pseudo-header checksums)
    DNS                    -- RFC 1035 query + answer
    HTTP/1.1               -- request/response carried as a TCP payload
    TLS record            -- RFC 8446 record + ClientHello/ServerHello stubs
"""

from __future__ import annotations

import socket
import struct

# ---- L2: Ethernet ----------------------------------------------------------

ETH_TYPE_IPV4 = 0x0800


def mac_to_bytes(mac: str) -> bytes:
    return bytes(int(b, 16) for b in mac.split(":"))


def eth_frame(dst_mac: str, src_mac: str, payload: bytes, ethertype: int = ETH_TYPE_IPV4) -> bytes:
    return mac_to_bytes(dst_mac) + mac_to_bytes(src_mac) + struct.pack("!H", ethertype) + payload


# ---- checksums -------------------------------------------------------------

def checksum16(data: bytes) -> int:
    """The standard Internet 16-bit one's-complement checksum (RFC 1071)."""
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) + data[i + 1]
    total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


# ---- L3: IPv4 --------------------------------------------------------------

IPPROTO_TCP = 6
IPPROTO_UDP = 17


def ipv4_packet(src_ip: str, dst_ip: str, proto: int, payload: bytes, ttl: int = 64,
                ident: int = 0) -> bytes:
    version_ihl = (4 << 4) | 5  # IPv4, 5 * 4 = 20-byte header (no options)
    tos = 0
    total_len = 20 + len(payload)
    flags_frag = 0
    header = struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl, tos, total_len, ident, flags_frag, ttl, proto, 0,
        socket.inet_aton(src_ip), socket.inet_aton(dst_ip),
    )
    chk = checksum16(header)
    header = header[:10] + struct.pack("!H", chk) + header[12:]
    return header + payload


def _l4_checksum(src_ip: str, dst_ip: str, proto: int, segment: bytes) -> int:
    pseudo = struct.pack(
        "!4s4sBBH",
        socket.inet_aton(src_ip), socket.inet_aton(dst_ip), 0, proto, len(segment),
    )
    return checksum16(pseudo + segment)


# ---- L4: TCP ---------------------------------------------------------------

# TCP flag bits
FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10


def tcp_segment(src_ip: str, dst_ip: str, sport: int, dport: int, seq: int, ack: int,
                flags: int, payload: bytes = b"", window: int = 64240) -> bytes:
    data_offset = (5 << 4)  # 5 * 4 = 20-byte header, no options
    header = struct.pack(
        "!HHIIBBHHH",
        sport, dport, seq, ack, data_offset, flags, window, 0, 0,
    )
    chk = _l4_checksum(src_ip, dst_ip, IPPROTO_TCP, header + payload)
    header = header[:16] + struct.pack("!H", chk) + header[18:]
    return header + payload


# ---- L4: UDP ---------------------------------------------------------------

def udp_segment(src_ip: str, dst_ip: str, sport: int, dport: int, payload: bytes) -> bytes:
    length = 8 + len(payload)
    header = struct.pack("!HHHH", sport, dport, length, 0)
    chk = _l4_checksum(src_ip, dst_ip, IPPROTO_UDP, header + payload)
    if chk == 0:
        chk = 0xFFFF
    header = header[:6] + struct.pack("!H", chk) + header[8:]
    return header + payload


# ---- L7: DNS ---------------------------------------------------------------

def _encode_dns_name(name: str) -> bytes:
    out = b""
    for label in name.rstrip(".").split("."):
        out += bytes([len(label)]) + label.encode("ascii")
    return out + b"\x00"


def dns_query(txn_id: int, qname: str, qtype: int = 1) -> bytes:
    """A standard recursion-desired A query (qtype 1 = A, 28 = AAAA)."""
    header = struct.pack("!HHHHHH", txn_id, 0x0100, 1, 0, 0, 0)  # RD set
    question = _encode_dns_name(qname) + struct.pack("!HH", qtype, 1)  # class IN
    return header + question


def dns_response(txn_id: int, qname: str, answer_ip: str, qtype: int = 1, ttl: int = 300) -> bytes:
    header = struct.pack("!HHHHHH", txn_id, 0x8180, 1, 1, 0, 0)  # QR + RA, 1 answer
    question = _encode_dns_name(qname) + struct.pack("!HH", qtype, 1)
    # Answer: name pointer to offset 12 (start of question), type A, class IN, ttl, rdata
    answer = struct.pack("!HHHIH", 0xC00C, qtype, 1, ttl, 4) + socket.inet_aton(answer_ip)
    return header + question + answer


# ---- L7: HTTP --------------------------------------------------------------

def http_request(method: str, path: str, host: str, body: bytes = b"") -> bytes:
    head = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"User-Agent: netlabs-synthetic/1.0\r\n"
        f"Accept: */*\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
    ).encode("ascii")
    return head + body


def http_response(status: int, reason: str, body: bytes,
                  content_type: str = "text/html") -> bytes:
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Server: netlabs-synthetic/1.0\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
    ).encode("ascii")
    return head + body


# ---- L7: TLS ---------------------------------------------------------------

TLS_HANDSHAKE = 0x16
TLS_CLIENT_HELLO = 0x01
TLS_SERVER_HELLO = 0x02


def _tls_record(content_type: int, body: bytes, version: int = 0x0303) -> bytes:
    return struct.pack("!BHH", content_type, version, len(body)) + body


def tls_client_hello(sni: str) -> bytes:
    """A minimal but structurally-valid ClientHello carrying an SNI extension."""
    random_bytes = bytes(range(32))
    session_id = b""
    cipher_suites = struct.pack("!HH", 2, 0x1301)  # len, TLS_AES_128_GCM_SHA256
    compression = b"\x01\x00"  # 1 method, null
    # SNI extension (type 0)
    name = sni.encode("ascii")
    server_name = struct.pack("!BH", 0, len(name)) + name
    sni_list = struct.pack("!H", len(server_name)) + server_name
    sni_ext = struct.pack("!HH", 0x0000, len(sni_list)) + sni_list
    extensions = struct.pack("!H", len(sni_ext)) + sni_ext
    body = (
        struct.pack("!H", 0x0303) + random_bytes
        + struct.pack("!B", len(session_id)) + session_id
        + cipher_suites + compression + extensions
    )
    handshake = struct.pack("!B", TLS_CLIENT_HELLO) + struct.pack("!I", len(body))[1:] + body
    return _tls_record(TLS_HANDSHAKE, handshake)


def tls_server_hello() -> bytes:
    random_bytes = bytes(range(31, -1, -1))
    body = struct.pack("!H", 0x0303) + random_bytes + b"\x00" + struct.pack("!H", 0x1301) + b"\x00"
    handshake = struct.pack("!B", TLS_SERVER_HELLO) + struct.pack("!I", len(body))[1:] + body
    return _tls_record(TLS_HANDSHAKE, handshake)
