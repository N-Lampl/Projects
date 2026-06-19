"""Minimal classic-libpcap (.pcap) reader/writer — pure stdlib.

The file format is the classic global header + per-packet records that tcpdump,
Wireshark and scapy all understand, so the trace we emit is a real, openable .pcap.
We use LINKTYPE_ETHERNET (1) and microsecond timestamps.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PCAP_MAGIC = 0xA1B2C3D4
LINKTYPE_ETHERNET = 1


@dataclass
class PcapRecord:
    ts_sec: int
    ts_usec: int
    data: bytes  # the full link-layer frame (Ethernet)


def write_pcap(path: str, records: list[PcapRecord], snaplen: int = 65535) -> None:
    with open(path, "wb") as f:
        # Global header: magic, ver_major, ver_minor, thiszone, sigfigs, snaplen, network
        f.write(struct.pack("!IHHiIII", PCAP_MAGIC, 2, 4, 0, 0, snaplen, LINKTYPE_ETHERNET))
        for r in records:
            f.write(struct.pack("!IIII", r.ts_sec, r.ts_usec, len(r.data), len(r.data)))
            f.write(r.data)


def read_pcap(path: str) -> list[PcapRecord]:
    records: list[PcapRecord] = []
    with open(path, "rb") as f:
        global_hdr = f.read(24)
        if len(global_hdr) < 24:
            raise ValueError("truncated pcap global header")
        magic = struct.unpack("!I", global_hdr[:4])[0]
        if magic == PCAP_MAGIC:
            endian = "!"
        elif magic == 0xD4C3B2A1:  # byte-swapped little-endian capture
            endian = "<"
        else:
            raise ValueError(f"not a classic pcap file (magic={magic:#x})")
        while True:
            rec_hdr = f.read(16)
            if len(rec_hdr) < 16:
                break
            ts_sec, ts_usec, incl_len, _orig_len = struct.unpack(endian + "IIII", rec_hdr)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            records.append(PcapRecord(ts_sec, ts_usec, data))
    return records
