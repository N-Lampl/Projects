# data/

There is **no external dataset** to download. The pcap analysed by this lab is
**self-generated and synthetic** — `scripts/make_pcap.py` (or `make pcap`) hand-crafts
real wire-format Ethernet/IPv4/TCP/UDP/DNS/HTTP/TLS frames and writes them to
`data/synthetic.pcap` in the classic libpcap format.

## Generate it

```bash
make pcap                       # pure-python engine (offline default)
# or, with the optional enhanced engine:
python3 scripts/make_pcap.py --engine scapy   # requires: pip install scapy==2.5.0
```

The resulting `data/synthetic.pcap` is a valid pcap you can also open in Wireshark or
`tcpdump -r data/synthetic.pcap`.

## License / ethics

The trace contains only synthetic traffic between RFC-1918 / documentation addresses
(`10.0.0.0/8`, `93.184.216.34`). No real hosts are contacted and **no live sniffing**
ever happens. Generated artifacts are git-ignored — never commit captures of real traffic.
