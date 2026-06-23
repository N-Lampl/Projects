# data/ (git-ignored)

**Nothing in this folder is committed.** The default run path needs *no* data: the
OHLCV series is generated deterministically in-repo by
[`src/market_manip/data.py`](../src/market_manip/data.py) (seeded GBM price +
synthetic volume with injected pump-and-dump and spoofing-like events). Just run
`make detect`.

## Optional: swap in real OHLCV

To re-run the same feature/detector stack on real market data, drop a CSV with
columns `date,open,high,low,close,volume` here and adapt `scripts/run_detect.py`
to load it instead of `generate()`. Real public sources:

- **Nasdaq ITCH / LOBSTER** sample limit-order-book data (academic, free sample
  files) — closest to the spoofing/quote-flashing signal. <https://lobsterdata.com/>
- **Stooq** free daily OHLCV CSV downloads (equities, FX, indices).
  e.g. `https://stooq.com/q/d/l/?s=aapl.us&i=d` (research use; check site terms).
- **Yahoo Finance** daily OHLCV via the `yfinance` package (unofficial; rate-limited).
- **Kaggle** "stock market" / "pump and dump" datasets (per-dataset licenses).

Note: real labelled market-manipulation events are scarce and confidential, which
is exactly why this project injects *known* manipulations into synthetic data so
detection precision/recall and lead-time can be measured against ground truth.
All data is git-ignored — never commit downloaded market data.
