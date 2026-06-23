"""Synthetic OHLCV generation with injected market-manipulation events.

A deterministic, seeded geometric-Brownian-motion (GBM) price series with a
matching synthetic volume process. Two families of manipulation are injected:

  * PUMP-AND-DUMP -- a coordinated ramp in price + volume over a few days,
    followed by a sharp crash back toward (or below) the pre-pump level.
  * SPOOFING-LIKE volume burst -- a short, sharp 1-2 bar volume spike with only
    a tiny transient price move (orders flashed then cancelled).

Every event is recorded with its exact bar span so detection precision/recall
and lead-time can be measured against ground truth. There is no I/O and no
download: this is the mandatory offline fallback for the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Event:
    """One injected manipulation, in integer bar indices (inclusive start)."""

    kind: str  # "pump_dump" | "spoof"
    start: int  # first manipulated bar
    end: int  # last manipulated bar (inclusive)
    peak: int  # bar of maximum disturbance (used for lead-time scoring)


@dataclass
class Series:
    """A generated OHLCV frame plus the ground-truth event list."""

    df: pd.DataFrame
    events: list[Event] = field(default_factory=list)

    @property
    def label(self) -> np.ndarray:
        """Per-bar binary ground truth: 1 inside any manipulation window."""
        y = np.zeros(len(self.df), dtype=int)
        for ev in self.events:
            y[ev.start : ev.end + 1] = 1
        return y


def _gbm(n: int, rng: np.random.Generator, s0: float, mu: float, sigma: float) -> np.ndarray:
    """Daily geometric Brownian motion closing prices."""
    dt = 1.0 / 252.0
    shocks = rng.standard_normal(n)
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    return s0 * np.exp(np.cumsum(log_ret))


def generate(
    n: int = 1500,
    n_pumps: int = 6,
    n_spoofs: int = 8,
    seed: int = 42,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.35,
) -> Series:
    """Generate a deterministic OHLCV series with injected manipulations.

    Returns a :class:`Series` whose ``df`` has columns
    ``[date, open, high, low, close, volume]`` and whose ``events`` list is the
    ground truth used to score detections.
    """
    rng = np.random.default_rng(seed)

    close = _gbm(n, rng, s0, mu, sigma)

    # Baseline volume: log-normal around a slowly-drifting mean (~1M shares).
    base = 1_000_000.0 * (1.0 + 0.15 * np.sin(np.linspace(0, 6 * np.pi, n)))
    volume = base * np.exp(rng.normal(0.0, 0.25, n))

    events: list[Event] = []

    # Keep events apart so windows do not overlap and lead-time is well defined.
    min_gap = 40
    used: list[tuple[int, int]] = []

    def _free_slot(width: int) -> int | None:
        for _ in range(200):
            start = int(rng.integers(min_gap, n - width - min_gap))
            span = (start - min_gap, start + width + min_gap)
            if all(span[1] < a or span[0] > b for a, b in used):
                used.append((start, start + width))
                return start
        return None

    # ---- PUMP-AND-DUMP: ramp price + volume, then crash. ----
    for _ in range(n_pumps):
        ramp = int(rng.integers(4, 8))  # build-up bars
        crash = int(rng.integers(2, 4))  # dump bars
        width = ramp + crash
        start = _free_slot(width)
        if start is None:
            continue

        pump_mag = rng.uniform(0.35, 0.70)  # total fractional run-up
        anchor = close[start - 1]
        # Compounding ramp up.
        for i in range(ramp):
            step = pump_mag / ramp
            close[start + i] = close[start + i - 1] * (1.0 + step)
            volume[start + i] *= rng.uniform(4.0, 9.0)  # coordinated buying
        peak = start + ramp - 1
        peak_price = close[peak]
        # Crash back toward (slightly below) the anchor.
        target = anchor * rng.uniform(0.90, 0.98)
        for j in range(crash):
            frac = (j + 1) / crash
            close[peak + 1 + j] = peak_price + (target - peak_price) * frac
            volume[peak + 1 + j] *= rng.uniform(3.0, 7.0)
        # Re-base the rest of the path so post-event drift continues from crash.
        drift = close[peak + crash] - close[start + width - 1]
        close[start + width :] += drift
        events.append(Event("pump_dump", start, start + width - 1, peak))

    # ---- SPOOFING-LIKE volume burst: huge volume, tiny price wobble. ----
    for _ in range(n_spoofs):
        width = int(rng.integers(1, 3))
        start = _free_slot(width + 2)
        if start is None:
            continue
        for i in range(width):
            volume[start + i] *= rng.uniform(8.0, 16.0)  # flashed orders
            # only a fleeting price tick, immediately reverted
            wobble = rng.choice([-1.0, 1.0]) * close[start + i] * rng.uniform(0.002, 0.006)
            close[start + i] += wobble
        events.append(Event("spoof", start, start + width - 1, start))

    close = np.maximum(close, 1.0)
    volume = np.maximum(volume, 1.0)

    # Build OHLC consistent with the close path and an intraday range.
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]
    rng_pct = np.abs(rng.normal(0.0, 0.01, n)) + 0.003
    high = np.maximum(open_, close) * (1.0 + rng_pct)
    low = np.minimum(open_, close) * (1.0 - rng_pct)

    dates = pd.bdate_range("2018-01-01", periods=n)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return Series(df=df, events=events)
