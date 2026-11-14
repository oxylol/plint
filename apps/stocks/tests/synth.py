"""A made-up market for the tests: real trading days, invented stocks, and effects planted on purpose.

What is planted, so tests can ask whether the machinery finds it:
- reversal: a stock's own move over the last 5 days is partly given back over the next days
- momentum: a slow drift in the direction of its last 120 days
- one stock splits 2-for-1 (S003), one pays a dividend (S004), one is halted for a week (S005), one lists
  late (S006), one stops trading for good (S007)
SPY is the common factor, BIL creeps up like a Treasury bill."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stocks.cal import trading_days


def market(n_stocks: int = 120, n_days: int = 700, seed: int = 0, reversal: float = 0.06, momentum: float = 0.0004,
           start: date = date(2018, 1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = trading_days(start, date(start.year + 6, 12, 31))[:n_days]
    d_int = np.array([d.year * 10000 + d.month * 100 + d.day for d in days], dtype=np.int32)
    T, N = len(days), n_stocks
    mkt = rng.normal(0.0004, 0.009, T)
    beta = rng.uniform(0.6, 1.5, N)
    vol = rng.uniform(0.012, 0.035, N)
    idio = np.zeros((T, N))
    eps = rng.normal(0, 1, (T, N)) * vol
    for t in range(T):
        rev = -reversal * idio[max(0, t - 5):t].sum(axis=0) if t else 0.0
        mom = momentum * np.sign(idio[max(0, t - 120):t].sum(axis=0)) if t > 20 else 0.0
        idio[t] = eps[t] + rev / 5.0 + mom
    lr = beta * mkt[:, None] + idio                                 # total log return, close to close
    share_ovn = rng.uniform(0.2, 0.6, (T, N))
    lov, lit = lr * share_ovn, lr * (1 - share_ovn)
    lit += rng.normal(0, 0.004, (T, N))
    lov -= lit - lr * (1 - share_ovn)                               # keep lov + lit = lr
    price0 = rng.uniform(15, 300, N)
    base_volume = np.exp(rng.normal(15.0, 1.0, N))                  # shares a day
    rows = []
    syms = [f"S{j:03d}" for j in range(N)]
    for j, s in enumerate(syms):
        c_prev, first, last = price0[j], 0, T
        if s == "S006":
            first = 260
        if s == "S007":
            last = T - 150
        for t in range(first, last):
            if s == "S005" and 300 <= t < 305:
                continue                                           # halted: no bar
            split = 2.0 if (s == "S003" and t == 400) else 1.0
            o = c_prev * np.exp(lov[t, j]) / split
            if s == "S004" and t == 410:
                o -= 0.02 * c_prev                                  # goes ex-dividend: 2 % paid out
            c = o * np.exp(lit[t, j])
            ret = np.exp(lr[t, j]) - 1.0 if t > first else np.nan
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
            v = base_volume[j] * split ** (t >= 400 and s == "S003") * np.exp(rng.normal(0, 0.35)) * (1 + 6 * abs(lr[t, j]))
            rows.append((d_int[t], s, o, hi, lo, c, v, (hi + lo + c) / 3, max(1, int(v / 180)), ret))
            c_prev = c
    for s, series, px in (("SPY", mkt, 270.0), ("BIL", np.full(T, 0.00015), 91.5)):
        c_prev = px
        for t in range(T):
            o = c_prev * np.exp(series[t] * 0.4)
            c = o * np.exp(series[t] * 0.6)
            rows.append((d_int[t], s, o, max(o, c) * 1.001, min(o, c) * 0.999, c, 6e7, (o + c) / 2, 300000,
                         np.exp(series[t]) - 1.0 if t else np.nan))
            c_prev = c
    return pd.DataFrame(rows, columns=["d", "sym", "o", "h", "l", "c", "v", "vw", "n", "ret"])


def panel(n_stocks: int = 120, n_days: int = 700, seed: int = 0, **kw):
    """A Panel of the made-up market with every invented stock in the universe from its first bar."""
    from stocks.panel import Panel
    p = Panel.from_frame(market(n_stocks, n_days, seed, **kw), "SPY")
    member = np.isfinite(p.c) | (np.maximum.accumulate(np.isfinite(p.c), axis=0))
    for s in ("SPY", "BIL"):
        member[:, p.ix[s]] = False
    p.member = member
    return p
