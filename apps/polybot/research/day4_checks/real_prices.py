"""day4's replay priced at real buys instead of the price history (the kick-off artifact check).

    python day4_checks/real_prices.py DATA_DIR [label]

The replay prices a follower's buy at the 5-minute history plus the fill model's cost (`est`). For every
day4-qualifying signal (as in consensus_filter.py), this looks up the real buys of the same outcome by the 10,325
collected wallets (the panel) within a window around the signal (default ±3 minutes), and re-prices the follower at their median price
(maker and taker fills mixed, so if anything it is a little cheap). Compares the return at `est` with the return at
the real price on the same signals.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
LABEL = sys.argv[2] if len(sys.argv) > 2 else "copied29"
LO, HI = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "-180,180").split(","))   # seconds around the signal
C = DATA / "group_cache"
U = dt.timezone.utc
TEST0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp()
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
rows = json.loads((C / f"consensus_rows_{LABEL}.json").read_text())
R = pd.DataFrame(rows)
R = R[(R.score >= 4) & R.est.between(0.05, 0.90) & (R.est <= R.avg + 0.05) & (R.t >= TEST0)].sort_values("t")
keep, last = [], {}
for i, a, t in zip(R.index, R.asset, R.t):
    if a in last and t - last[a] < 72 * 3600:
        continue
    last[a] = t
    keep.append(i)
R = R.loc[keep].copy()
assets = pd.read_parquet(DATA / "panel_assets.parquet")
a2c = dict(zip(assets.asset, assets.code))
P = pd.read_parquet(DATA / "panel.parquet", columns=["asset", "leader_px", "t_fill"])
P = P[P.asset.isin({a2c[a] for a in R.asset if a in a2c})]
by = {c: (g.t_fill.values, g.leader_px.values) for c, g in P.sort_values("t_fill").groupby("asset")}
real = []
for a, t in zip(R.asset, R.t):
    c = a2c.get(a)
    if c not in by:
        real.append(np.nan)
        continue
    tt, px = by[c]
    s = (tt >= t + LO) & (tt <= t + HI)
    real.append(float(np.median(px[s])) if s.any() else np.nan)
R["real"] = real
R["ret_real"] = R.final / R.real - 1 - R.fee * (1 - R.real)
ok = R.real.notna() & R.real.between(0.02, 0.98)
print(f"{len(R)} day4-qualifying signals from Sep 15, {ok.sum()} with real buys from {LO:+d} s to {HI:+d} s")
S = R[ok]
for name, s in (("all", S), ("underdogs (est < 50¢)", S[S.est < 0.5]), ("favourites (est ≥ 50¢)", S[S.est >= 0.5]),
                ("before the game", S[~S.in_play.astype(bool)]), ("in play", S[S.in_play.astype(bool)]),
                ("select (to Oct 19)", S[S.t < SELECT_END]), ("holdout (Oct 20 →)", S[S.t >= SELECT_END])):
    print(f"  {name:24s} {len(s):5d}  est {100 * s.est.mean():5.1f}¢  real {100 * s.real.mean():5.1f}¢  (real − est {100 * (s.real - s.est).mean():+5.2f}¢)  "
          f"return at est {100 * s.ret.mean():+5.1f}%  at real {100 * s.ret_real.mean():+5.1f}%")
