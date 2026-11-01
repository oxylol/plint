"""The shipped round-3 model's out-of-sample signals priced at real buys instead of the follower price (`p_f`).

    python day4_checks/real_prices_panel.py DATA_DIR

`oos_trades.parquet` holds analyze3.py's qualifying signals per family (row = panel row). For each, the median price
of the panel's real buys of the same outcome from the decision moment to 20 minutes after it replaces `p_f`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
O = pd.read_parquet(DATA / "oos_trades.parquet")
P = pd.read_parquet(DATA / "panel.parquet", columns=["asset", "leader_px", "t_fill", "t_d", "p_f", "final", "fee_rate", "cat"])
print(O.groupby(["family", "kind"]).size().to_string())
need = set(P.asset.values[O.row.values])
Q = P[P.asset.isin(need)].sort_values("t_fill")
by = {c: (g.t_fill.values, g.leader_px.values) for c, g in Q.groupby("asset")}
for (fam, kind), g in O.groupby(["family", "kind"]):
    r = P.iloc[g.row.values]
    real = []
    for a, t in zip(r.asset, r.t_d):
        tt, px = by[a]
        s = (tt >= t) & (tt <= t + 1200)
        real.append(np.median(px[s]) if s.any() else np.nan)
    real = np.array(real)
    ok = np.isfinite(real) & (real > 0.02) & (real < 0.98)
    rr = r[ok]
    ret_f = rr.final / rr.p_f - 1 - rr.fee_rate.fillna(0) * (1 - rr.p_f)
    ret_r = rr.final / real[ok] - 1 - rr.fee_rate.fillna(0) * (1 - real[ok])
    print(f"{fam:24s} {kind:8s} {len(g):6d} signals, {ok.sum():6d} with real buys: p_f {100 * rr.p_f.mean():5.1f}¢ real {100 * real[ok].mean():5.1f}¢ | "
          f"return at p_f {100 * ret_f.mean():+5.1f}%  at real {100 * ret_r.mean():+5.1f}%  (all signals at p_f {100 * g.ret.mean():+5.1f}%)")
