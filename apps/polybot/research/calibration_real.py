"""Are sports prices right at the prices people actually paid? Calibration and taker returns from real buys.

    python calibration_real.py DATA_DIR

The 5-minute price history of a thin market is not a price anyone could trade at, so this uses the panel's real buys
(prep3's signals: every buy of the 10,325 collected wallets, `leader_px` = what the wallet paid, controls = wallets no
board picked). One row per outcome (its first buy in the slice), sports only, resolved, priced 2–98¢. Shows won − paid
by price band before the game (last hour) and in play, then the return per dollar a taker would have made after the
taker fee, by league and side, on Sep 15 → Oct 19 and on Oct 20 → 31.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
U = dt.timezone.utc
TEST0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp()
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
rng = np.random.default_rng(3)

P = pd.read_parquet(DATA / "panel.parquet", columns=["cat", "final", "leader_px", "f_in_game", "f_pre_game_h_log", "fee_rate", "control",
                                                    "t_d", "asset", "ev"])
P = P[(P.cat == "sports") & P.final.notna() & P.leader_px.between(0.02, 0.98)].copy()
evs = pd.read_parquet(DATA / "panel_evs.parquet").set_index("code").ev
P["league"] = P.ev.map(evs).fillna("").str.split("-").str[0]
P["pre_h"] = np.expm1(P.f_pre_game_h_log)
P["ret"] = P.final / P.leader_px - 1 - P.fee_rate.fillna(0) * (1 - P.leader_px)
P["day"] = (P.t_d // 86400).astype(int)
BANDS = ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.98))


def boot(df):
    if len(df) < 30:
        return np.nan, np.nan
    g = df.groupby("day").ret.agg(["sum", "count"])
    dr = rng.integers(0, len(g), size=(2000, len(g)))
    bm = g["sum"].values[dr].sum(1) / g["count"].values[dr].sum(1)
    return np.quantile(bm, .05), np.quantile(bm, .95)


def calib(df, name):
    df = df.sort_values("t_d").drop_duplicates("asset")
    print(f"\n{name}: {len(df)} outcomes")
    print(f"  {'band':>8s} {'n':>6s} {'won':>6s} {'paid':>6s} {'gap':>9s} | taker return after fee [90% CI]")
    for lo, hi in BANDS:
        s = df[(df.leader_px >= lo) & (df.leader_px < hi)]
        if len(s) < 50:
            continue
        a, b = boot(s)
        print(f"  {100 * lo:3.0f}–{100 * hi:2.0f}¢ {len(s):6d} {100 * s.final.mean():5.1f}% {100 * s.leader_px.mean():5.1f}¢ {100 * (s.final - s.leader_px).mean():+6.2f} pts"
              f" | {100 * s.ret.mean():+6.1f}% [{100 * a:+5.1f}, {100 * b:+5.1f}]")


pre = P[(P.f_in_game == 0) & (P.pre_h <= 1.0)]
inp = P[P.f_in_game == 1]
calib(pre, "before the game (last hour), every wallet")
calib(pre[pre.control == 1], "before the game (last hour), control wallets only")
calib(inp, "in play, every wallet")

print("\nby league and side, before the game (last hour) and in play, every wallet, first buy per outcome; "
      "taker return after fee, Sep 15 → Oct 19 | Oct 20 → 31")
Q = P[(P.f_in_game == 1) | (P.pre_h <= 1.0)]
Q = Q[Q.t_d >= TEST0].sort_values("t_d").drop_duplicates("asset")
Q["side"] = np.where(Q.leader_px < 0.5, "underdog", "favourite")
top = Q.league.value_counts()
top = top[top >= 1500].index
both = 0
cells = 0
for lg in top:
    for side in ("underdog", "favourite"):
        s = Q[(Q.league == lg) & (Q.side == side)]
        a, b = s[s.t_d < SELECT_END], s[s.t_d >= SELECT_END]
        if len(a) < 100 or len(b) < 30:
            continue
        cells += 1
        both += (a.ret.mean() > 0) and (b.ret.mean() > 0)
        print(f"  {lg:8s} {side:9s} {len(a):6d} {100 * a.ret.mean():+6.1f}% | {len(b):5d} {100 * b.ret.mean():+6.1f}%")
print(f"{both} of {cells} league × side cells are positive in both periods")
