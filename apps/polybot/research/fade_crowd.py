"""Fade the crowd: when many traders of the point-in-time monthly pool have bought an outcome, buy the other side.

    python fade_crowd.py DATA_DIR [--follow]

The pool is the monthly top 200 rebuilt day by day from resolved P&L (no hindsight). For each outcome, the first moment
k or more pool members other than the signaller had bought it in the previous 48 hours (prep3's `crowd`) is one trade:
buy the complementary outcome at 1 − mid plus the book's cost for $5, taker fee paid, held to resolution. Reported per
crowd size, for sports and the rest, before the game and in play, on Sep 15 → Oct 19 and the Oct 20 → 31 holdout, with
90% intervals over days, and through the bot's book (10 buys a day, 20 open, 72 hours per market).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.argv_full = list(sys.argv)
sys.argv = sys.argv[:2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fillmodel3 import FillModel  # noqa: E402

DATA = Path(sys.argv[1])
fm = FillModel(DATA / "fill_model.json")
rng = np.random.default_rng(17)
P = pd.read_parquet(DATA / "panel.parquet", columns=["in_pool", "crowd", "mid_d", "cost", "p_f", "final", "fee_rate", "t_d", "resolved_at",
                                                    "cat", "f_in_game", "asset", "cid", "ev", "hours" if False else "end"])
P = P[(P.in_pool == 1)].sort_values("t_d").reset_index(drop=True)
T0 = pd.Timestamp("2026-09-15", tz="UTC").timestamp()
SE = pd.Timestamp("2026-10-20", tz="UTC").timestamp()
P = P[P.t_d >= T0]


def trades_for(k):
    q = P[P.crowd >= k]
    first = q.drop_duplicates("asset", keep="first").copy()   # one trade per outcome: the moment the crowd first reached k
    hours = ((first.end - first.t_d) / 3600).clip(lower=0).fillna(720).values
    opp_mid = 1 - first.mid_d.values
    cost = np.array([fm.cost(c, m, h, g > 0.5) for c, m, h, g in zip(first.cat.values, opp_mid, hours, first.f_in_game.values)])
    p = np.clip(opp_mid + cost, 0.01, 0.99)
    win = 1 - first.final.values
    fee = first.fee_rate.values
    first["opp_p"] = p
    first["ret_fade"] = win / p - 1 - fee * (1 - p)
    first["ret_follow"] = first.final.values / first.p_f.values - 1 - fee * (1 - first.p_f.values)
    return first[(p >= 0.05) & (p <= 0.95)]


def ci(df, col):
    d = (df.t_d // 86400).astype(int).values
    r = df[col].values
    ud = np.unique(d)
    if len(ud) < 3:
        return float("nan"), float("nan")
    s = np.array([r[d == x].sum() for x in ud])
    c = np.array([(d == x).sum() for x in ud])
    dr = rng.integers(0, len(ud), (4000, len(ud)))
    bm = s[dr].sum(1) / c[dr].sum(1)
    return np.quantile(bm, 0.05), np.quantile(bm, 0.95)


COL = "ret_follow" if "--follow" in sys.argv_full else "ret_fade"


def row(name, df):
    if len(df) < 20:
        print(f"  {name:40s} n {len(df)}")
        return
    lo, hi = ci(df, COL)
    if COL == "ret_follow":
        a, b = df[df.t_d < SE], df[df.t_d >= SE]
        print(f"  {name:40s} n {len(df):5d}  follow {100 * df[COL].mean():+6.1f}% [{100 * lo:+6.1f}, {100 * hi:+6.1f}]  select {100 * a[COL].mean():+6.1f}% "
              f"(n {len(a)})  holdout {100 * b[COL].mean():+6.1f}% (n {len(b)})  paid {100 * df.p_f.mean():.0f}¢")
        return
    a, b = df[df.t_d < SE], df[df.t_d >= SE]
    print(f"  {name:40s} n {len(df):5d}  fade {100 * df.ret_fade.mean():+6.1f}% [{100 * lo:+6.1f}, {100 * hi:+6.1f}]  select {100 * a.ret_fade.mean():+6.1f}% "
          f"holdout {100 * b.ret_fade.mean():+6.1f}% (n {len(b)})  | following them {100 * df.ret_follow.mean():+6.1f}%  paid {100 * df.opp_p.mean():.0f}¢")


def book(df, per_day=10, max_open=20):
    df = df.sort_values("t_d")
    open_, buys, acted, out = [], [], {}, []
    for _, r in df.iterrows():
        t = r.t_d
        open_ = [o for o in open_ if o > t]
        buys = [b for b in buys if b > t - 86400]
        if len(open_) >= max_open or len(buys) >= per_day or (r.cid in acted and t - acted[r.cid] < 72 * 3600):
            continue
        open_.append(r.resolved_at)
        buys.append(t)
        acted[r.cid] = t
        out.append(r)
    return pd.DataFrame(out)


for k in ((2, 3, 4, 5) if COL == "ret_follow" else (3, 5, 8)):
    df = trades_for(k)
    print(f"\ncrowd ≥ {k} (≥ {k + 1} pool traders on the outcome): {len(df)} outcomes")
    row("all", df)
    sp = df.cat == "sports"
    row("sports", df[sp])
    row("sports, before the game", df[sp & (df.f_in_game < 0.5)])
    row("sports, in play", df[sp & (df.f_in_game >= 0.5)])
    row("not sports", df[~sp])
    row("crowd's outcome priced 30–70¢", df[(df.mid_d >= 0.3) & (df.mid_d <= 0.7)])
    row("crowd's outcome a favourite (>50¢)", df[df.mid_d > 0.5])
    row("crowd's outcome an underdog (<50¢)", df[df.mid_d <= 0.5])
    bk = book(df)
    if len(bk):
        lo, hi = ci(bk, COL)
        span = max(7.0, (bk.t_d.max() - bk.t_d.min()) / 86400)
        print(f"  through the bot's book: {len(bk)} trades, {100 * bk[COL].mean():+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}], "
              f"${5 * bk[COL].sum() / span * 7:+.1f} a week at $5; select {100 * bk[bk.t_d < SE][COL].mean():+.1f}%, "
              f"holdout {100 * bk[bk.t_d >= SE][COL].mean():+.1f}%")
