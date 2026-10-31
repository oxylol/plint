"""Backtest the live `day4` setup's rule on the research data: four day-board leaders agree, the follower buys.

    python check_day4.py DATA_DIR [WALLETS.json]

The day board is rebuilt per day from resolved P&L (top 25 by the previous day's P&L among the 10k wallets). A
consensus forms when at least 4 of them net-bought an outcome (stance ≥ $25) within 24 hours, minus those on the
other side; the follower buys 5 minutes after the fourth at the midpoint plus the book's cost, with the live setup's
filters (5–90¢, at most 5¢ above the leaders, ≥ 1 hour to the end, one buy per outcome per 72 h), held to resolution.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common3 import gz_lines  # noqa: E402
from fillmodel3 import FillModel  # noqa: E402

DATA = Path(sys.argv[1])
DAY = 86400
rng = np.random.default_rng(3)

FIXED = Path(sys.argv[2]) if len(sys.argv) > 2 else None   # a JSON list of wallets: test that fixed group instead
if FIXED:
    members = set(json.loads(FIXED.read_text()))
    board = None
    print(f"fixed group: {len(members)} wallets (the ones the live day/day4 setups copied, 28–31 Oct)")
else:
    wd = pd.read_parquet(DATA / "wallet_days.parquet")
    wd["rank1"] = wd.groupby("day").pnl1.rank(ascending=False, method="first")
    board = set(zip(wd.wallet[(wd.rank1 <= 25) & (wd.pnl1 > 0)], wd.day[(wd.rank1 <= 25) & (wd.pnl1 > 0)]))
    print(f"rebuilt day boards: {len(board)} wallet-days, {len({w for w, _ in board})} distinct wallets")

dset = pds.dataset(DATA / "events", format="parquet")
cols = ["wallet", "cid", "asset", "t_d", "final", "resolved_at", "end", "fee_rate", "cat", "leader_px", "stance", "f_in_game"]
parts = []
for b in dset.to_batches(columns=cols, filter=(pds.field("has_market") == True) & (pds.field("stance") >= 25), batch_size=300_000):  # noqa: E712
    df = b.to_pandas()
    day = (df.t_d // DAY).astype(int)
    keep = np.array([(w in members) if board is None else ((w, d) in board) for w, d in zip(df.wallet.values, day.values)])
    parts.append(df[keep & df.final.notna().values])
ev = pd.concat(parts, ignore_index=True).sort_values("t_d").reset_index(drop=True)
print(f"{len(ev)} signals from day-board leaders (stance ≥ $25), {ev.wallet.nunique()} wallets")

# consensus: ≥4 distinct leaders on this outcome in 24 h, minus distinct leaders on the other outcome of the market
by_cid = defaultdict(list)
for i, (c, a, w, t) in enumerate(zip(ev.cid.values, ev.asset.values, ev.wallet.values, ev.t_d.values)):
    by_cid[c].append((t, a, w, i))
cons = []
for c, lst in by_cid.items():
    last = {}
    for k, (t, a, w, i) in enumerate(lst):
        win = [x for x in lst[: k + 1] if t - x[0] <= 24 * 3600]
        agree = {x[2] for x in win if x[1] == a}
        against = {x[2] for x in win if x[1] != a} - agree
        if len(agree) - len(against) >= 4 and (a not in last or t - last[a] >= 72 * 3600):
            last[a] = t
            rows = [x[3] for x in win if x[1] == a]
            cons.append((i, t, a, float(np.mean(ev.leader_px.values[rows])), len(agree)))
print(f"{len(cons)} consensus moments (4+ agreeing, net of dissent)")

# the follower's midpoint 5 minutes after the 4th leader (t_d already includes those 5 minutes)
want = defaultdict(list)
for j, (i, t, a, lp, n) in enumerate(cons):
    want[a].append((t, j))
mid = np.full(len(cons), np.nan)
for p in sorted((DATA / "prices").glob("*.jsonl.gz")):
    for line in gz_lines(p):
        try:
            jl = json.loads(line)
        except ValueError:
            continue
        ws = want.get(jl["a"])
        if not ws or not jl["p"]:
            continue
        tt = np.array([x[0] for x in jl["p"]], dtype=float)
        pp = np.array([x[1] for x in jl["p"]], dtype=float)
        w_ = 900 if jl["b"] <= 300 else 5400
        for t, j in ws:
            if np.isfinite(mid[j]) or not (jl["lo"] <= t <= jl["hi"]):
                continue
            k = np.searchsorted(tt, t)
            if k < len(tt) and tt[k] - t <= w_:
                mid[j] = pp[k]
fm = FillModel(DATA / "fill_model.json")
rows = []
for j, (i, t, a, lp, n) in enumerate(cons):
    if not np.isfinite(mid[j]):
        continue
    hours = (ev.end.values[i] - t) / 3600 if ev.end.values[i] == ev.end.values[i] else 24 * 30
    cat, game = ev.cat.values[i], ev.f_in_game.values[i] > 0.5
    p = min(0.99, mid[j] + fm.cost(cat, mid[j], max(0.0, hours), game))
    if not (0.05 <= p <= 0.90) or p > lp + 0.05 or hours < 1:
        continue
    fee = ev.fee_rate.values[i]
    r = ev.final.values[i] / p - 1 - fee * (1 - p)
    rows.append({"t": t, "day": int(t // DAY), "p": p, "r": r, "cat": cat, "in_game": game, "n": n, "won": ev.final.values[i]})
R = pd.DataFrame(rows)
print(f"{len(R)} buys after the live setup's filters ({len(cons) - int(np.isfinite(mid).sum())} had no price on file)\n")


def show(label, d):
    if len(d) < 3:
        print(f"  {label:34s} {len(d):4d} buys")
        return
    ud = np.unique(d.day.values)
    sums = np.array([d.r.values[d.day.values == x].sum() for x in ud])
    cnts = np.array([(d.day.values == x).sum() for x in ud])
    dr = rng.integers(0, len(ud), size=(4000, len(ud)))
    bm = sums[dr].sum(1) / cnts[dr].sum(1)
    print(f"  {label:34s} {len(d):4d} buys on {len(ud):2d} days · mean {100 * d.r.mean():+6.1f}% [90% CI {100 * np.quantile(bm, .05):+6.1f}, "
          f"{100 * np.quantile(bm, .95):+6.1f}] · won {100 * d.won.mean():.0f}% at {100 * d.p.mean():.0f}¢ avg")


s0 = pd.Timestamp("2026-09-15", tz="UTC").timestamp()
live = pd.Timestamp("2026-10-28", tz="UTC").timestamp()
show("Sep 15 → Oct 31, all", R[R.t >= s0])
show("  before the live run (→ Oct 27)", R[(R.t >= s0) & (R.t < live)])
show("  live paper period (Oct 28 →)", R[R.t >= live])
show("  sports", R[(R.t >= s0) & (R.cat == "sports")])
show("  not sports", R[(R.t >= s0) & (R.cat != "sports")])
show("  in play", R[(R.t >= s0) & R.in_game])
show("  before the game", R[(R.t >= s0) & ~R.in_game & (R.cat == "sports")])
show("  paid 40–60¢", R[(R.t >= s0) & (R.p >= 0.4) & (R.p <= 0.6)])
show("  paid > 60¢", R[(R.t >= s0) & (R.p > 0.6)])
show("  paid < 40¢", R[(R.t >= s0) & (R.p < 0.4)])
for wk in sorted(set(((R.t[R.t >= s0] - s0) // (7 * DAY)).astype(int))):
    d = R[(R.t >= s0 + wk * 7 * DAY) & (R.t < s0 + (wk + 1) * 7 * DAY)]
    show(f"  week of {pd.to_datetime(s0 + wk * 7 * DAY, unit='s'):%b %d}", d)
