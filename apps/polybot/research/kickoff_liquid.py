"""The pre-game underdog gap, only where real buys confirm the price (the bot's tradeable universe).

    python kickoff_liquid.py DATA_DIR [min_buys=3]

For each kick-off-set market (collect3_kickoff.py), counts the panel's real buys of either outcome in the hour before
kick-off and their median gap to the price history. Then the calibration of the underdog at 5 minutes before kick-off
(history price + the fill model's pre-game cost + taker fee, held to resolution) for markets with ≥ min_buys real buys
(the price is a real price) and for the rest, by price band, Sep 3 → Oct 19 and Oct 20 → 31.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
MIN_BUYS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
K = DATA / "kickoff"
U = dt.timezone.utc
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
FILL = json.loads((DATA / "fill_model.json").read_text())["cells"]
rng = np.random.default_rng(9)


def ts(x):
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def band(p):
    return "lt10" if p < 0.10 else "10_30" if p < 0.30 else "30_70" if p < 0.70 else "70_90" if p < 0.90 else "ge90"


mk = json.loads((K / "markets.json").read_text())
paths = {}
for line in (K / "prices.jsonl").open():
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d["p"]:
        v = sorted(d["p"])
        paths[d["cid"]] = (np.array([x[0] for x in v], float), np.array([x[1] for x in v]))
side = {}
for cid, m in mk.items():
    side[m["tokens"][0]] = (cid, 0)
    side[m["tokens"][1]] = (cid, 1)
assets = pd.read_parquet(DATA / "panel_assets.parquet")
code2asset = dict(zip(assets.code, assets.asset))
P = pd.read_parquet(DATA / "panel.parquet", columns=["asset", "leader_px", "t_fill"])
P["a"] = P.asset.map(code2asset)
P = P[P.a.isin(side.keys())]
nbuys, gaps = {}, {}
for a, px, t in zip(P.a, P.leader_px, P.t_fill):
    cid, k = side[a]
    g = ts(mk[cid]["game_start"])
    if not g - 3600 <= t < g or cid not in paths:
        continue
    tt, pp = paths[cid]
    j = np.searchsorted(tt, t, side="right") - 1
    if j < 0:
        continue
    h = pp[j] if k == 0 else 1 - pp[j]
    nbuys[cid] = nbuys.get(cid, 0) + 1
    gaps.setdefault(cid, []).append(px - h)
rows = []
for cid, (tt, pp) in paths.items():
    m = mk[cid]
    g = ts(m["game_start"])
    t = g - 300
    j = np.searchsorted(tt, t, side="right") - 1
    if j < 0 or t - tt[j] > 1800 or not 0.02 <= pp[j] <= 0.98:
        continue
    p0 = pp[j]
    y0 = 1.0 if m["prices"][0] > 0.5 else 0.0
    pu, yu = (p0, y0) if p0 <= 0.5 else (1 - p0, 1 - y0)
    cell = FILL.get(f"sports|{band(pu)}|lt6") or FILL["sports"]
    a = min(pu + cell["c5_med"], 0.995)
    rate = float(m.get("fee_rate") or 0)
    rows.append((t, pu, yu, yu / a - 1 - rate * (1 - a), nbuys.get(cid, 0), np.median(gaps[cid]) if cid in gaps else np.nan, m["sports_type"]))
D = pd.DataFrame(rows, columns=["t", "pu", "yu", "ret", "n", "gap", "type"])
D["day"] = (D.t // 86400).astype(int)
print(f"{len(D)} markets priced 5 min before kick-off; {(D.n >= MIN_BUYS).sum()} with ≥ {MIN_BUYS} real buys in the hour before "
      f"(median |paid − history| there {100 * D.gap[D.n >= MIN_BUYS].abs().median():.2f}¢)")


def boot(s):
    if len(s) < 30:
        return np.nan, np.nan
    g = s.groupby("day").ret.agg(["sum", "count"])
    dr = rng.integers(0, len(g), size=(2000, len(g)))
    bm = g["sum"].values[dr].sum(1) / g["count"].values[dr].sum(1)
    return np.quantile(bm, .05), np.quantile(bm, .95)


for name, S in ((f"real price (≥ {MIN_BUYS} buys)", D[D.n >= MIN_BUYS]), ("no / few real buys", D[D.n < MIN_BUYS])):
    print(f"\n{name}: {len(S)} markets")
    print(f"  {'underdog':>9s} {'n':>5s} {'won':>6s} {'price':>6s} {'gap':>7s} | buy underdog after cost+fee [90% CI] | to Oct 19 | Oct 20 →")
    for lo, hi in ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
        s = S[(S.pu >= lo) & (S.pu <= hi) & ((S.pu < hi) | (hi == 0.5))]
        if len(s) < 30:
            continue
        a, b = boot(s)
        s1, s2 = s[s.t < SELECT_END], s[s.t >= SELECT_END]
        print(f"  {100 * lo:3.0f}–{100 * hi:2.0f}¢  {len(s):5d} {100 * s.yu.mean():5.1f}% {100 * s.pu.mean():5.1f}¢ {100 * (s.yu - s.pu).mean():+5.2f} | "
              f"{100 * s.ret.mean():+6.1f}% [{100 * a:+5.1f}, {100 * b:+5.1f}] | {len(s1):4d} {100 * s1.ret.mean():+6.1f}% | {len(s2):4d} {100 * s2.ret.mean():+6.1f}%")
    for t_, s in S.groupby("type"):
        print(f"    {t_:16s} {len(s):5d} underdog won − price {100 * (s.yu - s.pu).mean():+5.2f} pts, buy underdog {100 * s.ret.mean():+6.1f}%")
