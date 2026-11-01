"""How far is the 5-minute price history from what people really paid? (Why the kick-off "edge" is not tradeable.)

    python kickoff_vs_real.py DATA_DIR

For every real buy in the panel (prep3's signals, `leader_px` = what the wallet paid) on a market in the kick-off set
(collect3_kickoff.py), within 3 hours before to 4 hours after kick-off, compares the price history's value for that
outcome at that moment with the price paid, by history-price band: the gap is the real cost above the "price" the
calibration table uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import datetime as dt

DATA = Path(sys.argv[1])


def ts(x):
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


K = DATA / "kickoff"
mk = json.loads((K / "markets.json").read_text())
side = {}
for cid, m in mk.items():
    side[m["tokens"][0]] = (cid, 0)
    side[m["tokens"][1]] = (cid, 1)
paths = {}
for line in (K / "prices.jsonl").open():
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d["p"]:
        v = sorted(d["p"])
        paths[d["cid"]] = (np.array([x[0] for x in v], float), np.array([x[1] for x in v]))
assets = pd.read_parquet(DATA / "panel_assets.parquet")
code2asset = dict(zip(assets.code, assets.asset if "asset" in assets else assets.iloc[:, 1]))
P = pd.read_parquet(DATA / "panel.parquet", columns=["asset", "leader_px", "t_fill", "t_d", "final", "control"])
P["a"] = P.asset.map(code2asset)
P = P[P.a.isin(side.keys())]
rows = []
for a, px, tf, td, fin, ctl in zip(P.a, P.leader_px, P.t_fill, P.t_d, P.final, P.control):
    cid, k = side[a]
    if cid not in paths:
        continue
    tt, pp = paths[cid]
    t = tf if np.isfinite(tf) else td - 300
    if not tt[0] <= t <= tt[-1]:
        continue
    j = np.searchsorted(tt, t, side="right") - 1
    h = pp[j] if k == 0 else 1 - pp[j]
    rows.append((h, px, fin, ctl, mk[cid]["sports_type"], float(mk[cid].get("vol") or 0), t < ts(mk[cid]["game_start"]), t - tt[j]))
R = pd.DataFrame(rows, columns=["h_", "paid", "final", "control", "type", "vol", "pre", "stale"])
print(f"{len(R)} real buys on kick-off-set markets inside the price window")
def table(R):
  print(f"\n{'history price':>14s} {'buys':>6s} {'history':>8s} {'paid':>6s} {'paid − history':>15s} {'|gap| > 3¢':>11s} | won   (vs history, vs paid)")
  for lo, hi in ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.98)):
    s = R[(R.h_ >= lo) & (R.h_ < hi)]
    if len(s) < 50:
        continue
    g = s.paid - s.h_
    print(f"{100 * lo:6.0f}–{100 * hi:2.0f}¢    {len(s):6d} {100 * s.h_.mean():7.1f}¢ {100 * s.paid.mean():5.1f}¢ {100 * g.mean():+10.2f}¢ "
          f"{100 * (g.abs() > 0.03).mean():10.0f}% | {100 * s.final.mean():5.1f}%  ({100 * (s.final - s.h_).mean():+5.2f}, {100 * (s.final - s.paid).mean():+5.2f} pts)")
print("\nall buys:"); table(R)
print("\nbefore kick-off (prices barely move, so a stale bucket cannot explain a gap):"); table(R[R.pre])
print("\nin play:"); table(R[~R.pre])
print("\nby market type and life volume, median |paid − history|:")
for t, s in R.groupby("type"):
    print(f"  {t:16s} {len(s):6d}  {100 * (s.paid - s.h_).abs().median():.2f}¢  (mean {100 * (s.paid - s.h_).abs().mean():.2f}¢)")
for lo, hi in ((5e3, 2e4), (2e4, 1e5), (1e5, 1e6), (1e6, 1e12)):
    s = R[(R.vol >= lo) & (R.vol < hi)]
    if len(s):
        print(f"  ${lo:>9,.0f}+ {len(s):6d}  {100 * (s.paid - s.h_).abs().median():.2f}¢  (mean {100 * (s.paid - s.h_).abs().mean():.2f}¢)")
