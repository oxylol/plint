"""Does Kalshi's price predict sports outcomes beyond Polymarket's, and can a Polymarket buyer use it after costs?

    python kalshi_vs_poly.py DATA_DIR

Uses collect_kalshi.py's matched games and candles and collect3_kickoff.py's Polymarket paths (first outcome, 5-minute
midpoint; before kick-off it matches real buys to 0.3¢, kickoff_vs_real.py). At Polymarket's 5-minute marks from
60 to 5 minutes before kick-off, the Kalshi midpoint (best bid + best ask) / 2 at that minute is compared with the
Polymarket price of the same team.

1. Information: log loss of Polymarket, Kalshi and their average; a logistic fit of the outcome on Polymarket's logit
   plus b × (Kalshi logit − Polymarket logit), game-clustered bootstrap for b.
2. Trading: buy the team (or the other side) on Polymarket when Kalshi's probability beats Polymarket's ask
   (price + 0.5¢) plus the taker fee by a cut. One bet per game, the first qualifying mark. Priced at the same mark
   (optimistic) and at the next mark 5 minutes later (what a bot polling Polymarket's history would have paid).
   Sep 3 → Oct 19 against Oct 20 → 31, day-block bootstrap.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
U = dt.timezone.utc
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
rng = np.random.default_rng(17)


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def lgt(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


matched = json.loads((DATA / "kalshi" / "matched.json").read_text())
candles = {}
for line in (DATA / "kalshi" / "candles.jsonl").open():
    d = json.loads(line)
    rows = [(t, float(b), float(a)) for t, b, a, _, _ in d["c"] if b is not None and a is not None and float(a) > float(b)]
    if rows:
        candles[d["ticker"]] = (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))
need = {m["cid"] for m in matched}
paths = {}
for line in (DATA / "kickoff" / "prices.jsonl").open():
    d = json.loads(line)
    if d["cid"] in need and d["p"]:
        v = sorted(d["p"])
        paths[d["cid"]] = (np.array([x[0] for x in v], float), np.array([x[1] for x in v]))

R = []          # one row per (Kalshi market, 5-minute mark)
for m in matched:
    if m["ticker"] not in candles or m["cid"] not in paths:
        continue
    kt, kb, ka = candles[m["ticker"]]
    pt, pp = paths[m["cid"]]
    g = ts(m["game_start"])
    y = 1.0 if m["final_poly"] > 0.5 else 0.0
    for j in range(len(pt)):
        s = pt[j]
        if not g - 3600 <= s <= g - 300:
            continue
        i = np.searchsorted(kt, s, side="right") - 1
        if i < 0 or s - kt[i] > 180 or ka[i] - kb[i] > 0.05:
            continue
        p = pp[j] if m["k"] == 0 else 1 - pp[j]
        p_next = (pp[j + 1] if m["k"] == 0 else 1 - pp[j + 1]) if j + 1 < len(pt) and pt[j + 1] - s <= 600 else np.nan
        if not 0.03 <= p <= 0.97:
            continue
        R.append({"ticker": m["ticker"], "game": m["event_ticker"], "lg": m["lg"], "t": s, "g": g, "p": p, "p_next": p_next,
                  "k": (kb[i] + ka[i]) / 2, "kspread": ka[i] - kb[i], "y": y, "fee": float(m.get("fee_rate") or 0)})
print(f"{len(matched)} matched markets, {len({r['ticker'] for r in R})} with both price series, {len(R)} market × 5-minute marks, "
      f"{len({r['game'] for r in R})} games")
p = np.array([r["p"] for r in R])
k = np.array([r["k"] for r in R])
y = np.array([r["y"] for r in R])
games = np.array([r["game"] for r in R])
gi = {gm: i for i, gm in enumerate(sorted(set(games)))}
gidx = np.array([gi[x] for x in games])
diff = k - p
print(f"Kalshi − Polymarket: mean {100 * diff.mean():+.2f}¢, median |gap| {100 * np.median(np.abs(diff)):.2f}¢, "
      f"|gap| > 2¢ in {100 * (np.abs(diff) > 0.02).mean():.0f}% of marks, > 4¢ in {100 * (np.abs(diff) > 0.04).mean():.0f}%")


def ll(q):
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))


print(f"\n## 1. Information\nlog loss: Polymarket {ll(p):.4f} · Kalshi {ll(k):.4f} · average {ll((p + k) / 2):.4f}")


def fit_b(idx):
    off = lgt(p[idx])
    x = lgt(k[idx]) - lgt(p[idx])
    b = 0.0
    a = 0.0
    for _ in range(30):
        q = sig(off + a + b * x)
        w = q * (1 - q)
        H = np.array([[w.sum() + 1e-6, (w * x).sum()], [(w * x).sum(), (w * x * x).sum() + 1e-6]])
        gr = np.array([(q - y[idx]).sum(), ((q - y[idx]) * x).sum()])
        a, b = np.array([a, b]) - np.linalg.solve(H, gr)
    return b


b0 = fit_b(np.arange(len(R)))
ng = len(gi)
members = defaultdict(list)
for i, g_ in enumerate(gidx):
    members[g_].append(i)
bs = []
for _ in range(400):
    pick = rng.integers(0, ng, ng)
    idx = np.concatenate([members[g_] for g_ in pick])
    bs.append(fit_b(idx))
lo, hi = np.quantile(bs, [0.05, 0.95])
print(f"outcome ~ Polymarket logit + b × (Kalshi logit − Polymarket logit): b = {b0:+.2f} [90% CI {lo:+.2f}, {hi:+.2f}] "
      f"(0 = Kalshi adds nothing, 1 = trust Kalshi fully)")

print("\n## 2. Trading on Polymarket when Kalshi disagrees (one bet per game, first qualifying mark)")
days = (np.array([r["t"] for r in R]) // 86400).astype(int)


def run(cut, entry):
    seen = set()
    out = []
    order = np.argsort([r["t"] for r in R], kind="stable")
    for i in order:
        r = R[i]
        if r["game"] in seen:
            continue
        for side in (1, 0):                      # 1: this team, 0: the other side
            q = r["k"] if side else 1 - r["k"]
            px = r["p"] if entry == "same" else r["p_next"]
            if np.isnan(px):
                continue
            a = (px if side else 1 - px) + 0.005
            if not 0.04 <= a <= 0.97:
                continue
            er = q / a - 1 - r["fee"] * (1 - a)
            if er >= cut:
                won = r["y"] if side else 1 - r["y"]
                out.append((i, won / a - 1 - r["fee"] * (1 - a), er))
                seen.add(r["game"])
                break
    return out


for entry in ("same", "next"):
    for cut in (0.0, 0.02, 0.04, 0.08):
        o = run(cut, entry)
        if len(o) < 10:
            print(f"  entry {entry:4s} cut {cut:+.2f}: {len(o)} bets")
            continue
        ii = np.array([x[0] for x in o])
        rr = np.array([x[1] for x in o])
        dd = days[ii]
        ud, inv = np.unique(dd, return_inverse=True)
        sums, cnts = np.bincount(inv, rr), np.bincount(inv)
        dr = rng.integers(0, len(ud), size=(3000, len(ud)))
        bm = sums[dr].sum(1) / cnts[dr].sum(1)
        tt = np.array([R[i]["t"] for i in ii])
        sel = tt < SELECT_END
        print(f"  entry {entry:4s} cut {cut:+.2f}: {len(rr):4d} bets, {100 * rr.mean():+6.1f}% a bet [90% CI {100 * np.quantile(bm, .05):+5.1f}, "
              f"{100 * np.quantile(bm, .95):+5.1f}], expected {100 * np.mean([x[2] for x in o]):+5.1f}% | select {sel.sum():3d} {100 * rr[sel].mean():+6.1f}% | "
              f"holdout {(~sel).sum():3d} {100 * rr[~sel].mean():+6.1f}% | ${35 * rr.sum() / max(1, len(np.unique(days))) :+.1f}/week at $5 (no cap)")
print("\nby league (cut +0.02, next-mark entry):")
o = run(0.02, "next")
for lg in sorted({R[x[0]]["lg"] for x in o}):
    rr = np.array([x[1] for x in o if R[x[0]]["lg"] == lg])
    print(f"  {lg:5s} {len(rr):4d} bets {100 * rr.mean():+6.1f}%")
