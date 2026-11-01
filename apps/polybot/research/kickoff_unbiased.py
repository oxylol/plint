"""The pre-game underdog gap without the look-ahead volume filter: every market, whatever it later traded.

    python kickoff_unbiased.py DATA_DIR          (wallet-derived list: ≥ $5k plus the weighted < $5k sample)
    python kickoff_unbiased.py DATA_DIR --uni    (a random sample of every closed market Gamma lists, collect3_universe.py)

Combines collect3_kickoff.py's markets traded ≥ $5k (all of them) with its random sample of the markets traded less
(`_low`, weighted up to their full count), and shows the underdog's calibration 60 and 5 minutes before kick-off and
the return of buying it at the history price + 0.5¢ (what live books charge, kickoff_books_now.py) + taker fee, held
to resolution: for ≥ $5k, < $5k and all markets together.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
K = DATA / "kickoff"
U = dt.timezone.utc
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()


def ts(x):
    return dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def load(suffix):
    mk = json.loads((K / f"markets{suffix}.json").read_text())
    mk.pop("n_universe", None)
    out = []
    for line in (K / f"prices{suffix}.jsonl").open():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not d["p"]:
            continue
        m = mk[d["cid"]]
        v = sorted(d["p"])
        tt, pp = np.array([x[0] for x in v], float), np.array([x[1] for x in v])
        g = ts(m["game_start"])
        y0 = 1.0 if m["prices"][0] > 0.5 else 0.0
        for e in (-60, -5):
            t = g + 60 * e
            j = np.searchsorted(tt, t, side="right") - 1
            if j < 0 or t - tt[j] > 1800 or not 0.02 <= pp[j] <= 0.98:
                continue
            pu, yu = (pp[j], y0) if pp[j] <= 0.5 else (1 - pp[j], 1 - y0)
            a = min(pu + 0.005, 0.995)
            out.append((e, t, pu, yu, yu / a - 1 - float(m.get("fee_rate") or 0) * (1 - a), m["sports_type"] in ("spreads", "totals"),
                        hash(m.get("slug", "").rsplit("-", 1)[0] if m["sports_type"] != "moneyline" else m.get("slug", "")) % 10**9))
    return np.array(out, float), len(mk)


if "--uni" in sys.argv:
    X, n = load("_uni")
    rng = np.random.default_rng(4)
    print(f"random sample of {n} markets from every closed sports market Gamma lists (no wallet involved)")

    def boot(s):
        """90% interval, resampling days (games on a day share results across their lines)."""
        d = (s[:, 1] // 86400).astype(int)
        ud, inv = np.unique(d, return_inverse=True)
        sums, cnts = np.bincount(inv, s[:, 4]), np.bincount(inv)
        dr = rng.integers(0, len(ud), size=(3000, len(ud)))
        bm = sums[dr].sum(1) / cnts[dr].sum(1)
        return np.quantile(bm, .05), np.quantile(bm, .95)
    for e in (-60, -5):
        print(f"\n{e:+d} min")
        for name, flag in (("moneylines", 0.0), ("spreads + totals", 1.0), ("all", None)):
            print(f"  {name}:")
            for lo, hi in ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.501)):
                s = X[(X[:, 0] == e) & (X[:, 2] >= lo) & (X[:, 2] < hi) & ((X[:, 5] == flag) if flag is not None else True)]
                if len(s) < 30:
                    continue
                a, b = boot(s)
                sel = s[:, 1] < SELECT_END
                print(f"    {100 * lo:3.0f}–{100 * min(hi, .5):2.0f}¢ {len(s):5d}  won {100 * s[:, 3].mean():5.1f}% vs {100 * s[:, 2].mean():5.1f}¢ "
                      f"({100 * (s[:, 3] - s[:, 2]).mean():+5.2f} pts) | buy underdog {100 * s[:, 4].mean():+6.1f}% [{100 * a:+5.1f}, {100 * b:+5.1f}] | "
                      f"select {100 * s[sel, 4].mean():+6.1f}%  holdout {100 * s[~sel, 4].mean():+6.1f}%")
    sys.exit()


hi_, n_hi = load("")
lo_, n_lo_s = load("_low")
n_lo_all = 68900   # markets under $5k in the window (collect3_kickoff.py's count before sampling)
w_lo = n_lo_all / n_lo_s
print(f"≥ $5k: {n_hi} markets; < $5k: sample of {n_lo_s} (weight {w_lo:.2f} each)")
for e in (-60, -5):
    print(f"\n{e:+d} min: underdog won − price (pts) and buy-underdog return after 0.5¢ and fee | all markets: Sep 3 → Oct 19 / Oct 20 → 31")
    print(f"  {'band':>8s} | {'≥ $5k':>22s} | {'< $5k':>22s} | {'all markets (weighted)':>24s} | {'select':>7s} {'holdout':>8s}")
    for lo, hi in ((0.02, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.501)):
        cells = []
        parts = []
        for X, w in ((hi_, 1.0), (lo_, w_lo)):
            s = X[(X[:, 0] == e) & (X[:, 2] >= lo) & (X[:, 2] < hi)]
            cells.append(f"{len(s):5d} {100 * (s[:, 3] - s[:, 2]).mean():+5.2f} {100 * s[:, 4].mean():+6.1f}%" if len(s) else "—")
            parts.append((s, w))
        S = np.vstack([p[0] for p in parts])
        W = np.concatenate([np.full(len(p[0]), p[1]) for p in parts])
        gap = np.average(S[:, 3] - S[:, 2], weights=W)
        ret = np.average(S[:, 4], weights=W)
        sel = S[:, 1] < SELECT_END
        rs = np.average(S[sel, 4], weights=W[sel])
        rh = np.average(S[~sel, 4], weights=W[~sel])
        print(f"  {100 * lo:3.0f}–{100 * min(hi, 0.5):2.0f}¢ | {cells[0]:>22s} | {cells[1]:>22s} | {gap * 100:+9.2f} {100 * ret:+8.1f}%     | {100 * rs:+6.1f}% {100 * rh:+7.1f}%")
    for name, flag in (("moneylines", 0.0), ("spreads + totals", 1.0)):
        S = np.vstack([hi_[(hi_[:, 0] == e) & (hi_[:, 5] == flag)], lo_[(lo_[:, 0] == e) & (lo_[:, 5] == flag)]])
        W = np.concatenate([np.ones(((hi_[:, 0] == e) & (hi_[:, 5] == flag)).sum()), np.full(((lo_[:, 0] == e) & (lo_[:, 5] == flag)).sum(), w_lo)])
        print(f"  {name:18s} all markets: underdog won − price {100 * np.average(S[:, 3] - S[:, 2], weights=W):+5.2f} pts, "
              f"buy underdog {100 * np.average(S[:, 4], weights=W):+6.1f}%")
