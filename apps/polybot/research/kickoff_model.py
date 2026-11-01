"""Is the sports price right at kick-off? A model of the price itself, with no leaders, held to resolution.

    python kickoff_model.py DATA_DIR [cost=med|p75] [min_moves]

Uses collect3_kickoff.py's paths (resolved moneyline / child-moneyline / spread / total markets, game start in the
last 58 days, first outcome's 5-minute midpoint from 3 h before to 4 h after kick-off). For each market and entry
moment (60 and 5 minutes before kick-off; 15, 30 and 60 minutes after) the underdog side is the one priced ≤ 50¢.
A buy pays the midpoint plus the fill model's cost of a $5 market buy (sports pre-game / in play, by price band) and
the market's taker fee (rate × (1 − price) per dollar), and is held to resolution.

1. Calibration: underdog and favourite, by price band and moment, every market (no model).
2. Models refitted every week on games that started before it (ridge logistic, anchored on the price's logit):
   price slope and intercept by market type, + league, + the last hour's move and pre-game activity. A side is bought
   when its expected return per dollar after cost and fee clears a cut. Picked on Sep 15 → Oct 19, judged on
   Oct 20 → 31, day-block bootstrap CIs, every qualifying signal and the bot's first-10-a-day book.

The universe filter (traded ≥ $5k over the market's life) peeks at the future; activity before the entry (5-minute
buckets with a price change in the 3 hours before) is the filter the bot could use, and the results are shown by both.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
COST = sys.argv[2] if len(sys.argv) > 2 else "med"
MIN_MOVES = int(sys.argv[3]) if len(sys.argv) > 3 else 0
K = DATA / "kickoff"
U = dt.timezone.utc
TEST0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp()
SELECT_END = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
MOMENTS = (-60, -5, 15, 30, 60)
TYPES = ("moneyline", "child_moneyline", "spreads", "totals")
rng = np.random.default_rng(5)
FILL = json.loads((DATA / "fill_model.json").read_text())["cells"]


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def band(p):
    return "lt10" if p < 0.10 else "10_30" if p < 0.30 else "30_70" if p < 0.70 else "70_90" if p < 0.90 else "ge90"


def cost(p, live):
    g = "sports_live" if live else "sports"
    cell = FILL.get(f"{g}|{band(p)}|lt6") or FILL.get(f"{g}|{band(p)}") or FILL[g]
    return cell["c5_p75" if COST == "p75" else "c5_med"]


def lgt(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


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
print(f"{len(mk)} markets, {len(paths)} with a price path")


def league(slug):
    return (slug or "").split("-")[0]


lg_count = Counter(league(m["slug"]) for c, m in mk.items() if c in paths)
LEAGUES = [g for g, n in lg_count.most_common(30) if n >= 300]

rows = []
for cid, (tt, pp) in paths.items():
    m = mk[cid]
    g = ts(m["game_start"])
    y0 = 1.0 if m["prices"][0] > 0.5 else 0.0
    rate = float(m.get("fee_rate") or 0.0)
    for e in MOMENTS:
        t = g + 60 * e
        k = np.searchsorted(tt, t, side="right") - 1
        if k < 0 or t - tt[k] > 1800:          # no print in the last half hour: no price to act on
            continue
        p0 = pp[k]
        if not 0.02 <= p0 <= 0.98:
            continue
        k1 = np.searchsorted(tt, t - 3300, side="right") - 1
        p_before = pp[k1] if k1 >= 0 else p0
        pre = (tt >= g - 3 * 3600) & (tt <= min(t, g))
        moves = int((np.diff(pp[pre]) != 0).sum()) if pre.sum() > 1 else 0
        under0 = p0 <= 0.5
        pu = p0 if under0 else 1 - p0
        yu = y0 if under0 else 1 - y0
        drift = lgt(np.array(pu))[()] - lgt(np.array(p_before if under0 else 1 - p_before))[()]
        rows.append((t, g, e, pu, yu, drift, moves, rate, TYPES.index(m["sports_type"]),
                     LEAGUES.index(league(m["slug"])) if league(m["slug"]) in LEAGUES else len(LEAGUES), float(m.get("vol") or 0), cid))
R = np.array([r[:11] for r in rows], float)
CID = np.array([r[11] for r in rows])
t_, g_, e_, pu, yu, drift, moves, rate, typ, lg, vol = R.T
live = e_ > 0
cu = np.array([cost(p, l) for p, l in zip(pu, live)])
cf = np.array([cost(1 - p, l) for p, l in zip(pu, live)])
au, af = np.minimum(pu + cu, 0.995), np.minimum(1 - pu + cf, 0.995)
ret_u = yu / au - 1 - rate * (1 - au)
ret_f = (1 - yu) / af - 1 - rate * (1 - af)
keep = (moves >= MIN_MOVES) & (t_ >= dt.datetime(2026, 9, 3, tzinfo=U).timestamp())
print(f"{len(R)} market-moments ({keep.sum()} with ≥ {MIN_MOVES} pre-game moves), {len(LEAGUES)} leagues: {', '.join(LEAGUES)}")
days = (t_ // 86400).astype(int)


def boot(sel, r):
    """Mean return per $ of the selected bets, with a day-block 90% bootstrap interval."""
    if sel.sum() < 20:
        return sel.sum(), np.nan, np.nan, np.nan
    rr, dd = r[sel], days[sel]
    ud, inv = np.unique(dd, return_inverse=True)
    sums, cnts = np.bincount(inv, rr), np.bincount(inv)
    dr = rng.integers(0, len(ud), size=(3000, len(ud)))
    bm = sums[dr].sum(1) / cnts[dr].sum(1)
    return sel.sum(), rr.mean(), np.quantile(bm, .05), np.quantile(bm, .95)


def fmt(b):
    n, m, lo, hi = b
    return f"{int(n):6d} {100 * m:+6.1f}% [{100 * lo:+5.1f}, {100 * hi:+5.1f}]" if n >= 20 else f"{int(n):6d}      —"


print(f"\n## 1. Calibration, every market (Sep 3 → Oct 31), cost {COST}, held to resolution\n")
print(f"{'moment':>7s} {'underdog price':>15s} | {'n':>6s} {'won':>6s} {'price':>6s} | {'buy underdog':>28s} | {'buy favourite':>28s}")
for e in MOMENTS:
    for lo_, hi_ in ((0.02, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50)):
        s = keep & (e_ == e) & (pu >= lo_) & (pu < hi_ if hi_ < 0.5 else pu <= hi_)
        if s.sum() < 20:
            continue
        print(f"{e:+5d}m  {100 * lo_:4.0f}–{100 * hi_:2.0f}¢       | {s.sum():6d} {100 * yu[s].mean():5.1f}% {100 * pu[s].mean():5.1f}¢ | "
              f"{fmt(boot(s, ret_u))} | {fmt(boot(s, ret_f))}")
    print()

print("by market type, 5 min before kick-off (won − price, in points; favourite and underdog returns):")
for k, name in enumerate(TYPES):
    s = keep & (e_ == -5) & (typ == k)
    if s.sum() >= 50:
        print(f"  {name:16s} {s.sum():6d}  underdog won {100 * (yu[s] - pu[s]).mean():+5.2f} pts | dog {fmt(boot(s, ret_u))} | fav {fmt(boot(s, ret_f))}")
print("by league, 5 min before kick-off:")
for k, name in enumerate(LEAGUES + ["other"]):
    s = keep & (e_ == -5) & (lg == k)
    if s.sum() >= 100:
        print(f"  {name:10s} {s.sum():6d}  underdog won {100 * (yu[s] - pu[s]).mean():+5.2f} pts | dog {fmt(boot(s, ret_u))} | fav {fmt(boot(s, ret_f))}")
print("selection check (the ≥ $5k life-volume filter peeks ahead): underdog won − price at −5 min by life volume")
for lo_, hi_ in ((5e3, 2e4), (2e4, 1e5), (1e5, 1e6), (1e6, 1e12)):
    s = keep & (e_ == -5) & (vol >= lo_) & (vol < hi_)
    if s.sum() >= 50:
        print(f"  ${lo_:>9,.0f}+ {s.sum():6d}  {100 * (yu[s] - pu[s]).mean():+5.2f} pts")
for lo_, hi_ in ((0, 3), (3, 10), (10, 25), (25, 99)):
    s = keep & (e_ == -5) & (moves >= lo_) & (moves < hi_)
    if s.sum() >= 50:
        print(f"  {lo_:2d}–{hi_:2d} pre-game moves {s.sum():6d}  {100 * (yu[s] - pu[s]).mean():+5.2f} pts")


# ---- 2. models, walk-forward by week
def design(idx, spec):
    cols = [lgt(pu[idx])]
    for k in range(len(TYPES)):
        cols += [(typ[idx] == k).astype(float), (typ[idx] == k) * lgt(pu[idx])]
    cols += [live[idx].astype(float), live[idx] * lgt(pu[idx])]
    if "league" in spec:
        for k in range(len(LEAGUES) + 1):
            cols += [(lg[idx] == k).astype(float)]
    if "flow" in spec:
        cols += [drift[idx], np.log1p(moves[idx])]
    return np.column_stack(cols)


def fit(tr, spec, lam):
    Xt = design(tr, spec)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Z = np.hstack([(Xt - mu) / sd, np.ones((len(tr), 1))])
    off = lgt(pu[tr])
    w = np.zeros(Z.shape[1])
    I = np.eye(Z.shape[1])
    I[-1, -1] = 0
    for _ in range(25):
        p = sig(Z @ w + off)
        w -= np.linalg.solve(Z.T @ (Z * (p * (1 - p))[:, None]) + lam * I, Z.T @ (p - yu[tr]) + lam * I @ w)
    return lambda j: sig(np.hstack([(design(j, spec) - mu) / sd, np.ones((len(j), 1))]) @ w + lgt(pu[j]))


SPECS = {"price by type": (), "+ league": ("league",), "+ league + flow": ("league", "flow")}
LAMS = (30.0, 300.0, 3000.0)
weeks = []
s = TEST0
while s < t_.max():
    weeks.append((s, s + 7 * 86400))
    s += 7 * 86400
P = {}
for spec in SPECS:
    for lam in LAMS:
        P[(spec, lam)] = np.full(len(R), np.nan)
for w0, w1 in weeks:
    tr = np.where(keep & (g_ < w0 - 6 * 3600))[0]
    te = np.where(keep & (t_ >= w0) & (t_ < w1))[0]
    if not len(te):
        continue
    for spec, sp in SPECS.items():
        for lam in LAMS:
            P[(spec, lam)][te] = fit(tr, sp, lam)(te)

test = keep & (t_ >= TEST0)
print(f"\n## 2. Models, refitted weekly, Sep 15 → Oct 31 ({test.sum()} market-moments; one side per market-moment)\n")
# log loss against the market
base_ll = -np.mean(yu[test] * np.log(pu[test]) + (1 - yu[test]) * np.log(1 - pu[test]))
print(f"market alone: log loss {base_ll:.4f}")
for key, pr in P.items():
    q = np.clip(pr[test], 1e-4, 1 - 1e-4)
    print(f"  {key[0]:18s} λ={key[1]:6.0f}: {-np.mean(yu[test] * np.log(q) + (1 - yu[test]) * np.log(1 - q)):.4f}")


def pick(pr, cut, moments=None):
    """For each market-moment, the side with the higher expected return if it clears the cut; returns (sel, ret)."""
    er_u = pr / au - 1 - rate * (1 - au)
    er_f = (1 - pr) / af - 1 - rate * (1 - af)
    take_u = (er_u >= er_f) & (er_u >= cut)
    take_f = (er_f > er_u) & (er_f >= cut)
    sel = test & (take_u | take_f) & np.isfinite(pr)
    if moments is not None:
        sel &= np.isin(e_, moments)
    r = np.where(take_u, ret_u, ret_f)
    return sel, r, np.where(take_u, er_u, er_f)


def one_per_market(sel):
    """Only the first qualifying moment of each market (a market is bought once)."""
    out = np.zeros_like(sel)
    seen = set()
    for i in np.where(sel)[0][np.argsort(t_[sel], kind="stable")]:
        if CID[i] in seen:
            continue
        seen.add(CID[i])
        out[i] = True
    return out


def book(sel, r):
    """The bot's book: $5 a bet, first 10 qualifying bets a day in time order. $ a week."""
    out = np.zeros_like(sel)
    for d in np.unique(days[sel]):
        ii = np.where(sel & (days == d))[0]
        out[ii[np.argsort(t_[ii])[:10]]] = True
    nd = len(np.unique(days[test]))
    return 7 * 5 * r[out].sum() / nd, out


print(f"\n{'model':18s} {'λ':>6s} {'when':>9s} {'cut':>5s} | {'every signal (first per market)':>34s} | {'select':>7s} {'holdout':>8s} | book $/wk (sel / hold)")
res = {}
for key, pr in P.items():
    for mom_name, moms in (("pre-game", (-60, -5)), ("in play", (15, 30, 60)), ("any", None)):
        for cut in (0.0, 0.02, 0.05):
            sel, r, er = pick(pr, cut, moms)
            sel = one_per_market(sel)
            b = boot(sel, r)
            ms = r[sel & (t_ < SELECT_END)].mean() if (sel & (t_ < SELECT_END)).sum() else np.nan
            mh = r[sel & (t_ >= SELECT_END)].mean() if (sel & (t_ >= SELECT_END)).sum() else np.nan
            wk, bsel = book(sel, r)
            nsd = len(np.unique(days[test & (t_ < SELECT_END)]))
            nhd = len(np.unique(days[test & (t_ >= SELECT_END)]))
            wk_s = 35 * r[bsel & (t_ < SELECT_END)].sum() / nsd
            wk_h = 35 * r[bsel & (t_ >= SELECT_END)].sum() / nhd
            res[(key, mom_name, cut)] = dict(n=int(b[0]), mean=b[1], lo=b[2], hi=b[3], select=ms, holdout=mh, wk=wk, wk_s=wk_s, wk_h=wk_h)
            print(f"{key[0]:18s} {key[1]:6.0f} {mom_name:>9s} {cut:+5.2f} | {fmt(b):>34s} | {100 * ms:+6.1f}% {100 * mh:+7.1f}% | "
                  f"{wk:+6.1f} ({wk_s:+5.1f} / {wk_h:+5.1f})")
best = max(res, key=lambda k: res[k]["wk_s"] if res[k]["n"] >= 100 else -1e9)
b = res[best]
print(f"\npicked on Sep 15 → Oct 19 by book $/week: {best[0][0]} λ={best[0][1]:.0f}, {best[1]}, cut {best[2]:+.2f} "
      f"(select ${b['wk_s']:+.1f}/wk) → holdout ${b['wk_h']:+.1f}/wk, {100 * b['holdout']:+.1f}% a bet; every signal {100 * b['mean']:+.1f}% "
      f"[{100 * b['lo']:+.1f}, {100 * b['hi']:+.1f}]")
json.dump({f"{k[0][0]}|{k[0][1]}|{k[1]}|{k[2]}": {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
          open(K / f"kickoff_model_{COST}_{MIN_MOVES}.json", "w"), indent=1)
