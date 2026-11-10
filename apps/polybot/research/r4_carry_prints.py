"""The carry rule on one built data set: buy the favourite once per market when
  ask in [LO, HI), 10-45 days to the scheduled end, the favourite's mid has stayed >= LO for >= STABLE hours,
  both sides printed within 2 h. Held to resolution. Prints: headline, neighbours, by period, by group, a small
  portfolio with caps.
    python carry_eval.py TAG [families] [fee_rate]
"""
import numpy as np, json, sys
R = '/opt/plint/data/polybot/research4'; TAG = sys.argv[1]
FAMS = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] != '-' else None
FEE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
meta = json.load(open(f'{R}/{TAG}_meta.json'))
C = np.load(f'{R}/{TAG}_C.npy')
if FAMS: C = C[np.isin(C[:, 12], [i for n, i in meta['fams'].items() if n in FAMS])]
C = C[np.lexsort((C[:, 1], C[:, 0]))]
mi, T, kf, mid, bid, ask, fwin, hte, ser, ntape, htc, age, famc, n72, usd72 = C.T
sname = {v: k for k, v in meta['sers'].items()}
new = np.r_[True, (mi[1:] != mi[:-1]) | (kf[1:] != kf[:-1])]
rng = np.random.default_rng(5)
def boot(r, c):
    u, idx = np.unique(c, return_inverse=True)
    num = np.bincount(idx, r); den = np.bincount(idx).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.integers(0, len(u), len(u)) for _ in range(800))]
    return np.percentile(bs, [5, 95])
def entries(lo, hi, S, h0, h1):
    below = mid < lo
    idx = np.maximum.accumulate(np.where(new | below, np.arange(len(T)), 0))
    stable = np.where(below, 0, (T - T[idx]) / 3600)
    c = (ask >= lo) & (ask < hi) & (hte >= h0) & (hte < h1) & (age <= 2) & (htc > 1) & (stable >= S) & (ask > bid - 0.02)
    ix = np.nonzero(c)[0]
    if not len(ix): return ix
    _, first = np.unique(mi[ix], return_index=True)
    return ix[first]
def line(nm, ix):
    if len(ix) < 30: return f"{nm:34s} n={len(ix)}"
    r = (fwin[ix] - ask[ix]) / ask[ix] - FEE * (1 - ask[ix])
    l, u = boot(r, ser[ix])
    us, idx = np.unique(ser[ix], return_inverse=True); sm = (np.bincount(idx, r) / np.bincount(idx)).mean()
    cm = ntape[ix] < 3000
    return f"{nm:34s} n={len(ix):5d} series={len(us):4d} paid {100*ask[ix].mean():5.1f}c won {100*fwin[ix].mean():5.1f}% ret {100*r.mean():+6.2f}% [{100*l:+5.2f},{100*u:+5.2f}] per-series {100*sm:+5.2f}% hold {np.median(htc[ix])/24:4.1f}d uncapped-tapes {100*r[cm].mean():+5.2f}%"
print(f"== {TAG} {FAMS or 'all'} fee {FEE}: {len(np.unique(mi))} markets with a favourite, {str(T.min().astype('datetime64[s]'))[:10]} -> {str(T.max().astype('datetime64[s]'))[:10]}")
ix = entries(.90, .985, 72, 240, 1080)
print(line("RULE 90-98.5c, stable 72h, 10-45d", ix))
for nm, a in (("  no stability filter", (.90, .985, 0, 240, 1080)), ("  stable 24h", (.90, .985, 24, 240, 1080)), ("  stable 168h", (.90, .985, 168, 240, 1080)),
              ("  floor 85c", (.85, .985, 72, 240, 1080)), ("  floor 95c", (.95, .985, 72, 240, 1080)), ("  cap 97c", (.90, .97, 72, 240, 1080)),
              ("  3-10 days left", (.90, .985, 72, 72, 240)), ("  10-20 days left", (.90, .985, 72, 240, 480)), ("  20-45 days left", (.90, .985, 72, 480, 1080)), ("  45-120 days left", (.90, .985, 72, 1080, 2880))):
    print(line(nm, entries(*a)))
if len(ix) >= 60:
    r = (fwin[ix] - ask[ix]) / ask[ix] - FEE * (1 - ask[ix])
    qs_ = np.quantile(usd72[ix], [1/3, 2/3])
    print('by activity before entry ($ traded in the 72 h before; cut points %.0f / %.0f): ' % tuple(qs_) + ' | '.join(f"{nm}: {100*r[m_].mean():+5.2f}% (n={m_.sum()}, lost {int((fwin[ix][m_] == 0).sum())})" for nm, m_ in (('quiet third', usd72[ix] <= qs_[0]), ('middle', (usd72[ix] > qs_[0]) & (usd72[ix] <= qs_[1])), ('busy third', usd72[ix] > qs_[1]))))
    cap_ = ntape[ix] >= 3000
    print(f'markets whose first tape hit the 3,000-trade cap: {100*r[cap_].mean() if cap_.sum() else float("nan"):+5.2f}% (n={cap_.sum()}, lost {int((fwin[ix][cap_] == 0).sum())}) | others {100*r[~cap_].mean():+5.2f}% (n={(~cap_).sum()}, lost {int((fwin[ix][~cap_] == 0).sum())})')
    hy = (T[ix].astype('datetime64[s]').astype('datetime64[M]').astype(int) // 6)
    print("by half-year of entry: " + " | ".join(f"{1970 + h // 2}H{h % 2 + 1}: {100*r[hy == h].mean():+5.2f}% (n={(hy == h).sum()}, lost {int(((fwin[ix] == 0) & (hy == h)).sum())})" for h in np.unique(hy)))
    us, idx = np.unique(ser[ix], return_inverse=True); pn = np.bincount(idx, r); ct = np.bincount(idx)
    print("losing positions:", int((fwin[ix] == 0).sum()), "of", len(ix), "| biggest losing series:", "; ".join(f"{sname[int(us[o])][:40]} ({int(((fwin[ix] == 0) & (idx == o)).sum())} lost of {ct[o]})" for o in np.argsort(pn)[:4]))
    # portfolio: $5 a position, at most 3 open per series, 10 new a day
    o = np.argsort(T[ix]); t0 = T[ix][o]; t1 = t0 + htc[ix][o] * 3600; rr = r[o]; ss = ser[ix][o]
    open_ = []; day_n = {}; taken = []
    for a, b, x, s in zip(t0, t1, rr, ss):
        open_ = [(e, s2) for e, s2 in open_ if e > a]
        d = int(a // 86400)
        if sum(1 for _, s2 in open_ if s2 == s) >= 3 or day_n.get(d, 0) >= 10: continue
        open_.append((b, s)); day_n[d] = day_n.get(d, 0) + 1; taken.append((a, b, x))
    tk = np.array(taken); pnl = 5 * tk[:, 2]
    mon = (tk[:, 1].astype('datetime64[s]').astype('datetime64[M]').astype(int)); months = np.arange(mon.min(), mon.max() + 1)
    mp = np.array([pnl[mon == m].sum() for m in months])
    grid = np.arange(tk[:, 0].min(), tk[:, 1].max(), 86400); cap = np.array([5 * ((tk[:, 0] <= g) & (tk[:, 1] > g)).sum() for g in grid])
    eq = np.cumsum(pnl[np.argsort(tk[:, 1])]); dd = (np.maximum.accumulate(eq) - eq).max()
    yrs = (tk[:, 1].max() - tk[:, 0].min()) / (365 * 86400)
    print(f"portfolio ($5 each, <=3 open per series, <=10 new a day): {len(tk)} positions over {yrs:.1f} y, total {pnl.sum():+.0f}$, capital in use avg {cap.mean():.0f}$ / peak {cap.max():.0f}$, "
          f"return on avg capital {100*pnl.sum()/cap.mean()/yrs:+.0f}%/y, months up {int((mp > 0).sum())}/{len(mp)}, worst month {mp.min():+.0f}$, best {mp.max():+.0f}$, max drawdown {dd:.0f}$")
