"""Carry rule on the whole geopolitics set without the busy-market hole: markets whose tape is complete are replayed
on their prints (ask = last print on the ask side); markets whose tape hit the 3,000-trade cap (the busiest ones, where
upsets pile up) are replayed on the exchange's 12-hour midpoint history (ask = mid + 0.5c).
    python carry_hybrid.py [lo hi stable_h d0 d1]
"""
import gzip, json, re, sys, datetime as dt, numpy as np
R = '/opt/plint/data/polybot/research4'
LO, HI, S, D0, D1 = (float(x) for x in sys.argv[1:6]) if len(sys.argv) > 5 else (0.90, 0.985, 72, 10, 45)
MON = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december|\d+")
meta = json.load(open(f'{R}/geo_meta.json')); C = np.load(f'{R}/geo_C.npy')
C = C[np.lexsort((C[:, 1], C[:, 0]))]
mi, T, kf, mid, bid, ask, fwin, hte, ser, ntape, htc, age, famc, n72, usd72 = C.T
sname = {v: k for k, v in meta['sers'].items()}
new = np.r_[True, (mi[1:] != mi[:-1]) | (kf[1:] != kf[:-1])]
below = mid < LO
idx = np.maximum.accumulate(np.where(new | below, np.arange(len(T)), 0))
stable = np.where(below, 0, (T - T[idx]) / 3600)
c = (ask >= LO) & (ask < HI) & (hte >= D0 * 24) & (hte < D1 * 24) & (age <= 2) & (htc > 1) & (stable >= S) & (ask > bid - 0.02) & (ntape < 3000)
ix = np.nonzero(c)[0]; _, first = np.unique(mi[ix], return_index=True); ix = ix[first]
E = [(T[i], ask[i], fwin[i], htc[i] / 24, sname[int(ser[i])], 0, usd72[i]) for i in ix]
info = {}
for tag in ('geopolitics', 'world', 'middle-east'):
    for l in gzip.open(f'{R}/events_long/{tag}.jsonl.gz', 'rt'):
        r = json.loads(l); info.setdefault(r['cid'], r)
ncap = 0; f = gzip.open(f'{R}/mids12h.jsonl.gz', 'rt'); done = set()
while True:
    try: l = f.readline()
    except EOFError: break
    if not l: break
    try: j = json.loads(l)
    except Exception: continue
    m = info.get(j['cid'])
    if not m or not j['p'] or not m['end'] or j['cid'] in done: continue
    done.add(j['cid']); ncap += 1
    Eend = dt.datetime.fromisoformat(m['end']).timestamp()
    Tcl = dt.datetime.fromisoformat(m['closed_time']).timestamp() if m['closed_time'] else Eend
    P = np.array(j['p']); t = P[:, 0]; y = P[:, 1]; fv = np.maximum(y, 1 - y); side = (y < 0.5).astype(int)
    need = max(1, int(S // 12))
    for i in range(len(t)):
        left = (Eend - t[i]) / 86400
        if not (D0 <= left < D1) or t[i] >= Tcl - 3600: continue
        w = (t >= t[i] - S * 3600 - 1) & (t <= t[i])
        if S > 0 and (w.sum() < need or (side[w] != side[i]).any() or fv[w].min() < LO): continue
        a = fv[i] + 0.005
        if not (LO <= a < HI): continue
        E.append((t[i], a, m['p'][side[i]], (Tcl - t[i]) / 86400, MON.sub('#', m['ev'] or m['slug']), 1, np.nan)); break
t0 = np.array([e[0] for e in E]); a = np.array([e[1] for e in E]); w = np.array([e[2] for e in E]); hold = np.array([e[3] for e in E]); cap = np.array([e[5] for e in E]); sr = np.array([e[4] for e in E]); u72 = np.array([e[6] for e in E])
r = (w - a) / a
rng = np.random.default_rng(3)
def boot(r_, s_):
    u, ii = np.unique(s_, return_inverse=True); num = np.bincount(ii, r_); den = np.bincount(ii).astype(float)
    bs = [num[k].sum() / max(den[k].sum(), 1) for k in (rng.integers(0, len(u), len(u)) for _ in range(1000))]
    return np.percentile(bs, [5, 95])
def show(nm, m_):
    if m_.sum() < 20: print(f"{nm}: n={m_.sum()}"); return
    l, u = boot(r[m_], sr[m_]); us, ii = np.unique(sr[m_], return_inverse=True); sm = (np.bincount(ii, r[m_]) / np.bincount(ii)).mean()
    print(f"{nm:38s} n={m_.sum():5d} series={len(us):4d} paid {100*a[m_].mean():5.1f}c won {100*w[m_].mean():5.1f}% lost {int((w[m_] == 0).sum()):3d}  ret {100*r[m_].mean():+6.2f}% [{100*l:+5.2f},{100*u:+5.2f}]  per-series {100*sm:+5.2f}%  hold {np.median(hold[m_]):4.1f}d")
print(f"rule: favourite ask {LO}-{HI}, stable {S:.0f} h, {D0:.0f}-{D1:.0f} days left | busy markets replayed on mids: {ncap}")
allm = np.ones(len(r), bool)
show("ALL", allm); show("  complete tapes (prints)", cap == 0); show("  capped tapes = busiest (mids)", cap == 1)
q = np.nanquantile(u72[cap == 0], [1 / 3, 2 / 3])
show(f"  prints: quiet third (<${q[0]:.0f}/72h)", (cap == 0) & (u72 <= q[0])); show("  prints: middle third", (cap == 0) & (u72 > q[0]) & (u72 <= q[1])); show(f"  prints: busy third (>${q[1]:.0f}/72h)", (cap == 0) & (u72 > q[1]))
hy = (t0.astype('datetime64[s]').astype('datetime64[M]').astype(int) // 6)
print("by half-year: " + " | ".join(f"{1970 + h // 2}H{h % 2 + 1} {100*r[hy == h].mean():+5.2f}% (n={(hy == h).sum()}, lost {int((w[hy == h] == 0).sum())})" for h in np.unique(hy)))
def portfolio(mask, per_series=3, per_day=10):
    o = np.argsort(t0[mask]); a0 = t0[mask][o]; b0 = a0 + hold[mask][o] * 86400; rr = r[mask][o]; ss = sr[mask][o]
    open_ = []; dn = {}; tk = []
    for x, y, z, s in zip(a0, b0, rr, ss):
        open_ = [(e, s2) for e, s2 in open_ if e > x]; d = int(x // 86400)
        if sum(1 for _, s2 in open_ if s2 == s) >= per_series or dn.get(d, 0) >= per_day: continue
        open_.append((y, s)); dn[d] = dn.get(d, 0) + 1; tk.append((x, y, z))
    tk = np.array(tk); pnl = 5 * tk[:, 2]
    mon = tk[:, 1].astype('datetime64[s]').astype('datetime64[M]').astype(int); months = np.arange(mon.min(), mon.max() + 1)
    mp = np.array([pnl[mon == m].sum() for m in months])
    grid = np.arange(tk[:, 0].min(), tk[:, 1].max(), 86400); capu = np.array([5 * ((tk[:, 0] <= g) & (tk[:, 1] > g)).sum() for g in grid])
    eq = np.cumsum(pnl[np.argsort(tk[:, 1])]); dd = (np.maximum.accumulate(eq) - eq).max(); yrs = (tk[:, 1].max() - tk[:, 0].min()) / (365 * 86400)
    return f"{len(tk)} positions over {yrs:.1f} y: total {pnl.sum():+.0f}$ on avg {capu.mean():.0f}$ in use (peak {capu.max():.0f}$) = {100*pnl.sum()/capu.mean()/yrs:+.0f}%/y; months up {int((mp > 0).sum())}/{len(mp)}, worst {mp.min():+.0f}$, best {mp.max():+.0f}$, max drawdown {dd:.0f}$"
print("portfolio, $5 each, <=3 open per series, <=10 new a day:", portfolio(allm))
ls = [(sr[i], w[i]) for i in range(len(r)) if w[i] == 0]
import collections
print("losses by series:", "; ".join(f"{k[:36]} x{v}" for k, v in collections.Counter(s for s, _ in ls).most_common(10)))
