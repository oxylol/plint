"""Non-sports, non-weather markets: (A) every taker trade from the maker's side, (B) resting bids below the last two-sided
prints, filled only by through-prints in the next hour. Same definitions as the weather scripts."""
import gzip, json, re, sys, glob, os, math, datetime as dt
import numpy as np

R = sys.argv[1]
MAN = sys.argv[2]
TAG = sys.argv[3]
LB, H = 6 * 3600, 3600
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]


def family(r):
    if r.get("tag"): return {"geopolitics": "geo", "world": "geo", "middle-east": "geo"}.get(r["tag"], r["tag"])
    s = r["slug"] or ""; ft = r["fee_type"]
    if re.search(r"of-tweets|truth-social-posts|-posts-", s): return "tweet counts"
    if re.search(r"^(bitcoin|ethereum|solana|xrp|will-bitcoin|will-ethereum|will-solana|will-xrp|what-price-will)", s) or ft == "crypto_fees_v2":
        return "crypto touch/range" if re.search(r"dip-to|reach|hit|between", s) else "crypto other"
    if ft == "finance_prices_fees" or re.search(r"close-above|closes-above|-above-\d+-on|dip-to|treasury-yield|gas-price|finish-week", s):
        return "stocks/commodities"
    if ft == "mentions_fees" or re.search(r"^will-.*-say-", s): return "mentions"
    if ft == "tech_fees": return "tech"
    if ft == "economics_fees": return "economics"
    if ft == "politics_fees": return "politics"
    if ft == "culture_fees": return "culture"
    if ft is None: return "geopolitics (no fee type)"
    return "zero-fee other"


want = set(json.loads(l)["cid"] for l in gzip.open(MAN, "rt") if os.path.exists(f"{R}/tapes/" + json.loads(l)["cid"] + ".json.gz"))
uni = {}
SRC = sys.argv[4] if len(sys.argv) > 4 else "universe"
for f in sorted(glob.glob(f"{R}/{SRC}/*.jsonl.gz")):
    for l in gzip.open(f, "rt"):
        c0 = l[9:75]
        if c0 in want and c0 not in uni:
            r = json.loads(l)
            uni[r["cid"]] = {k: r.get(k) for k in ("cid", "slug", "fee_type", "p", "end", "closed_time", "ev", "tag")}
fams, evs, sers = {}, {}, {}
MON = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december|\d+")
A, B, C = [], [], []
posted = {}
nm = 0
for l in gzip.open(MAN, "rt"):
    cid = json.loads(l)["cid"]
    m = uni.get(cid)
    p = f"{R}/tapes_deep/{cid}.json.gz"
    if not os.path.exists(p):
        p = f"{R}/tapes/{cid}.json.gz"
    if not m or not os.path.exists(p) or m["p"][0] not in (0.0, 1.0):
        continue
    try:
        tape = json.load(gzip.open(p, "rt"))
    except Exception:
        continue
    if len(tape) < 2:
        continue
    fam = fams.setdefault(family(m), len(fams))
    ev = evs.setdefault(m["ev"] or cid, len(evs))
    ser = sers.setdefault(MON.sub("#", m["ev"] or m["slug"] or cid), len(sers))
    a = np.array(tape, dtype=np.float64)
    a = a[(a[:, 4] >= 0.001) & (a[:, 4] <= 0.999)]
    if len(a) < 2:
        continue
    a = a[np.argsort(a[:, 0], kind="stable")]
    ts, tok, side, size, pr = a.T
    Tend = dt.datetime.fromisoformat(m["end"]).timestamp() if m["end"] else ts[-1]
    Tcl = dt.datetime.fromisoformat(m["closed_time"]).timestamp() if m["closed_time"] else ts[-1]
    fin = m["p"]
    # A: maker view of every trade
    mtok = np.where(side == 1, 1 - tok, tok)
    q = np.where(side == 1, 1 - pr, pr)
    win = np.where(mtok == 0, fin[0], fin[1])
    A.append(np.column_stack([q, win, size * q, mtok, (Tend - ts) / 3600, np.full(len(ts), fam), np.full(len(ts), ev), (Tcl - ts) / 3600, ts, np.full(len(ts), ser)]).astype(np.float32))
    # B: resting bids
    yes_hit = ((tok == 0) & (side == -1)) | ((tok == 1) & (side == 1))
    hits = []
    for k, msk in ((0, yes_hit), (1, ~yes_hit)):
        hits.append((ts[msk], np.where(tok[msk] == k, pr[msk], 1 - pr[msk])))
    if len(hits[0][0]) and len(hits[1][0]):
        T = np.arange(math.ceil(ts[0] / 3600) * 3600 + 3600, ts[-1], 3600.0)
        if len(T):
            last = []; ages = []
            for t_k, p_k in hits:
                i = np.searchsorted(t_k, T, side="right") - 1
                ok = (i >= 0) & (t_k[np.maximum(i, 0)] >= T - LB)
                last.append(np.where(ok, p_k[np.maximum(i, 0)], np.nan))
                ages.append(np.where(ok, T - t_k[np.maximum(i, 0)], np.inf))
            age = np.maximum(ages[0], ages[1]) / 3600
            b, n = last
            r = (b + (1 - n)) / 2
            spread = (1 - n) - b
            okT = ~np.isnan(r) & (spread > -0.05) & (spread < 0.2)
            n72 = np.searchsorted(ts, T, side='right') - np.searchsorted(ts, T - 72 * 3600, side='right')
            cusd = np.concatenate([[0], np.cumsum(size * pr)])
            usd72 = cusd[np.searchsorted(ts, T, side='right')] - cusd[np.searchsorted(ts, T - 72 * 3600, side='right')]
            kf = np.where(r >= 0.5, 0, 1)
            bidf = np.where(kf == 0, b, n); askf = np.where(kf == 0, 1 - n, 1 - b)
            sel = okT & (np.maximum(r, 1 - r) >= 0.7)
            if sel.any():
                C.append(np.column_stack([np.full(sel.sum(), nm), T[sel], kf[sel], np.maximum(r, 1 - r)[sel], bidf[sel], askf[sel],
                                          np.where(kf[sel] == 0, fin[0], fin[1]), (Tend - T[sel]) / 3600, np.full(sel.sum(), ser), np.full(sel.sum(), len(tape)), (Tcl - T[sel]) / 3600, age[sel], np.full(sel.sum(), fam), n72[sel], usd72[sel]]))
            for k in (0, 1):
                t_k, p_k = hits[k]
                i0 = np.searchsorted(t_k, T, side="right"); i1 = np.searchsorted(t_k, T + H, side="right")
                wmin = np.full(len(T), np.inf)
                for j in np.nonzero((i1 > i0) & okT)[0]:
                    wmin[j] = p_k[i0[j]:i1[j]].min()
                rk = r if k == 0 else 1 - r
                for di, d in enumerate(DELTAS):
                    L = np.floor((rk - d) * 100 + 1e-9) / 100
                    valid = okT & (L >= 0.01) & (L <= 0.99)
                    posted[(fam, di, k)] = posted.get((fam, di, k), 0) + int(valid.sum())
                    f = valid & (wmin < L - 1e-9)
                    for j in np.nonzero(f)[0]:
                        B.append((di, k, L[j], fin[k], fam, ev, (Tend - T[j]) / 3600, rk[j], spread[j], T[j], nm, ser, len(tape)))
    nm += 1
np.save(f"{R}/{TAG}_A.npy", np.vstack(A)); np.save(f"{R}/{TAG}_B.npy", np.array(B)); np.save(f"{R}/{TAG}_C.npy", np.vstack(C) if C else np.zeros((0, 15)))
json.dump({"fams": fams, "posted": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in posted.items()}, "n_ev": len(evs), "evs": evs, "sers": sers}, open(f"{R}/{TAG}_meta.json", "w"))
print("markets", nm, "trades", sum(len(x) for x in A), "fills", len(B), "events", len(evs))
