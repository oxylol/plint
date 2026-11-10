"""Resting bids below the market in temperature ladders, replayed on the taker tapes.

Every hour T, per market, the reference r (Yes terms) is the midpoint of the last print that hit a Yes bid and the
last print that hit a No bid (both within LB hours before T): only prints already on the tape. A $5 bid rests for one
hour at r - d (Yes) and (1-r) - d (No). It counts as filled only if a later print in that hour went THROUGH the level
(strictly below it), which under price-time priority means the level was consumed. Held to resolution, no fee (maker).
"""
import gzip, json, sys, math, datetime as dt, os
from zoneinfo import ZoneInfo
import numpy as np

D = sys.argv[1] + "/weather/"
OUT = sys.argv[2]
LB, H = 2 * 3600, 3600
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
st = json.load(open(D + "stations.json"))
city_tz, city_ix = {}, {}
for icao, s in st.items():
    if s["city"] not in city_tz:
        city_tz[s["city"]] = ZoneInfo(s["tz"]); city_ix[s["city"]] = len(city_ix)
mk = {}
for l in gzip.open(D + "markets.jsonl.gz", "rt"):
    m = json.loads(l)
    if m["closed"] and m["kind"] == "highest" and m["prices"] and set(m["prices"]) == {0.0, 1.0}:
        mk[m["cid"]] = m
out = []
posted = np.zeros((len(DELTAS), 2, 3))  # delta, token, phase -> orders posted
nm = 0
for fn in os.listdir(D + "tapes"):
    m = mk.get(fn.split(".")[0])
    if not m or m["city"] not in city_tz:
        continue
    try:
        tape = json.load(gzip.open(D + "tapes/" + fn, "rt"))
    except Exception:
        continue
    if len(tape) < 5:
        continue
    a = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in tape], dtype=np.float64)
    a = a[(a[:, 4] >= 0.001) & (a[:, 4] <= 0.999)]
    a = a[np.argsort(a[:, 0], kind="stable")]
    ts, tok, side, size, p = a.T
    # prints that hit a Yes bid (maker buys Yes) / a No bid, as the price the maker paid
    yes_hit = ((tok == 0) & (side == -1)) | ((tok == 1) & (side == 1))
    hits = []
    for k, msk in ((0, yes_hit), (1, ~yes_hit)):
        t_k = ts[msk]
        p_k = np.where(tok[msk] == k, p[msk], 1 - p[msk])
        hits.append((t_k, p_k))
    if len(hits[0][0]) == 0 or len(hits[1][0]) == 0:
        continue
    tz = city_tz[m["city"]]
    d0 = dt.datetime.fromisoformat(m["day"]).replace(tzinfo=tz).timestamp()
    T = np.arange(math.ceil(ts[0] / 3600) * 3600 + 3600, ts[-1], 3600.0)
    if len(T) == 0:
        continue
    last = []
    for t_k, p_k in hits:
        i = np.searchsorted(t_k, T, side="right") - 1
        ok = (i >= 0) & (t_k[np.maximum(i, 0)] >= T - LB)
        last.append((np.where(ok, p_k[np.maximum(i, 0)], np.nan)))
    b, n = last
    r = (b + (1 - n)) / 2
    spread = (1 - n) - b
    okT = ~np.isnan(r) & (spread > -0.05) & (spread < 0.2)
    if not okT.any():
        continue
    phase = np.where(T < d0, 0, np.where(T < d0 + 86400, 1, 2)).astype(int)
    fin = m["prices"]
    dord = dt.date.fromisoformat(m["day"]).toordinal()
    for k in (0, 1):
        t_k, p_k = hits[k]
        i0 = np.searchsorted(t_k, T, side="right")
        i1 = np.searchsorted(t_k, T + H, side="right")
        wmin = np.full(len(T), np.inf)
        for j in np.nonzero((i1 > i0) & okT)[0]:
            wmin[j] = p_k[i0[j]:i1[j]].min()
        rk = r if k == 0 else 1 - r
        for di, d in enumerate(DELTAS):
            L = np.floor((rk - d) * 100 + 1e-9) / 100
            valid = okT & (L >= 0.01) & (L <= 0.99)
            np.add.at(posted[di, k], phase[valid], 1)
            f = valid & (wmin < L - 1e-9)
            for j in np.nonzero(f)[0]:
                out.append((di, k, L[j], fin[k], phase[j], (d0 - T[j]) / 3600, dord, city_ix[m["city"]], rk[j], spread[j], T[j], nm))
    nm += 1
np.save(OUT, np.array(out))
np.save(OUT.replace(".npy", "_posted.npy"), posted)
print("markets", nm, "fills", len(out))
