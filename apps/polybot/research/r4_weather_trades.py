"""Who wins each taker trade in the temperature markets? Every tape row is a taker trade; the maker on the other
side holds the opposite claim to resolution. Maker 'buys' token X at q:
  taker BUY tok i at p  -> maker holds the complement at q = 1-p, pays off when tok i ends 0
  taker SELL tok i at p -> maker holds tok i at q = p
Reports maker return per dollar at risk (no fee: makers pay none), by q, by clock phase, by which side the taker took.
"""
import gzip, json, sys, math, datetime as dt, os
from collections import defaultdict
from zoneinfo import ZoneInfo
import numpy as np

D = sys.argv[1] + "/weather/"
st = json.load(open(D + "stations.json"))
city_tz = {}
for icao, s in st.items():
    city_tz.setdefault(s["city"], ZoneInfo(s["tz"]))
U = dt.timezone.utc
rows = []  # q, win, usd, phase, takerside(0 buyYes,1 buyNo,2 sellYes,3 sellNo), day_ord, eventhash, hours_to_daystart
mk = {}
for l in gzip.open(D + "markets.jsonl.gz", "rt"):
    m = json.loads(l)
    if m["closed"] and m["kind"] == "highest" and m["prices"] and set(m["prices"]) == {0.0, 1.0}:
        mk[m["cid"]] = m
n = 0
for fn in os.listdir(D + "tapes"):
    cid = fn.split(".")[0]
    m = mk.get(cid)
    if not m or m["city"] not in city_tz:
        continue
    tz = city_tz[m["city"]]
    d0 = dt.datetime.fromisoformat(m["day"]).replace(tzinfo=tz)
    t0, t1 = d0.timestamp(), d0.timestamp() + 86400
    fin = m["prices"]
    try:
        tape = json.load(gzip.open(D + "tapes/" + fn, "rt"))
    except Exception:
        continue
    ev = hash(m["ev"]) & 0xFFFFFFF
    dord = dt.date.fromisoformat(m["day"]).toordinal()
    for ts, tok, side, size, p, w in tape:
        if not (0.001 <= p <= 0.999) or size <= 0:
            continue
        if side == 1:   # taker buys tok -> maker holds complement
            q, win = 1 - p, 1.0 - fin[tok]
            tside = tok           # 0 buyYes 1 buyNo
            mtok = 1 - tok
        else:
            q, win = p, fin[tok]
            tside = 2 + tok
            mtok = tok
        usd = size * q
        phase = 0 if ts < t0 else (1 if ts < t1 else 2)
        rows.append((q, win, usd, phase, tside, dord, ev, (t0 - ts) / 3600.0, mtok, size))
    n += 1
a = np.array(rows, dtype=np.float64)
np.save(sys.argv[2], a)
print("markets", n, "trades", len(a))
