"""Microstructure of the temperature markets, from the trade tapes: what takers pay a maker, and what the winner
trades at once the day is over.

    python weather_micro.py DATA_DIR

Per market, trades are split by the city's clock: before the day (quotes only face forecast changes), in play
(METAR readers are informed), after the day (the max is known, only the resolution is pending). Spread = the gap
between what takers paid when buying Yes and what they got when selling Yes within two hours of each other.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

DATA = Path(sys.argv[1]) / "weather"
U = dt.timezone.utc


def rnd(x):
    return math.floor(x + 0.5)


mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items():
    city_st.setdefault(s["city"], icao)

obs_day = {}
for icao, s in st.items():
    p = DATA / "obs" / f"{icao}.csv"
    if not p.exists():
        continue
    tz = ZoneInfo(s["tz"])
    vals = defaultdict(list)
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""):
            continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U).astimezone(tz)
            f = float(parts[2])
        except ValueError:
            continue
        v = rnd(f) if s["unit"] == "F" else rnd((f - 32) * 5 / 9)
        vals[t.date().isoformat()].append(v)
    for d, vs in vals.items():
        if len(vs) >= 12:
            obs_day[(icao, d)] = max(vs)

# resolution vs METAR, by unit
agree = defaultdict(lambda: [0, 0])
events = defaultdict(list)
for m in mk:
    if m["closed"] and m["kind"] == "highest":
        pr = m.get("prices") or [None, None]
        if pr[0] is not None and (pr[0] >= 0.99 or pr[0] <= 0.01):
            m["final"] = 1.0 if pr[0] >= 0.99 else 0.0
            events[(m["city"], m["day"])].append(m)
for (city, day), ms in events.items():
    icao = city_st.get(city)
    val = obs_day.get((icao, day)) if icao else None
    w = [m for m in ms if m["final"] == 1.0]
    if val is None or len(w) != 1:
        continue
    w = w[0]
    ok = (w["lo"] is None or val >= w["lo"]) and (w["hi"] is None or val <= w["hi"])
    agree[ms[0]["unit"]][0 if ok else 1] += 1
for u, (a, b) in agree.items():
    print(f"resolution matches the METAR max: °{u} cities {100 * a / (a + b):.1f}% ({b} of {a + b} disagree)")

spreads = defaultdict(list)
vol = defaultdict(float)
n_mk = 0
win_buy = defaultdict(lambda: [0.0, 0.0])     # window → [$ paid, shares] takers buying the winning token
win_sell = defaultdict(lambda: defaultdict(float))  # window → price ceiling → $ takers sold the winner for
days = set()
for (city, day), ms in events.items():
    icao = city_st.get(city)
    if not icao:
        continue
    tz = ZoneInfo(st[icao]["tz"])
    d0 = dt.datetime.fromisoformat(day).replace(tzinfo=tz).timestamp()
    d1 = d0 + 86400
    for m in ms:
        pth = DATA / "tapes" / f"{m['cid']}.json.gz"
        if not pth.exists():
            continue
        try:
            rows = json.load(gzip.open(pth, "rt"))
        except Exception:
            continue
        rows.sort(key=lambda r: r[0])
        n_mk += 1
        days.add(day)
        tr = []
        for ts_, idx, side, size, price, _w in rows:
            yes_px = price if idx == 0 else 1 - price
            buys_yes = (idx == 0 and side == 1) or (idx == 1 and side == -1)
            win = m["final"] == 1.0
            win_px = yes_px if win else 1 - yes_px
            buys_win = buys_yes if win else not buys_yes
            wdw = "before" if ts_ < d0 else "in play" if ts_ < d1 else "after"
            tr.append((ts_, yes_px, buys_yes, size, wdw))
            vol[wdw] += size * price
            if wdw == "after" or (wdw == "in play" and ts_ >= d1 - 4 * 3600):
                key = "after" if wdw == "after" else "last 4 h"
                if buys_win:
                    win_buy[key][0] += size * win_px
                    win_buy[key][1] += size
                else:
                    for c in (0.90, 0.95, 0.97, 0.98, 0.99):
                        if win_px <= c:
                            win_sell[key][c] += size * win_px
        # spread: consecutive opposite-side trades within 2 h
        for (t1, p1, b1, _s1, w1), (t2, p2, b2, _s2, w2) in zip(tr, tr[1:]):
            if b1 != b2 and t2 - t1 <= 7200 and w1 == w2:
                ask = p1 if b1 else p2
                bid = p2 if b1 else p1
                spreads[w1].append(ask - bid)

nd = max(1, len(days))
print(f"\n{n_mk} markets with tapes over {nd} market days")
print("taker volume per market day: " + " · ".join(f"{w} ${v / nd:,.0f}" for w, v in vol.items()))
for w in ("before", "in play", "after"):
    s = np.array(spreads[w])
    if len(s):
        print(f"spread takers crossed, {w:8s}: median {100 * np.median(s):.1f}¢ · mean {100 * s.mean():.1f}¢ · 25–75% {100 * np.quantile(s, .25):.1f}–{100 * np.quantile(s, .75):.1f}¢ ({len(s)} pairs)")
for key in ("last 4 h", "after"):
    paid, sh = win_buy[key]
    print(f"\n{key} of the day, the winning token: takers bought ${paid / nd:,.0f} a day at {100 * paid / sh if sh else 0:.1f}¢ avg")
    print("   takers sold it for at most: " + " · ".join(f"{100 * c:.0f}¢ ${v / nd:,.0f}/day" for c, v in sorted(win_sell[key].items())))
