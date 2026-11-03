"""When is a day's high final enough to buy, and is anything still for sale then?

    python weather_lock.py DATA_DIR

For each lock rule (from local hour H, once the reading is M °C below the running high) and each finished event:
the lock moment from the station's reports, whether the final high ended in the same bucket (else the position
loses), and from the trade tapes what takers paid for the winning tokens (the running winner's Yes, the other
buckets' No) in the hour after the lock at 97.5¢ or less, i.e. what a taker could have bought then.
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
BAD = {"Hong Kong", "Shenzhen", "Denver", "Moscow", "Seoul"}
MAXP = 0.975
rnd = lambda x: math.floor(x + 0.5)  # noqa: E731

mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items():
    city_st.setdefault(s["city"], icao)
obs = defaultdict(list)   # (icao, day) → [(local dt, °C float)]
for icao, s in st.items():
    p = DATA / "obs" / f"{icao}.csv"
    if not p.exists():
        continue
    tz = ZoneInfo(s["tz"])
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""):
            continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U).astimezone(tz)
            f = float(parts[2])
        except ValueError:
            continue
        obs[(icao, t.date().isoformat())].append((t, (f - 32) * 5 / 9))
for k in obs:
    obs[k].sort()
events = defaultdict(list)
for m in mk:
    if m["closed"] and m["kind"] == "highest" and m["city"] not in BAD:
        pr = m.get("prices") or [None, None]
        if pr[0] is not None and (pr[0] >= 0.99 or pr[0] <= 0.01):
            m["final"] = 1.0 if pr[0] >= 0.99 else 0.0
            events[(m["city"], m["day"])].append(m)


def bucket_of(ms, v):
    for m in ms:
        if (m["lo"] is None or v >= m["lo"]) and (m["hi"] is None or v <= m["hi"]):
            return m
    return None


def tape(cid):
    p = DATA / "tapes" / f"{cid}.json.gz"
    if not p.exists():
        return None
    try:
        rows = json.load(gzip.open(p, "rt"))
    except Exception:
        return None
    rows.sort(key=lambda r: r[0])
    return rows


RULES = [(h, m) for h in (15, 16, 17, 18, 19, 20) for m in (0.5, 1.0, 2.0)]
res = {r: {"locked": 0, "wrong": 0, "usd": 0.0, "paid": 0.0, "shares": 0.0, "first": [], "events_with_flow": 0, "payoff": 0.0, "fee": 0.0,
           "band": defaultdict(lambda: [0.0, 0.0, 0.0])} for r in RULES}
days = set()
tapes_cache = {}
for (city, day), ms in events.items():
    icao = city_st.get(city)
    o = obs.get((icao, day)) if icao else None
    if not o or len(o) < 12:
        continue
    unit = ms[0]["unit"]
    conv = (lambda c: c * 9 / 5 + 32) if unit == "F" else (lambda c: c)
    final_max = max(c for _, c in o)
    final_bucket = bucket_of(ms, rnd(conv(final_max)))
    resolved = [m for m in ms if m["final"] == 1.0]
    if final_bucket is None or len(resolved) != 1 or resolved[0]["cid"] != final_bucket["cid"]:
        continue   # the resolution did not follow the station's reports: not a case the rule can know about
    days.add(day)
    tz = ZoneInfo(st[icao]["tz"])
    for h, margin in RULES:
        run_max = None
        t_lock = None
        for t, c in o:
            run_max = c if run_max is None else max(run_max, c)
            if t.hour >= h and run_max - c >= margin:
                t_lock = t
                break
        if t_lock is None:
            continue
        r = res[(h, margin)]
        r["locked"] += 1
        wb = bucket_of(ms, rnd(conv(run_max)))
        if wb is None:
            continue
        if wb["cid"] != final_bucket["cid"]:
            r["wrong"] += 1
        t0 = t_lock.timestamp()
        got = False
        firsts = []
        for m in ms:
            win = m["cid"] == wb["cid"]
            rows = tapes_cache.get(m["cid"])
            if rows is None:
                rows = tapes_cache[m["cid"]] = tape(m["cid"]) or []
            for ts_, idx, side, size, price, _w in rows:
                if ts_ <= t0:
                    continue
                if ts_ > t0 + 3600:
                    break
                yes_px = price if idx == 0 else 1 - price
                buys_yes = (idx == 0 and side == 1) or (idx == 1 and side == -1)
                win_px = yes_px if win else 1 - yes_px
                buys_win = buys_yes if win else not buys_yes
                if buys_win:
                    if not firsts or firsts[-1][0] != m["cid"]:
                        firsts.append((m["cid"], win_px))
                    if win_px <= MAXP:
                        r["usd"] += size * win_px
                        r["paid"] += size * win_px
                        r["shares"] += size
                        pays = (m["cid"] == final_bucket["cid"]) if win else (m["cid"] != final_bucket["cid"])
                        r["payoff"] += size * (1.0 if pays else 0.0)
                        r["fee"] += size * m.get("fee_rate", 0.05) * win_px * (1 - win_px)
                        b = r["band"]["≤90¢" if win_px <= 0.90 else "90–95¢" if win_px <= 0.95 else "95–97.5¢"]
                        b[0] += size * win_px; b[1] += size * (1.0 if pays else 0.0); b[2] += size * m.get("fee_rate", 0.05) * win_px * (1 - win_px)
                        got = True
        if got:
            r["events_with_flow"] += 1
        r["first"].extend(px for _, px in firsts)
nd = max(1, len(days))
print(f"{len(days)} market days, {sum(1 for k in events)} events\n")
print("lock rule                 locked  wrong   buyable ≤97.5¢/h after   paid   events/day   first buy   return on the flow after fee   by price band (return)")
for (h, margin), r in res.items():
    if not r["locked"] or not r["paid"]:
        continue
    vw = r["paid"] / r["shares"] if r["shares"] else float("nan")
    first = np.median(r["first"]) if r["first"] else float("nan")
    ret = (r["payoff"] - r["paid"] - r["fee"]) / r["paid"]
    bands = " · ".join(f"{k} ${v[0] / nd:,.0f}/d {100 * (v[1] - v[0] - v[2]) / v[0]:+.1f}%" for k, v in sorted(r["band"].items()) if v[0] > 0)
    print(f"from {h:02d}:00, {margin:3.1f}°C below   {r['locked']:5d}  {100 * r['wrong'] / r['locked']:5.2f}%   ${r['usd'] / nd:7,.0f}   {100 * vw:5.1f}¢   {r['events_with_flow'] / nd:5.1f}      {100 * first:5.1f}¢      {100 * ret:+6.1f}%    {bands}")
