"""One hour for every city, or each city's own? The afternoon rule timed three ways.

    python weather_hour.py DATA_DIR

For every finished day: the hour the high was first reached, and the running bucket's real Yes ask at each hour
from 11:00 to 19:00 (the next price a taker paid within the hour). Then the rule "buy Yes on the running bucket when
the ask is 20–90¢" timed by (A) one hour everywhere, (B) each city's own hour: the hour by which its high had been
reached on q% of its earlier days (walk-forward, at least 15 days of history, else the global one), (C) the day's
forecast peak hour plus an offset. Returns per trade, held to resolution, taker fee paid, 90% intervals by day.
"""
from __future__ import annotations

import bisect
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
MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
HOURS = list(range(11, 20))
LO, HI = 0.20, 0.90
rng = np.random.default_rng(11)
rnd = lambda x: math.floor(x + 0.5)  # noqa: E731

mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items():
    city_st.setdefault(s["city"], icao)
obs = defaultdict(list)
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
fc_peak: dict[tuple, int] = {}
for icao, s in st.items():
    p = DATA / "forecasts" / f"{icao}.json"
    if not p.exists():
        continue
    h = (json.loads(p.read_text()).get("temperature_2m_previous_day1") or {})
    times = h.get("time") or []
    per_day = defaultdict(lambda: defaultdict(lambda: [None] * 24))
    for m in MODELS:
        for t, v in zip(times, h.get(f"temperature_2m_previous_day1_{m}") or []):
            if v is not None:
                per_day[t[:10]][m][int(t[11:13])] = v
    for day, bym in per_day.items():
        curves = [np.array(v, dtype=float) for v in bym.values() if all(x is not None for x in v)]
        if len(curves) >= 3:
            fc_peak[(icao, day)] = int(np.argmax(np.vstack(curves).mean(0)))
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


_tapes = {}


def ask_yes(cid, t, window=3600):
    tp = _tapes.get(cid)
    if tp is None:
        p = DATA / "tapes" / f"{cid}.json.gz"
        tp = _tapes[cid] = False
        if p.exists():
            try:
                rows = sorted(json.load(gzip.open(p, "rt")), key=lambda r: r[0])
                tp = _tapes[cid] = ([r[0] for r in rows], [(r[4] if r[1] == 0 else 1 - r[4]) for r in rows],
                                    [((r[1] == 0 and r[2] == 1) or (r[1] == 1 and r[2] == -1)) for r in rows])
            except Exception:
                pass
    if tp is False:
        return None
    ts, px, by = tp
    for j in range(bisect.bisect_right(ts, t), len(ts)):
        if ts[j] > t + window:
            return None
        if by[j]:
            return px[j]
    return None


# ── per event: the hour of the high, and at each hour the running bucket's ask and whether it held ──
ev = []
for (city, day), ms in sorted(events.items(), key=lambda kv: kv[0][1]):
    icao = city_st.get(city)
    o = obs.get((icao, day)) if icao else None
    if not o or len(o) < 12:
        continue
    unit = ms[0]["unit"]
    conv = (lambda c: c * 9 / 5 + 32) if unit == "F" else (lambda c: c)
    final_max = max(c for _, c in o)
    fb = bucket_of(ms, rnd(conv(final_max)))
    resolved = [m for m in ms if m["final"] == 1.0]
    if fb is None or len(resolved) != 1 or resolved[0]["cid"] != fb["cid"]:
        continue
    tz = ZoneInfo(st[icao]["tz"])
    first = next(t for t, c in o if c == final_max)
    e = {"city": city, "day": day, "max_hour": first.hour + first.minute / 60, "peak": fc_peak.get((icao, day)), "at": {}}
    for H in HOURS:
        t_dec = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(H, 0), tzinfo=tz)
        seen = [c for t, c in o if t <= t_dec]
        if not seen:
            continue
        wb = bucket_of(ms, rnd(conv(max(seen))))
        if wb is None:
            continue
        a = ask_yes(wb["cid"], t_dec.timestamp())
        if a is None:
            continue
        e["at"][H] = (a, 1.0 if wb["cid"] == fb["cid"] else 0.0, wb.get("fee_rate", 0.05))
    ev.append(e)
print(f"{len(ev)} finished days, {len({e['city'] for e in ev})} cities")

by_city = defaultdict(list)
for e in ev:
    by_city[e["city"]].append(e)


def trade(e, H):
    x = e["at"].get(H)
    if x is None:
        return None
    a, y, fee = x
    if not (LO <= a <= HI):
        return None
    return y / a - 1 - fee * (1 - a), a, y


def summarise(name, picks):
    picks = [p for p in picks if p is not None]
    if len(picks) < 30:
        print(f"  {name:58s} {len(picks)} trades")
        return
    r = np.array([p[0] for p in picks])
    dd = np.array([p[3] for p in picks])
    ud = sorted(set(dd))
    sums = np.array([r[dd == x].sum() for x in ud])
    cnts = np.array([(dd == x).sum() for x in ud])
    draws = rng.integers(0, len(ud), size=(2000, len(ud)))
    bm = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(bm, [0.05, 0.95])
    print(f"  {name:58s} {len(r):5d} trades on {len(ud):3d} days · {100 * r.mean():+5.1f}% per trade [90% CI {100 * lo:+.1f}, {100 * hi:+.1f}] · "
          f"paid {100 * np.mean([p[1] for p in picks]):.0f}¢ · held {100 * np.mean([p[2] for p in picks]):.0f}% · ${5 * r.sum() / len(ud) * 7:+.1f}/week at $5")


print("\n## A · one hour everywhere")
for H in (13, 14, 15, 16, 17):
    summarise(f"at {H:02d}:00", [(lambda t: t and (*t, e["day"]))(trade(e, H)) for e in ev])

print("\n## B · each city's own hour: the hour by which its high had been reached on q% of its earlier days (walk-forward)")
all_hours = np.array([e["max_hour"] for e in ev])
for q in (50, 60, 70, 80, 90):
    for shift in (0, 1):
        picks = []
        for city, es in by_city.items():
            es = sorted(es, key=lambda e: e["day"])
            hist = []
            for e in es:
                if len(hist) >= 15:
                    h = math.ceil(np.quantile(hist, q / 100)) + shift
                else:
                    h = math.ceil(np.quantile(all_hours, q / 100)) + shift
                h = min(max(h, 11), 19)
                t = trade(e, h)
                picks.append((*t, e["day"]) if t else None)
                hist.append(e["max_hour"])
        summarise(f"city's {q}th percentile hour{' + 1 h' if shift else ''}", picks)

print("\n## C · the day's forecast peak hour (day-before runs) plus an offset")
for k in (0, 1, 2, 3):
    picks = []
    for e in ev:
        if e["peak"] is None:
            continue
        h = min(max(e["peak"] + k, 11), 19)
        t = trade(e, h)
        picks.append((*t, e["day"]) if t else None)
    summarise(f"forecast peak + {k} h", picks)

print("\n## Per city: when the high lands, and the rule at 15:00 vs at the city's 80th-percentile hour (small samples: illustration only)")
print("  city             days  high reached: median / 80%   at 15:00: n  return      own hour: h  n  return")
for city, es in sorted(by_city.items(), key=lambda kv: np.median([e["max_hour"] for e in kv[1]])):
    mh = np.array([e["max_hour"] for e in es])
    h80 = min(max(math.ceil(np.quantile(mh, 0.8)), 11), 19)
    r15 = [trade(e, 15) for e in es]
    r15 = [t[0] for t in r15 if t]
    ro = [trade(e, h80) for e in es]
    ro = [t[0] for t in ro if t]
    print(f"  {city:16s} {len(es):4d}   {np.median(mh):5.1f} / {np.quantile(mh, 0.8):5.1f}           {len(r15):3d}  {100 * np.mean(r15) if r15 else float('nan'):+6.1f}%        {h80:02d}:00 {len(ro):3d}  {100 * np.mean(ro) if ro else float('nan'):+6.1f}%")

# ── the shipped fallback: each city's hour from every day here, until the bot has its own history ──
out = {}
for city, es in by_city.items():
    mh = np.array([e["max_hour"] for e in es])
    if len(mh) >= 15:
        out[city] = {"days": int(len(mh)), "median": round(float(np.median(mh)), 1),
                     **{f"q{q}": int(min(max(math.ceil(np.quantile(mh, q / 100)), 11), 19)) for q in (50, 60, 70, 80, 90)}}
Path("polybot/afternoon_hours.json").write_text(json.dumps({"from": "research/weather_hour.py", "days": f"{min(e['day'] for e in ev)} → {max(e['day'] for e in ev)}", "cities": dict(sorted(out.items()))}, indent=1))
print(f"\nwrote polybot/afternoon_hours.json for {len(out)} cities")
