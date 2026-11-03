import datetime as dt, gzip, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
from zoneinfo import ZoneInfo
DATA = Path(sys.argv[1]) / "weather"; U = dt.timezone.utc
rnd = lambda x: math.floor(x + 0.5)
mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items(): city_st.setdefault(s["city"], icao)
obs = {}; per_day = []
for icao, s in st.items():
    p = DATA / "obs" / f"{icao}.csv"
    if not p.exists(): continue
    tz = ZoneInfo(s["tz"]); vals = defaultdict(list); mins = defaultdict(set)
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""): continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U).astimezone(tz); f = float(parts[2])
        except ValueError: continue
        d = t.date().isoformat(); vals[d].append(f); mins[d].add(t.minute)
    for d, vs in vals.items():
        if len(vs) >= 12:
            per_day.append(len(vs))
            c = [(x - 32) * 5 / 9 for x in vs]
            obs[(icao, d)] = {"maxf": max(vs), "maxc": max(c), "n": len(vs), "minutes": sorted(mins[d])[:4]}
import statistics
print("reports per station-day: median", statistics.median(per_day), "· 90th pct", sorted(per_day)[int(.9*len(per_day))], "· sample minute marks", obs[next(iter(obs))]["minutes"])
events = defaultdict(list)
for m in mk:
    if m["closed"] and m["kind"] == "highest":
        pr = m.get("prices") or [None, None]
        if pr[0] is not None and (pr[0] >= 0.99 or pr[0] <= 0.01):
            m["final"] = 1.0 if pr[0] >= 0.99 else 0.0; events[(m["city"], m["day"])].append(m)
diff = defaultdict(Counter); bycity = Counter(); tot = Counter(); frac = defaultdict(Counter)
for (city, day), ms in events.items():
    icao = city_st.get(city); o = obs.get((icao, day)) if icao else None
    w = [m for m in ms if m["final"] == 1.0]
    if o is None or len(w) != 1: continue
    w = w[0]; unit = w["unit"]
    raw = o["maxf"] if unit == "F" else o["maxc"]; val = rnd(raw)
    tot[city] += 1
    ok = (w["lo"] is None or val >= w["lo"]) and (w["hi"] is None or val <= w["hi"])
    if ok: continue
    ref = w["lo"] if w["lo"] is not None else w["hi"]
    d = int(ref - val) if w["lo"] is not None and w["hi"] is not None else ("above" if w["lo"] is not None and val < w["lo"] else "below")
    diff[unit][d] += 1; bycity[city] += 1
    frac[unit][round(raw - math.floor(raw), 1)] += 1
for u in diff: print(f"°{u} mismatches, winner minus hourly METAR max (degrees; 'above'/'below' = open-ended bucket):", dict(sorted(diff[u].items(), key=lambda kv: str(kv[0]))))
print("fractional part of the raw max in the mismatches (°C: .4/.5/.6 = rounding edge):", {u: dict(sorted(frac[u].items())) for u in frac})
print("cities with the most disagreements (disagree/events):", [(c, f"{bycity[c]}/{tot[c]}") for c, _ in bycity.most_common(10)])
