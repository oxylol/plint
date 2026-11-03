import datetime as dt, gzip, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
DATA = Path(sys.argv[1]) / "weather"; U = dt.timezone.utc
BAD = {"Hong Kong", "Shenzhen", "Denver", "Moscow", "Seoul"}
rnd = lambda x: math.floor(x + 0.5)
mk = [json.loads(l) for l in gzip.open(DATA / "markets.jsonl.gz", "rt")]
st = json.loads((DATA / "stations.json").read_text())
city_st = {}
for icao, s in st.items(): city_st.setdefault(s["city"], icao)
obs = defaultdict(list)
for icao, s in st.items():
    p = DATA / "obs" / f"{icao}.csv"
    if not p.exists(): continue
    tz = ZoneInfo(s["tz"])
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3 or parts[2] in ("M", ""): continue
        try:
            t = dt.datetime.strptime(parts[1], "%Y-%m-%d %H:%M").replace(tzinfo=U).astimezone(tz); f = float(parts[2])
        except ValueError: continue
        v = rnd(f) if s["unit"] == "F" else rnd((f - 32) * 5 / 9)
        obs[(icao, t.date().isoformat())].append((t, v))
events = defaultdict(list)
for m in mk:
    if m["closed"] and m["kind"] == "highest" and m["city"] not in BAD:
        pr = m.get("prices") or [None, None]
        if pr[0] is not None and (pr[0] >= 0.99 or pr[0] <= 0.01):
            m["final"] = 1.0 if pr[0] >= 0.99 else 0.0; events[(m["city"], m["day"])].append(m)
# when is the day's max first reached, and is the resolution right (clean cities)?
hours = []; agree = [0, 0]
for (city, day), ms in events.items():
    icao = city_st.get(city); o = obs.get((icao, day)) if icao else None
    w = [m for m in ms if m["final"] == 1.0]
    if not o or len(o) < 12 or len(w) != 1: continue
    mx = max(v for _, v in o); hours.append(next(t for t, v in o if v == mx).hour + 0.0)
    w = w[0]; ok = (w["lo"] is None or mx >= w["lo"]) and (w["hi"] is None or mx <= w["hi"]); agree[0 if ok else 1] += 1
hours = np.array(hours)
print(f"clean cities: resolution matches the station max in {100 * agree[0] / sum(agree):.2f}% of {sum(agree)} events ({agree[1]} disagree)")
print("share of days whose max is first reached after local " + " · ".join(f"{h}:00 {100 * (hours >= h).mean():.1f}%" for h in (16, 17, 18, 19, 20, 21, 22)))
# the carry: winner tokens sold by takers after the day, per event and per day
days = set(); sold = defaultdict(lambda: defaultdict(float)); bought = defaultdict(lambda: [0.0, 0.0]); per_event = defaultdict(float)
for (city, day), ms in events.items():
    icao = city_st.get(city)
    if not icao: continue
    tz = ZoneInfo(st[icao]["tz"]); d0 = dt.datetime.fromisoformat(day).replace(tzinfo=tz).timestamp(); d1 = d0 + 86400
    for m in ms:
        pth = DATA / "tapes" / f"{m['cid']}.json.gz"
        if not pth.exists(): continue
        try: rows = json.load(gzip.open(pth, "rt"))
        except Exception: continue
        days.add(day)
        win = m["final"] == 1.0
        for ts_, idx, side, size, price, _w in rows:
            yes_px = price if idx == 0 else 1 - price
            buys_yes = (idx == 0 and side == 1) or (idx == 1 and side == -1)
            win_px = yes_px if win else 1 - yes_px; buys_win = buys_yes if win else not buys_yes
            if ts_ < d0 + 20 * 3600: continue
            key = "after 20:00 local" if ts_ < d1 else "next day, before resolution"
            if buys_win:
                bought[key][0] += size * win_px; bought[key][1] += size
            else:
                for c in (0.95, 0.97, 0.98, 0.99):
                    if win_px <= c: sold[key][c] += size * win_px
                if win_px <= 0.98: per_event[(city, day)] += size * win_px
nd = max(1, len(days))
for key in ("after 20:00 local", "next day, before resolution"):
    paid, sh = bought[key]
    print(f"\n{key}: takers bought the winner ${paid / nd:,.0f} a day at {100 * paid / sh if sh else 0:.1f}¢ avg; takers sold it at ≤ " + " · ".join(f"{100 * c:.0f}¢ ${v / nd:,.0f}/day" for c, v in sorted(sold[key].items())))
pe = np.array(list(per_event.values()))
print(f"\nwinner sold at ≤98¢ from 20:00 local on, per event: {len(pe)} events with any · median ${np.median(pe):.0f} · 75th pct ${np.quantile(pe, .75):.0f} · 90th ${np.quantile(pe, .9):.0f} · events per day {len(pe) / nd:.1f}")
