"""Daily temperature markets with what would have been known: the forecasts, the METAR actuals, the prices.

    python collect3_weather.py OUT_DIR markets [since YYYY-MM-DD]   # every daily temperature market, closed and open
    python collect3_weather.py OUT_DIR obs                          # METAR (IEM archive) for every resolving station
    python collect3_weather.py OUT_DIR forecasts                    # Open-Meteo previous-day runs of the bot's models
    python collect3_weather.py OUT_DIR tapes [workers]              # every trade in every temperature market (prices at decision time)

Writes OUT_DIR/weather/{markets.jsonl.gz, obs/<ICAO>.csv, forecasts/<ICAO>.json, tapes/<cid>.json.gz}. Resumable.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot.weather import parse_question, parse_station, station_info  # noqa: E402

OUT = Path(sys.argv[1]) / "weather"
STAGE = sys.argv[2]
ARGS = sys.argv[3:]
OUT.mkdir(parents=True, exist_ok=True)
U = dt.timezone.utc
c = PublicClient()
lock = threading.Lock()
MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]


def log(*a):
    with lock:
        print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def call(fn, tries=6):
    delay = 2
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay = min(60, delay * 2)


def get_json(url, tries=5):
    delay = 3
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def stage_markets():
    since = dt.datetime.fromisoformat(ARGS[0]).replace(tzinfo=U) if ARGS else dt.datetime(2026, 5, 1, tzinfo=U)
    path = OUT / "markets.jsonl.gz"
    have = set()
    if path.exists():
        with gzip.open(path, "rt") as f:
            for line in f:
                have.add(json.loads(line)["cid"])
    out = gzip.open(path, "at")
    day = since
    end_all = dt.datetime.now(U) + dt.timedelta(days=3)
    n_all = 0
    while day < end_all:
        lo = day.replace(hour=11, minute=58)
        n = 0
        for closed in (True, False):
            pages = c.list_markets(closed=closed, end_date_min=lo, end_date_max=lo + dt.timedelta(minutes=4), page_size=100)
            try:
                page = call(pages.first_page)
            except Exception as e:
                log(day.date(), "failed", str(e)[:60])
                continue
            k = 0
            while True:
                for m in page.items:
                    q = m.question or ""
                    if "temperature in" not in q or not m.condition_id or str(m.condition_id) in have:
                        continue
                    b = parse_question(q, m.state.end_date.year if m.state.end_date else day.year)
                    if b is None:
                        continue
                    have.add(str(m.condition_id))
                    tr = m.trading
                    fs = tr.fee_schedule if tr is not None else None
                    out.write(json.dumps({
                        "cid": str(m.condition_id), "slug": m.slug, "q": q, "ev": m.events[0].slug if m.events else None,
                        "kind": b.kind, "city": b.city, "unit": b.unit, "lo": b.lo, "hi": b.hi, "day": b.day.isoformat(),
                        "station": parse_station(m.description, b.city),
                        "tokens": [str(m.outcomes.yes.token_id or ""), str(m.outcomes.no.token_id or "")],
                        "prices": [float(m.outcomes.yes.price) if m.outcomes.yes.price is not None else None,
                                   float(m.outcomes.no.price) if m.outcomes.no.price is not None else None],
                        "closed": bool(m.state.closed), "end": m.state.end_date.isoformat() if m.state.end_date else None,
                        "start": m.state.start_date.isoformat() if m.state.start_date else None,
                        "closed_time": m.state.closed_time.isoformat() if m.state.closed_time else None,
                        "fee_rate": float(fs.rate or 0) if (fs is not None and tr.fees_enabled) else 0.0,
                        "liq": float(m.metrics.liquidity_num or 0), "vol": float(m.metrics.volume_num or 0), "nr": bool(m.state.neg_risk),
                    }) + "\n")
                    n += 1
                k += 1
                if not page.next_cursor or k >= 60:
                    break
                cur = page.next_cursor
                try:
                    page = call(lambda: pages.from_cursor(cur).first_page())
                except Exception:
                    break
        out.flush()
        n_all += n
        if n:
            log(f"{day.date()}: {n} temperature markets (total {n_all})")
        day += dt.timedelta(days=1)
    out.close()


def markets():
    with gzip.open(OUT / "markets.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


# Cities whose market text names the airport but not its ICAO code (read off the resolution rules by hand)
MORE_STATIONS = {"Seoul": "RKSI", "Jakarta": "WIHH", "Lagos": "DNMM", "Jinan": "ZSJN", "Zhengzhou": "ZHCC"}


def city_stations() -> dict[str, str]:
    """City → resolving station: every market of a city resolves on the same airport."""
    out = dict(MORE_STATIONS)
    for m in markets():
        if m.get("station"):
            out.setdefault(m["city"], m["station"])
    return out


def stations():
    out = {}
    cs = city_stations()
    for m in markets():
        icao = cs.get(m["city"])
        if icao and icao not in out:
            info = station_info(icao)
            if info:
                out[icao] = {**info, "city": m["city"], "unit": m["unit"]}
    (OUT / "stations.json").write_text(json.dumps(out, indent=1))
    return out


def stage_obs():
    st = stations()
    d = OUT / "obs"
    d.mkdir(exist_ok=True)
    days = [m["day"] for m in markets()]
    start = dt.date.fromisoformat(min(days)) - dt.timedelta(days=2)
    end = dt.date.today() + dt.timedelta(days=1)
    for icao in st:
        f = d / f"{icao}.csv"
        if f.exists() and f.stat().st_size > 1000 and dt.datetime.fromtimestamp(f.stat().st_mtime).date() >= dt.date.today():
            continue
        url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(
            {"station": icao, "data": "tmpf", "year1": start.year, "month1": start.month, "day1": start.day,
             "year2": end.year, "month2": end.month, "day2": end.day, "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
             "missing": "M", "trace": "T", "direct": "no"}) + "&report_type=3&report_type=4")
        for i in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    f.write_bytes(r.read())
                break
            except Exception as e:
                log(icao, "obs failed", str(e)[:60])
                time.sleep(5 * (i + 1))
        log(icao, f"{f.stat().st_size if f.exists() else 0} bytes")
        time.sleep(1)


def stage_forecasts():
    st = stations()
    d = OUT / "forecasts"
    d.mkdir(exist_ok=True)
    days = [m["day"] for m in markets()]
    start = dt.date.fromisoformat(min(days)) - dt.timedelta(days=2)
    end = min(dt.date.today() + dt.timedelta(days=2), dt.date.fromisoformat(max(days)))
    for icao, s in st.items():
        f = d / f"{icao}.json"
        if f.exists():
            continue
        out = {"station": icao, "tz": s["tz"]}
        for var in ("temperature_2m_previous_day1", "temperature_2m_previous_day2"):
            url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                "latitude": f"{s['lat']:.4f}", "longitude": f"{s['lon']:.4f}", "hourly": var, "models": ",".join(MODELS),
                "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": s["tz"]})
            try:
                out[var] = get_json(url).get("hourly")
            except Exception as e:
                log(icao, var, "failed", str(e)[:80])
            time.sleep(0.5)
        f.write_text(json.dumps(out))
        log(icao, "forecasts", list(out.get("temperature_2m_previous_day1", {}) or {})[:3])


def stage_tapes():
    """Every taker trade of the closed markets (kind, since): the prices a follower could really have had.
    Paced gently: the wallet collector shares the data API's /trades limit."""
    workers = int(ARGS[0]) if ARGS else 3
    kind = ARGS[1] if len(ARGS) > 1 else "highest"
    since = ARGS[2] if len(ARGS) > 2 else "2026-08-15"
    rate = float(ARGS[3]) if len(ARGS) > 3 else 5.0
    d = OUT / "tapes"
    d.mkdir(exist_ok=True)
    todo = [m for m in markets() if m["closed"] and m["kind"] == kind and m["day"] >= since and not (d / f"{m['cid']}.json.gz").exists()]
    todo.sort(key=lambda m: m["day"], reverse=True)
    log(f"{len(todo)} tapes to fetch")
    gap = 1.0 / rate
    nxt = [time.monotonic()]

    def pace():
        with lock:
            t = max(time.monotonic(), nxt[0])
            nxt[0] = t + gap
        time.sleep(max(0.0, t - time.monotonic()))

    def one(m):
        rows = []
        try:
            pages = c.list_trades(condition_id=m["cid"], full_history=True, page_size=500)
            pace()
            page = call(pages.first_page)
            k = 1
            while True:
                rows += [[int(t.timestamp.timestamp()), 0 if str(t.asset_id) == m["tokens"][0] else 1, 1 if str(t.side).upper() == "BUY" else -1,
                          float(t.size), float(t.price), str(t.wallet).lower()[-8:]] for t in page.items]
                if not page.next_cursor or k >= 6:
                    break
                cur = page.next_cursor
                pace()
                page = call(lambda: pages.from_cursor(cur).first_page())
                k += 1
        except Exception as e:
            log("tape failed", m["slug"], str(e)[:60])
            return 0
        tmp = d / f"{m['cid']}.tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(rows, f)
        tmp.replace(d / f"{m['cid']}.json.gz")
        return len(rows)

    n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(one, todo):
            n += 1
            if n % 500 == 0:
                log(f"{n}/{len(todo)} tapes, {time.time() - t0:.0f}s")


def stage_prices():
    """Midpoint history of every closed market's Yes token from the start of the day before to the end: 5-minute
    buckets while Polymarket keeps them (about 60 days), hourly before that."""
    workers = int(ARGS[0]) if ARGS else 6
    since = ARGS[1] if len(ARGS) > 1 else "2026-07-01"
    path = OUT / "prices.jsonl.gz"
    have = set()
    if path.exists():
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    have.add(json.loads(line)["cid"])
                except Exception:
                    pass
    five = time.time() - 59 * 86400
    todo = [m for m in markets() if m["closed"] and m["day"] >= since and m["cid"] not in have and m["tokens"][0]]
    todo.sort(key=lambda m: m["day"], reverse=True)
    log(f"{len(todo)} markets to price")
    out = gzip.open(path, "at")

    def one(m):
        d = dt.datetime.fromisoformat(m["day"]).replace(tzinfo=U)
        lo, hi = d - dt.timedelta(days=1), d + dt.timedelta(hours=12)
        b = 300 if lo.timestamp() >= five else 3600
        for i in range(5):
            try:
                pts = c.list_price_history(asset_id=m["tokens"][0], start=lo, end=hi, bucket_seconds=b).first_page().items
                row = {"cid": m["cid"], "b": b, "p": [[int(p.timestamp.timestamp()), float(p.price)] for p in pts]}
                with lock:
                    out.write(json.dumps(row) + "\n")
                return len(pts)
            except Exception:
                time.sleep(2 * (i + 1))
        return -1

    n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(one, todo):
            n += 1
            if n % 2000 == 0:
                with lock:
                    out.flush()
                log(f"{n}/{len(todo)}, {time.time() - t0:.0f}s")
    out.close()


{"markets": stage_markets, "obs": stage_obs, "forecasts": stage_forecasts, "tapes": stage_tapes, "prices": stage_prices}[STAGE]()
