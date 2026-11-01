"""Kalshi's price for the same games, minute by minute before kick-off: is the other exchange information Polymarket lacks?

    python collect_kalshi.py DATA_DIR [rate=8]

Lists Kalshi's settled game-winner markets (public API, no key) for the leagues both exchanges carry, matches each to a
Polymarket moneyline in collect3_kickoff.py's set by league, local game date and team codes (Kalshi
`KXNBAGAME-26OCT31ATLNOP-NOP` ↔ Polymarket `nba-atl-nop-2026-10-31`, the team's token by slug order; soccer
`…-SUN` ↔ `epl-sun-lee-2026-10-25-sun` Yes), and fetches Kalshi's 1-minute candles (bid, ask, last, volume) from
75 minutes before kick-off to 5 after. Writes DATA_DIR/kalshi/{markets.json, matched.json, candles.jsonl}.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(sys.argv[1])
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"nfl": "KXNFLGAME", "nba": "KXNBAGAME", "nhl": "KXNHLGAME", "mlb": "KXMLBGAME", "wnba": "KXWNBAGAME", "cfb": "KXNCAAFGAME",
          "epl": "KXEPLGAME", "ucl": "KXUCLGAME", "lal": "KXLALIGAGAME", "bun": "KXBUNDESLIGAGAME", "fl1": "KXLIGUE1GAME",
          "sea": "KXSERIEAGAME", "mls": "KXMLSGAME"}
MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
OUT = DATA / "kalshi"
OUT.mkdir(exist_ok=True)
lock = threading.Lock()
nxt = [time.monotonic()]


def get(path):
    for i in range(6):
        with lock:
            w = max(time.monotonic(), nxt[0])
            nxt[0] = w + 1 / RATE
        time.sleep(max(0.0, w - time.monotonic()))
        try:
            with urllib.request.urlopen(API + path, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def list_markets():
    p = OUT / "markets.json"
    if p.exists():
        return json.loads(p.read_text())
    since = int(dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc).timestamp())
    rows = []
    for lg, s in SERIES.items():
        cur = ""
        n = 0
        while True:
            d = get(f"/markets?series_ticker={s}&status=settled&min_close_ts={since}&limit=1000" + (f"&cursor={cur}" if cur else ""))
            if not d:
                break
            for m in d.get("markets", []):
                rows.append({"lg": lg, **{k: m.get(k) for k in ("ticker", "event_ticker", "yes_sub_title", "result", "occurrence_datetime",
                                                               "close_time", "volume_fp")}})
                n += 1
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
        print(f"{lg}: {n} settled Kalshi markets", flush=True)
    p.write_text(json.dumps(rows))
    return rows


def match(rows):
    mk = json.loads((DATA / "kickoff" / "markets.json").read_text())
    poly = {}
    for m in mk.values():
        if m["sports_type"] not in ("moneyline", "child_moneyline"):
            continue
        parts = m["slug"].split("-")
        lg = parts[0]
        if lg not in SERIES:
            continue
        try:
            date = dt.date(int(parts[-4]), int(parts[-3]), int(parts[-2])) if parts[-1].isalpha() else dt.date(int(parts[-3]), int(parts[-2]), int(parts[-1]))
        except (ValueError, IndexError):
            continue
        if parts[-1].isalpha():           # soccer: one Yes/No market per team, slug ends with the team
            a, b, team = parts[1], parts[2], parts[-1]
            poly[(lg, date, team.upper(), frozenset((a.upper(), b.upper())))] = (m["cid"], 0)
        else:                             # two-way: tokens in slug order
            a, b = parts[1], parts[2]
            poly[(lg, date, a.upper(), frozenset((a.upper(), b.upper())))] = (m["cid"], 0)
            poly[(lg, date, b.upper(), frozenset((a.upper(), b.upper())))] = (m["cid"], 1)
    out = []
    for r in rows:
        ev = r["event_ticker"].split("-", 1)[1] if "-" in r["event_ticker"] else ""
        try:
            date = dt.date(2000 + int(ev[:2]), MONTHS[ev[2:5]], int(ev[5:7]))
        except (KeyError, ValueError):
            continue
        teams = ev[7:]
        side = r["ticker"].rsplit("-", 1)[-1]
        if side == "TIE":
            continue
        other = teams.replace(side, "", 1) if teams.startswith(side) or teams.endswith(side) else None
        if not other:
            continue
        key = (r["lg"], date, side, frozenset((side, other)))
        if key in poly:
            cid, k = poly[key]
            out.append({**r, "cid": cid, "k": k, "game_start": mk[cid]["game_start"], "final_poly": mk[cid]["prices"][k],
                        "fee_rate": mk[cid].get("fee_rate"), "vol_poly": mk[cid].get("vol")})
    return out


def main():
    rows = list_markets()
    m = match(rows)
    agree = sum((r["result"] == "yes") == (r["final_poly"] > 0.5) for r in m)
    print(f"{len(rows)} Kalshi markets, {len(m)} matched to Polymarket; outcomes agree on {agree}", flush=True)
    (OUT / "matched.json").write_text(json.dumps(m))
    cp = OUT / "candles.jsonl"
    done = set()
    if cp.exists():
        done = {json.loads(line)["ticker"] for line in cp.open() if line.strip()}
    todo = [r for r in m if r["ticker"] not in done]

    def one(r):
        g = int(ts(r["game_start"]))
        s = SERIES[r["lg"]]
        d = get(f"/series/{s}/markets/{r['ticker']}/candlesticks?start_ts={g - 75 * 60}&end_ts={g + 300}&period_interval=1")
        return r["ticker"], (d or {}).get("candlesticks")
    with cp.open("a") as f, ThreadPoolExecutor(max_workers=4) as ex:
        for n, (t, c) in enumerate(ex.map(one, todo), 1):
            if c is not None:
                f.write(json.dumps({"ticker": t, "c": [[x["end_period_ts"], x["yes_bid"].get("close_dollars"), x["yes_ask"].get("close_dollars"),
                                                        x["price"].get("close_dollars"), x.get("volume_fp")] for x in c]}) + "\n")
            if n % 500 == 0:
                f.flush()
                print(f"{dt.datetime.now():%H:%M:%S} {n}/{len(todo)}", flush=True)
    print("kalshi done", flush=True)


main()
