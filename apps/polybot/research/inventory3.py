"""Sizes, date ranges and gaps of every round-3 dataset, for REPORT3.md.

    python inventory3.py DATA_DIR
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(sys.argv[1])
U = dt.timezone.utc


def d(t):
    return dt.datetime.fromtimestamp(t, U).strftime("%Y-%m-%d") if t else "-"


def size(p):
    if p.is_file():
        return p.stat().st_size
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file())


out = []
boards = sorted((DATA / "boards").glob("*.json"))
uni = json.loads((DATA / "universe.json").read_text()) if (DATA / "universe.json").exists() else {}
rows_b = sum(len(json.loads(b.read_text())["rows"]) for b in boards)
out.append(f"| leaderboards | {len(boards)} boards (4 windows × 11 categories, by profit and by volume), {rows_b} rows, "
           f"{len({r['wallet'] for b in boards for r in json.loads(b.read_text())['rows']})} distinct wallets | snapshot 2026-10-31 | {size(DATA / 'boards') / 1e6:.0f} MB |")
ctl = json.loads((DATA / "controls.json").read_text()) if (DATA / "controls.json").exists() else {"markets": [], "takers": {}}
out.append(f"| control takers | {len(ctl['takers'])} wallets trading in {len(ctl['markets'])} randomly drawn resolved markets "
           f"(3000 of those seen in 2+ markets join the universe) | markets ending 2026-06-01 → 10-30 | - |")
n_w = n_c = n_r = n_o = n_f = n_tr = n_ctl = n_pnl = 0
fmin, fmax = None, None
for p in (DATA / "wallets").glob("*/*.json.gz"):
    try:
        w = json.load(gzip.open(p, "rt"))
    except Exception:
        continue
    n_w += 1
    n_c += len(w.get("closed", []))
    n_r += len(w.get("redeemable", []))
    n_o += len(w.get("open", []))
    f = w.get("fills") or []
    n_f += len(f)
    if f:
        a, b = min(x[0] for x in f), max(x[0] for x in f)
        fmin = a if fmin is None else min(fmin, a)
        fmax = b if fmax is None else max(fmax, b)
    n_tr += bool(w.get("truncated", {}).get("fills"))
    info = w.get("info", {})
    n_ctl += bool(info.get("control"))
    n_pnl += any(b[2] == "PNL" for b in info.get("boards", []))
out.append(f"| wallets | {n_w} collected of {len(uni)} in the universe: {n_pnl} on a profit board, {n_ctl} controls, "
           f"{n_w - n_pnl} not picked by profit | collected 2026-10-31 | {size(DATA / 'wallets') / 1e9:.2f} GB |")
out.append(f"| positions | {n_c} CLOSED + {n_r} REDEEMABLE + {n_o} OPEN rows (up to 3,000 per status per wallet) | full history | (in wallets) |")
out.append(f"| fills | {n_f} fills, maker and taker (newest 6,000 per wallet; {n_tr} wallets truncated, their signals start where their fills do) | "
           f"{d(fmin)} → {d(fmax)} | (in wallets) |")
nm = miss = closed = 0
if (DATA / "markets" / "markets.jsonl.gz").exists():
    for line in gzip.open(DATA / "markets" / "markets.jsonl.gz", "rt"):
        m = json.loads(line)
        nm += 1
        miss += bool(m.get("missing"))
        closed += bool(m.get("closed"))
out.append(f"| markets | {nm} touched by the positions and fills: {closed} closed with outcome, {miss} not found | - | {size(DATA / 'markets') / 1e6:.0f} MB |")
bk = sorted((DATA / "books").glob("*.jsonl.gz"))
nb = 0
for f in bk:
    try:
        nb += sum(1 for _ in gzip.open(f, "rt"))
    except EOFError:
        pass
out.append(f"| order books | {nb} snapshots (top 12 levels a side, $1–$100 walk costs) of ~6,000 markets every 20 min | "
           f"{bk[0].name[:10] if bk else '-'} → {bk[-1].name[:10] if bk else '-'} (UTC hours) | {size(DATA / 'books') / 1e6:.0f} MB |")
ns = npts = 0
b5 = b60 = 0
for f in (DATA / "prices").glob("*.jsonl.gz"):
    try:
        for line in gzip.open(f, "rt"):
            j = json.loads(line)
            ns += 1
            npts += len(j["p"])
            b5 += j["b"] == 300
            b60 += j["b"] == 3600
    except (EOFError, ValueError):
        pass
out.append(f"| price paths | {ns} spans ({b5} at 5-minute, {b60} at hourly buckets; midpoints), {npts} points, signal to resolution, ≤15 days each | "
           f"5-minute buckets exist from ~2026-09-01 only | {size(DATA / 'prices') / 1e6:.0f} MB |")
nt = len(list((DATA / "tapes").glob("*.json.gz"))) if (DATA / "tapes").exists() else 0
out.append(f"| thin-market tapes | {nt} markets (lifetime volume < $250k or temperature), every taker trade back to the first signal | signals since ~2026-08-31 | "
           f"{size(DATA / 'tapes') / 1e6 if (DATA / 'tapes').exists() else 0:.0f} MB |")
wx = DATA / "weather"
if wx.exists():
    mk = [json.loads(l) for l in gzip.open(wx / "markets.jsonl.gz", "rt")]
    kinds = Counter(m["kind"] for m in mk)
    days = sorted({m["day"] for m in mk})
    out.append(f"| temperature markets | {len(mk)} ({kinds['highest']} highest, {kinds['lowest']} lowest), 54 cities, 53 stations | {days[0]} → {days[-1]} | "
               f"{size(wx / 'markets.jsonl.gz') / 1e6:.0f} MB |")
    out.append(f"| METAR | {len(list((wx / 'obs').glob('*.csv')))} stations, every routine and special report (IEM archive) | 2026-04-29 → 2026-11-01 | {size(wx / 'obs') / 1e6:.0f} MB |")
    out.append(f"| forecasts | Open-Meteo previous-day and two-days-before runs, 5 models (the bot's), hourly, {len(list((wx / 'forecasts').glob('*.json')))} stations | "
               f"2026-04-29 → 2026-11-02 | {size(wx / 'forecasts') / 1e6:.0f} MB |")
    nwt = len(list((wx / "tapes").glob("*.json.gz"))) if (wx / "tapes").exists() else 0
    out.append(f"| temperature tapes | {nwt} highest-temperature markets, every taker trade | days 2026-08-15 → 10-31 | {size(wx / 'tapes') / 1e6 if (wx / 'tapes').exists() else 0:.0f} MB |")
if (DATA / "panel.parquet").exists():
    import pandas as pd
    P = pd.read_parquet(DATA / "panel.parquet", columns=["t_d", "wallet", "cid", "in_pool"])
    out.append(f"| signals (panel) | {len(P)} resolved, priced signals from {P.wallet.nunique()} wallets in {P.cid.nunique()} markets; {int(P.in_pool.sum())} from the rebuilt monthly top 200 | "
               f"{d(P.t_d.min())} → {d(P.t_d.max())} | {size(DATA / 'panel.parquet') / 1e6:.0f} MB |")
print("| dataset | what | dates | size |\n|---|---|---|---|")
print("\n".join(out))
print(f"\ntotal on disk: {size(DATA) / 1e9:.2f} GB")
