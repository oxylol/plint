"""The live ledger, read every way: standings, checks, nested filters, pooled patterns and the sample still needed.

    python research/ledger_week.py DATA_DIR [OUT.json]      (DATA_DIR = data/polybot, holding ledger.jsonl and trades.jsonl)

1. Standings: every setup's return per signal at the real $5 ask, day-bootstrap 90% CI, paired by day against day4 and
   against edge ($ a day at $5 a signal).
2. Checks: the ledger against the paper books (same setup, same outcome: price logged vs price paid, outcome).
3. Nested filters on day4's own signals: what day4x, day4pre and day4model dropped and kept, and the paired gap.
4. Pooled patterns, one row per outcome (its first logged signal across all setups): in play or not, price band, kind.
5. The sample still needed: signals per setup for a 90% CI of ±5 and ±10 points per signal, and the days that takes at
   the live signal rate.
Outcomes come from Gamma (closed markets, winner at ≥ 99¢), cached in DATA_DIR/research3/ledger_finals.json.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polybot.categories import category as market_category  # noqa: E402
from polybot.ledger import settle  # noqa: E402

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "research3" / "ledger_week.json"
rows = [json.loads(line) for line in (DATA / "ledger.jsonl").open() if line.strip()]
cache_p = DATA / "research3" / "ledger_finals.json"
finals = json.loads(cache_p.read_text()) if cache_p.exists() else {}


def resolve():
    c = PublicClient()
    cids = sorted({r["condition_id"] for r in rows if r["asset_id"] not in finals})
    for i in range(0, len(cids), 40):
        ids = cids[i:i + 40]
        got = []
        for kw in ({"closed": True}, {}):
            try:
                got += c.list_markets(condition_ids=ids, page_size=len(ids), **kw).first_page().items
            except Exception as e:
                print("lookup failed", str(e)[:60])
        for m in got:
            if not m.state.closed:
                continue
            for o in (m.outcomes.yes, m.outcomes.no):
                if o.price is None:
                    continue
                p = float(o.price)
                f = 1.0 if p >= 0.99 else 0.0 if p <= 0.01 else None
                if f is not None:
                    for ident in (o.token_id, o.position_id):
                        if ident:
                            finals[str(ident)] = f
    cache_p.write_text(json.dumps(finals))


resolve()
S = settle(rows, finals)
for r in S:
    r["day"] = r["t"][:10]
    r["kind"] = market_category(r.get("title", ""), r.get("slug", ""))
    r["band"] = "<30¢" if r["price"] < 0.30 else "30–50¢" if r["price"] < 0.50 else "50–70¢" if r["price"] < 0.70 else "≥70¢"
first, last = min(r["t"] for r in rows)[:10], max(r["t"] for r in rows)[:10]
ndays = (dt.date.fromisoformat(last) - dt.date.fromisoformat(first)).days + 1
print(f"ledger {first} → {last} ({ndays} days): {len(rows)} signals logged, {len(S)} resolved, {len(set(r['asset_id'] for r in S))} distinct outcomes")
out = {"first": first, "last": last, "logged": len(rows), "resolved": len(S)}


def boot_mean(xs_by_day, n=4000, seed=1):
    days = list(xs_by_day)
    if len(days) < 3:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    ms = []
    for _ in range(n):
        pick = [days[rng.randrange(len(days))] for _ in days]
        cnt = sum(len(xs_by_day[d]) for d in pick)
        ms.append(sum(sum(xs_by_day[d]) for d in pick) / cnt)
    ms.sort()
    return ms[int(0.05 * n)], ms[int(0.95 * n)]


def stats(sel):
    by = defaultdict(list)
    for r in sel:
        by[r["day"]].append(r["ret"])
    xs = [r["ret"] for r in sel]
    lo, hi = boot_mean(by)
    sd = (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")
    return {"n": len(xs), "mean": sum(xs) / len(xs) if xs else float("nan"), "lo": lo, "hi": hi, "sd": sd, "days": len(by)}


def pct(x):
    return "   —  " if x != x else f"{100 * x:+6.1f}%"


# ── 1. standings
by_setup = defaultdict(list)
for r in S:
    by_setup[r["strategy"]].append(r)
logged = defaultdict(int)
for r in rows:
    logged[r["strategy"]] += 1


def paired(a, b, seed=3):
    """$ a day at $5 a signal, a − b, on the days both resolved signals; day bootstrap."""
    da, db = defaultdict(float), defaultdict(float)
    for r in by_setup[a]:
        da[r["day"]] += 5 * r["ret"]
    for r in by_setup[b]:
        db[r["day"]] += 5 * r["ret"]
    common = sorted(set(da) & set(db))
    if len(common) < 3:
        return None
    d = [da[x] - db[x] for x in common]
    rng = random.Random(seed)
    bm = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(4000))
    return sum(d) / len(d), bm[200], bm[3799], len(common)


print("\n## 1. Standings (every logged signal at its real $5 ask, held to resolution, after fee)")
print(f"{'setup':11s} {'logged':>6s} {'resolved':>8s} {'/day':>5s} {'per signal':>10s} {'90% CI':>17s} | {'vs day4 $/day [90% CI]':>28s} | {'vs edge $/day [90% CI]':>28s}")
standings = []
for name in sorted(logged, key=lambda k: -stats(by_setup[k])["mean"] if by_setup[k] else 9):
    st = stats(by_setup[name])
    g4, ge = paired(name, "day4") if name != "day4" else None, paired(name, "edge") if name != "edge" else None
    f = lambda g: f"{g[0]:+6.2f} [{g[1]:+6.2f}, {g[2]:+6.2f}] {g[3]}d" if g else ""  # noqa: E731
    print(f"{name:11s} {logged[name]:6d} {st['n']:8d} {logged[name] / ndays:5.1f} {pct(st['mean']):>10s} [{pct(st['lo'])}, {pct(st['hi'])}] | {f(g4):>28s} | {f(ge):>28s}")
    standings.append({"setup": name, "logged": logged[name], "per_day": logged[name] / ndays, **st,
                      "vs_day4": g4, "vs_edge": ge})
out["standings"] = standings
beats = [s for s in standings if s["vs_day4"] and s["vs_day4"][1] > 0]
print("setups beating day4 with the paired 90% CI above zero:", ", ".join(s["setup"] for s in beats) or "none")

# ── 2. checks against the paper books
trades = [json.loads(line) for line in (DATA / "trades.jsonl").open() if line.strip()]
closed = {(t.get("strategy"), t["asset_id"]): t for t in trades if t.get("side") == "SELL" and t.get("reason") == "resolved"}
bought = {(t.get("strategy"), t["asset_id"]): t for t in trades if t.get("side") == "BUY"}
n_chk = n_px = n_out = 0
gaps = []
for r in S:
    if not r.get("taken"):
        continue
    k = (r["strategy"], r["asset_id"])
    if k in bought:
        n_px += 1
        gaps.append(float(bought[k]["price"]) - float(r["price"]))
    if k in closed:
        n_chk += 1
        n_out += (float(closed[k]["price"]) >= 0.5) == (r["final"] >= 0.5)
print(f"\n## 2. Checks: {sum(1 for r in S if r.get('taken'))} resolved ledger signals were bought; {n_px} matched to a paper buy "
      f"(paid − logged: median {100 * sorted(gaps)[len(gaps) // 2] if gaps else 0:+.2f}¢, max |gap| {100 * max(map(abs, gaps)) if gaps else 0:.2f}¢); "
      f"{n_chk} settled in the book, outcome agrees on {n_out}")
out["checks"] = {"matched_buys": n_px, "settled": n_chk, "outcome_agree": n_out,
                 "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else None}

# ── 3. nested filters on day4's signals
print("\n## 3. Nested filters on day4's own signals (matched by outcome)")
d4 = {r["asset_id"]: r for r in by_setup["day4"]}
nested = {}
for name in ("day4x", "day4pre", "day4model"):
    mine = {r["asset_id"] for r in by_setup[name]}
    kept = [r for a, r in d4.items() if a in mine]
    dropped = [r for a, r in d4.items() if a not in mine]
    extra = [r for r in by_setup[name] if r["asset_id"] not in d4]
    sk, sd_ = stats(kept) if kept else {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}, \
        stats(dropped) if dropped else {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    print(f"  {name:10s} kept {sk['n']:3d} of day4's resolved signals at {pct(sk['mean'])}; dropped {sd_['n']:3d} at {pct(sd_['mean'])} "
          f"[{pct(sd_['lo'])}, {pct(sd_['hi'])}]; {len(extra)} signals day4 did not log")
    nested[name] = {"kept": sk, "dropped": sd_, "extra": len(extra)}
out["nested"] = nested

# ── 4. pooled patterns, one row per outcome
firsts = {}
for r in sorted(S, key=lambda r: r["t"]):
    firsts.setdefault(r["asset_id"], r)
P = list(firsts.values())
print(f"\n## 4. Pooled: {len(P)} distinct resolved outcomes (first signal across all setups)")
pooled = {}
for label, key in (("in play", lambda r: "in play" if r.get("in_play") else "before / not a game"), ("price", lambda r: r["band"]),
                   ("kind", lambda r: r["kind"])):
    groups = defaultdict(list)
    for r in P:
        groups[key(r)].append(r)
    for g, sel in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(sel) < 15:
            continue
        st = stats(sel)
        print(f"  {label:8s} {g:22s} {st['n']:4d}  {pct(st['mean'])} [{pct(st['lo'])}, {pct(st['hi'])}]")
        pooled[f"{label}|{g}"] = st
out["pooled"] = pooled

# ── 5. the sample still needed
print("\n## 5. Signals needed for a 90% CI of ±10 / ±5 points per signal (from each setup's spread of returns)")
need = {}
for s in standings:
    if s["n"] < 10 or s["sd"] != s["sd"]:
        continue
    n10, n5 = math.ceil((1.645 * s["sd"] / 0.10) ** 2), math.ceil((1.645 * s["sd"] / 0.05) ** 2)
    rate = max(s["per_day"], 1e-9)
    print(f"  {s['setup']:11s} sd {s['sd']:.2f}: ±10 pts ≈ {n10:5d} signals ({n10 / rate:6.0f} days at {rate:.1f}/day) · ±5 pts ≈ {n5:5d} ({n5 / rate:6.0f} days)")
    need[s["setup"]] = {"sd": s["sd"], "n10": n10, "n5": n5, "days10": n10 / rate, "days5": n5 / rate}
out["need"] = need
OUT.write_text(json.dumps(out, indent=1, default=float))
