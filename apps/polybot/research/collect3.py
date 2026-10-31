"""Round 3 collector: a wide, mostly unbiased wallet universe with full positions and fills.

    python collect3.py OUT_DIR boards                 # leaderboards (PNL and VOLUME, every window × category)
    python collect3.py OUT_DIR controls [n_markets]   # control wallets: takers in a random sample of resolved markets
    python collect3.py OUT_DIR wallets [workers]      # per wallet: CLOSED + REDEEMABLE + OPEN positions, fills since FILLS_SINCE, join date
    python collect3.py OUT_DIR markets [workers]      # metadata for every market the positions and fills touch

Everything is cached on disk and every stage resumes where it stopped:
boards/<window>_<category>_<sort>.json, controls.json, wallets/<aa>/<wallet>.json.gz,
markets/markets.jsonl.gz. Requests are paced below Polymarket's published limits
(data API /v2/trades 300 per 10 s, /v2/positions 200 per 10 s).

Why the universe looks like this: a profit leaderboard picks traders *because* their
recent positions won, so a model trained only on them learns "leaderboard trader vs
nobody" and its test folds are flattered by the very wins that put the traders on the
board. Volume boards (selected on activity, not profit) and takers drawn from the
tapes of randomly chosen resolved markets give a control group whose selection does
not depend on outcomes.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from polymarket import PublicClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common3 import pos_row  # noqa: E402

OUT = Path(sys.argv[1])
STAGE = sys.argv[2]
ARGS = sys.argv[3:]
OUT.mkdir(parents=True, exist_ok=True)
U = dt.timezone.utc
FILLS_SINCE = dt.datetime(2026, 6, 1, tzinfo=U)
POS_PAGES = 6          # × 500 positions per status
FILL_PAGES = 12        # × 500 fills, newest first (market makers get truncated; flagged)
c = PublicClient()
lock = threading.Lock()


def log(*a):
    with lock:
        print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


class Pace:
    """A shared token bucket: at most `rate` calls a second across threads."""
    def __init__(self, rate: float):
        self.gap = 1.0 / rate
        self.next = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


PACE = {"trades": Pace(24), "positions": Pace(16), "gamma": Pace(20), "other": Pace(20)}


def call(fn, kind="other", tries=6):
    delay = 2
    for i in range(tries):
        PACE[kind].wait()
        try:
            return fn()
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()
            if i == tries - 1 or "NotFound" in name or ("Rejected" in name and "rate" not in msg and "timed out" not in msg):
                raise
            time.sleep(delay)
            delay = min(60, delay * 2)


def pages(paginator, kind, max_pages):
    page = call(paginator.first_page, kind)
    n = 1
    yield page.items
    while page.next_cursor and n < max_pages:
        cur = page.next_cursor
        page = call(lambda: paginator.from_cursor(cur).first_page(), kind)
        n += 1
        yield page.items
    if page.next_cursor:
        yield None   # marker: truncated


# ── boards ───────────────────────────────────────────────────
CATS = ["politics", "sports", "crypto", "culture", "economics", "tech", "finance", "weather", "mentions", "esports"]
BOARDS = [("day", None, "PNL", 3000), ("week", None, "PNL", 5000), ("month", None, "PNL", 8000), ("all", None, "PNL", 5000),
          ("day", None, "VOLUME", 1000), ("week", None, "VOLUME", 3000), ("month", None, "VOLUME", 5000), ("all", None, "VOLUME", 3000)]
for cat in CATS:
    BOARDS += [("month", cat, "PNL", 1500), ("week", cat, "PNL", 800), ("all", cat, "PNL", 800), ("day", cat, "PNL", 300),
               ("month", cat, "VOLUME", 500)]


def stage_boards():
    d = OUT / "boards"
    d.mkdir(exist_ok=True)
    for window, cat, sort, depth in BOARDS:
        f = d / f"{window}_{cat or 'overall'}_{sort}.json"
        if f.exists():
            continue
        rows = []
        try:
            kw = {"window": window, "sort_by": sort, "page_size": min(depth, 1000)}
            if cat:
                kw["category"] = cat
            for items in pages(c.list_trader_leaderboard(**kw), "other", (depth + 999) // 1000):
                if items is None:
                    break
                rows += [{"wallet": str(e.wallet).lower(), "name": e.user_name or "", "rank": int(e.rank), "pnl": float(e.pnl),
                          "volume": float(e.volume)} for e in items]
        except Exception as e:
            log("board", window, cat, sort, "failed:", str(e)[:80])
            continue
        f.write_text(json.dumps({"window": window, "category": cat or "overall", "sort": sort,
                                 "at": dt.datetime.now(U).isoformat(timespec="seconds"), "rows": rows[:depth]}))
        log(f"board {window}/{cat or 'overall'}/{sort}: {len(rows)}")


# ── controls: takers drawn from random resolved markets ───────
def stage_controls():
    n_markets = int(ARGS[0]) if ARGS else 1500
    f = OUT / "controls.json"
    if f.exists():
        log("controls.json exists")
        return
    rnd = random.Random(31)
    picked = []
    now = dt.datetime.now(U)
    day = FILLS_SINCE
    per_day = max(1, n_markets // max(1, (now - FILLS_SINCE).days))
    while day < now - dt.timedelta(days=1):
        try:
            items = call(lambda: c.list_markets(closed=True, end_date_min=day, end_date_max=day + dt.timedelta(days=1),
                                                order="volume", ascending=False, page_size=100).first_page().items, "gamma")
        except Exception as e:
            log("markets", day.date(), str(e)[:60])
            items = []
        # a random draw among the day's top-100 by volume (not just the top): busy enough to have a crowd
        items = [m for m in items if m.condition_id and float(m.metrics.volume_num or 0) > 5000]
        rnd.shuffle(items)
        picked += [(str(m.condition_id), m.slug, m.state.end_date.isoformat() if m.state.end_date else None) for m in items[:per_day]]
        day += dt.timedelta(days=1)
    log(f"{len(picked)} markets sampled for controls")
    takers: dict[str, dict] = {}
    for i, (cid, slug, end) in enumerate(picked):
        # the last 500 takers (often in-play) and 500 from before the final 12 hours
        items = []
        for kw in ({}, {"end": dt.datetime.fromisoformat(end) - dt.timedelta(hours=12)} if end else None):
            if kw is None:
                continue
            try:
                items += call(lambda: c.list_trades(condition_id=cid, page_size=500, **kw).first_page().items, "trades")
            except Exception:
                pass
        seen = set()
        for t in items:
            w = str(t.wallet).lower()
            if w in seen:
                continue
            seen.add(w)
            takers.setdefault(w, {"wallet": w, "name": t.name or t.pseudonym or "", "markets": 0})
            takers[w]["markets"] += 1
        if i % 200 == 0:
            log(f"  {i}/{len(picked)} markets, {len(takers)} takers")
    f.write_text(json.dumps({"markets": picked, "takers": takers}))
    log(f"controls: {len(takers)} takers from {len(picked)} markets")


# ── the universe, in priority order ──────────────────────────
# How deep into each board the full collection goes (about 11k wallets with the controls; the boards
# themselves are kept whole for reference). Fills are ~100 KB a page, so this keeps the pull near 10 GB.
CAPS = {("overall", "PNL"): {"day": 1000, "week": 1500, "month": 3000, "all": 1500},
        ("overall", "VOLUME"): {"day": 500, "week": 1000, "month": 1500, "all": 1000},
        ("cat", "PNL"): {"day": 100, "week": 150, "month": 300, "all": 150}, ("cat", "VOLUME"): {"month": 100}}


def depth_cap(window: str, cat: str, sort: str) -> int:
    return CAPS[("overall" if cat == "overall" else "cat", sort)].get(window, 0)


def universe() -> list[tuple[str, dict]]:
    order: dict[str, dict] = {}
    boards = sorted((OUT / "boards").glob("*.json"))
    # interleave boards by rank so a partial collection is balanced across boards and controls
    lists = []
    for f in boards:
        b = json.loads(f.read_text())
        lists.append((b["window"], b["category"], b["sort"], b["rows"][: depth_cap(b["window"], b["category"], b["sort"])]))
    ctl = []
    if (OUT / "controls.json").exists():
        tk = json.loads((OUT / "controls.json").read_text())["takers"]
        # takers seen in at least two sampled markets (enough history to learn from), in random order, 3000 of them
        ctl = sorted((t for t in tk.values() if t["markets"] >= 2), key=lambda t: t["wallet"])
        random.Random(17).shuffle(ctl)
        ctl = ctl[:3000]
    depth = max([len(r) for *_, r in lists] + [len(ctl)])
    rnd = random.Random(5)
    for i in range(depth):
        for window, cat, sort, rows in lists:
            if i < len(rows):
                r = rows[i]
                t = order.setdefault(r["wallet"], {"wallet": r["wallet"], "name": r["name"], "boards": [], "control": False})
                t["boards"].append([window, cat, sort, r["rank"]])
        if i < len(ctl):
            t = order.setdefault(ctl[i]["wallet"], {"wallet": ctl[i]["wallet"], "name": ctl[i]["name"], "boards": [], "control": False})
            t["control"] = True
    return list(order.items())


def wallet_path(w: str) -> Path:
    return OUT / "wallets" / w[2:4] / f"{w}.json.gz"


def collect_wallet(item) -> str:
    w, info = item
    path = wallet_path(w)
    if path.exists():
        return "skip"
    try:
        out = {"wallet": w, "info": info, "at": dt.datetime.now(U).isoformat(timespec="seconds"), "truncated": {}}
        for status in ("CLOSED", "REDEEMABLE", "OPEN"):
            rows = []
            for items in pages(c.list_positions(user=w, status=status, sort_by="TIMESTAMP", sort_direction="DESC", page_size=500),
                               "positions", POS_PAGES):
                if items is None:
                    out["truncated"][status] = True
                    break
                rows += [pos_row(p) for p in items]
            out[status.lower()] = rows
        fills = []
        for items in pages(c.list_trades(user=w, start=FILLS_SINCE, taker_only=False, page_size=500), "trades", FILL_PAGES):
            if items is None:
                out["truncated"]["fills"] = True
                break
            fills += [[int(t.timestamp.timestamp()), str(t.asset_id), str(t.condition_id), 1 if str(t.side).upper() == "BUY" else -1,
                       float(t.size), float(t.price), t.outcome_index, t.slug or "", t.event_slug or "", (t.title or "")[:90], t.outcome or "",
                       str(t.transaction_hash or "")[-10:]] for t in items]
        out["fills"] = fills
        # taker-only fills since June: which of their fills crossed the spread
        try:
            tk = []
            for items in pages(c.list_trades(user=w, start=FILLS_SINCE, taker_only=True, page_size=500), "trades", 4):
                if items is None:
                    break
                tk += [[int(t.timestamp.timestamp()), str(t.asset_id), str(t.transaction_hash or "")[-10:]] for t in items]
            out["taker"] = tk
        except Exception:
            out["taker"] = None
        try:
            st = call(lambda: c.get_user_stats(user=w))
            out["joined"] = st.join_date.isoformat() if (st is not None and st.join_date) else None
        except Exception:
            out["joined"] = None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(out, f)
        tmp.replace(path)
        return "ok"
    except Exception as e:
        log("  failed", w[:10], type(e).__name__, str(e)[:100])
        return "fail"


def stage_wallets():
    workers = int(ARGS[0]) if ARGS else 8
    limit = int(ARGS[1]) if len(ARGS) > 1 else 10**9
    uni = universe()[:limit]
    (OUT / "universe.json").write_text(json.dumps(dict(uni)))
    todo = [x for x in uni if not wallet_path(x[0]).exists()]
    log(f"universe {len(uni)} wallets ({sum(1 for _, t in uni if t['control'])} controls), {len(todo)} to collect")
    t0 = time.time()
    n = ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(collect_wallet, todo):
            n += 1
            ok += res == "ok"
            if n % 200 == 0:
                rate = n / (time.time() - t0)
                log(f"{n}/{len(todo)} wallets ({ok} ok), {rate:.2f}/s, eta {(len(todo) - n) / max(rate, 1e-6) / 3600:.1f} h")
    log("wallets done")


# ── markets ──────────────────────────────────────────────────
def market_row(m) -> dict:
    tr = m.trading
    fs = tr.fee_schedule if tr is not None else None
    yes, no = m.outcomes.yes, m.outcomes.no
    return {
        "cid": str(m.condition_id), "slug": m.slug, "q": (m.question or "")[:160], "version": str(getattr(m.version, "value", m.version)),
        "labels": [yes.label, no.label], "prices": [float(yes.price) if yes.price is not None else None, float(no.price) if no.price is not None else None],
        "tokens": [str(yes.token_id or ""), str(no.token_id or "")], "pos_ids": [str(yes.position_id or ""), str(no.position_id or "")],
        "start": m.state.start_date.isoformat() if m.state.start_date else None,
        "end": m.state.end_date.isoformat() if m.state.end_date else None,
        "closed_time": m.state.closed_time.isoformat() if m.state.closed_time else None, "closed": bool(m.state.closed),
        "nr": bool(m.state.neg_risk), "category": m.category, "tags": [t.label for t in (m.tags or []) if getattr(t, "label", None)],
        "ev": m.events[0].slug if m.events else None, "ev_id": str(m.events[0].id) if m.events else None,
        "ev_title": (m.events[0].title or "")[:120] if m.events else None,
        "sports_type": m.sports.sports_market_type if m.sports else None,
        "game_start": m.sports.game_start_time.isoformat() if (m.sports and m.sports.game_start_time) else None,
        "fee_on": bool(tr.fees_enabled) if tr is not None else False, "fee_rate": float(fs.rate or 0) if fs is not None else 0.0,
        "fee_type": tr.fee_type if tr is not None else None, "min_size": float(tr.minimum_order_size or 0) if tr is not None else None,
        "tick": float(tr.minimum_tick_size or 0) if tr is not None else None,
        "liq": float(m.metrics.liquidity_num or 0), "vol": float(m.metrics.volume_num or 0),
    }


def stage_markets():
    workers = int(ARGS[0]) if ARGS else 6
    d = OUT / "markets"
    d.mkdir(exist_ok=True)
    have = set()
    mp = d / "markets.jsonl.gz"
    if mp.exists():
        with gzip.open(mp, "rt") as f:
            for line in f:
                try:
                    have.add(json.loads(line)["cid"])
                except Exception:
                    pass
    want: dict[str, str] = {}
    for p in (OUT / "wallets").glob("*/*.json.gz"):
        try:
            with gzip.open(p, "rt") as f:
                w = json.load(f)
        except Exception:
            continue
        for k in ("closed", "redeemable", "open"):
            for r in w.get(k, []):
                if r["cid"] not in have:
                    want.setdefault(r["cid"], r.get("slug") or "")
        for fl in w.get("fills", []):
            if fl[2] not in have:
                want.setdefault(fl[2], fl[7])
    log(f"{len(have)} markets on file, {len(want)} to look up")
    items = list(want.items())
    out = gzip.open(mp, "at")
    t0 = time.time()

    def batch(chunk):
        rows = []
        got = {}
        for closed in (True, None):   # the condition-id listing returns open markets unless asked for closed ones
            ids = [x[0] for x in chunk if x[0] not in got]
            if not ids:
                break
            try:
                kw = {"closed": True} if closed else {}
                ms = call(lambda: c.list_markets(condition_ids=ids, page_size=len(ids), **kw).first_page().items, "gamma")
                got.update({str(m.condition_id): m for m in ms})
            except Exception as e:
                log("batch failed", str(e)[:80])
        for cid, slug in chunk:
            m = got.get(cid)
            if m is None and slug:
                try:
                    m = call(lambda: c.get_market(slug=slug), "gamma")
                except Exception:
                    m = None
            if m is not None and str(m.condition_id) == cid:
                try:
                    rows.append(market_row(m))
                except Exception as e:
                    log("row failed", slug, str(e)[:60])
            else:
                rows.append({"cid": cid, "slug": slug, "missing": True})
        return rows

    chunks = [items[i:i + 40] for i in range(0, len(items), 40)]
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(batch, chunks):
            with lock:
                for r in rows:
                    out.write(json.dumps(r) + "\n")
            n += 1
            if n % 100 == 0:
                out.flush()
                log(f"{n * 40}/{len(items)} markets, {time.time() - t0:.0f}s")
    out.close()
    log("markets done")


{"boards": stage_boards, "controls": stage_controls, "wallets": stage_wallets, "markets": stage_markets}[STAGE]()
