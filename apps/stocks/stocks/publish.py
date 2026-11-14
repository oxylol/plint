"""The page's JSON (shapes in PAGE.md), built from the books, the ledger and the reports."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ledger
from .bars import d2s
from .book import Record
from .cal import NY
from .metrics import YEAR, clean, max_drawdown, thin
from .models import AGENT, ORDER, ScoreModel
from .panel import Panel
from .store import Store

SUMMARY_KEYS = ("id", "title", "kind", "what", "state", "state_text", "since", "days", "equity", "ret", "index_ret",
                "vs_index", "day_ret", "day_vs_index", "max_dd", "n_pos", "cash", "trades", "costs", "turnover",
                "verdict", "spark", "spark_index", "backtest")
DETAIL_KEYS = ("curve", "positions", "pending", "trades", "today", "ranking_history", "signals", "memo", "reads",
               "report", "by_year")


def index_curve(panel: Panel, index_symbol: str, t0: int, start: float) -> np.ndarray:
    """The index from the close of row t0, starting at `start` dollars."""
    j = panel.ix[index_symbol]
    return start * np.r_[1.0, np.cumprod(1.0 + np.nan_to_num(panel.ret[t0 + 1:, j]))]


def report_block(store: Store, mid: str) -> tuple[dict | None, str | None, list | None]:
    """(the backtest figures, the report's text, the by-year table) as the research run left them."""
    rep = store.read_json(f"reports/{mid}.json")
    if not rep:
        return None, None, None
    text_path = Path(store.dir) / "reports" / f"{mid}.md"
    return rep.get("backtest"), (text_path.read_text(encoding="utf-8") if text_path.exists() else None), rep.get("by_year")


def model_json(meta: dict, rec: Record | None, panel: Panel | None, index_symbol: str, store: Store,
               names: dict[str, str], start_cash: float, state: str, state_text: str, horizon: int | None,
               scores_all: bool, pending: list | None = None, today_note: str = "") -> tuple[dict, dict]:
    """(summary, detail) for one model."""
    mid = meta["id"]
    backtest, report, years = report_block(store, mid)
    s = {"id": mid, "title": meta["title"], "kind": meta["kind"], "what": meta["what"], "state": state,
         "state_text": state_text, "since": None, "days": 0, "equity": None, "ret": None, "index_ret": None,
         "vs_index": None, "day_ret": None, "day_vs_index": None, "max_dd": None, "n_pos": 0, "cash": None,
         "trades": 0, "costs": 0.0, "turnover": None, "verdict": None, "spark": [], "spark_index": [],
         "backtest": backtest}
    d = {"curve": [], "positions": [], "pending": [], "trades": [], "today": None, "ranking_history": [],
         "signals": [], "memo": None, "reads": [], "report": report, "by_year": years}
    name = lambda sym: names.get(sym, sym)

    # the ledger: rankings scored, forecasts settled
    if scores_all and horizon:
        series = store.ic_series(mid, horizon)
        if series:
            s["verdict"] = ledger.verdict(np.array([ic for _, ic, _ in series]), horizon)
            d["ranking_history"] = [{"date": dt, "ic": round(ic, 4), "n": n} for dt, ic, n in series[-250:]]
        else:
            s["verdict"] = ledger.verdict(np.array([]), horizon)
    elif horizon and meta["kind"] != "index":
        sig = store.signals(mid, limit=200)
        d["signals"] = [{"date": x["date"], "symbol": x["symbol"], "horizon": x["horizon"], "p": x["p"],
                         "note": x["note"], "excess": x["excess"]} for x in sig]
        settled = store.signals(mid, settled=True)
        s["verdict"] = ledger.signal_verdict(settled, horizon) or ledger.verdict(np.array([]), horizon, what="evenings")

    if rec is not None and rec.days and panel is not None:
        eq = np.array(rec.equity)
        t0 = panel.t_of(rec.days[0])
        ix = index_curve(panel, index_symbol, t0, start_cash)[: len(eq)]
        book = rec.book
        years_run = max((len(eq) - 1) / YEAR, 1e-9)
        s.update({
            "since": d2s(rec.days[0]), "days": len(eq) - 1, "equity": round(float(eq[-1]), 2),
            "ret": float(eq[-1] / start_cash - 1.0), "index_ret": float(ix[-1] / start_cash - 1.0),
            "max_dd": max_drawdown(eq), "n_pos": len(book.pos), "cash": float(book.cash / eq[-1]) if eq[-1] > 0 else None,
            "trades": len(book.fills), "costs": round(float(book.costs), 2),
            "turnover": float(book.traded / eq.mean() / years_run) if len(eq) > 20 else None,
            "spark": thin(eq / start_cash), "spark_index": thin(ix / start_cash),
        })
        s["vs_index"] = s["ret"] - s["index_ret"]
        if len(eq) > 1:
            s["day_ret"] = float(eq[-1] / eq[-2] - 1.0)
            s["day_vs_index"] = s["day_ret"] - float(ix[-1] / ix[-2] - 1.0)
        d["curve"] = [[d2s(day), round(float(a), 2), round(float(b), 2)] for day, a, b in zip(rec.days, eq, ix)]
        d["positions"] = sorted(
            [{"symbol": sym, "name": name(sym), "weight": float(p.value / eq[-1]), "value": round(float(p.value), 2),
              "since": d2s(p.since), "days": p.days, "ret": float(p.value / p.basis - 1.0) if p.basis > 0 else None,
              "why": p.why} for sym, p in book.pos.items()], key=lambda x: -x["weight"])
        d["trades"] = [{"date": d2s(f.d), "symbol": f.sym, "name": name(f.sym), "side": f.side, "usd": round(f.usd, 2),
                        "price": round(f.price, 4) if np.isfinite(f.price) else None,
                        "shares": round(f.usd / f.price, 4) if np.isfinite(f.price) and f.price > 0 else None,
                        "cost": round(f.cost, 3), "why": f.why, "ret": f.ret} for f in reversed(book.fills[-300:])]
    if pending:
        now = {}
        if rec is not None and rec.book and rec.equity and rec.equity[-1] > 0:
            now = {sym: p.value / rec.equity[-1] for sym, p in rec.book.pos.items()}
        d["pending"] = [{"symbol": o.symbol, "name": name(o.symbol),
                         "side": "sell" if o.weight < now.get(o.symbol, 0.0) else "buy",
                         "weight": float(o.weight), "why": o.why} for o in pending]

    if scores_all:
        dates = store.score_dates(mid)
        if dates:
            sc = store.load_scores(mid, dates[-1]) or {}
            ranked = sorted(sc.items(), key=lambda kv: -kv[1])
            held = set(rec.book.pos) if rec is not None and rec.book else set()
            row = lambda i, sym, v: {"symbol": sym, "name": name(sym), "score": round(float(v), 4), "rank": i + 1, "held": sym in held}
            n = len(ranked)
            d["today"] = {"date": dates[-1], "note": today_note,
                          "top": [row(i, sym, v) for i, (sym, v) in enumerate(ranked[:12])],
                          "bottom": [row(n - 1 - i, sym, v) for i, (sym, v) in enumerate(reversed(ranked[-8:]))]}
    if meta["kind"] == "agent":
        memo = store.memo(mid)
        if memo:
            d["memo"] = {"date": memo[0], "text": memo[1]}
        d["reads"] = [{"date": r["date"], "symbol": r["symbol"], "name": name(r["symbol"]),
                       **{k: r.get(k) for k in ("thesis", "facts", "priced_in", "p5", "p20", "conviction", "risks", "sources")}}
                      for r in store.reads(limit=40)]
    return clean(s), clean({**s, **d})


def spend_block(store: Store, cap: float, stopped: str | None, now: datetime) -> dict:
    utc = now.astimezone(timezone.utc)               # the spend log keeps UTC days, so the meter does too
    month, day = utc.strftime("%Y-%m"), utc.strftime("%Y-%m-%d")
    sm = store.spend_summary(month)
    return {"month": month, "used": round(sm["used"], 2), "cap": cap, "today": round(store.spend_day(day), 2),
            "calls": sm["calls"], "stopped": stopped, "by_day": [[d, round(u, 2)] for d, u in sm["by_day"]]}


def write(store: Store, state: dict, details: dict[str, dict]) -> None:
    store.write_json("state.json", clean(state))
    for mid, d in details.items():
        store.write_json(f"model_{mid}.json", d)


def waiting_all(meta_list: list[dict], store: Store, text: str) -> tuple[list[dict], dict[str, dict]]:
    """Every model listed as waiting: the page on the first day."""
    summaries, details = [], {}
    for meta in meta_list:
        s, d = model_json(meta, None, None, "", store, {}, 0.0, "waiting", text, None, False)
        summaries.append(s)
        details[meta["id"]] = d
    return summaries, details


def metas(models: list) -> list[dict]:
    """The models in the page's order, the agent among them."""
    by_id = {m.id: {"id": m.id, "title": m.title, "kind": m.kind, "what": m.what} for m in models}
    by_id["agent"] = dict(AGENT)
    return [by_id[i] for i in ORDER if i in by_id]


def now_local() -> datetime:
    return datetime.now().astimezone()


def iso(t: datetime | None) -> str | None:
    return t.isoformat(timespec="seconds") if t else None


def ny(t: datetime) -> datetime:
    return t.astimezone(NY)
