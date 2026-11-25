"""The copy study (`python -m stocks copy`): read the House reports, build every person's copy book, rank
them, make the half-yearly picks, replay the two combos, write the reports and the page's block.

    reports/copy.json, reports/copy.md   the fair combo's backtest, like every model's
    copy/study.json                      the page's block: the people, the picks, both combos
    copy/picks.json                      the picks the live `copy` model follows
    <cache>/copytrade/people.parquet, trades.parquet (with `copyable`, the evening check, filled in)

Every person's book runs from early 2016 to the newest close: the picks need the three years before each
half year. The figures shown are the design stretch's (research.BACKTEST_START to the day before
EXAM_START); the exam stretch is held back as for every model. Rules: DESIGN.md, "Copy trades"."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import backtest, cal as calmod
from ..bars import d2s
from ..book import Book, Costs
from ..metrics import YEAR, by_year, clean, summarize
from ..models.base import Env
from ..models.rules import Index
from ..panel import Panel
from ..research import BACKTEST_START, EXAM_START, _num, _pct, _report
from . import trades as tt
from .house import House, Stopped
from .model import CopyBook, Period
from .plan import Plan, liquid, make

log = logging.getLogger("stocks.copytrade.study")
TOP = 10                    # people in a combo
LOOKBACK = 756              # trading days a pick looks back over (three years)
MIN_LOOK = 252              # and at least this many
MIN_OPENED = 10             # positions opened in that span to be picked, or ranked (an insider: purchases copied)
ACTIVE = 252                # a copied trade made public within this many trading days, to be picked
WARM = 25                   # rows at the panel's start before a book opens (the cost model reads 21 days back)


# ── the data ────────────────────────────────────────────────────────────────────────────────────
def read_house(lab, say) -> dict:
    """The House reports not read yet, from the first year of price history to this one."""
    house = House(lab.cfg.cache_dir)
    years = list(range(int(lab.cfg.history_start[:4]), datetime.now(timezone.utc).year + 1))
    try:
        got = house.sync(years, say)
    except Stopped as e:
        lab.say(f"House reports: the run was stopped, and what it had read is kept ({e})", "warn")
        got = {"stopped": str(e)}
    return got


def study_panel(lab, symbols: set[str]) -> Panel:
    have = lab.bars.symbols()
    syms = sorted((set(lab.universe.ever()) | set(lab.instruments()) | symbols) & (have | set(lab.instruments())))
    panel = Panel.load(lab.bars, syms, lab.cfg.index_symbol)
    panel.set_members(lab.universe)
    return panel


def flags(trades: pd.DataFrame, panel: Panel, ok) -> pd.Series:
    """The evening check (a price, $1 million a day) for every purchase the panel reaches; None elsewhere."""
    cal = panel.days.astype(np.int64)
    out = []
    for d, s, side in zip(trades["day"].to_numpy(dtype=np.int64), trades["sym"], trades["side"]):
        i = int(np.searchsorted(cal, d))
        out.append(ok(int(cal[i]), s) if side == "buy" and i < len(cal) else None)
    return pd.Series(out, index=trades.index, dtype=object)


def record(plan: Plan, lo: int, hi: int) -> int:
    """What counts toward the minimum, lo < day <= hi: positions opened; for an insider, who trades one company
    and so can open only one position at a time, the purchases copied."""
    return plan.count("copied" if plan.pid.startswith("in:") else "opened", lo, hi)


def plans_for(trades: pd.DataFrame, cal: np.ndarray, ok) -> dict[str, Plan]:
    out = {}
    for pid, g in tt.by_person(trades).items():
        if (g["side"] == "buy").any():
            out[pid] = make(pid, g, cal, ok, pid.startswith("in:"))
    return out


# ── every person's book ─────────────────────────────────────────────────────────────────────────
def run_book(panel: Panel, model, env: Env, t0: int, costs: Costs, start_cash: float, bought_out: frozenset,
             renamed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One book from the close of row t0 to the newest: equity, traded and costs per row (rows before t0
    are NaN). The same Book and decide() as a backtest; only the record is lighter, so a thousand fit."""
    book = Book(start_cash, bought_out=bought_out, renamed=dict(renamed))
    T = panel.T
    eq = np.full(T, np.nan)
    traded = np.zeros(T, dtype=np.float32)
    cost = np.zeros(T, dtype=np.float32)
    eq[t0] = start_cash
    change = getattr(model, "change", None)
    pending = model.decide(t0, book.view()).orders
    for t in range(t0 + 1, T):
        fills = book.step(panel, t, pending, costs) if pending or book.pos else []
        if fills:
            traded[t] = sum(f.usd for f in fills)
            cost[t] = sum(f.cost for f in fills)
        eq[t] = book.equity()
        pending = model.decide(t, book.view()).orders if change is None or t in change or not book.pos else []
    return eq, traded, cost


def window_score(e: np.ndarray, ix: np.ndarray, a: int, b: int) -> float:
    """What the book made against the index over rows a..b, a year."""
    if b - a < 1 or not (np.isfinite(e[a]) and np.isfinite(e[b])) or e[a] <= 0:
        return float("nan")
    rel = (e[b] / e[a]) / (ix[b] / ix[a])
    return float(rel ** (YEAR / (b - a)) - 1.0)


def half_years(first: int, next_day: int) -> list[int]:
    """Every 1 January and 1 July from `first` up to the half year the next trading day falls in."""
    out = []
    y, m = first // 10000, 1 if first % 10000 < 701 else 7
    while True:
        h = y * 10000 + m * 100 + 1
        if h > next_day:
            break
        out.append(h)
        y, m = (y, 7) if m == 1 else (y + 1, 1)
    return out


def picks(panel: Panel, plans: dict[str, Plan], books: dict[str, np.ndarray], ix: np.ndarray, t_open: int,
          first_half: int, next_day: int) -> list[dict]:
    """The fair combo's picks: for each half year, the TOP people by their book against the index over the
    LOOKBACK rows before it, among those who opened MIN_OPENED positions in them and had a copied trade made
    public in the last ACTIVE rows. Picked from records up to the close before the half year starts."""
    cal = panel.days.astype(np.int64)
    out = []
    for h in half_years(first_half, next_day):
        t_h = int(np.searchsorted(cal, h))
        b = t_h - 1
        a = max(t_open, b - LOOKBACK)
        if b - a < MIN_LOOK:
            continue
        scored = []
        for pid, plan in plans.items():
            if record(plan, int(cal[a]), int(cal[b])) < MIN_OPENED:
                continue
            if plan.count("copied", int(cal[max(a, b - ACTIVE)]), int(cal[b])) < 1:
                continue
            s = window_score(books[pid], ix, a, b)
            if np.isfinite(s):
                scored.append((s, pid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:TOP]
        start = int(cal[t_h]) if t_h < len(cal) else next_day
        out.append({"start": start, "half": h, "from": d2s(int(cal[a])), "to": d2s(int(cal[b])),
                    "people": [pid for _, pid in top], "scores": [s for s, _ in top], "eligible": len(scored)})
    return out


# ── the study ───────────────────────────────────────────────────────────────────────────────────
def run(lab, say=None) -> dict:
    cfg, store = lab.cfg, lab.store
    say = say or (lambda text: (log.info(text), lab.doing(text)))
    say("reading the House trade reports (all of them the first time: an hour or two)")
    got = read_house(lab, say)
    if got.get("reports"):
        lab.say(f"House trade reports: {got['reports']} read, {got.get('trades', 0)} trades in them, "
                f"{got.get('paper', 0)} filed on paper (scans, not read)")
    house = House(cfg.cache_dir)
    people, trades, counts = tt.build(house, lab.events.insiders, lab.assets.aliases())
    if trades.empty:
        raise RuntimeError("no trades to copy: neither House reports nor insider purchases are in")
    say(f"building the copy books: {counts['house_people']} House members, {counts['insider_people']} insiders")
    panel = study_panel(lab, set(trades["sym"]))
    costs = Costs(panel, cfg.slip_bps, cfg.impact_coef)
    ok = liquid(panel, costs.adv)
    trades["copyable"] = flags(trades, panel, ok)
    folder = Path(cfg.cache_dir) / "copytrade"
    tt.save(folder, people, trades)
    known = trades[trades["copyable"].notna()]
    cal_days = panel.days.astype(np.int64)
    pd.DataFrame({"day": cal_days[np.searchsorted(cal_days, known["day"].to_numpy(dtype=np.int64))], "sym": known["sym"],
                  "ok": known["copyable"].astype(bool)}).drop_duplicates(["day", "sym"]).to_parquet(folder / "flags.parquet", index=False)
    np.save(folder / "calendar.npy", cal_days)
    cal = panel.days.astype(np.int64)
    plans = plans_for(trades, cal, ok)
    names = dict(zip(people["pid"], people["name"]))
    env = Env(index=cfg.index_symbol, cash=cfg.cash_symbol, names=lab.assets.names())
    gone, renamed = frozenset(lab.assets.gone()), lab.assets.aliases()
    t_open = WARM
    index = Index()
    index.prepare(panel, env)
    ix, _, _ = run_book(panel, index, env, t_open, costs, cfg.start_cash, gone, renamed)

    books: dict[str, np.ndarray] = {}
    flows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for n, (pid, plan) in enumerate(sorted(plans.items())):
        if not len(plan.opened) or (pid.startswith("in:") and len(plan.copied) < MIN_OPENED):
            continue                                # an insider who never opened enough can be neither ranked nor picked
        m = CopyBook(pid, names.get(pid, pid), [Period(int(cal[t_open]), [plan], names)])
        m.prepare(panel, env)
        e, tr, co = run_book(panel, m, env, t_open, costs, cfg.start_cash, gone, renamed)
        books[pid], flows[pid] = e, (tr, co)
        if n % 100 == 0:
            say(f"copy books: {n} of {len(plans)}")

    t0 = int(np.searchsorted(cal, BACKTEST_START))
    t_exam = int(np.searchsorted(cal, EXAM_START))
    t1 = min(t_exam, panel.T) - 1
    lo_day, hi_day = int(cal[t0 - 1]), int(cal[t1])

    # the ranking: every person's book over the design stretch
    rows = []
    pool = dict(zip(people["pid"], people["pool"]))
    label = dict(zip(people["pid"], people["label"]))
    by_pid = tt.by_person(trades)
    for pid, e in books.items():
        plan = plans[pid]
        g = by_pid[pid]
        inside = g[(g["day"] > lo_day) & (g["day"] <= hi_day)]
        if inside.empty and plan.count("opened", lo_day, hi_day) == 0:
            continue
        if pool.get(pid) == "insider" and record(plan, lo_day, hi_day) < MIN_OPENED:
            continue                                # listed: every House member, and the insiders who are ranked
        tr, co = flows[pid]
        s = summarize(e[t0:t1 + 1], ix[t0:t1 + 1], tr[t0:t1 + 1], co[t0:t1 + 1], ci=False)
        dated = inside[inside["traded"] > 0]           # insiders' purchases carry no trade date of their own here
        lag = dated["filed"].map(_ordinal) - dated["traded"].map(_ordinal)
        opened = record(plan, lo_day, hi_day)
        rows.append({"pid": pid, "name": names.get(pid, pid), "pool": pool.get(pid, ""), "label": label.get(pid, ""),
                     "ret_ann": s["ret_ann"], "vs_index_ann": s["vs_index_ann"], "max_dd": s["max_dd"],
                     "turnover": s["turnover"], "opened": opened, "copied": plan.count("copied", lo_day, hi_day),
                     "trades": int(len(inside)), "first": d2s(int(inside["day"].min())) if len(inside) else None,
                     "last": d2s(int(inside["day"].max())) if len(inside) else None,
                     "lag_days": float(lag.median()) if len(lag) else None, "ranked": opened >= MIN_OPENED})
    rows.sort(key=lambda r: (not r["ranked"], -_or(r["vs_index_ann"], -9.0)))
    for i, r in enumerate(x for x in rows if x["ranked"]):
        r["rank"] = i + 1

    # the picks, the fair combo, the hindsight combo
    next_day = _next_trading_day(int(cal[-1]))
    made = picks(panel, plans, books, ix, t_open, BACKTEST_START, next_day)
    store.write_json("copy/picks.json", clean({"made_at": datetime.now(timezone.utc).isoformat(), "to": d2s(int(cal[-1])),
                                                 "picks": made}))
    fair_periods = [Period(p["start"], [plans[pid] for pid in p["people"]], names, 1.0 / max(len(p["people"]), 1))
                    for p in made if p["start"] <= hi_day]
    best = [r for r in rows if r["ranked"]][:TOP]
    hind_periods = [Period(int(cal[t0]), [plans[r["pid"]] for r in best], names, 1.0 / max(len(best), 1))]
    from .model import Copy
    fair = CopyBook("copy", Copy.title, fair_periods, Copy.what)
    hind = CopyBook("copy_hindsight", "Hindsight: the best ten", hind_periods,
                    "The ten people whose copy books did best over the whole stretch, picked after seeing it, followed from its first day.")
    say("replaying the two combos")
    trio = [Index(), fair, hind]
    out = backtest.run(panel, trio, env, t0, t1, cfg.start_cash, cfg.slip_bps, cfg.impact_coef, bought_out=gone)
    backtest.score(panel, out, "index", cfg.cash_symbol, t0, t1)
    sens = {mid: {} for mid in out}
    for key, kw in ((2, {"cost_scale": 2.0}), (3, {"cost_scale": 3.0}), ("fx", {"extra_bps": cfg.fx_fee_bps})):
        alt = backtest.run(panel, trio, env, t0, t1, cfg.start_cash, cfg.slip_bps, cfg.impact_coef,
                           prepare=False, bought_out=gone, **kw)
        ie = np.array(alt["index"].record.equity)
        years = max((len(ie) - 1) / YEAR, 1e-9)
        for mid, o in alt.items():
            e = np.array(o.record.equity)
            sens[mid][key] = float((e[-1] / e[0]) ** (1 / years) - (ie[-1] / ie[0]) ** (1 / years))
    each_day = np.nanmean(np.where(panel.member, panel.ret, np.nan), axis=1)[t0 + 1:t1 + 1]
    basket = float(np.prod(1.0 + np.nan_to_num(each_day)) ** (YEAR / max(len(each_day), 1)) - 1.0)
    variants, exam_runs = store.get("variants") or [], store.get("exam_runs") or 0

    def block(o):
        s = o.stats
        return {"from": d2s(int(cal[t0])), "to": d2s(int(cal[t1])), "ret_ann": s["ret_ann"], "index_ann": s["index_ann"],
                "vs_index_ann": s["vs_index_ann"], "ci": s["ci"], "sharpe": s["sharpe"], "max_dd": s["max_dd"],
                "turnover": s["turnover"], "cost_drag": s["cost_drag"], "ic": None, "ic_t": None}

    combos = {}
    for mid in ("copy", "copy_hindsight"):
        o = out[mid]
        combos[mid] = {**block(o), "basket": basket, "vs_basket": o.stats["ret_ann"] - basket, "costs_x2": sens[mid].get(2),
                       "costs_x3": sens[mid].get(3), "fx": sens[mid].get("fx"), "by_year": o.years,
                       "curve": [[d2s(int(d)), float(a), float(b)] for d, a, b in
                                 zip(o.record.days[::5], o.record.equity[::5], out["index"].record.equity[::5])]}
    text = _report(out["copy"], panel.days, t0, t1, cfg, sens["copy"], [], None, variants, exam_runs, "backtest", basket)
    text += copy_section(rows, made, combos, names, counts, got, hi_day)
    rep = store.read_json("reports/copy.json") or {}
    store.write_json("reports/copy.json", clean({"backtest": {**block(out["copy"]), "exam": (rep.get("backtest") or {}).get("exam")},
                                                 "by_year": out["copy"].years, "made_at": datetime.now(timezone.utc).isoformat()}))
    (Path(store.dir) / "reports").mkdir(parents=True, exist_ok=True)
    (Path(store.dir) / "reports" / "copy.md").write_text(text, encoding="utf-8")

    now_pick = made[-1] if made else None
    study = {
        "made_at": datetime.now(timezone.utc).isoformat(), "from": d2s(int(cal[t0])), "to": d2s(int(cal[t1])),
        "counts": {**counts, "people_with_books": len(books), "ranked": sum(r["ranked"] for r in rows), "listed": len(rows),
                   "insiders_with_trades": int(trades.loc[trades["pid"].str.startswith("in:") & (trades["day"] > lo_day)
                                                          & (trades["day"] <= hi_day), "pid"].nunique()),
                   "house_listed": sum(r["pool"] == "house" for r in rows), "insider_listed": sum(r["pool"] == "insider" for r in rows),
                   "house_ranked": sum(r["pool"] == "house" and r["ranked"] for r in rows),
                   "paper_reports": got.get("paper")},
        "rules": {"top": TOP, "lookback_years": LOOKBACK / YEAR, "min_opened": MIN_OPENED},
        "people": rows,
        "now": None if now_pick is None else {"start": d2s(now_pick["start"]), "from": now_pick["from"], "to": now_pick["to"],
                                              "eligible": now_pick["eligible"],
                                              "people": [{"pid": pid, "name": names.get(pid, pid), "pool": pool.get(pid, ""),
                                                          "label": label.get(pid, ""), "score": sc}
                                                         for pid, sc in zip(now_pick["people"], now_pick["scores"])]},
        "picks": [{"start": d2s(p["start"]), "people": [names.get(pid, pid) for pid in p["people"]], "eligible": p["eligible"]}
                  for p in made],
        "fair": combos["copy"], "hindsight": {**combos["copy_hindsight"], "people": [
            {"pid": r["pid"], "name": r["name"], "pool": r["pool"], "label": r["label"], "vs_index_ann": r["vs_index_ann"]} for r in best]},
    }
    store.write_json("copy/study.json", clean(study))
    line = (f"copy study: {len(rows)} people ({study['counts']['house_listed']} House members), "
            f"fair combo {_pct(combos['copy']['vs_index_ann'])} a year against the index, "
            f"hindsight {_pct(combos['copy_hindsight']['vs_index_ann'])}")
    lab.say(line)
    return {"people": len(rows), "fair": combos["copy"]["vs_index_ann"], "hindsight": combos["copy_hindsight"]["vs_index_ann"],
            "picks": len(made)}


def _or(x, default: float) -> float:
    return float(x) if x is not None and np.isfinite(x) else default


def _ordinal(d: int) -> int:
    d = int(d)
    return date(d // 10000, d // 100 % 100, d % 100).toordinal()


def _next_trading_day(d: int) -> int:
    n = calmod.next_trading_day(date(d // 10000, d // 100 % 100, d % 100))
    return n.year * 10000 + n.month * 100 + n.day


def copy_section(rows: list[dict], made: list[dict], combos: dict, names: dict, counts: dict, got: dict, hi_day: int) -> str:
    ranked = [r for r in rows if r["ranked"]]
    house = [r for r in rows if r["pool"] == "house"]
    f, h = combos["copy"], combos["copy_hindsight"]
    lines = ["", "## Copy trades: what was looked at", "",
             f"- {len(rows)} people listed: {len(house)} House members with a copied trade in the stretch, and the "
             f"{len(rows) - len(house)} company insiders (one person at one company) who bought often enough to be ranked. "
             f"{len(ranked)} opened at least {MIN_OPENED} positions and are ranked.",
             f"- House reports: {counts.get('rows', 0)} trades read from the electronic reports; {counts.get('copyable', 0)} of them "
             f"could be copied (a stock, a fund or a call option with a ticker), {counts.get('repeats', 0)} were repeats in "
             "amended reports. Reports filed on paper are scans and were not read.",
             "", "## The two combos", "",
             "| | a year | against the index | against a blind basket | worst drop |", "|---|---|---|---|---|",
             f"| fair: re-picked every half year from the three years before | {_pct(f['ret_ann'])} | **{_pct(f['vs_index_ann'])}** | "
             f"{_pct(f['vs_basket'])} | {_pct(f['max_dd'])} |",
             f"| hindsight: the ten best over the whole stretch | {_pct(h['ret_ann'])} | **{_pct(h['vs_index_ann'])}** | "
             f"{_pct(h['vs_basket'])} | {_pct(h['max_dd'])} |", "",
             "The hindsight combo was picked after seeing the whole stretch, so it is bound to look good: picking the best ten of "
             f"{len(ranked)} afterwards always finds some. The fair one is what could have been done at the time.", "",
             "## Who the fair combo followed", "", "| from | people | could be picked |", "|---|---|---|"]
    for p in made:
        lines.append(f"| {d2s(p['start'])} | {', '.join(names.get(pid, pid) for pid in p['people']) or 'nobody qualified'} | {p['eligible']} |")
    lines += ["", "## The best and the worst to copy (design stretch, a year against the index)", "",
              "| | who | | against the index | positions |", "|---|---|---|---|---|"]
    show = ranked[:15] + ranked[-10:] if len(ranked) > 25 else ranked
    for r in show:
        lines.append(f"| {r.get('rank', '')} | {r['name']} | {r['label']} | {_pct(r['vs_index_ann'])} | {r['opened']} |")
    lag = [r["lag_days"] for r in house if r.get("lag_days") is not None]
    lines += ["", "## What this does not show", "",
              f"- A House member's trade reaches a copy weeks after it was made (the median member's median: "
              f"{_num(float(np.median(lag)) if lag else float('nan'), 0)} days from the trade to the report). The copy buys then, not when they did.",
              "- Amounts are ranges; a copy's sizes follow the middle of each range.",
              "- Senators, Trump and funds' quarterly holdings are not in (DESIGN.md says why).",
              "- An insider's purchase is held six months; their sales are not copied."]
    return "\n".join(lines) + "\n"


