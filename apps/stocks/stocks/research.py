"""Before anything trades: fit the learned models walk-forward, replay history, write a report per model.

Two stretches, kept apart:
    design   BACKTEST_START .. the day before EXAM_START    looked at as often as needed while building
    exam     EXAM_START .. the newest day                   run once, after the design is frozen
`python -m stocks research` does the design stretch; `--exam` the other. The count of exam runs is kept and
printed in every report, so a second look cannot pass as a first."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import backtest
from .bars import d2s
from .learn import dataset, fit
from .learn.net import NetCfg
from .ledger import nw_t, table_ic
from .metrics import clean
from .models import Env, ScoreModel, build
from .models.events import EventModel
from .panel import Panel
from .pipeline import Lab

log = logging.getLogger("stocks.research")
BACKTEST_START = 20190102
EXAM_START = 20250514
ABLATIONS = (("net_mlp", "mlp"), ("net_pool", "pool"))        # the network without its view of the whole day


def full_panel(lab: Lab) -> Panel:
    syms = sorted(set(lab.universe.ever()) | set(lab.instruments()))
    panel = Panel.load(lab.bars, syms, lab.cfg.index_symbol)
    panel.set_members(lab.universe)
    return panel


def _table(lab: Lab, panel: Panel, say) -> dataset.Table:
    out = lab.cfg.cache_dir / "table"
    try:
        t = dataset.Table.load(out)
        if len(t.days) == panel.T and int(t.days[-1]) == int(panel.days[-1]) and t.syms == panel.syms:
            return t
    except (FileNotFoundError, ValueError):
        pass
    say(f"building the training table: {panel.T} days, {panel.N} stocks")
    return dataset.build(panel, lab.cfg.index_symbol, out)


def _scores(lab: Lab, table: dataset.Table, name: str, algo: str, t0: int, t1: int, n_cols: int, say, **kw) -> np.ndarray:
    """A walk-forward score table, kept on disk so a crashed run does not start over."""
    path = lab.cfg.cache_dir / "scores" / f"{name}_{int(table.days[t0])}_{int(table.days[t1])}.npy"
    if path.exists():
        S = np.load(path)
        if S.shape == (table.T, n_cols):
            return S
    S = fit.walk_forward(table, algo, t0, t1, n_cols, log=say, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, S)
    return S


def _pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{digits}f} %"


def _num(x, digits: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _event_stats(model: EventModel, fwd_rel: np.ndarray, t0: int, t1: int) -> dict:
    """What followed each qualifying event over 20 days, against the universe: mean, t (by day), count."""
    by_day, n = [], 0
    for t in range(t0, t1 + 1):
        cols = np.flatnonzero(model.fire[t] > 0)
        vals = fwd_rel[t, cols]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            by_day.append(vals.mean())
            n += len(vals)
    mean, _se, tstat, days = nw_t(np.array(by_day), 19)
    return {"mean": mean, "t": tstat, "events": n, "days": days}


def _report(o: backtest.Outcome, days: np.ndarray, t0: int, t1: int, cfg, sens: dict, ladder: list, events: dict | None,
            variants: list[str], exam_runs: int, title: str = "backtest") -> str:
    m, s = o.model, o.stats
    lines = [f"# {m.title}: {title}", "", m.what, "",
             f"**{d2s(days[t0])} to {d2s(days[t1])}**, {s['days']} trading days, ${cfg.start_cash:,.0f} to start, costs included. "
             "Every decision used only what was known that evening and was filled at the next open.", "",
             "| | this model | the index |", "|---|---|---|",
             f"| a year | {_pct(s['ret_ann'])} | {_pct(s['index_ann'])} |",
             f"| in all | {_pct(s['ret'])} | {_pct(s['index_ret'])} |", "",
             f"- Against the index: **{_pct(s['vs_index_ann'])} a year** (90 % range {_pct(s['ci'][0])} to {_pct(s['ci'][1])}).",
             f"- Worst drop from a high: {_pct(s['max_dd'])}. Sharpe ratio {_num(s['sharpe'])}.",
             f"- It traded {_num(s['turnover'], 1)} times its book a year in {s['trades']} trades; costs took {_pct(s['cost_drag'], 2)} a year."]
    if sens:
        lines.append(f"- With costs doubled: {_pct(sens.get(2))} a year against the index; tripled: {_pct(sens.get(3))}. "
                     f"With a Swedish broker's currency fee on every trade ({cfg.fx_fee_bps / 100:.2f} %): {_pct(sens.get('fx'))}.")
    if o.ic:
        lines += ["", "## How its rankings held up", "",
                  "Each evening's ranking against what the stocks then did, over the universe's average. "
                  "IC is the rank correlation: 0 is no skill, 0.02 is good for daily data. t above 2 means it is unlikely to be luck.", "",
                  "| days ahead | IC | t | evenings | top tenth minus bottom tenth |", "|---|---|---|---|---|"]
        lines += [f"| {h} | {_num(v['ic'], 4)} | {_num(v['t'], 1)} | {v['n']} | {_pct(v['spread'], 2)} |" for h, v in sorted(o.ic.items())]
    if events:
        lines += ["", "## What followed its events", "",
                  f"{events['events']} qualifying events on {events['days']} evenings. Over the next 20 days they did "
                  f"**{_pct(events['mean'], 2)}** against the universe's average (t {_num(events['t'], 1)}), before costs."]
    if ladder and m.id in ("net", "trees", "ridge", "blend"):
        lines += ["", "## The ladder", "",
                  "The same inputs through plainer and fancier models (5 days ahead). Each step up has to earn its place.", "",
                  "| model | IC | t |", "|---|---|---|"]
        lines += [f"| {name} | {_num(ic, 4)} | {_num(t, 1)} |" for name, ic, t in ladder]
    if o.years:
        lines += ["", "## Year by year", "", "| year | this model | the index |", "|---|---|---|"]
        lines += [f"| {y['year']} | {_pct(y['ret'])} | {_pct(y['index'])} |" for y in o.years]
    lines += ["", "## What this does not show", "",
              "- Free data has little on companies that left the market. A stock that went to zero and vanished is missing, "
              "so every stock picker's history is flattered a little; the index's is not.",
              "- Fills are at the first price of the day on the consolidated tape. A real order at the open gets the auction's price, which is near it, not equal.",
              "- The backtest refits the learned models twice a year; live they are refitted every week.",
              f"- The design was changed {len(variants)} time(s) after first seeing these figures"
              + (": " + "; ".join(variants) if variants else "") + ".",
              f"- The exam stretch ({d2s(EXAM_START)} onward) has been run {exam_runs} time(s)."]
    return "\n".join(lines) + "\n"


LEARNED = {"ridge", "trees", "net", "blend"}


def run(lab: Lab, exam: bool = False, say=None, months: int = 6, only: tuple[str, ...] = ()) -> dict:
    """The design stretch (default) or the exam. Writes reports/<id>.json and .md; returns a short summary.
    `only` limits it to some models (the index always comes along, as the bar); without a learned model
    among them nothing is fitted, which turns hours into a minute."""
    cfg, store = lab.cfg, lab.store
    say = say or (lambda text: (log.info(text), lab.doing(text)))
    panel = full_panel(lab)
    table = _table(lab, panel, say)
    days = panel.days
    first_fit = dataset.WARMUP + fit.MIN_TRAIN_DAYS + 2 * (dataset.H_MAX + 2) + fit.VAL_DAYS
    t_start = max(int(np.searchsorted(days, BACKTEST_START)), first_fit)
    t_exam = int(np.searchsorted(days, EXAM_START))
    if exam:
        t0, t1 = t_exam, panel.T - 1
        runs = (store.get("exam_runs") or 0) + 1
        store.put("exam_runs", runs)
        say(f"exam run number {runs}: {d2s(days[t0])} to {d2s(days[t1])}")
    else:
        t0, t1 = t_start, min(t_exam, panel.T) - 1
    if t1 - t0 < 60:
        raise RuntimeError(f"only {t1 - t0} days between {d2s(days[t0])} and {d2s(days[min(t1, panel.T - 1)])}: not enough history")

    env = Env(index=cfg.index_symbol, cash=cfg.cash_symbol, names=lab.assets.names(),
              events=None if lab.events.empty() else lab.events)
    quick = {"seeds": 1, "months": 12} if cfg.fast else {"months": months}
    fitted = not only or bool(LEARNED & set(only))
    for algo in ("ridge", "trees", "net") if fitted else ():
        say(f"walk-forward fits: {algo}")
        kw = {**quick, "cfg": NetCfg(epochs=3, patience=2)} if cfg.fast and algo == "net" else quick
        env.scores[algo] = _scores(lab, table, algo, algo, t0, t1, panel.N, say, **kw)
    ladder = []
    fwd5 = dataset.relative(dataset.forward(panel, 5), panel.member)
    if fitted and not exam and not cfg.fast:
        for name, mode in ABLATIONS:
            say(f"walk-forward fits: the network as '{mode}'")
            S = _scores(lab, table, name, "net", t0, t1, panel.N, say, months=12, seeds=1, cfg=NetCfg(mode=mode))
            ic, _, t, _ = nw_t(table_ic(S, fwd5, panel.member, t0, t1 + 1), 4)
            ladder.append((f"network, {'each stock alone' if mode == 'mlp' else 'with the day average'}", ic, t))

    say("replaying history")
    models = [m for m in build(cfg.models) if (not only or m.id in only or m.id == "index") and (fitted or m.id not in LEARNED)]
    out = backtest.run(panel, models, env, t0, t1, cfg.start_cash, cfg.slip_bps, cfg.impact_coef)
    backtest.score(panel, out, "index", cfg.cash_symbol, t0, t1)
    sens: dict[str, dict] = {mid: {} for mid in out}
    for key, kw in ((2, {"cost_scale": 2.0}), (3, {"cost_scale": 3.0}), ("fx", {"extra_bps": cfg.fx_fee_bps})):
        alt = backtest.run(panel, models, env, t0, t1, cfg.start_cash, cfg.slip_bps, cfg.impact_coef, prepare=False, **kw)
        ix = np.array(alt["index"].record.equity)
        years = max((len(ix) - 1) / 252.0, 1e-9)
        for mid, o in alt.items():
            e = np.array(o.record.equity)
            sens[mid][key] = float((e[-1] / e[0]) ** (1 / years) - (ix[-1] / ix[0]) ** (1 / years))
    for name in ("ridge", "trees", "net", "blend"):
        if name in out and 5 in out[name].ic:
            label = {"ridge": "straight line", "trees": "boosted trees", "net": "network, reading the whole day", "blend": "blend"}[name]
            ladder.append((label, out[name].ic[5]["ic"], out[name].ic[5]["t"]))
    fwd20 = dataset.relative(dataset.forward(panel, 20), panel.member)
    variants = store.get("variants") or []
    exam_runs = store.get("exam_runs") or 0

    summary = []
    for mid, o in out.items():
        m, s = o.model, o.stats
        waiting = m.waiting() if not isinstance(m, ScoreModel) else ""
        if waiting:                                 # an event model without its filings has nothing to report
            say(f"{mid}: no report ({waiting})")
            continue
        ev = _event_stats(m, fwd20, t0, t1) if isinstance(m, EventModel) else None
        main = o.ic.get(m.horizon if m.horizon in o.ic else 5, {}) if o.ic else {}
        block = {"from": d2s(days[t0]), "to": d2s(days[t1]), "ret_ann": s["ret_ann"], "index_ann": s["index_ann"],
                 "vs_index_ann": s["vs_index_ann"], "ci": s["ci"], "sharpe": s["sharpe"], "max_dd": s["max_dd"],
                 "turnover": s["turnover"], "cost_drag": s["cost_drag"],
                 "ic": main.get("ic") if main else (ev["mean"] if ev else None),
                 "ic_t": main.get("t") if main else (ev["t"] if ev else None)}
        text = _report(o, days, t0, t1, cfg, sens[mid], ladder, ev, variants, exam_runs, "the exam" if exam else "backtest")
        rep = store.read_json(f"reports/{mid}.json") or {}
        path = Path(store.dir) / "reports" / f"{mid}.md"
        if exam:
            if not rep.get("backtest"):
                say(f"{mid}: the exam was run before the design backtest; its result is kept on its own")
                rep["backtest"] = None
            else:
                rep["backtest"]["exam"] = {k: block[k] for k in ("from", "to", "ret_ann", "index_ann", "vs_index_ann", "ci", "ic", "ic_t")}
            rep["exam_at"] = datetime.now(timezone.utc).isoformat()
            old = path.read_text(encoding="utf-8") if path.exists() else ""
            marker = "\n---\n\n# "
            text = (old.split(marker)[0].rstrip() + "\n\n---\n\n" + text) if old else text
        else:
            rep = {"backtest": {**block, "exam": (rep.get("backtest") or {}).get("exam")}, "by_year": o.years,
                   "made_at": datetime.now(timezone.utc).isoformat()}
        store.write_json(f"reports/{mid}.json", clean(rep))
        path.write_text(text, encoding="utf-8")
        summary.append({"id": mid, "vs_index_ann": s["vs_index_ann"], "ci": s["ci"], "ic": block["ic"], "ic_t": block["ic_t"]})
        say(f"{mid}: {_pct(s['vs_index_ann'])} a year against the index, IC {_num(block['ic'], 4)} (t {_num(block['ic_t'], 1)})")
    if not only:
        store.write_json("reports/summary.json", clean({"exam": exam, "from": d2s(days[t0]), "to": d2s(days[t1]), "models": summary}))
    return {"from": d2s(days[t0]), "to": d2s(days[t1]), "models": summary}


def fit_live(lab: Lab, say=None, algos: tuple[str, ...] = ("ridge", "trees", "net")) -> dict:
    """Fit the learned models on everything known now and keep the fits for the evening runs."""
    say = say or (lambda text: (log.info(text), lab.doing(text)))
    panel = full_panel(lab)
    table = _table(lab, panel, say)
    day = int(panel.days[-1])
    out = {}
    for algo in algos:
        say(f"fitting {algo} on prices up to {d2s(day)}")
        kw = {"cfg": NetCfg(epochs=3, patience=2)} if lab.cfg.fast and algo == "net" else {}
        fits = fit.fit_at(table, algo, table.T, 1 if lab.cfg.fast else fit.SEEDS_LIVE[algo], **kw)
        lab.fits.save(algo, day, fits)
        out[algo] = float(np.mean([f.get("val_ic", np.nan) for f in fits]))
        lab.say(f"{algo} fitted on prices up to {d2s(day)} ({len(fits)} seeds, validation IC {out[algo]:+.4f})")
    return out
