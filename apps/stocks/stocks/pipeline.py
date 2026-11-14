"""The evening run: prices in, books marked, tomorrow decided, the page written.

One function, `evening`, that can be run again at any time: a model that has already decided for a day
does not decide twice (the first decision stands), prices are pulled again for the last two weeks and
corrections logged, and the books are replayed from the decisions each time."""
from __future__ import annotations

import logging
import traceback
from datetime import date, datetime, timedelta, timezone

import numpy as np

from . import cal, ingest, ledger, publish
from .alpaca import Alpaca, AlpacaError
from .bars import BarStore, d2s, s2d
from .book import Costs, Record, replay
from .config import Config
from .events import Events
from .learn.dataset import forward, relative
from .learn.fit import Fits
from .models import AGENT, Env, ScoreModel, build
from .notify import Notifier
from .panel import Panel
from .store import Store
from .types import BookView, Decision, Order, Signal
from .universe import Assets, Universe, month_of

log = logging.getLogger("stocks")
HORIZONS = (1, 5, 20)
NEEDS_NO_REPORT = {"index", "agent"}         # the index has nothing to prove; the agent cannot be backtested


class Lab:
    """Everything the run needs, opened once."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.data_dir)
        self.bars = BarStore(cfg.cache_dir)
        self.assets = Assets(cfg.cache_dir)
        self.universe = Universe(cfg.cache_dir)
        self.events = Events(cfg.cache_dir)
        self.fits = Fits(cfg.data_dir)
        self.api = Alpaca(cfg.alpaca_key, cfg.alpaca_secret) if cfg.has_market_data else None
        self.notifier = Notifier(cfg.ntfy_url if cfg.notify != "off" else "", cfg.ntfy_token)
        self.sec = None
        if cfg.has_sec:
            from .sec import Sec
            self.sec = Sec(cfg.sec_contact)
        self.busy: str | None = None
        self.agent_stopped: str | None = None

    def say(self, text: str, level: str = "info") -> None:
        (log.warning if level != "info" else log.info)(text)
        self.store.log(text, level)

    def doing(self, what: str | None) -> None:
        self.busy = what
        st = self.store.read_json("state.json")
        if st:                                     # let the page see what it is busy with before the run ends
            st.setdefault("clock", {})["busy"] = what
            self.store.write_json("state.json", st)

    # ── which symbols, which days ───────────────────────────────────────────────────────────────
    def instruments(self) -> list[str]:
        return [self.cfg.index_symbol, self.cfg.cash_symbol]

    def thin(self, syms: list[str]) -> list[str]:
        """All of them, or (a trial run) an even sample plus the two instruments."""
        n = self.cfg.symbol_limit
        if not n or len(syms) <= n:
            return syms
        return sorted(set(syms[:: max(1, len(syms) // n)]) | set(self.instruments()))

    def have_history(self) -> bool:
        """The long history is in (the flag is set when the backfill has finished, not while it runs)."""
        return bool(self.store.get("history_done")) and bool(self.bars.years())

    def model_ids(self) -> list[str]:
        ids = [m.id for m in build(self.cfg.models)]
        if not self.cfg.models or "agent" in self.cfg.models:
            ids.append("agent")
        return ids

    def live_panel(self, today: int) -> Panel:
        """The days the books need: from a good year before the oldest record to today."""
        sinces = [s2d(s) for s in (self.store.since(m) for m in self.model_ids()) if s]
        first = min(sinces + [today])
        y, m, d = first // 10000, first // 100 % 100, first % 100
        start = date(y, m, d) - timedelta(days=500)
        syms: set[str] = set(self.instruments())
        month = month_of(first)
        while month <= month_of(today):
            syms.update(self.universe.members(month * 100 + 1))
            month = month + 1 if month % 100 < 12 else (month // 100 + 1) * 100 + 1
        for mid in self.model_ids():
            for dec in self.store.decisions(mid, self.store.since(mid)):
                syms.update(o.symbol for o in dec.orders)
        panel = Panel.load(self.bars, sorted(syms), self.cfg.index_symbol, start.year * 10000 + start.month * 100 + start.day, today)
        panel.set_members(self.universe)
        return panel


# ── steps ────────────────────────────────────────────────────────────────────────────────────────
def refresh_assets(lab: Lab, force: bool = False) -> None:
    last = lab.store.get("assets_at")
    if not force and last and (datetime.now(timezone.utc) - datetime.fromisoformat(last)).days < 7 and not lab.assets.frame().empty:
        return
    lab.doing("reading the list of stocks")
    lab.assets.save(lab.api.assets())
    try:                                             # the companies bought out since the last look (all of them the first time)
        since = (lab.store.get("gone_at") or lab.cfg.history_start)[:10]
        today = datetime.now(timezone.utc).date().isoformat()
        lab.assets.save_gone(ingest.gone_symbols(lab.api, since, today, set(lab.assets.common())))
        lab.store.put("gone_at", today)
    except AlpacaError as e:
        lab.say(f"could not read the list of bought-out companies: {e}", "warn")
    lab.store.put("assets_at", datetime.now(timezone.utc).isoformat())
    lab.say(f"asset list read: {len(lab.assets.frame())} symbols, {len(lab.assets.common(True))} listed common stocks")


def backfill(lab: Lab, today: date) -> None:
    """The long history, once. Takes about an hour on the free plan's rate."""
    syms = lab.thin(sorted(set(lab.assets.common()) | set(lab.instruments())))
    lab.say(f"fetching history from {lab.cfg.history_start} for {len(syms)} symbols")

    def progress(done: int, total: int, rows: int) -> None:
        lab.doing(f"fetching price history: {done} of {total} symbols")

    ingest.backfill(lab.api, lab.bars, syms, lab.cfg.history_start, today.isoformat(), progress=progress)
    lab.doing("picking each month's universe")
    months = lab.universe.build(lab.bars, lab.assets.common(), lab.cfg.index_symbol, lab.cfg.universe_size, lab.cfg.min_price)
    lab.store.put("history_done", today.isoformat())
    lab.say(f"history in: {len(lab.bars.days(lab.cfg.index_symbol))} trading days, universe built for {months} months")


def update_prices(lab: Lab, today: date, held: set[str]) -> dict:
    lab.doing("fetching today's prices")
    syms = sorted(set(lab.thin(sorted(set(lab.assets.common(active_only=True)) | set(lab.instruments())))) | held)
    return ingest.update(lab.api, lab.bars, syms, today, note=lambda t: lab.say(t, "warn"))


def refresh_universe(lab: Lab, today: int) -> None:
    """Build next month's list on the month's last trading day, and any month that is missing."""
    d = date(today // 10000, today // 100 % 100, today % 100)
    nxt = cal.next_trading_day(d)
    want = nxt.year * 100 + nxt.month
    have = lab.universe.months()
    if month_of(today) not in have or want not in have:
        lab.doing("picking the month's universe")
        lab.universe.build(lab.bars, lab.assets.common(), lab.cfg.index_symbol, lab.cfg.universe_size,
                           lab.cfg.min_price, since_month=min(month_of(today), want))
        lab.say(f"universe for {want // 100}-{want % 100:02d}: {len(lab.universe.members(want * 100 + 1))} stocks")


def books(lab: Lab, panel: Panel) -> dict[str, Record]:
    """Every model's book, replayed from its decisions."""
    costs = Costs(panel, lab.cfg.slip_bps, lab.cfg.impact_coef)
    out = {}
    for mid in lab.model_ids():
        since = lab.store.since(mid)
        if not since:
            continue
        orders = {s2d(dec.date): dec.orders for dec in lab.store.decisions(mid, since)}
        try:
            out[mid] = replay(panel, orders, lab.cfg.start_cash, costs, s2d(since), frozenset(lab.assets.gone()))
        except KeyError:
            lab.say(f"{mid}: its first day {since} is not in the price table", "warn")
    return out


def may_start(lab: Lab, mid: str) -> bool:
    """A model trades only once its backtest report is written (list line 6)."""
    return mid in NEEDS_NO_REPORT or lab.store.read_json(f"reports/{mid}.json") is not None


def decide(lab: Lab, panel: Panel, recs: dict[str, Record], today_iso: str) -> dict[str, Decision]:
    """Each model's decision for tomorrow's open. Returns the ones made in this run or stored earlier today."""
    env = Env(index=lab.cfg.index_symbol, cash=lab.cfg.cash_symbol, names=lab.assets.names(),
              events=lab.events if (lab.cfg.has_sec or not lab.events.empty()) else None, fits=lab.fits)
    models = build(lab.cfg.models)
    for m in models:
        env.peers[m.id] = m
    out: dict[str, Decision] = {}
    t = panel.T - 1
    lab.models = models
    for m in sorted(models, key=lambda m: m.id == "blend"):
        try:
            m.prepare(panel, env)
        except Exception as e:
            lab.say(f"{m.id}: could not prepare: {e}", "warn")
            log.debug(traceback.format_exc())
            m._failed = str(e)
    for m in models:
        if getattr(m, "_failed", None) or m.waiting() or not may_start(lab, m.id):
            continue
        if lab.store.last_decision_date(m.id) == today_iso:
            stored = lab.store.decisions(m.id, today_iso)
            if stored:
                out[m.id] = stored[-1]
            continue
        rec = recs.get(m.id)
        view = rec.book.view() if rec is not None and rec.book else BookView(lab.cfg.start_cash, 1.0)
        try:
            d = m.decide(t, view)
        except Exception as e:
            lab.say(f"{m.id}: could not decide: {e}", "warn")
            log.debug(traceback.format_exc())
            continue
        if lab.store.save_decision(d):
            if not lab.store.since(m.id):
                lab.store.set_since(m.id, today_iso)
                lab.say(f"{m.title} starts its record on {today_iso}")
            for sig in d.meta.get("signals", []):
                lab.store.save_signal(Signal(m.id, today_iso, sig["symbol"], sig["horizon"], sig["p"], sig.get("note", "")))
        out[m.id] = d
    return out


def run_agent(lab: Lab, panel: Panel, recs: dict[str, Record], today_iso: str) -> Decision | None:
    if not (lab.cfg.has_agent and (not lab.cfg.models or "agent" in lab.cfg.models)):
        return None
    if lab.store.last_decision_date("agent") == today_iso:
        stored = lab.store.decisions("agent", today_iso)
        return stored[-1] if stored else None
    from .agent.data import LiveDeskData
    from .agent.desk import Desk
    lab.doing("the agent is reading")
    t = panel.T - 1
    names = lab.assets.names()
    frame = lab.assets.frame()
    data = LiveDeskData(panel, lab.cfg.index_symbol, names, lab.universe.ranks(int(panel.days[t])),
                        events=lab.events if lab.cfg.has_sec else None, sec=lab.sec, alpaca=lab.api,
                        exchanges=dict(zip(frame["sym"], frame["exchange"])) if len(frame) else {})
    rec = recs.get("agent")
    view = rec.book.view() if rec is not None and rec.book else BookView(lab.cfg.start_cash, 1.0)
    result = Desk(lab.cfg, lab.store).run(data, view)
    lab.agent_stopped = result.stopped
    if result.stopped == "no key":
        return None
    d = Decision("agent", today_iso, list(result.orders),
                 note=f"{result.triaged} items triaged, {len(result.reads)} companies read, ${result.usd:.2f} spent"
                      + (f"; stopped early: {result.stopped}" if result.stopped else ""),
                 meta={"usd": result.usd, "stopped": result.stopped})
    if lab.store.save_decision(d) and not lab.store.since("agent"):
        lab.store.set_since("agent", today_iso)
        lab.say(f"The agent starts its record on {today_iso}")
    return d


def settle(lab: Lab, panel: Panel) -> None:
    fwd = {h: relative(forward(panel, h), panel.member) for h in HORIZONS}
    n = 0
    for m in getattr(lab, "models", []):
        if isinstance(m, ScoreModel):
            n += ledger.settle_scores(lab.store, panel, fwd, m.id, HORIZONS)
    n += ledger.settle_signals(lab.store, panel, lab.cfg.index_symbol)
    if n:
        lab.say(f"{n} rankings and forecasts scored against what followed")


def daily_line(lab: Lab, panel: Panel, recs: dict[str, Record], today_iso: str) -> None:
    """One line to the phone, once a day."""
    if lab.cfg.notify == "off" or lab.store.get("notified") == today_iso or not recs:
        return
    j = panel.ix[lab.cfg.index_symbol]
    idx_day = float(panel.ret[-1, j]) if np.isfinite(panel.ret[-1, j]) else 0.0
    rows = []
    for mid, rec in recs.items():
        if len(rec.equity) < 2 or rec.days[-1] != int(panel.days[-1]):
            continue
        day = rec.equity[-1] / rec.equity[-2] - 1.0
        fills = [f for f in rec.book.fills if f.d == rec.days[-1]]
        rows.append((mid, day, sum(f.side == "buy" for f in fills), sum(f.side == "sell" for f in fills)))
    if not rows:
        return
    by = {r[0]: r for r in rows}
    contest = sorted((r for r in rows if r[0] != "index"), key=lambda r: -r[1])
    parts = [f"index {idx_day:+.2%}"]
    if contest:
        parts.append(f"best {contest[0][0]} {contest[0][1]:+.2%}")
        if len(contest) > 1:
            parts.append(f"worst {contest[-1][0]} {contest[-1][1]:+.2%}")
    for key in ("net", "agent"):
        if key in by and (not contest or key not in (contest[0][0], contest[-1][0])):
            parts.append(f"{key} {by[key][1]:+.2%}")
    buys, sells = sum(r[2] for r in rows), sum(r[3] for r in rows)
    parts.append(f"{buys} bought, {sells} sold" if buys or sells else "no trades")
    if lab.notifier.send(f"Stocks {today_iso}", " · ".join(parts)):
        lab.store.put("notified", today_iso)
    if lab.cfg.notify == "trades":
        for mid, rec in recs.items():
            for f in rec.book.fills:
                if f.d == rec.days[-1]:
                    lab.notifier.send(f"Stocks: {mid} {f.side} {f.sym}", f"${f.usd:,.0f} at {f.price:,.2f} · {f.why}", "low")


# ── the page ─────────────────────────────────────────────────────────────────────────────────────
def phase_of(lab: Lab, recs: dict[str, Record]) -> tuple[str, str]:
    if not lab.cfg.has_market_data:
        return "keys", "Waiting for the Alpaca keys"
    if not lab.have_history():
        return "backfill", "Fetching ten years of prices"
    if not recs:
        return "research", "Prices are in; fitting and backtesting before anything trades"
    return "running", "Running"


def notices(lab: Lab, phase: str, data_day: int | None, today: int | None) -> list[dict]:
    cfg, out = lab.cfg, []
    if not cfg.has_market_data:
        out.append({"level": "warn", "text": "No market data yet: add ALPACA_API_KEY and ALPACA_API_SECRET to .env, then `plint up stocks`."})
    if not cfg.has_agent:
        out.append({"level": "info", "text": "The agent is off until STOCKS_ANTHROPIC_API_KEY is in .env."})
    if not cfg.has_sec:
        out.append({"level": "info", "text": "Filings are off until SEC_CONTACT (an email address: the SEC asks who is calling) is in .env. "
                                             "Without it the after-earnings and insider models wait, and the agent reads headlines only."})
    fault = lab.store.get("fault")
    if fault:
        out.insert(0, {"level": "fault", "text": fault})
    if data_day and today and data_day < today and phase == "running":
        out.insert(0, {"level": "fault", "text": f"No new prices since {d2s(data_day)}. The books stand at that close."})
    if lab.agent_stopped in ("month cap", "day allowance"):
        text = (f"The agent reached its ${cfg.agent_cap_usd:,.0f} for the month and holds what it has until the 1st."
                if lab.agent_stopped == "month cap" else "The agent used up today's share of its budget and stopped reading early.")
        out.append({"level": "warn", "text": text})
    return out


def publish_state(lab: Lab, panel: Panel | None, recs: dict[str, Record], decisions: dict[str, Decision],
                  next_run: datetime | None = None, clock: datetime | None = None) -> None:
    cfg = lab.cfg
    models = getattr(lab, "models", None) or build(cfg.models)
    metas = publish.metas(models)
    by_id = {m.id: m for m in models}
    names = lab.assets.names()
    now = publish.now_local()
    phase, phase_text = phase_of(lab, recs)
    today = int(panel.days[-1]) if panel is not None and panel.T else None
    summaries, details = [], {}
    for meta in metas:
        mid = meta["id"]
        m = by_id.get(mid)
        rec = recs.get(mid)
        if mid == "agent":
            state, text = ("running", "") if rec else ("waiting", "waiting for its API key" if not cfg.has_agent else "starts at the next evening run")
            if lab.agent_stopped == "month cap":
                state, text = "off", "month cap reached"
            horizon, scores_all = 20, False
        else:
            wait = getattr(m, "_failed", None) or (m.waiting() if hasattr(m, "panel") else "")
            if phase in ("keys", "backfill"):
                wait = "waiting for market data"
            elif not wait and not may_start(lab, mid):
                wait = "waiting for its backtest"
            state, text = ("running", "") if rec and not wait else ("waiting", wait or "starts at the next evening run")
            if rec and wait:
                state, text = "waiting", wait
            horizon, scores_all = m.horizon, isinstance(m, ScoreModel)
        dec = decisions.get(mid)
        s, d = publish.model_json(meta, rec, panel, cfg.index_symbol, lab.store, names, cfg.start_cash, state, text,
                                  horizon, scores_all, dec.orders if dec else None, dec.note if dec else "")
        summaries.append(s)
        details[mid] = d
    data_day = lab.bars.last_day(cfg.index_symbol) if cfg.has_market_data else None
    want = cal.last_close(clock or datetime.now(timezone.utc), cfg.run_at)
    state = {
        "app": "stocks", "generated": publish.iso(now), "asof": d2s(today) if today and recs else None,
        "phase": phase, "phase_text": phase_text, "keys": cfg.keys(),
        "notices": notices(lab, phase, data_day, want.year * 10000 + want.month * 100 + want.day),
        "clock": {"last_run": lab.store.get("last_run"), "next_run": publish.iso(next_run), "busy": lab.busy},
        "start_cash": cfg.start_cash,
        "index": {"symbol": cfg.index_symbol, "title": "S&P 500 (SPY)" if cfg.index_symbol == "SPY" else cfg.index_symbol},
        "spend": publish.spend_block(lab.store, cfg.agent_cap_usd, lab.agent_stopped, now),
        "models": summaries, "log": lab.store.recent_log(40),
    }
    publish.write(lab.store, state, details)


# ── the run ──────────────────────────────────────────────────────────────────────────────────────
def evening(lab: Lab, now: datetime | None = None, next_run: datetime | None = None) -> str:
    """One evening. Returns a line saying what happened."""
    cfg = lab.cfg
    now = now or datetime.now(timezone.utc)
    if not cfg.has_market_data:
        publish_state(lab, None, {}, {}, next_run, now)
        return "no Alpaca keys: nothing to do"
    today = cal.last_close(now, cfg.run_at)
    today_i = today.year * 10000 + today.month * 100 + today.day
    try:
        refresh_assets(lab)
        if not lab.have_history():
            publish_state(lab, None, {}, {}, next_run, now)
            backfill(lab, today)
        else:
            held = set()
            for mid in lab.model_ids():
                st = lab.store.read_json(f"model_{mid}.json") or {}
                held.update(p["symbol"] for p in st.get("positions", []))
            update_prices(lab, today, held)
        refresh_universe(lab, lab.bars.last_day(cfg.index_symbol) or today_i)
        lab.store.put("fault", None)
    except AlpacaError as e:
        lab.store.put("fault", f"Alpaca: {e}. Prices were not updated; the books stand at the last close.")
        lab.say(f"prices not updated: {e}", "warn")
        if not lab.have_history():
            publish_state(lab, None, {}, {}, next_run, now)
            return f"stopped: {e}"
    data_day = lab.bars.last_day(cfg.index_symbol)
    lab.doing("marking the books")
    panel = lab.live_panel(data_day)
    today_iso = d2s(int(panel.days[-1]))
    if lab.sec is not None:
        try:
            from . import filings
            if not lab.store.get("filings_backfilled"):
                lab.doing("reading ten years of filings: results days and insider purchases")
                got = filings.backfill(lab.sec, lab.events, lab.universe.ever(), int(cfg.history_start[:4]), log=lab.say)
                lab.store.put("filings_backfilled", today.isoformat())
                lab.say(f"filings history in: {len(lab.events.earnings)} results days, {len(lab.events.insiders)} insider purchases")
            lab.doing("reading today's filings")
            since = lab.store.get("filings_since") or (now - timedelta(days=5)).astimezone(cal.NY).strftime("%Y-%m-%dT%H:%M:%S")
            got = filings.update(lab.sec, lab.events, [panel.syms[j] for j in panel.members(panel.T - 1)], since, log=lab.say)
            lab.store.put("filings_since", now.astimezone(cal.NY).strftime("%Y-%m-%dT%H:%M:%S"))
            log.info("filings: %s", {k: v for k, v in got.items() if k != "failures"})
        except Exception as e:
            lab.say(f"filings not read: {e}", "warn")
            log.debug(traceback.format_exc())
    recs = books(lab, panel)
    daily_line(lab, panel, recs, today_iso)
    lab.doing("deciding for tomorrow")
    decisions = decide(lab, panel, recs, today_iso)
    try:
        agent_decision = run_agent(lab, panel, recs, today_iso)
        if agent_decision:
            decisions["agent"] = agent_decision
    except Exception as e:
        lab.say(f"the agent failed tonight and holds what it has: {e}", "warn")
        log.warning(traceback.format_exc())
    settle(lab, panel)
    lab.store.put("last_run", publish.iso(publish.now_local()))
    lab.doing(None)
    recs = books(lab, panel)                     # again, so models that started tonight show their first day
    publish_state(lab, panel, recs, decisions, next_run, now)
    orders = sum(len(d.orders) for d in decisions.values())
    line = f"evening run done for {today_iso}: {len(recs)} books, {orders} orders for the next open"
    lab.say(line)
    return line
