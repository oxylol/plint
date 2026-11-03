"""Strategy 4, "collector": buy the tokens a finished day has already decided, before the market resolves.

No forecasting. Each daily temperature event resolves on the station's routine
METAR reports (the rule is in the market's description, the source is the
Aviation Weather Center's API); the bot reads the same reports the same way.
Once a city's high is set (from a lock hour, with the reading well below the
high, or when the day is over) the winning bucket is known, yet its Yes and the
other buckets' No still trade below $1 until the first report of the next day
settles the event. The bot buys those at up to a maximum price and redeems them
at $1. The research behind it: research/weather_carry.py.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal

from .collector import day_summary, lock_state, payers, resolution_source, winning_bucket
from .core import Core
from .metar import Report, fetch_reports
from .weather import Bucket, local_now, parse_question, station_info

log = logging.getLogger("polybot.collector")


class CollectorStrategy:
    name = "collector"
    kind = "collector"
    label = "Collector · known winners after the day"
    MARKETS_TTL = timedelta(minutes=30)
    SKIP_AFTER_LOSS = timedelta(days=30)

    def __init__(self, core: Core):
        self.core = core
        self.cfg = core.cfg
        self.store = core.store
        self.events: dict[tuple, dict] = {}         # (city, day, kind) → {"items": [(Bucket, market)], "icao", "read_at", "metar_rule"}
        self.stations: dict[str, dict | None] = {}  # icao → {lat, lon, tz, name}
        self.reports: dict[str, tuple[datetime, list[Report]]] = {}
        self.rows: list[dict] = []
        self.decisions: dict[tuple, str] = {}
        self._markets_at: datetime | None = None
        self._alerted: set[str] = set()
        self.trade = True                            # the `collector` command scans without buying

    @property
    def interval(self) -> int:
        return self.cfg.cl_tick_seconds

    # ── markets ──────────────────────────────────────────────
    def load_markets(self, now: datetime) -> None:
        if self._markets_at and now - self._markets_at < self.MARKETS_TTL:
            return
        core = self.core
        kinds = {"highest"} | ({"lowest"} if self.cfg.cl_lowest else set())
        events: dict[tuple, dict] = {}
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Temperature markets end at noon UTC on their day and stay open until the next day's first report settles
        # them; a city's day can be the UTC day before or after, so look at three days.
        for d in range(-1, 2):
            lo = start + timedelta(days=d, hours=11, minutes=58)
            pages = core.public.list_markets(closed=False, end_date_min=lo, end_date_max=lo + timedelta(minutes=4), page_size=100)
            page = core.call(pages.first_page)
            n = 0
            while True:
                for m in page.items:
                    if "temperature in" not in (m.question or ""):
                        continue
                    b = parse_question(m.question, (m.state.end_date.year if m.state.end_date else now.year))
                    if b is None or b.kind not in kinds:
                        continue
                    ev = events.get((b.city, b.day, b.kind))
                    if ev is None:
                        icao, read_at, metar_rule = resolution_source(m.description)
                        ev = events[(b.city, b.day, b.kind)] = {"items": [], "icao": icao, "read_at": read_at, "metar_rule": metar_rule}
                        if icao and icao not in self.stations:
                            self.stations[icao] = station_info(icao)
                    ev["items"].append((b, m))
                n += 1
                if not page.next_cursor or n >= 80:
                    break
                page = core.call(pages.from_cursor(page.next_cursor).first_page)
        self.events = events
        self._markets_at = now
        log.info("%d temperature events over three days (%d cities)", len(events), len({k[0] for k in events}))

    def reports_for(self, icaos: list[str], now: datetime) -> None:
        """Fetch what the tick needs, ten stations a request; a failed station keeps its last reports."""
        want = [i for i in icaos if i not in self.reports or now - self.reports[i][0] >= timedelta(seconds=self.cfg.cl_tick_seconds) - timedelta(seconds=30)]
        for i in range(0, len(want), 10):
            batch = want[i:i + 10]
            try:
                got = fetch_reports(batch)
            except Exception as e:
                self.core.warn("metar", f"could not read the station reports: {e}")
                return
            for icao, rs in got.items():
                if rs or icao not in self.reports:
                    self.reports[icao] = (now, rs)

    # ── bookkeeping ──────────────────────────────────────────
    def skipped_city(self, city: str, now: datetime) -> str | None:
        if city in {c.strip() for c in self.cfg.cl_skip_cities.split(",") if c.strip()}:
            return "city skipped in CL_SKIP_CITIES"
        since = now - self.SKIP_AFTER_LOSS
        for c in self.store.state.get("closed", []):
            if c.get("strategy") != self.name or float(c.get("pnl", 0)) >= 0:
                continue
            b = parse_question(c.get("title") or "", now.year)
            closed_at = c.get("closed_at") or ""
            if b is not None and b.city == city and closed_at >= since.isoformat(timespec="seconds"):
                return f"a position here lost on {closed_at[:10]}: skipped for 30 days"
        return None

    def check_losses(self, now: datetime) -> None:
        """A lost collector position means the rule was read wrong somewhere: say so loudly."""
        for c in self.store.state.get("closed", []):
            if c.get("strategy") != self.name or float(c.get("pnl", 0)) >= 0:
                continue
            key = f"{c.get('asset_id')}|{c.get('closed_at')}"
            if key in self._alerted:
                continue
            self._alerted.add(key)
            if (c.get("closed_at") or "") < (now - timedelta(days=2)).isoformat(timespec="seconds"):
                continue
            msg = f"[collector] {c.get('title')} · {c.get('outcome')} lost ${-float(c.get('pnl', 0)):.2f}: the day's value was read wrong. The city is skipped for 30 days."
            log.error(msg)
            self.core.record_error(msg)
            self.core.notifier.send("Polybot: collector was wrong", msg, priority="high", tags="rotating_light")

    # ── one tick ─────────────────────────────────────────────
    def run(self, now: datetime) -> None:
        if not self.cfg.cl_enabled:
            return
        self.load_markets(now)
        self.check_losses(now)
        cfg = self.cfg
        rows: list[dict] = []
        pending: list[tuple] = []
        for (city, day, kind), ev in sorted(self.events.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
            icao = ev["icao"]
            st = self.stations.get(icao) if icao else None
            row = {"city": city, "day": day.isoformat(), "kind": kind, "station": icao or "", "unit": ev["items"][0][0].unit,
                   "markets": len(ev["items"]), "read_at": ev["read_at"].isoformat() if ev["read_at"] else None}
            why = self.skipped_city(city, now)
            if not ev["metar_rule"]:
                row["decision"] = "resolves on another source, not the METAR rule"
            elif why:
                row["decision"] = why
            elif not st:
                row["decision"] = f"unknown station {icao or '?'}"
            else:
                lt = local_now(st["tz"], now)
                row["local_time"] = lt.strftime("%a %H:%M")
                if lt.date() < day:
                    row["decision"] = "day not started there"
                elif lt.date() == day and kind == "lowest":
                    row["decision"] = "lows wait for the day to end"
                elif lt.date() == day and lt.hour < cfg.cl_lock_hour - 1:
                    row["decision"] = f"waiting for {cfg.cl_lock_hour:02d}:00 there"
                else:
                    pending.append((row, ev, st, lt))
            rows.append(row)
        self.reports_for(sorted({ev["icao"] for _, ev, _, _ in pending}), now)
        candidates: list[tuple] = []
        for row, ev, st, lt in pending:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(st["tz"])
            day = date.fromisoformat(row["day"])
            reports = self.reports.get(ev["icao"], (None, []))[1]
            s = day_summary(reports, tz, day, until=lt)
            unit = row["unit"]
            val = s.value(row["kind"])
            row.update({"reports": s.n, "latest": s.latest_tenths / 10 if s.latest_tenths is not None else None,
                        "latest_at": s.latest_at.strftime("%H:%M") if s.latest_at else None,
                        "value_c": val / 10 if val is not None else None, "value_at": s.value_at(row["kind"]).strftime("%H:%M") if s.value_at(row["kind"]) else None,
                        "gap_min": round(s.gap_min)})
            locked, why = lock_state(row["kind"], s, lt, cfg.cl_lock_hour, int(round(cfg.cl_lock_margin_c * 10)), cfg.cl_max_gap_min)
            row["locked"] = locked
            row["status"] = why
            if not locked:
                row["decision"] = why
                continue
            buckets = [b for b, _ in ev["items"]]
            w = winning_bucket(buckets, val, unit)
            if w is None:
                row["decision"] = "no bucket contains the value"
                continue
            row["winner"] = w.label
            tokens = []
            for b, outcome in payers(buckets, w):
                m = next(mk for bb, mk in ev["items"] if bb is b)
                o = m.outcomes.yes if outcome == "Yes" else m.outcomes.no
                asset = self.core.trading_id(m, o)
                tokens.append({"bucket": b.label, "outcome": outcome, "asset_id": asset, "price": float(o.price) if o.price is not None else None,
                               "condition_id": str(m.condition_id), "slug": m.slug, "event_slug": m.events[0].slug if m.events else "",
                               "question": b.question, "held": bool(asset and self.store.position(asset, self.name))})
            row["tokens"] = tokens
            candidates.append((row, ev, w, tokens, st))
        bought_total = 0
        for row, ev, w, tokens, st in candidates:
            bought = []
            for tk in tokens:
                if not self.trade:
                    tk["decision"] = "scan only"
                    continue
                if bought_total >= cfg.cl_max_per_tick:
                    tk["decision"] = "enough for this tick"
                    continue
                try:
                    if self.consider(row, tk, now):
                        bought.append(tk)
                        bought_total += 1
                except Exception as e:
                    log.exception("could not act on %s", tk["question"])
                    tk["decision"] = f"error: {e}"
                    self.core.record_error(f"collector {row['city']}: {e}")
            held = sum(1 for tk in tokens if tk.get("held"))
            row["decision"] = (f"{len(bought)} bought" if bought else "") + (f" · {held} held" if held else "") or "nothing to buy"
            if bought:
                verb = "bought" if self.core.mode == "live" else "paper buy"
                self.core.notifier.send(f"Polybot: collector {verb}",
                                        f"[collector] {row['city']} {row['day']} {row['kind']}: {row['winner']} is set ({row['status']}).\n"
                                        + "\n".join(f"{tk['outcome']} on {tk['bucket']} at {tk['fill_price']:.3f} (${tk['fill_cost']:.2f})" for tk in bought),
                                        tags="white_check_mark")
        self.rows = rows

    def consider(self, row: dict, tk: dict, now: datetime) -> bool:
        cfg = self.cfg
        core = self.core
        asset = tk["asset_id"]
        what = f"{tk['question']} · {tk['outcome']}"

        def skip(reason: str, final: bool = False) -> bool:
            tk["decision"] = reason
            if final and asset:
                self.store.mark_acted(asset, self.name, now)
            key = (row["city"], row["day"], tk["bucket"], tk["outcome"])
            if self.decisions.get(key) != reason:
                log.info("skip %s: %s", what, reason)
            self.decisions[key] = reason
            return False

        if not asset:
            return skip("no tradeable token", final=True)
        if tk.get("held"):
            return skip("holding")
        if self.store.acted_at(asset, self.name):
            return skip("already decided")
        if len(self.store.open_positions(self.name)) >= cfg.cl_max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24), self.name) + cfg.cl_stake_usd > cfg.cl_daily_budget_usd:
            return skip("daily budget used up")
        m = core.market(tk["condition_id"], tk["slug"] or "")
        if m is None:
            return skip("market not found")
        why = core.not_tradeable(m, now)
        if why:
            return skip(why, final=True)
        est = core.estimate(asset, "BUY", amount=cfg.cl_stake_usd)
        if est is None:
            return skip("not enough liquidity")
        if est > cfg.cl_max_price:
            return skip(f"{est:.3f}: too little to gain")
        if est < cfg.cl_min_price:
            return skip(f"{est:.2f}: the market disagrees with the reports", final=True)
        if core.mode == "live" and core.balance is not None and core.balance < cfg.cl_stake_usd:
            core.warn("funds", f"not enough pUSD for a ${cfg.cl_stake_usd} order")
            return skip("not enough pUSD in the wallet")
        fill = core.buy(asset, cfg.cl_stake_usd, est, what, fee_rate=core.fee_rate(m))
        if fill is None:
            return skip("order rejected", final=True)
        rec = {
            "strategy": self.name, "side": "BUY", "asset_id": asset, "condition_id": tk["condition_id"], "title": tk["question"],
            "outcome": tk["outcome"], "slug": tk["slug"] or "", "event_slug": tk["event_slug"], "city": row["city"], "day": row["day"],
            "kind": row["kind"], "station": row["station"], "value_c": row.get("value_c"), "winner": row.get("winner"), "lock": row.get("status"),
            "bucket": tk["bucket"], "mode": core.mode, **fill,
        }
        self.store.open_position(rec, now)
        tk["decision"] = "bought" if core.mode == "live" else "paper bought"
        tk["held"] = True
        tk["fill_price"] = rec["price"]
        tk["fill_cost"] = rec["cost"]
        log.info("%s %s on %s at %.3f ($%s): %s", "Bought" if core.mode == "live" else "Paper buy", tk["outcome"], tk["question"], rec["price"], rec["cost"], row.get("status"))
        return True

    # ── the page ─────────────────────────────────────────────
    def status(self) -> dict:
        """The cities in their evening window (reports in hand) first, then anything skipped; the merely waiting are counted."""
        order = {True: 0, False: 1}
        quiet = ("day not started there", "lows wait for the day to end")
        rows = [r for r in self.rows if "reports" in r or not (r.get("decision", "").startswith("waiting for") or r.get("decision") in quiet)]
        rows = sorted(rows, key=lambda r: (order.get(r.get("locked"), 2), r.get("day", ""), r.get("city", "")))
        return {"enabled": self.cfg.cl_enabled, "lock_hour": self.cfg.cl_lock_hour, "margin_c": self.cfg.cl_lock_margin_c,
                "max_price": float(self.cfg.cl_max_price), "rows": rows[:120], "events": len(self.events), "waiting": len(self.rows) - len(rows)}
