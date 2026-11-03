"""Strategy 5, "afternoon": Yes on the running high's bucket at 15:00 local, at the real ask.

Filip's idea: it is usually hottest in the middle of the day, so by mid-afternoon
the day's high is probably in, and the market sells that bucket's Yes a few
points too cheap (research/weather_hasmax.py: at 15:00 the real asks for the
running bucket's Yes paid 27→held 29, 42→45, 58→61, 73→76; buying at 20–90¢
made +5.6% a trade, 90% CI +1.7 to +9.4, over 1,358 trades; buying No instead
lost 22%). The bot reads the station's reports the way the markets' rule does
(shared with the collector), fixes the bucket the running high falls in at the
first look after the city's hour, and buys its Yes within that hour when it
costs between AF_MIN_PRICE and AF_MAX_PRICE, once per city and day, held to
resolution.

Each city has its own hour (research/weather_hour.py: timing every city at
15:00 made +5.6% a trade; timing each at the hour by which its own high had
landed on 60% of its earlier days made +10.4%, CI +7.3 to +13.4, on 1,790
trades): the AF_QUANTILE point of when the high was first reached over the
last AF_HISTORY_DAYS days of the station's reports (IEM archive, refreshed
daily), the shipped polybot/afternoon_hours.json until that history is in,
and AF_HOUR when neither is.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .collector import city_hour, daily_max_hours, day_summary, in_window, winning_bucket
from .core import Core
from .metar import fetch_history
from .store import iso
from .weather import local_now

SHIPPED = Path(__file__).resolve().parent / "afternoon_hours.json"

log = logging.getLogger("polybot.afternoon")


class AfternoonStrategy:
    name = "afternoon"
    kind = "afternoon"

    def __init__(self, core: Core, collector):
        self.core = core
        self.cfg = core.cfg
        self.store = core.store
        self.collector = collector            # shares its markets, stations and station reports
        self.label = "Afternoon · Yes on the running high at each city's hour"
        self.picks: dict[tuple, dict] = {}    # (city, day) → the bucket fixed at the first look in the window
        self.hours: dict[str, dict] = {}      # city → {"hour", "source", "days", "median", "at"}
        try:
            self.shipped = json.loads(SHIPPED.read_text()).get("cities", {})
        except Exception:
            self.shipped = {}
        self.rows: list[dict] = []
        self.decisions: dict[tuple, str] = {}
        self.trade = True

    @property
    def interval(self) -> int:
        return self.cfg.af_tick_seconds

    HISTORY_TTL = timedelta(hours=24)
    REFRESH_PER_TICK = 8

    def refresh_hours(self, now: datetime) -> None:
        """Keep each city's hour current: a few stations a tick, each once a day, from the archive's last weeks."""
        col = self.collector
        q = self.cfg.af_quantile
        cities = {}
        for (city, _day, kind), ev in col.events.items():
            if kind == "highest" and ev["icao"] and ev["metar_rule"] and col.stations.get(ev["icao"]):
                cities[city] = ev["icao"]
        due = [c for c in cities if c not in self.hours or self.hours[c].get("source") != "station history"
               or now - datetime.fromisoformat(self.hours[c]["at"]) >= self.HISTORY_TTL]
        due.sort(key=lambda c: self.hours.get(c, {}).get("at", ""))
        for city in due[:self.REFRESH_PER_TICK]:
            icao = cities[city]
            st = col.stations[icao]
            try:
                reports = fetch_history(icao, self.cfg.af_history_days)
                hist = daily_max_hours(reports, ZoneInfo(st["tz"]), before=local_now(st["tz"], now).date())
                hist = hist[-self.cfg.af_history_days:]
                if len(hist) >= self.cfg.af_min_history:
                    hs = [h for _, h in hist]
                    self.hours[city] = {"hour": city_hour(hs, q), "source": "station history", "days": len(hs),
                                        "median": round(sorted(hs)[len(hs) // 2], 1), "at": iso(now)}
                    continue
                note = f"only {len(hist)} complete days in the archive"
            except Exception as e:
                note = f"archive: {str(e)[:60]}"
                log.info("history for %s (%s): %s", city, icao, note)
            self.hours[city] = {**self.fallback(city), "at": iso(now), "note": note}

    def fallback(self, city: str) -> dict:
        s = self.shipped.get(city)
        key = f"q{int(round(self.cfg.af_quantile * 100))}"
        if s and key in s:
            return {"hour": int(s[key]), "source": "shipped table", "days": s.get("days"), "median": s.get("median")}
        return {"hour": self.cfg.af_hour, "source": "AF_HOUR", "days": 0, "median": None}

    def hour_for(self, city: str) -> dict:
        h = self.hours.get(city)
        return h if h else self.fallback(city)

    def skipped_city(self, city: str) -> str | None:
        if city in {c.strip() for c in self.cfg.af_skip_cities.split(",") if c.strip()}:
            return "city skipped in AF_SKIP_CITIES"
        return None

    def run(self, now: datetime) -> None:
        if not self.cfg.af_enabled:
            return
        cfg = self.cfg
        col = self.collector
        col.load_markets(now)
        self.refresh_hours(now)
        rows: list[dict] = []
        pending: list[tuple] = []
        for (city, day, kind), ev in sorted(col.events.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if kind != "highest":
                continue
            icao = ev["icao"]
            st = col.stations.get(icao) if icao else None
            row = {"city": city, "day": day.isoformat(), "station": icao or "", "unit": ev["items"][0][0].unit}
            why = self.skipped_city(city)
            if not ev["metar_rule"]:
                continue                       # another resolution source (Hong Kong): the running high is not theirs
            if why:
                row["decision"] = why
            elif not st:
                row["decision"] = f"unknown station {icao or '?'}"
            else:
                lt = local_now(st["tz"], now)
                if lt.date() != day:
                    continue                   # only the day each city is in right now
                row["local_time"] = lt.strftime("%a %H:%M")
                hc = self.hour_for(city)
                row.update({"hour": hc["hour"], "hour_source": hc["source"], "hour_days": hc.get("days"), "median": hc.get("median")})
                w = in_window(lt, hc["hour"])
                pick = self.picks.get((city, day))
                if pick is not None:
                    row.update({"value_c": pick["value_c"], "bucket": pick["bucket"], "ask": pick.get("ask"), "reports": pick.get("reports")})
                if w < 0:
                    row["decision"] = f"waiting for {hc['hour']:02d}:00 there"
                elif pick is not None and pick.get("final"):
                    row["decision"] = pick["final"]
                elif w > 0:
                    row["decision"] = "window over" if pick is None else pick.get("last", "window over")
                else:
                    pending.append((row, ev, st, lt))
            rows.append(row)
        col.reports_for(sorted({ev["icao"] for _, ev, _, _ in pending}), now)
        bought: list[tuple] = []
        n = 0
        for row, ev, st, lt in pending:
            key = (row["city"], row["day"])
            pick = self.picks.get(key)
            if pick is None:
                reports = col.reports.get(ev["icao"], (None, []))[1]
                s = day_summary(reports, ZoneInfo(st["tz"]), date.fromisoformat(row["day"]), until=lt)
                if s.n == 0 or s.max_tenths is None:
                    row["decision"] = "no reports yet"
                    continue
                buckets = [b for b, _ in ev["items"]]
                b = winning_bucket(buckets, s.max_tenths, row["unit"])
                if b is None:
                    row["decision"] = "no bucket contains the running high"
                    continue
                m = next(mk for bb, mk in ev["items"] if bb is b)
                o = m.outcomes.yes
                pick = self.picks[key] = {"value_c": s.max_tenths / 10, "reports": s.n, "bucket": b.label, "question": b.question, "hour": row.get("hour"),
                                          "asset_id": self.core.trading_id(m, o), "condition_id": str(m.condition_id), "slug": m.slug,
                                          "event_slug": m.events[0].slug if m.events else "", "gamma": float(o.price) if o.price is not None else None,
                                          "at": lt.strftime("%H:%M")}
                row.update({"value_c": pick["value_c"], "bucket": pick["bucket"], "reports": pick["reports"]})
            if not self.trade:
                row["decision"] = f"scan only · Yes on {pick['bucket']} shows {pick['gamma']}"
                continue
            if n >= cfg.af_max_per_tick:
                row["decision"] = "enough for this tick"
                continue
            try:
                if self.consider(row, pick, now):
                    bought.append((row, pick))
                    n += 1
            except Exception as e:
                log.exception("could not act on %s", pick["question"])
                row["decision"] = f"error: {e}"
                self.core.record_error(f"afternoon {row['city']}: {e}")
        if bought:
            verb = "bought" if self.core.mode == "live" else "paper buy"
            self.core.notifier.send(f"Polybot: afternoon {verb}",
                                    "\n".join(f"[afternoon] {r['city']}: Yes on {p['bucket']} at {p['fill_price']:.2f} (${p['fill_cost']:.2f}), high so far {p['value_c']:g}°C" for r, p in bought),
                                    tags="sunny")
        cutoff = (now - timedelta(days=2)).date().isoformat()
        self.picks = {k: v for k, v in self.picks.items() if k[1] >= cutoff}
        self.rows = rows

    def consider(self, row: dict, pick: dict, now: datetime) -> bool:
        cfg = self.cfg
        core = self.core
        asset = pick["asset_id"]
        what = f"{pick['question']} · Yes"

        def skip(reason: str, final: bool = False) -> bool:
            row["decision"] = reason
            pick["last"] = reason
            if final:
                pick["final"] = reason
                if asset:
                    self.store.mark_acted(asset, self.name, now)
            if self.decisions.get((row["city"], row["day"])) != reason:
                log.info("skip %s: %s", what, reason)
            self.decisions[(row["city"], row["day"])] = reason
            return False

        if not asset:
            return skip("no tradeable token", final=True)
        if self.store.position(asset, self.name):
            return skip("holding", final=True)
        if self.store.acted_at(asset, self.name):
            return skip("already decided", final=True)
        if len(self.store.open_positions(self.name)) >= cfg.af_max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24), self.name) + cfg.af_stake_usd > cfg.af_daily_budget_usd:
            return skip("daily budget used up")
        m = core.market(pick["condition_id"], pick["slug"] or "")
        if m is None:
            return skip("market not found")
        why = core.not_tradeable(m, now)
        if why:
            return skip(why, final=True)
        est = core.estimate(asset, "BUY", amount=cfg.af_stake_usd)
        if est is None:
            return skip("not enough liquidity")        # try again next tick, within the hour
        pick["ask"] = float(est)
        row["ask"] = float(est)
        if est > cfg.af_max_price:
            return skip(f"Yes at {est:.2f}: above {cfg.af_max_price}", final=True)
        if est < cfg.af_min_price:
            return skip(f"Yes at {est:.2f}: below {cfg.af_min_price}", final=True)
        if core.mode == "live" and core.balance is not None and core.balance < cfg.af_stake_usd:
            core.warn("funds", f"not enough pUSD for a ${cfg.af_stake_usd} order")
            return skip("not enough pUSD in the wallet")
        fill = core.buy(asset, cfg.af_stake_usd, est, what, fee_rate=core.fee_rate(m))
        if fill is None:
            return skip("order rejected", final=True)
        rec = {
            "strategy": self.name, "side": "BUY", "asset_id": asset, "condition_id": pick["condition_id"], "title": pick["question"],
            "outcome": "Yes", "slug": pick["slug"] or "", "event_slug": pick["event_slug"], "city": row["city"], "day": row["day"],
            "kind": "highest", "station": row["station"], "value_c": pick["value_c"], "bucket": pick["bucket"], "picked_at": pick["at"], "city_hour": pick.get("hour"),
            "mode": core.mode, **fill,
        }
        self.store.open_position(rec, now)
        pick["final"] = "bought" if core.mode == "live" else "paper bought"
        pick["fill_price"] = rec["price"]
        pick["fill_cost"] = rec["cost"]
        row["decision"] = pick["final"]
        log.info("%s Yes on %s at %.2f ($%s): high so far %g°C from %d reports", "Bought" if core.mode == "live" else "Paper buy",
                 pick["question"], rec["price"], rec["cost"], pick["value_c"], pick["reports"] or 0)
        return True

    def status(self) -> dict:
        def rank(r):
            d = r.get("decision", "")
            return 0 if "bought" in d or d == "holding" else 1 if not d.startswith("waiting") else 2
        rows = sorted(self.rows, key=lambda r: (rank(r), r.get("city", "")))
        src = {}
        for h in self.hours.values():
            src[h["source"]] = src.get(h["source"], 0) + 1
        return {"enabled": self.cfg.af_enabled, "hour": self.cfg.af_hour, "quantile": self.cfg.af_quantile, "min_price": float(self.cfg.af_min_price),
                "max_price": float(self.cfg.af_max_price), "rows": rows[:80], "today": len(self.rows), "hours": src}
