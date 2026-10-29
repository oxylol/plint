"""Strategy 3, "forecast": bet No on tomorrow's high being the forecast plus a margin.

Filip's idea: pull the weather forecast for each city Polymarket has a daily
temperature event for, take the highest expected maximum across several
models, add 2°C, and buy "No" on that degree's market. The forecast is
rarely that far off, and the market tends to give the tail more than it
deserves. Positions are held to resolution; if a fresh forecast climbs into
the bucket we bet against, the position is sold.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal

from .core import CENT, Core
from .store import iso, parse_iso
from .weather import (Bucket, Forecast, fetch_forecast, geocode, local_now, parse_question, parse_station, pick_bucket,
                      station_info, target_value)

log = logging.getLogger("polybot.forecast")


class ForecastStrategy:
    name = "forecast"
    kind = "forecast"
    label = "Forecast · No at high +2°C"
    MARKETS_TTL = timedelta(minutes=30)
    FORECAST_TTL = timedelta(minutes=30)

    def __init__(self, core: Core):
        self.core = core
        self.cfg = core.cfg
        self.store = core.store
        self.label = f"Forecast · No at high +{self.cfg.fc_plus_c:g}°C"
        self.events: dict[tuple, list] = {}        # (city, day) → [(Bucket, market)]
        self.stations: dict[str, dict | None] = {}  # city → {icao, lat, lon, tz, name}
        self.forecasts: dict[tuple, Forecast] = {}  # (icao, day) → Forecast
        self.rows: list[dict] = []
        self._markets_at: datetime | None = None
        self.decisions: dict[tuple, str] = {}

    @property
    def interval(self) -> int:
        return self.cfg.fc_tick_seconds

    # ── markets ──────────────────────────────────────────────
    def load_markets(self, now: datetime) -> None:
        if self._markets_at and now - self._markets_at < self.MARKETS_TTL:
            return
        core = self.core
        kinds = {"highest"} | ({"lowest"} if self.cfg.fc_lowest else set())
        events: dict[tuple, list] = defaultdict(list)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Temperature markets end at noon UTC on their day; asking for that minute keeps the paging short
        # (a whole day holds thousands of other markets and the listing is capped).
        for d in range(self.cfg.fc_days_ahead + 2):
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
                    events[(b.city, b.day, b.kind)].append((b, m))
                    if b.city not in self.stations:
                        icao = parse_station(m.description, b.city)
                        info = station_info(icao) if icao else None
                        if info is None:
                            info = geocode(b.city)
                            if info:
                                info["icao"] = icao or ""
                        else:
                            info["icao"] = icao
                        self.stations[b.city] = info
                n += 1
                if not page.next_cursor or n >= 80:
                    break
                page = core.call(pages.from_cursor(page.next_cursor).first_page)
        self.events = dict(events)
        self._markets_at = now
        log.info("%d temperature events for the next days (%d cities)", len(self.events), len({k[0] for k in self.events}))

    def forecast_for(self, city: str, day: date, now: datetime) -> Forecast | None:
        st = self.stations.get(city)
        if not st:
            return None
        key = (st.get("icao") or city, day)
        fc = self.forecasts.get(key)
        if fc and fc.fetched_at and now - fc.fetched_at < self.FORECAST_TTL:
            return fc
        try:
            fc = fetch_forecast(st["lat"], st["lon"], day, [m for m in self.cfg.fc_models.split(",") if m])
        except Exception as e:
            self.core.warn(f"forecast:{city}", f"forecast for {city} failed: {e}")
            return self.forecasts.get(key)
        self.forecasts[key] = fc
        return fc

    # ── one tick ─────────────────────────────────────────────
    def run(self, now: datetime) -> None:
        if not self.cfg.fc_enabled:
            return
        self.load_markets(now)
        self.manage_positions(now)
        rows = []
        candidates = []
        for (city, day, kind), items in sorted(self.events.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            st = self.stations.get(city)
            row = {"city": city, "day": day.isoformat(), "kind": kind, "station": (st or {}).get("icao", ""), "station_name": (st or {}).get("name", "")}
            if not st:
                row["decision"] = "no station"
                rows.append(row)
                continue
            lt = local_now(st["tz"], now)
            row["local_time"] = lt.strftime("%a %H:%M")
            fc = self.forecast_for(city, day, now)
            if fc is None or fc.agg(kind, self.cfg.fc_agg) is None:
                row["decision"] = "no forecast yet"
                rows.append(row)
                continue
            agg = fc.agg(kind, self.cfg.fc_agg)
            buckets = [b for b, _ in items]
            unit = buckets[0].unit
            plus = self.cfg.fc_plus_c if kind == "highest" else -self.cfg.fc_plus_c
            tv = target_value(agg, plus, unit)
            bucket = pick_bucket(buckets, tv)
            row.update({"models": {k: round(v, 1) for k, v in (fc.highs if kind == "highest" else fc.lows).items()},
                        "forecast": round(agg, 1), "spread": round(fc.spread(kind) or 0, 1), "target": round(tv, 1), "unit": unit,
                        "bucket": bucket.label if bucket else None})
            if bucket is None:
                row["decision"] = "no bucket"
                rows.append(row)
                continue
            m = next(mk for b, mk in items if b is bucket)
            no = m.outcomes.no
            asset = self.core.trading_id(m, no)
            row.update({"condition_id": str(m.condition_id), "slug": m.slug, "asset_id": asset, "no_price": float(no.price) if no.price is not None else None,
                        "event_slug": m.events[0].slug if m.events else ""})
            pos = self.store.position(asset, self.name) if asset else None
            row["holding"] = pos is not None
            if pos is not None:
                row["decision"] = "holding"
            elif self.store.acted_at(asset, self.name) if asset else False:
                row["decision"] = "already decided"
            elif lt.date() >= day:
                row["decision"] = "day already started there"
            elif (day - lt.date()).days > self.cfg.fc_days_ahead:
                row["decision"] = f"{(day - lt.date()).days} days ahead: too early"
            else:
                candidates.append((fc.spread(kind) or 0.0, row, m, bucket, asset))
            rows.append(row)
        # the surest forecasts first (models agree), then the cheaper No
        candidates.sort(key=lambda c: (c[0], c[1].get("no_price") or 1.0))
        bought = 0
        for spread, row, m, bucket, asset in candidates:
            if bought >= self.cfg.fc_max_per_tick:
                row["decision"] = "enough for this tick"
                continue
            try:
                if self.consider(row, m, bucket, asset, now):
                    bought += 1
            except Exception as e:
                log.exception("could not act on %s", row["city"])
                row["decision"] = f"error: {e}"
                self.core.record_error(f"forecast {row['city']}: {e}")
        self.rows = rows

    def consider(self, row: dict, m, bucket: Bucket, asset: str, now: datetime) -> bool:
        cfg = self.cfg
        core = self.core
        what = f"{bucket.question} · No"

        def skip(reason: str, final: bool = False) -> bool:
            row["decision"] = reason
            if final and asset:
                self.store.mark_acted(asset, self.name, now)
            if self.decisions.get((row["city"], row["day"])) != reason:
                log.info("skip %s: %s", what, reason)
            self.decisions[(row["city"], row["day"])] = reason
            return False

        if not asset:
            return skip("no tradeable token", final=True)
        if len(self.store.open_positions(self.name)) >= cfg.fc_max_open:
            return skip("max open positions reached")
        if self.store.spent_since(now - timedelta(hours=24), self.name) + cfg.fc_stake_usd > cfg.fc_daily_budget_usd:
            return skip("daily budget used up")
        why = core.not_tradeable(m, now)
        if why:
            return skip(why, final=True)
        est = core.estimate(asset, "BUY", amount=cfg.fc_stake_usd)
        if est is None:
            return skip("not enough liquidity")
        if est > cfg.fc_max_price:
            return skip(f"No at {est:.2f}: too little to gain", final=True)
        if est < cfg.fc_min_price:
            return skip(f"No at {est:.2f}: the market strongly disagrees", final=True)
        if core.mode == "live" and core.balance is not None and core.balance < cfg.fc_stake_usd:
            core.warn("funds", f"not enough pUSD for a ${cfg.fc_stake_usd} order")
            return skip("not enough pUSD in the wallet")
        fill = core.buy(asset, cfg.fc_stake_usd, est, what, fee_rate=core.fee_rate(m))
        if fill is None:
            return skip("order rejected", final=True)
        rec = {
            "strategy": self.name, "side": "BUY", "asset_id": asset, "condition_id": str(m.condition_id), "title": bucket.question,
            "outcome": "No", "slug": m.slug or "", "event_slug": row.get("event_slug", ""), "city": row["city"], "day": row["day"],
            "kind": row["kind"], "station": row["station"], "forecast": row["forecast"], "spread": row["spread"], "target": row["target"],
            "bucket": bucket.label, "unit": bucket.unit, "mode": core.mode, **fill,
        }
        self.store.open_position(rec, now)
        row["decision"] = "bought" if core.mode == "live" else "paper bought"
        row["holding"] = True
        verb = "Bought" if core.mode == "live" else "Paper buy"
        log.info("%s No on %s at %.2f ($%s): forecast %s %s°C, models within %s°C", verb, bucket.question, rec["price"], rec["cost"],
                 row["kind"], row["forecast"], row["spread"])
        core.notifier.send(f"Polybot: {verb.lower()}",
                           f"[forecast] {row['city']}: No on {bucket.label} for {row['day']} at {rec['price']:.2f} (${rec['cost']:.2f})\n"
                           f"forecast {row['kind']} {row['forecast']}°C (models within {row['spread']}°C), target {row['target']}°{bucket.unit}",
                           tags="partly_sunny")
        return True

    def manage_positions(self, now: datetime) -> None:
        """Sell if a fresh forecast has climbed into the bucket we bet against (before the day there)."""
        if not self.cfg.fc_exit_if_forecast_moves:
            return
        for pos in self.store.open_positions(self.name):
            try:
                day = date.fromisoformat(pos.get("day", ""))
            except Exception:
                continue
            st = self.stations.get(pos.get("city", ""))
            if not st or local_now(st["tz"], now).date() >= day:
                continue   # the day has started: let it resolve
            fc = self.forecast_for(pos["city"], day, now)
            if fc is None:
                continue
            agg = fc.agg(pos.get("kind", "highest"), self.cfg.fc_agg)
            if agg is None:
                continue
            unit = pos.get("unit", "C")
            now_target = target_value(agg, 0, unit)
            edge = Decimal(str(pos["target"])) - Decimal(str(round(now_target, 1)))   # how far the forecast still is from our bucket
            if pos.get("kind", "highest") == "lowest":
                edge = -edge
            if edge > Decimal(str(self.cfg.fc_plus_c)) * Decimal("0.5"):
                continue   # still comfortably away
            shares = Decimal(str(pos.get("shares", 0)))
            est = self.core.estimate(pos["asset_id"], "SELL", shares=shares)
            if est is None:
                continue
            m = self.core.market(pos["condition_id"], pos.get("slug", ""))
            fill = self.core.sell(pos["asset_id"], shares, est, pos.get("title", ""), fee_rate=self.core.fee_rate(m) if m else Decimal(0))
            if fill:
                rec = self.store.close_position(pos["asset_id"], Decimal(str(fill["price"])), Decimal(str(fill["proceeds"])), "forecast moved", now,
                                                mode=self.core.mode, strategy=self.name)
                log.info("sold No on %s: forecast now %.1f, P&L $%.2f", pos.get("title"), now_target, rec["pnl"])
                self.core.notifier.send("Polybot: forecast moved",
                                        f"[forecast] {pos.get('city')}: sold No on {pos.get('bucket')} at {fill['price']:.2f}, forecast now {now_target:.1f}°{unit}, P&L ${rec['pnl']:+.2f}",
                                        tags="cloud")

    # ── the page ─────────────────────────────────────────────
    def status(self) -> dict:
        # Tomorrow's cities first (where a decision is still live), today's at the end.
        rows = sorted(self.rows, key=lambda r: ((r.get("decision") or "").startswith("day already"), r.get("day", ""), r.get("city", "")))
        return {"enabled": self.cfg.fc_enabled, "plus_c": self.cfg.fc_plus_c, "agg": self.cfg.fc_agg, "models": self.cfg.fc_models.split(","),
                "rows": rows[:100], "events": len(self.events)}
