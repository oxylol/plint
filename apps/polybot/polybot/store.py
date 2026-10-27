"""On-disk state: open/closed positions, the trade log and the status page's JSON."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

log = logging.getLogger("polybot.store")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, datetime):
        return iso(o)
    return str(o)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)  # mkstemp makes it private; backups and `cat` should see it
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class Store:
    KEEP_TRADES = 300

    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.status_path = self.dir / "status.json"
        self.log_path = self.dir / "trades.jsonl"
        self.state = self._load()

    def _load(self) -> dict:
        default = {"positions": {}, "closed": [], "acted": {}, "trades": []}
        if not self.state_path.exists():
            return default
        try:
            data = json.loads(self.state_path.read_text())
            for k, v in default.items():
                data.setdefault(k, v)
            return data
        except Exception as e:  # a damaged file must not stop the bot
            log.error("state.json unreadable (%s); starting fresh, old file kept as state.json.bad", e)
            self.state_path.replace(self.dir / "state.json.bad")
            return default

    def save(self) -> None:
        _atomic_write(self.state_path, json.dumps(self.state, indent=1, default=_json_default))

    # ---- positions ----
    def position(self, asset_id: str) -> dict | None:
        return self.state["positions"].get(asset_id)

    def open_positions(self) -> list[dict]:
        return list(self.state["positions"].values())

    def find_position(self, condition_id: str, outcome: str) -> dict | None:
        for p in self.state["positions"].values():
            if p.get("condition_id") == condition_id and str(p.get("outcome", "")).lower() == (outcome or "").lower():
                return p
        return None

    def acted_at(self, asset_id: str) -> datetime | None:
        s = self.state["acted"].get(asset_id)
        return parse_iso(s) if s else None

    def spent_since(self, since: datetime) -> Decimal:
        total = Decimal(0)
        for t in self.state["trades"]:
            if t.get("side") == "BUY" and parse_iso(t["at"]) >= since:
                total += Decimal(str(t.get("cost", 0)))
        return total

    def open_position(self, rec: dict, now: datetime) -> dict:
        rec = dict(rec)
        rec["opened_at"] = iso(now)
        rec.setdefault("current_price", rec.get("price"))
        self.state["positions"][rec["asset_id"]] = rec
        self.state["acted"][rec["asset_id"]] = iso(now)
        self._log({**rec, "at": iso(now)})
        self.save()
        return rec

    def close_position(self, asset_id: str, price: Decimal, proceeds: Decimal, reason: str, now: datetime,
                       mode: str | None = None) -> dict | None:
        pos = self.state["positions"].pop(asset_id, None)
        if pos is None:
            return None
        cost = Decimal(str(pos.get("cost", 0)))
        rec = {
            "side": "SELL", "reason": reason, "asset_id": asset_id, "condition_id": pos.get("condition_id"),
            "title": pos.get("title"), "outcome": pos.get("outcome"), "slug": pos.get("slug"),
            "event_slug": pos.get("event_slug"), "shares": pos.get("shares"), "price": float(price),
            "proceeds": float(proceeds), "cost": float(cost), "pnl": float(proceeds - cost),
            "opened_at": pos.get("opened_at"), "closed_at": iso(now), "mode": mode or pos.get("mode", "paper"),
        }
        self.state["closed"].append(rec)
        self.state["closed"] = self.state["closed"][-500:]
        self.state["acted"][asset_id] = iso(now)
        self._log({**rec, "at": iso(now)})
        self.save()
        return rec

    def update_price(self, asset_id: str, price: Decimal) -> None:
        pos = self.state["positions"].get(asset_id)
        if pos is None:
            return
        pos["current_price"] = float(price)
        pos["value"] = float(Decimal(str(pos.get("shares", 0))) * price)
        pos["priced_at"] = iso(utcnow())

    def note_position(self, asset_id: str, note: str) -> None:
        pos = self.state["positions"].get(asset_id)
        if pos is not None:
            pos["note"] = note

    # ---- trades ----
    def _log(self, rec: dict) -> None:
        self.state["trades"].append(rec)
        self.state["trades"] = self.state["trades"][-self.KEEP_TRADES:]
        try:
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        except OSError as e:
            log.warning("could not append to trades.jsonl: %s", e)

    def recent_trades(self, n: int = 30) -> list[dict]:
        return list(reversed(self.state["trades"][-n:]))

    def totals(self, now: datetime | None = None) -> dict:
        now = now or utcnow()
        open_cost = Decimal(0)
        open_value = Decimal(0)
        for p in self.state["positions"].values():
            open_cost += Decimal(str(p.get("cost", 0)))
            shares = Decimal(str(p.get("shares", 0)))
            price = Decimal(str(p.get("current_price") or p.get("price") or 0))
            open_value += shares * price
        closed = self.state["closed"]
        realized = sum((Decimal(str(c.get("pnl", 0))) for c in closed), Decimal(0))
        wins = sum(1 for c in closed if float(c.get("pnl", 0)) > 0)
        losses = sum(1 for c in closed if float(c.get("pnl", 0)) < 0)
        from datetime import timedelta
        return {
            "open": len(self.state["positions"]), "closed": len(closed), "wins": wins, "losses": losses,
            "open_cost": float(open_cost), "open_value": float(open_value),
            "unrealized": float(open_value - open_cost), "realized": float(realized),
            "total": float(open_value - open_cost + realized),
            "spent_24h": float(self.spent_since(now - timedelta(hours=24))),
        }

    # ---- status for the page ----
    def write_status(self, status: dict) -> None:
        _atomic_write(self.status_path, json.dumps(status, default=_json_default))

    def read_status(self) -> dict | None:
        try:
            return json.loads(self.status_path.read_text())
        except Exception:
            return None
