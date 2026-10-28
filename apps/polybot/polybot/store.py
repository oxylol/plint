"""On-disk state: open/closed positions per strategy, the trade log and the page's JSON."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
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


def pkey(strategy: str, asset_id: str) -> str:
    """Positions are per strategy: two strategies may hold the same outcome."""
    return f"{strategy}|{asset_id}"


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
    KEEP_TRADES = 500

    def __init__(self, data_dir: str):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.status_path = self.dir / "status.json"
        self.log_path = self.dir / "trades.jsonl"
        self.state = self._load()

    def _load(self) -> dict:
        default = {"positions": {}, "closed": [], "acted": {}, "trades": [], "leaders": {}, "since": {}}
        if not self.state_path.exists():
            return default
        try:
            data = json.loads(self.state_path.read_text())
            for k, v in default.items():
                data.setdefault(k, v)
        except Exception as e:  # a damaged file must not stop the bot
            log.error("state.json unreadable (%s); starting fresh, old file kept as state.json.bad", e)
            self.state_path.replace(self.dir / "state.json.bad")
            return default
        # Older files keyed positions by asset id alone; make them per strategy.
        for k in list(data["positions"]):
            if "|" not in k:
                rec = data["positions"].pop(k)
                rec.setdefault("strategy", "consensus")
                data["positions"][pkey(rec["strategy"], k)] = rec
        for k in list(data["acted"]):
            if "|" not in k:
                v = data["acted"].pop(k)
                data["acted"][pkey("consensus", k)] = v
                data["acted"][pkey("underdog", k)] = v
        return data

    def save(self) -> None:
        _atomic_write(self.state_path, json.dumps(self.state, indent=1, default=_json_default))

    # ---- positions ----
    def position(self, asset_id: str, strategy: str) -> dict | None:
        return self.state["positions"].get(pkey(strategy, asset_id))

    def positions_for_asset(self, asset_id: str) -> list[dict]:
        return [p for p in self.state["positions"].values() if p.get("asset_id") == asset_id]

    def open_positions(self, strategy: str | None = None) -> list[dict]:
        return [p for p in self.state["positions"].values() if strategy is None or p.get("strategy", "consensus") == strategy]

    def find_position(self, condition_id: str, outcome: str, strategy: str) -> dict | None:
        for p in self.open_positions(strategy):
            if p.get("condition_id") == condition_id and str(p.get("outcome", "")).lower() == (outcome or "").lower():
                return p
        return None

    def acted_at(self, asset_id: str, strategy: str) -> datetime | None:
        s = self.state["acted"].get(pkey(strategy, asset_id))
        return parse_iso(s) if s else None

    def mark_acted(self, asset_id: str, strategy: str, now: datetime) -> None:
        self.state["acted"][pkey(strategy, asset_id)] = iso(now)

    def spent_since(self, since: datetime, strategy: str | None = None) -> Decimal:
        total = Decimal(0)
        for t in self.state["trades"]:
            if strategy is not None and t.get("strategy", "consensus") != strategy:
                continue
            if t.get("side") == "BUY" and parse_iso(t["at"]) >= since:
                total += Decimal(str(t.get("cost", 0)))
        return total

    def open_position(self, rec: dict, now: datetime) -> dict:
        rec = dict(rec)
        rec["opened_at"] = iso(now)
        rec.setdefault("strategy", "consensus")
        rec.setdefault("current_price", rec.get("price"))
        self.state["positions"][pkey(rec["strategy"], rec["asset_id"])] = rec
        self.mark_acted(rec["asset_id"], rec["strategy"], now)
        self._log({**rec, "at": iso(now)})
        self.save()
        return rec

    def close_position(self, asset_id: str, price: Decimal, proceeds: Decimal, reason: str, now: datetime,
                       mode: str | None = None, strategy: str = "consensus") -> dict | None:
        pos = self.state["positions"].pop(pkey(strategy, asset_id), None)
        if pos is None:
            return None
        cost = Decimal(str(pos.get("cost", 0)))
        rec = {
            "side": "SELL", "reason": reason, "asset_id": asset_id, "condition_id": pos.get("condition_id"),
            "title": pos.get("title"), "outcome": pos.get("outcome"), "slug": pos.get("slug"),
            "event_slug": pos.get("event_slug"), "shares": pos.get("shares"), "price": float(price),
            "proceeds": float(proceeds), "cost": float(cost), "pnl": float(proceeds - cost),
            "opened_at": pos.get("opened_at"), "closed_at": iso(now), "mode": mode or pos.get("mode", "paper"),
            "strategy": strategy,
        }
        self.state["closed"].append(rec)
        self.state["closed"] = self.state["closed"][-2000:]
        self.mark_acted(asset_id, strategy, now)
        self._log({**rec, "at": iso(now)})
        for w in pos.get("agree_wallets") or []:
            self.record_follow(strategy, w, pos.get("agree_names", {}).get(w, ""), rec["pnl"])
        self.save()
        return rec

    # ---- starting over ----
    def reset_strategy(self, strategy: str, now: datetime) -> dict:
        """Forget one setup's book so its record starts fresh (positions, trades, closed, cooldowns, leader stats)."""
        st = self.state
        before = {"positions": len(self.open_positions(strategy)), "trades": sum(1 for t in st["trades"] if t.get("strategy") == strategy),
                  "closed": sum(1 for c in st["closed"] if c.get("strategy") == strategy)}
        st["positions"] = {k: v for k, v in st["positions"].items() if v.get("strategy", "consensus") != strategy}
        st["trades"] = [t for t in st["trades"] if t.get("strategy", "consensus") != strategy]
        st["closed"] = [c for c in st["closed"] if c.get("strategy", "consensus") != strategy]
        st["acted"] = {k: v for k, v in st["acted"].items() if not k.startswith(strategy + "|")}
        st.get("leaders", {}).pop(strategy, None)
        st.setdefault("since", {})[strategy] = iso(now)
        self._log({"strategy": strategy, "side": "RESET", "at": iso(now), "note": "book cleared, record starts here"})
        self.save()
        return before

    def since(self, strategy: str) -> str | None:
        return self.state.get("since", {}).get(strategy)

    # ---- who was worth following ----
    def leader(self, strategy: str, wallet: str) -> dict:
        return self.state["leaders"].setdefault(strategy, {}).setdefault(wallet, {"name": "", "follows": 0, "wins": 0, "pnl": 0.0, "trust": None})

    def trust(self, strategy: str, wallet: str, default: float) -> float:
        t = self.state["leaders"].get(strategy, {}).get(wallet, {}).get("trust")
        return float(t) if t is not None else default

    def record_follow(self, strategy: str, wallet: str, name: str, pnl: float, default_trust: float = 1.0) -> None:
        """A copied position closed: remember how the leader did, and trust them a little more or less."""
        entry = self.leader(strategy, wallet)
        if name:
            entry["name"] = name
        entry["follows"] += 1
        entry["wins"] += 1 if pnl > 0 else 0
        entry["pnl"] = round(entry["pnl"] + pnl, 4)
        trust = entry["trust"] if entry["trust"] is not None else default_trust
        trust = trust * (1.1 if pnl > 0 else 0.8)
        entry["trust"] = round(min(2.0, max(0.2, trust)), 3)

    def leaders_followed(self, strategy: str) -> list[dict]:
        rows = [{"wallet": w, **v} for w, v in self.state["leaders"].get(strategy, {}).items()]
        rows.sort(key=lambda r: (-r["pnl"], -r["follows"]))
        return rows

    def update_price(self, asset_id: str, price: Decimal) -> None:
        for pos in self.positions_for_asset(asset_id):
            pos["current_price"] = float(price)
            pos["value"] = float(Decimal(str(pos.get("shares", 0))) * price)
            pos["priced_at"] = iso(utcnow())

    def note_position(self, asset_id: str, note: str, strategy: str) -> None:
        pos = self.position(asset_id, strategy)
        if pos is not None:
            pos["note"] = note

    def update_position(self, asset_id: str, strategy: str, **fields) -> None:
        pos = self.position(asset_id, strategy)
        if pos is not None:
            pos.update(fields)

    # ---- trades ----
    def _log(self, rec: dict) -> None:
        if rec.get("side") == "RESET":   # only the file remembers resets; the in-memory list is trades
            try:
                with self.log_path.open("a") as f:
                    f.write(json.dumps(rec, default=_json_default) + "\n")
            except OSError:
                pass
            return
        self.state["trades"].append(rec)
        self.state["trades"] = self.state["trades"][-self.KEEP_TRADES:]
        try:
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        except OSError as e:
            log.warning("could not append to trades.jsonl: %s", e)

    def recent_trades(self, n: int = 30, strategy: str | None = None) -> list[dict]:
        rows = [t for t in self.state["trades"] if strategy is None or t.get("strategy", "consensus") == strategy]
        return list(reversed(rows[-n:]))

    def totals(self, now: datetime | None = None, strategy: str | None = None) -> dict:
        now = now or utcnow()
        open_cost = Decimal(0)
        open_value = Decimal(0)
        positions = self.open_positions(strategy)
        for p in positions:
            open_cost += Decimal(str(p.get("cost", 0)))
            shares = Decimal(str(p.get("shares", 0)))
            price = Decimal(str(p.get("current_price") or p.get("price") or 0))
            open_value += shares * price
        closed = [c for c in self.state["closed"] if strategy is None or c.get("strategy", "consensus") == strategy]
        realized = sum((Decimal(str(c.get("pnl", 0))) for c in closed), Decimal(0))
        wins = sum(1 for c in closed if float(c.get("pnl", 0)) > 0)
        losses = sum(1 for c in closed if float(c.get("pnl", 0)) < 0)
        returns = [float(c["pnl"]) / float(c["cost"]) for c in closed if float(c.get("cost", 0)) > 0]
        return {
            "open": len(positions), "closed": len(closed), "wins": wins, "losses": losses,
            "win_rate": (wins / len(closed)) if closed else None,
            "avg_return": (sum(returns) / len(returns)) if returns else None,
            "open_cost": float(open_cost), "open_value": float(open_value),
            "unrealized": float(open_value - open_cost), "realized": float(realized),
            "total": float(open_value - open_cost + realized),
            "spent_24h": float(self.spent_since(now - timedelta(hours=24), strategy)),
        }

    # ---- status for the page ----
    def write_status(self, status: dict) -> None:
        _atomic_write(self.status_path, json.dumps(status, default=_json_default))

    def read_status(self) -> dict | None:
        try:
            return json.loads(self.status_path.read_text())
        except Exception:
            return None
