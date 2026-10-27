"""Configuration from environment variables (all POLY_*; see README.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from decimal import Decimal

WINDOWS = ("day", "week", "month", "all")


def _str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    return v.strip() if v.strip() else default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    return int(_str(name, str(default)))


def _dec(name: str, default: str) -> Decimal:
    return Decimal(_str(name, default))


@dataclass(frozen=True)
class Config:
    # Account. Leave the key empty to paper trade with public data only.
    private_key: str
    wallet: str
    relayer_key: str
    relayer_address: str
    live: bool
    # Who to follow
    window: str          # leaderboard window: day | week | month | all
    category: str        # overall, or a Polymarket category such as politics / sports
    top_n: int           # how many leaders to watch
    # What counts as agreement
    min_agree: int       # distinct leaders net-buying the same outcome (minus dissenters)
    lookback_hours: float
    min_leader_usd: Decimal   # a leader's net flow below this is noise
    # How to trade
    stake_usd: Decimal
    max_price: Decimal
    min_price: Decimal
    max_above_leaders: Decimal  # don't pay more than this above the leaders' average entry
    max_open: int
    daily_budget_usd: Decimal
    cooldown_hours: float
    min_hours_to_end: float
    follow_sells: bool
    auto_redeem: bool
    poll_seconds: int
    # Plumbing
    data_dir: str
    port: int
    ntfy_url: str
    ntfy_token: str
    tz: str

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            private_key=_str("POLY_PRIVATE_KEY", ""),
            wallet=_str("POLY_WALLET", ""),
            relayer_key=_str("POLY_RELAYER_API_KEY", ""),
            relayer_address=_str("POLY_RELAYER_API_KEY_ADDRESS", ""),
            live=_bool("POLY_LIVE", False),
            window=_str("POLY_WINDOW", "week").lower(),
            category=_str("POLY_CATEGORY", "overall").lower(),
            top_n=_int("POLY_TOP_N", 25),
            min_agree=_int("POLY_MIN_AGREE", 5),
            lookback_hours=float(_str("POLY_LOOKBACK_HOURS", "24")),
            min_leader_usd=_dec("POLY_MIN_LEADER_USD", "25"),
            stake_usd=_dec("POLY_STAKE_USD", "5"),
            max_price=_dec("POLY_MAX_PRICE", "0.90"),
            min_price=_dec("POLY_MIN_PRICE", "0.05"),
            max_above_leaders=_dec("POLY_MAX_ABOVE_LEADERS", "0.05"),
            max_open=_int("POLY_MAX_OPEN", 20),
            daily_budget_usd=_dec("POLY_DAILY_BUDGET_USD", "50"),
            cooldown_hours=float(_str("POLY_COOLDOWN_HOURS", "72")),
            min_hours_to_end=float(_str("POLY_MIN_HOURS_TO_END", "1")),
            follow_sells=_bool("POLY_FOLLOW_SELLS", True),
            auto_redeem=_bool("POLY_AUTO_REDEEM", True),
            poll_seconds=_int("POLY_POLL_SECONDS", 300),
            data_dir=_str("DATA_DIR", "/data"),
            port=_int("PORT", 8000),
            ntfy_url=_str("NTFY_URL", ""),
            ntfy_token=_str("NTFY_TOKEN", ""),
            tz=_str("TZ", "UTC"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.window not in WINDOWS:
            raise ValueError(f"POLY_WINDOW must be one of {WINDOWS}, not {self.window!r}")
        if self.top_n < 1 or self.top_n > 1000:
            raise ValueError("POLY_TOP_N must be between 1 and 1000")
        if self.min_agree < 1:
            raise ValueError("POLY_MIN_AGREE must be at least 1")
        if self.stake_usd < 1:
            raise ValueError("POLY_STAKE_USD must be at least 1")
        if not (0 < self.min_price < self.max_price <= 1):
            raise ValueError("need 0 < POLY_MIN_PRICE < POLY_MAX_PRICE <= 1")
        if self.poll_seconds < 30:
            raise ValueError("POLY_POLL_SECONDS must be at least 30")

    @property
    def can_trade(self) -> bool:
        return bool(self.private_key and self.wallet)

    @property
    def mode(self) -> str:
        return "live" if (self.live and self.can_trade) else "paper"

    def public(self) -> dict:
        """The non-secret settings, for the status page."""
        skip = {"private_key", "relayer_key", "relayer_address", "ntfy_token", "ntfy_url", "data_dir", "port"}
        out = {}
        for f in fields(self):
            if f.name in skip:
                continue
            v = getattr(self, f.name)
            out[f.name] = float(v) if isinstance(v, Decimal) else v
        out["wallet"] = (self.wallet[:6] + "…" + self.wallet[-4:]) if self.wallet else ""
        out["mode"] = self.mode
        out["has_key"] = bool(self.private_key)
        out["has_relayer"] = bool(self.relayer_key and self.relayer_address)
        return out
