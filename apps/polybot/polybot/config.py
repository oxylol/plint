"""Configuration from environment variables (all POLY_*; see README.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from decimal import Decimal

WINDOWS = ("day", "week", "month", "all")

# Settings a consensus variant may override (the POLY_* name without the prefix).
VARIANT_KEYS = ("label", "window", "category", "top_n", "min_agree", "lookback_hours", "min_leader_usd", "stake_usd",
                "max_price", "min_price", "max_above_leaders", "max_open", "daily_budget_usd", "cooldown_hours",
                "min_hours_to_end", "follow_sells", "poll_seconds",
                # credible-pack options
                "pack", "pool", "min_weight", "fresh_only", "adaptive_trust", "max_hold_days", "max_per_cycle")
DEFAULT_VARIANTS = ";".join([
    "week4:min_agree=4",
    "month:window=month", "month4:window=month,min_agree=4",
    "day:window=day", "day4:window=day,min_agree=4",
    "all:window=all", "all4:window=all,min_agree=4",
    # only credible, active traders from the monthly board; two of them agreeing (weighted) is enough
    "credible:pack=1,window=month,min_agree=2,min_weight=2.5,lookback_hours=48,min_hours_to_end=12,max_per_cycle=3,label=Credible pack",
    # a single credible trader's fresh conviction bet, with trust that adapts to how their copies do
    "smart:pack=1,window=month,min_agree=1,min_weight=1.5,fresh_only=1,adaptive_trust=1,lookback_hours=48,min_hours_to_end=12,max_per_cycle=3,label=Smart money",
])
RESERVED_NAMES = ("consensus", "underdog")


def parse_variants(raw: str) -> list[tuple[str, dict[str, str]]]:
    """`month:window=month;day:window=day,min_agree=4` → [("month", {...}), ("day", {...})].

    Each variant is a name, optionally followed by a colon and comma-separated
    overrides; everything else comes from the base POLY_* settings.
    """
    out: list[tuple[str, dict[str, str]]] = []
    seen: set[str] = set()
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, rest = chunk.partition(":")
        name = name.strip().lower()
        if not name or name in RESERVED_NAMES or name in seen or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError(f"POLY_VARIANTS: bad or duplicate variant name {name!r}")
        overrides: dict[str, str] = {}
        for kv in rest.split(","):
            kv = kv.strip()
            if not kv:
                continue
            key, eq, val = kv.partition("=")
            key = key.strip().lower().removeprefix("poly_")
            if not eq or key not in VARIANT_KEYS:
                raise ValueError(f"POLY_VARIANTS: variant {name!r} has an unknown setting {kv!r}")
            overrides[key] = val.strip()
        out.append((name, overrides))
        seen.add(name)
    return out


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
    variants: str                # extra consensus setups to compare, see parse_variants()
    # Credibility rules for pack=1 setups (CR_*): who on the board is worth copying
    cr_pool: int                 # how many of the board to measure
    cr_pack_size: int            # how many credible traders to follow
    cr_refresh_hours: float
    cr_min_fills_7d: int
    cr_max_idle_hours: float
    cr_min_closed: int
    cr_min_profit_factor: float
    cr_min_win_rate: float
    cr_max_top_share: float
    cr_min_hold_hours: float
    cr_min_edge: float
    cr_min_age_days: float
    cr_max_two_sided: float
    # Underdog strategy: buy the cheaper side of a game at kickoff, sell into the early bounce
    ud_enabled: bool
    ud_tick_seconds: int
    ud_lookahead_hours: float    # games this far ahead show up on the page
    ud_buy_before_min: float     # buy from this many minutes before the start…
    ud_buy_until_min: float      # …until this many minutes after it
    ud_min_price: Decimal
    ud_max_price: Decimal
    ud_max_spread: Decimal       # skip when ask minus bid is wider than this
    ud_min_liquidity: Decimal    # skip thin markets (Polymarket's liquidity figure)
    ud_stake_usd: Decimal
    ud_take_profit_pct: Decimal  # sell when the price is this much above entry
    ud_stop_loss_pct: Decimal    # or this much below
    ud_max_hold_min: float       # or this many minutes after the start, whatever comes first
    ud_max_open: int
    ud_daily_budget_usd: Decimal
    ud_sports: str               # comma-separated slug prefixes (nba,nhl,epl…); empty = every sport
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
            variants=_str("POLY_VARIANTS", DEFAULT_VARIANTS),
            cr_pool=_int("CR_POOL", 80),
            cr_pack_size=_int("CR_PACK_SIZE", 12),
            cr_refresh_hours=float(_str("CR_REFRESH_HOURS", "6")),
            cr_min_fills_7d=_int("CR_MIN_FILLS_7D", 10),
            cr_max_idle_hours=float(_str("CR_MAX_IDLE_HOURS", "72")),
            cr_min_closed=_int("CR_MIN_CLOSED", 30),
            cr_min_profit_factor=float(_str("CR_MIN_PROFIT_FACTOR", "1.3")),
            cr_min_win_rate=float(_str("CR_MIN_WIN_RATE", "0.40")),
            cr_max_top_share=float(_str("CR_MAX_TOP_SHARE", "0.50")),
            cr_min_hold_hours=float(_str("CR_MIN_HOLD_HOURS", "12")),
            cr_min_edge=float(_str("CR_MIN_EDGE", "0.02")),
            cr_min_age_days=float(_str("CR_MIN_AGE_DAYS", "30")),
            cr_max_two_sided=float(_str("CR_MAX_TWO_SIDED", "0.30")),
            ud_enabled=_bool("UD_ENABLED", True),
            ud_tick_seconds=_int("UD_TICK_SECONDS", 30),
            ud_lookahead_hours=float(_str("UD_LOOKAHEAD_HOURS", "3")),
            ud_buy_before_min=float(_str("UD_BUY_BEFORE_MIN", "3")),
            ud_buy_until_min=float(_str("UD_BUY_UNTIL_MIN", "2")),
            ud_min_price=_dec("UD_MIN_PRICE", "0.10"),
            ud_max_price=_dec("UD_MAX_PRICE", "0.45"),
            ud_max_spread=_dec("UD_MAX_SPREAD", "0.03"),
            ud_min_liquidity=_dec("UD_MIN_LIQUIDITY", "5000"),
            ud_stake_usd=_dec("UD_STAKE_USD", "5"),
            ud_take_profit_pct=_dec("UD_TAKE_PROFIT_PCT", "15"),
            ud_stop_loss_pct=_dec("UD_STOP_LOSS_PCT", "30"),
            ud_max_hold_min=float(_str("UD_MAX_HOLD_MIN", "30")),
            ud_max_open=_int("UD_MAX_OPEN", 6),
            ud_daily_budget_usd=_dec("UD_DAILY_BUDGET_USD", "50"),
            ud_sports=_str("UD_SPORTS", "").lower().replace(" ", ""),
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
        parse_variants(self.variants)
        if not (1 <= self.cr_pack_size <= self.cr_pool <= 1000) or self.cr_refresh_hours < 1:
            raise ValueError("need 1 <= CR_PACK_SIZE <= CR_POOL <= 1000 and CR_REFRESH_HOURS >= 1")
        if self.ud_tick_seconds < 10:
            raise ValueError("UD_TICK_SECONDS must be at least 10")
        if not (0 < self.ud_min_price < self.ud_max_price < 1):
            raise ValueError("need 0 < UD_MIN_PRICE < UD_MAX_PRICE < 1")
        if self.ud_stake_usd < 1 or self.ud_take_profit_pct <= 0 or self.ud_stop_loss_pct <= 0:
            raise ValueError("UD_STAKE_USD, UD_TAKE_PROFIT_PCT and UD_STOP_LOSS_PCT must be positive")

    def rules(self):
        from .credibility import Rules
        return Rules(min_fills_7d=self.cr_min_fills_7d, max_idle_hours=self.cr_max_idle_hours, min_closed=self.cr_min_closed,
                     min_profit_factor=self.cr_min_profit_factor, min_win_rate=self.cr_min_win_rate,
                     max_top_share=self.cr_max_top_share, min_hold_hours=self.cr_min_hold_hours,
                     min_edge=self.cr_min_edge, min_age_days=self.cr_min_age_days, max_two_sided=self.cr_max_two_sided)

    @property
    def can_trade(self) -> bool:
        return bool(self.private_key and self.wallet)

    @property
    def mode(self) -> str:
        return "live" if (self.live and self.can_trade) else "paper"

    def public(self) -> dict:
        """The non-secret settings, for the status page."""
        skip = {"private_key", "relayer_key", "relayer_address", "ntfy_token", "ntfy_url", "data_dir", "port", "variants"}
        out = {}
        for f in fields(self):
            if f.name in skip:
                continue
            v = getattr(self, f.name)
            out[f.name] = float(v) if isinstance(v, Decimal) else v
        out["wallet"] = (self.wallet[:6] + "…" + self.wallet[-4:]) if self.wallet else ""
        out["mode"] = self.mode
        out["sports"] = [x for x in self.ud_sports.split(",") if x]
        out["has_key"] = bool(self.private_key)
        out["has_relayer"] = bool(self.relayer_key and self.relayer_address)
        return out
