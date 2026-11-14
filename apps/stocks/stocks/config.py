"""Settings, all from the environment. The defaults are the declared design (DESIGN.md): change one and the
model it touches starts its record over (`python -m stocks reset <model>`)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _s(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _f(name: str, default: float) -> float:
    try:
        return float(_s(name) or default)
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(float(_s(name) or default))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    v = _s(name).lower()
    return default if not v else v in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    data_dir: Path            # the record: decisions, ledger, fitted models, reports (small, backed up)
    cache_dir: Path           # market data and feature tables (large, can be fetched again)
    port: int

    # keys; each one missing switches off the part that needs it and the page says so
    alpaca_key: str
    alpaca_secret: str
    anthropic_key: str
    sec_contact: str          # SEC asks every script to name a contact in its User-Agent
    ntfy_url: str
    ntfy_token: str

    # the lab
    start_cash: float         # pretend money per model
    universe_size: int        # the N most traded common stocks, re-picked each month
    min_price: float
    history_start: str
    index_symbol: str
    cash_symbol: str          # what "out of the market" holds (a T-bill fund)
    models: tuple[str, ...]   # which models run; empty = all

    # clock, New York time
    run_at: str               # the evening run: after the close plus the 15 minutes free data is held back
    notify: str               # "daily" | "trades" | "off"

    # costs, in basis points of the traded value (book.py)
    slip_bps: float
    impact_coef: float
    fx_fee_bps: float         # shown in reports only: a Swedish broker's currency fee per trade

    # the agent
    agent_cap_usd: float      # hard stop per calendar month
    agent_model: str
    agent_small_model: str
    agent_deep_reads: int     # most companies read in depth per day
    agent_searches: int       # most web searches per day
    agent_effort: str
    fast: bool                # tests only: one seed, three epochs, no ablations

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            data_dir=Path(_s("DATA_DIR", "/data")),
            cache_dir=Path(_s("CACHE_DIR", "/cache")),
            port=_i("PORT", 8000),
            alpaca_key=_s("ALPACA_API_KEY"),
            alpaca_secret=_s("ALPACA_API_SECRET"),
            anthropic_key=_s("STOCKS_ANTHROPIC_API_KEY"),
            sec_contact=_s("SEC_CONTACT"),
            ntfy_url=_s("NTFY_URL"),
            ntfy_token=_s("NTFY_TOKEN"),
            start_cash=_f("STK_START_CASH", 10_000.0),
            universe_size=_i("STK_UNIVERSE", 1000),
            min_price=_f("STK_MIN_PRICE", 5.0),
            history_start=_s("STK_HISTORY_START", "2016-01-01"),
            index_symbol=_s("STK_INDEX", "SPY"),
            cash_symbol=_s("STK_CASH", "BIL"),
            models=tuple(m for m in _s("STK_MODELS").replace(",", " ").split() if m),
            run_at=_s("STK_RUN_AT", "16:25"),
            notify=_s("STK_NOTIFY", "daily").lower(),
            slip_bps=_f("STK_SLIP_BPS", 1.0),
            impact_coef=_f("STK_IMPACT", 1.0),
            fx_fee_bps=_f("STK_FX_FEE_BPS", 25.0),
            agent_cap_usd=_f("STK_AGENT_CAP_USD", 200.0),
            agent_model=_s("STK_AGENT_MODEL", "claude-opus-5-5"),
            agent_small_model=_s("STK_AGENT_SMALL_MODEL", "claude-haiku-5-5"),
            agent_deep_reads=_i("STK_AGENT_DEEP_READS", 30),
            agent_searches=_i("STK_AGENT_SEARCHES", 30),
            agent_effort=_s("STK_AGENT_EFFORT", "high"),
            fast=_b("STK_FAST", False),
        )

    # what is switched on
    @property
    def has_market_data(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def has_agent(self) -> bool:
        return bool(self.anthropic_key)

    @property
    def has_sec(self) -> bool:
        return bool(self.sec_contact)

    def keys(self) -> dict:
        """For the page: which keys are in place. Never the keys themselves."""
        return {"alpaca": self.has_market_data, "anthropic": self.has_agent, "sec": self.has_sec,
                "ntfy": bool(self.ntfy_url)}
