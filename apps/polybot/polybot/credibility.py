"""Who on a leaderboard is worth copying? Pure rules over a trader's track record.

A leaderboard ranks by profit, which rewards three kinds of trader we cannot
copy: in-game sports scalpers (their edge is gone minutes later), market
makers (their profit is the spread, not a view) and one-bet whales (one lucky
position). A credible trader is active now, has a real history of closed
positions with a good profit factor, wins often enough, is not carried by a
single position, holds long enough for a copy to matter, and earns a real
edge per dollar traded.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rules:
    min_fills_7d: int = 10       # active: fills in the last week…
    max_idle_hours: float = 72   # …and one recently
    min_closed: int = 30         # history: closed positions with real money in them
    min_profit_factor: float = 1.3
    min_win_rate: float = 0.40
    max_top_share: float = 0.50  # the best position may carry at most half the profit
    min_hold_hours: float = 12   # median hold; shorter is in-game trading
    min_edge: float = 0.02       # profit per dollar of volume; makers sit near zero
    min_age_days: float = 30
    max_two_sided: float = 0.30  # share of markets where they held both outcomes (hedgers, makers)


@dataclass
class Record:
    """A trader's measured track record (what pack.py collects)."""
    wallet: str
    name: str
    rank: int = 0
    pnl: float = 0.0             # leaderboard profit for the window
    volume: float = 0.0          # leaderboard volume for the window
    closed_n: int = 0            # closed positions with cost ≥ $1
    wins: int = 0
    losses: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0      # positive number
    sum_pnl: float = 0.0
    sum_cost: float = 0.0
    top_pnl: float = 0.0         # best single position
    median_hold_hours: float | None = None
    median_cost: float | None = None   # typical position size in USD
    fills_7d: int = 0
    taker_share: float | None = None
    hours_since_fill: float | None = None
    age_days: float | None = None
    top_category: str = ""
    top_category_share: float = 0.0
    two_sided_share: float = 0.0  # markets where they held both outcomes

    @property
    def win_rate(self) -> float:
        n = self.wins + self.losses
        return self.wins / n if n else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            return 99.0 if self.gross_win > 0 else 0.0
        return self.gross_win / self.gross_loss

    @property
    def return_on_cost(self) -> float:
        return self.sum_pnl / self.sum_cost if self.sum_cost > 0 else 0.0

    @property
    def top_share(self) -> float:
        return self.top_pnl / self.sum_pnl if self.sum_pnl > 0 else 1.0

    @property
    def edge(self) -> float:
        return self.pnl / self.volume if self.volume > 0 else 0.0


@dataclass
class Judgement:
    ok: bool
    score: float                 # 0..1 among the credible; 0 otherwise
    reasons: list[str] = field(default_factory=list)

    @property
    def weight(self) -> float:
        """A vote weight: 0.5 for a borderline trader, 1.5 for an excellent one."""
        return 0.5 + self.score if self.ok else 0.0


def judge(r: Record, rules: Rules = Rules()) -> Judgement:
    reasons: list[str] = []
    if r.fills_7d < rules.min_fills_7d or r.hours_since_fill is None or r.hours_since_fill > rules.max_idle_hours:
        reasons.append("not trading right now")
    if r.closed_n < rules.min_closed:
        reasons.append(f"only {r.closed_n} closed positions")
    if r.age_days is not None and r.age_days < rules.min_age_days:
        reasons.append(f"account {r.age_days:.0f} days old")
    if r.closed_n >= rules.min_closed:
        if r.profit_factor < rules.min_profit_factor:
            reasons.append(f"profit factor {r.profit_factor:.2f}")
        if r.win_rate < rules.min_win_rate:
            reasons.append(f"wins {r.win_rate:.0%} of positions")
        if r.top_share > rules.max_top_share:
            reasons.append(f"one position is {min(r.top_share, 9.99):.0%} of the profit")
        if r.median_hold_hours is not None and r.median_hold_hours < rules.min_hold_hours:
            reasons.append(f"holds {r.median_hold_hours:.0f} h (in-game trader)")
    if r.edge < rules.min_edge:
        reasons.append(f"edge {r.edge:.1%} of volume (market maker)")
    if r.taker_share is not None and r.taker_share < 0.2 and (r.median_hold_hours or 0) < 24:
        reasons.append("rests orders and flips them (market maker)")
    if r.closed_n >= rules.min_closed and r.two_sided_share > rules.max_two_sided:
        reasons.append(f"holds both sides in {r.two_sided_share:.0%} of markets (hedger)")
    if reasons:
        return Judgement(False, 0.0, reasons)
    pf = min(r.profit_factor, 5.0) / 5.0
    roc = min(max(r.return_on_cost, 0.0), 0.30) / 0.30
    history = min(r.closed_n, 200) / 200
    score = 0.35 * pf + 0.25 * r.win_rate + 0.20 * roc + 0.10 * (1 - r.top_share) + 0.10 * history
    return Judgement(True, round(min(max(score, 0.0), 1.0), 3), [])


def conviction(stance_usd: float, median_cost: float | None) -> float:
    """How big a bet is for this trader: 1 = their usual size, capped at 0.5–2."""
    if not median_cost or median_cost <= 0:
        return 1.0
    return min(2.0, max(0.5, abs(stance_usd) / median_cost))
