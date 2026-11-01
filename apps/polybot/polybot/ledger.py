"""Every qualifying signal at its real price, bought or not: settle the ledger and compare the setups.

The budget lets a setup buy about ten signals a day, so its paper book needs months to tell a few percent from luck.
The ledger (written by the consensus setups, `Store.ledger_add`) keeps every signal that passed a setup's rules at the
price a $5 market buy would have paid at that moment. Settled at resolution, it gives each setup's return per signal
on the whole population, at real prices, with no hindsight. `python -m polybot ledger [base]` prints the table.
"""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime


def signal_return(price: float, final: float, fee_rate: float) -> float:
    """Return per dollar of a buy at `price` held to resolution, after the taker fee (rate × (1 − price) per dollar)."""
    return final / price - 1 - fee_rate * (1 - price)


def settle(rows: list[dict], finals: dict[str, float]) -> list[dict]:
    """The rows whose outcome has resolved (finals: asset id → 1 or 0), each with its return."""
    out = []
    for r in rows:
        f = finals.get(r["asset_id"])
        if f is None or not r.get("price"):
            continue
        out.append({**r, "final": f, "ret": signal_return(float(r["price"]), f, float(r.get("fee_rate") or 0))})
    return out


def _boot(per_day: dict[str, list[float]], n: int = 2000, seed: int = 7) -> tuple[float, float]:
    """90% interval of the mean return per signal, resampling days."""
    days = list(per_day)
    if len(days) < 3:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [days[rng.randrange(len(days))] for _ in days]
        tot = sum(sum(per_day[d]) for d in pick)
        cnt = sum(len(per_day[d]) for d in pick)
        means.append(tot / cnt)
    means.sort()
    return means[int(0.05 * n)], means[int(0.95 * n)]


def summarize(settled: list[dict], logged: list[dict], base: str = "day4") -> list[dict]:
    """Per setup: signals logged, resolved, mean return per signal [90% CI], the bought ones, and the gap to `base`
    on the days both had resolved signals ($ per day at $5 a signal, paired by day, 90% CI)."""
    n_logged = defaultdict(int)
    for r in logged:
        n_logged[r["strategy"]] += 1
    by = defaultdict(lambda: defaultdict(list))
    taken = defaultdict(list)
    for r in settled:
        day = r["t"][:10]
        by[r["strategy"]][day].append(r["ret"])
        if r.get("taken"):
            taken[r["strategy"]].append(r["ret"])
    out = []
    for name in sorted(n_logged, key=lambda k: (k != base, k)):
        per_day = by.get(name, {})
        rets = [x for v in per_day.values() for x in v]
        lo, hi = _boot(per_day) if rets else (float("nan"), float("nan"))
        row = {"setup": name, "logged": n_logged[name], "resolved": len(rets), "mean": sum(rets) / len(rets) if rets else float("nan"),
               "lo": lo, "hi": hi, "taken_n": len(taken[name]), "taken_mean": sum(taken[name]) / len(taken[name]) if taken[name] else float("nan")}
        if name != base and base in by and per_day:
            common = sorted(set(per_day) & set(by[base]))
            if len(common) >= 3:
                diffs = [5 * (sum(per_day[d]) - sum(by[base][d])) for d in common]
                rng = random.Random(11)
                bm = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(2000))
                row.update(gap_day=sum(diffs) / len(diffs), gap_lo=bm[100], gap_hi=bm[1899], gap_days=len(common))
        out.append(row)
    return out


def finals_from_markets(rows: list[dict], market_of) -> dict[str, float]:
    """Look each logged outcome's market up (market_of(condition_id, slug) → Gamma market or None) and return the
    resolved ones: asset id → 1.0 (won) or 0.0 (lost)."""
    finals: dict[str, float] = {}
    seen = set()
    for r in rows:
        key = (r["condition_id"], r.get("slug", ""))
        if key in seen:
            continue
        seen.add(key)
        try:
            m = market_of(*key)
        except Exception:
            continue
        if m is None or not m.state.closed:
            continue
        for o in (m.outcomes.yes, m.outcomes.no):
            if o.price is None:
                continue
            p = float(o.price)
            final = 1.0 if p >= 0.99 else 0.0 if p <= 0.01 else None
            if final is None:
                continue
            for ident in (o.token_id, o.position_id):
                if ident:
                    finals[str(ident)] = final
    return finals


def report(rows: list[dict], finals: dict[str, float], base: str = "day4", now: datetime | None = None) -> str:
    settled = settle(rows, finals)
    lines = [f"ledger: {len(rows)} signals logged, {len(settled)} resolved"
             + (f" (since {min(r['t'] for r in rows)[:10]})" if rows else ""),
             f"{'setup':14s} {'logged':>6s} {'resolved':>8s} {'per signal [90% CI]':>26s} {'bought':>6s} {'bought avg':>10s}  vs {base}: $/day at $5 a signal [90% CI]"]
    for r in summarize(settled, rows, base):
        ci = f"{100 * r['mean']:+6.1f}% [{100 * r['lo']:+5.1f}, {100 * r['hi']:+5.1f}]" if r["resolved"] else "—"
        gap = (f"{r['gap_day']:+6.2f} [{r['gap_lo']:+6.2f}, {r['gap_hi']:+6.2f}] over {r['gap_days']} days" if "gap_day" in r else "")
        tk = f"{100 * r['taken_mean']:+6.1f}%" if r["taken_n"] else "—"
        lines.append(f"{r['setup']:14s} {r['logged']:6d} {r['resolved']:8d} {ci:>26s} {r['taken_n']:6d} {tk:>10s}  {gap}")
    return "\n".join(lines)
