"""The figures a book is judged by. One set of definitions for the backtest report and the live page."""
from __future__ import annotations

import numpy as np

YEAR = 252.0


def max_drawdown(equity: np.ndarray) -> float:
    """The worst fall from a high, as a negative fraction."""
    e = np.asarray(equity, dtype=np.float64)
    if len(e) < 2:
        return 0.0
    return float((e / np.maximum.accumulate(e) - 1.0).min())


def annual(equity: np.ndarray) -> float:
    """Growth per year, compounded."""
    e = np.asarray(equity, dtype=np.float64)
    years = (len(e) - 1) / YEAR
    if years <= 0 or e[0] <= 0 or e[-1] <= 0:
        return float("nan")
    return float((e[-1] / e[0]) ** (1.0 / years) - 1.0)


def daily(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, dtype=np.float64)
    return e[1:] / e[:-1] - 1.0


def bootstrap_ci(active: np.ndarray, n: int = 2000, block: int = 21, seed: int = 0, level: float = 0.90) -> tuple[float, float]:
    """A 90 % range for the yearly edge over the index, from the daily differences. Days are drawn in runs of
    a month (a circular block bootstrap) because good and bad stretches cluster."""
    a = np.asarray(active, dtype=np.float64)
    a = a[np.isfinite(a)]
    T = len(a)
    if T < 40:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    blocks = -(-T // block)
    starts = rng.integers(T, size=(n, blocks))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(n, -1)[:, :T]
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo * YEAR), float(hi * YEAR)


def summarize(equity: np.ndarray, index_equity: np.ndarray, traded: np.ndarray, cost: np.ndarray,
              cash_equity: np.ndarray | None = None, ci: bool = True) -> dict:
    """A book against the index over the same days."""
    e, ix = np.asarray(equity, dtype=np.float64), np.asarray(index_equity, dtype=np.float64)
    r, ri = daily(e), daily(ix)
    rc = daily(cash_equity) if cash_equity is not None and len(cash_equity) == len(e) else np.zeros_like(r)
    years = max((len(e) - 1) / YEAR, 1e-9)
    ex = r - rc
    active = np.log1p(r) - np.log1p(ri)
    mean_eq = float(e.mean()) if len(e) else 1.0
    out = {
        "days": int(len(e) - 1),
        "ret": float(e[-1] / e[0] - 1.0), "index_ret": float(ix[-1] / ix[0] - 1.0),
        "ret_ann": annual(e), "index_ann": annual(ix),
        "vol_ann": float(r.std() * np.sqrt(YEAR)) if len(r) > 1 else float("nan"),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(YEAR)) if len(r) > 1 and ex.std() > 1e-9 else float("nan"),
        "max_dd": max_drawdown(e),
        "ir": float(active.mean() / active.std() * np.sqrt(YEAR)) if len(r) > 1 and active.std() > 1e-9 else float("nan"),
        "tracking": float(active.std() * np.sqrt(YEAR)) if len(r) > 1 else float("nan"),
        "turnover": float(np.sum(traded) / mean_eq / years),
        "cost_drag": float(np.sum(cost) / mean_eq / years),
        "costs": float(np.sum(cost)),
    }
    out["vs_index"] = out["ret"] - out["index_ret"]
    out["vs_index_ann"] = out["ret_ann"] - out["index_ann"]
    out["ci"] = list(bootstrap_ci(active)) if ci else [float("nan"), float("nan")]
    return out


def by_year(days: np.ndarray, equity: np.ndarray, index_equity: np.ndarray) -> list[dict]:
    """Calendar years: the model's return and the index's. The first year starts where the record starts."""
    days, e, ix = np.asarray(days), np.asarray(equity, dtype=np.float64), np.asarray(index_equity, dtype=np.float64)
    out = []
    years = days // 10000
    for y in np.unique(years):
        rows = np.flatnonzero(years == y)
        a = max(rows[0] - 1, 0)                     # from the last close of the year before
        b = rows[-1]
        if b > a:
            out.append({"year": int(y), "ret": float(e[b] / e[a] - 1.0), "index": float(ix[b] / ix[a] - 1.0)})
    return out


def thin(xs, k: int = 60) -> list[float]:
    """At most k points, evenly picked, the last one kept."""
    xs = list(xs)
    if len(xs) <= k:
        return [round(float(x), 5) for x in xs]
    step = (len(xs) - 1) / (k - 1)
    return [round(float(xs[round(i * step)]), 5) for i in range(k)]


def clean(obj):
    """NaN and infinity cannot go into JSON: they become null. Numpy numbers become plain ones."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
