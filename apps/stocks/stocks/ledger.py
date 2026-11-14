"""The faster verdict: every evening's ranking, scored against what followed.

A book's profit needs years to tell skill from luck: with a few percent a year of edge against ten percent
a year of wobble around the index, the t-statistic reaches 2 after a decade or more. A ranking of a
thousand stocks is a thousand small forecasts a day. Its rank correlation with the returns that followed
(the "IC") averages out much faster: a real daily IC of 0.02 shows at t = 2 in about three months.

The rule for the lamp, fixed before any data was seen:
    fewer than MIN_DAYS scored            → none      (too early)
    t >= 2 and the mean above zero        → likely    (skill likely)
    mean + 2 standard errors below FLOOR  → unlikely  (even the kind reading shows nothing worth trading)
    otherwise                             → none
t is Newey-West with as many lags as the horizon, because 5-day outcomes of neighbouring evenings overlap."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

MIN_DAYS = 40
FLOOR = 0.005
FLOOR_EXCESS = 0.001        # for forecasts scored by excess return: 0.1 % per call
ABOUT = 120                 # days the page says a verdict usually takes


def spearman(a: np.ndarray, b: np.ndarray, min_n: int = 20) -> tuple[float, int]:
    """Rank correlation over the pairs where both are known. (nan, n) when too few."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < min_n:
        return float("nan"), n
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (float((ra * rb).sum() / den) if den > 0 else float("nan")), n


def nw_t(x: np.ndarray, lags: int) -> tuple[float, float, float, int]:
    """(mean, standard error, t, n) of a series whose neighbours overlap by `lags` steps."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan"), n
    m = x.mean()
    e = x - m
    var = (e * e).mean()
    for k in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] * e[:-k]).mean()
    se = np.sqrt(max(var, 1e-18) / n)
    return float(m), float(se), float(m / se), n


def verdict(values: np.ndarray, horizon: int, floor: float = FLOOR, what: str = "days") -> dict:
    mean, se, t, n = nw_t(values, max(horizon - 1, 0))
    if n < MIN_DAYS:
        lamp, text = "none", f"No verdict yet: {n} of about {ABOUT} {what} scored"
    elif t >= 2.0 and mean > 0:
        lamp, text = "likely", f"Skill likely: {n} {what} scored"
    elif mean + 2.0 * se < floor:
        lamp, text = "unlikely", f"Skill unlikely: {n} {what} scored"
    else:
        lamp, text = "none", f"No verdict yet: {n} {what} scored, the answer is still inside the noise"
    r = lambda v, k: None if not np.isfinite(v) else round(float(v), k)
    return {"lamp": lamp, "text": text, "ic": r(mean, 4), "t": r(t, 2), "n": n, "horizon": horizon}


def table_ic(S: np.ndarray, R: np.ndarray, member: np.ndarray, t0: int = 0, t1: int | None = None) -> np.ndarray:
    """Daily IC of a score table against a table of what followed, over each day's members. NaN where a day
    has no scores or its outcome is not known."""
    t1 = S.shape[0] if t1 is None else t1
    out = np.full(S.shape[0], np.nan)
    for t in range(t0, t1):
        m = member[t]
        out[t], _ = spearman(S[t, m], R[t, m])
    return out


def decile_spread(S: np.ndarray, R: np.ndarray, member: np.ndarray, t0: int, t1: int) -> float:
    """Mean of (top tenth's outcome - bottom tenth's) per day: the ranking's reach in return terms."""
    out = []
    for t in range(t0, t1):
        m = member[t] & np.isfinite(S[t]) & np.isfinite(R[t])
        if m.sum() < 50:
            continue
        s, r = S[t, m], R[t, m]
        k = max(1, len(s) // 10)
        order = np.argsort(s)
        out.append(r[order[-k:]].mean() - r[order[:k]].mean())
    return float(np.mean(out)) if out else float("nan")


def settle_scores(store, panel, fwd_rel: dict[int, np.ndarray], model: str, horizons: tuple[int, ...]) -> int:
    """Live: score every stored evening whose outcome is now known. `fwd_rel[h]` is the [T, N] table of
    what followed (learn.dataset.relative(forward(panel, h))). Returns how many (day, horizon) were settled."""
    done = {h: {d for d, _, _ in store.ic_series(model, h)} for h in horizons}
    count = 0
    for date in store.score_dates(model):
        d = int(date.replace("-", ""))
        try:
            t = panel.t_of(d)
        except KeyError:
            continue
        scores = None
        for h in horizons:
            if date in done[h] or not np.isfinite(fwd_rel[h][t]).any():
                continue
            scores = scores if scores is not None else store.load_scores(model, date)
            s = np.full(panel.N, np.nan)
            for sym, v in scores.items():
                j = panel.ix.get(sym)
                if j is not None:
                    s[j] = v
            ic, n = spearman(s, fwd_rel[h][t])
            if np.isfinite(ic):
                store.save_ic(model, date, h, ic, n)
                count += 1
    return count


def settle_signals(store, panel, index_symbol: str) -> int:
    """Live: forecasts without a book (the agent's, the event models'). Each is scored by the stock's return
    over the index from the next open, over its own horizon."""
    from .learn.dataset import forward
    cache: dict[int, np.ndarray] = {}
    count = 0
    ji = panel.ix.get(index_symbol)
    for model, date, sym, h, _p in store.open_signals():
        j = panel.ix.get(sym)
        try:
            t = panel.t_of(int(date.replace("-", "")))
        except KeyError:
            continue
        if j is None or ji is None:
            continue
        if h not in cache:
            cache[h] = forward(panel, h)
        own, idx = cache[h][t, j], cache[h][t, ji]
        if np.isfinite(own) and np.isfinite(idx):
            store.settle_signal(model, date, sym, h, float(own - idx), _iso(panel.days[-1]))
            count += 1
    return count


def signal_verdict(signals: list[dict], horizon: int) -> dict | None:
    """One lamp for a list of settled forecasts: the mean excess return of the calls, each taken in the
    direction it was made (p below 0.5 is a call against), averaged per evening first."""
    by_day: dict[str, list[float]] = {}
    for s in signals:
        if s.get("excess") is None or s["horizon"] != horizon:
            continue
        side = 1.0 if s["p"] >= 0.5 else -1.0
        by_day.setdefault(s["date"], []).append(side * s["excess"])
    if not by_day:
        return None
    series = np.array([np.mean(v) for _, v in sorted(by_day.items())])
    return verdict(series, horizon, FLOOR_EXCESS, "evenings")


def brier(signals: list[dict], horizon: int) -> tuple[float, int] | None:
    """How good the stated chances were: mean of (p - outcome)^2 over settled forecasts; 0.25 is a coin."""
    xs = [(s["p"] - (1.0 if s["excess"] > 0 else 0.0)) ** 2 for s in signals
          if s.get("excess") is not None and s["horizon"] == horizon]
    return (float(np.mean(xs)), len(xs)) if xs else None


def _iso(d: int) -> str:
    d = int(d)
    return f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"
