"""When the learned models are fitted and on what: the walk-forward schedule, and where fits are kept.

A fit that takes over on day t has seen only outcomes that were complete before t. Its last half year is
set aside to choose how long to train (epochs, trees, ridge strength), with a gap before it as long as the
longest outcome so no training row's outcome reaches into it; then the model is trained again on all of it.

In a backtest the models are refitted every six months. Live they are refitted every week, so the backtest
is, if anything, run with staler models than the ones that trade."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from .. import features
from . import net, ridge, trees
from .dataset import H_MAX, MAIN, WARMUP, Table

ALGOS = {"ridge": ridge, "trees": trees, "net": net}
SEEDS_BACKTEST = {"ridge": 1, "trees": 1, "net": 3}
SEEDS_LIVE = {"ridge": 1, "trees": 2, "net": 5}
VAL_DAYS = 126
MIN_TRAIN_DAYS = 250


class NotEnoughHistory(Exception):
    pass


def split(table: Table, t_fit: int, val_days: int = VAL_DAYS) -> tuple[list[int], list[int]]:
    """(training days, validation days) for a fit that takes over at row t_fit."""
    last = min(t_fit, table.T) - H_MAX - 2          # the newest day whose 20-day outcome was complete before t_fit
    v0 = last - val_days + 1
    train_end = v0 - H_MAX - 2                      # and a gap, so no training outcome overlaps the validation span
    train, val = list(range(WARMUP, train_end + 1)), list(range(v0, last + 1))
    if len(train) < MIN_TRAIN_DAYS:
        raise NotEnoughHistory(f"{len(train)} training days before row {t_fit}; {MIN_TRAIN_DAYS} needed")
    return train, val


def refit_rows(days: np.ndarray, t_first: int, t_last: int, months: int = 6) -> list[int]:
    """Rows where a new fit takes over: t_first, then the first trading day of every `months`-th month."""
    out = [t_first]
    for t in range(t_first + 1, t_last + 1):
        m, pm = int(days[t]) // 100 % 100, int(days[t - 1]) // 100 % 100
        if m != pm and (m - 1) % months == 0:
            out.append(t)
    return out


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def ensemble(algo: str, fits: list[dict], X: np.ndarray, ctx: np.ndarray, head: int = MAIN) -> np.ndarray:
    """One day's scores: each seed's output put on a common scale, then averaged. `head` picks the horizon
    (only the network has more than the 5-day one)."""
    kw = {"head": head} if algo == "net" else {}
    return np.mean([_z(np.asarray(ALGOS[algo].predict(p, X, ctx, **kw), dtype=np.float64)) for p in fits], axis=0).astype(np.float32)


def fit_at(table: Table, algo: str, t_fit: int, seeds: int, log: Callable | None = None, **fit_kw) -> list[dict]:
    train, val = split(table, t_fit)
    out = []
    for s in range(seeds):
        kw = {**fit_kw, "log": log} if algo == "net" and log else fit_kw
        out.append(ALGOS[algo].fit(table, train, val, seed=s, **kw))
        if log:
            log(f"{algo}: fit for {int(table.days[min(t_fit, table.T - 1)])}, seed {s + 1} of {seeds}, "
                f"{len(train)} training days, validation IC {out[-1].get('val_ic', float('nan')):+.4f}")
    return out


def walk_forward(table: Table, algo: str, t_first: int, t_last: int, n_cols: int, months: int = 6,
                 seeds: int | None = None, log: Callable | None = None, heads: tuple[int, ...] | None = None,
                 **fit_kw):
    """A score table [T, n_cols] over rows t_first..t_last, every row scored by a fit made before it.
    With `heads` (the network only), one table per horizon head from the same fits: {head: table}."""
    tables = {h: np.full((table.T, n_cols), np.nan, dtype=np.float32) for h in (heads or (MAIN,))}
    points = refit_rows(table.days, t_first, t_last, months)
    seeds = seeds or SEEDS_BACKTEST[algo]
    for i, tf in enumerate(points):
        train, val = split(table, tf)
        fits = [ALGOS[algo].fit(table, train, val, seed=s, **fit_kw) for s in range(seeds)]
        if log:
            log(f"{algo}: fitted for {int(table.days[tf])} on {len(train)} days "
                f"(validation IC {np.mean([f.get('val_ic', np.nan) for f in fits]):+.4f})")
        t_end = points[i + 1] if i + 1 < len(points) else t_last + 1
        for t in range(tf, t_end):
            sl = table.rows(t)
            if sl.stop > sl.start:
                X = np.asarray(table.X[sl], dtype=np.float32)
                for h, S in tables.items():
                    S[t, table.col[sl]] = ensemble(algo, fits, X, table.ctx[t], h)
    return tables if heads else tables[MAIN]


class Fits:
    """Fitted models on disk: <data>/models/<algo>/<day>.pkl, newest wins."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "models"

    def save(self, algo: str, day: int, fits: list[dict], note: str = "") -> Path:
        path = self.dir / algo / f"{int(day)}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in f.items() if not k.startswith("_")} for f in fits]
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"algo": algo, "day": int(day), "fits": clean, "features": features.NAMES,
                                      "context": features.CTX_NAMES, "note": note}))
        tmp.replace(path)
        for old in sorted((self.dir / algo).glob("*.pkl"))[:-6]:     # keep the last few
            old.unlink()
        return path

    def latest(self, algo: str, on_or_before: int | None = None) -> dict | None:
        """The newest fit made on or before a day, if it was fitted on today's figures."""
        folder = self.dir / algo
        if not folder.is_dir():
            return None
        for path in sorted(folder.glob("*.pkl"), reverse=True):
            if on_or_before is not None and int(path.stem) > on_or_before:
                continue
            got = pickle.loads(path.read_bytes())
            if got.get("features") == features.NAMES and got.get("context") == features.CTX_NAMES:
                return got
        return None
