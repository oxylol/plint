"""The straight line: a ridge regression on the same inputs as the network.

It is the floor. Whatever a linear fit of fifty-six ranks finds, any fancier model finds too; what the
network is worth is what it adds on top of this."""
from __future__ import annotations

import numpy as np

from ..ledger import spearman
from .dataset import MAIN, Table

SHRINK = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)      # ridge strength, as a share of the inputs' own scale


def _moments(table: Table, days: list[int], chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    F = table.X.shape[1]
    A, b = np.zeros((F, F)), np.zeros(F)
    for i in range(0, len(days), chunk):
        part = days[i:i + chunk]
        rows = np.concatenate([np.arange(table.ptr[t], table.ptr[t + 1]) for t in part]) if part else np.array([], dtype=int)
        if not len(rows):
            continue
        y = table.Y[rows, MAIN]
        ok = np.isfinite(y)
        X = np.asarray(table.X[rows[ok]], dtype=np.float64)
        A += X.T @ X
        b += X.T @ y[ok]
    return A, b


def _solve(A: np.ndarray, b: np.ndarray, shrink: float) -> np.ndarray:
    lam = shrink * np.trace(A) / len(A)
    return np.linalg.solve(A + lam * np.eye(len(A)), b)


def day_ic(table: Table, days: list[int], score) -> float:
    """Mean daily rank correlation of `score(X_day, ctx_day)` with what followed."""
    ics = []
    for t in days:
        sl = table.rows(t)
        if sl.stop - sl.start < 20:
            continue
        ic, _ = spearman(score(np.asarray(table.X[sl], dtype=np.float32), table.ctx[t]), table.R[sl, MAIN])
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def fit(table: Table, train: list[int], val: list[int], seed: int = 0) -> dict:
    A, b = _moments(table, train)
    best, best_ic = SHRINK[len(SHRINK) // 2], -np.inf
    for s in SHRINK:
        w = _solve(A, b, s)
        ic = day_ic(table, val, lambda X, _c, w=w: X @ w)
        if np.isfinite(ic) and ic > best_ic:
            best, best_ic = s, ic
    Av, bv = _moments(table, val)                    # then the same strength on everything known
    return {"w": _solve(A + Av, b + bv, best).astype(np.float32), "shrink": best, "val_ic": best_ic}


def predict(params: dict, X: np.ndarray, ctx: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(X, dtype=np.float32) @ params["w"]
