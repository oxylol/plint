"""Gradient-boosted trees (LightGBM) on the same inputs as the network.

On tables of figures like these, boosted trees are the method to beat; published comparisons put them level
with neural networks more often than behind. They are here so the network has a real rival."""
from __future__ import annotations

import numpy as np

from .dataset import MAIN, Table

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=400,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=63,
              verbosity=-1, num_threads=3)
MAX_ROWS = 1_200_000        # more than this is thinned by whole days: Plint's memory, not the method's limit
ROUNDS, PATIENCE = 1000, 60


def _matrix(table: Table, days: list[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = sum(int(table.ptr[t + 1] - table.ptr[t]) for t in days)
    if n > MAX_ROWS:
        keep = rng.random(len(days)) < MAX_ROWS / n
        days = [t for t, k in zip(days, keep) if k]
    rows = np.concatenate([np.arange(table.ptr[t], table.ptr[t + 1]) for t in days])
    y = table.Y[rows, MAIN]
    ok = np.isfinite(y)
    return np.asarray(table.X[rows[ok]], dtype=np.float32), y[ok]


def fit(table: Table, train: list[int], val: list[int], seed: int = 0) -> dict:
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    params = {**PARAMS, "seed": seed}
    Xt, yt = _matrix(table, train, rng)
    Xv, yv = _matrix(table, val, rng)
    probe = lgb.train(params, lgb.Dataset(Xt, yt), ROUNDS, valid_sets=[lgb.Dataset(Xv, yv)],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    rounds = max(20, int((probe.best_iteration or ROUNDS) * 1.1))       # a little more for the larger table
    del Xt, yt, Xv, yv, probe
    X, y = _matrix(table, sorted(train + val), rng)
    final = lgb.train(params, lgb.Dataset(X, y), rounds)
    return {"model": final.model_to_string(), "rounds": rounds}


def predict(params: dict, X: np.ndarray, ctx: np.ndarray | None = None) -> np.ndarray:
    import lightgbm as lgb
    booster = params.get("_booster")
    if booster is None:
        booster = params["_booster"] = lgb.Booster(model_str=params["model"])
    return booster.predict(np.asarray(X, dtype=np.float32))
