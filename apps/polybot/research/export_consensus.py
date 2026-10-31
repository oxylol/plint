"""Fit the consensus model on every resolved candidate and write it for the bot.

    python export_consensus.py DATA_DIR OUT.json [label] [lambda]

Reads consensus_model.py's saved table (group_cache/consensus_*), fits the market-anchored L2 logistic regression
that won (lambda 1000), writes a round-3 model file (inputs "consensus") with the policy the walk-forward chose, and
checks the bot's scorer against the fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot.model import Model  # noqa: E402

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LABEL = sys.argv[3] if len(sys.argv) > 3 else "copied29"
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0
C = DATA / "group_cache"
feats = json.loads((C / f"consensus_{LABEL}.json").read_text())["feats"]
X = np.load(C / f"consensus_X_{LABEL}.npy")
rows = json.loads((C / f"consensus_rows_{LABEL}.json").read_text())
y = np.array([r["final"] for r in rows], float)
est = np.array([r["est"] for r in rows], float)
off = np.log(np.clip(est, 1e-4, 1 - 1e-4) / (1 - np.clip(est, 1e-4, 1 - 1e-4)))
mu, sd = X.mean(0), X.std(0) + 1e-6
Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
w = np.zeros(Z.shape[1])
I = np.eye(Z.shape[1])
I[-1, -1] = 0
for _ in range(40):
    p = 1 / (1 + np.exp(-np.clip(Z @ w + off, -30, 30)))
    w -= np.linalg.solve(Z.T @ (Z * (p * (1 - p))[:, None]) + LAM * I, Z.T @ (p - y) + LAM * I @ w)
out = {"kind": "linear", "version": 3, "target": "anchor", "anchor": "logit_p", "inputs": "consensus", "features": feats,
       "mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist(),
       "policy": {"threshold": -0.02, "decision": "expected return per $ after fee", "min_score": 4, "min_price": 0.05, "max_price": 0.90,
                  "max_above_leaders": 0.05, "no_in_play": False, "per_day": 0},
       "tournament": {"rows": len(rows), "lambda": LAM, "walk_forward": "Sep 29-Oct 31, book of 10 a day: +12.0%/trade [+4.5, +19.4], "
                      "+$37.7/week vs day4 rule +7.6% / +$25.3; holdout Oct 20-31 +16.9% vs +1.3%"},
       "trained_on": "consensus moments of the 29 wallets the live day4 setup copied, Sep 2 - Oct 31 2026 (research/consensus_model.py)"}
OUT.write_text(json.dumps(out, indent=1))
m = Model(OUT)
chk = np.random.default_rng(1).choice(len(rows), 500, replace=False)
want = 1 / (1 + np.exp(-(Z[chk] @ w + off[chk])))
got = np.array([m.win_probability({f: float(X[i, j]) for j, f in enumerate(feats)}) for i in chk])
print(f"{OUT.name}: {len(rows)} rows, bot vs fit max |Δp| {np.max(np.abs(got - want)):.1e}")
for f, v in sorted(zip(feats, w[:-1]), key=lambda t: -abs(t[1]))[:8]:
    print(f"  {f:16s} {v:+.4f}")
