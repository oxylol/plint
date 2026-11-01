# Which rule tweaks hold in both periods AND in both leader groups (copied29, sports144)? Per-signal, day4-qualifying signals.
import json, datetime as dt
import numpy as np
C = "/data/group_cache"; U = dt.timezone.utc
T0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp(); SE = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
def load(label):
    feats = json.load(open(f"{C}/consensus_{label}.json"))["feats"]
    X = np.load(f"{C}/consensus_X_{label}.npy"); rows = json.load(open(f"{C}/consensus_rows_{label}.json"))
    f = {k: X[:, i] for i, k in enumerate(feats)}
    tt = np.array([r["t"] for r in rows]); est = np.array([r["est"] for r in rows]); ret = np.array([r["ret"] for r in rows])
    score = np.array([r["score"] for r in rows]); avg = np.array([r["avg"] for r in rows]); asset = np.array([r["asset"] for r in rows])
    qual = (score >= 4) & (est >= .05) & (est <= .9) & (est <= avg + .05) & (tt >= T0)
    last = {}; pop = []
    for i in np.argsort(tt, kind="stable"):
        if qual[i] and not (asset[i] in last and tt[i] - last[asset[i]] < 72 * 3600):
            last[asset[i]] = tt[i]; pop.append(i)
    pop = np.array(pop)
    f["est"] = est; f["score_"] = score.astype(float)
    return {k: v[pop] for k, v in f.items()}, ret[pop], tt[pop]
D = {lab: load(lab) for lab in ("copied29", "sports144")}
rules = {
    "drop point spreads": lambda f: f["spread"] < .5,
    "drop in-play": lambda f: f["in_play"] < .5,
    "drop NFL": lambda f: f["lg_nfl"] < .5,
    "drop esports": lambda f: f["lg_esports"] < .5,
    "drop MLB/NHL": lambda f: (f["lg_mlb"] < .5) & (f["lg_nhl"] < .5),
    "drop spreads + esports": lambda f: (f["spread"] < .5) & (f["lg_esports"] < .5),
    "only 30-70c": lambda f: (f["est"] >= .3) & (f["est"] <= .7),
    "drop 30-70c underdogs (<50c)": lambda f: ~((f["est"] >= .3) & (f["est"] < .5)),
    "favourites only (>50c)": lambda f: f["est"] > .5,
    "score 4 only": lambda f: f["score_"] == 4,
    "kick-off within 12h or in play": lambda f: (f["in_play"] > .5) | (np.expm1(f["pre_game_h_log"]) <= 12),
}
print(f"{'rule':32s} | {'copied29: all':>13s} {'select':>7s} {'holdout':>8s} | {'sports144: all':>14s} {'select':>7s} {'holdout':>8s} | passes")
for lab in D:
    f, r, t = D[lab]
for name, rule in [("(day4 as is)", lambda f: np.ones(len(f["est"]), bool))] + list(rules.items()):
    cells, passes = [], 0
    for lab, (f, r, t) in D.items():
        base_s, base_h = r[t < SE].mean(), r[t >= SE].mean()
        m = rule(f)
        a, s_, h_ = r[m].mean(), r[m & (t < SE)].mean(), r[m & (t >= SE)].mean()
        passes += (s_ > base_s) + (h_ > base_h)
        cells.append(f"{100*a:+6.1f}% (n{m.sum():4d}) {100*s_:+6.1f}% {100*h_:+7.1f}%")
    print(f"{name:32s} | {cells[0]} | {cells[1]} | {passes}/4")
