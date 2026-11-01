# Robust splits of day4's qualifying signals: per-signal return in the selection weeks and the holdout.
import json, datetime as dt
import numpy as np
C = "/data/group_cache"; U = dt.timezone.utc
feats = json.load(open(f"{C}/consensus_copied29.json"))["feats"]
X = np.load(f"{C}/consensus_X_copied29.npy"); rows = json.load(open(f"{C}/consensus_rows_copied29.json"))
f = {k: X[:, i] for i, k in enumerate(feats)}
tt = np.array([r["t"] for r in rows]); est = np.array([r["est"] for r in rows]); ret = np.array([r["ret"] for r in rows])
score = np.array([r["score"] for r in rows]); avg = np.array([r["avg"] for r in rows]); asset = np.array([r["asset"] for r in rows])
T0 = dt.datetime(2026, 9, 15, tzinfo=U).timestamp(); SE = dt.datetime(2026, 10, 20, tzinfo=U).timestamp()
qual = (score >= 4) & (est >= .05) & (est <= .9) & (est <= avg + .05) & (tt >= T0)
last = {}; pop = []
for i in np.argsort(tt, kind="stable"):
    if qual[i] and not (asset[i] in last and tt[i] - last[asset[i]] < 72 * 3600):
        last[asset[i]] = tt[i]; pop.append(i)
pop = np.array(pop)
rng = np.random.default_rng(2)
def show(name, m):
    s = pop[m[pop]]
    if len(s) < 25: return
    a, b = s[tt[s] < SE], s[tt[s] >= SE]
    d = (tt[s] // 86400).astype(int); ud = np.unique(d)
    sums = np.array([ret[s][d == x].sum() for x in ud]); cnt = np.array([(d == x).sum() for x in ud])
    dr = rng.integers(0, len(ud), (3000, len(ud))); bm = sums[dr].sum(1) / cnt[dr].sum(1)
    print(f"  {name:30s} n {len(s):4d} ({len(s)/47:4.1f}/day)  {100*ret[s].mean():+6.1f}% [{100*np.quantile(bm,.05):+6.1f}, {100*np.quantile(bm,.95):+6.1f}]  "
          f"select {100*ret[a].mean():+6.1f}% (n {len(a)})  holdout {100*ret[b].mean():+6.1f}% (n {len(b)})")
pre = f["in_play"] < .5
preh = np.where(f["pre_game_h_log"] > 0, np.expm1(f["pre_game_h_log"]), np.where(pre, 999, 0))
show("all day4 signals", np.ones(len(rows), bool))
show("pre-game", pre); show("in play", ~pre)
show("pre-game, kick-off within 3h", pre & (preh <= 3)); show("pre-game, 3-12h", pre & (preh > 3) & (preh <= 12)); show("pre-game, >12h", pre & (preh > 12))
for s_ in (4, 5, 6):
    show(f"score {s_}" + ("+" if s_ == 6 else ""), (score >= 6) if s_ == 6 else (score == s_))
show("price 30-70c", (est >= .3) & (est <= .7)); show("price <30c", est < .3); show("price >70c", est > .7)
show("favourite (>50c)", est > .5); show("underdog (<50c)", est <= .5)
show("spread", f["spread"] > .5); show("total", f["total"] > .5); show("moneyline/other", (f["spread"] < .5) & (f["total"] < .5))
for lg in ("nba", "nfl", "cfb", "cbb", "mlb", "nhl", "soccer", "tennis", "esports"):
    show(f"league {lg}", f[f"lg_{lg}"] > .5)
show("other leagues / not sports", sum(f[f"lg_{k}"] for k in ("nba", "nfl", "cfb", "cbb", "mlb", "nhl", "soccer", "tennis", "esports", "fight")) < .5)
show("pre-game & price 30-70c", pre & (est >= .3) & (est <= .7))
show("pre-game & not NFL", pre & (f["lg_nfl"] < .5))
