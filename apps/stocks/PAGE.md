# The page and its JSON

One page (`static/index.html`, plus any files beside it) served by the app at `/`. It reads two endpoints and
nothing else. `tests/fixture_state.py` writes examples of both into `tests/fixtures/`; `tests/serve_fixture.py`
serves the page over them the way Caddy will (theme files injected).

All money is dollars. All returns are fractions (0.0123 = +1.23 %). Dates are `YYYY-MM-DD`, times ISO 8601.
A field may be `null` wherever that is written; arrays may be empty. Unknown fields must be ignored.

## GET /api/state

```
{
  "app": "stocks",
  "generated": "2026-11-13T22:31:05+01:00",
  "asof": "2026-11-13" | null,              // the last close in the books
  "phase": "keys" | "backfill" | "research" | "running",
  "phase_text": "Waiting for the Alpaca keys",
  "keys": {"alpaca": false, "anthropic": false, "sec": false, "ntfy": true},
  "notices": [{"level": "info" | "warn" | "fault", "text": "…"}],
  "clock": {"last_run": iso | null, "next_run": iso | null, "busy": "fitting the network" | null},
  "start_cash": 10000,
  "index": {"symbol": "SPY", "title": "S&P 500 (SPY)"},
  "spend": {"month": "2026-11", "used": 37.42, "cap": 200.0, "today": 4.10, "calls": 512,
            "stopped": null | "month cap" | "day allowance",
            "by_day": [["2026-11-03", 3.2], …]},
  "models": [ModelSummary, …],              // in the order to list them when nothing else decides
  "log": [{"t": iso, "level": "info" | "warn", "text": "…"}]     // newest first, at most 40
}
```

`ModelSummary`:

```
{
  "id": "net", "title": "Neural network", "kind": "index" | "rule" | "learned" | "agent",
  "what": "one sentence in plain words",
  "state": "running" | "waiting" | "off", "state_text": "waiting for its first fit",
  "since": "2026-11-20" | null, "days": 23,      // trading days of record
  "equity": 10234.51, "ret": 0.0235, "index_ret": 0.0112, "vs_index": 0.0123,   // since `since`
  "day_ret": 0.0041, "day_vs_index": 0.0012,
  "max_dd": -0.031,                               // worst drop from a high, a negative fraction
  "n_pos": 38, "cash": 0.021,                     // cash as a share of the book
  "trades": 84, "costs": 12.31,                   // costs paid so far, dollars
  "turnover": 4.2,                                // value traded in a year / average book value
  "verdict": {"lamp": "none" | "likely" | "unlikely",
              "text": "No verdict yet: 14 of about 120 days scored",
              "ic": 0.012, "t": 0.8, "n": 14, "horizon": 5} | null,
  "spark": [1.0, 1.002, …], "spark_index": [1.0, 1.001, …],      // value / start, at most 60 points each
  "backtest": null | {
      "from": "2020-01-02", "to": "2025-05-13",
      "ret_ann": 0.14, "index_ann": 0.12, "vs_index_ann": 0.02, "ci": [-0.03, 0.07],
      "sharpe": 0.8, "max_dd": -0.34, "turnover": 9.1, "cost_drag": 0.021,
      "ic": 0.011 | null, "ic_t": 2.4 | null,
      "exam": null | {"from": "2025-05-14", "to": "2026-11-13", "ret_ann": …, "index_ann": …,
                      "vs_index_ann": …, "ci": […, …], "ic": … | null, "ic_t": … | null}}
}
```

- `kind: "index"` is the bar to beat, not a contestant: show it as the reference line.
- `verdict` is the faster test (each evening's ranking or forecast scored against what followed). `lamp`
  `likely` = skill likely, `unlikely` = skill unlikely, `none` = too early. `ic` is the mean rank correlation,
  `t` its t-statistic, `n` the days (or forecasts) scored, `horizon` the trading days ahead it speaks about.
- `backtest` is history replayed with costs; `exam` is the stretch held back until the design was frozen.
  The agent has no backtest (it has read the past): say so where the backtest would be.
- `state: "waiting"` models have no record yet; every number may be `null`.

## GET /api/model/<id>

Everything in `ModelSummary`, plus:

```
{
  "curve": [["2026-11-03", 10012.2, 10004.0], …],    // date, the model's value, the index from the same start
  "positions": [{"symbol": "NVDA", "name": "NVIDIA Corporation", "weight": 0.027, "value": 271.3,
                 "since": "2026-11-10", "days": 3, "ret": 0.021, "why": "rank 3 of 1000"}],
  "pending": [{"symbol": "AMD", "name": "…", "side": "buy" | "sell", "weight": 0.025, "why": "…"}],
                                                      // decided tonight, filled at the next open
  "trades": [{"date": "2026-11-13", "symbol": "XOM", "name": "…", "side": "buy" | "sell", "usd": 251.2,
              "price": 118.2, "shares": 2.125, "cost": 0.08, "why": "fell to rank 312",
              "ret": 0.034 | null}],                  // newest first, at most 300; `ret` on a sell: what the position made
  "today": null | {"date": "2026-11-13", "note": "…",
                   "top": [{"symbol": "…", "name": "…", "score": 0.93, "rank": 1, "held": true}],
                   "bottom": [{…}]},                  // models that score every stock
  "ranking_history": [{"date": "2026-10-30", "ic": 0.031, "n": 998}],     // settled evenings, oldest first, ≤ 250
  "signals": [{"date": "…", "symbol": "…", "horizon": 5, "p": 0.62, "note": "…", "excess": 0.013 | null}],
                                                      // event models and the agent, newest first, ≤ 200;
                                                      // `excess` = return over the index once settled
  "memo": null | {"date": "2026-11-13", "text": "markdown"},               // agent
  "reads": [{"date": "…", "symbol": "…", "name": "…", "thesis": "…", "facts": ["…"], "priced_in": "…",
             "p5": 0.58, "p20": 0.61, "conviction": 0.4, "risks": "…",
             "sources": [{"title": "…", "url": "…"}]}],                    // agent, ≤ 40
  "report": "markdown" | null,                        // the backtest report
  "by_year": [{"year": 2020, "ret": 0.21, "index": 0.18}] | null,
  "copy": null | CopyStudy                            // the copy model only (copytrade/study.py)
}
```

`CopyStudy` (the copy study, DESIGN.md "Copy trades"; figures are the design stretch's):

```
{
  "made_at": iso, "from": "2019-01-02", "to": "2025-05-13",
  "counts": {"house_listed": 180, "insider_listed": 640, "ranked": 300, "listed": 820, "paper_reports": 1500, …},
  "rules": {"top": 10, "lookback_years": 3.0, "min_opened": 10},
  "people": [{"pid": "house:pelosi:nancy:ca", "name": "Nancy Pelosi", "pool": "house" | "insider",
              "label": "House, CA" | "CEO at ABT", "ret_ann": 0.21, "vs_index_ann": 0.05, "max_dd": -0.3,
              "turnover": 1.2, "opened": 34, "copied": 61, "trades": 80, "first": "2019-02-01", "last": "2025-04-02",
              "lag_days": 24 | null, "ranked": true, "rank": 8}],      // ranked first, best first
  "now": null | {"start": "2026-07-01", "from": "2023-06-29", "to": "2026-06-30", "eligible": 61,
                 "people": [{"pid", "name", "pool", "label", "score": 0.21}]},   // whom the live book follows
  "picks": [{"start": "2019-01-02", "people": ["name", …], "eligible": 48}],      // oldest first
  "fair": Combo, "hindsight": Combo + {"people": [{"pid", "name", "pool", "label", "vs_index_ann"}]}
}
Combo: {"from", "to", "ret_ann", "index_ann", "vs_index_ann", "ci", "sharpe", "max_dd", "turnover", "cost_drag",
        "basket", "vs_basket", "costs_x2", "costs_x3", "fx", "by_year": […], "curve": [["2019-01-02", 10000.0, 10000.0], …]}
```

## What the page must let him do (the list in .scratch/spec-stocks.md)

1. See every model at once, on a PC and on a phone: return against the index, worst drop, trades, costs.
   The index is the bar; the rest are ordered best to worst against it.
2. Open a model: what it holds, each trade with price and cost, and why (the network's score, the agent's memo).
3. For the models that rank every stock: today's ranking, and how past rankings held up (`ranking_history`).
4. For the agent: tonight's memo, its reads, and a meter of the month's spend against the cap, with the
   word that it stopped when it did.
5. For each model: the backtest in a few figures, the exam beside it, and the report to read in full.
6. Know at a glance what is missing: a key, the data, a first fit (`phase`, `keys`, `notices`, `state`).
7. Tell paper from real: this page is paper only, and says so.
8. The copy book: whom it copies now, the fair combo beside the hindsight one, everyone copied (House members
   and insiders, with a filter), and whom it followed each half year.
