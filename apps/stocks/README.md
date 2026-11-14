# Stocks

Stock-trading models on paper, side by side, at `stocks.${DOMAIN}`: a neural network, an agent on the Claude
API with a monthly budget, the index as the bar, and the classic rules. Each has its own $10,000 of pretend
money. One run per US trading day after the close; what a model decides is filled at the next open.
**There is no broker connection: nothing here can place a real order.**

Why it is built the way it is: `DESIGN.md`. The page and its JSON: `PAGE.md`.

## Keys (in `/opt/plint/.env`, then `plint up stocks`)

| key | for | without it |
|---|---|---|
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | prices (a free paper account at alpaca.markets, email only) | nothing runs; the page says so |
| `STOCKS_ANTHROPIC_API_KEY` | the agent | the agent waits |
| `SEC_CONTACT` | an email address the SEC asks for when its filings are read | after-earnings and insider models wait; the agent reads headlines only |

`STK_AGENT_CAP_USD` (200) is the agent's hard stop per calendar month. `STK_NOTIFY` is `daily` (one line after
the close), `trades` or `off`.

## What happens once the Alpaca keys are in

1. The first run fetches ten years of daily prices (about an hour at the free plan's pace) and picks each
   month's universe. The index fund starts that evening.
2. The first backtests start by themselves: walk-forward fits, history replayed, a report per model in
   `data/stocks/reports/`. Several hours, at low priority. No model but the index trades before its report
   is written.
3. From the next evening every model with a report decides daily. The learned models are refitted at the
   weekend.

The exam stretch (the last eighteen months, held back) is never run by itself:
`docker exec stocks python -m stocks research --exam`, once, when the design is frozen.

## Commands

```
docker exec stocks python -m stocks status            keys, data, books, spend
docker exec stocks python -m stocks run               one evening run now
docker exec stocks python -m stocks research          the design-stretch backtests again
docker exec stocks python -m stocks research --exam   the held-back stretch (counted every time)
docker exec stocks python -m stocks fit               refit the learned models now
docker exec stocks python -m stocks filings           read the SEC now (the ten years of history first, if not in)
docker exec stocks python -m stocks review            the agent's weekly look at its own record, now
docker exec stocks python -m stocks universe          this month's stocks, and what the name filter kept out
docker exec stocks python -m stocks variant "…"       write down a design change made after seeing figures
docker exec stocks python -m stocks reset <model>     start a model's record over (after its rules changed)
```

`plint logs stocks` for the log; the page's own log shows corrections and faults in plain words.

## Where things are

```
stocks/
  alpaca.py  bars.py  ingest.py     the price source (read only), the price table, fetching into it
  universe.py  panel.py             which stocks each month; prices as day-by-stock tables
  features.py                       the 56 figures the learned models read
  book.py  portfolio.py             the paper book; from a ranking to orders
  models/                           the contestants (rules.py, learned.py, events.py)
  learn/                            the training table, ridge / trees / net, the walk-forward schedule
  agent/                            the desk: prompts, the model calls, the budget
  sec.py  filings.py  events.py     SEC filings: results days, insider purchases
  ledger.py  metrics.py             rankings scored against what followed; the figures a book is judged by
  backtest.py  research.py          history replayed; the reports
  pipeline.py  runner.py            the evening run; the clock (every job is a child process)
  publish.py  web.py                the page's JSON; the server
static/                             the page
tests/                              unittest; `synth.py` is a made-up market with effects planted on purpose
```

State: `data/stocks/` (decisions, ledger, fits, reports: small, backed up). Prices and training tables:
`/mnt/storage/plint/stocks/` (large, can be fetched again).

## Tests

```
cd apps/stocks
docker build -t plint/stocks:dev .
docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev \
  python -m unittest discover -s tests -p 'test_*.py'
```

`test_e2e` runs the whole path on the made-up market (history, universe, fits, backtests, fifteen evenings,
the page's JSON). `test_leak` is the one that matters most: it replaces the future with noise and requires
the same figures and the same orders. The page's own checks: `tests/page.js` (its head says how).

Try it without touching the live app: run the image on another port with its own folders,

```
docker run --rm -p 127.0.0.1:5259:8000 -v /opt/plint/.scratch/stocks-dev/data:/data \
  -v /opt/plint/.scratch/stocks-dev/cache:/cache --user 1002:1002 --read-only --tmpfs /tmp -e HOME=/tmp plint/stocks:dev
```
