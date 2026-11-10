# Polybot

Two Polymarket strategies, paper-traded side by side so they can be compared,
with the option to go live. Runs on Plint as the `poly` stack with a page at
`https://poly.<domain>`.

## Strategy 1 · Consensus

Follow the **leaderboard**. Every five minutes the bot

1. takes the top *N* traders by profit (day / week / month / all-time board),
2. pulls every fill they made in the last *L* hours,
3. works out each leader's **stance** per market: money into one outcome minus
   money into the other, so market makers who buy both sides count as flat.
   A leader *agrees* with an outcome when the stance is at least
   `POLY_MIN_LEADER_USD`, and is *against* it below minus that,
4. **buys** when `agree − against ≥ POLY_MIN_AGREE`, with a market order of
   `POLY_STAKE_USD`, if the price is inside the band and not more than
   `POLY_MAX_ABOVE_LEADERS` above what the leaders paid,
5. **sells** a position when the same number of leaders net-sell it
   (`POLY_FOLLOW_SELLS`), and books resolved markets at 1 or 0.

### Variants: comparing consensus setups

`POLY_VARIANTS` runs extra copies of the consensus strategy with some settings
changed, each with its own paper book, so the page can compare them. The
syntax is `name:setting=value,setting=value;name2:...` where the settings are
the `POLY_*` names without the prefix. The built-in set (used when
`POLY_VARIANTS` is empty) compares the four leaderboard windows, each at the
base threshold and at four agreeing, the two credible-pack setups, the three
round-2 data-model setups and the round-3 edge model below
(`polybot/config.py` has the full string):

```
week4:min_agree=4;month:window=month;month4:window=month,min_agree=4;day:window=day;day4:window=day,min_agree=4;all:window=all;all4:window=all,min_agree=4;credible:pack=1,window=month,min_agree=2,min_weight=2.5,lookback_hours=48,min_hours_to_end=12,max_per_cycle=3,label=Credible pack;smart:pack=1,window=month,min_agree=1,min_weight=1.5,fresh_only=1,adaptive_trust=1,lookback_hours=48,min_hours_to_end=12,max_per_cycle=3,label=Smart money;learned:pack=1,model=1,model_file=model_r2.json,…;weather5:…;weather10:…;edge:pack=1,model=1,model_file=model.json,window=month,min_agree=1,lookback_hours=48,max_per_cycle=3,label=Edge model (round 3)
```

Other ideas: `week4:min_agree=4` (a lower bar), `top50:top_n=50` (a bigger
pack), `politics:category=politics` (one category's board),
`fast:lookback_hours=6,min_agree=3`. Leaders and fills are fetched once and
shared, so variants are cheap. The base setup is always called `consensus`.

### Credible pack: only take advice from traders worth copying

A leaderboard ranks by profit, and profit comes from three kinds of trader a
copier cannot use: in-game sports scalpers (the edge is gone minutes later),
market makers (the profit is the spread, not a view) and one-bet whales. Setups
with `pack=1` therefore measure the top `CR_POOL` of their board every
`CR_REFRESH_HOURS` (closed positions, a week of fills, account age) and keep
the `CR_PACK_SIZE` most credible traders: active now (`CR_MIN_FILLS_7D`,
`CR_MAX_IDLE_HOURS`), a real history (`CR_MIN_CLOSED` closed positions,
account ≥ `CR_MIN_AGE_DAYS` days), profit factor ≥ `CR_MIN_PROFIT_FACTOR`,
win rate ≥ `CR_MIN_WIN_RATE`, best position ≤ `CR_MAX_TOP_SHARE` of the
profit, median hold ≥ `CR_MIN_HOLD_HOURS`, and profit ≥ `CR_MIN_EDGE` per
dollar of volume. Each gets a score (profit factor, win rate, return on cost,
spread of profit, length of history) and a vote weight of ½ + score. A vote is
then weight × conviction, where conviction is the size of this bet relative to
the trader's usual position (capped at ½–2). The page lists who passed, who
did not and why.

Two such setups run by default:

- **Credible pack** (`credible`): the monthly board's credible traders, acting
  when at least two agree and the weighted agreement minus dissent reaches
  2.5, on markets with 12 hours or more to go, with a 48-hour lookback.
- **Smart money** (`smart`): my own system. One credible trader is enough
  when the bet is a *fresh* conviction entry (they had at most a quarter of it
  before the window), weighted 1.5 or more. Trust in each trader then adapts:
  a copied position that closes with a profit raises their weight by 10%, a
  loss lowers it by 20%, and below 0.5 they are muted. The page's "Who was
  worth following" table shows the result per trader. The idea: real edge in
  prediction markets is mostly idiosyncratic, one informed trader at a time,
  so waiting for a crowd mostly selects in-game sports; a fresh, large bet by
  someone with a proven record, caught early, is the better signal.

### Data model: a system reconstructed from the data

`research/collect.py` downloads the closed positions of hundreds of traders
(every leaderboard plus a control group of ordinary holders of busy markets)
and `research/analyze.py` measures what predicted that a position was worth
following, using only what a follower could know at entry time: the trader's
record *before* that position, the size of the bet for them, the entry price,
days to the market's end, the market type, their specialisation and win rate
in that type, how many others in the pool took the side, their taker share and
their board membership. It prints the tables, fits a small ridge model for the
return and a logistic one for winning on positions closed before a time split,
judges them on the positions closed after it, and writes `polybot/model.json`.
The **Data model** setup (`learned`, `model=1`) scores every signal from every
measured trader on the monthly board with that model and buys when the best
predicted return clears the model's top-quarter cut (`min_pred=` overrides).
It skips sports (`no_sports=1`) and buys between 15¢ and 75¢, because that is
what the data said; the findings are in `research/REPORT.md`.

```
# 1. traders and their finished positions (both passes), threaded; resumable
docker run --rm -v $PWD/research:/r -v /tmp/research2:/out python:3.12-slim sh -c "pip install -q polymarket-client && python /r/collect2.py /out 3000 4"
# 2. optional: closed markets with tags, for better market types
docker run --rm -v $PWD/research:/r -v /tmp/research2:/out python:3.12-slim sh -c "pip install -q polymarket-client && python /r/collect_markets.py /out 2026-01-01"
# 3. the tournament: models × trading rules × walk-forward folds × bootstrap; writes the winner with its policy
docker run --rm -v $PWD:/app -v /tmp/research2:/data python:3.12-slim sh -c "pip install -q numpy scikit-learn && python /app/research/analyze2.py /data --model /app/polybot/model.json --report /app/research/TOURNAMENT.md"
```

Things the data taught us that every script now handles: the API's `CLOSED` set
omits resolved losses (they stay `REDEEMABLE` forever); the cost basis of "No"
positions in multi-candidate (negative-risk) markets is unreliable; a trader's
own exit flatters a follower, so positions are scored at the market's
resolution (a row's current price is the token's final price once the market
ended); leaderboard membership is known today, not back then, so it is not a
feature; and a single win counts for at most +300% when ranking rules.

Two more copies of it run on weather markets only (`only=weather`), at $5
and at $10 a trade, to see where those thin books start to cost more than the
edge is worth; paper fills walk the live order book for the stake, so the
two rows will diverge if the books are too thin for $10.

Pack options for `POLY_VARIANTS`: `pack=1`, `pool=`, `min_weight=`,
`fresh_only=1`, `adaptive_trust=1`, `max_hold_days=` (sell what is still open
after that many days; 0 holds to resolution), `only=` (market types to trade,
comma-separated: weather, politics, geopolitics, crypto, culture, economics,
tech, sports, other). Any setup also takes
`max_per_cycle=` (buys per cycle, strongest signals first; 5 by default, 3 for
the pack setups) so a budget is spread over a day rather than spent at once.

### Edge model (round 3): the follower's price, not the leader's

`research/REPORT3.md` has the findings; in short, round 3 rebuilt the data
model around what a follower can actually get. The **Edge model** setup
(`edge`, `model=1`, `model_file=model.json`) uses it:

- **Who:** everyone measured on the monthly board (the same pool as `learned`),
  judged by a record that counts resolved losses too (CLOSED + REDEEMABLE,
  scored at resolution; `polybot/features.py`). The live `learned` setup reads
  CLOSED only, which put the average leader's win rate at 0.71 instead of 0.52.
- **When:** a single leader's fresh bet is enough (`min_show=1`), but only
  while it is fresh (`max_signal_age_min`, 15): the model was trained on bets
  caught five minutes after they were made, and the edge decays.
- **At what price:** each bet is scored at what a $5 buy costs right now (the
  book walked for the stake), and the model anchors on that price: its win
  chance is sigmoid(logit(price) + trees), so it only buys when the leader adds
  information the market has not priced. The decision quantity is the expected
  return per dollar after the taker fee.
- **How much:** at most one open position per event (`per_event=1`, no more
  "No on both horses"), at most a quarter of the book behind one leader
  (`leader_cap`), plus the threshold, price band, horizon and in-play rule that
  won the walk-forward tournament. All of it ships in `model.json`'s policy.

`polybot/model_r2.json` keeps the round-2 model for `learned`, `weather5` and
`weather10`, so their records stay comparable. Round-3 setup options:
`model_file=`, `min_show=`, `per_event=`, `leader_cap=`, `max_hours_to_end=`,
`max_signal_age_min=`, `no_in_play=1`.

```
# 1. boards, controls, wallets (positions + fills), markets; book snapshots and weather run alongside
python research/collect3.py DATA boards && python research/collect3.py DATA controls 1500
python research/collect3.py DATA wallets 14 && python research/collect3.py DATA markets 6
python research/collect3_books.py DATA 20 48 &
python research/collect3_weather.py DATA markets 2026-05-01   # then obs, forecasts, tapes
# 2. signals, prices, thin-market tapes, panel
python research/build3.py DATA 2 && python research/collect3_prices.py DATA 8 0 0.15
python research/collect3_tapes.py DATA 4 250000 4 20 && python research/fillmodel3.py DATA && python research/prep3.py DATA
# 3. the tournament (writes model.json with its policy) and the weather study
python research/analyze3.py DATA --model polybot/model.json --report research/TOURNAMENT3.md
python research/analyze3_weather.py DATA --late
```

They run in the `plint/polybot` image plus numpy, pandas, pyarrow,
scikit-learn, lightgbm and scipy; `research/common3.py` is the research side's
numpy-free part, so `tests/test_features.py` can prove in the bot's own image
that research and bot compute the same inputs.

## Strategy 2 · Underdog

Filip's observation: the underdog gets a boost once a match begins. Every
30 seconds the bot

1. keeps a schedule of moneyline (match winner) markets starting in the next
   `UD_LOOKAHEAD_HOURS`, with the cheaper side priced between `UD_MIN_PRICE`
   and `UD_MAX_PRICE` (team markets use the cheaper side; soccer-style
   "Will X win?" markets use Yes only),
2. in the window from `UD_BUY_BEFORE_MIN` before kickoff to `UD_BUY_UNTIL_MIN`
   after, **buys** the underdog for `UD_STAKE_USD` if the market is liquid
   enough and the spread is tight enough,
3. **sells** at `+UD_TAKE_PROFIT_PCT` (live: a resting sell order at that
   price; paper: the first trade at or above it), at `−UD_STOP_LOSS_PCT`, or
   `UD_MAX_HOLD_MIN` minutes after kickoff, whichever comes first.

A quick backtest on four days of games (349 underdogs priced 10–45¢, minute
price history): from the first in-game print the underdog was up on average
about 3% after 15 minutes and 5% after 30, and the "+15% target, out after 30
minutes" rule averaged about +1% to +5% per trade before fees, winning about
70% of the time. Sports markets charge a taker fee of about that size, which
is why paper trades charge the fee too. The page will show whether anything is
left.

## Strategy 3 · Forecast

Filip's second idea: Polymarket runs a daily temperature event for about fifty
cities (one market per degree: "Will the highest temperature in Paris be 24°C
on October 30?", with "or higher" / "or below" at the ends, °F ranges for US
cities), resolved on the METAR observations of a named airport. Every half
hour the bot

1. loads the next days' temperature events and finds each one's station,
2. pulls the day's forecast for that airport from Open-Meteo in several weather
   models (`FC_MODELS`) and takes the highest expected maximum across them
   (`FC_AGG`, "the highest expected from the data"),
3. adds `FC_PLUS_C` (2°C) and buys **No** on the degree bucket that would
   resolve for that temperature, for `FC_STAKE_USD`, if No is between
   `FC_MIN_PRICE` and `FC_MAX_PRICE`, before the day has started in that city;
   the surest forecasts (models agree) go first, `FC_MAX_PER_TICK` a tick,
4. holds to resolution, or sells if a fresh forecast climbs to within half the
   margin of the bucket (`FC_EXIT_IF_FORECAST_MOVES`). `FC_LOWEST=1` also
   plays the lows (No on the forecast low minus the margin); `FC_DAYS_AHEAD`
   (1) is how far ahead it bets, in the city's own calendar.

## Strategy 4 · Collector

No forecasting at all. Every daily temperature event resolves on the station's routine
METAR reports: the market's description spells the rule out (the hourly temperature
remark in tenths of °C, else the body temperature; SPECI reports don't count; nearest
whole degree with halves up, converted to °F first for US cities; an observation
belongs to the local date its time stamp falls on) and names the Aviation Weather
Center API as the source. The bot reads the same reports (`polybot/metar.py`) the same
way (`polybot/collector.py`). Once a city's high is set, from `CL_LOCK_HOUR` (20:00
local) with the latest reading `CL_LOCK_MARGIN_C` below it, or once the day is over
(lows only then), the winning bucket is known, and its Yes and the other buckets' No
pay $1 when the next day's first report settles the event. The tapes temper the idea
(`research/weather_lock.py`): by the time the reports make a high certain the market
is already there (the first buy after a lock is at 99.7–99.9¢; live books hold no
asks, only bid walls at 99.9¢), and what trades at 97.5¢ or less sits in days that
were still uncertain and returns about zero after the fee. The one band with a small
positive is 95–97.5¢ after a 16:00 lock with a 2°C margin: +2 to +3% on a few trades
a week, wrong on 0.14% of days. That is what the defaults test. Every
`CL_TICK_SECONDS` the bot

1. loads the three days of temperature events a city's clock can be on and reads each
   event's station and rule from its description (cities on another source, such as
   Hong Kong's Observatory, are skipped; so is anything in `CL_SKIP_CITIES`),
2. fetches the reports of the stations whose day is in its window,
3. buys every token that is now worth $1 (the winner's Yes, the losers' No) for
   `CL_STAKE_USD` when it costs between `CL_MIN_PRICE` and `CL_MAX_PRICE` (below the
   minimum the market disagrees with the reports, so something is off; above the
   maximum there is too little to gain), `CL_MAX_PER_TICK` a tick, within the budget,
4. holds to resolution. A silence of more than `CL_MAX_GAP_MIN` from the station blocks
   a buy (a report may still turn up). A lost position means the rule was read wrong:
   it is reported loudly and the city is skipped for 30 days.

`python -m polybot collector` scans the finished days without buying and prints what
it sees.

## Strategy 5 · Afternoon

Filip's idea: it is usually hottest in the middle of the day, so by mid-afternoon the
day's high is probably in. The study (`research/weather_hasmax.py`) says the high is
first reached by 14:00 on half the days, by 16:00 on 84%, and that the market sells
the running high's bucket a few points too cheap in the early afternoon (real asks at
15:00: paid 27→held 29, 42→45, 58→61, 73→76). A model of the state adds nothing beyond
the price, so the rule is the plain one: at the city's hour the bot fixes the bucket
the day's running high falls in (the station's routine reports, read the way the
markets' rule reads them, shared with the collector) and buys its Yes for
`AF_STAKE_USD` when the real ask is between `AF_MIN_PRICE` and `AF_MAX_PRICE`, within
that hour, once per city and day, held to resolution. Every city has its own hour
(Filip's point; `research/weather_hour.py`): the high lands at 11:30 in Taipei and
17:00 in Madrid, and timing each city at the hour by which its own high had landed on
60% of its earlier days made +10.4% a trade (90% CI +7.3 to +13.4, 1,790 trades)
against +5.6% for 15:00 everywhere; buying No instead lost 22%. The hour is the
`AF_QUANTILE` point of when the high was first reached over the station's last
`AF_HISTORY_DAYS` days (IEM archive, a few stations a tick, each refreshed daily),
`polybot/afternoon_hours.json` (from the study) until `AF_MIN_HISTORY` complete days
are in, and `AF_HOUR` when neither exists. Hong Kong is skipped (`AF_SKIP_CITIES`): it
resolves on the Observatory, not the airport.

`python -m polybot afternoon` scans the cities in their window without buying.

## Comparing them

The page's **Strategies** table lists every setup: paper (or live) P&L,
number of closed trades, win rate and average return per trade, with what is
open underneath. The consensus section has a picker to look at any variant's
leaders and signals. Positions and trades carry a strategy tag. Everything is
tracked per strategy in `data/polybot/state.json` and `trades.jsonl`.

## Paper first

Without a key the bot runs in **paper mode**: the same decisions, no orders,
fills at the prices the order book shows at that moment, minus Polymarket's
taker fee. Leave it there for a week or two before going live.

## Going live

1. On polymarket.com → profile menu → copy the **wallet address**.
2. Export the account's **private key** (Settings → Export private key for
   email/Google logins; for a browser wallet it is that wallet's key).
3. Optional, for claiming winnings automatically: Settings → API Keys →
   **Relayer API Key** (copy the key and its signer address).
4. In `/opt/plint/.env`:
   ```
   POLY_PRIVATE_KEY=0x…
   POLY_WALLET=0x…
   POLY_RELAYER_API_KEY=…            # optional
   POLY_RELAYER_API_KEY_ADDRESS=0x…  # optional
   POLY_LIVE=1
   ```
5. `plint restart poly`, then `docker exec polybot python -m polybot check`
   should show the wallet type, the pUSD balance and `approved: true`.

The bot only spends pUSD that is already in the account; it cannot withdraw.
Going live applies to both strategies; set `UD_ENABLED=0` to run only the
consensus one.

## Settings (`.env`, all optional)

Consensus:

| variable | default | meaning |
|---|---|---|
| `POLY_WINDOW` | `week` | leaderboard window: `day`, `week`, `month`, `all` |
| `POLY_CATEGORY` | `overall` | or a Polymarket category such as `politics`, `sports`, `crypto` |
| `POLY_TOP_N` | `25` | how many leaders to watch |
| `POLY_MIN_AGREE` | `5` | agree minus against needed to trade |
| `POLY_LOOKBACK_HOURS` | `24` | how far back a leader's fills count |
| `POLY_MIN_LEADER_USD` | `25` | a leader's stance below this is noise |
| `POLY_STAKE_USD` | `5` | per trade (Polymarket's minimum order is about $5) |
| `POLY_MAX_PRICE` / `POLY_MIN_PRICE` | `0.90` / `0.05` | only buy inside this band |
| `POLY_MAX_ABOVE_LEADERS` | `0.05` | never pay more than this above the leaders' average |
| `POLY_MAX_OPEN` | `20` | open positions at most |
| `POLY_DAILY_BUDGET_USD` | `50` | buys per rolling 24 h at most |
| `POLY_COOLDOWN_HOURS` | `72` | don't re-enter the same outcome sooner |
| `POLY_MIN_HOURS_TO_END` | `1` | skip markets that end sooner |
| `POLY_FOLLOW_SELLS` | `1` | sell when the leaders sell |
| `POLY_AUTO_REDEEM` | `1` | claim winnings (needs the Relayer key) |
| `POLY_POLL_SECONDS` | `300` | how often to look |
| `POLY_VARIANTS` | the built-in set above | extra setups to compare |
| `CR_POOL` / `CR_PACK_SIZE` / `CR_REFRESH_HOURS` | `80` / `12` / `6` | credible pack: how many to measure, keep, how often |
| `CR_MIN_FILLS_7D` / `CR_MAX_IDLE_HOURS` | `10` / `72` | active right now |
| `CR_MIN_CLOSED` / `CR_MIN_AGE_DAYS` | `30` / `30` | a real history |
| `CR_MIN_PROFIT_FACTOR` / `CR_MIN_WIN_RATE` | `1.3` / `0.40` | quality of that history |
| `CR_MAX_TOP_SHARE` | `0.50` | not one lucky bet |
| `CR_MIN_HOLD_HOURS` / `CR_MIN_EDGE` | `12` / `0.02` | not an in-game scalper or a market maker |

Underdog:

| variable | default | meaning |
|---|---|---|
| `UD_ENABLED` | `1` | run the strategy at all |
| `UD_SPORTS` | *(all)* | comma-separated slug prefixes to allow, e.g. `nba,nhl,epl,wta` |
| `UD_STAKE_USD` | `5` | per game |
| `UD_TAKE_PROFIT_PCT` | `15` | sell this much above entry |
| `UD_STOP_LOSS_PCT` | `30` | or this much below |
| `UD_MAX_HOLD_MIN` | `30` | or this many minutes after kickoff |
| `UD_MIN_PRICE` / `UD_MAX_PRICE` | `0.10` / `0.45` | what counts as an underdog |
| `UD_MAX_SPREAD` | `0.03` | skip wider books |
| `UD_MIN_LIQUIDITY` | `5000` | skip thinner markets |
| `UD_BUY_BEFORE_MIN` / `UD_BUY_UNTIL_MIN` | `3` / `2` | the buying window around kickoff |
| `UD_MAX_OPEN` | `6` | open games at most |
| `UD_DAILY_BUDGET_USD` | `50` | buys per rolling 24 h at most |
| `UD_LOOKAHEAD_HOURS` | `3` | games this far ahead are watched |
| `UD_TICK_SECONDS` | `30` | how often to look |

Change a value, then `plint restart poly`.

## Files

- `polybot/core.py` — Polymarket calls, paper and live orders, resting sells, settlement, fees
- `polybot/consensus.py` + `strategy_consensus.py` — the agreement logic (pure, tested) and its strategy, variants included
- `polybot/credibility.py` + `pack.py` — who is worth copying (pure rules, tested) and the measuring of a board
- `polybot/model.py` + `model.json` (round 3) / `model_r2.json`, `polybot/features.py`, `research/` — the data models, the shared feature code and the scripts that collected and fitted them
- `research/REPORT4.md` — round 4 (2026-11-10): makers, anchors and carry; no edge found, the `carry` rule is specified there but not built (`research/r4_carry_live.py` lists what it would hold)
- `polybot/underdog.py` + `strategy_underdog.py` — the underdog rules (pure, tested) and its strategy
- `polybot/weather.py` + `strategy_forecast.py` — temperature markets, stations, Open-Meteo forecasts (pure parts tested) and the forecast strategy
- `polybot/metar.py` + `collector.py` + `strategy_collector.py` — the station reports read by the markets' rule, when a day's value is final (tested), and the collector strategy
- `polybot/strategy_afternoon.py` — the afternoon strategy (Yes on the running high's bucket at 15:00 local)
- `polybot/runner.py` — runs both on their own clocks, writes the page's JSON
- `polybot/store.py` — `data/polybot/state.json`, `trades.jsonl`, `status.json`
- `static/index.html` — the page (Plint Soft UI tokens)

Commands: `docker exec polybot python -m polybot once` runs one cycle of every
strategy and prints their reasoning; `… check` shows the account state; `… collector` scans the finished temperature days without buying; `… afternoon` scans the cities in their afternoon window.
`… ledger [base]` settles the ledger (`data/polybot/ledger.jsonl`: every signal that passed a consensus setup's rules, at
the real price of a $5 buy, bought or not) and compares setups per signal against `base` (default `day4`), paired by day.
Tests: `docker run --rm -v $PWD:/app -w /app python:3.12-slim sh -c "pip install -q -r requirements.txt && python -m unittest discover -s tests"`.

Uses the official [`polymarket-client`](https://pypi.org/project/polymarket-client/)
SDK; data from Gamma and `data-api.polymarket.com`, orders through the CLOB.
