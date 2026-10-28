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
base threshold and at four agreeing, plus the two credible-pack setups below:

```
week4:min_agree=4;month:window=month;month4:window=month,min_agree=4;day:window=day;day4:window=day,min_agree=4;all:window=all;all4:window=all,min_agree=4;credible:pack=1,window=month,min_agree=2,min_weight=2.5,lookback_hours=48,min_hours_to_end=12,max_per_cycle=3,label=Credible pack;smart:pack=1,window=month,min_agree=1,min_weight=1.5,fresh_only=1,adaptive_trust=1,lookback_hours=48,min_hours_to_end=12,max_per_cycle=3,label=Smart money
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
- `polybot/model.py` + `model.json`, `research/` — the data model and the scripts that collected and fitted it
- `polybot/underdog.py` + `strategy_underdog.py` — the underdog rules (pure, tested) and its strategy
- `polybot/runner.py` — runs both on their own clocks, writes the page's JSON
- `polybot/store.py` — `data/polybot/state.json`, `trades.jsonl`, `status.json`
- `static/index.html` — the page (Plint Soft UI tokens)

Commands: `docker exec polybot python -m polybot once` runs one cycle of both
strategies and prints their reasoning; `… check` shows the account state.
Tests: `docker run --rm -v $PWD:/app -w /app python:3.12-slim sh -c "pip install -q -r requirements.txt && python -m unittest discover -s tests"`.

Uses the official [`polymarket-client`](https://pypi.org/project/polymarket-client/)
SDK; data from Gamma and `data-api.polymarket.com`, orders through the CLOB.
