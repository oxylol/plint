# Stocks: the design

Written 2026-11-14, before a single real price was fetched. Every number below was chosen then. That is the
point of writing it down first: a rule tuned after looking at ten years of results will always look good on
those ten years.

## The question

Can any of these models beat simply holding the index, after costs, on paper? And how soon can that be told
from luck?

Polybot asked the same of prediction markets through four rounds and found no edge at real prices. Shares are
a harder market than that. So the likely answer here is "no, or not by enough", and the lab is built to give
that answer honestly rather than to find a way around it.

## The rules every model lives under

**One clock.** Each US trading day at 16:25 New York time (22:25 in Sweden): prices in, books marked, each
model decides. What it decides is filled at the next day's open (15:30 in Sweden).

**What a model may see.** The price table up to that evening's close and nothing after. This is not left to
care: a model is handed a table that ends on the evening in question (`Panel.upto`), and `tests/test_leak.py`
replaces everything after a day with noise and requires the same figures and the same orders.

**The price table.** Raw daily bars as they traded, plus each day's total return (splits and dividends in).
Raw prices never change afterwards, so the table can be added to day by day and stay true. The last two weeks
are fetched again every evening; a correction on the source's side is applied and logged on the page.

**The universe.** At each month end, the 1,000 most-traded common stocks on NYSE, Nasdaq and NYSE American
at $5 or more (median daily dollar volume over three months), for the month that follows. Funds, warrants,
units and preferred shares are kept out by name. Picked only from what was known at that month end.

**The book.** $10,000 of pretend money each. Long only, no borrowing, no shorting, cash earns nothing.
A book is not stored: it is replayed from the model's logged decisions over the price table every evening,
so a corrected price or a fixed bug reaches every figure. The first decision of an evening stands; running
the evening twice changes nothing.

**Costs.** Per dollar traded at the open: half the spread (1 basis point for the busiest stocks up to 20 for
the thinnest, read off the dollars traded in them), 1 basis point of slip, and the push of the order itself
(daily volatility × √(order ÷ a day's trading)). Reports also show the result with costs doubled and tripled,
and with a Swedish broker's 0.25 % currency fee on every trade.

**One holding rule** for every model that ranks stocks (`portfolio.TopK`): hold 40, each 1/40 of the book;
keep a holding while it still ranks in the top fifth, and for five days at most unless it is still one of the
top 40; change at most 8 names a day. The models differ in what they know, not in how their book was tuned.
(The five-day limit came from Filip on 2026-11-14, "the network should probably not hold for long", before
any full backtest existed. Five days is what the ranking speaks about. The rule without the limit is kept in
the backtest as a yardstick: the network's report has a table of the same fits at three paces.)

## How a model is judged

Four ways, slowest to fastest, and they must not be confused:

1. **Backtest** (2019-01-02 to 2025-05-13): history replayed with the live code. The learned models are
   refitted walk-forward every six months, each fit seeing only outcomes complete before its day.
2. **Exam** (2025-05-14 onward): the last eighteen months, held back. Run once, after the design is frozen.
   Every run is counted and the count is printed in every report.
3. **Live paper**: the only thing that cannot be fooled, and the slowest. With a few percent a year of edge
   against ten percent a year of wobble around the index, profit alone needs a decade to reach t = 2.
4. **The ledger**: every evening's ranking of 1,000 stocks is scored against what the stocks then did (rank
   correlation, the "IC"). A real daily IC of 0.02 shows at t = 2 in about three months. This is the lamp on
   the page: *likely* at t ≥ 2 with at least 40 evenings scored, *unlikely* when even mean + 2 standard errors
   is under 0.005, otherwise *no verdict*. The t is Newey-West, because 5-day outcomes of neighbouring
   evenings overlap.

A few months of paper can show that a model is broken. It cannot show that it is good.

## The contestants

| | what it does | why it is here |
|---|---|---|
| **index** | buys SPY once, holds | the bar |
| **net** | the neural network: ranks the universe, holds the top 40 | what was asked for |
| **netfast** | the same network on its one-day view: holdings last a day or two | Filip: it may trade more often if that pays. This shows whether it does |
| **trees** | LightGBM on the network's inputs | the usual winner on tables of figures: the rival |
| **ridge** | a linear fit on the same inputs | the floor: what the network adds is what it has over this |
| **blend** | average of the three rankings | an average is usually steadier than its parts |
| **agent** | Claude reads filings and news, trades ≤ 4 names a day | what was asked for |
| **momentum** | past year's winners, last month skipped, monthly | Jegadeesh & Titman 1993; also checks the machinery |
| **reversal** | last week's losers against their own volatility, a week | Lehmann 1990; lives or dies on costs |
| **drift** | buys the day after a strong reaction to results, 40 days | Bernard & Thomas 1989; slow, a small account can reach it |
| **insiders** | ≥ 2 officers or directors buy within a month, 60 days | Lakonishok & Lee 2001; purchases, in clusters |
| **trend** | SPY above its 200-day average, else Treasury bills | Faber 2007; a risk rule, not a stock picker |

## The neural network

**What it reads.** Fifty-six figures per stock per evening, all from price and volume (`features.py`): how
far it has moved over 1 to 252 days, overnight and in-session moves apart, volatility at four spans and its
shape, its slope against the index and what is left over, dollars traded and how unusual today's volume was,
distance from its highs, lows and averages, where in the day's range it closed, the same month a year ago.
Each is turned into the stock's place among that day's 1,000 (−1 lowest, +1 highest), so no figure's scale or
drift over the years matters. Plus eighteen figures about the market that day.

**What it is asked.** To order the stocks by how they will do against each other from the next open: over 1,
5, 10 and 20 days at once, each return taken over the universe's average and divided by the stock's own
volatility before ranking (so "did well for its temperament", not "is wild"). The loss is the day-by-day
correlation between its scores and those ranked outcomes. The 5-day head is the one that trades.

**What makes it a network rather than trees.** It reads the whole day at once. Each stock is encoded alone;
sixteen learned slots then attend over all thousand encodings together with the market's row, and each stock
reads the slots back (an induced set-attention block). A stock's score can so depend on what the rest of the
market looks like that evening. Trees cannot see that.

**Whether that helps is measured, not assumed.** The report shows a ladder at the 5-day horizon: ridge, trees,
the network with each stock alone, with the day's plain average, and with the slots. Each step up has to
earn its place. If the slots do not beat "each stock alone" on the design stretch, the plainer network trades
and the report says so.

**Sizes.** About 100,000 weights, five seeds averaged live (three in the backtest). Small on purpose: the
signal in daily returns is faint, and averaging seeds does more for a faint signal than one larger network.
It also fits Plint: four cores, no usable GPU.

**Training.** Walk-forward. The last half year before a fit is set aside to choose how long to train, with a
gap of 22 days before it and after it so no outcome crosses a boundary; then the model is trained again on
all of it for that long. Live it is refitted every week.

**Not in it, on purpose, for now.** Fundamentals from filings (they matter at monthly horizons more than at
five days, and the tags are a swamp) and a sequence branch over raw daily bars (ten times the compute; it
gets one honest trial on a single split before it earns a place). Both are next steps, not promises.

## The agent

**Each evening** (`agent/desk.py`):
1. *Triage*, small model (Haiku 5.5): every filing and headline about a universe company since the last run,
   plus the day's largest moves. Material or not, which way, how sure.
2. *Deep reads*, main model (Opus 5.5): at most 30 companies. The filing itself, the stock's recent history,
   up to two web searches. It must say what is already in the price, then state the chance the stock beats
   the index over 5 and 20 days.
3. *Decide*, main model: sees its book, tonight's reads, the market and its own playbook. Returns orders and
   a memo. Code then enforces the limits whatever it said: at most 15 names, 12 % each, 4 new a day, long
   only, universe only.
4. Weekly it reads its own closed trades and settled forecasts and rewrites its playbook.

**The budget.** $200 a calendar month, a hard stop: before every call its worst case is added to the month's
spend and the call is not made if it would cross. A day's allowance is what is left divided by the trading
days left, times 1.5, and never more than 1.5 even days of the month (about $14): the cap is a ceiling, not a
target, so a month begun late is not made up for by spending faster. When money runs short it gives up
searches first, then deep reads, then the decision. Measured on the first real evening (14 Nov 2026, three
reads): screening 139 items cost under a cent, a read 4 to 24 cents (the dearer ones are those that search
the web), the decision 4 cents.

**It cannot be backtested.** The model has read about the past: any run over history leaks. It is judged on
live paper and on its stated chances (every deep read is a forecast, traded or not), from the day it starts.

**Where its edge could be, if anywhere.** Not speed: headlines are read by machines in milliseconds. Slow
information: what a filing changes about the next quarters, and whether the price has moved enough.

**Untrusted text.** Filings and headlines are data, never instructions; every prompt that carries them says
so, and no order can be placed from triage output alone.

## The traps, and what closes each

| trap (most of them met in Polybot) | what closes it here |
|---|---|
| the live rule differs from the simulated one | one `decide()` and one `Book.step()` for both; `test_leak` |
| lookahead | models are handed tables that end that evening; outcomes live in one function (`learn.dataset.forward`) |
| pricing at a price nobody could get | fills at the next open plus costs, never at the close that made the signal |
| universe built from survivors | membership picked month by month from what was known then |
| tuning on the test set | declared design, the exam held back and counted, `stocks variant` logs every change |
| the best of ten cells | one holding rule for all rankers; the ladder shows every rung, not the best one |
| overlapping outcomes read as independent | Newey-West t in the ledger; block bootstrap for the yearly edge |
| an adjusted price series rewritten under you | raw bars plus total return; the tail re-pulled and corrections logged |
| a split that looks like a crash | the adjusted series is the judge; a late split is fixed only from the action list |
| one company counted twice | Alpaca answers for a renamed company under both tickers (META's bars reach back through FB's years, and FB's are those years again): a symbol whose whole history is another's is kept out (`ingest.aliases`) |
| events known only for survivors | the SEC's ticker list names today's companies only; the results days and insider purchases of companies since bought or delisted are found through the tickers their insiders filed under, each for its own years (`filings.links`) |

## What is frozen, and how it may change

Frozen now: the universe rule, the cost model, the holding rule, the 56 figures, the targets and loss, the
network's shape and sizes, the fit schedule, every rule model's numbers, the ledger's lamp, the two stretches.

A change after seeing design-stretch figures is allowed, once it is written down (`python -m stocks variant
"what and why"`); the list is printed in every report. The first entry is the five-day limit above: it was made
after a trial of the machinery on a 150-stock sample had been looked at, so it is on the list, though it did
not come from those figures. After the exam has been run, a change to a model
starts its record over (`python -m stocks reset <model>`).

## What this cannot show

- **Vanished companies.** Free data is patchy on shares that stopped trading. Companies bought out since 2020
  and companies pushed off their exchange are in the table (found through Alpaca's corporate-action list and
  its over-the-counter symbols); many that left before 2020 are not, so the early years of every stock picker's
  history are flattered a little. The index's are not. A holding that collapses and then stops trading is
  written off; one that simply stops is cashed in at its last price. Live paper has no such problem.
- **The auction.** Fills use the first price of the day on the tape. A real order at the open gets the
  auction's price: near it, not equal.
- **Size.** At $10,000 a book the orders move nothing. The cost model has a term for larger books; it is a
  rule of thumb, not a measurement.
- **Going live.** There is no broker in this app. Nordnet's API is closed to new customers; a real version
  would be its own project.
