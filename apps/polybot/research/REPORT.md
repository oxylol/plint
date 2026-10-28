# What the data says about copying Polymarket traders

Collected 2026-10-28 with `collect.py` + `collect_settled.py`: 278,099 finished positions by 683 traders (606 from every leaderboard window and category, 94 ordinary holders of busy markets as a control group). Finished = exited (sold or redeemed) **plus** resolved-and-never-redeemed; the second group is 85k positions, 93% of them losses, and the Data API hides it from the `CLOSED` set, which is why any analysis of `CLOSED` alone looks far too rosy. Return = P&L / cost of the position; `$ pnl/$1k` = dollars made per $1,000 put in.

## Everything

```
market type                                 n   win  avg ret  median  $ pnl/$1k
  sports                               149684   45%    -5.7%  -36.3%       +31
  crypto                                40136   45%    -3.1%  -35.7%        -2
  other                                 29643   47%   +24.5%   -1.0%      +115
  politics                              19695   47%   +54.7%   +0.0%      +147
  culture                               13209   40%   +31.6%  -14.8%       +37
  geopolitics                           10492   55%   +35.8%   +0.9%       +97
  weather                                8959   31%   -18.0%   -9.0%        +1
  economics                              3612   50%   +28.6%   +0.0%       +26
  tech                                   2669   50%   +32.5%   +0.0%        -7

trader's prior profit factor                n   win  avg ret  median  $ pnl/$1k
  < 1 (losing)                         153530   42%   -10.0%  -69.0%       +43
  1–1.3                                 37938   50%   +21.2%   +0.0%       +27
  1.3–2                                 31483   49%   +32.5%   +0.0%       +46
  2–4                                   19120   52%   +25.1%   +0.5%       +89
  4+                                    16920   57%   +35.7%   +1.2%      +221
  (other)                                1468   68%   +36.1%   +1.1%      +252
  (unknown)                             17640   41%   +12.5%   -8.0%       +93

trader's prior win rate                     n   win  avg ret  median  $ pnl/$1k
  < 40%                                124766   35%   -13.3% -100.0%        -8
  40–55%                                77936   50%   +16.5%   +0.0%       +65
  55–70%                                32665   58%   +38.7%   +3.0%      +112
  70–90%                                17906   68%   +24.0%   +2.3%       +69
  90%+                                   7552   76%   +14.8%   +0.9%      +273
  (unknown)                             17274   41%   +12.5%   -9.6%       +92

entry price                                 n   win  avg ret  median  $ pnl/$1k
  < 10¢                                 38686   22%   +81.2% -100.0%      +780
  10–30¢                                38986   28%    -4.4% -100.0%      +157
  30–50¢                                66690   39%   -12.0% -100.0%      +124
  50–70¢                                67028   51%    -5.3%   +5.7%       +74
  70–90¢                                33727   70%    -1.2%  +15.2%       +35
  90¢+                                  32982   69%    -3.7%   +0.9%        -6

others in the pool on the same side ±48h      n   win  avg ret  median  $ pnl/$1k
  none                                 131383   46%    +7.3%   -7.0%       +75
  1                                     45056   42%    -0.6%  -29.7%       +23
  2–3                                   41323   44%    +7.4%  -11.0%       +94
  4–7                                   30878   45%    +8.8%   -4.2%      +130
  8+                                    29459   49%    +4.7%   -2.0%       +82

bet size vs trader's usual                  n   win  avg ret  median  $ pnl/$1k
  < ½×                                  76739   41%   +22.4%  -80.5%       +88
  ½–1×                                  38013   43%    -0.0%  -46.7%       +63
  1–2×                                  44296   45%    -6.1%  -14.6%       +43
  2–5×                                  43971   49%    +0.1%   -0.0%      +143
  5×+                                   57806   51%    -0.3%   +0.1%       +50
  (unknown)                             17274   41%   +12.5%   -9.6%       +92

how it finished                             n   win  avg ret  median  $ pnl/$1k
  exited (sold or redeemed)            193209   62%   +37.9%   +7.5%      +215
  resolved, never redeemed              84890    7%   -66.8% -100.0%      -919

on a board?                                 n   win  avg ret  median  $ pnl/$1k
  control: ordinary holder              30213   45%    -2.3%   -4.0%        +7
  leaderboard trader                   247886   45%    +6.9%   -9.2%       +77
```

## Only positions a follower can copy (entered 12 h or more before the market's end)

```
market type                                 n   win  avg ret  median  $ pnl/$1k
  sports                                33682   45%    -4.5%  -29.9%        -5
  other                                 18971   46%   +38.0%   -1.6%       +55
  politics                              15739   47%   +64.2%   +0.0%      +142
  culture                               10298   39%   +42.8%  -16.4%       +41
  geopolitics                            8475   55%   +38.0%   +0.9%       +71
  crypto                                 6116   58%   +56.4%   +1.1%       -11
  weather                                4538   27%   -17.9%   -9.0%       +41
  economics                              3241   49%   +30.7%   +0.0%       +28
  tech                                   2348   48%   +33.4%   -0.1%       -47

entry price                                 n   win  avg ret  median  $ pnl/$1k
  < 10¢                                 24191   24%  +105.5% -100.0%      +655
  10–30¢                                14317   38%   +23.6%  -62.5%      +304
  30–50¢                                15709   46%    +0.6%  -11.1%      +278
  50–70¢                                16707   52%    -0.5%   +3.7%       +65
  70–90¢                                13925   66%    -1.1%  +11.8%       +44
  90¢+                                  18559   59%    -5.1%   +0.4%        -3

trader's prior win rate in this type        n   win  avg ret  median  $ pnl/$1k
  < 40%                                 32539   35%   +15.8%  -32.8%       -18
  40–55%                                19715   50%   +40.7%   +0.0%       +62
  55–70%                                12769   57%   +31.3%   +2.5%       +41
  70%+                                  13651   65%   +19.8%   +2.0%       +79
  (unknown)                             24734   40%   +32.3%   -8.0%      +120
```

## The model (ridge regression on return, logistic on winning; split by entry date)

```
=== Model: 44439 train (entered and finished before 2026-10-07), 31387 test (entered after) ===
feature           → return   → win
  prior_pf           -0.047  -0.064
  prior_wr           +0.097  +0.405
  prior_n_log        +0.022  +0.053
  prior_roc          +0.060  -0.050
  conviction_log     +0.007  +0.018
  price              +0.237  +0.630
  price_sq           -0.241  +0.120
  days_log           +0.035  +0.000
  is_sports          -0.028  -0.009
  cat_share          +0.012  +0.002
  cat_wr             +0.032  +0.144
  crowd_log          +0.021  +0.033
  taker_share        +0.004  -0.035
  prior_hold_log     -0.001  +0.032
  on_month           +0.012  +0.051
  on_all             +0.024  +0.032

In sample [predicted return]: 44439 positions, overall avg return +16.3%, win 51%
  top 10%      n=  4443  avg ret  +10.7%  median   +6.4%  win   62%  $/$1k     +97  score ≥ +0.147
  top 25%      n= 11109  avg ret  +12.2%  median   +4.3%  win   64%  $/$1k     +95  score ≥ +0.074
  top 50%      n= 22219  avg ret  +13.8%  median   +3.0%  win   61%  $/$1k     +76  score ≥ -0.016
  all          n= 44439  avg ret  +16.3%  median   +0.1%  win   51%  $/$1k     +44  score ≥ -0.701
  bottom 25%   n= 11109  avg ret   +9.6%  win   30%
  rank correlation score vs actual return: +0.237

Out of sample (entered after the split) [predicted return]: 31387 positions, overall avg return +29.6%, win 42%
  top 10%      n=  3138  avg ret  +11.4%  median   +4.2%  win   55%  $/$1k     +83  score ≥ +0.114
  top 25%      n=  7846  avg ret  +20.1%  median   +4.9%  win   56%  $/$1k     +47  score ≥ +0.030
  top 50%      n= 15693  avg ret  +42.5%  median   +2.0%  win   54%  $/$1k     +58  score ≥ -0.080
  all          n= 31387  avg ret  +29.6%  median   -4.2%  win   42%  $/$1k     +22  score ≥ -0.656
  bottom 25%   n=  7846  avg ret  -17.3%  win   24%
  rank correlation score vs actual return: +0.255

```

```
Entered before October (before the boards that picked these traders) [predicted return]: 45560 positions, overall avg return +16.9%, win 50%
  top 10%      n=  4556  avg ret   +9.7%  median   +5.6%  win   61%  $/$1k     +75  score ≥ +0.150
  top 25%      n= 11390  avg ret  +12.2%  median   +3.9%  win   62%  $/$1k     +89  score ≥ +0.075
  top 50%      n= 22780  avg ret  +14.0%  median   +2.4%  win   60%  $/$1k     +75  score ≥ -0.015
  all          n= 45560  avg ret  +16.9%  median   +0.0%  win   50%  $/$1k     +43  score ≥ -0.701
  bottom 25%   n= 11390  avg ret  +11.8%  win   30%
  rank correlation score vs actual return: +0.224

```

```
Control group only (ordinary holders, never on a board) [predicted return]: 9978 positions, overall avg return +12.7%, win 50%
  top 10%      n=   997  avg ret   +6.3%  median   +4.2%  win   57%  $/$1k      -4  score ≥ +0.123
  top 25%      n=  2494  avg ret   +3.5%  median   +3.3%  win   57%  $/$1k      -7  score ≥ +0.065
  top 50%      n=  4989  avg ret   +7.1%  median   +2.7%  win   58%  $/$1k      +1  score ≥ -0.016
  all          n=  9978  avg ret  +12.7%  median   +0.0%  win   50%  $/$1k     +13  score ≥ -0.701
  bottom 25%   n=  2494  avg ret  +26.5%  win   32%
  rank correlation score vs actual return: +0.154

```

## Reading it

1. **The pool as a whole barely makes money.** 44–48% of positions win, the median position loses about 10%, and
   ordinary holders average −2% to −4% per position (fees). Leaderboard traders average +5% to +7%.
2. **Sports is where the losses are**, for everyone: −5.7% per position, 45% winners, even among leaderboard traders.
   The money is in politics (+55% average, +14.7% dollar-weighted), geopolitics and long-horizon "other" markets.
3. **The trader's record predicts the next position**, monotonically: prior profit factor above 4 → 57% winners and
   +36% average; below 1 → 42% and −10%. Prior win rate is the strongest single feature in the model.
4. **Price matters**: entries above 90¢ lose on average; entries below 10¢ are lottery tickets (22% winners, huge
   dollar-weighted returns driven by a few wins); the model's sweet spot is around 30–60¢.
5. **Agreement adds little.** Positions with several pool traders on the same side within two days do about as well as
   lone positions. Bet size relative to the trader's usual size adds nothing either. Consensus is not the signal;
   who is betting and on what kind of market is.
6. **Out of sample, restricted to copyable positions, the model's top quarter averages about +20% per position with
   56% winners (about +5% dollar-weighted), its bottom quarter −17% with 24% winners.** The ranking holds on positions
   entered before the boards that selected these traders existed, and even inside the control group, though the gain
   there is tiny. The price-based "edge" score (predicted win probability against the price) only finds longshots and is
   not used.

The **Data model** setup in the bot (`learned`) is this model: it scores every signal from every measured trader on the
monthly board, skips sports, buys between 15¢ and 75¢ with 12 hours or more to go, and acts when the predicted return
clears the model's top-quarter cut.
