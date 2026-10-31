# Round 3 tournament

840310 resolved signals with a follower price, 6563 wallets, 98726 markets, 2026-08-01 → 2026-10-31; 51290 from the rebuilt monthly top 200 (the bot's pool)
9 weekly folds from Sep 01; the last one holds the live paper period (Oct 28 →)
  fold Sep 01: train 101125 (used 101125), test 40645  (97s)
  fold Sep 08: train 141720 (used 141720), test 49507  (40s)
  fold Sep 15: train 187430 (used 187430), test 53616  (49s)
  fold Sep 22: train 241092 (used 241092), test 57586  (260s)
  fold Sep 29: train 296409 (used 250000), test 79918  (58s)
  fold Oct 06: train 381338 (used 250000), test 88432  (55s)
  fold Oct 13: train 467331 (used 250000), test 105405  (238s)
  fold Oct 20: train 569539 (used 250000), test 139665  (64s)
  fold Oct 27: train 708568 (used 250000), test 110474  (64s)

## How well each model predicts (out of sample, all folds)

Calibration slope 1 and intercept 0 are perfect; below 1 means over-confident. 'top-decile ret' is the mean return per $ of the 10% of signals with the highest expected return (P/price − 1 − fee), capped at +300%.

model                   n  logloss  Δ vs mkt   Brier    AUC  slope   icpt top-dec ret pool top-dec
market             725248   0.5638   +0.0000  0.1925  0.772   1.01  +0.01       -8.2%        -6.2%
skill_eb           725248   0.5635   -0.0003  0.1924  0.773   0.99  -0.05       -5.0%        -8.7%
logit_anchor       725248   0.5631   -0.0007  0.1923  0.773   1.00  +0.02      -10.0%       -10.6%
gbm_raw            725248   0.5627   -0.0011  0.1921  0.774   1.00  +0.02       -7.4%       -10.9%
gbm_anchor         725248   0.5623   -0.0015  0.1920  0.774   1.00  +0.02       -3.9%       -11.7%
gbm_decay30        725248   0.5623   -0.0015  0.1920  0.774   0.99  +0.02       -2.2%        -9.3%
gbm_percat         725248   0.5649   +0.0010  0.1930  0.771   0.96  +0.02       -5.3%        -8.3%
gbm_wallet_te      725248   0.5623   -0.0016  0.1919  0.774   1.00  +0.02       -3.1%        -9.1%
rule_skill         725248   0.5731   +0.0093  0.1950  0.771   1.09  +0.19       -6.3%        -4.1%
rank               725248   0.5666   +0.0028  0.1933  0.772   0.90  +0.07       -1.2%        -2.4%
abl_-bet           725248   0.5624   -0.0014  0.1920  0.774   1.00  +0.02       -4.2%       -10.6%
abl_-market        725248   0.5625   -0.0013  0.1920  0.774   1.00  +0.02       -4.8%       -11.3%
abl_-style         725248   0.5624   -0.0014  0.1920  0.774   1.00  +0.02       -4.2%       -10.1%
abl_-skill         725248   0.5624   -0.0014  0.1920  0.774   1.00  +0.02       -4.8%       -11.1%
abl_-cat_skill     725248   0.5622   -0.0016  0.1919  0.774   1.00  +0.02       -3.8%       -10.6%
old_r2             725248   0.6035   +0.0396  0.2088  0.732   0.83  -0.09      -31.0%       -26.0%

## Selection quality without the budget (pool signals, out of sample)

Each model's top 10% and top 2% of the pool's signals by expected return, every one taken: what the model predicted (mean P/price − 1 − fee) against what happened (capped at +300%), 90% CI over days.

model            top 10%: n predicted  realized            90% CI | top 2%: n predicted  realized            90% CI
market                 4486     +0.9%     -6.2% [  -8.8,   -3.7] |       936     +3.9%    -15.1% [ -23.6,   -6.7]
skill_eb               4465     +0.1%     -8.7% [ -11.3,   -6.5] |       893     +0.2%     -8.9% [ -19.8,   -1.5]
logit_anchor           4465     +2.7%    -10.6% [ -13.7,   -7.7] |       893     +8.1%    -25.6% [ -35.3,  -16.7]
gbm_raw                4465     +6.2%    -10.9% [ -14.3,   -7.8] |       893    +19.8%    -30.4% [ -37.8,  -22.3]
gbm_anchor             4465     +5.5%    -11.7% [ -15.3,   -8.2] |       893    +18.2%    -28.3% [ -37.3,  -18.8]
gbm_decay30            4465     +6.0%     -9.3% [ -12.8,   -6.1] |       893    +17.6%    -20.1% [ -28.6,  -10.4]
gbm_percat             4465    +11.6%     -8.3% [ -12.0,   -5.0] |       893    +27.5%    -19.7% [ -30.9,   -7.9]
gbm_wallet_te          4465     +5.2%     -9.1% [ -12.7,   -5.7] |       893    +16.8%    -25.4% [ -34.9,  -15.4]
rule_skill             4467     +8.5%     -4.1% [  -7.2,   -1.1] |       899     +9.9%     +2.5% [  -5.0,  +10.5]
rank                   4499     -0.9%     -2.4% [  -4.1,   -0.8] |       899     -0.0%     -1.2% [  -2.8,   +0.4]
abl_-bet               4465     +5.3%    -10.6% [ -14.2,   -7.3] |       893    +17.9%    -31.5% [ -39.6,  -23.4]
abl_-market            4465     +4.3%    -11.3% [ -14.8,   -7.9] |       893    +14.4%    -26.6% [ -34.0,  -19.3]
abl_-style             4465     +5.6%    -10.1% [ -13.8,   -6.5] |       893    +18.5%    -29.9% [ -37.0,  -21.2]
abl_-skill             4465     +6.3%    -11.1% [ -14.9,   -7.6] |       893    +20.8%    -33.0% [ -40.9,  -25.4]
abl_-cat_skill         4465     +5.5%    -10.6% [ -14.2,   -7.4] |       893    +17.9%    -30.5% [ -39.4,  -20.9]
old_r2                 4465   +105.5%    -26.0% [ -29.8,  -21.8] |       893   +306.7%    -59.5% [ -67.4,  -51.5]

432 policies per model; each fold's policy is the one with the most P&L on the folds before it (at least 20 trades); the winning family is picked on folds before Oct 20 and judged on the folds after (holdout)

## Every qualifying pool signal, out of sample (nested policy choice); $/week at $5 a trade within the budget

'all' is every fold from Sep 15; 'select' the folds the winner is chosen on; 'holdout' Oct 20 → (never chosen on). 'book' is the same policies run through the bot's budget (10 a day, 20 open, 3 a cycle, 25% per leader), as a check. 'capped' counts any single win at most +300%.

model            signals mean/trade            90% CI  capped   win   $/week | select: n    mean    $/wk | holdout: n    mean            90% CI    $/wk | book: n, mean, $/wk
market              448      -1.4% [  -8.3,   +6.1]   -3.2%   55%    -4.72 |       339   -0.9%   -3.16 |        109   -2.8% [ -17.5,  +15.7]   -9.70 | book  307   -0.4%   -0.88 
skill_eb            369      +4.7% [  -8.6,  +22.3]   -0.1%   55%   +13.34 |       178   -2.6%   -4.55 |        191  +11.6% [ -10.7,  +45.4]  +40.45 | book  156  -18.4%  -21.78 
logit_anchor        940      +1.1% [  -4.1,   +6.5]   +1.1%   55%    +3.94 |       696   +1.4%   +4.79 |        244   +0.4% [ -10.4,  +11.7]   +1.54 | book  301   +0.5%   +1.06 
gbm_raw             812      -4.1% [ -11.0,   +2.5]   -5.1%   54%   -14.26 |       520   -5.8%  -20.41 |        292   -0.9% [ -15.9,  +11.0]   -3.29 | book  347   +0.1%   +0.25 
gbm_anchor          978      -0.9% [  -5.7,   +3.6]   -0.9%   55%    -3.04 |       606   +0.2%   +0.67 |        372   -2.6% [ -12.4,   +6.5]   -9.09 | book  354   +1.3%   +3.54 
gbm_decay30        1090      -0.1% [  -5.5,   +5.1]   -0.1%   55%    -0.42 |       725   -0.0%   -0.06 |        365   -0.3% [  -9.4,   +9.0]   -1.12 | book  356   -2.5%   -6.72 
gbm_percat          335      +1.3% [  -7.3,  +10.3]   +0.2%   52%    +3.33 |       176   +4.7%   +8.61 |        159   -2.6% [ -16.7,  +11.6]   -8.98 | book  197   +1.1%   +1.68 
gbm_wallet_te       852      -0.5% [  -6.5,   +5.9]   -0.5%   56%    -1.59 |       508   +3.3%  +11.54 |        344   -6.0% [ -16.5,   +6.0]  -20.97 | book  341   -4.1%  -10.47 
rule_skill          460      -0.6% [  -5.7,   +4.2]   -0.7%   64%    -2.04 |       285   +2.5%   +7.18 |        175   -5.7% [ -14.2,   +3.4]  -19.84 | book  244   +0.7%   +1.33 
rank                 46      -6.0% [ -16.9,   +0.9]   -6.0%   89%    -2.20 |         4   +5.8%   +0.25 |         42   -7.1% [ -20.1,   +0.4]   -8.75 | book   42   -7.1%   -2.37 
old_r2              468      +3.9% [  -4.9,  +12.7]   +3.9%   50%   +13.55 |       317   -1.5%   -4.90 |        151  +15.3% [  +4.5,  +24.8]  +53.53 | book  270   +4.9%   +9.94 
abl_-bet            999      +4.4% [  -1.3,  +10.0]   +4.4%   57%   +15.23 |       621   +5.7%  +19.99 |        378   +2.1% [  -7.5,  +12.6]   +7.41 (gbm_anchor's policies)
abl_-market         749      -4.9% [ -10.1,   -0.1]   -4.9%   52%   -17.05 |       432   -3.9%  -13.57 |        317   -6.2% [ -15.9,   +1.6]  -21.78 (gbm_anchor's policies)
abl_-style          991      +1.1% [  -4.1,   +6.9]   +1.1%   56%    +3.91 |       579   +2.8%   +9.89 |        412   -1.3% [ -11.1,   +9.0]   -4.50 (gbm_anchor's policies)
abl_-skill         1127      +1.3% [  -3.8,   +6.6]   +1.3%   57%    +4.40 |       717   +1.9%   +6.58 |        410   +0.2% [  -7.6,   +9.9]   +0.58 (gbm_anchor's policies)
abl_-cat_skill     1027      -1.3% [  -6.3,   +3.8]   -1.3%   55%    -4.54 |       613   -0.4%   -1.41 |        414   -2.6% [ -10.0,   +6.2]   -9.16 (gbm_anchor's policies)

## Rival rules on the same book

day_sports_40_60   1730      -4.5% [  -8.2,   -0.7]   -4.5%   49%   -15.91 |      1061   -5.9%  -20.75 |        669   -2.3% [  -6.1,   +1.7]   -8.22 | book  409   -3.0%   -9.16 
copy_pool_all     22914      -5.8% [  -6.8,   -4.8]   -8.2%   52%   -20.33 |     14039   -5.9%  -20.79 |       8875   -5.6% [  -7.5,   -3.9]  -19.61 | book  464  -15.4%  -53.09 
learned_now         162     +10.2% [  -7.0,  +28.1]   +7.5%   48%   +12.27 |       116   +9.6%  +11.15 |         46  +11.6% [ -20.9,  +50.3]  +15.52 | book  130  +16.9%  +16.40 

The selection folds' best family is gbm_wallet_te (508 signals, +3.3% per trade); on the holdout it made -6.0% [-16.5, +6.0]. It needs a per-wallet table the bot cannot rebuild, so the same rule over the shippable families picks logit_anchor; everything below describes logit_anchor.

## Winner: logit_anchor

chosen on the folds before Oct 20 (696 trades, +1.4% per trade, $+4.79/week); holdout after: 244 trades, +0.4% per trade [90% CI -10.4, +11.7], $+1.54/week; through the bot's book: 301 trades, +0.5% per trade, $+1.06/week

Winner minus each rival, mean return per trade, all out-of-sample folds (bootstrap over days):
  vs market          :  +2.6 points per trade (median of bootstrap), 5th pct   -6.0, P(winner better) 69%; $/week +3.94 vs -4.72
  vs skill_eb        :  -2.9 points per trade (median of bootstrap), 5th pct  -20.3, P(winner better) 38%; $/week +3.94 vs +13.34
  vs gbm_raw         :  +5.3 points per trade (median of bootstrap), 5th pct   -2.0, P(winner better) 89%; $/week +3.94 vs -14.26
  vs gbm_anchor      :  +2.0 points per trade (median of bootstrap), 5th pct   -2.1, P(winner better) 77%; $/week +3.94 vs -3.04
  vs gbm_decay30     :  +1.3 points per trade (median of bootstrap), 5th pct   -4.4, P(winner better) 64%; $/week +3.94 vs -0.42
  vs gbm_percat      :  -0.2 points per trade (median of bootstrap), 5th pct   -9.5, P(winner better) 48%; $/week +3.94 vs +3.33
  vs gbm_wallet_te   :  +1.6 points per trade (median of bootstrap), 5th pct   -3.4, P(winner better) 69%; $/week +3.94 vs -1.59
  vs rule_skill      :  +1.7 points per trade (median of bootstrap), 5th pct   -4.5, P(winner better) 68%; $/week +3.94 vs -2.04
  vs rank            :  +7.6 points per trade (median of bootstrap), 5th pct   -2.3, P(winner better) 90%; $/week +3.94 vs -2.20
  vs old_r2          :  -2.8 points per trade (median of bootstrap), 5th pct  -12.3, P(winner better) 33%; $/week +3.94 vs +13.55
  vs abl_-bet        :  -3.5 points per trade (median of bootstrap), 5th pct   -8.0, P(winner better) 14%; $/week +3.94 vs +15.23
  vs abl_-market     :  +6.2 points per trade (median of bootstrap), 5th pct   -0.8, P(winner better) 92%; $/week +3.94 vs -17.05
  vs abl_-style      :  -0.1 points per trade (median of bootstrap), 5th pct   -4.1, P(winner better) 49%; $/week +3.94 vs +3.91
  vs abl_-skill      :  -0.2 points per trade (median of bootstrap), 5th pct   -5.5, P(winner better) 47%; $/week +3.94 vs +4.40
  vs abl_-cat_skill  :  +2.3 points per trade (median of bootstrap), 5th pct   -2.1, P(winner better) 80%; $/week +3.94 vs -4.54
  vs day_sports_40_60:  +5.8 points per trade (median of bootstrap), 5th pct   -0.1, P(winner better) 94%; $/week +3.94 vs -15.91
  vs copy_pool_all   :  +6.9 points per trade (median of bootstrap), 5th pct   +1.3, P(winner better) 98%; $/week +3.94 vs -20.33
  vs learned_now     :  -9.2 points per trade (median of bootstrap), 5th pct  -28.7, P(winner better) 21%; $/week +3.94 vs +12.27
  by type: crypto 9 (+12%) · economics 2 (+53%) · geopolitics 4 (-10%) · other 196 (-5%) · politics 14 (+8%) · sports 708 (+2%) · tech 7 (+21%)
  by price paid: 30–50¢ 303 (-2%) · 50–70¢ 627 (+2%) · ≥70¢ 10 (+13%)
  by time to the end: <6h 207 (+6%) · 6–48h 396 (-4%) · 2–14d 319 (+2%) · >14d 18 (+41%)
  in play: in play 86 (+3%) · not in play 854 (+1%)
  leaders: 187 distinct; the busiest is behind 5% of trades, the top five 19%
  live paper period (Oct 28 →): 38 trades, mean +1.0% per trade, win 50%
    day_sports_40_60: 208 trades, mean +1.3% per trade
    learned_now: 10 trades, mean +118.2% per trade

## Policy rules, one at a time (the winner's model and fold policies otherwise)

Every qualifying signal (left) and the bot's book (right); 'two-sided' counts events held on two outcomes at once.

  as chosen                    signals   940, mean   +1.1% [-3.6, +6.3], two-sided   0 | book  301, mean   +0.5%, $+1.06/week, two-sided 0
  no per-event limit           signals  1689, mean   -0.9% [-4.4, +2.4], two-sided 279 | book  241, mean   -5.4%, $-9.95/week, two-sided 27
  no leader cap                signals   940, mean   +1.1% [-4.0, +6.3], two-sided   0 | book  279, mean   +4.1%, $+8.61/week, two-sided 0
  exit when the leader sells   signals   962, mean   -0.1% [-5.3, +4.9], two-sided   0 | book  317, mean   -1.8%, $-4.24/week, two-sided 0
  in-play signals allowed      signals   970, mean   +0.5% [-4.5, +5.9], two-sided   0 | book  304, mean   -5.1%, $-11.75/week, two-sided 0
  5 a day instead of 10        signals   940, mean   +1.1% [-3.9, +6.6], two-sided   0 | book  182, mean   -7.8%, $-10.78/week, two-sided 0

## Robustness of the winner

wallets not picked by a profit board (random takers, volume boards): 2281 wallets, 253877 signals
  unbiased wallets, any         2087 signals, mean -1.6% [-4.6, +1.6]
  profit-board wallets, any     4233 signals, mean +2.6% [-0.2, +5.3]
  same trades, entered at the leader's own price: +0.1% (n 940); 5 minutes after (the simulation): +1.1% (n 940); 30 min: -1.8% (n 940); 60 min: -2.0% (n 923); 4 h: -5.7% (n 819)
  same trades at the 75th-percentile book cost instead of the median: -0.0% (n 940); exiting when the leader sells: +0.1% (n 940)
  475 of 940 trades priced off the real tape (thin markets); mean cost above mid on the others 0.8¢
  stake each trade can take while keeping half its expected edge: median $50, share at $50: 94%; at 24.1 trades a day that is about $1148 a day

exported logit_anchor (linear, 250000 rows, config 10000.0) → /data/model_r3.json; bot scorer vs LightGBM on 2000 rows: max |Δp| = 1.38e-08
