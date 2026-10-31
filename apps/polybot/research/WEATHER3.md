# Weather study (analyze3_weather.py), three timings

## 12:00 UTC the day before, day-before Open-Meteo runs (contains look-ahead for the Americas)
```
150861 temperature markets, 147757 resolved in 13388 events, 53 cities with a station
cost of a $5 buy above the mid on temperature books (median, from 32629 snapshots): <15¢ 2.1¢, 15–85¢ 4.8¢, >85¢ 0.7¢
13062 events with observations, forecasts and one winning bucket; METAR max/min falls in the winning bucket in 95.9% of them (541 disagree)
22645 priced markets with a walk-forward forecast distribution (3717 events, 2026-08-15 → 2026-10-31)

## Calibration (every priced bucket)
  market mid: log loss 0.3318, Brier 0.1056; predicted→actual 0–2%: 0.9→0.8 (n 5061) · 2–5%: 2.9→2.9 (n 4027) · 5–10%: 7.0→6.5 (n 2802) · 10–20%: 14.8→13.4 (n 2918) · 20–35%: 27.3→27.9 (n 3671) · 35–50%: 41.3→40.7 (n 2822) · 50–65%: 55.3→56.3 (n 1184) · 65–80%: 68.4→64.0 (n 150)
  forecast: log loss 0.3706, Brier 0.1161; predicted→actual 0–2%: 0.8→2.0 (n 5839) · 2–5%: 3.3→5.4 (n 3431) · 5–10%: 7.3→9.6 (n 2779) · 10–20%: 14.5→16.9 (n 3099) · 20–35%: 27.4→27.5 (n 3386) · 35–50%: 42.1→38.0 (n 3054) · 50–65%: 53.3→45.8 (n 1023)

blend fitted walk-forward on 19230 buckets (from 2026-08-26); last weights: market +0.90, forecast +0.15, const +0.08
  market mid (same buckets): log loss 0.3339, Brier 0.1065; predicted→actual 0–2%: 0.9→0.8 (n 4255) · 2–5%: 2.9→2.8 (n 3441) · 5–10%: 7.1→6.1 (n 2312) · 10–20%: 14.7→13.7 (n 2467) · 20–35%: 27.3→28.0 (n 3127) · 35–50%: 41.4→40.9 (n 2446) · 50–65%: 55.3→55.9 (n 1041) · 65–80%: 68.3→63.4 (n 134)
  forecast (same buckets): log loss 0.3750, Brier 0.1177; predicted→actual 0–2%: 0.8→2.0 (n 4856) · 2–5%: 3.3→5.5 (n 2894) · 5–10%: 7.3→9.5 (n 2381) · 10–20%: 14.5→17.1 (n 2651) · 20–35%: 27.4→27.6 (n 2903) · 35–50%: 42.2→38.2 (n 2615) · 50–65%: 53.3→45.2 (n 908)
  blend: log loss 0.3330, Brier 0.1064; predicted→actual 0–2%: 1.0→1.0 (n 5268) · 2–5%: 3.2→3.1 (n 2849) · 5–10%: 7.3→6.7 (n 2068) · 10–20%: 14.6→14.8 (n 2447) · 20–35%: 27.4→28.4 (n 3024) · 35–50%: 41.9→41.0 (n 2444) · 50–65%: 55.7→56.8 (n 1004) · 65–80%: 68.5→65.3 (n 118)

## Trading, $5 a trade, at most 10 a day, held to resolution, half-spread and fee paid
  bot rule: No at max forecast +2°C              544 trades on  66 days · mean   +4.0% per trade [90% CI +1.3, +6.7] · win 89% · paid 85¢ avg · $+11.6/week at $5
  forecast: best edge ≥ 3¢                       662 trades on  67 days · mean  +24.0% per trade [90% CI +7.3, +41.1] · win 43% · paid 36¢ avg · $+83.1/week at $5
  forecast: best edge ≥ 5¢                       662 trades on  67 days · mean  +24.0% per trade [90% CI +8.3, +41.5] · win 43% · paid 36¢ avg · $+83.1/week at $5
  forecast: best edge ≥ 8¢                       662 trades on  67 days · mean  +24.0% per trade [90% CI +7.6, +41.7] · win 43% · paid 36¢ avg · $+83.1/week at $5
  forecast: best edge ≥ 12¢                      662 trades on  67 days · mean  +24.0% per trade [90% CI +7.2, +41.6] · win 43% · paid 36¢ avg · $+83.1/week at $5
  forecast: No only, edge ≥ 3¢, No 50–97¢        660 trades on  66 days · mean   +5.1% per trade [90% CI +0.1, +10.0] · win 67% · paid 63¢ avg · $+17.7/week at $5
  forecast: No only, edge ≥ 5¢, No 50–97¢        660 trades on  66 days · mean   +5.1% per trade [90% CI +0.2, +10.0] · win 67% · paid 63¢ avg · $+17.7/week at $5
  forecast: No only, edge ≥ 8¢, No 50–97¢        660 trades on  66 days · mean   +5.1% per trade [90% CI -0.0, +9.8] · win 67% · paid 63¢ avg · $+17.7/week at $5
  blend: best edge ≥ 3¢                          661 trades on  67 days · mean   +8.4% per trade [90% CI +0.4, +16.9] · win 60% · paid 54¢ avg · $+29.1/week at $5
  blend: best edge ≥ 5¢                          647 trades on  66 days · mean   +9.2% per trade [90% CI +1.2, +18.0] · win 60% · paid 54¢ avg · $+31.7/week at $5
  blend: best edge ≥ 8¢                          360 trades on  65 days · mean   +8.8% per trade [90% CI -1.2, +19.1] · win 60% · paid 54¢ avg · $+17.2/week at $5
  blend: best edge ≥ 12¢                          80 trades on  43 days · mean   +7.1% per trade [90% CI -14.3, +28.7] · win 57% · paid 54¢ avg · $+4.6/week at $5
  blend: No only, edge ≥ 3¢, No 50–97¢           658 trades on  66 days · mean   +6.1% per trade [90% CI +1.9, +10.0] · win 76% · paid 71¢ avg · $+21.1/week at $5
  blend: No only, edge ≥ 5¢, No 50–97¢           577 trades on  66 days · mean   +7.3% per trade [90% CI +3.3, +11.2] · win 76% · paid 71¢ avg · $+22.5/week at $5
  blend: No only, edge ≥ 8¢, No 50–97¢           234 trades on  59 days · mean  +12.0% per trade [90% CI +5.8, +18.5] · win 76% · paid 68¢ avg · $+16.6/week at $5

best (forecast: best edge ≥ 5¢) minus the bot rule, per trade, paired by day over 66 common days: median +20.5 points, 90% CI [+3.4, +38.2]
```

## 12:00 UTC the day before, two-days-before runs (certainly known in time)
```
150861 temperature markets, 147757 resolved in 13388 events, 53 cities with a station
cost of a $5 buy above the mid on temperature books (median, from 32629 snapshots): <15¢ 2.1¢, 15–85¢ 4.8¢, >85¢ 0.7¢
13062 events with observations, forecasts and one winning bucket; METAR max/min falls in the winning bucket in 95.9% of them (541 disagree)
22645 priced markets with a walk-forward forecast distribution (3717 events, 2026-08-15 → 2026-10-31)

## Calibration (every priced bucket)
  market mid: log loss 0.3318, Brier 0.1056; predicted→actual 0–2%: 0.9→0.8 (n 5061) · 2–5%: 2.9→2.9 (n 4027) · 5–10%: 7.0→6.5 (n 2802) · 10–20%: 14.8→13.4 (n 2918) · 20–35%: 27.3→27.9 (n 3671) · 35–50%: 41.3→40.7 (n 2822) · 50–65%: 55.3→56.3 (n 1184) · 65–80%: 68.4→64.0 (n 150)
  forecast: log loss 0.3951, Brier 0.1221; predicted→actual 0–2%: 0.9→3.0 (n 4895) · 2–5%: 3.3→5.9 (n 3475) · 5–10%: 7.3→10.8 (n 3151) · 10–20%: 14.7→16.3 (n 3475) · 20–35%: 27.3→27.1 (n 4092) · 35–50%: 41.4→35.8 (n 3009) · 50–65%: 52.9→41.8 (n 522)

blend fitted walk-forward on 19230 buckets (from 2026-08-26); last weights: market +1.04, forecast -0.02, const -0.01
  market mid (same buckets): log loss 0.3339, Brier 0.1065; predicted→actual 0–2%: 0.9→0.8 (n 4255) · 2–5%: 2.9→2.8 (n 3441) · 5–10%: 7.1→6.1 (n 2312) · 10–20%: 14.7→13.7 (n 2467) · 20–35%: 27.3→28.0 (n 3127) · 35–50%: 41.4→40.9 (n 2446) · 50–65%: 55.3→55.9 (n 1041) · 65–80%: 68.3→63.4 (n 134)
  forecast (same buckets): log loss 0.4000, Brier 0.1235; predicted→actual 0–2%: 0.9→3.2 (n 4079) · 2–5%: 3.3→6.0 (n 2924) · 5–10%: 7.3→10.4 (n 2678) · 10–20%: 14.7→16.3 (n 2988) · 20–35%: 27.3→27.3 (n 3482) · 35–50%: 41.5→35.9 (n 2608) · 50–65%: 52.8→41.7 (n 453)
  blend: log loss 0.3340, Brier 0.1066; predicted→actual 0–2%: 1.0→0.9 (n 5407) · 2–5%: 3.3→3.7 (n 2762) · 5–10%: 7.3→6.9 (n 1987) · 10–20%: 14.7→14.2 (n 2436) · 20–35%: 27.4→28.4 (n 3087) · 35–50%: 41.7→41.4 (n 2385) · 50–65%: 55.8→55.6 (n 1010) · 65–80%: 69.0→63.1 (n 149)

## Trading, $5 a trade, at most 10 a day, held to resolution, half-spread and fee paid
  bot rule: No at max forecast +2°C              505 trades on  66 days · mean   +4.7% per trade [90% CI +1.8, +7.4] · win 88% · paid 84¢ avg · $+12.6/week at $5
  forecast: best edge ≥ 3¢                       662 trades on  67 days · mean   -5.3% per trade [90% CI -17.7, +6.9] · win 40% · paid 37¢ avg · $-18.4/week at $5
  forecast: best edge ≥ 5¢                       662 trades on  67 days · mean   -5.3% per trade [90% CI -17.8, +6.8] · win 40% · paid 37¢ avg · $-18.4/week at $5
  forecast: best edge ≥ 8¢                       662 trades on  67 days · mean   -5.3% per trade [90% CI -16.8, +6.7] · win 40% · paid 37¢ avg · $-18.4/week at $5
  forecast: best edge ≥ 12¢                      662 trades on  67 days · mean   -5.3% per trade [90% CI -16.9, +7.0] · win 40% · paid 37¢ avg · $-18.4/week at $5
  forecast: No only, edge ≥ 3¢, No 50–97¢        660 trades on  66 days · mean   -1.2% per trade [90% CI -6.3, +3.7] · win 62% · paid 62¢ avg · $-4.1/week at $5
  forecast: No only, edge ≥ 5¢, No 50–97¢        660 trades on  66 days · mean   -1.2% per trade [90% CI -5.8, +4.0] · win 62% · paid 62¢ avg · $-4.1/week at $5
  forecast: No only, edge ≥ 8¢, No 50–97¢        660 trades on  66 days · mean   -1.2% per trade [90% CI -5.9, +3.8] · win 62% · paid 62¢ avg · $-4.1/week at $5
  blend: best edge ≥ 3¢                          652 trades on  67 days · mean   +1.8% per trade [90% CI -8.2, +12.3] · win 54% · paid 51¢ avg · $+6.1/week at $5
  blend: best edge ≥ 5¢                          462 trades on  66 days · mean   +0.7% per trade [90% CI -11.2, +13.9] · win 53% · paid 50¢ avg · $+1.8/week at $5
  blend: best edge ≥ 8¢                          153 trades on  53 days · mean  -16.7% per trade [90% CI -31.9, +0.0] · win 46% · paid 49¢ avg · $-16.9/week at $5
  blend: best edge ≥ 12¢                          33 trades on  22 days · mean  -12.0% per trade [90% CI -47.6, +30.0] · win 45% · paid 50¢ avg · $-6.3/week at $5
  blend: No only, edge ≥ 3¢, No 50–97¢           537 trades on  67 days · mean   +4.1% per trade [90% CI -0.6, +8.8] · win 77% · paid 73¢ avg · $+11.5/week at $5
  blend: No only, edge ≥ 5¢, No 50–97¢           278 trades on  63 days · mean   +4.7% per trade [90% CI -1.5, +11.0] · win 75% · paid 71¢ avg · $+7.2/week at $5
  blend: No only, edge ≥ 8¢, No 50–97¢            80 trades on  40 days · mean   +3.3% per trade [90% CI -8.4, +16.0] · win 70% · paid 68¢ avg · $+2.3/week at $5

best (blend: No only, edge ≥ 3¢, No 50–97¢) minus the bot rule, per trade, paired by day over 66 common days: median -1.6 points, 90% CI [-7.1, +4.3]
```

## Just before the city's day starts (the bot's timing), day-before runs
```
150861 temperature markets, 147757 resolved in 13388 events, 53 cities with a station
cost of a $5 buy above the mid on temperature books (median, from 32629 snapshots): <15¢ 2.1¢, 15–85¢ 4.8¢, >85¢ 0.7¢
13062 events with observations, forecasts and one winning bucket; METAR max/min falls in the winning bucket in 95.9% of them (541 disagree)
21303 priced markets with a walk-forward forecast distribution (3736 events, 2026-08-15 → 2026-10-31)

## Calibration (every priced bucket)
  market mid: log loss 0.3440, Brier 0.1097; predicted→actual 0–2%: 1.0→1.0 (n 4236) · 2–5%: 2.9→2.2 (n 3538) · 5–10%: 7.0→6.5 (n 2820) · 10–20%: 14.7→13.0 (n 2933) · 20–35%: 27.3→28.0 (n 3487) · 35–50%: 41.5→40.0 (n 2717) · 50–65%: 55.7→58.0 (n 1332) · 65–80%: 68.6→71.3 (n 230)
  forecast: log loss 0.3963, Brier 0.1242; predicted→actual 0–2%: 0.9→2.7 (n 4606) · 2–5%: 3.3→5.6 (n 3295) · 5–10%: 7.3→9.6 (n 2755) · 10–20%: 14.5→17.0 (n 3106) · 20–35%: 27.4→27.4 (n 3411) · 35–50%: 42.1→38.1 (n 3070) · 50–65%: 53.3→45.7 (n 1026)

blend fitted walk-forward on 17762 buckets (from 2026-08-27); last weights: market +1.01, forecast +0.05, const +0.05
  market mid (same buckets): log loss 0.3454, Brier 0.1104; predicted→actual 0–2%: 1.0→1.0 (n 3535) · 2–5%: 2.9→2.2 (n 2933) · 5–10%: 7.0→6.5 (n 2325) · 10–20%: 14.8→12.7 (n 2409) · 20–35%: 27.3→28.2 (n 2904) · 35–50%: 41.5→40.4 (n 2295) · 50–65%: 55.6→57.8 (n 1148) · 65–80%: 68.5→70.4 (n 206)
  forecast (same buckets): log loss 0.4008, Brier 0.1258; predicted→actual 0–2%: 0.9→2.7 (n 3772) · 2–5%: 3.3→5.7 (n 2718) · 5–10%: 7.3→9.6 (n 2304) · 10–20%: 14.5→17.1 (n 2609) · 20–35%: 27.4→27.5 (n 2871) · 35–50%: 42.2→38.3 (n 2567) · 50–65%: 53.3→44.9 (n 900)
  blend: log loss 0.3453, Brier 0.1104; predicted→actual 0–2%: 1.0→1.1 (n 4558) · 2–5%: 3.3→3.2 (n 2463) · 5–10%: 7.2→6.8 (n 1969) · 10–20%: 14.7→14.1 (n 2366) · 20–35%: 27.4→28.7 (n 2858) · 35–50%: 42.0→40.7 (n 2208) · 50–65%: 56.1→57.5 (n 1116) · 65–80%: 69.0→71.6 (n 218)

## Trading, $5 a trade, at most 10 a day, held to resolution, half-spread and fee paid
  bot rule: No at max forecast +2°C              516 trades on  65 days · mean   +1.3% per trade [90% CI -1.7, +4.4] · win 87% · paid 85¢ avg · $+3.7/week at $5
  forecast: best edge ≥ 3¢                       652 trades on  66 days · mean   +2.7% per trade [90% CI -11.8, +17.9] · win 34% · paid 34¢ avg · $+9.2/week at $5
  forecast: best edge ≥ 5¢                       652 trades on  66 days · mean   +2.7% per trade [90% CI -12.0, +18.1] · win 34% · paid 34¢ avg · $+9.2/week at $5
  forecast: best edge ≥ 8¢                       652 trades on  66 days · mean   +2.7% per trade [90% CI -11.4, +17.6] · win 34% · paid 34¢ avg · $+9.2/week at $5
  forecast: best edge ≥ 12¢                      652 trades on  66 days · mean   +2.7% per trade [90% CI -11.9, +18.5] · win 34% · paid 34¢ avg · $+9.2/week at $5
  forecast: No only, edge ≥ 3¢, No 50–97¢        650 trades on  65 days · mean   -4.2% per trade [90% CI -9.7, +1.0] · win 61% · paid 63¢ avg · $-14.6/week at $5
  forecast: No only, edge ≥ 5¢, No 50–97¢        650 trades on  65 days · mean   -4.2% per trade [90% CI -9.5, +1.1] · win 61% · paid 63¢ avg · $-14.6/week at $5
  forecast: No only, edge ≥ 8¢, No 50–97¢        650 trades on  65 days · mean   -4.2% per trade [90% CI -9.6, +1.1] · win 61% · paid 63¢ avg · $-14.6/week at $5
  blend: best edge ≥ 3¢                          650 trades on  65 days · mean   +1.0% per trade [90% CI -7.7, +10.6] · win 50% · paid 49¢ avg · $+3.5/week at $5
  blend: best edge ≥ 5¢                          547 trades on  65 days · mean   -0.1% per trade [90% CI -10.4, +10.3] · win 50% · paid 49¢ avg · $-0.4/week at $5
  blend: best edge ≥ 8¢                          246 trades on  60 days · mean  -10.3% per trade [90% CI -22.5, +2.4] · win 47% · paid 48¢ avg · $-14.7/week at $5
  blend: best edge ≥ 12¢                          72 trades on  39 days · mean   -3.2% per trade [90% CI -24.5, +20.9] · win 46% · paid 43¢ avg · $-2.1/week at $5
  blend: No only, edge ≥ 3¢, No 50–97¢           612 trades on  65 days · mean   -5.0% per trade [90% CI -9.2, -0.8] · win 70% · paid 73¢ avg · $-16.5/week at $5
  blend: No only, edge ≥ 5¢, No 50–97¢           362 trades on  65 days · mean   -4.2% per trade [90% CI -10.6, +2.3] · win 68% · paid 70¢ avg · $-8.2/week at $5
  blend: No only, edge ≥ 8¢, No 50–97¢           125 trades on  50 days · mean   -2.0% per trade [90% CI -10.2, +6.4] · win 68% · paid 69¢ avg · $-1.8/week at $5

best (blend: best edge ≥ 3¢) minus the bot rule, per trade, paired by day over 65 common days: median -0.1 points, 90% CI [-10.3, +10.6]
```
