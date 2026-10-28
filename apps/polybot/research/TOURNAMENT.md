415913 markets in the positions, 96173 of them with tags/resolutions on file
1120886 resolved positions from 2792 traders (300559 resolved-unclaimed; returns are at resolution), loaded in 38s
features built in 129s
1050373 positions with 5+ prior positions for the models
folds (train before, test the next 14 days): Jul 22, Aug 05, Aug 19, Sep 02, Sep 16, Sep 30, Oct 14
6 models × 108 policies × 7 folds, bootstrap 300 per fold
  fold Jul 22: train 423855, test 29245 (252s)
  fold Aug 05: train 452676, test 33127 (509s)
  fold Aug 19: train 485134, test 43956 (765s)
  fold Sep 02: train 528898, test 50104 (1025s)
  fold Sep 16: train 578074, test 64282 (1282s)
  fold Sep 30: train 641974, test 101962 (1536s)
  fold Oct 14: train 744363, test 291360 (1796s)

648 (model, policy) pairs had enough trades; ranked by the average bootstrap 5th percentile of the mean return per trade

model               q  k sports        band days | trades /fold    mean 5th pct   worst  win
gbm_win          0.90  5    yes 0.15–0.75  0.5 |    525    75  +65.3%  +34.3%  +28.5%  66%
gbm_win          0.50  5    yes 0.15–0.75  0.5 |    525    75  +65.3%  +33.7%  +28.5%  66%
gbm_win          0.75  5    yes 0.15–0.75  0.5 |    525    75  +65.3%  +33.5%  +28.5%  66%
gbm_win          0.90  5     no 0.15–0.75  0.5 |    520    74  +64.7%  +33.3%  +28.5%  66%
gbm_win          0.50  5     no 0.15–0.75  0.5 |    522    75  +64.1%  +33.2%  +28.5%  66%
gbm_win          0.75  5     no 0.15–0.75  0.5 |    521    74  +64.4%  +32.8%  +28.5%  66%
gbm_win          0.90  2     no 0.25–0.70  0.5 |    208    30  +70.0%  +31.0%  +23.6%  69%
gbm              0.90  5    yes 0.15–0.75  0.5 |    525    75  +59.3%  +31.0%  +14.6%  61%
gbm              0.75  5    yes 0.15–0.75  0.5 |    525    75  +59.3%  +30.8%  +14.6%  61%
gbm_win          0.90  2    yes 0.25–0.70  0.5 |    210    30  +69.7%  +30.7%  +23.6%  69%
gbm_win          0.75  2     no 0.25–0.70  0.5 |    209    30  +69.2%  +30.2%  +23.6%  68%
logit            0.75  5     no 0.15–0.75  0.5 |    521    74  +55.2%  +30.2%  +25.8%  68%
logit            0.90  3     no 0.15–0.75  0.5 |    313    45  +60.2%  +30.0%  +31.8%  69%
logit            0.75  3     no 0.15–0.75  0.5 |    313    45  +60.2%  +29.8%  +31.8%  69%
gbm_win          0.50  3    yes 0.15–0.75  0.5 |    315    45  +69.2%  +29.6%  +32.3%  63%
logit            0.50  5     no 0.15–0.75  0.5 |    522    75  +54.9%  +29.6%  +25.8%  68%
gbm_win          0.75  5    yes 0.05–0.95  0.5 |    525    75  +62.7%  +29.5%  +22.5%  61%
gbm_win          0.50  2     no 0.25–0.70  0.5 |    210    30  +68.4%  +29.3%  +23.6%  68%
gbm_win          0.75  3    yes 0.15–0.75  0.5 |    315    45  +69.2%  +29.3%  +32.3%  63%
gbm              0.50  5    yes 0.15–0.75  0.5 |    525    75  +59.3%  +29.3%  +14.6%  61%
logit            0.50  3     no 0.15–0.75  0.5 |    314    45  +59.7%  +29.2%  +31.8%  68%
logit            0.90  5     no 0.15–0.75  0.5 |    521    74  +55.2%  +29.1%  +25.8%  68%
gbm_win          0.90  3    yes 0.15–0.75  0.5 |    315    45  +69.2%  +28.9%  +32.3%  63%
gbm_win          0.90  3    yes 0.25–0.70  0.5 |    315    45  +60.8%  +28.9%  +31.5%  69%
gbm_win          0.90  5    yes 0.05–0.95  0.5 |    525    75  +62.7%  +28.9%  +22.5%  61%
gbm              0.75  2    yes 0.15–0.75  0.5 |    210    30  +76.7%  +28.7%  +54.7%  60%
gbm_win          0.75  2    yes 0.25–0.70  0.5 |    210    30  +69.7%  +28.6%  +23.6%  69%
gbm_win          0.75  3     no 0.25–0.70  0.5 |    313    45  +60.5%  +28.6%  +31.5%  69%
gbm_win          0.75  3     no 0.15–0.75  0.5 |    313    45  +69.1%  +28.6%  +35.5%  63%
gbm_win          0.90  3     no 0.25–0.70  0.5 |    312    45  +61.0%  +28.4%  +31.5%  69%

Best per model:
  gbm_win          mean  +65.3%  5th pct  +34.3%  win  66%  trades/fold 75  (q 0.9, k 5, sports yes, band 0.15–0.75, days ≥ 0.5)
  gbm              mean  +59.3%  5th pct  +31.0%  win  61%  trades/fold 75  (q 0.9, k 5, sports yes, band 0.15–0.75, days ≥ 0.5)
  logit            mean  +55.2%  5th pct  +30.2%  win  68%  trades/fold 74  (q 0.75, k 5, sports no, band 0.15–0.75, days ≥ 0.5)
  ridge            mean  +50.3%  5th pct  +25.6%  win  72%  trades/fold 45  (q 0.9, k 3, sports yes, band 0.05–0.95, days ≥ 0.5)
  rule:credible    mean  +50.5%  5th pct  +24.6%  win  67%  trades/fold 74  (q 0.9, k 5, sports yes, band 0.15–0.75, days ≥ 0.5)
  rule:leaderboard mean   -7.5%  5th pct  -27.1%  win  47%  trades/fold 75  (q 0.75, k 5, sports yes, band 0.25–0.7, days ≥ 0.5)

For scale, 'copy the leaderboard, anything, 3 a day': mean -12.2% per trade, 5th pct -40.0%, win 46%

What the winner trades (last three folds):
  fold Sep 16: 75 trades, mean +91%, win 77% · types {'geopolitics': 6, 'culture': 5, 'crypto': 4, 'weather': 40, 'politics': 5, 'economics': 5, 'other': 9, 'sports': 1} · price <30¢ 32%, 30–70¢ 55%, >70¢ 13%
      +300%  weather     0.22  Will the highest temperature in Paris be 24°C on · Yes  (trader prior wr 100%, n 105)
      +300%  weather     0.19  Will the highest temperature in Manila be 32°C o · Yes  (trader prior wr 97%, n 198)
      +300%  weather     0.22  Will the highest temperature in Shanghai be 24°C · Yes  (trader prior wr 97%, n 190)
      +300%  weather     0.23  Will the highest temperature in Singapore be 33° · Yes  (trader prior wr 97%, n 190)
  fold Sep 30: 75 trades, mean +84%, win 68% · types {'weather': 48, 'culture': 7, 'politics': 8, 'other': 4, 'geopolitics': 3, 'tech': 3, 'sports': 2} · price <30¢ 31%, 30–70¢ 61%, >70¢ 8%
      +300%  culture     0.17  Will "Other Mommy" Opening Weekend Box Office be · Yes  (trader prior wr 100%, n 191)
      +300%  weather     0.18  Will the highest temperature in Kuala Lumpur be  · Yes  (trader prior wr 100%, n 6)
      +300%  weather     0.22  Will the highest temperature in Miami be between · Yes  (trader prior wr 100%, n 6)
      +300%  geopolitics 0.15  Israel military action against Lebanon on Octobe · Yes  (trader prior wr 88%, n 78)
  fold Oct 14: 75 trades, mean +71%, win 63% · types {'geopolitics': 1, 'weather': 59, 'sports': 10, 'tech': 1, 'other': 2, 'crypto': 1, 'culture': 1} · price <30¢ 47%, 30–70¢ 48%, >70¢ 5%
      +300%  weather     0.18  Will the highest temperature in Chengdu be 27°C  · Yes  (trader prior wr 100%, n 17)
      +300%  weather     0.21  Will the highest temperature in Tel Aviv be 26°C · Yes  (trader prior wr 100%, n 12)
      +300%  weather     0.22  Will the highest temperature in Amsterdam be 11° · Yes  (trader prior wr 100%, n 12)
      +300%  weather     0.23  Will the highest temperature in Miami be between · Yes  (trader prior wr 100%, n 12)

winner: gbm_win with q=0.9, k=5, sports=yes, band (0.15, 0.75), days ≥ 0.5 → /out/model_big.json (threshold +0.0496, 966211 rows)

total 2069s