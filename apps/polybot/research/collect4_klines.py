import json, sys, time, urllib.request
import numpy as np
out = sys.argv[1]
t0 = int(time.mktime(time.strptime("2026-07-25", "%Y-%m-%d"))) * 1000
for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
    rows, t = [], t0
    while True:
        for i in range(5):
            try:
                with urllib.request.urlopen(f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=1m&limit=1000&startTime={t}", timeout=30) as r:
                    j = json.load(r)
                break
            except Exception as e:
                time.sleep(2 * (i + 1))
        if not j:
            break
        rows += [(k[0] / 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in j]
        t = j[-1][0] + 60000
        if len(j) < 1000:
            break
        time.sleep(0.05)
    a = np.array(rows)
    np.save(f"{out}/{sym}.npy", a)
    print(sym, len(a), time.strftime("%Y-%m-%d %H:%M", time.gmtime(a[0, 0])), time.strftime("%Y-%m-%d %H:%M", time.gmtime(a[-1, 0])), flush=True)
