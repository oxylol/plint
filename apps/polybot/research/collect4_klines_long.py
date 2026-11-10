import json, sys, time, urllib.request, calendar
import numpy as np
out, sym = sys.argv[1], sys.argv[2]
t = calendar.timegm(time.strptime("2021-12-01", "%Y-%m-%d")) * 1000
rows = []
while True:
    j = None
    for i in range(6):
        try:
            with urllib.request.urlopen(f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=15m&limit=1000&startTime={t}", timeout=30) as r:
                j = json.load(r)
            break
        except Exception:
            time.sleep(2 * (i + 1))
    if not j:
        break
    rows += [(k[0] / 1000, float(k[4])) for k in j]
    t = j[-1][0] + 900000
    if len(j) < 1000:
        break
a = np.array(rows); np.save(f"{out}/{sym}_15m.npy", a)
print(sym, len(a))
