import json
from cnctwin.wear import calibration as CAL, build as B
_c = CAL.calibrate
def cal(entry, *a, **k):
    r = _c(entry, *a, **k)
    print("CAL", json.dumps({"A": r.A_per_Pa, "B": r.B_K, "entry": str(entry)[:600], "report": r.report}, default=str), flush=True)
    return r
CAL.calibrate = cal
B.CAL.calibrate = cal
