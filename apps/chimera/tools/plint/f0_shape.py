"""How pitch moves over a read sentence: real speakers (TSP) or a folder of renders.

usage: f0_shape.py <tsp dir> --prefix M            the real men (speakers MA, MB, …)
       f0_shape.py <run>/tsp/mfa/corpus --renders male   a judge run's renders of the training lists (1-60)

Per sentence, in semitones re that sentence's median f0: the 10th and 90th percentiles, the slope
of a line through the voiced frames (per sentence, start to end), the median of the first and last
fifth of the voiced frames, and the lowest tenth. Prints the median over sentences and the spread
across speakers. Training speakers only (A-J): K and L are the judge's held-out speakers.
"""
import glob
import os
import sys

import numpy as np
import parselmouth

root = sys.argv[1]
prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else ""
renders = sys.argv[sys.argv.index("--renders") + 1] if "--renders" in sys.argv else ""
rows = {}
for path in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)):
    name = os.path.basename(path)
    if renders:
        # <body>-<list>_<sentence>/…: the training lists only
        d = os.path.basename(os.path.dirname(path))
        if not d.startswith(renders + "-") or int(d.split("-")[1].split("_")[0]) > 60:
            continue
    if prefix and not name.startswith(prefix):
        continue
    if prefix and name[1] in "KL":
        continue
    snd = parselmouth.Sound(path)
    lo, hi = (60, 300) if prefix == "M" or renders == "male" else (100, 500)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=lo, pitch_ceiling=hi)
    f = p.selected_array["frequency"]
    t = p.xs()
    v = f > 0
    if v.sum() < 30:
        continue
    st = 12 * np.log2(f[v] / np.median(f[v]))
    tt = (t[v] - t[v][0]) / (t[v][-1] - t[v][0])
    n = len(st)
    slope = np.polyfit(tt, st, 1)[0]
    rows.setdefault(name[:2] if prefix else "all", []).append(
        [np.median(f[v]), np.percentile(st, 10), np.percentile(st, 90), slope, np.median(st[: n // 5]), np.median(st[-n // 5 :]), np.percentile(st, 2), np.percentile(st, 98)]
    )
names = ["median Hz", "p10 st", "p90 st", "slope st", "first fifth", "last fifth", "p2 st", "p98 st"]
per = np.array([np.median(np.array(r), axis=0) for r in rows.values()])
print(f"{len(rows)} voices, {sum(len(r) for r in rows.values())} sentences")
for i, n in enumerate(names):
    print(f"  {n:12} {np.median(per[:, i]):7.2f}   (voices {per[:, i].min():6.2f} … {per[:, i].max():6.2f})")
print(f"  {'p90 − p10':12} {np.median(per[:, 2] - per[:, 1]):7.2f}")
