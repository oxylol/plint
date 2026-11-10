import json, sys, glob, os
qs = ["cutting_force","drilling_thrust","drilling_torque","chip_ratio","shear_angle","tool_temperature","chip_temperature","coolant_reach","taylor_life"]
def load(m, q):
    f = f"/opt/plint/.claude/worktrees/softening/.scratch/s/val_{m}_{q}.json"
    if not os.path.exists(f) or not os.path.getsize(f): return None
    try: return json.load(open(f))[0]
    except Exception as e: return None
for q in qs:
    for m in ("before","after"):
        e = load(m, q)
        if e is None: print(q, m, "missing"); continue
        parts = []
        for g in ("independent","calibration_dependent"):
            s = e.get(g) or {}
            if s.get("cases"):
                parts.append(f"{g[:5]}: pass {s['passing']}/{s['cases']} med {s['median_abs_rel']} p90 {s['p90_abs_rel']} worst {s['worst_case']} x{s['worst_miss_ratio']}")
        print(q, m, e["verdict"], "|", " | ".join(parts))
    if len(sys.argv) > 1:
        b, a = load("before", q), load("after", q)
        if b and a:
            for cb, ca in zip(b["cases"], a["cases"]):
                sb, sa = cb.get("stats") or {}, ca.get("stats") or {}
                print("   ", ca["id"], ca["status"], {k: (sb.get(k), sa.get(k)) for k in ("median_abs_rel", "median_rel", "max_abs_rel") if k in sa})
