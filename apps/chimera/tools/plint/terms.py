"""The training terms of one or two `chimera compare tsp` runs, side by side.

usage: terms.py <score.json> [<earlier score.json>] [--sex male|female] [--groups voice,formants,…] [--all]

Reads only the training half (the held-out half never guides a change: docs/hill-climb/README.md).
Without --all, terms whose cost is 0 in every run given are left out.
"""
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {a.split("=")[0]: (a.split("=") + [""])[1] for a in sys.argv[1:] if a.startswith("--")}
for i, a in enumerate(sys.argv[1:]):
    if a in ("--sex", "--groups") and i + 2 < len(sys.argv):
        opt[a] = sys.argv[i + 2]
        args = [x for x in args if x != sys.argv[i + 2]]
runs = [json.load(open(p))["train"] for p in args]
sexes = [opt["--sex"]] if opt.get("--sex") else ["male", "female"]
groups = opt["--groups"].split(",") if opt.get("--groups") else None
num = lambda x: "    —   " if x is None else f"{x:8.2f}"
for sex in sexes:
    print(f"\n{sex}: score " + " ← ".join(f"{r[sex]['score']:.3f}" for r in runs))
    print("  groups: " + ", ".join(f"{g} " + " ← ".join(f"{r[sex]['groups'].get(g) or 0:.2f}" for r in runs) for g in runs[0][sex]["groups"]))
    terms = runs[0][sex]["terms"]
    rows = []
    for name, t in terms.items():
        if groups and t["group"] not in groups:
            continue
        old = runs[1][sex]["terms"].get(name, {}) if len(runs) > 1 else {}
        if "--all" not in opt and not (t.get("cost") or old.get("cost")):
            continue
        rows.append((-(t.get("cost") or 0), name, t, old))
    for _, name, t, old in sorted(rows, key=lambda r: (r[0], r[1])):
        line = f"  {name:34} people {num(t.get('centre'))} ±{num(t.get('spread'))}  synth {num(t.get('synth'))}  cost {num(t.get('cost'))}"
        if old:
            line += f"   was {num(old.get('synth'))} cost {num(old.get('cost'))}"
        print(line + ("" if t.get("scored") else "   (diagnostic)"))
