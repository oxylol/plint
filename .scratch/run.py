"""Run examples/thin-wall-al7075 in-process: run.py <run_dir> <options-json> [patch-module]."""
import json, sys, time, resource, importlib
sys.path.insert(0, "/opt/plint/.claude/worktrees/thinwall/.scratch")
run_dir, opts = sys.argv[1], json.loads(sys.argv[2])
if len(sys.argv) > 3:
    for m in sys.argv[3].split(","): importlib.import_module(m)
from cnctwin.core.loop import run_project
t0w, t0c = time.time(), time.process_time()
last = [0.0]
def cb(f):
    s = f.get("sim_time_s")
    if s is not None and s - last[0] >= 0.05:
        last[0] = s
        print(f"sim {s:.3f} wall {time.time()-t0w:.0f} cpu {time.process_time()-t0c:.0f} rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
    elif "checkpoint" in f or "message" in f:
        print(f, flush=True)
try:
    res = run_project("/opt/plint/.claude/worktrees/thinwall/apps/cnc-twin/examples/thin-wall-al7075/project.yaml", run_dir, opts, cb)
    print("RESULT", json.dumps({k: res.get(k) for k in ("verdict", "truncated", "stopped", "status", "error")}, default=str)[:3000])
finally:
    print(f"END wall {time.time()-t0w:.0f} cpu {time.process_time()-t0c:.0f}", flush=True)
