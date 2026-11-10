import sys
sys.path.insert(0, "/opt/plint/.claude/worktrees/softening/apps/cnc-twin/tests/unit/mechanics")
if len(sys.argv) > 1 and sys.argv[1] == "before":
    sys.path.insert(0, "/opt/plint/.claude/worktrees/softening/.scratch/s"); import nosoft
import test_validation_orthogonal as T
import conftest
from cnctwin.mechanics.materials import ToolSide
from cnctwin.db.simap import wrap
from cnctwin.mechanics import coefficients as CF
carb = ToolSide.from_db(None, wrap(conftest.CARBIDE), None)
for name, mat, data in (("ck45", conftest.work_from(conftest.STEEL_1045), T.CK45), ("ti64", conftest.work_from(T.TI), T.TI64)):
    res, cal = T._predict(mat, carb, data)
    reg = T._regimes(mat, carb, data)
    for key, (f_raw, f_cal, t2, (fc, ff, t2m)) in res.items():
        print(name, key, "OX" if reg[key] == CF.REGIME_OXLEY else "MER",
              f"raw Fc {f_raw[0]/fc-1:+.3f} Ff {f_raw[1]/ff-1:+.3f} | cal Fc {f_cal[0]/fc-1:+.3f} Ff {f_cal[1]/ff-1:+.3f} | t2 {t2/t2m-1:+.3f}")
# softening of the ETH Ck45 cases
if len(sys.argv) > 2:
    import numpy as np
    mat = conftest.work_from(conftest.STEEL_1045)
    speeds = np.array(sorted({v for v, _ in T.CK45})) / 60.0
    pmap = CF.build_process_map(mat, carb, 0.0, speed_grid_m_s=speeds)
    for (v, h) in sorted(T.CK45):
        gc = CF.group_coefficients(mat, carb, T.GROUP, CF.SlowState(speed_m_s=v / 60.0, tool_temp_K=293.15), np.array([h * 1e-3]), pmap)
        print("soft", v, h, round(float(gc.edge_softening[0]), 3), round(gc.edge_softening_ref, 3), round(float(gc.edge_flow_temp_K[0]) - 273.15))
    print("C", mat.jc[3])
