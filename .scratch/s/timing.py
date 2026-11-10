import sys, time, math
import numpy as np
if sys.argv[1] == "before":
    sys.path.insert(0, "/opt/plint/.claude/worktrees/softening/.scratch/s"); import nosoft
sys.path.insert(0, "/opt/plint/.claude/worktrees/softening/apps/cnc-twin/tests/unit/thermal")
import test_surface_damage as T
from cnctwin.db.database import Database
from cnctwin.mechanics import coefficients as CO
db = Database.load()
kr = math.radians(95.0)
from cnctwin.validation.harnesses.contact import ContactModel
from cnctwin.db.simap import wrap
from cnctwin.mechanics.materials import ToolSide, WorkMaterial
work = WorkMaterial.from_db(db.material("aisi4140-qt30"))
item = {"type": "turning_insert", "coating": {"id": "al2o3-cvd"}, "geometry": {"insert": {}}}
tool = ToolSide.from_db(wrap(item), db.tool_material("carbide-p30"), db.coating("al2o3-cvd"))
model = ContactModel(work, tool, rake_rad=0.0, clearance_rad=0.1, edge_radius_m=40e-6, work_temp_K=293.15, speeds_m_s=[2.2], h_values_m=[2.5e-4])
h = np.geomspace(2e-6, 1e-3, 24)
slow = CO.SlowState(2.2, tool_temp_K=600.0, tool_flank_temp_K=650.0, VB_m=5e-5, contact_width_m=1.5e-3)
CO.group_coefficients(work, tool, model.group, slow, h, model.pmap)
t = time.perf_counter()
for i in range(20):
    slow = CO.SlowState(2.2, tool_temp_K=600.0 + i, tool_flank_temp_K=650.0 + i, VB_m=5e-5, contact_width_m=1.5e-3)
    CO.group_coefficients(work, tool, model.group, slow, h, model.pmap)
print(sys.argv[1], "ms per group (24 nodes):", (time.perf_counter() - t) / 20 * 1e3)
