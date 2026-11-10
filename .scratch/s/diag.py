import math, sys
import numpy as np
sys.path.insert(0, "/opt/plint/.claude/worktrees/softening/apps/cnc-twin/tests/unit/thermal")
import test_surface_damage as T
from cnctwin.db.database import Database
from cnctwin.mechanics import coefficients as CO, contact as C
from cnctwin.mechanics.oxley import effective_mu
db = Database.load()
kr = math.radians(95.0)
cases = {
 "warmup": ("aisi4140-qt30", "carbide-p30", 0.0, 40e-6, 132/60, 0.25e-3*math.sin(kr), 1.5e-3/math.sin(kr), 50e-6, "al2o3-cvd"),
 "chen0.1": ("aisi52100-62hrc", "cbn-low", -5.0, 25e-6, 100/60, None, None, 0.0, None),
 "worn": ("aisi52100-62hrc", "cbn-low", -6.0, 20e-6, 200/60, 1e-4*math.sin(kr), 2e-4/math.sin(kr), 3e-4, None),
}
f_mm = 0.1; engaged = 0.8e-3*math.acos(1-0.1/0.8)+0.5*f_mm*1e-3
cases["chen0.1"] = cases["chen0.1"][:5] + (f_mm*1e-3*0.1e-3/engaged, engaged) + cases["chen0.1"][7:]
for name, (mid, tm, rake, r, v, h, w, vb, coat) in cases.items():
    work, st, src = T._contact(db, mid, tm, rake, r, v, h, w, vb, coat=coat)
    from cnctwin.validation.harnesses.contact import ContactModel
    from cnctwin.db.simap import wrap
    from cnctwin.mechanics.materials import ToolSide
    item = {"type": "turning_insert", "coating": {"id": coat or "none"}, "geometry": {"insert": {}}}
    tool = ToolSide.from_db(wrap(item), db.tool_material(tm), db.coating(coat) if coat else None)
    model = ContactModel(work, tool, rake_rad=math.radians(rake), clearance_rad=math.radians(6.0), edge_radius_m=r,
                         work_temp_K=293.15, speeds_m_s=[v], h_values_m=[h])
    slow = CO.SlowState(v, tool_temp_K=600.0, tool_flank_temp_K=600.0, work_temp_K=293.15, VB_m=vb, contact_width_m=w, contact_time_s=1.0)
    k_ref, prim, t_ref = CO._reference_state(work, model.pmap, model.h_min, slow)
    mu = effective_mu(work, tool, 1.0)
    gc = CO.group_coefficients(work, tool, model.group, slow, np.array([h]), model.pmap)
    sig0, tau0 = C.flank_reference_stresses(k_ref, mu); sm, tm_ = C.flank_stresses(vb, sig0, tau0)
    t_ws = float(gc.T_shear_K[0])
    for label, tws in (("node", t_ws), ("ref", t_ref)):
        kc, share, lf, info = CO.softened_flank_contact(work, tool, r, math.radians(rake), 0.0, model.h_min, model.h_min, k_ref, prim, mu, vb, tm_/k_ref, v, w, tws, 600.0, 1.0)
        print(name, label, f"k_ref {k_ref/1e6:.0f} MPa prim eps {prim.strain:.2f} rate {prim.strain_rate_1_s:.3g} T_AB {prim.temp_K-273.15:.0f}C | T_ws {tws-273.15:.0f}C "
              f"T_flow {info['T_flow_K']-273.15:.0f}C rate_c {info['strain_rate_1_s']:.3g} s {info['softening']:.3f} k_c {kc/1e6:.0f} MPa share {share:.3f} "
              f"solidus {info['solidus_bound']} T_contact(clamped) {info['T_contact_K']-273.15:.0f}C sol {CO.solidus_K(work)}")
    print(name, "Kte", gc.Kte_N_per_m/1e3, "N/mm; sigma_mean", gc.flank_sigma_mean_Pa/1e6, "tau_mean", gc.flank_tau_mean_Pa/1e6, "frac", gc.frac_flank_to_tool, "pfe", gc.power_frac_edge)
