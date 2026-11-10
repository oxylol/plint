"""Probe: warmup 4140 OD pass and Chen 2017 cases through the harness chain."""
import math, sys, json
sys.path.insert(0, "/opt/plint/.claude/worktrees/softening/apps/cnc-twin/tests/unit/thermal")
import test_surface_damage as T
from cnctwin.db.database import Database
from cnctwin.thermal import surfacetemp as ST
db = Database.load()
out = {}
kr = math.radians(95.0)
work, st, src = T._contact(db, "aisi4140-qt30", "carbide-p30", 0.0, 40e-6, 132 / 60, 0.25e-3 * math.sin(kr),
                           1.5e-3 / math.sin(kr), 50e-6, coat="al2o3-cvd")
hist, v = T._judge(db, "aisi4140-qt30", work, src)
k, a = work.conductivity(293.15), work.diffusivity(293.15)
out["warmup"] = dict(Ft=st.force_t_N, Fr=st.force_r_N, edge_W=st.edge_W, land_W=st.land_W, shear_W=st.shear_W,
                     f_flank=st.frac_flank_to_tool, l_f=st.flank_contact_m, arc_W=src.arc_W, plough_W=src.plough_W,
                     edge_Npmm=st.edge_W / st.width_m / st.speed_m_s / 1e3, T_shear=st.T_shear_K - 273.15,
                     T_flank=st.flank_temp_K - 273.15, peak_um=[round(float(x) - 273.15) for x in hist.peak_K()], depths_um=[round(float(z)*1e6,2) for z in hist.depth_m],
                     T_edge=293.15 + ST.rise(src, k, a, 0.0, 0.0, T.GLX, T.GLW) - T.ZERO,
                     T_mean=293.15 + ST.contact_mean_rise(src, k, a, T.GLX, T.GLW) - T.ZERO,
                     verdicts={kk: (vv.damaged, round(float(vv.metric), 1), round(float(vv.limit), 1),
                                    float(vv.depth_m) * 1e6) for kk, vv in v.items()})
chen = []
for f_mm in T.CHEN_FEEDS_MM:
    engaged = 0.8e-3 * math.acos(1.0 - 0.1 / 0.8) + 0.5 * f_mm * 1e-3
    h = f_mm * 1e-3 * 0.1e-3 / engaged
    work, st, src = T._contact(db, "aisi52100-62hrc", "cbn-low", -5.0, 25e-6, 100 / 60, h, engaged, 0.0)
    k, a = work.conductivity(293.15), work.diffusivity(293.15)
    chen.append(dict(f=f_mm, edge_W=st.edge_W, edge_Npmm=st.edge_W / st.width_m / st.speed_m_s / 1e3,
                     f_flank=st.frac_flank_to_tool, l_f=st.flank_contact_m * 1e6,
                     q_work_GW=(src.arc_W + src.plough_W) / (src.arc_length_m * src.width_m) / 1e9,
                     T_edge=293.15 + ST.rise(src, k, a, 0.0, 0.0, T.GLX, T.GLW) - T.ZERO,
                     T_mean=293.15 + ST.contact_mean_rise(src, k, a, T.GLX, T.GLW) - T.ZERO))
out["chen"] = chen
worn = T._contact(db, "aisi52100-62hrc", "cbn-low", -6.0, 20e-6, 200 / 60, 1e-4 * math.sin(kr), 2e-4 / math.sin(kr), 3e-4)
hist, v = T._judge(db, "aisi52100-62hrc", worn[0], worn[2])
w = worn[0]; k, a = w.conductivity(293.15), w.diffusivity(293.15)
out["worn_cbn"] = dict(edge_W=worn[1].edge_W, land_W=worn[1].land_W,
                       T_surface_peak=293.15 + ST.contact_mean_rise(worn[2], k, a, T.GLX, T.GLW) - T.ZERO,
                       peak=[round(float(x) - 273.15) for x in hist.peak_K()],
                       wl_depth=float(v["white_layer"].depth_m) * 1e6)
print(json.dumps(out, indent=1, default=float))
