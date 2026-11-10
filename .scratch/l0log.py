"""L0-step log in Python around the compiled kernels (env L0LOG=path, L0_T0, L0_T1 in s)."""
import os, json, numpy as np
from cnctwin.core import hotloop as HL
from cnctwin.cutting import abi as CA
path = os.environ["L0LOG"]
T0, T1 = float(os.environ.get("L0_T0", "0")), float(os.environ.get("L0_T1", "1e9"))
out = open(path, "a")
_cut, _corr, _rm, _cs = HL.cut_step, HL.modal_correct, HL.run_macro, HL.contact_steps
st = {"hb": 0.0}
def cut_step(tool, table, mem, iv, refill, step, istep, pose, dfl, occ, pack, axes, kst, acc, cacc, cout, ev):
    hb0 = cacc.scalars[CA.CA_H_BEYOND]
    _cut(tool, table, mem, iv, refill, step, istep, pose, dfl, occ, pack, axes, kst, acc, cacc, cout, ev)
    st["rec"] = {"tk": int(istep[CA.SI_TICK_END]), "F": [round(float(v), 2) for v in cout.force_S_N],
                 "eng": int(cout.misc[CA.SO_ENGAGED]), "hmax": float(cacc.scalars[CA.CA_H_MAX]) * 1e6,
                 "hb": float(cacc.scalars[CA.CA_H_BEYOND] - hb0), "hbm": float(cacc.scalars[CA.CA_H_BEYOND_MAX]),
                 "u_st": [round(float(v) * 1e6, 3) for v in dfl.u_st_T0_m[:, :3].max(axis=0)],
                 "u_st_min": [round(float(v) * 1e6, 3) for v in dfl.u_st_T0_m[:, :3].min(axis=0)],
                 "v_st": [round(float(v), 4) for v in np.abs(dfl.v_st_T0_m_s[:, :3]).max(axis=0)],
                 "uw": [round(float(v) * 1e6, 3) for v in dfl.u_work_S_m], "vw": [round(float(v), 4) for v in dfl.v_work_S_m_s],
                 "stw_max": float(np.abs(cout.station_T0[:, :3]).max())}
def modal_correct(sys, stw, f, tick, acc):
    _corr(sys, stw, f, tick, acc)
    r = st.pop("rec", {"tk": int(tick)})
    x = sys.state.x
    r["xq"] = float(np.abs(x[:, 0]).max()); r["xv"] = float(np.abs(x[:, 1]).max())
    r["kmax"] = int(np.argmax(np.abs(x[:, 0])))
    r["fcut"] = float(np.abs(sys.ds.f_cut).max())
    r["wf"] = [round(float(v), 2) for v in sys.ds.work_force_S]
    if "dumped" not in st:
        st["dumped"] = 1
        out.write(json.dumps({"omega_Hz": [float(w) / 6.2832 for w in sys.dm.omega_rad_s], "is_work": [int(v) for v in sys.model.is_work],
                              "row": [int(v) for v in sys.dm.row], "dt": float(sys.model.dt_s[0]), "zeta": sys.dm.zeta.tolist(), "relax": sys.dm.relax.tolist(),
                              "scal": sys.dm.scal.tolist() if hasattr(sys.dm, "scal") else None}) + "\n")
    r["xvall"] = [float("%.3g" % v) for v in x[:, 1]]
    r["fx"] = [float("%.3g" % v) for v in sys.fx]
    from cnctwin.core.abi import ACC_PROCESS_DAMPING_J as _APD
    r["pdJ"] = float(acc.scalars[_APD])
    out.write(json.dumps(r) + "\n")
_run_step = HL.HotLoop.run_step
def run_step(self, feeder):
    t = int(self.state.clock.tick) * self.ctx.tick_s
    on = T0 <= t < T1
    if on:
        HL.cut_step, HL.modal_correct, HL.run_macro, HL.contact_steps = cut_step, modal_correct, _rm.py_func, _cs.py_func
        out.write(json.dumps({"macro": t}) + "\n")
    try:
        return _run_step(self, feeder)
    finally:
        if on:
            HL.cut_step, HL.modal_correct, HL.run_macro, HL.contact_steps = _cut, _corr, _rm, _cs
            out.flush()
HL.HotLoop.run_step = run_step
