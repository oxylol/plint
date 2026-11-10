"""Switches (env SW=comma list) and a per-macro-step log (env STEPLOG=path, from STEPLOG_T0 s)."""
import os, json, numpy as np
from cnctwin.core import loop as L
SW = set(filter(None, os.environ.get("SW", "").split(",")))
print("switches", SW, flush=True)
if "norecut" in SW or "nopack" in SW:
    from cnctwin.cutting import domain as CD, abi as CA
    if "norecut" in SW:
        CD.CuttingDomain._occupancy = staticmethod(lambda bus: CA.empty_occupancy())
    if "nopack" in SW:
        CD.CuttingDomain._packing = staticmethod(lambda bus, slot: CA.CutPacking(np.zeros(1), np.zeros(3)))
if "nopd" in SW:
    from cnctwin.cutting import domain as CD, abi as CA
    _b = CD.build_cut_tool
    def bct(*a, **k):
        t = _b(*a, **k)
        t.pk[:, CA.PK_PD] = 0.0
        return t
    CD.build_cut_tool = bct
for name in ("thermal", "holding", "coolant", "structure"):
    if "no" + name in SW:
        _orig_init = L.Run._init_domains
        def _mk(name, _orig_init):
            def init(self):
                _orig_init(self)
                dom = self.domains[name]
                dom.macro_step = lambda *a, **k: None
            return init
        L.Run._init_domains = _mk(name, _orig_init)

path = os.environ.get("STEPLOG")
if path:
    t0 = float(os.environ.get("STEPLOG_T0", "0"))
    out = open(path, "a")
    _step = L.Run.step
    def step(self):
        _step(self)
        t = int(self.state.clock.tick) * self.ctx.tick_s
        if t < t0:
            return
        nb = self.state.bus_prev
        rec = {"t": round(t, 5)}
        for c in nb.cutting.contacts:
            if c.in_cut_s <= 0 and c.h_beyond == 0:
                continue
            rec.update(slot=c.tool_slot, incut=round(c.in_cut_s, 5), Fpk=round(float(c.force_peak_inst_N), 1),
                       Fm=[round(float(v), 1) for v in np.mean(c.wrench_tool_T0_mean[:, :3], axis=0)] if len(c.wrench_tool_T0_mean) else None,
                       hmax=round(c.h_max_step_m * 1e6, 2), hb=c.h_beyond, hbm=c.h_beyond_max_m, rf=c.refill_missing,
                       recutJ=c.recut_J, packJ=c.packing_J, pdJ=c.process_damping_J, defJ=c.deflection_work_J,
                       Tpk=round(c.torque_peak_Nm, 3))
        for d in getattr(nb.dynamics, "contacts", []):
            rec.update(vpp=d.vibration_pp_m * 1e6, dpk=[round(float(v) * 1e6, 2) for v in d.deflection_peak_T0_m],
                       Emod=d.modal_energy_J, nmod=d.n_modes, sv=d.structure_version)
        out.write(json.dumps(rec, default=float) + "\n"); out.flush()
    L.Run.step = step

# frz_<domain>: that domain sees thermal's record frozen at its first macro step of the run (resume at 4.3 s)
import dataclasses
for name in ("tooling", "cutting", "structure", "holding", "wear", "coolant", "surface", "chips", "dynamics", "removal"):
    if "frz_" + name in SW:
        _oi = L.Run._init_domains
        def _mk2(name, _oi):
            def init(self):
                _oi(self)
                dom = self.domains[name]
                orig = dom.macro_step
                box = {}
                def ms(ctx, state, prev, nxt):
                    if "th" not in box:
                        box["th"] = prev.thermal
                    return orig(ctx, state, dataclasses.replace(prev, thermal=box["th"]), nxt)
                dom.macro_step = ms
            return init
        L.Run._init_domains = _mk2(name, _oi)

if "frz_motion" in SW:
    from cnctwin.motion import domain as MD
    _de = MD.MotionDomain._dynamic_errors
    _box = {}
    def _dyn_err(self, bus):
        if bus is not None and "th" not in _box:
            _box["th"] = bus.thermal
        if bus is not None:
            bus = dataclasses.replace(bus, thermal=_box["th"])
        return _de(self, bus)
    MD.MotionDomain._dynamic_errors = _dyn_err
if path:
    _step2 = L.Run.step
    def step2(self):
        th = self.state.bus_prev.thermal
        t = int(self.state.clock.tick) * self.ctx.tick_s
        if t >= t0 and (not os.environ.get("STEPLOG_SPARSE") or int(round(t * 100)) % 50 == 0):
            rec = {"th_t": round(t, 4), "sg": {k: [round(float(x) * 1e6, 3) for x in v] for k, v in (th.spindle_growth_M_m or {}).items()},
                   "sd": [round(float(x) * 1e6, 3) for x in th.structure_drift_M_m],
                   "ad": {k: [round(float(x) * 1e6, 3) for x in v] for k, v in (th.axis_drift or {}).items()},
                   "lg": {k: round(float(getattr(v, "length_growth_m", 0)) * 1e6, 3) for k, v in (th.tools or {}).items()}}
            out.write(json.dumps(rec) + "\n")
        _step2(self)
    L.Run.step = step2

if os.environ.get("PROBE_KC"):
    from cnctwin.dynamics import modal as MDm, domain as DDm
    _bs = MDm.build_system
    _kc = open(os.environ["PROBE_KC"], "a")
    def build_system(modes, **kw):
        sys_, lay = _bs(modes, **kw)
        Kc = kw.get("K_c_T0"); Kr = kw.get("K_res_T0")
        rec = {"nm": len(modes), "l0": kw.get("l0_ticks"), "ref": kw.get("ref_station"),
               "Kc": None if Kc is None else np.round(np.asarray(Kc), -3).tolist(),
               "Kres": None if Kr is None or isinstance(Kr, str) else np.round(np.asarray(Kr), -3).tolist(),
               "relax_diag": np.round(np.diag(sys_.dm.relax), 3).tolist(),
               "res_relax": np.round(sys_.dm.scal[MDm.MS_RES_RELAX:MDm.MS_RES_RELAX + 9], 3).tolist() if hasattr(MDm, "MS_RES_RELAX") else None,
               "G1": [float("%.3g" % v) for v in sys_.model.G1[:, 0]],
               "cshape": np.round(sys_.dm.contact_shape, 3).tolist()}
        _kc.write(json.dumps(rec) + "\n"); _kc.flush()
        return sys_, lay
    MDm.build_system = build_system

if path:
    _step3 = L.Run.step
    def step3(self):
        _step3(self)
        t = int(self.state.clock.tick) * self.ctx.tick_s
        if t < t0 or (os.environ.get("STEPLOG_SPARSE") and int(round(t * 100)) % 50 != 0):
            return
        b = self.state.bus_prev
        rec = {"w_t": round(t, 4)}
        w = b.wear.tools.get(0) if hasattr(b.wear, "tools") else None
        if w is not None:
            for k in ("VB_m", "bue_height_m", "notch_m", "chipping_m", "edge_radius_m"):
                v = getattr(w, k, None)
                if v is not None and np.size(v):
                    rec[k] = float(np.nanmax(np.asarray(v, float)))
        try:
            eff = self.state.tooling.effective[0]
            rec["er_max"] = float(np.max(eff.elem_edge_radius_m)); rec["er_min"] = float(np.min(eff.elem_edge_radius_m))
        except Exception as ex:
            rec["er_err"] = str(ex)[:80]
        co = b.cutting.coefficients.get(0) if hasattr(b.cutting, "coefficients") else None
        if co is not None:
            rec["Ksp"] = np.asarray(co.K_sp_N_per_m3, float).tolist()
        tc = self.state.cutting.tools.get(0)
        if tc is not None and tc.tool is not None:
            from cnctwin.cutting import abi as CA
            pk = tc.tool.pk
            rec["pd0_max"] = float(pk[:, CA.PK_PD].max()); rec["land_max"] = float(pk[:, CA.PK_PD + 1].max())
            rec["VBpk_max"] = float(pk[:, CA.PK_VB].max())
        out.write(json.dumps(rec, default=float) + "\n"); out.flush()
    L.Run.step = step3

if os.environ.get("PROBE_WEAR"):
    from cnctwin.wear import domain as WD, kernels as WK
    _adv = WK.advance_elements
    _pw = open(os.environ["PROBE_WEAR"], "a")
    def adv(geo, inp, law, elem, out):
        vn0 = elem.vn.copy(); kt0 = elem.kt.copy()
        _adv(geo, inp, law, elem, out)
        i = np.unravel_index(int(np.argmax(out.d_vn)), out.d_vn.shape)
        rec = {"d_vn_max": float(out.d_vn.max()), "at": [int(v) for v in i], "vn0": float(vn0[i]), "VB": float(out.VB[i]),
               "in_cut": float(inp.in_cut_s[i]), "sv_flank": float(inp.sv_flank[i]), "sv_rake": float(inp.sv_rake[i]),
               "flank_T": np.asarray(inp.flank_T[i]).round(1).tolist(), "rake_T": np.asarray(inp.rake_T[i]).round(1).tolist(),
               "cycle_dT": float(inp.cycle_dT[i]), "entries": float(inp.entries[i]), "flankT_max": float(np.max(inp.flank_T)),
               "rakeT_max": float(np.max(inp.rake_T)), "sv_flank_max": float(np.max(inp.sv_flank)), "d_kt_max": float(out.d_kt.max()),
               "d_D_max": float(out.d_D.max()), "edge_r_max": float(out.edge_r.max()), "d_notch_max": float(out.d_notch.max()),
               "flA": law.fl_A.tolist(), "flB": law.fl_B.tolist(), "fllo": law.fl_lo.tolist()}
        _pw.write(json.dumps(rec, default=float) + "\n"); _pw.flush()
    WD.K.advance_elements = adv

if os.environ.get("SKIP_FINAL_KEYFRAME"):
    # harness only: the end-of-run full keyframe (every surface brick meshed at once) exceeds the 1.8 GB cap
    from cnctwin.core import loop_outputs as LO
    _wk = LO.write_keyframe
    def wk(run, tick, full=False):
        if full:
            print("skipped the final full keyframe", flush=True)
            return None
        return _wk(run, tick, full)
    LO.write_keyframe = wk
