"""Run one level of a study and measure its quantities (docs/convergence.md "Quantities").

* :func:`run_driver` - one operation through cnctwin.core.drive.OperationDriver (controller -> motion -> hot loop ->
  absorb -> the L2 steps of motion, cutting, removal and dynamics; the slow domains off): per tick the tool tip in S,
  the tick-mean and tick-peak tool wrench (T0), the spindle torque; per step the field's removed volume (exact
  ledger, and its x / y / z family estimates), cutting's analytic chip volume, removal's engaged angle, dynamics'
  vibration and chatter flags; per tooth pass cutting's emission rows; the final dexel and polar fields.
* :func:`run_loop` - a whole project through cnctwin.core.loop.Run (every domain on; run_project's class, with an
  observer of the bus after every macro step).
* Field measurements: wall positions on the dexel field, the turned profile on the polar field.

Everything is deterministic (the simulator is; no wall-clock value enters a result); CPU times are reported
separately as the cost of a level.
"""

from __future__ import annotations

import math
import time

import numpy as np

TWO_PI = 2.0 * math.pi


# ------------------------------------------------------------------------------------------------ driver runs
class DriverRun:
    """Per-tick and per-step records of one OperationDriver run (see module doc)."""

    def __init__(self, drv, *, dynamics: bool):
        self.drv = drv
        self.dynamics = dynamics
        self.ticks = []          # per tick: dict rows appended by blocks
        self.steps = []
        self.emissions = []
        drv.observers.append(self._observe)

    def _observe(self, d, rec) -> None:
        nxt = d.state.bus_next
        fr = nxt.motion.paths[0]
        tips = np.asarray(fr.tip_S_m, float)
        omega = np.asarray(fr.spindle_omega_rad_s, float)
        n = rec.tick1 - rec.tick0
        ipo = int(d.ctx.ipo_ticks)
        nt = n // ipo
        cc = nxt.cutting.contacts[0] if nxt.cutting.contacts else None
        rc = nxt.removal.contacts[0] if nxt.removal.contacts else None
        dc = nxt.dynamics.contacts[0] if (self.dynamics and getattr(nxt.dynamics, "contacts", None)) else None
        W = np.asarray(cc.wrench_tool_T0_mean, float) if cc is not None else np.zeros((nt, 6))
        P = np.asarray(cc.wrench_tool_T0_peak, float) if cc is not None else np.zeros((nt, 6))
        T = np.asarray(cc.spindle_torque_Nm, float) if cc is not None else np.zeros(nt)
        if W.shape[0] != nt:
            W = np.zeros((nt, 6))
            P = np.zeros((nt, 6))
            T = np.zeros(nt)
        om = np.abs(omega.reshape(-1)) if omega.size else np.zeros(1)
        om_ticks = om[:nt] if om.size >= nt else np.full(nt, float(om.mean()))
        self.ticks.append({"tick": rec.tick0 + ipo * np.arange(nt, dtype=np.int64), "tip": tips[:nt],
                           "W": W, "P": P, "T": T, "omega": om_ticks})
        g = rc.geometry if rc is not None else None
        row = {"tick0": rec.tick0, "tick1": rec.tick1, "in_cut_s": float(rec.in_cut_s),
               "removed_m3": float(rec.removed_m3), "analytic_m3": float(rec.analytic_m3),
               "removed_xyz_m3": np.asarray(rc.removed_volume_xyz_m3, float).tolist() if rc is not None else [0, 0, 0],
               "engaged_rad": float(g.engaged_angle_rad) if g is not None else 0.0,
               "ae_m": float(g.ae_m) if g is not None else 0.0,
               "tip_lo": tips.min(axis=0).tolist(), "tip_hi": tips.max(axis=0).tolist(),
               "l0": sorted(int(v) for v in rec.l0_ticks.values()), "cpu": dict(rec.cpu),
               "omega": float(om.mean())}
        if cc is not None:
            hdt = np.asarray(cc.int_h_dt, float)
            if hdt.ndim == 2 and hdt.size:
                row["h_dt_tooth"] = np.nansum(hdt, axis=1).tolist()     # per tooth: sum over elements of int h dt
        if dc is not None:
            row.update(vib_pp_m=float(dc.vibration_pp_m), chatter=bool(dc.chatter),
                       chatter_index=float(dc.chatter_index), chatter_freq_Hz=float(dc.chatter_freq_Hz),
                       depth_limit_m=float(dc.depth_limit_m))
        self.steps.append(row)
        if cc is not None:
            em = np.asarray(cc.emissions, float)
            if em.size:
                self.emissions.append(em)

    def run(self, max_steps: int = 20000):
        drv = self.drv
        n = 0
        t0 = time.process_time()
        while not drv.feeder.finished() and n < max_steps:
            drv.step()
            n += 1
        self.cpu_s = time.process_time() - t0
        self.result = drv.result(drv.feeder.finished())
        return self

    # -------------------------------------------------------------- selections
    def tick_arrays(self):
        if not self.ticks:
            return {}
        return {k: np.concatenate([b[k] for b in self.ticks], axis=0) for k in self.ticks[0]}

    def cost(self) -> dict:
        cut = [s for s in self.steps if s["in_cut_s"] > 0.0]
        return {"cpu_total_s": self.cpu_s, "cpu_in_cut_s": float(sum(s["cpu"].get("step", 0.0) for s in cut)),
                "in_cut_sim_s": float(sum(s["in_cut_s"] for s in cut)),
                "l0_steps": float(sum(s["cpu"].get("l0_steps", 0.0) for s in self.steps)),
                "strides": sorted({v for s in cut for v in s["l0"]})}


def steady_means(run: DriverRun, tick_mask: np.ndarray, *, teeth: int, rho: float, ipo_s: float) -> dict:
    """Quantities over the steady ticks (``tick_mask`` over :meth:`DriverRun.tick_arrays`): mean force (an integer
    number of revolutions when the revolution is a whole number of ticks), its components, the peak |F| of the
    ticks' peak samples, the mean torque; over the steps entirely inside: the field's and the analytic chip volume
    per tooth pass, the engaged angle; over the tooth passes inside: the per-tooth chip-volume spread."""
    a = run.tick_arrays()
    idx = np.flatnonzero(tick_mask)
    out = {}
    if idx.size == 0:
        return out
    # contiguous run of steady ticks, trimmed to whole revolutions
    om = float(np.mean(a["omega"][idx]))
    tpr = TWO_PI / (om * ipo_s) if om > 0.0 else 0.0
    n = idx.size
    if tpr > 0.0 and abs(tpr - round(tpr)) < 0.02 and round(tpr) >= 1:
        k = int(round(tpr))
        n = (idx.size // k) * k
    sel = idx[:n]
    W = a["W"][sel]
    F = W[:, :3].mean(axis=0)
    out["F_mean_N"] = float(np.linalg.norm(F))
    out["Fx_N"], out["Fy_N"], out["Fz_N"] = (float(v) for v in F)
    out["F_peak_N"] = float(np.max(np.linalg.norm(a["P"][idx][:, :3], axis=1)))
    out["torque_Nm"] = float(abs(np.mean(a["T"][sel])))
    out["steady_ticks"] = int(sel.size)
    out["revolutions"] = float(om * ipo_s * sel.size / TWO_PI)
    # steps entirely inside the steady ticks
    tick_set = set(int(t) for t in a["tick"][idx])
    ipo = int(round(ipo_s / 1e-6))
    inside = []
    for s in run.steps:
        ticks = range(int(s["tick0"]), int(s["tick1"]), ipo)
        if s["in_cut_s"] > 0.0 and all(t in tick_set for t in ticks):
            inside.append(s)
    if inside:
        dt = sum((s["tick1"] - s["tick0"]) * 1e-6 for s in inside)
        passes = sum(s["omega"] * (s["tick1"] - s["tick0"]) * 1e-6 * teeth / TWO_PI for s in inside)
        out["chip_volume_field_mm3"] = 1e9 * sum(s["removed_m3"] for s in inside) / passes
        out["chip_volume_analytic_mm3"] = 1e9 * sum(s["analytic_m3"] for s in inside) / passes
        xyz = np.sum([s["removed_xyz_m3"] for s in inside], axis=0)
        out["chip_volume_xyz_mm3"] = [float(1e9 * v / passes) for v in xyz]
        out["engaged_deg"] = float(np.degrees(np.mean([s["engaged_rad"] for s in inside])))
        out["steady_steps"] = len(inside)
        out["steady_s"] = float(dt)
        vib = [s["vib_pp_m"] for s in inside if "vib_pp_m" in s]
        if vib:
            out["vib_pp_um"] = float(1e6 * np.mean(vib))
        hdt = [s["h_dt_tooth"] for s in inside if s.get("h_dt_tooth")]
        if hdt and len({len(v) for v in hdt}) == 1:
            per = np.sum(np.asarray(hdt, float), axis=0)
            if per.size == teeth and per.mean() > 0.0:
                # every tooth spends the same time per revolution: the spread of int h dt between the teeth is the
                # spread of their mean chip (runout, pitch, regeneration)
                out["chip_tooth_spread"] = float((per.max() - per.min()) / per.mean())
    # tooth passes (emission rows) inside the steady ticks
    if run.emissions:
        from cnctwin.chips.abi import EM
        em = np.concatenate(run.emissions, axis=0)
        t_lo, t_hi = int(a["tick"][idx].min()), int(a["tick"][idx].max())
        rows = em[(em[:, EM["tick"]] >= t_lo) & (em[:, EM["tick"]] <= t_hi)]
        if rows.shape[0] >= teeth:
            # a tooth pass may be booked in more than one row (a pass split by a step boundary): per tooth the sum of
            # its rows over the window (every tooth passes equally often) gives the spread between the teeth
            vols = rows[:, EM["mass_kg"]] / rho
            w = vols / max(float(vols.sum()), 1e-300)
            out["h_mean_um"] = float(1e6 * np.sum(w * rows[:, EM["h_m"]]))
    return out


# ------------------------------------------------------------------------------------------------ dexel field
def ray_endpoints(field, fam: int, iu: int, iv: int, level: int = 0) -> list[tuple[float, float]]:
    """Material segments (lo, hi) in S metres of a level's ray of family ``fam`` at transverse indices (iu, iv)."""
    from cnctwin.removal import abi as RA
    from cnctwin.removal.reference import ray_index, ray_segments
    ray = ray_index(field, level, fam, iu, iv)
    o = float(field.lv_origin_m[level, fam])
    return [(o + lo * RA.QUANTUM_M, o + hi * RA.QUANTUM_M) for lo, hi, _, _ in ray_segments(field, ray)]


def wall_samples(field, fam: int, nominal_m: float, u_range, v_range, *, window_m: float = 0.5e-3,
                 level: int = 0) -> np.ndarray:
    """Positions along family ``fam`` of the material boundary nearest to ``nominal_m`` (within ``window_m``) on
    every ray of the family whose transverse coordinates lie in u_range x v_range (S metres; u, v the family's
    transverse axes in removal's order: x family (y, z), y family (x, z), z family (x, y))."""
    from cnctwin.removal import abi as RA
    u_ax, v_ax = RA.TRANS_U[fam], RA.TRANS_V[fam]
    a = float(field.lv_cell_m[level])
    o = field.lv_origin_m[level]
    nu, nv = int(field.lv_dims[level, u_ax]), int(field.lv_dims[level, v_ax])
    us = o[u_ax] + a * np.arange(nu)
    vs = o[v_ax] + a * np.arange(nv)
    iu = np.flatnonzero((us >= u_range[0]) & (us <= u_range[1]))
    iv = np.flatnonzero((vs >= v_range[0]) & (vs <= v_range[1]))
    out = []
    for i in iu:
        for j in iv:
            best = None
            for lo, hi in ray_endpoints(field, fam, int(i), int(j), level):
                for e in (lo, hi):
                    if abs(e - nominal_m) <= window_m and (best is None or abs(e - nominal_m) < abs(best - nominal_m)):
                        best = e
            if best is not None:
                out.append(best)
    return np.asarray(out, float)


def wall_stats(samples: np.ndarray, nominal_m: float) -> dict:
    if samples.size == 0:
        return {}
    return {"wall_mm": float(1e3 * np.mean(samples)), "wall_dev_um": float(1e6 * (np.mean(samples) - nominal_m)),
            "wall_min_mm": float(1e3 * samples.min()), "wall_max_mm": float(1e3 * samples.max()),
            "wall_n": int(samples.size)}


def field_volume_m3(field) -> list[float]:
    from cnctwin.removal import field as FD
    return [float(FD.volume_m3(field, 0, f)) for f in range(3)]


# ------------------------------------------------------------------------------------------------ polar field
def turned_profile(pf, z_lo_m: float, z_hi_m: float):
    """(z [nz_sel], r [n_theta, nz_sel]) outer radius of every bin over the rows whose centres lie in [z_lo, z_hi]."""
    from cnctwin.removal import abi as RA
    g = pf.geom
    nth, nz = int(g[RA.PG_NTHETA]), int(g[RA.PG_NZ])
    z0, dz, q = float(g[RA.PG_Z0]), float(g[RA.PG_DZ]), float(g[RA.PG_QUANTUM])
    zc = z0 + (np.arange(nz) + 0.5) * dz
    rows = np.flatnonzero((zc >= z_lo_m) & (zc <= z_hi_m))
    cnt = np.asarray(pf.rcnt, np.int64).reshape(nth, nz)[:, rows]
    seg = np.asarray(pf.rseg).reshape(nth, nz, -1, 2)[:, rows]
    last = np.maximum(cnt - 1, 0)
    r = np.take_along_axis(seg[..., 1], last[..., None], axis=2)[..., 0] * q
    r = np.where(cnt > 0, r, 0.0)
    return zc[rows], r


def roughness(z: np.ndarray, r: np.ndarray) -> dict:
    """Ra and Rt of the turned profiles r(z) (one per bin): the mean line a least-squares straight line per bin
    (removes taper and the deflection slope), Ra the mean absolute deviation, Rt the peak-to-valley; both averaged
    over the bins."""
    if r.size == 0 or z.size < 3:
        return {}
    A = np.vstack([z - z.mean(), np.ones_like(z)]).T
    coef, *_ = np.linalg.lstsq(A, r.T, rcond=None)
    dev = r - (A @ coef).T
    ra = np.mean(np.abs(dev), axis=1)
    rt = dev.max(axis=1) - dev.min(axis=1)
    return {"Ra_um": float(1e6 * ra.mean()), "Rt_um": float(1e6 * rt.mean()), "r_mean_mm": float(1e3 * r.mean())}


# ------------------------------------------------------------------------------------------------ the whole loop
def run_loop(project_path, run_dir, *, options=None, observe=None):
    """cnctwin.core.loop.Run (what run_project runs) with ``observe(run)`` called after every macro step (the bus of
    the step is ``run.state.bus_prev`` then). Returns (result dict, Run)."""
    from cnctwin.core.loop import Run

    class Observed(Run):
        def step(self):
            super().step()
            if observe is not None:
                observe(self)

    opts = {"outputs": False, "sanity": False}
    opts.update(options or {})
    phases = {}
    last = {"phase": "setup", "t": time.process_time()}

    def progress(fields):
        phase = fields.get("phase")
        if phase and phase != last["phase"]:
            now = time.process_time()
            phases[last["phase"]] = phases.get(last["phase"], 0.0) + now - last["t"]
            last.update(phase=phase, t=now)

    run = Observed(project_path, run_dir, opts, progress)
    result = run.execute()
    now = time.process_time()
    phases[last["phase"]] = phases.get(last["phase"], 0.0) + now - last["t"]
    result["cpu_phases_s"] = phases
    return result, run
