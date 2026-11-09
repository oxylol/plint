"""Plain-Python reference of the milling kernel (docs/models/cutting.md "Verification").

:func:`cut_step_ref` evaluates one L0 step of :func:`cnctwin.cutting.kernels.cut_step` with an
independent formulation: explicit rotation matrices (S <- T0 <- T), vectors in S for positions, normals
and velocities, the memory interpolated in S, the engaged part of each swept arc from a direct
intersection of angle intervals, the exact rotation of clipped elements' forces (the kernel uses a
first-order rotation), station loads summed directly in T0, and the energy split computed from -F.v
per element. It returns the step's results without touching its inputs: forces and moments, station
loads, energies, volumes, per-element increments, the element chip thicknesses, and the memory and
pending writes after the step. Bookkeeping that the kernel shares with turning (shocks, passes, events)
is checked by the scenario tests, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import abi as A
from .coefficients import table_forces

TWO_PI = 2.0 * math.pi


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class RefResult:
    wrench_T0: np.ndarray
    station_T0: np.ndarray
    heat_shear_J: float = 0.0
    heat_rake_J: float = 0.0
    heat_flank_J: float = 0.0
    ploughing_J: float = 0.0
    process_damping_J: float = 0.0
    chip_volume_m3: float = 0.0
    displaced_m3: float = 0.0
    shaft_J: float = 0.0
    feed_J: float = 0.0
    deflection_J: float = 0.0
    elem_acc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    h: dict = field(default_factory=dict)            # element -> h at its evaluation
    in_cut: dict = field(default_factory=dict)       # element -> engaged fraction
    memory: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mem_tick: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pend: np.ndarray = field(default_factory=lambda: np.zeros(0))


def engaged_part(intervals: list[tuple[float, float]], a_mid: float, dal: float) -> tuple[float, float, float, float]:
    """(fraction, centroid s, s_lo, s_hi) of the arc angle(s) = a_mid + (s - 1/2) dal, s in [0, 1], inside the
    union of the intervals [lo, hi] (taken modulo 2 pi)."""
    if dal == 0.0:
        for lo, hi in intervals:
            for m in (-1, 0, 1):
                if lo + m * TWO_PI <= a_mid <= hi + m * TWO_PI:
                    return 1.0, 0.5, 0.0, 1.0
        return 0.0, 0.0, 0.0, 0.0
    a0 = a_mid - 0.5 * abs(dal)
    a1 = a_mid + 0.5 * abs(dal)
    pieces = []
    for lo, hi in intervals:
        for m in (-1, 0, 1):
            x0 = max(lo + m * TWO_PI, a0)
            x1 = min(hi + m * TWO_PI, a1)
            if x1 > x0:
                pieces.append((x0, x1))
    if not pieces:
        return 0.0, 0.0, 0.0, 0.0
    total = sum(x1 - x0 for x0, x1 in pieces)
    frac = total / (a1 - a0)
    if frac >= 1.0 - 1e-12:
        return 1.0, 0.5, 0.0, 1.0
    first = min(p[0] for p in pieces)
    last = max(p[1] for p in pieces)
    s_vals = sorted([0.5 + (first - a_mid) / dal, 0.5 + (last - a_mid) / dal])
    s_lo, s_hi = max(s_vals[0], 0.0), min(s_vals[1], 1.0)
    return frac, 0.5 * (s_lo + s_hi), s_lo, s_hi


def commit_ref(mem_pts, mem_tick, mem_valid, s, b, p, h, h_write, ratio, tick) -> None:
    if h < h_write:
        return
    if h_write <= 0.0 or ratio <= 0.0 or h >= (1.0 + ratio) * h_write or mem_valid[s, b] == 0:
        w = 1.0
    else:
        w = (h - h_write) / (ratio * h_write)
    mem_pts[s, b] = mem_pts[s, b] + w * (np.asarray(p) - mem_pts[s, b])
    mem_tick[s, b] = tick
    mem_valid[s, b] = 1


def wave_area_ref(ring: np.ndarray, aux: np.ndarray, e: int, u: np.ndarray, n: np.ndarray, travel: float,
                  land: float, tan_clear: float, x_max: float, dx: float) -> float:
    """Process damping's indented area from the element's wave memory (independent of kernels.wave_area): the
    profile delta(x) = (u - u(x)) . n - max(0, x - L) tan(gamma) behind the edge on the nodes (current point, last
    evaluation, ring samples), refined at the land's end and at x_max, and its positive part integrated exactly
    segment by segment (delta is linear between the refined nodes)."""
    cnt = int(aux[e, A.PDA_COUNT])
    if cnt <= 0 or x_max <= 0.0:
        return 0.0
    L = min(max(float(land), 0.0), x_max)
    nk = ring.shape[1]
    xs = [0.0, travel - float(aux[e, A.PDA_X])]
    us = [np.asarray(u, float), np.array(aux[e, A.PDA_UX:A.PDA_UZ + 1], float)]
    for g in range(cnt - 1, max(cnt - nk, 0) - 1, -1):
        xs.append(travel - g * dx)
        us.append(np.array(ring[e, g % nk], float))
    xs = np.array(xs)
    du = np.array([float((us[0] - v) @ n) for v in us])
    # refine: the land's end and x_max inside the node range (u linear between nodes)
    cuts = sorted({c for c in (L, x_max) if xs[0] < c < xs[-1]})
    grid = sorted(set(xs.tolist()) | set(cuts))
    grid = [x for x in grid if x <= x_max]
    dd = np.interp(grid, xs, du)
    prof = dd - np.maximum(np.array(grid) - L, 0.0) * tan_clear
    area = 0.0
    for i in range(len(grid) - 1):
        x0, x1, f0, f1 = grid[i], grid[i + 1], prof[i], prof[i + 1]
        w = x1 - x0
        if w <= 0.0:
            continue
        if f0 >= 0.0 and f1 >= 0.0:
            area += 0.5 * (f0 + f1) * w
        elif f0 > 0.0 or f1 > 0.0:
            fp = max(f0, f1)
            area += 0.5 * fp * fp / abs(f1 - f0) * w
    return area


def _field_depth(refill: A.Refill | None, s: int, b: int, p: np.ndarray, n: np.ndarray) -> float | None:
    """Depth of p below the field plane of bin (s, b) along n (None without a plane)."""
    if refill is None or not refill.valid[s, b]:
        return None
    m = np.asarray(refill.m_S[s, b], float)
    d = float((np.asarray(refill.q_S_m[s, b], float) - p) @ m)
    den = -float(n @ m)
    return d / den if den > 0.2 else d


def cut_step_ref(tool: A.CutTool, table: A.CutTable, mem: A.SurfaceMemory, iv: A.EngagedIntervals, step: np.ndarray,
                 istep: np.ndarray, pose: A.StepPose, defl: A.Deflection, st: A.CutState,
                 refill: A.Refill | None = None) -> RefResult:
    """Reference evaluation of one step (fresh, non-stale memory bins; no recut or packing). Elements that read
    the field (read row -1) take the planes of ``refill``."""
    dt = float(step[A.ST_DT_S])
    omega = float(step[A.ST_OMEGA_RAD_S])
    theta = float(step[A.ST_THETA_MID_RAD])
    tick_end = int(istep[A.SI_TICK_END])
    l0 = int(istep[A.SI_L0_TICKS])
    tick0 = tick_end - l0
    dal = omega * dt
    nb = mem.point_S_m.shape[1]
    width = TWO_PI / nb
    ratio = float(tool.scal[A.TC_RECOVERY_RATIO])
    pd_max = float(tool.scal[A.TC_PD_MAX_LEN])
    R = np.asarray(pose.R_S_T0, float)
    tip = np.asarray(pose.tip_S_m, float)
    v_tip = np.asarray(pose.v_tip_S_m_s, float)
    w_frame = np.asarray(pose.w_T0_S_rad_s, float)
    u_work = np.asarray(defl.u_work_S_m, float)
    v_work = np.asarray(defl.v_work_S_m_s, float)
    ez_S = R[:, 2]
    n = tool.p_T_m.shape[0]
    nst = tool.st_z_m.shape[0]
    mem_pts = np.array(mem.point_S_m, float)
    mem_tick = np.array(mem.tick)
    mem_valid = np.array(mem.valid)
    pend = np.array(st.pend, float)
    res = RefResult(np.zeros(6), np.zeros((nst, 6)), elem_acc=np.zeros((n, A.N_ELEM_ACC)))
    F_S_tot = np.zeros(3)
    M_S_tot = np.zeros(3)          # about the tip, S
    for s in range(nst):
        cnt = int(iv.count[s])
        if cnt <= 0:
            continue
        intervals = [(float(iv.lo_rad[s, i]), float(iv.hi_rad[s, i])) for i in range(cnt)]
        z_st = float(tool.st_z_m[s])
        u = np.asarray(defl.u_st_T0_m[s, :3], float).copy()
        u[2] += float(st.misc[A.MS_FLOAT_U]) if tool.scal[A.TC_FLOAT_K] > 0.0 else 0.0
        rotv = np.asarray(defl.u_st_T0_m[s, 3:], float)
        vu = np.asarray(defl.v_st_T0_m_s[s, :3], float)
        vr = np.asarray(defl.v_st_T0_m_s[s, 3:], float)
        F_st = np.zeros(3)
        M_st = np.zeros(3)
        for kk in range(int(tool.st_ptr[s]), int(tool.st_ptr[s + 1])):
            e = int(tool.st_elem[kk])
            a = (theta + float(tool.alpha0_rad[e])) % TWO_PI
            frac, s_c, s_lo, s_hi = engaged_part(intervals, a, dal)
            row = int(tool.row[e])
            h_write = float(table.h_write_m[row])
            ws = int(tool.write_row[e])

            def commit_pending():
                pb = int(pend[e, A.PD_BIN])
                if pb >= 0:
                    if ws >= 0:
                        commit_ref(mem_pts, mem_tick, mem_valid, ws, pb, pend[e, 1:4], pend[e, A.PD_H], h_write,
                                   ratio, int(st.pend_tick[e]))
                    pend[e, A.PD_BIN] = -1.0

            if frac <= 0.0:
                # outside every interval: a pending write is committed (the element has left the bin stencil)
                commit_pending()
                continue
            dl = (s_c - 0.5) * dal
            th = theta + dl
            Rz = rot_z(th)
            G = R @ Rz
            p_T = np.asarray(tool.p_T_m[e], float)
            p_T0 = Rz @ p_T
            pr_T0 = p_T0 - np.array([0.0, 0.0, z_st])
            u_e = Rz @ (rot_z(-th) @ u) + np.cross(rotv, pr_T0)
            tip_c = tip + (s_c - 0.5) * dt * (v_tip - v_work) + u_work * 0.0
            p_S = tip_c - u_work + R @ (p_T0 + u_e)
            n_S = G @ np.asarray(tool.n_T[e], float)
            ac = (a + dl) % TWO_PI
            x = ac / width - 0.5
            b0 = int(math.floor(x))
            wgt = x - b0
            u_b = wgt
            b0 %= nb
            b1 = (b0 + 1) % nb

            def inside(angle):
                xx = angle % TWO_PI
                return any(lo <= xx <= hi or lo <= xx + TWO_PI <= hi for lo, hi in intervals)

            near = any(hi - lo < TWO_PI - 1e-12 and any(abs(ac + m * TWO_PI - edge) < width
                                                           for m in (-1, 0, 1) for edge in (lo, hi))
                       for lo, hi in intervals)
            if frac < 1.0 or near:
                in0, in1 = inside((b0 + 0.5) * width), inside((b0 + 1.5) * width)
                if in0 and not in1:
                    wgt = 0.0
                elif in1 and not in0:
                    wgt = 1.0
            rs = int(tool.read_row[e])
            if rs < 0:
                d0 = _field_depth(refill, s, b0, p_S, n_S)
                d1 = _field_depth(refill, s, b1, p_S, n_S)
                hf = (1.0 - wgt) * (d0 or 0.0) + wgt * (d1 or 0.0)
                M = p_S - hf * n_S
            else:
                M = (1.0 - wgt) * mem_pts[rs, b0] + wgt * mem_pts[rs, b1]
            h = float((p_S - M) @ n_S)
            if rs >= 0:
                # chord of the stencil between two points of the swept circle (second order in the bin width)
                h -= 0.5 * width * width * ((1.0 - wgt) * u_b ** 2 + wgt * (1.0 - u_b) ** 2) * \
                    float(tool.n_T[e, 0] * p_T[0] + tool.n_T[e, 1] * p_T[1])
            res.h[e] = h
            res.in_cut[e] = frac
            # velocity relative to the material (S)
            v_vib = R @ (vu + np.cross(vr, pr_T0)) - v_work
            v_e = (v_tip - v_work) + np.cross(omega * ez_S + w_frame, R @ p_T0) + R @ (vu + np.cross(vr, pr_T0))
            vn = float(v_e @ n_S)
            vn_vib = float(v_vib @ n_S)
            c_S = G @ np.asarray(tool.c_T[e], float)
            Vc = float(v_e @ c_S)
            slope = h - float(st.h_last_m[e]) if int(st.tick_last[e]) == tick0 else 0.0
            # delayed memory write at bin-centre crossings
            bps = dal / width
            if bps != 0.0:
                xs = x - (s_c - 0.5) * bps - 0.5 * bps
                xe = xs + bps
                crossings = range(int(math.floor(xs)) + 1, int(math.floor(xe)) + 1) if bps > 0 else \
                    range(int(math.ceil(xs)) - 1, int(math.ceil(xe)) - 1, -1)
                for c in crossings:
                    sx = (c - xs) / bps
                    if not (s_lo <= sx <= s_hi):
                        continue
                    hx = h + (sx - s_c) * slope
                    commit_pending()
                    bi = c % nb
                    # first order along the velocity, plus the rotation's second-order (centripetal) term
                    q_S = p_S + (sx - s_c) * dt * v_e - 0.5 * ((sx - s_c) * dal) ** 2 * (G @ np.array([p_T[0], p_T[1], 0.0]))
                    base = None
                    if ws >= 0 and rs != ws:
                        if rs >= 0 and mem_valid[rs, bi]:
                            base = mem_pts[rs, bi].copy()
                        elif rs < 0:
                            df = _field_depth(refill, s, bi, q_S, n_S)
                            base = None if df is None else q_S - df * n_S
                    if ws < 0:
                        pass
                    elif base is not None:
                        # helix: pass the surface read from the read row on, renewed by the recovery rule
                        if h_write <= 0.0 or ratio <= 0.0:
                            wr = 1.0 if hx >= h_write else 0.0
                        elif hx < h_write:
                            wr = 0.0
                        elif hx >= (1.0 + ratio) * h_write:
                            wr = 1.0
                        else:
                            wr = (hx - h_write) / (ratio * h_write)
                        pend[e, A.PD_BIN] = float(bi)
                        pend[e, 1:4] = base + wr * (q_S - base)
                        pend[e, A.PD_H] = 1e300
                    elif hx >= h_write:
                        pend[e, A.PD_BIN] = float(bi)
                        pend[e, 1:4] = q_S
                        pend[e, A.PD_H] = hx
            was_in = (int(st.flags[e]) & A.FL_IN_CUT) != 0
            if h <= 0.0:
                if was_in:
                    commit_pending()          # the element left the cut (h fell to zero)
                continue
            if s_hi < 1.0 and frac < 1.0 and s_lo <= 0.0:
                exits_now = True              # the arc ends inside the step: the element leaves the interval
            else:
                exits_now = False
            col = table_forces(table, row, h)
            db = float(tool.db_m[e])
            dS = float(tool.dS_m[e])
            ft0 = col[A.COL_FT] * db + col[A.COL_ET] * dS
            fr0 = col[A.COL_FR] * db + col[A.COL_ER] * dS
            fa0 = col[A.COL_FA] * db + col[A.COL_EA] * dS
            fft, ffr = float(tool.flank_N[e, 0]), float(tool.flank_N[e, 1])
            fpd = 0.0
            if was_in and Vc > 0.0 and tool.pd[e, 0] > 0.0:
                # process damping: the flank indents the surface the edge generated behind it (wave memory)
                u_rel = u_e - R.T @ u_work
                n_T0 = Rz @ np.asarray(tool.n_T[e], float)
                n_teeth = float(tool.scal[A.TC_N_TEETH])
                x_lim = min(pd_max, TWO_PI * float(tool.r_xy_m[e]) / (n_teeth if n_teeth >= 1.0 else 1.0))
                area = wave_area_ref(st.pd_ring, st.pd_aux, e, u_rel, n_T0, float(st.pd_aux[e, A.PDA_X]) + Vc * frac * dt,
                                     float(tool.pd[e, 1]), float(tool.pd[e, 2]), x_lim, pd_max / (A.PD_RING_N - 1))
                fpd = float(tool.pd[e, 0]) * area
            mu = float(tool.pd[e, 3])
            ft = ft0 + fft + mu * fpd
            fr = fr0 + ffr + fpd
            fa = fa0
            dirs_S = [G @ np.asarray(tool.dir_T[e, k], float) for k in range(3)]
            F = ft * dirs_S[0] + fr * dirs_S[1] + fa * dirs_S[2]
            dtw = frac * dt
            arm = R @ pr_T0
            F_T0 = R.T @ F
            F_st += F_T0 * dtw
            M_st += np.cross(pr_T0, F_T0) * dtw
            F_S_tot += F * dtw
            M_S_tot += np.cross(R @ p_T0, F) * dtw
            # energies
            va = float(v_e @ dirs_S[2])
            p_tab = ft0 * Vc + fr0 * vn - fa0 * va
            p_fl = fft * Vc + ffr * vn + mu * fpd * Vc
            ps = col[A.COL_PS]
            pr = col[A.COL_PR]
            dV = h * db * float(tool.r_xy_m[e]) * abs(dal) * frac
            if tool.forming[e]:
                res.ploughing_J += p_tab * dtw
                res.displaced_m3 += dV
            else:
                res.heat_shear_J += p_tab * dtw * ps
                res.heat_rake_J += p_tab * dtw * pr
                res.ploughing_J += p_tab * dtw * (1.0 - ps - pr)
                res.chip_volume_m3 += dV
            res.heat_flank_J += p_fl * dtw
            res.process_damping_J += fpd * vn * dtw
            rc = col[A.COL_RC]
            ea = res.elem_acc
            ea[e, A.ELEM_IN_CUT_S] += dtw
            ea[e, A.ELEM_RAKE_SLIDING_M] += rc * Vc * dtw
            ea[e, A.ELEM_FLANK_SLIDING_M] += Vc * dtw
            ea[e, A.ELEM_INT_SIGMA_V_RAKE] += col[A.COL_SIG_RAKE] * rc * Vc * dtw
            ea[e, A.ELEM_INT_SIGMA_V_FLANK] += float(tool.sig_flank_Pa[e]) * Vc * dtw
            ea[e, A.ELEM_INT_H_DT] += h * dtw
            ea[e, A.ELEM_H_MAX] = max(ea[e, A.ELEM_H_MAX], h)
            if not was_in:
                ea[e, A.ELEM_ENTRIES] += 1.0
            if exits_now:
                commit_pending()
            del arm
        res.station_T0[s, :3] = F_st / dt
        res.station_T0[s, 3:] = M_st / dt
        # deflection work: force impulses on the deflection rates (T0)
        res.deflection_J += float(F_st @ vu + M_st @ vr)
    F_T0 = R.T @ F_S_tot
    M_T0 = R.T @ M_S_tot
    res.wrench_T0[:3] = F_T0 / dt
    res.wrench_T0[3:] = M_T0 / dt
    res.deflection_J -= float(F_S_tot @ v_work)
    res.shaft_J = -omega * float(M_T0[2])
    res.feed_J = -float(F_S_tot @ v_tip) - float(w_frame @ M_S_tot)
    res.memory = mem_pts
    res.mem_tick = mem_tick
    res.pend = pend
    return res


__all__ = ["cut_step_ref", "engaged_part", "RefResult", "rot_z", "commit_ref"]
