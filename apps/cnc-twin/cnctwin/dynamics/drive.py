"""Time-domain verification harness of the dynamics kernels (docs/models/dynamics.md "Verification").

The simulator's hot loop composes the dynamics kernels with cutting's force kernel; this harness replaces the
cutting kernel with the classic regenerative milling model of the benchmarks (straight teeth, linear
coefficients, Altintas' frame: x feed, y normal, immersion phi clockwise from +y; Insperger & Stepan 2004)::

    h_j = f_z sin phi_j + (u_x - u_x(t - tau_j)) sin phi_j + (u_y - u_y(t - tau_j)) cos phi_j
    F_x = sum_j g_j (-F_t cos phi_j - F_r sin phi_j - F_pd sin phi_j),
    F_y = sum_j g_j ( F_t sin phi_j - F_r cos phi_j - F_pd cos phi_j),
    F_t = K_t a max(h_j, 0), F_r = K_r a max(h_j, 0),
    F_pd = K_sp a A(v_n / V)  (flank indentation, process_damping_element; v_n = v . n_j, V = R Omega)

and drives the real kernels per step: external_force -> modal_predict -> modal_deflection -> force at the
predicted state -> modal_correct. The structure is a DynSystem with one station whose tool_shape rows carry the
x, y shapes of the modes. The delayed displacement is interpolated linearly in the node history. Stability is
measured on the regenerative difference d_k = max over period k of |u(t) - u(t - T)|, which removes the periodic
forced response: its growth rate per period is log |mu_max| of the linearised system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, types

from .abi import ACCUM_T, MI_L0_TICKS, SYSTEM_T, core_accum
from .kernels import (external_force, modal_begin, modal_correct, modal_deflection, modal_free, modal_predict,
                      process_damping_element)
from .modal import Mode, build_system, work_modes
from .sld import TWO_PI

OPTS = dict(cache=True, nogil=True, error_model="numpy")
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
i8 = types.int64

# cut parameter vector (BenchCut)
BC_N = 0          # teeth
BC_KT = 1         # K_t (Pa)
BC_KR = 2         # K_r (Pa)
BC_A = 3          # depth (m)
BC_FZ = 4         # feed per tooth (m)
BC_PHI_ST = 5     # entry / exit immersion (rad)
BC_PHI_EX = 6
BC_OMEGA = 7      # spindle speed (rad/s, positive: clockwise in Altintas' frame)
BC_DT = 8         # step (s)
BC_KSP = 9        # process damping: K_sp (N/m3), land (m), tan(clearance), max interference length (m), radius (m)
BC_LAND = 10
BC_TANC = 11
BC_PDMAX = 12
BC_R = 13
BC_LINEAR = 14    # 1: keep the force linear (no h > 0 clamp), for onset measurements
N_BC = 16
NODE_PHASE_STEPS = 0.3819660112501051   # 2 - golden ratio: an offset of the node grid no rational angle shares
FLOOR_REL = 1e-10                        # regenerative difference below this share of its start is round-off
OMEGA_DT_MAX = 0.3                       # the L0 stride limit of the governing modes (db/models/dynamics-modes.yaml)


@njit(types.void(SYSTEM_T, F1, F1, F1, F2, F2, i8, i8, ACCUM_T), **OPTS)
def bench_run(sys, cut, phase0, delays_s, hist, out_u, n_steps, tick_step, acc):
    """n_steps L0 steps of the benchmark (module doc). phase0 [N]: tooth angles at t = 0; delays_s [N];
    hist [n_hist, 2]: displacement history ring (node n at row n % n_hist); out_u [n_steps, 4]: t, u_x, u_y, F_x."""
    N = int(cut[BC_N])
    Kt = cut[BC_KT]
    Kr = cut[BC_KR]
    a = cut[BC_A]
    fz = cut[BC_FZ]
    phi_st = cut[BC_PHI_ST]
    phi_ex = cut[BC_PHI_EX]
    Om = cut[BC_OMEGA]
    dt = cut[BC_DT]
    Ksp = cut[BC_KSP]
    linear = cut[BC_LINEAR] != 0.0
    n_hist = hist.shape[0]
    base = np.zeros(7)
    station = np.zeros((1, 6))
    fS = np.zeros(3)
    two_pi = 2.0 * math.pi
    for n in range(n_steps):
        t1 = (n + 1) * dt
        external_force(sys, 0.0, 0.0, 0.0, base)
        modal_predict(sys)
        modal_deflection(sys)
        ux = sys.defl.u_st_T0_m[0, 0]
        uy = sys.defl.u_st_T0_m[0, 1]
        vx = sys.defl.v_st_T0_m_s[0, 0]
        vy = sys.defl.v_st_T0_m_s[0, 1]
        Fx = 0.0
        Fy = 0.0
        for j in range(N):
            phi = Om * t1 + phase0[j]
            phi -= two_pi * math.floor(phi / two_pi)
            if phi < phi_st or phi > phi_ex:
                continue
            # delayed displacement by linear interpolation in the node history
            td = t1 - delays_s[j]
            if td <= 0.0:
                uxd = 0.0
                uyd = 0.0
            else:
                pos = td / dt
                k0 = int(math.floor(pos))
                fr = pos - k0
                r0 = k0 % n_hist
                r1 = (k0 + 1) % n_hist
                uxd = (1.0 - fr) * hist[r0, 0] + fr * hist[r1, 0]
                uyd = (1.0 - fr) * hist[r0, 1] + fr * hist[r1, 1]
            s = math.sin(phi)
            c = math.cos(phi)
            h = fz * s + (ux - uxd) * s + (uy - uyd) * c
            if h < 0.0 and not linear:
                h = 0.0
            Ft = Kt * a * h
            Fr = Kr * a * h
            Fpd = 0.0
            if Ksp > 0.0:
                vn = vx * s + vy * c
                Fpd = process_damping_element(Ksp, a, cut[BC_LAND], cut[BC_TANC], cut[BC_PDMAX], vn,
                                              Om * cut[BC_R], dt, acc)
            Fx += -Ft * c - Fr * s - Fpd * s
            Fy += Ft * s - Fr * c - Fpd * c
        station[0, 0] = Fx
        station[0, 1] = Fy
        fS[0] = Fx
        fS[1] = Fy
        tick = sys.ds.node_tick[0] + sys.dm.iscal[MI_L0_TICKS]
        modal_correct(sys, station, fS, tick, acc)
        # node displacement (corrected state) into the history
        qx = 0.0
        qy = 0.0
        for k in range(sys.state.x.shape[0]):
            qx += sys.model.tool_shape[k, 0, 0] * sys.state.x[k, 0]
            qy += sys.model.tool_shape[k, 0, 1] * sys.state.x[k, 0]
        r = (n + 1) % n_hist
        hist[r, 0] = qx
        hist[r, 1] = qy
        out_u[n, 0] = t1
        out_u[n, 1] = qx
        out_u[n, 2] = qy
        out_u[n, 3] = Fx


@njit(types.void(SYSTEM_T, i8, F2, F1, ACCUM_T), **OPTS)
def cost_contact(sys, n_steps, station, force_S, acc):
    """n_steps L0 steps of the per-step protocol with fixed wrenches (cost measurement; the force kernel excluded):
    modal_begin (external_force, modal_predict, modal_deflection), modal_correct."""
    base = np.zeros(7)
    l0 = sys.dm.iscal[MI_L0_TICKS]
    for n in range(n_steps):
        modal_begin(sys, 0.001 * n, -1000.0, 0.0, base)
        modal_correct(sys, station, force_S, sys.ds.node_tick[0] + l0, acc)
        sys.ds.samp_n[0] = 0


@njit(types.void(SYSTEM_T, i8, i8), **OPTS)
def cost_free(sys, n_ticks, ipo_ticks):
    """n_ticks free ticks (cost measurement)."""
    base = np.zeros(7)
    for k in range(n_ticks):
        th = -1.0 * k
        modal_free(sys, th, th - 1.0, -1000.0, -1000.0, base, base, sys.ds.node_tick[0] + ipo_ticks)


@dataclass
class BenchModes:
    """Modes of the benchmark structure in Altintas' frame: (freq_Hz, zeta, mass_kg, direction [x, y])."""

    modes: list

    def build(self, l0_ticks: int, tick_s: float, K_c: np.ndarray | None = None, capacity: int = 16):
        ms = []
        for i, (f, z, mass, d) in enumerate(self.modes):
            w = TWO_PI * float(f)
            dv = np.array([float(d[0]), float(d[1]), 0.0])
            dv = dv / np.linalg.norm(dv)
            st = np.zeros((1, 6))
            st[0, :3] = dv / math.sqrt(float(mass))
            ms.append(Mode(0, w, float(z), "bench", f"bench {i}", mass_kg=float(mass), stations=st,
                           at_unbalance=st[0].copy(), key=("bench", i)))
        _, grid = work_modes(None, 0, np.eye(3), None)
        sys, layout = build_system(ms, nst=1, ref_station=0, residual=None, grid=grid, point_S=np.zeros(3),
                                   l0_ticks=l0_ticks, ipo_ticks=l0_ticks, tick_s=tick_s, n_ticks=1,
                                   capacity=capacity, rotating=False, unbalance_T=None, K_c_T0=K_c,
                                   R_S_T0=np.eye(3))
        return sys, layout


@dataclass
class BenchResult:
    t_s: np.ndarray
    u_m: np.ndarray            # [n, 2]
    Fx_N: np.ndarray
    growth_per_period: float   # mean log ratio of successive regenerative differences (late half of the run)
    d_k: np.ndarray            # regenerative difference per period
    steps_per_period: int


def run_benchmark(modes: BenchModes, *, n_teeth: int, Kt_Pa: float, Kr_Pa: float, depth_m: float, fz_m: float,
                  phi_st_rad: float, phi_ex_rad: float, speed_rad_s: float, steps_per_tooth: int, periods: int,
                  pitch_deg=None, linear: bool = True, relax: bool = True, process_damping: tuple | None = None,
                  kick_m_s: float = 1e-3) -> BenchResult:
    """Run the benchmark at one speed and depth; the step is a whole fraction of the tooth period (uniform pitch)
    or of the revolution / (N steps_per_tooth) (variable pitch). ``relax``: give the corrector the mean cutting
    stiffness (ZOA directional matrix) as its estimate."""
    from .sld import classic_milling
    N = int(n_teeth)
    if pitch_deg is None:
        pitch = np.full(N, TWO_PI / N)
    else:
        pitch = np.radians(np.asarray(pitch_deg, float))
    T_rev = TWO_PI / abs(speed_rad_s)
    # the L0 stride rule of the simulator (architecture 5.1): 2 pi f_max dt <= OMEGA_DT_MAX as well
    f_max = max(float(f) for f, _, _, _ in modes.modes)
    steps_per_tooth = max(int(steps_per_tooth), int(math.ceil(TWO_PI * f_max * T_rev / N / OMEGA_DT_MAX)))
    dt = T_rev / (N * steps_per_tooth)
    tick_s = dt                     # one base tick per step: node ticks count steps
    K_c = None
    if relax:
        cm = classic_milling(N, Kt_Pa, Kr_Pa, phi_st_rad, phi_ex_rad, pitch_deg)
        K_c = depth_m * cm.mean()
    sys, _ = modes.build(1, tick_s, K_c)
    # node angles are offset by a fraction of a step so no node falls exactly on the entry or exit angle (a node
    # there would be in or out of the cut by rounding, a period-to-period jump of the forcing)
    phase0 = np.concatenate([[0.0], np.cumsum(pitch)[:-1]]) + NODE_PHASE_STEPS * abs(speed_rad_s) * dt
    delays = pitch / abs(speed_rad_s)          # tooth j regenerates the surface of tooth j + 1 (ahead by pitch_j)
    n_steps = int(periods) * int(steps_per_tooth) * (N if pitch_deg is not None else 1)
    hist = np.zeros((N * steps_per_tooth * 4 + 8, 2))
    out = np.zeros((n_steps, 4))
    cut = np.zeros(N_BC)
    cut[BC_N] = N
    cut[BC_KT] = Kt_Pa
    cut[BC_KR] = Kr_Pa
    cut[BC_A] = depth_m
    cut[BC_FZ] = fz_m
    cut[BC_PHI_ST] = phi_st_rad
    cut[BC_PHI_EX] = phi_ex_rad
    cut[BC_OMEGA] = abs(speed_rad_s)
    cut[BC_DT] = dt
    cut[BC_LINEAR] = 1.0 if linear else 0.0
    if process_damping is not None:
        K_sp, land, tan_c, max_len, radius = process_damping
        cut[BC_KSP] = K_sp
        cut[BC_LAND] = land
        cut[BC_TANC] = tan_c
        cut[BC_PDMAX] = max_len
        cut[BC_R] = radius
    # a velocity kick excites every mode
    for k in range(sys.state.x.shape[0]):
        sys.state.x[k, 1] = kick_m_s * math.sqrt(modes.modes[k][2])
    acc = core_accum()
    bench_run(sys, cut, phase0, delays, hist, out, n_steps, 1, acc)
    per = int(steps_per_tooth) * (N if pitch_deg is not None else 1)
    u = out[:, 1:3]
    lag = per if pitch_deg is None else int(round(T_rev / dt))
    diff = np.linalg.norm(u[lag:] - u[:-lag], axis=1) if len(u) > lag else np.zeros(0)
    n_per = len(diff) // per
    d_k = np.array([float(np.max(diff[i * per:(i + 1) * per])) for i in range(n_per)]) if n_per else np.zeros(0)
    return BenchResult(out[:, 0].copy(), u.copy(), out[:, 3].copy(), growth_rate(d_k), d_k, per)


def growth_rate(d_k: np.ndarray) -> float:
    """Mean log growth per period of the regenerative difference over the late half of the run (least squares
    slope); a run whose difference fell to the round-off floor (FLOOR_REL of its start) is stable at the rate it
    had until then."""
    d = np.asarray(d_k, float)
    if len(d) < 4:
        return 0.0
    floor = FLOOR_REL * float(np.max(d[:max(2, len(d) // 20)]))
    below = np.nonzero(d <= floor)[0]
    if len(below):
        k = int(below[0])
        d0 = float(d[0]) if d[0] > 0.0 else floor
        return (math.log(max(float(d[k]), 1e-300)) - math.log(d0)) / max(k, 1)
    late = d[len(d) // 2:]
    k = np.arange(len(late), dtype=float)
    slope = np.polyfit(k, np.log(late), 1)[0]
    return float(slope)


def onset_depth(modes: BenchModes, *, a_lo: float, a_hi: float, tol_rel: float, **kw) -> float:
    """Depth where the benchmark's regenerative difference stops decaying (bisection on the growth sign)."""
    lo, hi = a_lo, a_hi
    if run_benchmark(modes, depth_m=hi, **kw).growth_per_period <= 0.0:
        return math.inf
    while hi - lo > tol_rel * hi:
        mid = 0.5 * (lo + hi)
        if run_benchmark(modes, depth_m=mid, **kw).growth_per_period > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


__all__ = ["bench_run", "cost_contact", "cost_free", "BenchModes", "BenchResult", "run_benchmark", "onset_depth",
           "growth_rate", "N_BC"]
