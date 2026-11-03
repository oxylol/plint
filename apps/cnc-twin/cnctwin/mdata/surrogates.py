"""Analytic models the calibration fits when it does not re-run the simulation (docs/models/calibration.md
"Models"). Pure functions of SI arrays; each states its equation, source and validity.

* :func:`spindle_electrical_power` - P_el = P_shaft / eta + P_idle,max (|w| / w_max)^x
  + max(J w dw/dt, 0) / eta (the production energy model, palmgren-1959 for x).
* :func:`spindle_motor_output` - P_out = P_shaft + P_bearing,max (|w| / w_max)^x_b + J w dw/dt (what a
  load meter measures: the motor's output torque times speed; harris-kotzalas-2007-bearings).
* :func:`first_order_filter` - y' = (u - y) / tau, exact for piecewise-linear u (a meter's low-pass).
* :func:`milling_power_kienzle` / :func:`milling_power_mechanistic` - mean shaft power of a milling
  pass (kienzle-1952, martellotti-1941 h = fz sin phi, altintas-2012-mfg-automation ch. 2).
* :func:`following_error` - de/dt = (1 - K_ff) v_cmd - K_v e (proportional position loop with velocity
  feed-forward and an ideal velocity loop, ellis-2012-control-design), exact for piecewise-linear v.
* :func:`axis_force` - F = m a + m g_z + sign(v) (mu N + F_seal) + F_process (production energy
  model; armstrong-1994-friction for the Coulomb form).
* :func:`block_queue` - interpreter / look-ahead queue: block k is ready t_block after block k-1 was
  ready, never before block k - B has finished; it starts when ready and when block k-1 ended.
* :func:`spindle_growth` - d delta/dt = (G (|w| / w_max)^x - delta) / tau (first-order thermal growth,
  bryan-1990-thermal-errors, mayr-2012-thermal-issues), exact for piecewise-constant speed.
* :func:`taylor_life` - T = T_ref (C / (vc (f/f_ref)^a (ap/ap_ref)^b))^(1/n) (taylor-1907, extended).
* :func:`usui_flank_wear` - VB = VB0 + (cot alpha - tan gamma) int A sigma_n V_s exp(-B / theta) dt
  (usui-1984-tool-wear; the wear-land geometry as in yen-2004-tool-wear-fem).
* :func:`part_dimension` - d = d_sim + c_r (r - r_sim) + d_defl (1 / s_k - 1) (linear in runout and
  compliance).
"""

from __future__ import annotations

import math

import numpy as np

from ..units import STANDARD_GRAVITY_M_S2


# ----------------------------------------------------------------------------- spindle

def _rate(t_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values))
    return np.gradient(values, t_s)


def spindle_electrical_power(t_s, omega_rad_s, shaft_W, *, efficiency: float, idle_power_max_W: float,
                             omega_max_rad_s: float, exponent: float, inertia_kgm2: float = 0.0) -> np.ndarray:
    """Electrical input power of the spindle drive (W) at each sample."""
    t = np.asarray(t_s, dtype=np.float64)
    w = np.abs(np.asarray(omega_rad_s, dtype=np.float64))
    shaft = np.asarray(shaft_W, dtype=np.float64)
    idle = idle_power_max_W * np.power(w / omega_max_rad_s, exponent)
    accel = np.maximum(inertia_kgm2 * w * _rate(t, w), 0.0)
    return shaft / efficiency + idle + accel / efficiency


def spindle_motor_output(t_s, omega_rad_s, shaft_W, *, bearing_power_max_W: float, omega_max_rad_s: float,
                         exponent: float, inertia_kgm2: float = 0.0) -> np.ndarray:
    """Mechanical output power of the spindle motor (W): cutting shaft power, bearing friction, inertia."""
    t = np.asarray(t_s, dtype=np.float64)
    w = np.abs(np.asarray(omega_rad_s, dtype=np.float64))
    shaft = np.asarray(shaft_W, dtype=np.float64)
    bearing = bearing_power_max_W * np.power(w / omega_max_rad_s, exponent)
    return shaft + bearing + inertia_kgm2 * w * _rate(t, w)


def first_order_filter(t_s, u, tau_s: float) -> np.ndarray:
    """y with y' = (u - y) / tau, y(t0) = u(t0); exact for u linear between samples. tau <= 0: y = u."""
    t = np.asarray(t_s, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if tau_s <= 0.0 or len(u) < 2:
        return u.copy()
    y = np.empty_like(u)
    y[0] = u[0]
    for k in range(1, len(u)):
        dt = t[k] - t[k - 1]
        if dt <= 0:
            y[k] = y[k - 1]
            continue
        e = math.exp(-dt / tau_s)
        slope = (u[k] - u[k - 1]) / dt
        # exact response to u(t) = u[k-1] + slope (t - t[k-1]) over the interval
        y[k] = u[k] - slope * tau_s + (y[k - 1] - u[k - 1] + slope * tau_s) * e
    return y


# ----------------------------------------------------------------------------- milling power

def _engagement(ae_m: float, radius_m: float) -> float:
    """Swept immersion angle phi_e = arccos(1 - ae / R) (0 .. pi)."""
    ratio = min(max(ae_m / radius_m, 0.0), 2.0)
    return math.acos(1.0 - ratio)


def sin_power_integral(p: float, phi_e: float) -> float:
    """int_0^phi_e sin(phi)^p dphi (p > -1) through the regularised incomplete beta function."""
    from scipy.special import beta, betainc

    a, b = 0.5 * (p + 1.0), 0.5
    full = beta(a, b)                       # int_0^pi sin^p = B(a, 1/2)
    if phi_e <= 0.5 * math.pi:
        return 0.5 * full * float(betainc(a, b, math.sin(phi_e) ** 2))
    return full - 0.5 * full * float(betainc(a, b, math.sin(phi_e) ** 2))


def milling_power_kienzle(*, spindle_rev_s: float, teeth: int, radius_m: float, ap_m: float, fz_m: float, ae_m: float,
                          kc11_Pa: float, mc: float, h_ref_m: float, edge_N_per_m: float = 0.0) -> float:
    """Mean shaft power (W) of a pass: P = n z R ap [kc11 h_ref^mc fz^(1-mc) I(1 - mc) + Kte phi_e],
    I(p) = int sin^p over the swept immersion. Straight-flute equivalent (exact for the mean over a
    revolution at constant immersion), h = fz sin(phi) (error O(fz / 2R))."""
    phi_e = _engagement(ae_m, radius_m)
    shear = kc11_Pa * h_ref_m ** mc * fz_m ** (1.0 - mc) * sin_power_integral(1.0 - mc, phi_e)
    return spindle_rev_s * teeth * radius_m * ap_m * (shear + edge_N_per_m * phi_e)


def milling_power_mechanistic(*, spindle_rev_s: float, teeth: int, radius_m: float, ap_m: float, fz_m: float,
                              ae_m: float, Ktc_Pa: float, Kte_N_per_m: float) -> float:
    """Mean shaft power (W) with the linear edge-force model F_t = Ktc ap h + Kte ap:
    P = Ktc ae ap vf + Kte z n ap R phi_e (vf = z fz n)."""
    phi_e = _engagement(ae_m, radius_m)
    vf = teeth * fz_m * spindle_rev_s
    return Ktc_Pa * ae_m * ap_m * vf + Kte_N_per_m * teeth * spindle_rev_s * ap_m * radius_m * phi_e


# ----------------------------------------------------------------------------- feed axes

def following_error(t_s, v_cmd_m_s, *, kv_1_s: float, feedforward: float, e0_m: float = 0.0) -> np.ndarray:
    """Following error (m) of a proportional position loop with velocity feed-forward:
    e' = (1 - K_ff) v(t) - K_v e, integrated exactly for v linear between samples."""
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(v_cmd_m_s, dtype=np.float64)
    e = np.empty_like(v)
    if len(v) == 0:
        return e
    e[0] = e0_m
    g = 1.0 - feedforward
    for k in range(1, len(v)):
        dt = t[k] - t[k - 1]
        if dt <= 0:
            e[k] = e[k - 1]
            continue
        x = kv_1_s * dt
        decay = math.exp(-x)
        a0 = -math.expm1(-x) / kv_1_s                      # int_0^dt e^{-kv (dt - s)} ds
        a1 = (dt - a0) / (kv_1_s * dt)                      # int_0^dt e^{-kv (dt - s)} s / dt ds
        e[k] = e[k - 1] * decay + g * (v[k - 1] * a0 + (v[k] - v[k - 1]) * a1)
    return e


def axis_force(v_m_s, a_m_s2, *, mass_kg: float, lift_frac: float, friction_coeff: float, normal_load_N: float,
               seal_drag_N: float, process_N=None) -> np.ndarray:
    """Drive force (N) of a feed axis: inertia, gravity on the vertical component, Coulomb friction."""
    v = np.asarray(v_m_s, dtype=np.float64)
    a = np.asarray(a_m_s2, dtype=np.float64)
    process = 0.0 if process_N is None else np.asarray(process_N, dtype=np.float64)
    friction = np.sign(v) * (friction_coeff * normal_load_N + seal_drag_N)
    return mass_kg * a + mass_kg * STANDARD_GRAVITY_M_S2 * lift_frac + friction + process


# ----------------------------------------------------------------------------- controller

def block_queue(motion_s, *, block_s: float, lookahead: int, start_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) of each block: ready_k = max(ready_{k-1}, end_{k-B}) + t_block,
    start_k = max(end_{k-1}, ready_k), end_k = start_k + m_k (m_k the motion time)."""
    m = np.asarray(motion_s, dtype=np.float64)
    n = len(m)
    start = np.empty(n)
    end = np.empty(n)
    ready_prev = start_s - block_s
    end_prev = start_s
    for k in range(n):
        free = end[k - lookahead] if lookahead > 0 and k >= lookahead else -math.inf
        ready = max(ready_prev, free) + block_s
        start[k] = max(end_prev, ready)
        end[k] = start[k] + m[k]
        ready_prev, end_prev = ready, end[k]
    return start, end


# ----------------------------------------------------------------------------- thermal

def spindle_growth(t_s, omega_rad_s, *, growth_max_m: float, tau_s: float, omega_max_rad_s: float, exponent: float,
                   initial_m: float = 0.0) -> np.ndarray:
    """Axial growth (m) with first-order lag towards G (|w| / w_max)^x, the speed held over each interval."""
    t = np.asarray(t_s, dtype=np.float64)
    w = np.abs(np.asarray(omega_rad_s, dtype=np.float64))
    target = growth_max_m * np.power(np.minimum(w / omega_max_rad_s, 10.0), exponent)
    out = np.empty_like(t)
    if len(t) == 0:
        return out
    out[0] = initial_m
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        decay = math.exp(-dt / tau_s) if dt > 0 else 1.0
        out[k] = target[k - 1] + (out[k - 1] - target[k - 1]) * decay
    return out


# ----------------------------------------------------------------------------- tool life and wear

def taylor_life(vc_m_s, *, C_m_s: float, n: float, T_ref_s: float, feed_m=None, ap_m=None, a: float = 0.0,
                b: float = 0.0, feed_ref_m: float = 1.0, ap_ref_m: float = 1.0) -> np.ndarray:
    """Tool life (s): vc (T / T_ref)^n (f / f_ref)^a (ap / ap_ref)^b = C."""
    vc = np.asarray(vc_m_s, dtype=np.float64)
    load = vc.copy()
    if feed_m is not None and a:
        load = load * np.power(np.asarray(feed_m, dtype=np.float64) / feed_ref_m, a)
    if ap_m is not None and b:
        load = load * np.power(np.asarray(ap_m, dtype=np.float64) / ap_ref_m, b)
    return T_ref_s * np.power(C_m_s / load, 1.0 / n)


def usui_flank_wear(t_s, *, sigma_n_Pa, sliding_m_s, temp_K, A_per_Pa: float, B_K: float, clearance_rad: float,
                    rake_rad: float, VB0_m: float = 0.0) -> np.ndarray:
    """Flank wear land (m) at the times t_s: the conditions of sample k hold over [t_k, t_k+1)."""
    t = np.asarray(t_s, dtype=np.float64)
    rate = A_per_Pa * np.asarray(sigma_n_Pa, dtype=np.float64) * np.asarray(sliding_m_s, dtype=np.float64) \
        * np.exp(-B_K / np.asarray(temp_K, dtype=np.float64))
    geometry = 1.0 / math.tan(clearance_rad) - math.tan(rake_rad)
    out = np.empty_like(t)
    if len(t) == 0:
        return out
    out[0] = VB0_m
    dt = np.diff(t)
    out[1:] = VB0_m + geometry * np.cumsum(rate[:-1] * dt)
    return out


def part_dimension(simulated_m, *, runout_sensitivity, deflection_m, runout_m: float, runout_sim_m: float,
                   stiffness_factor: float) -> np.ndarray:
    """Feature dimensions (m): the simulated value moved by the runout change and the compliance change."""
    simulated = np.asarray(simulated_m, dtype=np.float64)
    c = np.asarray(runout_sensitivity, dtype=np.float64)
    d = np.asarray(deflection_m, dtype=np.float64)
    return simulated + c * (runout_m - runout_sim_m) + d * (1.0 / stiffness_factor - 1.0)


__all__ = ["spindle_electrical_power", "spindle_motor_output", "first_order_filter", "sin_power_integral",
           "milling_power_kienzle", "milling_power_mechanistic", "following_error", "axis_force", "block_queue",
           "spindle_growth", "taylor_life", "usui_flank_wear", "part_dimension"]
