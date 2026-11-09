"""The per-macro-step checks of a held part (docs/models/holding.md section 5).

Loads per L1 tick of the step, as wrenches on the part about its centre of mass O in H: the cutting wrench
(``bus_prev.cutting`` ``wrench_work_W_mean`` and ``_peak`` of every contact on the workpiece, summed, moved
from W to H through the pose delta), the change of gravity since the clamped state (orientation and mass
removed), and inertia (``-m a`` from the part's acceleration in M for table-mounted parts; ``-I alpha``
about the spindle axis for parts in a chuck or collet).

For each contact group:

1. bonded nodes (adhesive, tabs) fail progressively at the worst tick (elliptic interaction of tension and
   shear against their strength; the worst node first, failures persistent);
2. every tick's peak load is solved for equilibrium; vacuum seal nodes that open break the seal;
3. the capacity factor lambda of the worst tick's load (``w0 + lambda w_peak``, bisection) is the margin;
   below 1 the part moves: the incipient motion m (tangent with the critical contacts released) normalised
   so that m^T M m = mass gives the generalized resistance ``Q_cap = m^T (w0 + lambda w_peak)`` and the
   demand ``Q(t) = m^T (w0 + w_mean) + m^T (w_peak - w_mean) cos(tooth phase)``;
4. the ratchet integrates the motion: friction modes (slip, rotation in the jaws, pull-out from a chuck)
   leave a persistent displacement (the pose delta), separation modes (lift-off, tipping against gravity and
   clamps) come back unless the part leaves its holding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import abi as A
from . import kernels as K
from .setup import Group

COVER_COS = 0.9995       # ticks within 1.8 deg of the worst tick's load direction ...
COVER_MARGIN = 0.05      # ... and not larger are held when the worst tick has 5 % reserve


@dataclass
class StepLoads:
    """Loads of one workpiece over one macro step (frame H, about the group's O)."""

    mean: np.ndarray                       # [n_ticks, 6] cutting + inertia (mean of each tick)
    peak: np.ndarray                       # [n_ticks, 6] cutting at the peak sample + inertia
    gravity_delta: np.ndarray              # [6] change of the weight wrench since the clamped state
    phase0_rad: np.ndarray                 # [n_ticks] tooth phase at each tick start
    rate_rad_s: np.ndarray                 # [n_ticks] tooth phase rate
    dt_tick_s: float = 1e-3
    cutting: bool = False                  # any cutting load in the step
    corners_H: np.ndarray | None = None    # [8, 3] corners of the part's bounding box (slip extents)
    # the cutting load's tooth-period pulse: |mean force| / |peak force| of each tick (1: steady or no cut), and
    # the speed the cut can push the part at its contact (m/s per tick: twice the tool's feed speed relative to a
    # table-mounted part - a part retreating faster than the tool advances plus its chip thinning per tooth loses
    # the cut -, the cutting speed at the contact of a part turning in a chuck - it cannot be turned backwards);
    # the contact point relative to O (None: no cut, no limit)
    cut_fraction: np.ndarray | None = None
    v_limit_m_s: np.ndarray | None = None
    contact_H: np.ndarray | None = None

    @property
    def n_ticks(self) -> int:
        return int(self.mean.shape[0])


@dataclass
class GroupResult:
    margin: float = math.inf
    held: bool = True
    mode: int = A.MODE_FRICTION
    mech: np.ndarray = field(default_factory=lambda: np.zeros(6))
    dq_m: float = 0.0
    q_max_m: float = 0.0
    work_J: float = 0.0
    in_J: float = 0.0                      # work the cutting and inertia loads did on the part's motion (ledger in)
    loss_J: float = 0.0                    # dissipated: friction work, inelastic landings (ledger out)
    mech_change_J: float = 0.0             # change of the motion's kinetic + potential energy (ledger out)
    moving_s: float = 0.0
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    failing_ticks: int = 0
    worst_tick: int = -1
    bond_failed: list = field(default_factory=list)     # node indices failed this step
    seal_open: bool = False
    forces_H: np.ndarray | None = None                  # node forces at the last tick's mean load
    states: np.ndarray | None = None
    q_cap_N: float = 0.0
    demand_peak_N: float = 0.0
    slip_extent_m: float = 0.0                          # largest corner displacement of the step's slip
    lift_extent_m: float = 0.0                          # largest corner displacement at the largest lift


def scaled(w: np.ndarray, l_c: float) -> float:
    return float(np.linalg.norm(w[:3]) + np.linalg.norm(w[3:]) / l_c)


def solve(group: Group, w: np.ndarray):
    n = group.n
    delta = np.zeros(6)
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    info = np.zeros(A.N_INFO)
    status = K.solve_equilibrium(group.contacts, np.ascontiguousarray(w, dtype=float), group.cfg, delta, f, st, info)
    return status, delta, f, st


def bond_utilisation(group: Group, f: np.ndarray) -> np.ndarray:
    """(tension / tensile capacity)^2 + (shear / shear capacity)^2, square-rooted, per bonded node (0 elsewhere)."""
    cs = group.contacts
    out = np.zeros(group.n)
    for i in np.flatnonzero(cs.kind == A.KIND_BONDED):
        fn = float(f[i] @ cs.n[i])
        ft = math.hypot(float(f[i] @ cs.t1[i]), float(f[i] @ cs.t2[i]))
        cap_t, cap_s = float(cs.cap_N[i, 0]), float(cs.cap_N[i, 1])
        t = max(-fn, 0.0) / cap_t if cap_t > 0.0 else (math.inf if fn < 0.0 else 0.0)
        s = ft / cap_s if cap_s > 0.0 else (math.inf if ft > 0.0 else 0.0)
        out[i] = math.hypot(t, s)
    return out


def fail_bonds(group: Group, w: np.ndarray) -> list:
    """Progressive failure of bonded nodes under the load w (worst node first); returns the failed nodes."""
    failed = []
    while np.any(group.contacts.kind == A.KIND_BONDED):
        status, _, f, _ = solve(group, w)
        if status != A.STATUS_HELD:
            break
        util = bond_utilisation(group, f)
        worst = int(np.argmax(util))
        if util[worst] <= 1.0:
            break
        group.contacts.kind[worst] = A.KIND_OFF
        failed.append(worst)
    return failed


def twist_extent(twist: np.ndarray, group: Group, corners_H: np.ndarray) -> float:
    """Largest displacement of the part's bounding-box corners for a small twist about O (m)."""
    r = corners_H - group.body.centroid_H
    d = twist[:3] + np.cross(twist[3:], r)
    return float(np.max(np.sqrt(np.einsum("ij,ij->i", d, d)))) if d.size else 0.0


def normalised_mechanism(m: np.ndarray, group: Group) -> np.ndarray:
    M = group.body.mass_matrix()
    gm = float(m @ M @ m)
    if gm <= 0.0 or group.body.mass_kg <= 0.0:
        return np.zeros(6)
    return m / math.sqrt(gm / group.body.mass_kg)


def covered_ticks(loads: StepLoads, l_c: float, k_star: int) -> np.ndarray:
    """Ticks whose peak load is nearly parallel to the worst tick's and not larger (scaled wrench)."""
    v = np.hstack([loads.peak[:, :3], loads.peak[:, 3:] / l_c])
    nv = np.sqrt(np.einsum("ij,ij->i", v, v))
    nb = nv[k_star]
    dots = v @ v[k_star]
    return (nv == 0.0) | ((nv <= nb) & (dots >= COVER_COS * nv * nb))


def _settle(group: Group, res: GroupResult, ratchet_state: np.ndarray) -> None:
    """A part held this step comes to rest: the motion's kinetic energy is dissipated in its contacts (and a lift
    that has not landed yet lands: its potential energy too), booked like the ratchet's own losses."""
    v = float(ratchet_state[1])
    if v != 0.0:
        ke = 0.5 * float(group.body.mass_kg) * v * v
        res.loss_J += ke
        res.mech_change_J -= ke
    ratchet_state[1] = 0.0
    if res.mode == A.MODE_SEPARATION:
        q = float(ratchet_state[0])
        if q > 0.0 and ratchet_state.shape[0] > 2:
            pe = float(ratchet_state[2]) * q          # the restoring force of the last lift (state[2])
            res.loss_J += pe
            res.mech_change_J -= pe
        ratchet_state[0] = 0.0


def evaluate_group(group: Group, loads: StepLoads, settings, ratchet_state: np.ndarray) -> GroupResult:
    """One macro step of a contact group (module doc). ``ratchet_state`` [q, v] is updated in place."""
    res = GroupResult()
    if group.n == 0 or not np.any(group.contacts.kind != A.KIND_OFF):
        res.held = False
        res.margin = 0.0
        return res
    w0 = group.w_ref + loads.gravity_delta
    l_c = group.body.l_c_m
    n_ticks = loads.n_ticks
    if n_ticks == 0:
        return res
    norms = np.sqrt(np.einsum("ij,ij->i", loads.peak[:, :3], loads.peak[:, :3])) + \
        np.sqrt(np.einsum("ij,ij->i", loads.peak[:, 3:], loads.peak[:, 3:])) / l_c
    k_star = int(np.argmax(norms))
    if np.any(group.bonded):
        res.bond_failed = fail_bonds(group, w0 + loads.peak[k_star])
    if not np.any(group.contacts.kind != A.KIND_OFF):
        res.held = False
        res.margin = 0.0
        return res
    n = group.n
    if scaled(loads.peak[k_star], l_c) <= 1e-12 * max(scaled(w0, l_c), 1.0):
        res.margin = settings.margin_cap
        _settle(group, res, ratchet_state)
        status, _, f, st = solve(group, w0 + loads.mean[-1])
        if status == A.STATUS_HELD:
            res.forces_H, res.states = f, st
        return res
    delta = np.zeros(6)
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    info = np.zeros(A.N_INFO)
    mech = np.zeros(6)
    mode = np.zeros(1)

    def capacity_at(k):
        lam_k = K.capacity(group.contacts, np.ascontiguousarray(w0), np.ascontiguousarray(loads.peak[k]), group.cfg,
                           delta, f, st, info, mech, mode)
        return float(lam_k), int(mode[0]), mech.copy()

    lam, res.mode, mech_k = capacity_at(k_star)
    failing = []
    covered = covered_ticks(loads, l_c, k_star) if lam >= 1.0 + COVER_MARGIN else np.zeros(n_ticks, bool)
    for k in range(n_ticks):
        if k == k_star:
            if lam < 1.0:
                failing.append(k)
            continue
        if covered[k]:
            continue
        status, _, _, st_k = solve(group, w0 + loads.peak[k])
        if status != A.STATUS_HELD:
            failing.append(k)
        elif np.any(group.seal & (st_k == A.ST_OPEN)):
            res.seal_open = True
    if lam >= 1.0:
        status, _, _, st_k = solve(group, w0 + loads.peak[k_star])
        if status == A.STATUS_HELD and np.any(group.seal & (st_k == A.ST_OPEN)):
            res.seal_open = True
    if failing and k_star not in failing:
        k_star = max(failing, key=lambda k: (norms[k], -k))
        lam, res.mode, mech_k = capacity_at(k_star)
    res.failing_ticks = len(failing)
    res.worst_tick = k_star
    res.margin = lam
    wv = loads.peak[k_star]
    if lam < 1.0:
        m_hat = normalised_mechanism(mech_k, group)
        if float(m_hat @ wv) < 0.0:
            m_hat = -m_hat
        res.mech = m_hat
        q_cap = float(m_hat @ (w0 + lam * wv))
        wave = A.empty_wave(n_ticks, 1)
        # the speed of the part's contact point per unit generalized speed (twist m_hat about O)
        u_c = float(np.linalg.norm(m_hat[:3] + np.cross(m_hat[3:], loads.contact_H))) \
            if loads.contact_H is not None else 0.0
        for k in range(n_ticks):
            wave.mean[k, 0] = float(m_hat @ (w0 + loads.mean[k]))
            wave.amp[k, 0] = float(m_hat @ (loads.peak[k] - loads.mean[k]))
            wave.phase0_rad[k] = loads.phase0_rad[k]
            wave.rate_rad_s[k] = loads.rate_rad_s[k]
            if loads.cut_fraction is not None:
                wave.alpha_rad[k, 0] = A.pulse_alpha(float(loads.cut_fraction[k]))
            if res.mode == A.MODE_FRICTION and loads.v_limit_m_s is not None and u_c > 0.0:
                wave.v_max[k, 0] = float(loads.v_limit_m_s[k]) / u_c
        q0, v0 = float(ratchet_state[0]), float(ratchet_state[1])
        out = np.zeros(A.N_RP)
        K.ratchet_part(wave, float(loads.dt_tick_s), q_cap, float(group.body.mass_kg), int(res.mode),
                       int(settings.sub_steps_per_tooth), int(settings.max_sub_steps), ratchet_state[:2], out)
        if ratchet_state.shape[0] > 2:
            ratchet_state[2] = q_cap
        res.q_cap_N = q_cap
        res.demand_peak_N = float(max(wave.mean[k, 0] + abs(wave.amp[k, 0]) for k in range(n_ticks)))
        res.dq_m = float(out[A.RP_DQ])
        res.q_max_m = float(out[A.RP_Q_MAX])
        res.work_J = float(out[A.RP_WORK_J])
        res.moving_s = float(out[A.RP_MOVING_S])
        mass = float(group.body.mass_kg)
        res.in_J = float(out[A.RP_IN_J])
        res.loss_J = float(out[A.RP_LOSS_J])
        res.mech_change_J = 0.5 * mass * (float(ratchet_state[1]) ** 2 - v0 * v0)
        if res.mode == A.MODE_SEPARATION:
            res.mech_change_J += q_cap * (float(ratchet_state[0]) - q0)
        res.held = False
        corners = loads.corners_H if loads.corners_H is not None else group.body.centroid_H[None, :]
        if res.mode == A.MODE_FRICTION:
            res.twist = m_hat * res.dq_m
            res.slip_extent_m = twist_extent(res.twist, group, corners)
        else:
            res.lift_extent_m = twist_extent(m_hat * res.q_max_m, group, corners)
    else:
        _settle(group, res, ratchet_state)
    if np.any(group.contacts.kind == A.KIND_BONDED):
        status, _, fb, _ = solve(group, w0 + loads.peak[k_star])
        if status == A.STATUS_HELD:
            util = float(np.max(bond_utilisation(group, fb)))
            if util > 0.0:
                res.margin = min(res.margin, 1.0 / util)
    status, _, fm, stm = solve(group, w0 + loads.mean[-1])
    if status == A.STATUS_HELD:
        res.forces_H, res.states = fm, stm
    return res


def _support_layout(group: Group, T_S_H: np.ndarray) -> dict:
    """The static part of a group's supports in S (cached on the group per placement)."""
    key = T_S_H.tobytes()
    cached = getattr(group, "_support_layout", None)
    if cached is not None and cached["key"] == key:
        return cached
    cs = group.contacts
    R = T_S_H[:3, :3]
    O = group.body.centroid_H
    slices = []
    for pi in range(len(group.patches)):
        idx = np.flatnonzero(group.node_patch == pi)
        if idx.size:
            slices.append((pi, idx))
    layout = {"key": key, "R": R.copy(), "slices": slices, "pos_S": (cs.r_m + O) @ R.T + T_S_H[:3, 3],
              "normal_S": cs.n @ R.T, "t1_S": cs.t1 @ R.T}
    group._support_layout = layout
    return layout


def supports(group: Group, forces_H: np.ndarray | None, states: np.ndarray | None, T_S_H: np.ndarray) -> list:
    """PartHolding.supports: one dict per contact patch with per-node arrays in S (docs/models/holding.md
    "Bus record"): positions, normals, tangents, current forces and clamped preloads on the part, stiffness,
    area shares and contact states (abi.ST_* codes)."""
    cs = group.contacts
    lay = _support_layout(group, T_S_H)
    R = lay["R"]
    pre = group.clamp_forces_H if group.clamp_forces_H.shape[0] == group.n else np.zeros((group.n, 3))
    pre_S = pre @ R.T
    forces_S = forces_H @ R.T if forces_H is not None else pre_S
    st = np.asarray(states, dtype=np.int8) if states is not None else np.full(group.n, -1, np.int8)
    st = np.where(cs.kind == A.KIND_OFF, np.int8(A.ST_OFF), st)
    out = []
    for pi, idx in lay["slices"]:
        patch = group.patches[pi]
        out.append({
            "id": patch.id, "role": patch.role, "law": patch.law, "device": patch.device, "group": group.name,
            "pos_S_m": lay["pos_S"][idx], "normal_S": lay["normal_S"][idx], "t1_S": lay["t1_S"][idx],
            "force_S_N": forces_S[idx], "preload_S_N": pre_S[idx], "stiffness_N_per_m": cs.kn_N_per_m[idx],
            "tangential_stiffness_N_per_m": cs.kt_N_per_m[idx], "friction": float(patch.mu),
            "area_m2": patch.area_m2 * patch.weight[group.node_local[idx]], "state": st[idx],
            "rotating": bool(patch.rotating)})
    return out


__all__ = ["StepLoads", "GroupResult", "evaluate_group", "supports", "solve", "fail_bonds", "bond_utilisation",
           "normalised_mechanism", "scaled", "covered_ticks", "twist_extent"]
