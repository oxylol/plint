"""Kernel layouts of the holding domain (docs/models/holding.md "Kernels"; architecture 5.5).

Everything a compiled holding kernel reads or writes is a NamedTuple of writable C-contiguous arrays
(or plain arrays and scalars), built at L2 in Python. Holding is an L2 domain: its kernels run once or a
few times per macro step (contact equilibrium, capacity search, slip integration), not in the hot loop.

Frames: contact sets are in H, the nominal workholding frame of the setup (W at setup, architecture 4:
the part frame W moves in H when the part slips: ``p_H = pose_delta p_W``); positions are relative to the
reference point O of the part (its centre of mass at setup). Wrenches are ``[F (3), M about O (3)]``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- contact kinds
KIND_FRICTION = 0       # unilateral contact with Coulomb friction (locators, jaws, clamps, plates, rollers, centres)
KIND_BONDED = 1         # bilateral bonded node with strength (adhesive films, tabs)
KIND_OFF = 2            # inactive (a failed bond node, a contact outside the part)

# ---------------------------------------------------------------- contact states written by the solver
ST_STICK = 0
ST_SLIP = 1
ST_OPEN = 2
ST_BONDED = 3
ST_OFF = 4

# ---------------------------------------------------------------- solver settings (cfg array)
CFG_TOL = 0             # relative residual tolerance
CFG_MAX_ITER = 1        # Newton iterations per load increment
CFG_N_INC = 2           # load increments from the reference configuration
CFG_MAX_DISP = 3        # translation of O beyond which the part is not held (m)
CFG_MAX_ROT = 4         # rotation beyond which the part is not held (rad)
CFG_K_REG = 5           # regularising stiffness on every translation (N/m); rotations k_reg * L_c^2
CFG_L_C = 6             # characteristic length of the part (m)
CFG_LAM_MAX = 7         # largest load factor searched (the margin cap)
CFG_LAM_TOL = 8         # relative tolerance of the capacity search
CFG_UTIL_EPS = 9        # friction utilisation above which a sticking contact counts as critical
N_CFG = 10

# ---------------------------------------------------------------- solver info array
INFO_STATUS = 0         # 0 held (converged), 1 not converged, 2 displacement beyond the limits
INFO_ITER = 1
INFO_RESIDUAL = 2       # scaled residual at the end
INFO_SCALE_N = 3        # force scale of the residual
N_INFO = 4

STATUS_HELD = 0
STATUS_NOT_CONVERGED = 1
STATUS_MOVES = 2

# ---------------------------------------------------------------- mechanism modes
MODE_FRICTION = 0       # the critical contacts slip: persistent, dissipative motion
MODE_SEPARATION = 1     # the critical contacts open: lift-off / tipping against gravity and clamps (reversible)

# ---------------------------------------------------------------- ratchet outputs
RP_DQ = 0               # net generalized displacement of the step (m)
RP_Q_MAX = 1            # largest displacement reached (m)
RP_WORK_J = 2           # work of the resistance over the motion (J)
RP_MOVING_S = 3         # time spent moving (s)
N_RP = 4

RT_DX1 = 0              # circumferential slip at the shank radius (m): slip angle = dx1 / r
RT_DX2 = 1              # axial walk out of the holder (m)
RT_WORK_J = 2           # friction work at the interface (J)
RT_MOVING_S = 3         # time spent slipping (s)
RT_PEAK_RATIO = 4       # largest |demand| / static capacity seen
N_RT = 5


class Contacts(NamedTuple):
    """The contact set of one part in one setup (frame H, positions relative to O)."""

    r_m: np.ndarray             # [n, 3] contact point relative to O
    n: np.ndarray               # [n, 3] unit normal: direction of the compressive force on the part
    t1: np.ndarray              # [n, 3] tangent (unit, normal to n)
    t2: np.ndarray              # [n, 3] tangent (unit, n x t1)
    kn_N_per_m: np.ndarray      # [n] normal stiffness (0: force-controlled contact)
    kt_N_per_m: np.ndarray      # [n, 2] tangential stiffness along t1, t2 (0: frictionless direction)
    mu: np.ndarray              # [n] friction coefficient (Coulomb, on the tangential vector)
    pre_N: np.ndarray           # [n, 3] force on the part at zero displacement (components along n, t1, t2)
    gap_m: np.ndarray           # [n] interference along n at zero displacement (debris under a locator)
    kind: np.ndarray            # [n] int32 KIND_*
    cap_N: np.ndarray           # [n, 2] bonded nodes: tensile and shear capacity (N)


class Wave(NamedTuple):
    """Per L1 tick of one macro step: mean and first tooth-harmonic amplitude of a demand (d components),
    and the tooth phase that times the harmonic: demand(t) = mean + amp cos(phase0 + rate (t - t_tick))."""

    mean: np.ndarray            # [n_ticks, d]
    amp: np.ndarray             # [n_ticks, d] (>= 0 means the peak sample lies above the mean)
    phase0_rad: np.ndarray      # [n_ticks] tooth phase at the start of the tick
    rate_rad_s: np.ndarray      # [n_ticks] tooth phase rate (teeth x |omega|; 0: steady demand)


def empty_contacts(n: int = 0) -> Contacts:
    return Contacts(np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n),
                    np.zeros((n, 2)), np.zeros(n), np.zeros((n, 3)), np.zeros(n), np.zeros(n, np.int32),
                    np.zeros((n, 2)))


def empty_wave(n_ticks: int, dims: int) -> Wave:
    return Wave(np.zeros((n_ticks, dims)), np.zeros((n_ticks, dims)), np.zeros(n_ticks), np.zeros(n_ticks))


def contiguous_contacts(cs: Contacts) -> Contacts:
    """Writable C-contiguous copies with the kernel dtypes."""
    out = []
    for name, value in zip(cs._fields, cs):
        dtype = np.int32 if name == "kind" else np.float64
        arr = np.ascontiguousarray(np.array(value, dtype=dtype, copy=True))
        out.append(arr)
    return Contacts(*out)


def cfg_array(tol: float, max_iter: int, n_inc: int, max_disp_m: float, max_rot_rad: float, k_reg_N_per_m: float,
              l_c_m: float, lam_max: float, lam_tol: float, util_eps: float) -> np.ndarray:
    cfg = np.zeros(N_CFG)
    cfg[CFG_TOL] = tol
    cfg[CFG_MAX_ITER] = max_iter
    cfg[CFG_N_INC] = n_inc
    cfg[CFG_MAX_DISP] = max_disp_m
    cfg[CFG_MAX_ROT] = max_rot_rad
    cfg[CFG_K_REG] = k_reg_N_per_m
    cfg[CFG_L_C] = l_c_m
    cfg[CFG_LAM_MAX] = lam_max
    cfg[CFG_LAM_TOL] = lam_tol
    cfg[CFG_UTIL_EPS] = util_eps
    return cfg


__all__ = ["Contacts", "Wave", "empty_contacts", "empty_wave", "contiguous_contacts", "cfg_array",
           "KIND_FRICTION", "KIND_BONDED", "KIND_OFF", "ST_STICK", "ST_SLIP", "ST_OPEN", "ST_BONDED", "ST_OFF",
           "STATUS_HELD", "STATUS_NOT_CONVERGED", "STATUS_MOVES", "MODE_FRICTION", "MODE_SEPARATION"]
