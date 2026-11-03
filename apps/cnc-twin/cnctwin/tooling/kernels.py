"""Compiled kernels of the tooling library (docs/models/tooling.md "Kernels"), each with a
plain-Python reference (``*_ref``) that the tests compare against.

* :func:`edge_points_T0` (hot loop, L0): midpoints of all elements in T0 at a spindle angle given
  by its cosine and sine, by angle addition from the EdgeTrig tables, with the tool's static
  runout and tilt (architecture 4.1 "runout and tilt"; deflection is added by the caller).
* :func:`edge_frames_T0` (hot loop, L0): tangent, chip-thickness normal, cutting direction and
  rake-face normal of all elements rotated to the spindle angle.
* :func:`compose_effective` (L2, once per macro step and tool): nominal EdgeModel + EdgeState
  (wear, thermal growth, pull-out, slip) -> effective EdgeModel (architecture 4.1 composition).
* :func:`envelope_radius_at` / :func:`envelope_radius_profile` (removal, collision broad phase):
  radius of the revolved envelope at a height (union of all chains, or one chain).

Rules (architecture 5.5): explicit signatures, cache, nogil, error_model numpy, no fastmath.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, typeof, types

from ..core.abi import EdgeModel
from .abi import (N_STATE, STATE_FREE_LENGTH_M, STATE_HOLDER_GROWTH_M, STATE_LENGTH_GROWTH_M, STATE_PULLOUT_M,
                  STATE_ROTATING, STATE_SHANK_DIR_R, STATE_SHANK_DIR_Z, STATE_SLIP_RAD, EdgeFrameT, EdgeState, EdgeTrig,
                  Envelope)


def _f1(n=1):
    return np.zeros(n)


def _f2():
    return np.zeros((1, 1))


def example_edge() -> EdgeModel:
    """An EdgeModel with the dtypes and dimensions tooling builds (for kernel signatures)."""
    i2 = np.zeros((1, 1), np.int32)
    return EdgeModel(_f1(), _f2(), _f2(), _f2(), _f2(), _f2(), _f2(), _f2(), _f2(), _f2(), _f2(), i2.copy(),
                     i2.copy(), i2.copy(), np.zeros((1, 1), np.int8), _f1(3), _f1(2), _f1(), _f1(), np.array([-1.0]))


def example_envelope() -> Envelope:
    i1 = np.zeros(1, np.int32)
    return Envelope(i1.copy(), i1.copy(), np.zeros(1, np.int8), *(_f1() for _ in range(12)), chain=i1.copy())


def example_state() -> EdgeState:
    return EdgeState(_f2(), _f2(), _f2(), _f2(), np.zeros(N_STATE), np.zeros(1, np.int8))


EDGE_T = typeof(example_edge())
TRIG_T = typeof(EdgeTrig(_f2(), _f2()))
FRAME_T = typeof(EdgeFrameT(*(np.zeros((1, 1, 3)) for _ in range(4))))
STATE_T = typeof(example_state())
ENV_T = typeof(example_envelope())
F3D = types.float64[:, :, ::1]
F1D = types.float64[::1]


# --------------------------------------------------------------------------- element positions
@njit(types.void(EDGE_T, TRIG_T, types.float64, types.float64, F3D), cache=True, nogil=True, error_model="numpy")
def edge_points_T0(edge, trig, cos_t, sin_t, out):
    """out[j, k] = midpoint of element (j, k) in T0 at spindle angle theta (cos_t, sin_t)."""
    nt, ne = edge.elem_r_m.shape
    ax = edge.tilt_rad[0]
    ay = edge.tilt_rad[1]
    ox = edge.runout_T_m[0]
    oy = edge.runout_T_m[1]
    oz = edge.runout_T_m[2]
    for j in range(nt):
        for k in range(ne):
            r = edge.elem_r_m[j, k]
            z = edge.elem_z_m[j, k]
            xt = r * trig.cos_a[j, k]
            yt = r * trig.sin_a[j, k]
            # small tilt omega = (ax, ay, 0): p + omega x p, then the runout offset (both fixed in T)
            px = xt + ay * z + ox
            py = yt - ax * z + oy
            pz = z + ax * yt - ay * xt + oz
            out[j, k, 0] = cos_t * px - sin_t * py
            out[j, k, 1] = sin_t * px + cos_t * py
            out[j, k, 2] = pz


def edge_points_T0_ref(edge, trig, cos_t, sin_t, out) -> None:
    theta = math.atan2(sin_t, cos_t)
    h = float(edge.hand_sign[0])
    for j in range(edge.elem_r_m.shape[0]):
        for k in range(edge.elem_r_m.shape[1]):
            a = h * (edge.tooth_phase_rad[j] + edge.elem_lag_rad[j, k])
            r, z = edge.elem_r_m[j, k], edge.elem_z_m[j, k]
            p = np.array([r * math.cos(a), r * math.sin(a), z])
            omega = np.array([edge.tilt_rad[0], edge.tilt_rad[1], 0.0])
            p = p + np.cross(omega, p) + edge.runout_T_m
            rot = np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0],
                            [0.0, 0.0, 1.0]])
            out[j, k] = rot @ p


@njit(types.void(FRAME_T, types.float64, types.float64, F3D, F3D, F3D, F3D), cache=True, nogil=True,
      error_model="numpy")
def edge_frames_T0(frames, cos_t, sin_t, tangent, normal, cut_dir, rake_normal):
    """Rotate the element frames (T, spindle angle 0) to T0 at spindle angle theta."""
    nt = frames.tangent.shape[0]
    ne = frames.tangent.shape[1]
    for j in range(nt):
        for k in range(ne):
            for src, dst in ((frames.tangent, tangent), (frames.normal, normal), (frames.cut_dir, cut_dir),
                             (frames.rake_normal, rake_normal)):
                x = src[j, k, 0]
                y = src[j, k, 1]
                dst[j, k, 0] = cos_t * x - sin_t * y
                dst[j, k, 1] = sin_t * x + cos_t * y
                dst[j, k, 2] = src[j, k, 2]


def edge_frames_T0_ref(frames, cos_t, sin_t, tangent, normal, cut_dir, rake_normal) -> None:
    rot = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    for src, dst in ((frames.tangent, tangent), (frames.normal, normal), (frames.cut_dir, cut_dir),
                     (frames.rake_normal, rake_normal)):
        dst[...] = np.einsum("ab,jkb->jka", rot, src)


# --------------------------------------------------------------------------- composition
@njit(types.void(EDGE_T, STATE_T, EDGE_T), cache=True, nogil=True, error_model="numpy")
def compose_effective(nominal, state, out):
    """Effective edge (architecture 4.1): nominal + rigid shifts along -shank_dir (holder growth,
    pull-out) + tool thermal elongation (uniform strain over the free length) + radial thermal
    growth along e_r - wear (radial along -e_r, axial along +z) ; slip shifts the tooth phases
    against the cutting direction (rotating tools); the worn hone radius replaces the nominal
    where wear gives one; a broken tool has no valid element. ``out`` has nominal's shapes."""
    s = state.scalars
    g_len = s[STATE_LENGTH_GROWTH_M]
    rigid = s[STATE_HOLDER_GROWTH_M] + s[STATE_PULLOUT_M]
    free = s[STATE_FREE_LENGTH_M]
    sr = s[STATE_SHANK_DIR_R]
    sz = s[STATE_SHANK_DIR_Z]
    slip = s[STATE_SLIP_RAD] if s[STATE_ROTATING] > 0.5 else 0.0
    broken = state.broken[0] != 0
    nt, ne = nominal.elem_r_m.shape
    two_pi = 2.0 * math.pi
    for j in range(nt):
        phase = nominal.tooth_phase_rad[j] - slip
        out.tooth_phase_rad[j] = phase - two_pi * math.floor(phase / two_pi)
        for k in range(ne):
            r = nominal.elem_r_m[j, k]
            z = nominal.elem_z_m[j, k]
            d = r * sr + z * sz
            frac = (free - d) / free if free > 0.0 else 1.0
            if frac < 0.0:
                frac = 0.0
            if frac > 1.0:
                frac = 1.0
            shift = rigid + g_len * frac
            out.elem_r_m[j, k] = r - shift * sr + state.radial_growth_m[j, k] - state.radial_wear_m[j, k]
            out.elem_z_m[j, k] = z - shift * sz + state.axial_wear_m[j, k]
            worn = state.edge_radius_m[j, k]
            out.elem_edge_radius_m[j, k] = worn if worn > 0.0 else nominal.elem_edge_radius_m[j, k]
            out.elem_valid[j, k] = 0 if broken else nominal.elem_valid[j, k]
            out.elem_lag_rad[j, k] = nominal.elem_lag_rad[j, k]
            out.elem_len_m[j, k] = nominal.elem_len_m[j, k]
            out.elem_db_m[j, k] = nominal.elem_db_m[j, k]
            out.elem_kappa_rad[j, k] = nominal.elem_kappa_rad[j, k]
            out.elem_rake_rad[j, k] = nominal.elem_rake_rad[j, k]
            out.elem_incl_rad[j, k] = nominal.elem_incl_rad[j, k]
            out.elem_clear_rad[j, k] = nominal.elem_clear_rad[j, k]
            out.elem_kind[j, k] = nominal.elem_kind[j, k]
            out.elem_group[j, k] = nominal.elem_group[j, k]
            out.elem_station[j, k] = nominal.elem_station[j, k]
    for i in range(nominal.station_z_m.shape[0]):
        r = nominal.station_r_m[i]
        z = nominal.station_z_m[i]
        d = r * sr + z * sz
        frac = (free - d) / free if free > 0.0 else 1.0
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        shift = rigid + g_len * frac
        out.station_r_m[i] = r - shift * sr
        out.station_z_m[i] = z - shift * sz
    for i in range(3):
        out.runout_T_m[i] = nominal.runout_T_m[i]
    for i in range(2):
        out.tilt_rad[i] = nominal.tilt_rad[i]
    out.hand_sign[0] = nominal.hand_sign[0]


def compose_effective_ref(nominal: EdgeModel, state: EdgeState) -> EdgeModel:
    s = state.scalars
    sr, sz = s[STATE_SHANK_DIR_R], s[STATE_SHANK_DIR_Z]
    free = s[STATE_FREE_LENGTH_M]
    rigid = s[STATE_HOLDER_GROWTH_M] + s[STATE_PULLOUT_M]

    def shift_of(r, z):
        frac = np.clip((free - (r * sr + z * sz)) / free, 0.0, 1.0) if free > 0 else np.ones_like(r)
        return rigid + s[STATE_LENGTH_GROWTH_M] * frac

    shift = shift_of(nominal.elem_r_m, nominal.elem_z_m)
    st_shift = shift_of(nominal.station_r_m, nominal.station_z_m)
    slip = s[STATE_SLIP_RAD] if s[STATE_ROTATING] > 0.5 else 0.0
    phase = nominal.tooth_phase_rad - slip
    return nominal._replace(
        tooth_phase_rad=phase - 2.0 * math.pi * np.floor(phase / (2.0 * math.pi)),
        elem_r_m=nominal.elem_r_m - shift * sr + state.radial_growth_m - state.radial_wear_m,
        elem_z_m=nominal.elem_z_m - shift * sz + state.axial_wear_m,
        elem_edge_radius_m=np.where(state.edge_radius_m > 0.0, state.edge_radius_m, nominal.elem_edge_radius_m),
        elem_valid=np.zeros_like(nominal.elem_valid) if state.broken[0] else nominal.elem_valid.copy(),
        station_r_m=nominal.station_r_m - st_shift * sr, station_z_m=nominal.station_z_m - st_shift * sz)


# --------------------------------------------------------------------------- envelope queries
@njit(types.float64(ENV_T, types.float64, types.int64), cache=True, nogil=True, error_model="numpy")
def envelope_radius_at(env, z, chain):
    """Radius of the revolved envelope at height z (all chains when chain < 0); 0 outside."""
    best = 0.0
    for i in range(env.kind.shape[0]):
        if chain >= 0 and env.chain[i] != chain:
            continue
        if z < env.z_lo_m[i] - 1e-15 or z > env.z_hi_m[i] + 1e-15:
            continue
        if env.kind[i] == 1:
            c = (env.zc_m[i] - z) / env.rho_m[i]
            if c > 1.0:
                c = 1.0
            if c < -1.0:
                c = -1.0
            r = env.rc_m[i] + env.rho_m[i] * math.sqrt(1.0 - c * c)
        elif env.z1_m[i] == env.z0_m[i]:
            r = env.r0_m[i] if env.r0_m[i] > env.r1_m[i] else env.r1_m[i]
        else:
            t = (z - env.z0_m[i]) / (env.z1_m[i] - env.z0_m[i])
            r = env.r0_m[i] + t * (env.r1_m[i] - env.r0_m[i])
        if r > best:
            best = r
    return best


@njit(types.void(ENV_T, F1D, types.int64, F1D), cache=True, nogil=True, error_model="numpy")
def envelope_radius_profile(env, z, chain, out):
    for i in range(z.shape[0]):
        out[i] = envelope_radius_at(env, z[i], chain)


def envelope_radius_ref(env, z, chain=-1) -> np.ndarray:
    from .props import envelope_radius
    return envelope_radius(env, z, chain=chain)
