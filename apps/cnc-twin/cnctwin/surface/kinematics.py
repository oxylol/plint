"""Tool motion at the surface events, from motion's PathFrame of the step (bus record ``motion``).

Per-tick arrays of a PathFrame are sampled at the base tick of each event by linear interpolation between
the L1 ticks (the frame's ``tick`` holds the end tick of every L1 tick): tool tip in S, its velocity
relative to the stock (finite differences of ``tip_S_m``), the tool axis in S, the spindle speed, and for
turning the cylindrical (rho, z) velocity of the tool tip about S z (S rotates with the work, so the
radial and axial components are the feed relative to the work). The tool axis in S is the end-of-step
``pose_S_T`` z column rotated per tick with ``axis_T_M`` through R_S_M = R_S_T R_M_T^T (exact for 3-axis
machines, first order within a step for moving rotary axes).
"""

from __future__ import annotations

import numpy as np

ISO_AXES = {"X": 0, "Y": 1, "Z": 2}      # ISO 841: + moves the tool relative to the work along +x_M, +y_M, +z_M


def frame_of(motion, slot: int, path: int | None = None):
    """The PathFrame of ``path``, else the first path whose tool slot is ``slot`` (None when none)."""
    paths = list(getattr(motion, "paths", []) or [])
    if path is not None and 0 <= path < len(paths):
        return paths[path]
    for frame in paths:
        if int(getattr(frame, "tool_slot", -1)) == int(slot):
            return frame
    return None


def _interp1(tq: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation, extrapolated linearly from the end intervals (an event in the first L1 tick of a
    step lies before the step's first tick end)."""
    out = np.interp(tq, t, y)
    if len(t) >= 2:
        lo = tq < t[0]
        if lo.any():
            out[lo] = y[0] + (tq[lo] - t[0]) * (y[1] - y[0]) / (t[1] - t[0])
        hi = tq > t[-1]
        if hi.any():
            out[hi] = y[-1] + (tq[hi] - t[-1]) * (y[-1] - y[-2]) / (t[-1] - t[-2])
    return out


def _interp_rows(t_query: np.ndarray, t: np.ndarray, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    if rows.ndim == 1:
        return _interp1(t_query, t, rows)
    return np.stack([_interp1(t_query, t, rows[:, c]) for c in range(rows.shape[1])], axis=1)


def rotation_S_M(frame) -> np.ndarray:
    R_ST = np.asarray(frame.pose_S_T, dtype=float)[:3, :3]
    R_MT = np.asarray(frame.pose_M_T, dtype=float)[:3, :3]
    return R_ST @ R_MT.T


def sample(frame, ticks: np.ndarray, tick_s: float) -> dict:
    """Motion quantities at the event ticks (see the module doc); empty dict without per-tick data."""
    ticks = np.asarray(ticks, dtype=float)
    t = np.asarray(getattr(frame, "tick", np.zeros(0)), dtype=float)
    tip = np.asarray(getattr(frame, "tip_S_m", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    if len(t) == 0 or len(tip) != len(t):
        return {}
    n = len(t)
    if n >= 2:
        dt = np.diff(t) * tick_s
        vel_mid = np.diff(tip, axis=0) / dt[:, None]
        t_mid = 0.5 * (t[1:] + t[:-1])
        vel = _interp_rows(ticks, t_mid, vel_mid)
        rho = np.hypot(tip[:, 0], tip[:, 1])
        v_rho = np.interp(ticks, t_mid, np.diff(rho) / dt) if len(t_mid) > 1 else np.full(len(ticks), np.diff(rho)[0] / dt[0])
        v_z = np.interp(ticks, t_mid, np.diff(tip[:, 2]) / dt) if len(t_mid) > 1 else np.full(len(ticks), np.diff(tip[:, 2])[0] / dt[0])
    else:
        feed = np.asarray(getattr(frame, "feed_vel_M_m_s", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
        R = rotation_S_M(frame)
        vel = np.tile((R @ feed[0]) if len(feed) else np.zeros(3), (len(ticks), 1))
        v_rho = np.zeros(len(ticks))
        v_z = vel[:, 2].copy()
    out = {"tip": _interp_rows(ticks, t, tip), "vel": vel, "v_rho": v_rho, "v_z": v_z}
    axis_M = np.asarray(getattr(frame, "axis_T_M", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    R = rotation_S_M(frame)
    if len(axis_M) == n:
        a = _interp_rows(ticks, t, axis_M @ R.T)
    else:
        a = np.tile(np.asarray(frame.pose_S_T, dtype=float)[:3, 2], (len(ticks), 1))
    norm = np.linalg.norm(a, axis=1)
    norm[norm == 0.0] = 1.0
    out["axis"] = a / norm[:, None]
    omega = np.asarray(getattr(frame, "spindle_omega_rad_s", np.zeros(0)), dtype=float)
    out["omega"] = np.interp(ticks, t, omega) if len(omega) == n else np.zeros(len(ticks))
    blocks = np.asarray(getattr(frame, "block_index", np.zeros(0)), dtype=np.int64)
    if len(blocks) == n:
        idx = np.clip(np.searchsorted(t, ticks, side="left"), 0, n - 1)
        out["block"] = blocks[idx]
    else:
        out["block"] = np.full(len(ticks), -1, np.int64)
    return out


def turning_tool_axes(frame) -> tuple[np.ndarray, np.ndarray]:
    """(x_T, z_T) of a non-rotating tool as (rho, z, theta) components about S z at the tool's angular
    position, from pose_W0_T (the non-rotating work frame W0 shares S z: architecture 5.4)."""
    pose = np.asarray(getattr(frame, "pose_W0_T", np.eye(4)), dtype=float)
    tip = pose[:3, 3]
    rho = float(np.hypot(tip[0], tip[1]))
    if rho > 0.0:
        rho_hat = np.array([tip[0] / rho, tip[1] / rho, 0.0])
    else:
        rho_hat = np.array([1.0, 0.0, 0.0])
    theta_hat = np.cross(np.array([0.0, 0.0, 1.0]), rho_hat)
    out = []
    for col in (0, 2):
        v = pose[:3, col]
        out.append(np.array([float(v @ rho_hat), float(v[2]), float(v @ theta_hat)]))
    return out[0], out[1]


def linear_axis_dirs_S(frame) -> dict:
    """Unit direction in S of the tool motion relative to the work per unit of each linear axis named X, Y
    or Z (ISO 841 axis convention), {name: (column, direction)}."""
    names = tuple(getattr(frame, "axis_names", ()) or ())
    R = rotation_S_M(frame)
    out = {}
    for col, name in enumerate(names):
        key = str(name).upper()
        if key in ISO_AXES:
            e = np.zeros(3)
            e[ISO_AXES[key]] = 1.0
            out[str(name)] = (col, R @ e)
    return out


__all__ = ["frame_of", "sample", "rotation_S_M", "turning_tool_axes", "linear_axis_dirs_S", "ISO_AXES"]
