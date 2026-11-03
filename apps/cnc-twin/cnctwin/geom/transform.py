"""Rigid transforms: 4x4 poses, rotations, Euler conventions of the controls, quaternions.

Conventions (docs/models/geom.md "Transforms")
---------------------------------------------
* A pose is a 4x4 homogeneous matrix ``T = [[R, t], [0, 0, 0, 1]]`` acting on column vectors.
  ``pose_A_B`` maps coordinates given in frame B to frame A: ``p_A = R p_B + t``; it is the pose of
  frame B in frame A (the columns of R are B's axes in A, t is B's origin in A). Composition:
  ``pose_A_C = pose_A_B @ pose_B_C``.
* Elementary rotations are right-handed and active: ``rot_z(a)`` turns +x toward +y.
* An **intrinsic** sequence ``axes = "zxz"`` with angles ``(a1, a2, a3)`` (in sequence order) is
  ``R = R_z(a1) R_x(a2) R_z(a3)``: about z, then about the new x, then about the newest z. The
  **extrinsic** sequence (fixed axes) with the same angles is ``R = R_z(a3) R_x(a2) R_z(a1)``.
* Project and database poses ``{translate_mm, rotate_deg: [a, b, c], order: xyz}`` index the angles
  by axis (a about x, b about y, c about z) and apply them intrinsically in ``order``
  (``R = R(order[0]) R(order[1]) R(order[2])``), identical to ``cnctwin.db.simap.pose_matrix`` and
  three.js ``Euler(a, b, c, 'XYZ')`` (docs/formats.md "Value forms").
* Quaternions are ``(w, x, y, z)``, Hamilton product, unit length, ``w >= 0`` after normalisation
  (``quat_from_matrix``); slerp takes the shorter arc (Shoemake 1985).

The plain functions are NumPy; the ``*_nb`` kernels are the Numba versions of the ones used per
step (explicit signatures, docs/architecture.md 5.5) and are checked against the plain ones.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numba import njit, types

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
f8 = types.float64
OPTS = dict(cache=True, nogil=True, error_model="numpy")

ROTATION_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_EPS = float(np.finfo(np.float64).eps)
# Below this angle a slerp falls back to a normalised lerp: the lerp's angular error grows as
# angle^3 / 12 and the sine formula's rounding as eps / angle; they balance at eps^(1/4) (~1.2e-4 rad).
SLERP_SMALL_ANGLE_RAD = _EPS ** 0.25
# A rotation matrix is accepted as rigid when R^T R differs from I by less than this (rounding of a
# product of a few dozen rotations stays far below it; any scale or shear exceeds it).
RIGID_TOLERANCE = 1e3 * _EPS


# ============================================================================ elementary rotations

def rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(axis: str, angle_rad: float) -> np.ndarray:
    """Rotation about a coordinate axis named ``x``, ``y`` or ``z``."""
    if axis == "x":
        return rot_x(angle_rad)
    if axis == "y":
        return rot_y(angle_rad)
    if axis == "z":
        return rot_z(angle_rad)
    raise ValueError(f"rotation axis {axis!r} is not x, y or z")


def unit(v, what: str = "vector") -> np.ndarray:
    """``v / |v|``; raises for a zero or non-finite vector."""
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{what} {arr.tolist()} has no direction")
    return arr / norm


def perpendicular(v) -> np.ndarray:
    """A unit vector perpendicular to ``v``, chosen deterministically: ``v`` crossed with the coordinate
    axis it is least aligned with (lowest index on ties)."""
    a = unit(v)
    k = int(np.argmin(np.abs(a)))
    e = np.zeros(3)
    e[k] = 1.0
    return unit(np.cross(a, e))


def axis_angle(axis, angle_rad: float) -> np.ndarray:
    """Rotation by ``angle_rad`` about the unit vector ``axis`` (Rodrigues' formula)."""
    k = unit(axis, "rotation axis")
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(k, k)


def to_axis_angle(R) -> tuple[np.ndarray, float]:
    """``(axis, angle)`` of a rotation, angle in [0, pi]; robust near 0 and pi (via the quaternion).
    The identity returns the axis +z and angle 0."""
    q = quat_from_matrix(R)
    v = q[1:]
    s = float(np.linalg.norm(v))
    if s == 0.0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    angle = 2.0 * math.atan2(s, q[0])
    return v / s, angle


def euler(angles_rad, axes: str, intrinsic: bool = True) -> np.ndarray:
    """Rotation of an Euler or Tait-Bryan sequence, angles in sequence order (see the module doc)."""
    if len(axes) != 3 or any(a not in AXIS_INDEX for a in axes):
        raise ValueError(f"Euler sequence {axes!r} must be three of x, y, z")
    if axes[0] == axes[1] or axes[1] == axes[2]:
        raise ValueError(f"Euler sequence {axes!r} repeats an axis in consecutive rotations")
    a = [float(v) for v in angles_rad]
    if len(a) != 3:
        raise ValueError("an Euler sequence needs three angles")
    mats = [rot_axis(ax, ang) for ax, ang in zip(axes, a)]
    if intrinsic:
        return mats[0] @ mats[1] @ mats[2]
    return mats[2] @ mats[1] @ mats[0]


def euler_angles(R, axes: str, intrinsic: bool = True) -> np.ndarray:
    """Angles (sequence order) of ``R`` for an Euler/Tait-Bryan sequence; the inverse of :func:`euler`.

    The middle angle is in [0, pi] for proper Euler sequences (``zxz`` ...) and in [-pi/2, pi/2] for
    Tait-Bryan ones (``xyz`` ...). At a gimbal lock (middle angle at a pole) the third angle is set to 0
    and the first carries the whole rotation (a deterministic choice). Algorithm: the general
    reduction of Shoemake (Graphics Gems IV, 1994), written for both families.
    """
    if not intrinsic:
        return euler_angles(R, axes[::-1], True)[::-1].copy()
    M = np.asarray(R, dtype=np.float64)
    i, j, k = (AXIS_INDEX[c] for c in axes)
    proper = i == k
    if proper:
        k = 3 - i - j
    # parity: +1 when (i, j, k) is a cyclic permutation of (0, 1, 2)
    sign = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    if proper:
        # R = R_i(a) R_j(b) R_i(c)
        sy = math.hypot(M[i, j], M[i, k])
        b = math.atan2(sy, M[i, i])
        if sy > 16.0 * _EPS:
            a = math.atan2(M[j, i], -sign * M[k, i])
            c = math.atan2(M[i, j], sign * M[i, k])
        else:
            a = math.atan2(sign * M[k, j], M[j, j])
            c = 0.0
        return np.array([a, b, c])
    # Tait-Bryan: R = R_i(a) R_j(b) R_k(c)
    cy = math.hypot(M[i, i], M[i, j])
    b = math.atan2(sign * M[i, k], cy)
    if cy > 16.0 * _EPS:
        a = math.atan2(-sign * M[j, k], M[k, k])
        c = math.atan2(-sign * M[i, j], M[i, i])
    else:
        a = math.atan2(sign * M[k, j], M[j, j])
        c = 0.0
    return np.array([a, b, c])


# ============================================================================ poses

def pose(R=None, t=None) -> np.ndarray:
    """4x4 pose from a 3x3 rotation (default identity) and a translation (default zero)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def translation(t) -> np.ndarray:
    return pose(None, t)


def rotation(R) -> np.ndarray:
    return pose(R, None)


def is_rigid(T, tol: float = RIGID_TOLERANCE) -> bool:
    """True when the 3x3 part is a proper rotation (orthonormal, det +1) and the last row is 0 0 0 1."""
    M = np.asarray(T, dtype=np.float64)
    R = M[:3, :3] if M.shape == (4, 4) else M
    if M.shape == (4, 4) and not np.array_equal(M[3], [0.0, 0.0, 0.0, 1.0]):
        return False
    return bool(np.max(np.abs(R.T @ R - np.eye(3))) <= tol and np.linalg.det(R) > 0.0)


def inverse(T) -> np.ndarray:
    """Inverse of a pose: ``[R^T, -R^T t]`` for a rigid pose, the general inverse otherwise."""
    M = np.asarray(T, dtype=np.float64)
    if is_rigid(M):
        R = M[:3, :3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ M[:3, 3]
        return out
    return np.linalg.inv(M)


def compose(*poses) -> np.ndarray:
    """Product left to right: ``compose(A_B, B_C, C_D)`` is ``A_D``."""
    out = np.eye(4)
    for T in poses:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def apply(T, points) -> np.ndarray:
    """Transform points ``[..., 3]``."""
    M = np.asarray(T, dtype=np.float64)
    P = np.asarray(points, dtype=np.float64)
    return P @ M[:3, :3].T + M[:3, 3]


def apply_dir(T, vectors) -> np.ndarray:
    """Rotate direction vectors ``[..., 3]`` (no translation)."""
    M = np.asarray(T, dtype=np.float64)
    return np.asarray(vectors, dtype=np.float64) @ M[:3, :3].T


def apply_normal(T, normals) -> np.ndarray:
    """Transform surface normals with the inverse transpose of the 3x3 part and renormalise."""
    M = np.asarray(T, dtype=np.float64)[:3, :3]
    N = np.asarray(normals, dtype=np.float64) @ np.linalg.inv(M)
    norm = np.linalg.norm(N, axis=-1, keepdims=True)
    return np.divide(N, norm, out=np.zeros_like(N), where=norm > 0.0)


def orthonormalize(R) -> np.ndarray:
    """The proper rotation nearest to ``R`` in the Frobenius norm (polar decomposition by SVD)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)) or 1.0)])
    return U @ D @ Vt


def frame_from_z(z_axis, x_hint=None) -> np.ndarray:
    """Rotation whose z column is ``z_axis``; x is ``x_hint`` made perpendicular (Gram-Schmidt), or a
    deterministic perpendicular (:func:`perpendicular`) when no hint is given or it is parallel."""
    z = unit(z_axis, "z axis")
    x = None
    if x_hint is not None:
        h = np.asarray(x_hint, dtype=np.float64).reshape(3)
        h = h - np.dot(h, z) * z
        n = float(np.linalg.norm(h))
        if n > math.sqrt(_EPS) * max(1.0, float(np.linalg.norm(x_hint))):
            x = h / n
    if x is None:
        x = perpendicular(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def frame_from_xz(x_axis, z_axis) -> np.ndarray:
    """Rotation with z along ``z_axis`` exactly and x as close as possible to ``x_axis``."""
    z = unit(z_axis, "z axis")
    x = np.asarray(x_axis, dtype=np.float64).reshape(3)
    x = x - np.dot(x, z) * z
    n = float(np.linalg.norm(x))
    if n <= math.sqrt(_EPS) * max(1.0, float(np.linalg.norm(x_axis))):
        raise ValueError(f"x direction {np.asarray(x_axis).tolist()} is parallel to the z direction")
    x = x / n
    return np.column_stack([x, np.cross(z, x), z])


def frame_from_points(p1, p2, p3) -> np.ndarray:
    """Rotation of the frame with x along p1->p2 and p3 on the +y side of the x axis (z = x x (p3 - p1))."""
    a = np.asarray(p1, dtype=np.float64).reshape(3)
    x = unit(np.asarray(p2, dtype=np.float64) - a, "direction point 1 -> point 2")
    v = np.asarray(p3, dtype=np.float64) - a
    z = np.cross(x, v)
    if float(np.linalg.norm(z)) <= math.sqrt(_EPS) * max(1.0, float(np.linalg.norm(v))):
        raise ValueError("the three points are collinear")
    z = unit(z)
    return np.column_stack([x, np.cross(z, x), z])


# ---------------------------------------------------------------------------- project dict poses

def pose_matrix(translate_m, rotate_rad, order: str = "xyz") -> np.ndarray:
    """4x4 of a pose ``{translate, rotate (by axis), order}`` in SI (see the module doc)."""
    if order not in ROTATION_ORDERS:
        raise ValueError(f"rotation order {order!r} is not one of {', '.join(ROTATION_ORDERS)}")
    angles = {"x": float(rotate_rad[0]), "y": float(rotate_rad[1]), "z": float(rotate_rad[2])}
    R = np.eye(3)
    for axis in order:
        R = R @ rot_axis(axis, angles[axis])
    return pose(R, [float(v) for v in translate_m])


def _plain_list(value, n: int) -> list[float]:
    if value is None:
        return [0.0] * n
    if hasattr(value, "value") and not isinstance(value, (list, tuple, np.ndarray)):
        value = value.value
    out = [float(getattr(v, "value", v)) for v in value]
    if len(out) != n:
        raise ValueError(f"expected {n} numbers, got {len(out)}")
    return out


def pose_from_dict(spec) -> np.ndarray:
    """4x4 SI pose of a project/database pose.

    Accepts the file form ``{translate_mm, rotate_deg, order}`` (also ``translate_m``/``rotate_rad``,
    ``translate_in``), an SI view (``cnctwin.db.simap.PoseMap``: uses its ``matrix()``), a 4x4 array, or
    ``None`` (identity). Missing translate/rotate are zero; ``order`` defaults to ``xyz``.
    """
    if spec is None:
        return np.eye(4)
    if hasattr(spec, "matrix") and callable(spec.matrix):
        return np.asarray(spec.matrix(), dtype=np.float64)
    if isinstance(spec, np.ndarray) or (isinstance(spec, Sequence) and not isinstance(spec, str)):
        M = np.asarray(spec, dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"a pose array must be 4x4, not {M.shape}")
        return M.copy()
    if not isinstance(spec, Mapping):
        raise ValueError(f"cannot read a pose from {type(spec).__name__}")
    known = {"translate_mm": 1e-3, "translate_m": 1.0, "translate_in": 0.0254, "translate_um": 1e-6}
    rot_known = {"rotate_deg": math.pi / 180.0, "rotate_rad": 1.0}
    allowed = set(known) | set(rot_known) | {"order", "note", "notes", "src"}
    unknown = sorted(k for k in spec if k not in allowed)
    if unknown:
        raise ValueError(f"unknown pose keys {unknown}: write {{translate_mm: [x, y, z], rotate_deg: [a, b, c], "
                         f"order: xyz}}")
    t_keys = [k for k in known if k in spec]
    r_keys = [k for k in rot_known if k in spec]
    if len(t_keys) > 1 or len(r_keys) > 1:
        raise ValueError(f"a pose gives its translation or rotation twice ({t_keys + r_keys})")
    translate = [0.0, 0.0, 0.0]
    if t_keys:
        translate = [v * known[t_keys[0]] for v in _plain_list(spec[t_keys[0]], 3)]
    rotate = [0.0, 0.0, 0.0]
    if r_keys:
        rotate = [v * rot_known[r_keys[0]] for v in _plain_list(spec[r_keys[0]], 3)]
    order = spec.get("order") or "xyz"
    return pose_matrix(translate, rotate, str(order))


def pose_to_dict(T, order: str = "xyz") -> dict:
    """The file form ``{translate_mm, rotate_deg, order}`` of a rigid pose (inverse of pose_from_dict)."""
    M = np.asarray(T, dtype=np.float64)
    if not is_rigid(M, 1e-9):
        raise ValueError("only a rigid pose has a {translate_mm, rotate_deg, order} form")
    angles_seq = euler_angles(M[:3, :3], order, intrinsic=True)
    by_axis = [0.0, 0.0, 0.0]
    for axis, angle in zip(order, angles_seq):
        by_axis[AXIS_INDEX[axis]] = float(angle)
    return {"translate_mm": [float(v) * 1e3 for v in M[:3, 3]],
            "rotate_deg": [math.degrees(a) for a in by_axis], "order": order}


# ============================================================================ quaternions

def quat_from_matrix(R) -> np.ndarray:
    """Unit quaternion (w, x, y, z), w >= 0, of a rotation matrix (Shepperd 1978: the largest of the four
    squared components is taken from the diagonal, so the division is always well conditioned)."""
    M = np.asarray(R, dtype=np.float64)[:3, :3]
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    cand = np.array([tr, M[0, 0], M[1, 1], M[2, 2]])
    k = int(np.argmax(cand))
    if k == 0:
        w = 0.5 * math.sqrt(max(1.0 + tr, 0.0))
        f = 0.25 / w
        q = np.array([w, (M[2, 1] - M[1, 2]) * f, (M[0, 2] - M[2, 0]) * f, (M[1, 0] - M[0, 1]) * f])
    elif k == 1:
        x = 0.5 * math.sqrt(max(1.0 + 2.0 * M[0, 0] - tr, 0.0))
        f = 0.25 / x
        q = np.array([(M[2, 1] - M[1, 2]) * f, x, (M[0, 1] + M[1, 0]) * f, (M[0, 2] + M[2, 0]) * f])
    elif k == 2:
        y = 0.5 * math.sqrt(max(1.0 + 2.0 * M[1, 1] - tr, 0.0))
        f = 0.25 / y
        q = np.array([(M[0, 2] - M[2, 0]) * f, (M[0, 1] + M[1, 0]) * f, y, (M[1, 2] + M[2, 1]) * f])
    else:
        z = 0.5 * math.sqrt(max(1.0 + 2.0 * M[2, 2] - tr, 0.0))
        f = 0.25 / z
        q = np.array([(M[1, 0] - M[0, 1]) * f, (M[0, 2] + M[2, 0]) * f, (M[1, 2] + M[2, 1]) * f, z])
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def matrix_from_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _angle_between_units(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between unit vectors, accurate at 0 and pi (Kahan: 2 atan2(|a - b|, |a + b|))."""
    return 2.0 * math.atan2(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b)))


def quat_slerp(q0, q1, s: float) -> np.ndarray:
    """Spherical linear interpolation of unit quaternions along the shorter arc (Shoemake 1985)."""
    a = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    b = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    theta = _angle_between_units(a, b)
    if theta < SLERP_SMALL_ANGLE_RAD:
        out = (1.0 - s) * a + s * b
    else:
        st = math.sin(theta)
        out = (math.sin((1.0 - s) * theta) / st) * a + (math.sin(s * theta) / st) * b
    return out / np.linalg.norm(out)


def vec_slerp(a, b, s: float, pole=None) -> np.ndarray:
    """Great-circle interpolation of unit vectors (the tool-axis interpolation of 5-axis controls).

    For (nearly) opposite vectors the plane of the great circle is undefined: it is taken through
    ``pole`` (a vector not parallel to ``a``) when given, else through :func:`perpendicular` of ``a``.
    """
    u = unit(a, "start direction")
    v = unit(b, "end direction")
    theta = _angle_between_units(u, v)
    if theta < SLERP_SMALL_ANGLE_RAD:
        out = (1.0 - s) * u + s * v
        return out / np.linalg.norm(out)
    if math.pi - theta < SLERP_SMALL_ANGLE_RAD:
        w = perpendicular(u) if pole is None else unit(np.cross(u, np.cross(pole, u)), "pole")
        return math.cos(s * math.pi) * u + math.sin(s * math.pi) * w
    st = math.sin(theta)
    out = (math.sin((1.0 - s) * theta) / st) * u + (math.sin(s * theta) / st) * v
    return out / np.linalg.norm(out)


def pose_interp(T0, T1, s: float) -> np.ndarray:
    """Pose between two rigid poses: rotation by quaternion slerp, translation linear (never a matrix lerp)."""
    A = np.asarray(T0, dtype=np.float64)
    B = np.asarray(T1, dtype=np.float64)
    q = quat_slerp(quat_from_matrix(A[:3, :3]), quat_from_matrix(B[:3, :3]), s)
    return pose(matrix_from_quat(q), (1.0 - s) * A[:3, 3] + s * B[:3, 3])


# ============================================================================ control conventions
# Each returns the pose of the tilted (feature) frame in the frame the control's command refers to
# (the active workpiece coordinate system), SI: translations in metres, angles in radians.

def fanuc_g68_2(origin_m, i_rad: float, j_rad: float, k_rad: float) -> np.ndarray:
    """Fanuc G68.2 (P0, Euler angles) ``G68.2 X Y Z I J K``: the feature coordinate system has its origin
    at (X, Y, Z) and is turned by I about Z, then by J about the new X, then by K about the new Z
    (intrinsic Z-X'-Z'')."""
    return pose(euler([i_rad, j_rad, k_rad], "zxz", intrinsic=True), origin_m)


_FANUC_Q = {123: "xyz", 132: "xzy", 213: "yxz", 231: "yzx", 312: "zxy", 321: "zyx"}


def fanuc_g68_2_rpy(origin_m, i_rad: float, j_rad: float, k_rad: float, q: int = 123) -> np.ndarray:
    """Fanuc G68.2 P1 (roll-pitch-yaw) ``G68.2 P1 Q.. X Y Z I J K``: I about X, J about Y, K about Z of the
    workpiece coordinate system (fixed axes), applied in the order Q (123 = X, Y, Z)."""
    if q not in _FANUC_Q:
        raise ValueError(f"G68.2 P1 rotation order Q{q} is not one of {sorted(_FANUC_Q)}")
    by_axis = {"x": i_rad, "y": j_rad, "z": k_rad}
    seq = _FANUC_Q[q]
    return pose(euler([by_axis[a] for a in seq], seq, intrinsic=False), origin_m)


def fanuc_g68_2_points(p1_m, p2_m, p3_m) -> np.ndarray:
    """Fanuc G68.2 P2 (three points): origin at point 1, +X toward point 2, point 3 on the +Y side."""
    return pose(frame_from_points(p1_m, p2_m, p3_m), p1_m)


def fanuc_g68_2_vectors(origin_m, x_dir, z_dir) -> np.ndarray:
    """Fanuc G68.2 P3 (two vectors): Z along the second vector, X along the first made perpendicular."""
    return pose(frame_from_xz(x_dir, z_dir), origin_m)


_SIEMENS_AXES = {1: "x", 2: "y", 3: "z"}


def siemens_cycle800(mode: int, a_rad: float, b_rad: float, c_rad: float,
                     reference_m=(0.0, 0.0, 0.0), zero_after_m=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Siemens CYCLE800 swivel frame ``TRANS(_X0,_Y0,_Z0) * R * TRANS(_X1,_Y1,_Z1)``.

    ``mode`` is the cycle's ``_MODE`` bit field: bits 7-6 the swivel mode, bits 1-0 / 3-2 / 5-4 the
    first / second / third rotation axis (01 X, 10 Y, 11 Z). Modes:

    * 00 axis by axis: ``R = R_ax1(_A) R_ax2(_B) R_ax3(_C)``, each rotation about the axes already
      rotated by the previous ones (``_MODE = 57``: X, then Y, then Z);
    * 01 solid angle: ``R = R_z(_A) R_y(_B) R_z(_C)`` (azimuth about Z, inclination about the new Y,
      then a turn about the new tool axis);
    * 10 projection angle: ``_A`` and ``_B`` are the projections of the swivelled plane on the planes
      normal to axis 1 and axis 2 (bits 1-0, 3-2: X and Y in either order, the tool axis Z), ``_C`` a
      turn about the new Z (see :func:`projection_angles`).

    Mode 11 (rotary axes direct) gives machine axis values, not a frame: it needs the machine's
    kinematic chain and is refused here.
    """
    swivel = (int(mode) >> 6) & 3
    ax = [(int(mode) >> shift) & 3 for shift in (0, 2, 4)]
    if swivel == 3:
        raise ValueError("CYCLE800 rotary-axis direct mode (_MODE bits 7-6 = 11) is resolved by the machine "
                         "kinematics, not by a frame")
    if swivel == 0:
        if 0 in ax:
            raise ValueError(f"CYCLE800 _MODE={mode}: axis-by-axis mode needs three rotation axes (bits 5-0)")
        R = euler([a_rad, b_rad, c_rad], "".join(_SIEMENS_AXES[v] for v in ax), intrinsic=True)
    elif swivel == 1:
        R = euler([a_rad, b_rad, c_rad], "zyz", intrinsic=True)
    else:
        if ax[0] == 0 or ax[1] == 0 or ax[0] == ax[1]:
            raise ValueError(f"CYCLE800 _MODE={mode}: projection mode needs two different axes (bits 3-0)")
        R = projection_angles(_SIEMENS_AXES[ax[0]], a_rad, _SIEMENS_AXES[ax[1]], b_rad) @ rot_z(c_rad)
    return translation(reference_m) @ pose(R) @ translation(zero_after_m)


def projection_angles(axis1: str, angle1_rad: float, axis2: str, angle2_rad: float) -> np.ndarray:
    """Rotation of a working plane given by two projection angles (CYCLE800 projection mode).

    ``angle1`` is the angle, about ``axis1``, of the line where the tilted plane cuts the coordinate plane
    normal to ``axis1``; ``angle2`` likewise about ``axis2``. The tool axis is the remaining coordinate
    axis, which must be z (orders XY and YX). Built as the intrinsic rotation ``R_axis1(a) R_axis2(b)``:
    the second axis after the first rotation stays in the first coordinate plane, so ``a = angle1``
    exactly, and ``tan(angle2) = tan(b) / cos(a)`` gives ``b`` (derivation in docs/models/geom.md).
    """
    if axis1 not in AXIS_INDEX or axis2 not in AXIS_INDEX or {axis1, axis2} != {"x", "y"}:
        raise ValueError(f"projection angles about {axis1}{axis2}: the plane's normal must be the remaining "
                         f"axis z (orders XY or YX)")
    a = float(angle1_rad)
    b = math.atan(math.tan(float(angle2_rad)) * math.cos(a))
    return rot_axis(axis1, a) @ rot_axis(axis2, b)


def heidenhain_plane_spatial(spa_rad: float, spb_rad: float, spc_rad: float) -> np.ndarray:
    """Heidenhain ``PLANE SPATIAL SPA SPB SPC``: rotations about the non-tilted X, Y and Z axes, in the
    order A, B, C (fixed axes): ``R = R_z(SPC) R_y(SPB) R_x(SPA)``. PLANE does not move the datum."""
    return pose(euler([spa_rad, spb_rad, spc_rad], "xyz", intrinsic=False))


def heidenhain_plane_euler(eulpr_rad: float, eulnu_rad: float, eulrot_rad: float) -> np.ndarray:
    """Heidenhain ``PLANE EULER EULPR EULNU EULROT``: precession about Z, nutation about the new X,
    rotation about the new Z (intrinsic Z-X'-Z'')."""
    return pose(euler([eulpr_rad, eulnu_rad, eulrot_rad], "zxz", intrinsic=True))


def heidenhain_plane_vector(base, normal) -> np.ndarray:
    """Heidenhain ``PLANE VECTOR BX BY BZ NX NY NZ``: tilted Z along the normal vector, tilted X along the
    base vector corrected to be perpendicular to it."""
    return pose(frame_from_xz(base, normal))


def heidenhain_plane_points(p1, p2, p3) -> np.ndarray:
    """Heidenhain ``PLANE POINTS``: X along point 1 -> point 2, point 3 on the +Y side; the datum stays."""
    return pose(frame_from_points(p1, p2, p3))


def heidenhain_plane_relative(current, axis: str, angle_rad: float) -> np.ndarray:
    """Heidenhain ``PLANE RELATIVE SP<axis>``: an incremental turn about an axis of the active tilted frame."""
    T = np.asarray(current, dtype=np.float64)
    return T @ pose(rot_axis(axis.lower(), angle_rad))


# ============================================================================ Numba kernels

@njit(types.void(F2, F2, F2), **OPTS)
def pose_apply_points_nb(T, P, out):
    """out[i] = R P[i] + t."""
    for i in range(P.shape[0]):
        x = P[i, 0]
        y = P[i, 1]
        z = P[i, 2]
        for r in range(3):
            out[i, r] = T[r, 0] * x + T[r, 1] * y + T[r, 2] * z + T[r, 3]


@njit(types.void(F2, F2, F2), **OPTS)
def pose_apply_dirs_nb(T, V, out):
    """out[i] = R V[i]."""
    for i in range(V.shape[0]):
        x = V[i, 0]
        y = V[i, 1]
        z = V[i, 2]
        for r in range(3):
            out[i, r] = T[r, 0] * x + T[r, 1] * y + T[r, 2] * z


@njit(types.void(F2, F2, F2), **OPTS)
def pose_mul_nb(A, B, out):
    """out = A B for 4x4 poses (out may not alias A or B)."""
    for r in range(4):
        for c in range(4):
            out[r, c] = A[r, 0] * B[0, c] + A[r, 1] * B[1, c] + A[r, 2] * B[2, c] + A[r, 3] * B[3, c]


@njit(types.void(F2, F2), **OPTS)
def pose_inv_nb(T, out):
    """Rigid inverse [R^T, -R^T t] (the caller guarantees a rigid pose)."""
    for r in range(3):
        for c in range(3):
            out[r, c] = T[c, r]
    for r in range(3):
        out[r, 3] = -(T[0, r] * T[0, 3] + T[1, r] * T[1, 3] + T[2, r] * T[2, 3])
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(types.void(F1, f8, F2), **OPTS)
def axis_angle_nb(axis, angle, out):
    """Rodrigues rotation about a unit axis into out[3, 3]."""
    kx = axis[0]
    ky = axis[1]
    kz = axis[2]
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c
    out[0, 0] = c + kx * kx * v
    out[0, 1] = kx * ky * v - kz * s
    out[0, 2] = kx * kz * v + ky * s
    out[1, 0] = ky * kx * v + kz * s
    out[1, 1] = c + ky * ky * v
    out[1, 2] = ky * kz * v - kx * s
    out[2, 0] = kz * kx * v - ky * s
    out[2, 1] = kz * ky * v + kx * s
    out[2, 2] = c + kz * kz * v


@njit(types.void(F2, F1), **OPTS)
def quat_from_matrix_nb(M, q):
    """Shepperd's method (see quat_from_matrix); q = (w, x, y, z) with w >= 0."""
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    k = 0
    best = tr
    if M[0, 0] > best:
        best = M[0, 0]
        k = 1
    if M[1, 1] > best:
        best = M[1, 1]
        k = 2
    if M[2, 2] > best:
        k = 3
    if k == 0:
        w = 0.5 * math.sqrt(max(1.0 + tr, 0.0))
        f = 0.25 / w
        q[0] = w
        q[1] = (M[2, 1] - M[1, 2]) * f
        q[2] = (M[0, 2] - M[2, 0]) * f
        q[3] = (M[1, 0] - M[0, 1]) * f
    elif k == 1:
        x = 0.5 * math.sqrt(max(1.0 + 2.0 * M[0, 0] - tr, 0.0))
        f = 0.25 / x
        q[0] = (M[2, 1] - M[1, 2]) * f
        q[1] = x
        q[2] = (M[0, 1] + M[1, 0]) * f
        q[3] = (M[0, 2] + M[2, 0]) * f
    elif k == 2:
        y = 0.5 * math.sqrt(max(1.0 + 2.0 * M[1, 1] - tr, 0.0))
        f = 0.25 / y
        q[0] = (M[0, 2] - M[2, 0]) * f
        q[1] = (M[0, 1] + M[1, 0]) * f
        q[2] = y
        q[3] = (M[1, 2] + M[2, 1]) * f
    else:
        z = 0.5 * math.sqrt(max(1.0 + 2.0 * M[2, 2] - tr, 0.0))
        f = 0.25 / z
        q[0] = (M[1, 0] - M[0, 1]) * f
        q[1] = (M[0, 2] + M[2, 0]) * f
        q[2] = (M[1, 2] + M[2, 1]) * f
        q[3] = z
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    sgn = 1.0 if q[0] >= 0.0 else -1.0
    for i in range(4):
        q[i] = sgn * q[i] / n


@njit(types.void(F1, F2), **OPTS)
def matrix_from_quat_nb(q, M):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    w = q[0] / n
    x = q[1] / n
    y = q[2] / n
    z = q[3] / n
    M[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    M[0, 1] = 2.0 * (x * y - w * z)
    M[0, 2] = 2.0 * (x * z + w * y)
    M[1, 0] = 2.0 * (x * y + w * z)
    M[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    M[1, 2] = 2.0 * (y * z - w * x)
    M[2, 0] = 2.0 * (x * z - w * y)
    M[2, 1] = 2.0 * (y * z + w * x)
    M[2, 2] = 1.0 - 2.0 * (x * x + y * y)


@njit(types.void(F1, F1, f8, F1), **OPTS)
def quat_slerp_nb(q0, q1, s, out):
    """Shorter-arc slerp of unit quaternions (see quat_slerp)."""
    n0 = math.sqrt(q0[0] ** 2 + q0[1] ** 2 + q0[2] ** 2 + q0[3] ** 2)
    n1 = math.sqrt(q1[0] ** 2 + q1[1] ** 2 + q1[2] ** 2 + q1[3] ** 2)
    dot = 0.0
    for i in range(4):
        dot += q0[i] * q1[i]
    sgn = 1.0 if dot >= 0.0 else -1.0
    dm = 0.0
    dp = 0.0
    for i in range(4):
        a = q0[i] / n0
        b = sgn * q1[i] / n1
        dm += (a - b) * (a - b)
        dp += (a + b) * (a + b)
    theta = 2.0 * math.atan2(math.sqrt(dm), math.sqrt(dp))
    if theta < SLERP_SMALL_ANGLE_RAD:
        w0 = 1.0 - s
        w1 = s
    else:
        st = math.sin(theta)
        w0 = math.sin((1.0 - s) * theta) / st
        w1 = math.sin(s * theta) / st
    norm = 0.0
    for i in range(4):
        out[i] = w0 * q0[i] / n0 + w1 * sgn * q1[i] / n1
        norm += out[i] * out[i]
    norm = math.sqrt(norm)
    for i in range(4):
        out[i] /= norm


@njit(types.void(F1, F1, f8, F1), **OPTS)
def vec_slerp_nb(a, b, s, out):
    """Great-circle interpolation of unit vectors; antipodal ends use the deterministic perpendicular."""
    na = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    nb = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
    ux = a[0] / na
    uy = a[1] / na
    uz = a[2] / na
    vx = b[0] / nb
    vy = b[1] / nb
    vz = b[2] / nb
    dm = math.sqrt((ux - vx) ** 2 + (uy - vy) ** 2 + (uz - vz) ** 2)
    dp = math.sqrt((ux + vx) ** 2 + (uy + vy) ** 2 + (uz + vz) ** 2)
    theta = 2.0 * math.atan2(dm, dp)
    if theta < SLERP_SMALL_ANGLE_RAD:
        ox = (1.0 - s) * ux + s * vx
        oy = (1.0 - s) * uy + s * vy
        oz = (1.0 - s) * uz + s * vz
    elif math.pi - theta < SLERP_SMALL_ANGLE_RAD:
        # w = u x e_k, e_k the axis u is least aligned with (lowest index on ties)
        ax = abs(ux)
        ay = abs(uy)
        az = abs(uz)
        if ax <= ay and ax <= az:
            wx, wy, wz = 0.0, uz, -uy
        elif ay <= az:
            wx, wy, wz = -uz, 0.0, ux
        else:
            wx, wy, wz = uy, -ux, 0.0
        wn = math.sqrt(wx * wx + wy * wy + wz * wz)
        c = math.cos(s * math.pi)
        sn = math.sin(s * math.pi)
        ox = c * ux + sn * wx / wn
        oy = c * uy + sn * wy / wn
        oz = c * uz + sn * wz / wn
    else:
        st = math.sin(theta)
        w0 = math.sin((1.0 - s) * theta) / st
        w1 = math.sin(s * theta) / st
        ox = w0 * ux + w1 * vx
        oy = w0 * uy + w1 * vy
        oz = w0 * uz + w1 * vz
    n = math.sqrt(ox * ox + oy * oy + oz * oz)
    out[0] = ox / n
    out[1] = oy / n
    out[2] = oz / n


@njit(types.void(F2, F2, f8, F2), **OPTS)
def pose_interp_nb(T0, T1, s, out):
    """Rigid pose between T0 (s = 0) and T1 (s = 1): quaternion slerp + linear translation."""
    q0 = np.empty(4)
    q1 = np.empty(4)
    q = np.empty(4)
    R0 = np.empty((3, 3))
    R1 = np.empty((3, 3))
    for r in range(3):
        for c in range(3):
            R0[r, c] = T0[r, c]
            R1[r, c] = T1[r, c]
    quat_from_matrix_nb(R0, q0)
    quat_from_matrix_nb(R1, q1)
    quat_slerp_nb(q0, q1, s, q)
    R = np.empty((3, 3))
    matrix_from_quat_nb(q, R)
    for r in range(3):
        for c in range(3):
            out[r, c] = R[r, c]
        out[r, 3] = (1.0 - s) * T0[r, 3] + s * T1[r, 3]
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


KERNELS = (pose_apply_points_nb, pose_apply_dirs_nb, pose_mul_nb, pose_inv_nb, axis_angle_nb, quat_from_matrix_nb,
           matrix_from_quat_nb, quat_slerp_nb, vec_slerp_nb, pose_interp_nb)

__all__ = ["rot_x", "rot_y", "rot_z", "rot_axis", "unit", "perpendicular", "axis_angle", "to_axis_angle", "euler",
           "euler_angles", "pose", "translation", "rotation", "is_rigid", "inverse", "compose", "apply", "apply_dir",
           "apply_normal", "orthonormalize", "frame_from_z", "frame_from_xz", "frame_from_points", "pose_matrix",
           "pose_from_dict", "pose_to_dict", "quat_from_matrix", "matrix_from_quat", "quat_mul", "quat_conj",
           "quat_slerp", "vec_slerp", "pose_interp", "fanuc_g68_2", "fanuc_g68_2_rpy", "fanuc_g68_2_points",
           "fanuc_g68_2_vectors", "siemens_cycle800", "projection_angles", "heidenhain_plane_spatial",
           "heidenhain_plane_euler", "heidenhain_plane_vector", "heidenhain_plane_points",
           "heidenhain_plane_relative", "ROTATION_ORDERS", "SLERP_SMALL_ANGLE_RAD", "KERNELS",
           "pose_apply_points_nb", "pose_apply_dirs_nb", "pose_mul_nb", "pose_inv_nb", "axis_angle_nb",
           "quat_from_matrix_nb", "matrix_from_quat_nb", "quat_slerp_nb", "vec_slerp_nb", "pose_interp_nb"]
