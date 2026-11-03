"""Transforms: Euler conventions, control frames, project poses, quaternions, slerp, and the Numba kernels
against the plain NumPy functions."""

import itertools
import math

import numpy as np
import pytest

from cnctwin.geom import transform as tf

RNG = np.random.default_rng(20261103)
SEQUENCES = [a + b + c for a, b, c in itertools.product("xyz", repeat=3) if a != b and b != c]


def random_rotation():
    return tf.axis_angle(RNG.normal(size=3), RNG.uniform(-math.pi, math.pi))


def test_elementary_rotations_are_right_handed():
    assert np.allclose(tf.rot_z(math.pi / 2) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(tf.rot_x(math.pi / 2) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(tf.rot_y(math.pi / 2) @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(tf.axis_angle([0, 0, 2], 0.3), tf.rot_z(0.3))


@pytest.mark.parametrize("seq", SEQUENCES)
@pytest.mark.parametrize("intrinsic", [True, False])
def test_euler_round_trip_including_gimbal_lock(seq, intrinsic):
    for _ in range(50):
        ang = RNG.uniform(-math.pi, math.pi, 3)
        R = tf.euler(ang, seq, intrinsic)
        assert np.allclose(tf.euler(tf.euler_angles(R, seq, intrinsic), seq, intrinsic), R, atol=1e-13)
    for mid in (0.0, math.pi / 2, -math.pi / 2, math.pi):
        R = tf.euler([0.7, mid, -0.4], seq, intrinsic)
        back = tf.euler_angles(R, seq, intrinsic)
        assert np.allclose(tf.euler(back, seq, intrinsic), R, atol=1e-12)


def test_intrinsic_is_reversed_extrinsic():
    ang = [0.3, -0.5, 1.1]
    assert np.allclose(tf.euler(ang, "zxz", True), tf.rot_z(0.3) @ tf.rot_x(-0.5) @ tf.rot_z(1.1))
    assert np.allclose(tf.euler(ang, "xyz", False), tf.rot_z(1.1) @ tf.rot_y(-0.5) @ tf.rot_x(0.3))


def test_pose_dict_matches_simap_and_round_trips():
    from cnctwin.db.simap import pose_matrix

    for order in tf.ROTATION_ORDERS:
        a = RNG.uniform(-3, 3, 3)
        t = RNG.uniform(-1, 1, 3)
        M = tf.pose_from_dict({"translate_mm": list(t * 1e3), "rotate_deg": list(np.degrees(a)), "order": order})
        assert np.allclose(M, pose_matrix(t, a, order), atol=1e-15)
        assert np.allclose(tf.pose_from_dict(tf.pose_to_dict(M, order)), M, atol=1e-12)
    assert np.allclose(tf.pose_from_dict(None), np.eye(4))
    assert np.allclose(tf.pose_from_dict({"translate_m": [1, 2, 3]})[:3, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        tf.pose_from_dict({"translate_mm": [0, 0, 0], "rotation": [1, 2, 3]})
    with pytest.raises(ValueError):
        tf.pose_from_dict({"rotate_deg": [0, 0, 0], "order": "xxz"})


def test_pose_from_simap_posemap():
    from cnctwin.db.simap import SIMap

    view = SIMap({"p": {"translate_mm": [10, 0, 0], "rotate_deg": [0, 0, 90]}})["p"]
    M = tf.pose_from_dict(view)
    assert np.allclose(M @ [1, 0, 0, 1], [0.01, 1, 0, 1])


def test_inverse_compose_apply():
    T = tf.pose(random_rotation(), [0.1, -0.2, 0.3])
    assert np.allclose(tf.compose(T, tf.inverse(T)), np.eye(4), atol=1e-15)
    p = RNG.normal(size=(5, 3))
    assert np.allclose(tf.apply(tf.inverse(T), tf.apply(T, p)), p, atol=1e-14)
    S = np.diag([2.0, 2.0, 2.0, 1.0])
    assert not tf.is_rigid(S)
    assert np.allclose(tf.inverse(S) @ S, np.eye(4))
    n = tf.apply_normal(np.diag([2.0, 1.0, 1.0, 1.0]), [[1.0, 1.0, 0.0]])
    assert np.allclose(n, np.array([[0.5, 1.0, 0.0]]) / np.linalg.norm([0.5, 1.0, 0.0]))


def test_axis_angle_round_trip_near_zero_and_pi():
    for angle in (0.0, 1e-9, 1.0, math.pi - 1e-9, math.pi):
        axis = tf.unit(RNG.normal(size=3))
        R = tf.axis_angle(axis, angle)
        a, ang = tf.to_axis_angle(R)
        assert np.allclose(tf.axis_angle(a, ang), R, atol=1e-12)


def test_frames_from_vectors_and_points():
    R = tf.frame_from_xz([1, 0.1, 0], [0, 0, 1])
    assert np.allclose(R[:, 2], [0, 0, 1]) and abs(np.dot(R[:, 0], R[:, 2])) < 1e-15
    R = tf.frame_from_points([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
    assert np.allclose(R, np.eye(3))
    with pytest.raises(ValueError):
        tf.frame_from_points([0, 0, 0], [1, 0, 0], [2, 0, 0])
    R = tf.frame_from_z([0, 0, 1])
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) > 0


def test_fanuc_g68_2_euler_zxz():
    # I = 0, J = 90 deg, K = 0: the feature plane tilts about X; its Z becomes the old -Y
    T = tf.fanuc_g68_2([0.01, 0.02, 0.03], 0.0, math.pi / 2, 0.0)
    assert np.allclose(T[:3, 2], [0, -1, 0], atol=1e-15)
    assert np.allclose(T[:3, 3], [0.01, 0.02, 0.03])
    # I = 90, J = 30: first about Z, then about the new X (which is the old Y)
    T = tf.fanuc_g68_2([0, 0, 0], math.pi / 2, math.radians(30), 0.0)
    assert np.allclose(T[:3, :3], tf.rot_z(math.pi / 2) @ tf.rot_x(math.radians(30)))
    assert np.allclose(T[:3, 0], [0, 1, 0], atol=1e-15)


def test_fanuc_g68_2_other_forms():
    T = tf.fanuc_g68_2_rpy([0, 0, 0], 0.1, 0.2, 0.3, q=123)
    assert np.allclose(T[:3, :3], tf.rot_z(0.3) @ tf.rot_y(0.2) @ tf.rot_x(0.1))
    T = tf.fanuc_g68_2_points([1, 1, 0], [2, 1, 0], [1, 2, 0])
    assert np.allclose(T[:3, :3], np.eye(3)) and np.allclose(T[:3, 3], [1, 1, 0])
    T = tf.fanuc_g68_2_vectors([0, 0, 0], [1, 0, 0], [0, -1, 0])
    assert np.allclose(T[:3, 2], [0, -1, 0]) and np.allclose(T[:3, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        tf.fanuc_g68_2_rpy([0, 0, 0], 0, 0, 0, q=111)


def test_siemens_cycle800_modes():
    a, b, c = math.radians(30), math.radians(-20), math.radians(10)
    # _MODE 57: axis by axis X, Y, Z, each about the rotated axes
    T = tf.siemens_cycle800(57, a, b, c)
    assert np.allclose(T[:3, :3], tf.rot_x(a) @ tf.rot_y(b) @ tf.rot_z(c))
    # axis order Z, Y, X: bits 1-0 = 11, 3-2 = 10, 5-4 = 01 -> 0b011011 = 27
    T = tf.siemens_cycle800(27, a, b, c)
    assert np.allclose(T[:3, :3], tf.rot_z(a) @ tf.rot_y(b) @ tf.rot_x(c))
    # solid angle: Z then the new Y
    T = tf.siemens_cycle800(64, a, b, 0.0)
    assert np.allclose(T[:3, :3], tf.rot_z(a) @ tf.rot_y(b))
    # frame: TRANS(X0) * R * TRANS(X1)
    T = tf.siemens_cycle800(57, 0.0, math.pi / 2, 0.0, reference_m=[0.1, 0, 0], zero_after_m=[0, 0, 0.05])
    assert np.allclose(T[:3, 3], [0.1 + 0.05, 0, 0], atol=1e-15)
    with pytest.raises(ValueError):
        tf.siemens_cycle800(192, 0, 0, 0)


def _projection_angle(axis: str, n: np.ndarray) -> float:
    """Angle about ``axis`` of the line where the plane with normal n cuts the coordinate plane normal to it,
    measured as R_axis turns its reference axis (+y for x, +x for y)."""
    if axis == "x":
        w = np.cross([1.0, 0.0, 0.0], n)
        w = -w if w[1] < 0 else w
        return math.atan2(w[2], w[1])
    w = np.cross([0.0, 1.0, 0.0], n)
    w = -w if w[0] < 0 else w
    return math.atan2(-w[2], w[0])


@pytest.mark.parametrize("mode,first,second", [(128 + 0b1001, "x", "y"), (128 + 0b0110, "y", "x")])
def test_siemens_projection_mode_reproduces_both_projection_angles(mode, first, second):
    alpha, beta = math.radians(30), math.radians(20)
    T = tf.siemens_cycle800(mode, alpha, beta, math.radians(15))
    n = T[:3, 2]
    assert math.isclose(_projection_angle(first, n), alpha, abs_tol=1e-12)
    assert math.isclose(_projection_angle(second, n), beta, abs_tol=1e-12)
    with pytest.raises(ValueError):
        tf.projection_angles("x", 0.1, "z", 0.2)


def test_heidenhain_plane_functions():
    spa, spb, spc = math.radians(10), math.radians(20), math.radians(30)
    T = tf.heidenhain_plane_spatial(spa, spb, spc)
    assert np.allclose(T[:3, :3], tf.rot_z(spc) @ tf.rot_y(spb) @ tf.rot_x(spa))
    assert np.allclose(T[:3, 3], 0.0)
    assert np.allclose(tf.heidenhain_plane_euler(0.1, 0.2, 0.3)[:3, :3], tf.fanuc_g68_2([0, 0, 0], 0.1, 0.2, 0.3)[:3, :3])
    T = tf.heidenhain_plane_vector([1, 0, 0.2], [0, 0, 1])
    assert np.allclose(T[:3, 0], [1, 0, 0]) and np.allclose(T[:3, 2], [0, 0, 1])
    T = tf.heidenhain_plane_relative(tf.heidenhain_plane_spatial(spa, 0, 0), "b", spb)
    assert np.allclose(T[:3, :3], tf.rot_x(spa) @ tf.rot_y(spb))
    assert np.allclose(tf.heidenhain_plane_points([0, 0, 0], [0, 1, 0], [-1, 0, 0])[:3, :3], tf.rot_z(math.pi / 2))


def test_quaternions_and_slerp():
    for _ in range(200):
        R = random_rotation()
        q = tf.quat_from_matrix(R)
        assert q[0] >= 0.0 and math.isclose(np.linalg.norm(q), 1.0, rel_tol=1e-15)
        assert np.allclose(tf.matrix_from_quat(q), R, atol=1e-14)
    q0 = tf.quat_from_matrix(np.eye(3))
    q1 = tf.quat_from_matrix(tf.rot_z(1.2))
    mid = tf.quat_slerp(q0, q1, 0.25)
    assert np.allclose(tf.matrix_from_quat(mid), tf.rot_z(0.3), atol=1e-14)
    # shorter arc: q and -q are the same rotation
    assert np.allclose(tf.matrix_from_quat(tf.quat_slerp(q0, -q1, 0.5)), tf.rot_z(0.6), atol=1e-14)
    assert np.allclose(tf.quat_mul(q1, tf.quat_conj(q1)), [1, 0, 0, 0], atol=1e-15)


def test_vector_slerp_is_a_great_circle():
    a = np.array([0.0, 0.0, 1.0])
    b = tf.unit([1.0, 0.0, 1.0])
    for s in np.linspace(0, 1, 7):
        v = tf.vec_slerp(a, b, s)
        assert math.isclose(np.linalg.norm(v), 1.0, rel_tol=1e-15)
        assert math.isclose(math.atan2(v[0], v[2]), s * math.pi / 4, abs_tol=1e-14)
    near = tf.unit([1e-9, 0.0, 1.0])
    assert np.allclose(tf.vec_slerp(a, near, 0.5), tf.unit([0.5e-9, 0, 1]), atol=1e-15)
    flip = tf.vec_slerp(a, -a, 0.5)
    assert abs(np.dot(flip, a)) < 1e-15
    flip = tf.vec_slerp(a, -a, 0.5, pole=[0, 1, 0])
    assert np.allclose(flip, [0, 1, 0], atol=1e-15)


def test_pose_interp_is_slerp_plus_lerp():
    T0 = tf.pose(np.eye(3), [0, 0, 0])
    T1 = tf.pose(tf.rot_x(1.0), [1, 2, 3])
    T = tf.pose_interp(T0, T1, 0.5)
    assert np.allclose(T[:3, :3], tf.rot_x(0.5)) and np.allclose(T[:3, 3], [0.5, 1, 1.5])


def test_kernels_match_numpy_functions():
    T = np.ascontiguousarray(tf.pose(random_rotation(), RNG.normal(size=3)))
    P = np.ascontiguousarray(RNG.normal(size=(20, 3)))
    out = np.empty_like(P)
    tf.pose_apply_points_nb(T, P, out)
    assert np.allclose(out, tf.apply(T, P), atol=1e-15)
    tf.pose_apply_dirs_nb(T, P, out)
    assert np.allclose(out, tf.apply_dir(T, P), atol=1e-15)
    B = np.ascontiguousarray(tf.pose(random_rotation(), RNG.normal(size=3)))
    M = np.empty((4, 4))
    tf.pose_mul_nb(T, B, M)
    assert np.allclose(M, T @ B, atol=1e-15)
    tf.pose_inv_nb(T, M)
    assert np.allclose(M, tf.inverse(T), atol=1e-15)
    R = np.empty((3, 3))
    axis = tf.unit(RNG.normal(size=3))
    tf.axis_angle_nb(axis, 0.7, R)
    assert np.allclose(R, tf.axis_angle(axis, 0.7), atol=1e-15)
    q = np.empty(4)
    for _ in range(100):
        Rr = np.ascontiguousarray(random_rotation())
        tf.quat_from_matrix_nb(Rr, q)
        assert np.allclose(q, tf.quat_from_matrix(Rr), atol=1e-15)
        tf.matrix_from_quat_nb(q, R)
        assert np.allclose(R, tf.matrix_from_quat(q), atol=1e-15)
    q0 = tf.quat_from_matrix(random_rotation())
    q1 = tf.quat_from_matrix(random_rotation())
    for s in (0.0, 0.3, 1.0):
        tf.quat_slerp_nb(q0, q1, s, q)
        assert np.allclose(q, tf.quat_slerp(q0, q1, s), atol=1e-14)
    v = np.empty(3)
    for a, b in ((tf.unit(RNG.normal(size=3)), tf.unit(RNG.normal(size=3))), (np.array([0., 0, 1]), np.array([0., 0, -1])),
                 (np.array([0., 0, 1]), tf.unit([1e-10, 0, 1]))):
        for s in (0.0, 0.4, 1.0):
            tf.vec_slerp_nb(np.ascontiguousarray(a), np.ascontiguousarray(b), s, v)
            assert np.allclose(v, tf.vec_slerp(a, b, s), atol=1e-14)
    T1 = np.ascontiguousarray(tf.pose(random_rotation(), RNG.normal(size=3)))
    tf.pose_interp_nb(T, T1, 0.35, M)
    assert np.allclose(M, tf.pose_interp(T, T1, 0.35), atol=1e-14)


def test_each_kernel_has_one_signature():
    for kernel in tf.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
