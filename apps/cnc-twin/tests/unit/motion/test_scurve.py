"""Jerk-limited profiles: the look-ahead generator against the analytic double-S, braking distance
against brute-force integration of the time-optimal jerk sequence."""

import numpy as np
import pytest

from cnctwin.motion.kernels import brake_distance, plan_ticks
from cnctwin.motion.scurve import double_s

DT = 1e-3


def generate(L, V, A, J, pieces=None, ready=None, stop=None, lookahead=50):
    Ls = np.array(pieces if pieces is not None else [L], float)
    n = len(Ls)
    vmax = np.full(n, float(V))
    am = np.full(n, float(A))
    jm = np.full(n, float(J))
    us = np.full(n, float(V))
    us[0] = 0.0
    rd = np.zeros(n, np.int64) if ready is None else np.asarray(ready, np.int64)
    se = np.zeros(n, np.int8) if stop is None else np.asarray(stop, np.int8)
    gi = np.zeros(1, np.int64)
    gf = np.zeros(3)
    N = 200000
    op = np.zeros(N, np.int32)
    os_ = np.zeros(N)
    ov = np.zeros(N)
    oa = np.zeros(N)
    ol = np.zeros(N, np.int8)
    oc = np.zeros(1, np.int32)
    status = plan_ticks(Ls, vmax, am, jm, us, rd, se, n, lookahead, gi, gf, 1000, 1000, DT, N, op, os_, ov, oa, ol, oc)
    c = int(oc[0])
    cum = np.concatenate([[0.0], np.cumsum(Ls)])
    s_abs = cum[op[:c]] + os_[:c]
    return status, c, s_abs, ov[:c], oa[:c]


CASES = [
    (0.100, 0.5, 5.0, 200.0),      # V and A reached
    (0.300, 0.2, 2.0, 50.0),       # V and A reached, long cruise
    (0.002, 0.5, 5.0, 200.0),      # neither V nor A reached
    (0.020, 0.5, 5.0, 200.0),      # A reached, V not
    (0.050, 0.05, 5.0, 20.0),      # V reached, A not (V J < A^2)
    (1e-5, 0.5, 5.0, 200.0),       # tiny move
]


@pytest.mark.parametrize("L,V,A,J", CASES)
def test_generator_reproduces_the_analytic_double_s(L, V, A, J):
    status, count, s, v, a = generate(L, V, A, J)
    prof = double_s(L, V, A, J)
    assert status == 2
    # time-optimal: done within one tick of the analytic duration (setpoints fall on ticks)
    assert count == pytest.approx(prof.duration / DT, abs=1.0)
    t = (np.arange(count) + 1) * DT
    s_ref = np.array([prof.position(x) for x in t])
    # position error bounded by the jerk over one tick (the discrete switching), and exact on ticks
    # where the analytic switching instants fall on tick boundaries
    assert np.max(np.abs(s - s_ref)) <= max(J * DT ** 3, 1e-12) * 2.0
    assert s[-1] == pytest.approx(L, abs=1e-12)
    assert np.max(v) <= V * (1 + 1e-9)
    assert np.max(np.abs(a)) <= A * (1 + 1e-9)
    assert np.all(np.abs(np.diff(a[:-1])) <= J * DT * (1 + 1e-6))


def test_double_s_conditions_match_biagiotti_melchiorri():
    p = double_s(0.1, 0.5, 5.0, 200.0)
    assert p.a_lim == pytest.approx(5.0)
    assert p.v_lim == pytest.approx(0.5)
    assert p.Tj == pytest.approx(5.0 / 200.0)
    assert p.Ta == pytest.approx(5.0 / 200.0 + 0.5 / 5.0)
    assert p.Tv == pytest.approx(0.1 / 0.5 - p.Ta)
    q = double_s(0.002, 0.5, 5.0, 200.0)       # D < 2 A^3 / J^2: A not reached
    assert q.Tj == pytest.approx((0.002 / 400.0) ** (1 / 3))
    assert q.position(q.duration) == pytest.approx(0.002)
    assert q.state(0.5 * q.duration)[1] == pytest.approx(q.v_lim)


def _brute_brake(v, a, u, A, J, h=2e-6):
    """Integrate the time-optimal braking sequence to (u, 0) numerically (fine steps, bisection on the
    switching to +J): the reference for brake_distance."""
    from cnctwin.motion.reference import brake_distance as ref
    # simulate jerk -J until a reaches a_p (from the closed form a_p), hold, then +J: here the sequence
    # is generated independently by a time-stepping controller with exact switching
    s = 0.0
    vv, aa = v, a
    for _ in range(10_000_000):
        if vv <= u + 1e-12 and aa >= -1e-12:
            break
        # ramp-up distance needed from (vv, aa) to a = 0
        nat = vv - aa * aa / (2 * J) if aa < 0 else vv + aa * aa / (2 * J)
        jerk = J if (aa < 0 and nat <= u) else -J
        a_new = aa + jerk * h
        if jerk < 0 and a_new < -A:
            jerk = (-A - aa) / h
            a_new = -A
        if jerk > 0 and a_new > 0:
            jerk = -aa / h
            a_new = 0.0
        s += vv * h + 0.5 * aa * h * h + jerk * h ** 3 / 6
        vv += aa * h + 0.5 * jerk * h * h
        aa = a_new
    return s, ref(v, a, u, A, J)


@pytest.mark.parametrize("v,a,u", [(0.4, 0.0, 0.0), (0.4, 2.0, 0.1), (0.3, -3.0, 0.05), (0.5, 4.0, 0.0),
                                   (0.2, -1.0, 0.15)])
def test_brake_distance_matches_brute_force(v, a, u):
    A, J = 5.0, 200.0
    brute, ref = _brute_brake(v, a, u, A, J)
    assert brake_distance(v, a, u, A, J) == pytest.approx(ref, rel=1e-12)
    assert brake_distance(v, a, u, A, J) == pytest.approx(brute, rel=2e-4, abs=1e-8)


def test_generator_respects_corner_bounds_and_stops_at_exact_stop():
    L = [0.01, 0.01, 0.01]
    n = 3
    vmax = np.array([0.3, 0.3, 0.3])
    am = np.full(n, 5.0)
    jm = np.full(n, 200.0)
    us = np.array([0.0, 0.05, 0.3])
    rd = np.zeros(n, np.int64)
    se = np.array([0, 1, 0], np.int8)
    gi = np.zeros(1, np.int64)
    gf = np.zeros(3)
    N = 2000
    bufs = [np.zeros(N, np.int32), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N, np.int8), np.zeros(1, np.int32)]
    status = plan_ticks(np.array(L), vmax, am, jm, us, rd, se, n, 10, gi, gf, 1000, 1000, DT, N, *bufs)
    op, os_, ov = bufs[0], bufs[1], bufs[2]
    c = int(bufs[5][0])
    assert status == 1 and gi[0] == 2          # stopped at the end of piece 1 (exact stop)
    # speed at the node between piece 0 and 1 is at most its bound (+ one tick of jerk)
    cross = np.nonzero(np.diff(op[:c]) > 0)[0]
    assert len(cross) == 1
    assert min(ov[cross[0]], ov[cross[0] + 1]) <= 0.05 + 200.0 * DT * DT
    assert ov[c - 1] == 0.0 and os_[c - 1] == pytest.approx(0.01, abs=1e-12)
