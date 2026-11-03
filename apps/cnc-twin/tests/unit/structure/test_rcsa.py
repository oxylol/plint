"""Receptance coupling: two free-free beams joined rigidly by RCSA against the monolithic FE, an elastic joint,
and a modal base against the assembled (reduced) system."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.structure import beam as B
from cnctwin.structure import rcsa

E, NU, RHO = 210e9, 0.3, 7800.0


def beam_matrices(L, d, n):
    sec = B.circle(d, E, NU, RHO)
    z = B.subdivide([0.0, L], L / n)
    asm = B.BeamAssembly(len(z))
    for i in range(len(z) - 1):
        asm.add_element(i, i + 1, z[i + 1] - z[i], sec)
    return asm.K, asm.M, len(z)


def test_two_beams_rigidly_coupled_equal_the_monolithic_beam():
    # A: free-free 60 mm of d 12 (tip at node 0, joint at its end); B: 40 mm of d 20, its far end clamped
    KA, MA, nA = beam_matrices(0.06, 0.012, 12)
    KB, MB, nB = beam_matrices(0.04, 0.020, 8)
    # monolithic: A's last node = B's first node, B's last node clamped
    n = nA + nB - 1
    K = np.zeros((6 * n, 6 * n))
    M = np.zeros((6 * n, 6 * n))
    K[:6 * nA, :6 * nA] += KA
    M[:6 * nA, :6 * nA] += MA
    off = 6 * (nA - 1)
    K[off:, off:] += KB
    M[off:, off:] += MB
    keep = np.arange(6 * (n - 1))
    tip = [B.UX, B.RY]
    joint_A = np.arange(6 * (nA - 1), 6 * nA)
    keep_B = np.arange(6 * (nB - 1))                   # B with its far end clamped
    for w in (2 * math.pi * 150.0, 2 * math.pi * 900.0, 2 * math.pi * 2600.0):
        G_mono = rcsa.fe_receptance(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], w, tip, tip)
        A_dofs = np.concatenate([tip, joint_A])
        H = rcsa.fe_receptance(KA, MA, w, A_dofs, A_dofs)
        G_B = rcsa.fe_receptance(KB[np.ix_(keep_B, keep_B)], MB[np.ix_(keep_B, keep_B)], w, np.arange(6), np.arange(6))
        G = rcsa.couple(H[:2, :2], H[:2, 2:], H[2:, :2], H[2:, 2:], G_B)
        assert np.allclose(G, G_mono, rtol=1e-8, atol=1e-12 * np.abs(G_mono).max())


def test_elastic_joint_softens_the_assembly():
    KA, MA, nA = beam_matrices(0.05, 0.01, 10)
    KB, MB, nB = beam_matrices(0.03, 0.02, 6)
    keep_B = np.arange(6 * (nB - 1))
    joint_A = np.arange(6 * (nA - 1), 6 * nA)
    w = 2 * math.pi * 50.0
    A_dofs = np.concatenate([[B.UX], joint_A])
    H = rcsa.fe_receptance(KA, MA, w, A_dofs, A_dofs)
    G_B = rcsa.fe_receptance(KB[np.ix_(keep_B, keep_B)], MB[np.ix_(keep_B, keep_B)], w, np.arange(6), np.arange(6))
    rigid = rcsa.couple(H[:1, :1], H[:1, 1:], H[1:, :1], H[1:, 1:], G_B)
    Kj = np.diag([1e7, 1e7, 1e9, 1e5, 1e5, 1e5])
    soft = rcsa.couple(H[:1, :1], H[:1, 1:], H[1:, :1], H[1:, 1:], G_B, joint_compliance=np.linalg.inv(Kj))
    assert abs(soft[0, 0]) > abs(rigid[0, 0])


def test_modal_base_coupling_equals_the_reduced_assembly():
    """RCSA with a modal base (unit-mass coordinates, shapes Psi at the joint) is the transformation-reduced system."""
    KA, MA, nA = beam_matrices(0.07, 0.012, 14)
    joint = np.arange(6 * (nA - 1), 6 * nA)
    w_b = np.array([2 * math.pi * 900.0, 2 * math.pi * 900.0])
    z_b = np.array([0.04, 0.04])
    m = 7e7 / w_b[0] ** 2
    Psi = np.zeros((6, 2))
    Psi[B.UX, 0] = Psi[B.UY, 1] = 1 / math.sqrt(m)
    Psi[B.RY, 0] = -1 / (0.3 * math.sqrt(m))
    Psi[B.RX, 1] = 1 / (0.3 * math.sqrt(m))
    # reduced system: joint DOFs = Psi q
    nf = 6 * (nA - 1)
    T = np.zeros((6 * nA, nf + 2))
    T[:nf, :nf] = np.eye(nf)
    T[nf:, nf:] = Psi
    Kr = T.T @ KA @ T
    Mr = T.T @ MA @ T
    Kr[nf:, nf:] += np.diag(w_b ** 2)
    Mr[nf:, nf:] += np.eye(2)
    C = np.zeros_like(Kr)
    C[nf:, nf:] = np.diag(2 * z_b * w_b)
    for w in (2 * math.pi * 300.0, 2 * math.pi * 880.0, 2 * math.pi * 4000.0):
        G_rcsa = rcsa.couple_fe_to_modal_base(KA, MA, w, [B.UX], joint, w_b, z_b, Psi)
        Z = rcsa.dynamic_stiffness(Kr, Mr, w, None, C)
        e = np.zeros(nf + 2)
        e[B.UX] = 1.0
        G_red = np.linalg.solve(Z, e.astype(complex))[B.UX]
        assert complex(G_rcsa[0, 0]) == pytest.approx(complex(G_red), rel=1e-8)


def test_rigid_base_fixes_the_joint():
    KA, MA, nA = beam_matrices(0.05, 0.01, 10)
    joint = np.arange(6 * (nA - 1), 6 * nA)
    w = 2 * math.pi * 100.0
    G = rcsa.couple_fe_to_modal_base(KA, MA, w, [B.UX], joint, np.zeros(0), np.zeros(0), np.zeros((6, 0)))
    keep = np.arange(6 * (nA - 1))
    G_fixed = rcsa.fe_receptance(KA[np.ix_(keep, keep)], MA[np.ix_(keep, keep)], w, [B.UX], [B.UX])
    assert complex(G[0, 0]) == pytest.approx(complex(G_fixed[0, 0]), rel=1e-12)
