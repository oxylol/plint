"""Receptance coupling substructure analysis (RCSA; Schmitz & Smith 2009 ch. 8, Erturk, Ozguven & Budak 2006,
Schmitz & Duncan 2005; docs/models/structure.md section 2.4).

A substructure is described by its receptance matrix ``H(omega)`` (displacement / rotation per force / moment)
between its DOFs. Two substructures A (internal DOFs a, coupling DOFs c) and B (coupling DOFs c', internal b)
joined at c = c' through a joint of complex stiffness K_j (rigid: K_j^-1 = 0) give

    G_aa = H_aa - H_ac (H_cc + G_c'c' + K_j^-1)^-1 H_ca
    G_ba = G_bc' (H_cc + G_c'c' + K_j^-1)^-1 H_ca
    G_bb = G_bb - G_bc' (H_cc + G_c'c' + K_j^-1)^-1 G_c'b

(the compatibility and equilibrium of the joint). The receptances of a finite-element substructure follow from
its dynamic stiffness ``Z = K (1 + i eta) + i omega C - omega^2 M`` (exact for the model) or from its modes; a
modal substructure (the spindle-holder base of the machine file) from ``sum_k psi_k psi_k^T / (omega_k^2 -
omega^2 + 2 i zeta_k omega_k omega)`` with mass-normalised shapes psi_k.
"""

from __future__ import annotations

import numpy as np


def dynamic_stiffness(K: np.ndarray, M: np.ndarray, omega: float, hysteretic: list | None = None,
                      C: np.ndarray | None = None) -> np.ndarray:
    """Z(omega) = K + i sum_p eta_p K_p + i omega C - omega^2 M (hysteretic parts as (eta, K_p))."""
    Z = K.astype(complex) - (omega * omega) * M
    for eta, Kp in hysteretic or []:
        Z = Z + 1j * eta * Kp
    if C is not None:
        Z = Z + 1j * omega * C
    return Z


def fe_receptance(K: np.ndarray, M: np.ndarray, omega: float, rows, cols, hysteretic: list | None = None,
                  C: np.ndarray | None = None) -> np.ndarray:
    """Receptance block H[rows, cols] of an FE substructure at omega (direct solve of the dynamic stiffness)."""
    Z = dynamic_stiffness(K, M, omega, hysteretic, C)
    E = np.zeros((K.shape[0], len(cols)), dtype=complex)
    E[np.asarray(cols), np.arange(len(cols))] = 1.0
    X = np.linalg.solve(Z, E)
    return X[np.asarray(rows), :]


def modal_receptance(omega: float, omega_k: np.ndarray, zeta_k: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """sum_k psi_k psi_k^T / (omega_k^2 - omega^2 + 2 i zeta_k omega_k omega); Psi [n_dof, n_modes]."""
    w = np.asarray(omega_k, dtype=float)
    z = np.asarray(zeta_k, dtype=float)
    den = w * w - omega * omega + 2j * z * w * omega
    return (Psi / den[None, :]) @ Psi.T


def couple(H_aa, H_ac, H_ca, H_cc, G_cc, G_bc=None, G_cb=None, G_bb=None, joint_compliance=None):
    """Assembly receptances of A (a, c) rigidly or elastically joined to B (c', b) at c = c' (module doc).
    Returns G_aa, or (G_aa, G_ba, G_bb) when B's internal blocks are given."""
    S = H_cc + G_cc
    if joint_compliance is not None:
        S = S + joint_compliance
    X = np.linalg.solve(S, H_ca)
    G_aa = H_aa - H_ac @ X
    if G_bc is None:
        return G_aa
    G_ba = G_bc @ X
    Y = np.linalg.solve(S, G_cb)
    G_bb_new = G_bb - G_bc @ Y
    return G_aa, G_ba, G_bb_new


def couple_fe_to_modal_base(K: np.ndarray, M: np.ndarray, omega: float, out_dofs, base_dofs, omega_k, zeta_k,
                            Psi: np.ndarray, hysteretic: list | None = None) -> np.ndarray:
    """Receptance at out_dofs of an FE substructure (free at base_dofs) rigidly joined at base_dofs to a modal
    base whose shapes at those DOFs are Psi [len(base_dofs), n_modes] (RCSA of the tool-holder assembly to the
    spindle-holder base). A base without modes is rigid: the FE DOFs base_dofs are fixed."""
    out_dofs = np.asarray(out_dofs)
    base_dofs = np.asarray(base_dofs)
    if Psi.shape[1] == 0:
        keep = np.setdiff1d(np.arange(K.shape[0]), base_dofs)
        pos = {int(d): i for i, d in enumerate(keep)}
        rows = np.array([pos[int(d)] for d in out_dofs])
        hyst = [(eta, Kp[np.ix_(keep, keep)]) for eta, Kp in hysteretic or []]
        return fe_receptance(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], omega, rows, rows, hyst)
    both = np.concatenate([out_dofs, base_dofs])
    H = fe_receptance(K, M, omega, both, both, hysteretic)
    n = len(out_dofs)
    G_base = modal_receptance(omega, omega_k, zeta_k, Psi)
    return couple(H[:n, :n], H[:n, n:], H[n:, :n], H[n:, n:], G_base)


__all__ = ["dynamic_stiffness", "fe_receptance", "modal_receptance", "couple", "couple_fe_to_modal_base"]
