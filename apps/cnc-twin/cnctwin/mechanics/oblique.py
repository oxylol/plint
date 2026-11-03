"""Oblique cutting (docs/models/mechanics.md, "Oblique cutting").

The orthogonal solution at the normal rake alpha_n gives the shear flow stress tau_s, the normal
shear angle phi_n and the friction angle beta (Armarego's equivalence); the chip flows at
eta_c = i (Stabler's flow rule). The specific cutting coefficients of Budak, Altintas &
Armarego (1996) follow, in the frame t (along the cutting velocity), r (normal to the machined
surface, the thrust/feed direction) and a (normal to both, along the tool axis for a helical
peripheral edge; the edge direction is e_a = sin(i) t + cos(i) a):

    Ktc = tau_s / sin(phi_n) (cos(beta_n - alpha_n) + tan(i) tan(eta) sin(beta_n)) / D
    Krc = tau_s / (sin(phi_n) cos(i)) sin(beta_n - alpha_n) / D
    Kac = tau_s / sin(phi_n) (cos(beta_n - alpha_n) tan(i) - tan(eta) sin(beta_n)) / D
    D = sqrt(cos^2(phi_n + beta_n - alpha_n) + tan^2(eta) sin^2(beta_n)),  tan(beta_n) = tan(beta) cos(eta)

so that F = K h b with h normal to the edge and b the chip width. At i = 0 they reduce to the
Oxley/Merchant forces exactly.
"""

from __future__ import annotations

import math

import numpy as np


def stabler_flow_angle(incl_rad: float, ratio: float = 1.0) -> float:
    """Chip flow angle eta_c = ratio x i (Stabler 1951; ratio 1)."""
    return ratio * incl_rad


def equivalent_rake(rake_n_rad: float, incl_rad: float, eta_c_rad: float) -> float:
    """Effective rake in the plane of the cutting and chip velocities (Stabler 1951):
    sin(alpha_e) = sin(eta_c) sin(i) + cos(eta_c) cos(i) sin(alpha_n)."""
    s = math.sin(eta_c_rad) * math.sin(incl_rad) + math.cos(eta_c_rad) * math.cos(incl_rad) * math.sin(rake_n_rad)
    return math.asin(max(-1.0, min(1.0, s)))


def normal_friction_angle(beta_rad, eta_c_rad):
    return np.arctan(np.tan(beta_rad) * np.cos(eta_c_rad))


def cutting_coefficients(tau_s_Pa, phi_n_rad, beta_rad, rake_n_rad: float, incl_rad: float,
                         eta_c_rad: float | None = None):
    """(Ktc, Krc, Kac) in Pa (arrays broadcast) by the transformation of Budak et al. (1996)."""
    eta = incl_rad if eta_c_rad is None else eta_c_rad
    tau = np.asarray(tau_s_Pa, dtype=float)
    phi = np.asarray(phi_n_rad, dtype=float)
    beta_n = normal_friction_angle(np.asarray(beta_rad, dtype=float), eta)
    tan_i = math.tan(incl_rad)
    tan_eta = math.tan(eta)
    d = np.sqrt(np.cos(phi + beta_n - rake_n_rad) ** 2 + (tan_eta * np.sin(beta_n)) ** 2)
    base = tau / (np.sin(phi) * d)
    ktc = base * (np.cos(beta_n - rake_n_rad) + tan_i * tan_eta * np.sin(beta_n))
    krc = base * np.sin(beta_n - rake_n_rad) / math.cos(incl_rad)
    kac = base * (np.cos(beta_n - rake_n_rad) * tan_i - tan_eta * np.sin(beta_n))
    return ktc, krc, kac


def chip_velocity_ratio(chip_ratio, incl_rad: float, eta_c_rad: float):
    """V_c / V from continuity: r_c cos(i) / cos(eta_c) (r_c = t1 / t2 in the normal plane)."""
    return np.asarray(chip_ratio) * math.cos(incl_rad) / math.cos(eta_c_rad)


def chip_flow_direction(rake_n_rad: float, incl_rad: float, eta_c_rad: float) -> np.ndarray:
    """Unit vector of the chip velocity relative to the tool in the (t, r, a) frame: up the rake
    face at eta_c from the edge normal, with the along-edge component opposite to the work's
    along-edge motion relative to the tool."""
    # rake-face 'up' direction in the normal plane, as (e_n, e_y) components
    up_n = -math.sin(rake_n_rad)
    up_y = math.cos(rake_n_rad)
    c_eta = math.cos(eta_c_rad)
    s_eta = math.sin(eta_c_rad)
    ci = math.cos(incl_rad)
    si = math.sin(incl_rad)
    # e_n = cos i t - sin i a ; e_a = sin i t + cos i a ; e_y = r
    v_n = c_eta * up_n
    v_y = c_eta * up_y
    v_a = -s_eta
    t = v_n * ci + v_a * si
    a = -v_n * si + v_a * ci
    vec = np.array([t, v_y, a])
    return vec / np.linalg.norm(vec)


def chip_width(width_m, incl_rad: float, eta_c_rad: float):
    """Chip width normal to its flow direction: b cos(eta_c) / cos(i) (= b with Stabler's rule)."""
    return np.asarray(width_m) * math.cos(eta_c_rad) / math.cos(incl_rad)


__all__ = ["stabler_flow_angle", "equivalent_rake", "normal_friction_angle", "cutting_coefficients",
           "chip_velocity_ratio", "chip_flow_direction", "chip_width"]
