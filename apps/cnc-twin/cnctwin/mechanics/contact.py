"""Edge, ploughing, indentation and flank-wear contact models (docs/models/mechanics.md, "Edge",
"Flank", "Fallbacks").

* **Edge forces** (h >= h_min): the material below the stagnation point (height h_min above the
  lowest point of the rounded edge) is pressed under the edge (Albrecht 1960). Pressure
  p = (2 + pi) k (Prandtl's flat punch, Hill 1950) and friction stress tau = min(mu p, k)
  (Coulomb-Tresca) act on the arc; the integrals are in :func:`kernels.edge_contact_forces`.
* **Ploughing** (h < h_min): the same contact at depth h; the interference area gives the
  indentation coefficient K_sp = K_re / A(h_min) the hot loop uses below h_min (continuous at
  h_min).
* **Indentation / extrusion** (chisel edges, normal rake below ``min_shear_rake_rad``, speeds
  below ``min_shear_speed_m_s``): depth h with the rake face in contact and the material above
  the stagnation height flowing up the face.
* **Flank wear land** (Smithey, Kapoor & DeVor 2001): stresses sigma_0 = (2 + pi) k and
  tau_0 = min(mu sigma_0, k) up to the critical land VB*, decaying as ((VB - x)/(VB - VB*))^2
  beyond it.
* **Shear-angle relations** of the fallback: Merchant (1945) and Lee & Shaffer (1951).
"""

from __future__ import annotations

import math

import numpy as np

from . import abi
from . import kernels as K
from . import params as P
from ..core.abi import EDGE_CHISEL
from .materials import WorkMaterial


def pressure_factor() -> float:
    return P.number(P.CONTACT, "indentation_pressure_factor")


def min_chip_thickness(work: WorkMaterial, edge_radius_m: float) -> float:
    """h_min = (database ratio or the model default) x edge radius."""
    ratio = work.min_chip_thickness_ratio
    if ratio is None:
        ratio = P.number(P.CONTACT, "default_min_chip_thickness_ratio")
    return float(ratio) * max(float(edge_radius_m), 0.0)


def contact_stresses(k_ref_Pa: float, mu: float) -> tuple[float, float]:
    """(pressure, friction stress) of material pressed under an edge or a wear land."""
    p = pressure_factor() * k_ref_Pa
    tau = min(mu * p, k_ref_Pa) if mu > 0.0 else k_ref_Pa
    return p, tau


def edge_forces(edge_radius_m: float, h_min_m: float, rake_rad: float, incl_rad: float, k_ref_Pa: float,
                mu: float) -> np.ndarray:
    """Edge (ploughing) forces per unit edge length for h >= h_min: abi.PL_* vector
    (tangential, thrust, axial, contact length, interference area)."""
    p, tau = contact_stresses(k_ref_Pa, mu)
    out = np.zeros(abi.N_PL)
    K.edge_contact_forces(float(h_min_m), float(edge_radius_m), float(h_min_m), float(rake_rad), float(incl_rad), p,
                          tau, out)
    return out


def ploughing_forces(h_m: float, edge_radius_m: float, h_min_m: float, rake_rad: float, incl_rad: float,
                     k_ref_Pa: float, mu: float) -> np.ndarray:
    """Forces per unit edge length below h_min (no chip): contact at depth h, all material flowing
    under the edge."""
    p, tau = contact_stresses(k_ref_Pa, mu)
    out = np.zeros(abi.N_PL)
    depth = min(float(h_m), float(h_min_m))
    K.edge_contact_forces(depth, float(edge_radius_m), float(h_min_m), float(rake_rad), float(incl_rad), p, tau, out)
    return out


def indentation_forces(h_m: float, edge_radius_m: float, h_min_m: float, rake_rad: float, incl_rad: float,
                       k_ref_Pa: float, mu: float) -> np.ndarray:
    """Indentation / extrusion forces per unit edge length (chisel edges, very negative rakes,
    speeds below v_min): contact up to depth h including the rake face; material above the
    stagnation height h_min flows up the face."""
    p, tau = contact_stresses(k_ref_Pa, mu)
    out = np.zeros(abi.N_PL)
    K.edge_contact_forces(float(h_m), float(edge_radius_m), float(h_min_m), float(rake_rad), float(incl_rad), p, tau,
                          out)
    return out


def interference_area(h_m: float, edge_radius_m: float, rake_rad: float = 0.0) -> float:
    """Cross-section area (m2 per m of edge) between the uncut surface at depth h and the edge."""
    out = np.zeros(abi.N_PL)
    K.edge_contact_forces(float(h_m), float(edge_radius_m), float(h_m), float(rake_rad), 0.0, 0.0, 0.0, out)
    return float(out[abi.PL_AREA])


def indentation_coefficient(work: WorkMaterial, edge_thrust_N_per_m: float, h_min_m: float, edge_radius_m: float,
                            rake_rad: float) -> tuple[float, str]:
    """K_sp (N/m3) for the hot loop's volume-based ploughing/process-damping model and where it
    came from: an identified database value (src not 'est'), else the model-consistent value
    K_re / A(h_min) that makes the indentation force continuous with the edge force at h_min."""
    if work.indentation_coeff_N_per_m3 is not None and not work.indentation_estimated:
        return float(work.indentation_coeff_N_per_m3), "database"
    area = interference_area(h_min_m, edge_radius_m, rake_rad)
    if area > 0.0 and edge_thrust_N_per_m > 0.0:
        return edge_thrust_N_per_m / area, "model"
    if work.indentation_coeff_N_per_m3 is not None:
        return float(work.indentation_coeff_N_per_m3), "database-estimate"
    return 0.0, "none"


def ploughing_shape(edge_radius_m: float, h_min_m: float, rake_rad: float, incl_rad: float, k_ref_Pa: float,
                    mu: float, n: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ploughing forces below h_min normalised by the edge forces, on x = h / h_min in [0, 1]:
    (x, tangential ratio, thrust ratio). 1 at x = 1 by construction."""
    x = np.linspace(0.0, 1.0, n)
    edge = edge_forces(edge_radius_m, h_min_m, rake_rad, incl_rad, k_ref_Pa, mu)
    ft = np.zeros(n)
    fr = np.zeros(n)
    for i, xi in enumerate(x):
        f = ploughing_forces(xi * h_min_m, edge_radius_m, h_min_m, rake_rad, incl_rad, k_ref_Pa, mu)
        ft[i] = f[abi.PL_FT] / edge[abi.PL_FT] if edge[abi.PL_FT] != 0.0 else 0.0
        fr[i] = f[abi.PL_FR] / edge[abi.PL_FR] if edge[abi.PL_FR] != 0.0 else 0.0
    return x, ft, fr


def flank_stresses(VB_m, sigma0_Pa: float, tau0_Pa: float, VB_star_m: float | None = None):
    """Mean normal and shear stress over a flank wear land of length VB (Smithey et al. 2001):
    plastic (sigma_0, tau_0) up to VB*, then decaying with the square of the distance to the end
    of the land, whose mean is one third. Works element-wise on arrays."""
    vb_star = P.number(P.CONTACT, "flank_plastic_land_m") if VB_star_m is None else float(VB_star_m)
    vb = np.asarray(VB_m, dtype=float)
    safe = np.where(vb > 0.0, vb, 1.0)
    plastic = np.minimum(vb, vb_star)
    elastic = np.maximum(vb - vb_star, 0.0)
    factor = np.where(vb > 0.0, (plastic + elastic / 3.0) / safe, 0.0)
    sigma = sigma0_Pa * factor
    tau = tau0_Pa * factor
    if np.ndim(VB_m) == 0:
        return float(sigma), float(tau)
    return sigma, tau


def flank_reference_stresses(k_ref_Pa: float, mu: float) -> tuple[float, float]:
    """sigma_0, tau_0 of the plastic part of a wear land."""
    return contact_stresses(k_ref_Pa, mu)


def shear_angle_merchant(friction_angle_rad: float, rake_rad: float, constant_rad: float = 0.5 * math.pi) -> float:
    """Merchant (1945): 2 phi + beta - alpha = C (C = pi/2 from minimum energy; modified: C fitted)."""
    return 0.5 * (constant_rad - friction_angle_rad + rake_rad)


def shear_angle_lee_shaffer(friction_angle_rad: float, rake_rad: float) -> float:
    """Lee & Shaffer (1951): phi = pi/4 - beta + alpha (negative for beta - alpha > 45 deg)."""
    return 0.25 * math.pi - friction_angle_rad + rake_rad


def merchant_forces(tau_s_Pa: float, t1_m: float, phi_rad: float, friction_angle_rad: float,
                    rake_rad: float) -> tuple[float, float]:
    """Cutting and thrust force per unit width of a shear-plane model with shear stress tau_s."""
    denom = math.sin(phi_rad) * math.cos(phi_rad + friction_angle_rad - rake_rad)
    return (tau_s_Pa * t1_m * math.cos(friction_angle_rad - rake_rad) / denom,
            tau_s_Pa * t1_m * math.sin(friction_angle_rad - rake_rad) / denom)


def uses_indentation(kind: int, rake_rad: float) -> bool:
    """Chisel edges and normal rakes below min_shear_rake_rad are treated by indentation."""
    return kind == EDGE_CHISEL or rake_rad < P.number(P.CONTACT, "min_shear_rake_rad")


def min_shear_speed() -> float:
    return P.number(P.CONTACT, "min_shear_speed_m_s")


__all__ = ["min_chip_thickness", "contact_stresses", "edge_forces", "ploughing_forces", "indentation_forces",
           "interference_area", "indentation_coefficient", "ploughing_shape", "flank_stresses",
           "flank_reference_stresses", "shear_angle_merchant", "shear_angle_lee_shaffer", "merchant_forces",
           "uses_indentation", "min_shear_speed", "pressure_factor"]
