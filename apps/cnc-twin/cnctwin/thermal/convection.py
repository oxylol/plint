"""Convection and radiation correlations (coefficients in db/models/thermal-convection.yaml).

* free convection: Churchill & Chu (1975) for vertical plates and horizontal cylinders,
  horizontal plates after Incropera et al. (2007, 9.6.2); air properties at the film temperature
  by Sutherland's law (White 2006), ideal-gas density;
* rotating cylinder in still air (tool and holder in the spindle): Nu_D = 0.133 Re^(2/3) Pr^(1/3)
  (Kays & Bjorklund 1958), combined with free convection by taking the larger h;
* pipe flow (through-tool coolant): Dittus-Boelter, laminar Nu = 4.36;
* radiation to an enclosure at T_sink, linearised: h_r = eps sigma (T^2 + Ts^2)(T + Ts).
"""

from __future__ import annotations

import functools
import math

import numpy as np

from ..db.models import params
from ..units import (AIR_CP_20C_J_KGK, AIR_DENSITY_20C_KG_M3, REFERENCE_TEMP_K, STANDARD_GRAVITY_M_S2,
                     STEFAN_BOLTZMANN_W_M2K4)


@functools.lru_cache(maxsize=1)
def _p() -> dict:
    return dict(params("thermal-convection"))


def air_properties(temp_K):
    """(kinematic viscosity m2/s, conductivity W/mK, Prandtl number, expansion 1/K) of air (scalars or
    arrays)."""
    p = _p()
    t = np.maximum(np.asarray(temp_K, dtype=float), 150.0)
    t0 = p["air_ref_temp_K"]
    s_mu = p["air_sutherland_mu_K"]
    s_k = p["air_sutherland_k_K"]
    mu = p["air_mu_ref_Pa_s"] * (t / t0) ** 1.5 * (t0 + s_mu) / (t + s_mu)
    k = p["air_k_ref_W_mK"] * (t / t0) ** 1.5 * (t0 + s_k) / (t + s_k)
    rho = AIR_DENSITY_20C_KG_M3 * REFERENCE_TEMP_K / t
    return mu / rho, k, mu * AIR_CP_20C_J_KGK / k, 1.0 / t


def rayleigh(surface_K, fluid_K, length_m):
    """(Ra, Pr, k_air) at the film temperature (scalars or arrays)."""
    nu, k, pr, beta = air_properties(0.5 * (np.asarray(surface_K, dtype=float) + fluid_K))
    ra = STANDARD_GRAVITY_M_S2 * beta * np.abs(np.asarray(surface_K, dtype=float) - fluid_K) * \
        np.asarray(length_m, dtype=float) ** 3 * pr / (nu * nu)
    return ra, pr, k


def _churchill_chu(ra: float, pr: float, a: float, b: float, c: float) -> float:
    return (a + b * ra ** (1.0 / 6.0) / (1.0 + (c / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2


def h_vertical_plate(surface_K: float, fluid_K: float, height_m: float) -> float:
    p = _p()
    ra, pr, k = rayleigh(surface_K, fluid_K, height_m)
    return float(_churchill_chu(ra, pr, p["cc_vertical_a"], p["cc_vertical_b"], p["cc_vertical_pr"]) * k / height_m)


def h_horizontal_plate(surface_K: float, fluid_K: float, length_m: float, faces_up: bool) -> float:
    """Horizontal face (L = A / P); ``faces_up``: outward normal points up."""
    p = _p()
    ra, _, k = rayleigh(surface_K, fluid_K, length_m)
    hot = surface_K > fluid_K
    if faces_up == hot:            # upper face of a hot plate or lower face of a cold one
        if ra <= p["plate_up_ra_switch"]:
            nu = p["plate_up_low_c"] * ra ** p["plate_up_low_n"]
        else:
            nu = p["plate_up_high_c"] * ra ** p["plate_up_high_n"]
    else:
        nu = p["plate_down_c"] * ra ** p["plate_down_n"]
    return float(nu * k / length_m)


def h_face(up_component: float, surface_K: float, fluid_K: float, height_m: float, plate_length_m: float) -> float:
    """Free convection on a face whose outward normal has vertical component ``up_component``."""
    if abs(up_component) <= _p()["face_vertical_cos"]:
        return h_vertical_plate(surface_K, fluid_K, height_m)
    return h_horizontal_plate(surface_K, fluid_K, plate_length_m, up_component > 0.0)


def h_horizontal_cylinder(surface_K, fluid_K, diameter_m):
    """Free convection on a long horizontal cylinder (scalars or arrays)."""
    p = _p()
    ra, pr, k = rayleigh(surface_K, fluid_K, diameter_m)
    return _churchill_chu(ra, pr, p["cc_cylinder_a"], p["cc_cylinder_b"], p["cc_cylinder_pr"]) * k / \
        np.asarray(diameter_m, dtype=float)


def h_rotating_cylinder(omega_rad_s, diameter_m, surface_K, fluid_K):
    """Rotating cylinder in still air, at least the free-convection value of a still cylinder
    (scalars or arrays)."""
    p = _p()
    d = np.asarray(diameter_m, dtype=float)
    nu, k, pr, _ = air_properties(0.5 * (np.asarray(surface_K, dtype=float) + fluid_K))
    re = np.minimum(abs(float(omega_rad_s)) * d * d / nu, p["rot_cyl_re_max"])
    h_rot = p["rot_cyl_c"] * re ** p["rot_cyl_re_exp"] * pr ** p["rot_cyl_pr_exp"] * k / d
    return np.maximum(h_rot, h_horizontal_cylinder(surface_K, fluid_K, d))


def h_pipe(velocity_m_s: float, diameter_m: float, density_kg_m3: float, viscosity_Pa_s: float,
           conductivity_W_mK: float, cp_J_kgK: float) -> float:
    """Fully developed pipe flow: laminar 4.36, turbulent Dittus-Boelter (the larger above transition)."""
    p = _p()
    re = density_kg_m3 * abs(velocity_m_s) * diameter_m / viscosity_Pa_s
    pr = viscosity_Pa_s * cp_J_kgK / conductivity_W_mK
    nu = p["laminar_nusselt"]
    if re > p["pipe_re_transition"]:
        nu = max(nu, p["dittus_boelter_c"] * re ** p["dittus_boelter_re_exp"] * pr ** p["dittus_boelter_pr_exp"])
    return nu * conductivity_W_mK / diameter_m


def h_radiation(emissivity, surface_K, sink_K):
    """Linearised radiation coefficient (scalars or arrays)."""
    t = np.asarray(surface_K, dtype=float)
    return emissivity * STEFAN_BOLTZMANN_W_M2K4 * (t * t + sink_K * sink_K) * (t + sink_K)


def emissivity(family: str | None) -> float:
    p = _p()
    families = list(p["emissivity_families"])
    values = list(p["emissivity_values"])
    if family in families:
        return float(values[families.index(family)])
    return float(p["emissivity_default"])


def gravity_up_in(frame_R_M: "list | None") -> tuple[float, float, float]:
    """Upward unit vector (+z_M) expressed in a frame whose rotation to M is ``frame_R_M``."""
    if frame_R_M is None:
        return 0.0, 0.0, 1.0
    r = frame_R_M
    up = (r[2][0], r[2][1], r[2][2])          # R^T e_z = third row of R
    norm = math.sqrt(sum(c * c for c in up)) or 1.0
    return up[0] / norm, up[1] / norm, up[2] / norm
