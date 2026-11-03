"""Model coefficients of the mechanics library, read from db/models/mechanics-*.yaml.

:func:`oxley_vector` packs the Oxley constants and solver settings into the flat parameter
vector of the kernels (slots ``abi.P_*``); the other models read their dictionaries directly.
Every file is validated (schema and sources) by :func:`cnctwin.db.models.params`.
"""

from __future__ import annotations

import functools

import numpy as np

from ..db import models
from .abi import (N_PRM, NO_REF, P_BETA_HA, P_BETA_HB, P_BETA_LA, P_BETA_LB, P_BETA_RTMIN, P_BETA_SWITCH, P_C0_MAX,
                  P_C0_MIN, P_C0_N, P_C0_RES_MAX, P_CONTACT_ASPECT, P_DELTA_ITER, P_DELTA_MAX, P_DELTA_MIN,
                  P_DELTA_N, P_DTM_A, P_DTM_B, P_DTM_C, P_ETA, P_ITER_MAX, P_MERCHANT_C0, P_MERCHANT_DELTA, P_MU,
                  P_MU_FALLBACK, P_PHI_FLOOR, P_PHI_MAX, P_PHI_MIN, P_PHI_N, P_PSI, P_RATE_MIN, P_REF_C0, P_REF_CM,
                  P_REF_DELTA, P_REF_LAMBDA, P_TOL_C0, P_TOL_PHI, P_TOL_T, P_TSTAR_MAX)

OXLEY = "mechanics-oxley"
CONTACT = "mechanics-contact"
CALIBRATION = "mechanics-calibration"
CHIP = "mechanics-chip"
COLOUR = "mechanics-colour"


@functools.lru_cache(maxsize=None)
def get(model: str) -> dict:
    """name -> value of one mechanics model file (cached; the files do not change in a run)."""
    return dict(models.params(model))


def number(model: str, name: str) -> float:
    return float(get(model)[name])


@functools.lru_cache(maxsize=None)
def _oxley_base() -> tuple:
    ox = get(OXLEY)
    ct = get(CONTACT)
    prm = np.zeros(N_PRM)
    prm[P_ETA] = ox["eta_shear_plane_temp"]
    prm[P_PSI] = ox["psi_interface_temp"]
    prm[P_BETA_LA] = ox["beta_low_a"]
    prm[P_BETA_LB] = ox["beta_low_b"]
    prm[P_BETA_HA] = ox["beta_high_a"]
    prm[P_BETA_HB] = ox["beta_high_b"]
    prm[P_BETA_SWITCH] = ox["beta_switch_RTtanphi"]
    prm[P_BETA_RTMIN] = ox["min_beta_RTtanphi"]
    prm[P_DTM_A] = ox["dTM_a"]
    prm[P_DTM_B] = ox["dTM_b"]
    prm[P_DTM_C] = ox["dTM_c"]
    prm[P_PHI_MIN] = ox["min_phi_rad"]
    prm[P_PHI_MAX] = ox["max_phi_rad"]
    prm[P_PHI_N] = ox["phi_grid_n"]
    prm[P_C0_MIN] = ox["min_C0"]
    prm[P_C0_MAX] = ox["max_C0"]
    prm[P_C0_N] = ox["C0_grid_n"]
    prm[P_DELTA_MIN] = ox["min_delta"]
    prm[P_DELTA_MAX] = ox["max_delta"]
    prm[P_DELTA_N] = ox["delta_grid_n"]
    prm[P_DELTA_ITER] = ox["delta_refine_iter"]
    prm[P_C0_RES_MAX] = ox["max_c0_residual"]
    prm[P_TOL_PHI] = ox["tol_phi_rad"]
    prm[P_TOL_C0] = ox["tol_C0"]
    prm[P_TOL_T] = ox["tol_T_K"]
    prm[P_ITER_MAX] = ox["max_iter"]
    prm[P_TSTAR_MAX] = ox["max_jc_Tstar"]
    prm[P_RATE_MIN] = ox["min_jc_rate_ratio"]
    prm[P_CONTACT_ASPECT] = ct["default_contact_aspect"]
    prm[P_MERCHANT_C0] = ct["merchant_C0"]
    prm[P_MERCHANT_DELTA] = ct["merchant_delta"]
    prm[P_PHI_FLOOR] = ct["min_merchant_phi_rad"]
    prm[P_REF_LAMBDA] = NO_REF
    prm[P_REF_CM] = 0.0
    prm[P_REF_C0] = 0.0
    prm[P_REF_DELTA] = 0.0
    return tuple(float(v) for v in prm)


def oxley_vector(mu_limit: float, mu_fallback: float) -> np.ndarray:
    """Kernel parameter vector with the interface Coulomb limit ``mu_limit`` (<= 0: Oxley's
    sticking condition only) and the fallback friction coefficient ``mu_fallback``."""
    prm = np.array(_oxley_base(), dtype=np.float64)
    prm[P_MU] = float(mu_limit)
    prm[P_MU_FALLBACK] = float(mu_fallback)
    return prm


def bue_vector() -> np.ndarray:
    """[saturation ratio, growth time, decay time, break-off rate, residual fraction] for kernels.bue_step."""
    ch = get(CHIP)
    return np.array([ch["bue_saturation_ratio"], ch["bue_growth_time_s"], ch["bue_decay_time_s"],
                     ch["bue_breakoff_rate_1_s"], ch["bue_residual_fraction"]], dtype=np.float64)
