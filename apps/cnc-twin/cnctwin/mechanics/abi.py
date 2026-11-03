"""Array layouts of the mechanics kernels (docs/models/mechanics.md, "Kernels").

The Oxley solver and the table lookups are Numba kernels; their scalar inputs travel in flat
float64 vectors whose slots are named here, and their results in an output vector of
``N_OUT`` slots. A :class:`ProcessMap` holds those output vectors on a grid of effective uncut
chip thickness, cutting speed and tool rake temperature, so the coefficient pipeline can read
them by trilinear interpolation instead of solving Oxley's equations every macro step.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ------------------------------------------------------------------ work material vector (mat)
MAT_A_PA = 0            # Johnson-Cook yield term A
MAT_B_PA = 1            # hardening modulus B
MAT_N = 2               # hardening exponent n
MAT_C = 3               # strain-rate constant C
MAT_M = 4               # thermal exponent m
MAT_EPS0_1_S = 5        # reference strain rate
MAT_TREF_K = 6          # reference temperature
MAT_TMELT_K = 7         # melting temperature of the JC law
MAT_RHO_KG_M3 = 8       # density (20 degC)
N_MAT = 9

# ------------------------------------------------------------------ tool-side vector (tool)
TOOL_T_RAKE_K = 0       # tool rake-face temperature at the contact (<= 0: adiabatic tool, all rake heat to the chip)
TOOL_RHO_KG_M3 = 1      # tool substrate density
TOOL_R_COAT_M2K_W = 2   # thermal resistance of the coating layer (thickness / conductivity)
TOOL_WIDTH_M = 3        # width of the contact along the edge (<= 0: default aspect ratio x contact length)
TOOL_TIME_S = 4         # duration of one contact (milling engagement; <= 0: steady, continuous cutting)
N_TOOL = 5

# ------------------------------------------------------------------ parameter vector (prm)
P_ETA = 0
P_PSI = 1
P_BETA_LA = 2
P_BETA_LB = 3
P_BETA_HA = 4
P_BETA_HB = 5
P_BETA_SWITCH = 6
P_BETA_RTMIN = 7
P_DTM_A = 8
P_DTM_B = 9
P_DTM_C = 10
P_PHI_MIN = 11
P_PHI_MAX = 12
P_PHI_N = 13
P_C0_MIN = 14
P_C0_MAX = 15
P_C0_N = 16
P_DELTA_MIN = 17
P_DELTA_MAX = 18
P_DELTA_N = 19
P_DELTA_ITER = 20
P_C0_RES_MAX = 21
P_TOL_PHI = 22
P_TOL_C0 = 23
P_TOL_T = 24
P_ITER_MAX = 25
P_TSTAR_MAX = 26
P_RATE_MIN = 27
P_MU = 28               # Coulomb limit of the interface shear stress (mu x friction factor); <= 0: none (Oxley)
P_CONTACT_ASPECT = 29
P_MERCHANT_C0 = 30
P_MERCHANT_DELTA = 31
P_PHI_FLOOR = 32
P_REF_LAMBDA = 33       # friction angle of the nearest converged solution (< -9: none)
P_REF_CM = 34           # Merchant constant 2 phi + lambda - alpha of that solution
P_REF_C0 = 35
P_REF_DELTA = 36
P_MU_FALLBACK = 37      # friction coefficient of the fallback when no converged neighbour exists
N_PRM = 38
NO_REF = -10.0

# ------------------------------------------------------------------ output vector (out)
OX_STATUS = 0           # STATUS_*
OX_REGIME = 1           # REGIME_*
OX_PHI = 2              # shear angle (normal plane)
OX_C0 = 3
OX_DELTA = 4
OX_FC = 5               # cutting force per unit width (N/m), along the cutting velocity
OX_FT = 6               # thrust force per unit width, normal to the machined surface
OX_F = 7                # friction force on the rake face per unit width
OX_N = 8                # normal force on the rake face per unit width
OX_FS = 9               # shear force on AB per unit width
OX_THETA = 10
OX_LAMBDA = 11          # mean friction angle on the rake face
OX_T2 = 12              # chip thickness
OX_CHIP_RATIO = 13      # t1 / t2
OX_LC = 14              # tool-chip contact length
OX_TAU_INT = 15         # mean interface shear stress
OX_SIGMA_N = 16         # mean interface normal stress
OX_SIGMA_NP = 17        # normal stress at B from the shear-plane stress field
OX_K_AB = 18            # shear flow stress on AB (tau_s)
OX_K_CHIP = 19          # shear flow stress of the chip at the interface
OX_N_EQ = 20            # equivalent strain-hardening index at AB
OX_T_AB = 21            # shear-plane temperature
OX_T_INT = 22           # mean tool-chip interface temperature
OX_T_CHIP = 23          # mean chip temperature leaving the contact
OX_DT_SZ = 24           # temperature rise in the primary zone
OX_DT_C = 25            # mean chip temperature rise from all rake friction heat
OX_DT_M = 26            # maximum chip temperature rise at the interface (all rake heat to the chip)
OX_BETA = 27            # fraction of the shear-zone heat conducted into the work
OX_R_CHIP = 28          # fraction of the rake friction heat entering the chip
OX_EPS_AB = 29
OX_EPSDOT_AB = 30
OX_EPS_INT = 31
OX_EPSDOT_INT = 32
OX_VS = 33              # shear velocity
OX_VC = 34              # chip velocity
OX_C0_RES = 35          # |sigma_N' - sigma_N| / k_AB at the solution (0 when the condition holds)
OX_G1_RES = 36          # |tau_int - tau_lim| / k_AB at the solution
OX_T_CAPPED = 37        # 1 if the shear-plane temperature hit the homologous-temperature clamp
OX_P_SHEAR = 38         # shear-plane power / cutting power (the rest is rake friction)
OX_STICK = 39           # Zorev sticking fraction of the contact (filled by the Python layer)
N_OUT = 40

STATUS_INVALID = 0.0
STATUS_OXLEY = 1.0
STATUS_MERCHANT = 2.0
STATUS_INDENTATION = 3.0

REGIME_STICKING = 1.0   # interface shear = chip shear flow stress (Oxley)
REGIME_SLIDING = 2.0    # interface shear = mu x normal stress (Coulomb-Tresca limit active)

# Output slots that scale with the uncut chip thickness: the map interpolates them divided by t1.
SCALES_WITH_T1 = (OX_FC, OX_FT, OX_F, OX_N, OX_FS, OX_T2, OX_LC)
# Slots that are flags or codes: the map takes them from the nearest node.
DISCRETE_SLOTS = (OX_STATUS, OX_REGIME, OX_T_CAPPED)

# ------------------------------------------------------------------ edge / ploughing model output
PL_FT = 0               # tangential force per unit edge length (along the cutting velocity)
PL_FR = 1               # thrust per unit edge length (normal to the machined surface)
PL_FA = 2               # along the axis perpendicular to the velocity in the cut surface
PL_CONTACT = 3          # contact arc length per unit edge length (m)
PL_AREA = 4             # interference area per unit edge length (m2)
N_PL = 5


class ProcessMap(NamedTuple):
    """Oxley outputs on a (tool temperature, speed, effective uncut chip thickness) grid.

    ``values[iT, iv, ih, :]`` is the output vector (``OX_*``) of the orthogonal solution at
    ``Ttool_grid_K[iT]``, ``speed_grid_m_s[iv]`` and ``t1_grid_m[ih]``, for one work material,
    tool, normal rake, workpiece temperature and friction state (``context``).
    """

    t1_grid_m: np.ndarray           # [nh] ascending, log-spaced
    speed_grid_m_s: np.ndarray      # [nv] ascending, log-spaced
    Ttool_grid_K: np.ndarray        # [nT] ascending (a single node for an adiabatic tool)
    values: np.ndarray              # [nT, nv, nh, N_OUT]
    context: np.ndarray             # [N_CTX] the fixed conditions the map was built for


CTX_RAKE_RAD = 0
CTX_WORK_TEMP_K = 1
CTX_MU = 2
CTX_ADIABATIC = 3
CTX_CONTACT_WIDTH_M = 4
CTX_CONTACT_TIME_S = 5
N_CTX = 6


def empty_out() -> np.ndarray:
    return np.zeros(N_OUT)
