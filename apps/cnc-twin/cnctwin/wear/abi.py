"""Kernel layouts of the wear domain (docs/models/wear.md "Kernels"; rules: architecture 5.5).

Every layout is a NamedTuple of writable C-contiguous arrays with fixed dtypes (float64, int64, int8),
so each kernel in :mod:`cnctwin.wear.kernels` has exactly one compiled signature. Per-element arrays
are ``[nt, ne]`` (the tool's EdgeModel); temperatures at the in-cut quadrature points are
``[nt, ne, nq]``. Scalars a kernel needs travel in small float64 ``scal`` arrays indexed by the
constants below.
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

# ------------------------------------------------------------------------------- Law.scal
LS_ABR = 0              # Archard term K x abrasiveness (dimensionless; divided by the hardness at T in the kernel)
LS_COAT_FACTOR = 1      # factor on the Usui rate while coating covers the contact (coating wear_rate_factor)
LS_COAT_TMAX_K = 2      # above it the coating no longer lowers the Usui rate (oxidation onset)
LS_NOTCH_EXCESS = 3     # f_notch - 1: excess wear rate at the depth-of-cut line
LS_ROUNDING = 4         # hone-radius growth per unit edge recession
LS_BUE_PROTECT = 5      # share of the rates suppressed under a stable BUE
LS_PLUCK_M = 6          # edge recession per BUE break-off (attrition)
LS_SKIN_FACTOR = 7      # skin abrasiveness factor (wear-rate multiplier while cutting the skin)
LS_FAT_SIGMA_PA = 8     # thermal-cycle fatigue curve: Basquin coefficient sigma_f' (0: no data)
LS_FAT_B = 9            # its exponent b (< 0)
LS_CRACK_A0_M = 10      # initial flaw depth (from K_IC and the characteristic strength)
LS_CRACK_GROWTH = 11    # Miner damage scale of comb-crack growth beyond initiation
LS_DOC_RATIO = 12       # depth-of-cut end: h at the end >= ratio x the run's largest h
LS_MIN_CLEAR = 13       # clearance of the land geometry is clipped below at this value (rad; g_vb's clip)
N_LS = 16

# ------------------------------------------------------------------------------- ElemIn.scal
IS_DT_S = 0             # macro step duration
N_IS = 4

# ------------------------------------------------------------------------------- ElemOut.scal
OS_COAT_THROUGH = 0     # elements whose flank coating wore through this step
OS_CRACK_INIT = 1       # elements whose thermal damage reached 1 this step
OS_MAX_RATE_M_S = 2     # largest flank wear-depth rate (m per s in cut)
OS_MAX_D = 3            # largest thermal damage
OS_SUM_DVN = 4          # sum of the flank depth increments
OS_RAKE_THROUGH = 5     # elements whose rake coating wore through this step
N_OS = 8

# ------------------------------------------------------------------------------- ChipLaw.scal
CL_M = 0                # Weibull modulus of the tool material
CL_SIGMA0_PA = 1        # characteristic strength at the reference effective volume
CL_VREF_M3 = 2          # effective volume of the TRS test piece
CL_KIC = 3              # fracture toughness (0: no crack term)
CL_Y = 4                # crack shape factor
CL_LC_RATIO = 5         # entry contact length / uncut chip thickness
CL_CHIP_RATIO = 6       # chip depth / entry contact length
CL_EXIT_FACTOR = 7      # stress factor of exit / breakout / breakthrough rows
CL_MU = 8               # tool-chip friction coefficient (force direction)
CL_COAT_SIGMA0_PA = 9   # coating spallation characteristic contact pressure
CL_COAT_M = 10          # its Weibull modulus
CL_BETA_MIN = 11        # wedge-angle guards
CL_BETA_MAX = 12
CL_NTHETA = 13          # nodes of the angular effective-volume integral
CL_ROUNDING = 14        # hone-radius growth per unit recession (as LS_ROUNDING)
CL_FATIGUE_B = 15       # Basquin exponent b (< 0) of the tool material: the edge's strength after N load cycles
N_CL = 16

# ------------------------------------------------------------------------------- chip_trials output columns
CO_P = 0                # chipping probability of the row
CO_CHIPPED = 1          # 1: the edge chipped at this row
CO_SIGMA_PA = 2         # largest tensile wedge stress at the end of the contact
CO_P_COAT = 3           # coating spallation probability
CO_SPALLED = 4          # 1: the coating spalled at this row
CO_DEPTH_M = 5          # chip depth (when chipped)
CO_VEFF_M3 = 6          # effective stressed volume
CO_HAZARD = 7           # the element's cumulative hazard after this row (fatigue sum and cracks)
N_CO = 8

# ------------------------------------------------------------------------------- shock rows (core.bus CuttingContact.shocks)
SH_TICK = 0
SH_TOOTH = 1
SH_ELEM = 2
SH_H = 3
SH_ANGLE = 4
SH_FORCE = 5
SH_LOADING_S = 6
SH_KIND = 7
SHOCK_WIDTH = 8
# kinds as written by cutting (cnctwin/cutting/abi.py SHOCK_*): 1 entry, 2 exit, 3 breakthrough, 4 breakout
SHOCK_ENTRY = 1

# ------------------------------------------------------------------------------- hash keys
KEY_SLOT = 1 << 32      # element key = slot * KEY_SLOT + tooth * KEY_TOOTH + elem
KEY_TOOTH = 1 << 16


class ElemGeo(NamedTuple):
    """Static element geometry of one tool (from its nominal EdgeModel), with derived factors."""

    valid: np.ndarray           # [nt, ne] int8
    kind: np.ndarray            # [nt, ne] int64 core.abi EDGE_*
    sin_k: np.ndarray           # [nt, ne] radial share of the edge recession (sin kappa)
    cos_k: np.ndarray           # [nt, ne] axial share (cos kappa)
    rake: np.ndarray            # [nt, ne] normal rake
    clear: np.ndarray           # [nt, ne] normal clearance
    g_vb: np.ndarray            # [nt, ne] VB / VN = cot(a) - tan(g) (a clipped below)
    c_coat: np.ndarray          # [nt, ne] depth into the flank coating at the edge per unit VN
    beta: np.ndarray            # [nt, ne] wedge angle pi/2 - g - a (clipped)
    r_e0: np.ndarray            # [nt, ne] initial hone radius
    db: np.ndarray              # [nt, ne] chip width


class ElemIn(NamedTuple):
    """One tool's inputs for one macro step (cutting's integrals, thermal's temperatures)."""

    in_cut_s: np.ndarray        # [nt, ne]
    sv_flank: np.ndarray        # [nt, ne] integral of sigma_n V_s dt on the flank (Pa m)
    sv_rake: np.ndarray         # [nt, ne] ... on the rake (Pa m)
    h_int: np.ndarray           # [nt, ne] integral of h dt (m s)
    entries: np.ndarray         # [nt, ne] entries (thermal and load cycles) in the step
    skin_frac: np.ndarray       # [nt, ne] share of the in-cut time spent in the stock's skin layer
    bue_broke: np.ndarray       # [nt, ne] BUE break-offs in the step (0/1)
    flank_T: np.ndarray         # [nt, ne, nq] flank-contact temperature at the quadrature points (K)
    rake_T: np.ndarray          # [nt, ne, nq] rake-contact temperature at the quadrature points (K)
    w_q: np.ndarray             # [nq] quadrature weights (sum 1)
    cycle_dT: np.ndarray        # [nt, ne] edge temperature range of one thermal cycle (K)
    therm_coef: np.ndarray      # [nt, ne] alpha E / (2 (1 - nu)) at the element (Pa / K)
    diffusivity: np.ndarray     # [nt, ne] tool thermal diffusivity at the element (m2/s)
    scal: np.ndarray            # [N_IS]


class Law(NamedTuple):
    """Wear law of one tool: Usui branches per face, hardness tables, scalars (LS_*)."""

    fl_A: np.ndarray            # [nbf] Usui A of the flank branches (1/Pa: dW/dt = A sigma_n V_s exp(-B/T))
    fl_B: np.ndarray            # [nbf] activation temperature (K)
    fl_lo: np.ndarray           # [nbf] lower end of the branch's temperature range (K, ascending)
    rk_A: np.ndarray            # [nbr] rake (crater) branches
    rk_B: np.ndarray
    rk_lo: np.ndarray
    hs_T: np.ndarray            # [nh] substrate hardness table: temperature (K)
    hs_Pa: np.ndarray           # [nh] hardness (Pa) (hot-hardness table, or one value)
    hc_T: np.ndarray            # [nhc] coating hardness table
    hc_Pa: np.ndarray
    scal: np.ndarray            # [N_LS]


class ElemState(NamedTuple):
    """Wear state of one tool's elements (checkpointed)."""

    vn: np.ndarray              # [nt, ne] flank wear depth normal to the machined surface (m)
    vn_notch: np.ndarray        # [nt, ne] extra recession at the depth-of-cut line (m)
    kt: np.ndarray              # [nt, ne] crater depth (m)
    chip_d: np.ndarray          # [nt, ne] depth of the edge chip, along the faces from the edge (m)
    coat_f: np.ndarray          # [nt, ne] coating left on the flank at the edge (m)
    coat_r: np.ndarray          # [nt, ne] coating left in the crater zone (m)
    D_th: np.ndarray            # [nt, ne] thermal-fatigue damage (Miner)
    crack_m: np.ndarray         # [nt, ne] comb-crack depth (m)
    cut_s: np.ndarray           # [nt, ne] accumulated time in cut (s)
    bue_m: np.ndarray           # [nt, ne] BUE height (m)
    used: np.ndarray            # [nt, ne] int8: the element has cut
    chipped: np.ndarray         # [nt, ne] int8 chips of the edge so far (its generation; nonzero: chipped)
    doc_hit: np.ndarray         # [nt, ne] int8: the element has been at the depth-of-cut line
    n_chip: np.ndarray          # [nt, ne] int64 load cycles (entry/exit shock rows) of the edge
    n_coat: np.ndarray          # [nt, ne] int64 (stream wear.coating)
    n_bue: np.ndarray           # [nt, ne] int64 (stream wear.bue_breakoff)
    edge_fat: np.ndarray        # [nt, ne] Miner sum of the edge's load cycles: sum (sigma_n / sigma_0)^(-1/b)


class ElemOut(NamedTuple):
    """Step increments and the bus arrays of one tool."""

    d_vn: np.ndarray            # [nt, ne] flank depth increment of the step
    d_notch: np.ndarray
    d_kt: np.ndarray
    d_D: np.ndarray
    VB: np.ndarray              # [nt, ne] flank wear land (m)
    VB_N: np.ndarray            # [nt, ne] notch wear land at the depth-of-cut line (0 elsewhere)
    radial: np.ndarray          # [nt, ne] radial wear (> 0 recedes the edge from the material)
    axial: np.ndarray           # [nt, ne] axial wear (> 0 along +z_T)
    edge_r: np.ndarray          # [nt, ne] worn hone radius
    doc: np.ndarray             # [nt, ne] int8: at the depth-of-cut line this step
    scal: np.ndarray            # [N_OS]


class ChipLaw(NamedTuple):
    scal: np.ndarray            # [N_CL]


def _f(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def new_geo(nt: int, ne: int) -> ElemGeo:
    return ElemGeo(np.zeros((nt, ne), np.int8), np.zeros((nt, ne), np.int64), *(_f(nt, ne) for _ in range(9)))


def new_input(nt: int, ne: int, nq: int) -> ElemIn:
    return ElemIn(*(_f(nt, ne) for _ in range(7)), _f(nt, ne, nq), _f(nt, ne, nq), _f(nq), _f(nt, ne), _f(nt, ne),
                  _f(nt, ne), _f(N_IS))


def new_law(nbf: int = 1, nbr: int = 1, nh: int = 1, nhc: int = 1) -> Law:
    return Law(_f(nbf), _f(nbf), _f(nbf), _f(nbr), _f(nbr), _f(nbr), _f(nh), _f(nh), _f(nhc), _f(nhc), _f(N_LS))


def new_state(nt: int, ne: int) -> ElemState:
    return ElemState(*(_f(nt, ne) for _ in range(10)), *(np.zeros((nt, ne), np.int8) for _ in range(3)),
                     *(np.zeros((nt, ne), np.int64) for _ in range(3)), _f(nt, ne))


def new_output(nt: int, ne: int) -> ElemOut:
    return ElemOut(*(_f(nt, ne) for _ in range(9)), np.zeros((nt, ne), np.int8), _f(N_OS))


def new_chip_law() -> ChipLaw:
    return ChipLaw(_f(N_CL))


def is_kernel_layout(nt) -> bool:
    """Every array of a layout is writable and C-contiguous (kernel argument rule)."""
    return all(isinstance(a, np.ndarray) and a.flags.c_contiguous and a.flags.writeable for a in nt)


def element_keys(slot: int, nt: int, ne: int) -> np.ndarray:
    """Counter-based hash keys of the elements of a tool slot, [nt, ne] int64."""
    j = np.arange(nt, dtype=np.int64)[:, None]
    k = np.arange(ne, dtype=np.int64)[None, :]
    return np.ascontiguousarray(np.int64(slot) * KEY_SLOT + j * KEY_TOOTH + k)


# ------------------------------------------------------------------------------- Numba types
GEO_T = numba.typeof(new_geo(1, 1))
IN_T = numba.typeof(new_input(1, 1, 1))
LAW_T = numba.typeof(new_law())
STATE_T = numba.typeof(new_state(1, 1))
OUT_T = numba.typeof(new_output(1, 1))
CHIPLAW_T = numba.typeof(new_chip_law())

__all__ = [name for name in dir() if name.isupper() or name in (
    "ElemGeo", "ElemIn", "Law", "ElemState", "ElemOut", "ChipLaw", "new_geo", "new_input", "new_law", "new_state",
    "new_output", "new_chip_law", "is_kernel_layout", "element_keys")]
