"""Kernel layouts of the thermal domain (docs/architecture.md 5.5, docs/models/thermal.md).

Every layout is a NamedTuple of writable C-contiguous arrays; the ``new_*`` builders allocate
them with the exact dtypes the compiled kernels are typed for (cnctwin/thermal/kernels.py).
Scalars that a kernel needs are passed in small float64 ``scal`` arrays indexed by the
constants below, so one compiled signature serves every call.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ------------------------------------------------------------------ faces of a grid cell
# bit b of an exposure / contact mask is face b: -x, +x, -y, +y, -z, +z
FACE_NORMALS = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])


class MatTable(NamedTuple):
    """Temperature-dependent properties of one material on a common temperature grid.

    Piecewise linear in T between the points, the end values held outside (the database
    rule); ``H_J_kg`` is the specific enthalpy at the points relative to 20 degC (exact
    integral of the piecewise-linear cp)."""

    T_K: np.ndarray            # [n] ascending
    k_W_mK: np.ndarray         # [n]
    cp_J_kgK: np.ndarray       # [n]
    H_J_kg: np.ndarray         # [n]
    alpha_1_K: np.ndarray      # [n] mean CTE from 20 degC
    rho_kg_m3: np.ndarray      # [1] density at 20 degC (material configuration)


# ------------------------------------------------------------------ workpiece grid
class WorkGrid(NamedTuple):
    """Finite-volume grid of one workpiece, co-registered with the stock lattice (frame S,
    material configuration). Cell (i, j, k) has its centre at origin + (i + 1/2, j + 1/2, k + 1/2) dx."""

    f: np.ndarray              # [nx, ny, nz] float64 material fraction (cut cells 0 < f < 1)
    h: np.ndarray              # [nx, ny, nz] float64 specific enthalpy (J/kg, rel. 20 degC), 0 where empty
    T: np.ndarray              # [nx, ny, nz] float64 temperature (K), NaN where empty
    contact: np.ndarray        # [nx, ny, nz] uint8 bitmask of faces touching the fixture
    geom: np.ndarray           # [4] origin x, y, z (m, S) and cell size dx (m)


class WorkScratch(NamedTuple):
    """Scratch arrays of the ADI step (allocated once per grid)."""

    src: np.ndarray            # [nx, ny, nz] float64: energy to deposit (J) on entry; scratch after
    a: np.ndarray              # [nx, ny, nz] float64 scratch (delta T of the sweeps)
    cap: np.ndarray            # [nx, ny, nz] float64 heat capacity of the cell (J/K)
    rx: np.ndarray             # [nx, ny, nz] float32 half-cell conduction resistance along x (K/W)
    ry: np.ndarray             # [nx, ny, nz] float32
    rz: np.ndarray             # [nx, ny, nz] float32
    bits: np.ndarray           # [nx, ny, nz] uint8 exposed faces (neighbour empty or outside)
    line_c: np.ndarray         # [max(nx, ny), nz] float64 Thomas coefficients
    line_d: np.ndarray         # [max(nx, ny), nz] float64
    gprev: np.ndarray          # [max(nx, ny, nz) + 1] float64


# indices of WorkBoundary.scal
B_T_AIR = 0          # ambient air (K)
B_T_FLUID = 1        # coolant (K)
B_H_WET = 2          # coolant h on wetted faces away from the cut (W/m2K)
B_ALL_WET = 3        # 1: every exposed face is wetted (coolant on, no wetted grid)
B_EPS = 4            # emissivity
B_T_RAD = 5          # radiation sink (enclosure) temperature (K)
B_H_CONTACT = 6      # part-fixture contact conductance (W/m2K)
B_T_FIX = 7          # fixture temperature (K)
B_FMIN = 8           # cells with f <= fmin are empty
B_COOLANT_ON = 9     # 1 when the coolant flows
B_SIGMA = 10         # Stefan-Boltzmann constant
N_BSCAL = 12


class WorkBoundary(NamedTuple):
    h_air: np.ndarray          # [6] free-convection h per face direction (W/m2K)
    near: np.ndarray           # [n, 5] coolant near-cut zones: x, y, z (S, m), radius (m), h (W/m2K)
    wet: np.ndarray            # [nwx, nwy] int8 wetted columns over the S xy-plane
    wet_geom: np.ndarray       # [3] origin x, y (m, S) and cell (m) of ``wet``
    scal: np.ndarray           # [N_BSCAL] float64, B_* indices


# indices of the work ledger array (kernels write, Python books)
L_SRC = 0            # energy deposited this step (J)
L_AIR = 1            # convective loss to air (J)
L_COOL = 2           # convective loss to coolant (J)
L_RAD = 3            # radiative loss (J)
L_CONTACT = 4        # conduction into the fixture (J)
L_DU = 5             # sum of C dT (J)
L_U0 = 6             # internal energy before (J, rel. 20 degC)
L_U1 = 7             # internal energy after (J)
L_AREA = 8           # exposed (convective) area (m2)
L_TSURF = 9          # area-weighted mean temperature of the exposed surface (K)
L_TMAX = 10          # max cell temperature (K)
L_MASS = 11          # mass (kg)
L_TMEAN = 12         # mass-weighted mean temperature (K)
L_LOST = 13          # source energy found on empty cells (J; booked as unplaced)
N_L = 16


# ------------------------------------------------------------------ tool / holder ladder
class Ladder(NamedTuple):
    """Axial 1-D finite volumes of a tool and its holder, coupled radially over the clamp,
    in a banded node order (bandwidth ``band.shape[1] - 1``)."""

    m: np.ndarray              # [n] float64 node mass (kg)
    h: np.ndarray              # [n] float64 specific enthalpy (J/kg, rel. 20 degC)
    T: np.ndarray              # [n] float64 temperature (K)
    mat: np.ndarray            # [n] int8: 0 tool material, 1 holder material
    area: np.ndarray           # [n] float64 axial cross-section (m2)
    dz: np.ndarray             # [n] float64 node length (m)
    link_a: np.ndarray         # [nl] int32
    link_b: np.ndarray         # [nl] int32
    link_kind: np.ndarray      # [nl] int8: 0 axial conduction (series half cells), 1 fixed conductance
    link_g: np.ndarray         # [nl] float64 fixed conductance (W/K) of kind-1 links
    g_air: np.ndarray          # [n] float64 conductance to air (W/K)
    g_cool: np.ndarray         # [n] float64 conductance to coolant (W/K)
    g_sp: np.ndarray           # [n] float64 conductance to the spindle nose (W/K)
    src: np.ndarray            # [n] float64 heat input this step (J)
    band: np.ndarray           # [n, bw + 1] float64 scratch (banded Cholesky factor)
    rhs: np.ndarray            # [n] float64 scratch
    scal: np.ndarray           # [4] T_air, T_cool, T_spindle (K), unused


LD_SRC = 0
LD_AIR = 1
LD_COOL = 2
LD_SP = 3
LD_DU = 4
LD_U0 = 5
LD_U1 = 6
N_LD = 8


# ------------------------------------------------------------------ cutting-edge local field
class EdgeResponse(NamedTuple):
    """Dimensionless step response of a uniformly heated rectangle 2a x 2b on a half space
    (docs/models/thermal.md "Edge"): theta = (q a / k) Phi(Fo = alpha t / a^2, m = b / a)."""

    log_fo: np.ndarray         # [nF] log10 Fo, uniform
    log_m: np.ndarray          # [nm] log10 m, uniform
    log_phi_c: np.ndarray      # [nm, nF] log10 Phi at the centre
    log_phi_m: np.ndarray      # [nm, nF] log10 Phi averaged over the rectangle
    phi_inf_c: np.ndarray      # [nm] steady centre value
    phi_inf_m: np.ndarray      # [nm] steady mean value


EDGE_NONE = 0
EDGE_INTERRUPTED = 1
EDGE_CONTINUOUS = 2


class EdgeInput(NamedTuple):
    """Per element [nt, ne] inputs of the periodic edge field (Stephenson & Ali 1992)."""

    mode: np.ndarray           # int8 EDGE_*
    q_r: np.ndarray            # rake flux into the tool while in cut (W/m2)
    a_r: np.ndarray            # rake contact length l_c (m): half length of the mirrored source
    q_f: np.ndarray            # flank flux into the tool while in cut (W/m2)
    a_f: np.ndarray            # flank contact width (m)
    b: np.ndarray              # half element width db / 2 (m)
    t_h: np.ndarray            # heating time per cycle (s)
    period: np.ndarray         # cycle period (s)
    n_cyc: np.ndarray          # cycles since the element entered continuous engagement
    t_eng: np.ndarray          # continuous engagement time (s)
    k: np.ndarray              # tool conductivity (W/mK)
    alpha: np.ndarray          # tool diffusivity (m2/s)
    wedge: np.ndarray          # (pi/2) / wedge angle
    r_far: np.ndarray          # far-field radius (m)
    T_body: np.ndarray         # body temperature at the element (K)
    xi: np.ndarray             # [nq] quadrature nodes on (0, 1) of the heating phase
    scal: np.ndarray           # [2] n_direct, n_window


class EdgeOutput(NamedTuple):
    rake_q: np.ndarray         # [nt, ne, nq] mean rake-contact temperature (K)
    peak_q: np.ndarray         # [nt, ne, nq] cutting-edge temperature (K)
    flank_q: np.ndarray        # [nt, ne, nq] mean flank-contact temperature (K)
    amp: np.ndarray            # [nt, ne] cycle amplitude at the cutting edge (K)


# EdgeInput.scal[2]: what edge_temperatures returns
EDGE_OUT_TEMPERATURE = 0   # body temperature + local rise minus the far field the 1-D body holds
EDGE_OUT_DEVIATION = 1     # periodic deviation from the cycle mean (the 3-D field holds the mean), K


# ------------------------------------------------------------------ 3-D cutting-region field (toolfield.py)
class FieldMesh(NamedTuple):
    """Static finite-volume mesh of the cutting region, solved together with the tool ladder
    (docs/models/thermal.md 2.4). Unknowns 0..n-1 are the field cells, n..n+nL-1 the ladder nodes;
    ``row_ptr``/``col`` is the strictly lower CSR pattern of the combined symmetric system (columns
    ascending in each row) and the ``*_slot`` arrays the CSR entry of every off-diagonal coupling."""

    m: np.ndarray              # [n] cell mass (kg), times the multiplicity
    link_a: np.ndarray         # [nl] int32 cell on one side of an internal face
    link_b: np.ndarray         # [nl] int32 the other cell (> link_a)
    link_area: np.ndarray      # [nl] face area (m2), times the multiplicity
    link_da: np.ndarray        # [nl] distance of cell a's centroid from the face (m)
    link_db: np.ndarray        # [nl] same for cell b
    face_cell: np.ndarray      # [nf] int32 cell of a boundary face
    face_area: np.ndarray      # [nf] face area (m2), times the multiplicity
    face_d: np.ndarray         # [nf] distance of the cell centroid from the face (m)
    face_node: np.ndarray      # [nf] int32 ladder node of a coupled face (insert seat, tooth root), -1: exposed
    face_rc: np.ndarray        # [nf] contact resistance of a coupled face (m2K/W)
    face_lk: np.ndarray        # [nf] conduction length on the ladder side (m): resistance face_lk / k(node)
    row_ptr: np.ndarray        # [N + 1] int32
    col: np.ndarray            # [nnz] int32
    link_slot: np.ndarray      # [nl] int32
    face_slot: np.ndarray      # [nf] int32 (-1: exposed face)
    lad_slot: np.ndarray       # [nll] int32 per ladder link (-1: the link is not in the system)
    blk_ptr: np.ndarray        # [nb + 1] int32 contiguous diagonal blocks (the slabs of the field, the ladder) of
                               # the block symmetric Gauss-Seidel preconditioner; each banded within band.shape[1] - 1


class FieldSystem(NamedTuple):
    """State and per-step arrays of the field (written by ToolField, read and updated by the kernel)."""

    h: np.ndarray              # [n] specific enthalpy (J/kg, rel. 20 degC)
    T: np.ndarray              # [n] temperature (K)
    src: np.ndarray            # [n] energy entering the cells this step (J)
    fa_h: np.ndarray           # [nf] film coefficient to air incl. linearised radiation (W/m2K)
    fa_w: np.ndarray           # [nf] face area exposed to air (m2, times the multiplicity)
    fc_h: np.ndarray           # [nf] film coefficient to the coolant (W/m2K)
    fc_w: np.ndarray           # [nf] face area wetted by the coolant (m2, times the multiplicity)
    active: np.ndarray         # [nL] int8 ladder node in the system (0: a view of the field)
    cap: np.ndarray            # [N] scratch: heat capacities (J/K)
    kk: np.ndarray             # [N] scratch: conductivities (W/mK)
    dia: np.ndarray            # [N] scratch: matrix diagonal
    val: np.ndarray            # [nnz] scratch: strictly lower matrix entries
    band: np.ndarray           # [N, bw + 1] scratch: Cholesky factors of the diagonal blocks (band[i, d] = L[i, i - d])
    acc: np.ndarray            # [N] scratch of the preconditioner
    x: np.ndarray              # [N] scratch: solution (temperature change)
    r: np.ndarray              # [N] scratch: residual
    z: np.ndarray              # [N] scratch
    p: np.ndarray              # [N] scratch
    q: np.ndarray              # [N] scratch
    b: np.ndarray              # [N] scratch: right-hand side
    scal: np.ndarray           # [N_FS] FS_* settings


FS_TOL = 0                 # relative residual of the conjugate-gradient solve
FS_MAX_ITER = 1            # iteration limit
N_FS = 4

FD_SRC = 0                 # energy entering (field cells + ladder nodes) (J)
FD_AIR = 1                 # loss to air (J)
FD_COOL = 2                # loss to the coolant (J)
FD_SP = 3                  # loss to the spindle nose / turret (ladder end) (J)
FD_DU = 4                  # sum of C dT (J)
FD_U0 = 5                  # internal energy before (cells + solved ladder nodes) (J, rel. 20 degC)
FD_U1 = 6                  # after
FD_ITER = 7                # conjugate-gradient iterations
FD_RES = 8                 # final relative residual |b - A x| / |b|
FD_COUPLE = 9              # heat from the field into the ladder through the coupled faces (J)
N_FD = 12


# ------------------------------------------------------------------ builders
def f64(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def new_mat_table(T_K, k, cp, H, alpha, rho) -> MatTable:
    return MatTable(*(np.ascontiguousarray(np.asarray(v, dtype=np.float64)).copy()
                      for v in (T_K, k, cp, H, alpha)), np.array([float(rho)]))


def new_work_grid(shape: tuple[int, int, int], origin_S_m, cell_m: float) -> WorkGrid:
    f = f64(shape)
    T = np.full(shape, np.nan)
    geom = np.array([float(origin_S_m[0]), float(origin_S_m[1]), float(origin_S_m[2]), float(cell_m)])
    return WorkGrid(f, f64(shape), T, np.zeros(shape, dtype=np.uint8), geom)


def new_work_scratch(shape: tuple[int, int, int]) -> WorkScratch:
    nx, ny, nz = shape
    line = max(nx, ny)
    return WorkScratch(f64(shape), f64(shape), f64(shape), np.zeros(shape, np.float32),
                       np.zeros(shape, np.float32), np.zeros(shape, np.float32),
                       np.zeros(shape, np.uint8), f64((line, nz)), f64((line, nz)), f64(max(nx, ny, nz) + 1))


def new_work_boundary(n_near: int = 0, wet_shape=(0, 0)) -> WorkBoundary:
    return WorkBoundary(f64(6), f64((n_near, 5)), np.zeros(wet_shape, np.int8), f64(3), f64(N_BSCAL))


def new_edge_input(shape: tuple[int, int], nq: int, out_mode: int | None = None) -> EdgeInput:
    """scal: [0] n_direct, [1] n_window and, when ``out_mode`` is given, [2] the EDGE_OUT_* output (without
    it the kernel returns temperatures)."""
    scal = f64(2) if out_mode is None else np.array([0.0, 0.0, float(out_mode)])
    return EdgeInput(np.zeros(shape, np.int8), *(f64(shape) for _ in range(14)), f64(nq), scal)


def new_field_system(n: int, nf: int, n_ladder: int, nnz: int, bw: int) -> FieldSystem:
    big = n + n_ladder
    return FieldSystem(f64(n), f64(n), f64(n), f64(nf), f64(nf), f64(nf), f64(nf), np.ones(n_ladder, np.int8),
                       f64(big), f64(big), f64(big), f64(nnz), f64((big, bw + 1)), f64(big), f64(big), f64(big),
                       f64(big), f64(big), f64(big), f64(big), f64(N_FS))


def field_mesh_sample() -> FieldMesh:
    i32 = lambda k: np.zeros(k, np.int32)
    return FieldMesh(f64(2), i32(1), i32(1), f64(1), f64(1), f64(1), i32(1), f64(1), f64(1), i32(1), f64(1),
                     f64(1), i32(5), i32(1), i32(1), i32(1), i32(1), i32(3))


def new_edge_output(shape: tuple[int, int], nq: int) -> EdgeOutput:
    return EdgeOutput(f64(shape + (nq,)), f64(shape + (nq,)), f64(shape + (nq,)), f64(shape))


def sample_types():
    """Small instances of every layout, for building the kernels' explicit signatures."""
    mat = new_mat_table([293.15, 400.0], [50.0, 45.0], [460.0, 480.0], [0.0, 1.0], [1e-5, 1e-5], 7850.0)
    grid = new_work_grid((2, 2, 2), (0.0, 0.0, 0.0), 1e-3)
    scratch = new_work_scratch((2, 2, 2))
    bnd = new_work_boundary(1, (1, 1))
    ladder = Ladder(f64(2), f64(2), f64(2), np.zeros(2, np.int8), f64(2), f64(2), np.zeros(1, np.int32),
                    np.zeros(1, np.int32), np.zeros(1, np.int8), f64(1), f64(2), f64(2), f64(2), f64(2),
                    f64((2, 3)), f64(2), f64(4))
    resp = EdgeResponse(f64(2), f64(2), f64((2, 2)), f64((2, 2)), f64(2), f64(2))
    edge_in = new_edge_input((1, 1), 2)
    edge_out = new_edge_output((1, 1), 2)
    return mat, grid, scratch, bnd, ladder, resp, edge_in, edge_out


def field_sample_types():
    """Small instances of the field layouts, for the field kernel's explicit signature."""
    return field_mesh_sample(), new_field_system(2, 1, 2, 1, 1)
