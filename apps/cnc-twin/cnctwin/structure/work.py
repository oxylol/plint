"""One workpiece refresh job: modes, residual compliance and displacement grids of the current material and
supports (docs/models/structure.md sections 3-5). A job reads only its inputs (the geometry at its start, the
supports and the temperatures of bus_prev at its start, the previous solution for the Ritz basis), so it can run
while the loop goes on; the domain applies its result a fixed number of macro steps later.

Voxel parts: :mod:`voxel` (FE), :mod:`thinwall` (walls below ``thin_wall_cells``), :mod:`displacement` (loads).
Long turned parts: :mod:`lathe`. Both publish on the same grid: modes and compliance at cell centres
(``origin + (i + 1/2) cell``, the WorkGrid convention of dynamics), displacements at nodes
(``origin + i cell``, the Displacement convention removal samples).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from ..core.bus import Displacement, WorkModes
from ..core.errors import InputError
from . import displacement as DS
from . import kernels as KN
from . import lathe as LA
from . import modal as MO
from . import params as P
from . import thinwall as TW
from . import voxel as VX


@dataclass
class WorkInputs:
    workpiece: int
    grid: VX.VoxelGrid
    material: object                       # materials.WorkMaterial
    settings: object                       # schema.StructureSettings
    geometry: object                       # protocols.WorkGeometry
    stock_geometry: object                 # the as-supplied stock (initial residual-stress reference)
    form: str                              # stock form (residual-stress profiles)
    stock_lo: np.ndarray
    stock_hi: np.ndarray
    kind: str                              # voxel | beam
    supports: VX.Supports
    lathe: LA.LatheSupports
    clamped: bool
    thermal: tuple | None                  # (origin_S_m, cell_m, temp_K grid) or None
    cylinder: bool = False


@dataclass
class WorkSolution:
    kind: str
    lam: np.ndarray                        # all computed eigenvalues (published + residual)
    vectors: np.ndarray                    # voxel: [n, nnx, nny, nnz, 3] node lattice; beam: [n, 6 n_nodes]
    active: np.ndarray                     # voxel: cells in the model [nx, ny, nz] bool; beam: empty
    modes: WorkModes
    displacement: Displacement
    digest: str                            # supports and boundary-condition identity (Ritz basis validity)
    stats: dict = field(default_factory=dict)

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        m, d = self.modes, self.displacement
        arrays = {prefix + "lam": self.lam, prefix + "vectors": self.vectors, prefix + "active": self.active.astype(np.int8),
                  prefix + "m.freq": m.freq_Hz, prefix + "m.zeta": m.zeta, prefix + "m.origin": m.origin_S_m,
                  prefix + "m.shape": m.shape, prefix + "m.comp": m.static_compliance, prefix + "d.origin": d.origin_S_m,
                  prefix + "d.total": d.total_m, prefix + "d.thermal": d.thermal_m, prefix + "d.clamp": d.clamp_m,
                  prefix + "d.residual": d.residual_m}
        meta = {"kind": self.kind, "digest": self.digest, "stats": _json(self.stats), "m.cell": float(m.cell_m),
                "m.n_wall_modes": int(getattr(m, "n_wall_modes", 0)), "m.model": str(getattr(m, "model", "")),
                "m.version": int(getattr(m, "version", 0)), "d.cell": float(d.cell_m),
                "d.clamped": bool(getattr(d, "clamped", False)), "d.version": int(getattr(d, "version", 0))}
        return arrays, meta

    @classmethod
    def from_arrays(cls, a: dict, meta: dict, prefix: str) -> "WorkSolution":
        g = lambda k: np.ascontiguousarray(np.array(a[prefix + k]))  # noqa: E731
        modes = WorkModes(freq_Hz=g("m.freq"), zeta=g("m.zeta"), origin_S_m=g("m.origin"), cell_m=float(meta["m.cell"]),
                          shape=g("m.shape"), static_compliance=g("m.comp"))
        modes.n_wall_modes = int(meta["m.n_wall_modes"])
        modes.model = str(meta["m.model"])
        modes.version = int(meta["m.version"])
        disp = Displacement(origin_S_m=g("d.origin"), cell_m=float(meta["d.cell"]), total_m=g("d.total"),
                            thermal_m=g("d.thermal"), clamp_m=g("d.clamp"), residual_m=g("d.residual"))
        disp.clamped = bool(meta["d.clamped"])
        disp.version = int(meta["d.version"])
        return cls(meta["kind"], g("lam"), g("vectors"), g("active").astype(bool), modes, disp, meta["digest"],
                   dict(meta["stats"]))


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def digest(inputs: WorkInputs) -> str:
    h = hashlib.sha256()
    h.update(inputs.kind.encode())
    h.update(b"1" if inputs.clamped else b"0")
    for arr in (inputs.supports.points, inputs.supports.k_held):
        h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    h.update(repr(inputs.lathe.to_dict()).encode())
    return h.hexdigest()[:24]


def model_key(inputs: WorkInputs, active: np.ndarray) -> str:
    """Identity of everything a voxel solution's stiffness, mass and boundary conditions depend on: the cells in
    the model, the grid, the material's elastic constants and density, the clamped state and every support array
    (held, locating and locked springs, clamping preloads). Two jobs with the same key solve the same eigenproblem
    and the same clamping and residual-stress systems; only their temperatures can differ."""
    h = hashlib.sha256()
    h.update(inputs.kind.encode())
    h.update(b"1" if inputs.clamped else b"0")
    g = inputs.grid
    h.update(np.asarray(list(g.origin_S_m) + [g.cell_m], dtype=float).tobytes())
    h.update(np.asarray(g.shape, dtype=np.int64).tobytes())
    mat = inputs.material
    h.update(np.asarray([mat.E_Pa, mat.nu, mat.rho_kg_m3], dtype=float).tobytes())
    h.update(np.packbits(np.ascontiguousarray(active, dtype=bool)).tobytes())
    sup = inputs.supports
    for arr in (sup.points, sup.k_held, sup.k_locate, sup.k_locked, sup.preload_N):
        h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    h.update(repr(inputs.lathe.to_dict()).encode())
    return h.hexdigest()[:24]


@dataclass
class WorkCache:
    """Runtime objects of one workpiece's latest voxel model, kept between jobs and never checkpointed: the
    assembled model and its held system (with the multigrid hierarchy), keyed by :func:`model_key`. A job whose
    model is unchanged reuses them; any other job builds them anew (a resumed run rebuilds them from the same
    inputs, so the results do not depend on the cache)."""

    key: str = ""
    model: object = None
    held: tuple | None = None           # (System, spring matrix or None, fixed DOFs or None)

    def take(self, key: str):
        return (self.model, self.held) if (key and key == self.key) else (None, None)

    def keep(self, key: str, model, held) -> None:
        self.key, self.model, self.held = key, model, held


def fill_nodes(field: np.ndarray, ok: np.ndarray, passes: int = 3) -> np.ndarray:
    """Lattice values outside the model filled by the mean of model neighbours (3 x 3 x 3), repeatedly: a
    trilinear sample near the surface then sees the surface's own displacement."""
    out = np.array(field, dtype=float, copy=True)
    have = np.array(ok, dtype=bool, copy=True)
    for _ in range(passes):
        if np.all(have):
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(have.shape)
        padded = np.pad(out * have[..., None], ((1, 1), (1, 1), (1, 1), (0, 0)))
        hp = np.pad(have.astype(float), 1)
        nx, ny, nz = have.shape
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    acc += padded[di:di + nx, dj:dj + ny, dk:dk + nz]
                    cnt += hp[di:di + nx, dj:dj + ny, dk:dk + nz]
        new = (~have) & (cnt > 0)
        out[new] = acc[new] / cnt[new][:, None]
        have = have | new
    return out


def _cell_values(lattice: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """[nx, ny, nz, m] cell-centre values (mean of the cell's model nodes), empty cells filled from neighbours."""
    shape = tuple(s - 1 for s in ok.shape)
    out = np.zeros(shape + (lattice.shape[-1],))
    has = np.zeros(shape, np.int8)
    KN.cell_average(np.ascontiguousarray(lattice, dtype=float), np.ascontiguousarray(ok.astype(np.int8)), out, has)
    if not np.all(has):
        out = fill_nodes(out, has.astype(bool), passes=max(shape))
    return out


# ------------------------------------------------------------------------------- voxel parts
def _active_cells(inputs: WorkInputs):
    """(cells in the model, material fractions, cells dropped): fraction above fill_threshold, face-connected to
    the cells holding support points (or the largest component)."""
    st = inputs.settings
    g = inputs.grid
    fill = np.clip(np.asarray(inputs.geometry.fill_fraction(inputs.workpiece, g.origin, g.cell_m, g.shape), dtype=float),
                   0.0, 1.0)
    active = fill > st.fill_threshold
    seeds = None
    if len(inputs.supports):
        seeds = np.zeros(g.shape, bool)
        idx = np.clip(np.floor((inputs.supports.points - g.origin) / g.cell_m).astype(np.int64), 0,
                      np.array(g.shape) - 1)
        seeds[tuple(idx.T)] = True
        seeds = _dilate(seeds)
    active, dropped = VX.keep_connected(active, seeds)
    return active, fill, dropped


def _model(inputs: WorkInputs, prev: WorkSolution | None):
    active, fill, dropped = _active_cells(inputs)
    mat = inputs.material
    return VX.VoxelModel(inputs.grid, active, mat.E_Pa, mat.nu, mat.rho_kg_m3), fill, dropped


def _dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    import scipy.ndimage as ndi
    return ndi.binary_dilation(mask, iterations=radius)


def _held_system(model: VX.VoxelModel, inputs: WorkInputs):
    """(system, spring matrix or None, fixed DOFs or None): the held part on its support springs, or on a
    minimal 3-2-1 support when it is not clamped (or its springs leave a rigid-body motion free)."""
    if inputs.clamped and len(inputs.supports) and np.any(inputs.supports.k_held != 0.0):
        data = model.spring_data(inputs.supports.points, inputs.supports.k_held)
        Ks = model.pattern_matrix(data)
        R = model.rigid_modes()
        S = R.T @ (Ks @ R)
        lam = np.linalg.eigvalsh(0.5 * (S + S.T))
        if lam[0] > 1e-9 * lam[-1]:
            return VX.System(model, data, None, inputs.settings), Ks, None
    fixed = VX.minimal_constraints(model)
    return VX.System(model, None, fixed, inputs.settings), None, fixed


def _modes(model, system, inputs: WorkInputs, prev: WorkSolution | None, dig: str, stats: dict):
    st = inputs.settings
    n_all = int(st.work_modes + st.work_residual_modes)
    lam = V = None
    if prev is not None and prev.kind == "voxel" and prev.digest == dig and prev.vectors.shape[0] > 0:
        guess = np.stack([model.from_lattice(prev.vectors[k]) for k in range(prev.vectors.shape[0])], axis=1)
        if np.all(np.isfinite(guess)) and np.any(guess != 0.0):
            n = min(n_all, len(system.free) - 2, guess.shape[1])
            lam_r, V_r, res, ok = system.ritz(guess[:, :n], prev.lam[:n], n, st.ritz_tol, int(st.work_modes))
            stats["ritz_residual"] = res
            if ok and n == min(n_all, len(system.free) - 2):
                lam, V = lam_r, V_r
                stats["solve"] = "ritz"
    if lam is None:
        lam, V = system.eigen(n_all, st.eig_tol)
        stats["solve"] = "full"
    return lam, V


def _voxel_modes_on_grid(model, lam, V, Ks, inputs: WorkInputs, fill):
    st = inputs.settings
    g = inputs.grid
    n_pub = int(min(st.work_modes, len(lam)))
    ok = model.node_ok
    shapes = np.zeros((n_pub,) + g.shape + (3,))
    for k in range(n_pub):
        shapes[k] = _cell_values(model.to_lattice(V[:, k]), ok)
    comp = np.zeros(g.shape + (3, 3))
    for k in range(n_pub, len(lam)):
        c = _cell_values(model.to_lattice(V[:, k]), ok)
        comp += np.einsum("...i,...j->...ij", c, c) / lam[k]
    s = VX.support_fraction(model, Ks, lam[:n_pub], V[:, :n_pub])
    eta_w = P.p(P.WORK, "work_loss_factor")
    eta_s = P.p(P.WORK, "support_loss_factor")
    zeta = 0.5 * eta_w * (1.0 - s) + 0.5 * eta_s * s
    freq = MO.freq_Hz(lam[:n_pub])
    walls = TW.detect_walls(fill, model.active, g.origin, g.cell_m, st, inputs.geometry, inputs.workpiece)
    wall_freq, wall_zeta, wall_shapes = [], [], []
    mat = inputs.material
    for wall in walls:
        TW.solve_wall(wall, g.origin, g.cell_m, mat.E_Pa, mat.nu, mat.rho_kg_m3, st)
        idx = tuple(wall.cells.T)
        root = tuple(wall.root_cells.T)
        w = wall.weight
        n = np.zeros(3)
        n[wall.axis] = 1.0
        shapes[(slice(None),) + idx] = (1.0 - w)[None, :, None] * shapes[(slice(None),) + idx] \
            + w[None, :, None] * shapes[(slice(None),) + root]
        comp[idx] = (1.0 - w)[:, None, None] * comp[idx] + w[:, None, None] * (
            comp[root] + wall.residual_m_per_N[:, None, None] * np.outer(n, n)[None, :, :])
        for m in range(len(wall.omega_rad_s)):
            sh = np.zeros(g.shape + (3,))
            sh[idx] = (w * wall.shapes[m])[:, None] * n[None, :]
            wall_freq.append(wall.omega_rad_s[m] / (2.0 * math.pi))
            wall_zeta.append(0.5 * eta_w)
            wall_shapes.append(sh)
    if wall_shapes:
        shapes = np.concatenate([shapes, np.stack(wall_shapes)], axis=0)
        freq = np.concatenate([freq, wall_freq])
        zeta = np.concatenate([zeta, wall_zeta])
    modes = WorkModes(freq_Hz=np.asarray(freq, float), zeta=np.asarray(zeta, float), origin_S_m=g.origin.copy(),
                      cell_m=float(g.cell_m), shape=np.ascontiguousarray(shapes),
                      static_compliance=np.ascontiguousarray(0.5 * (comp + np.swapaxes(comp, -1, -2))))
    modes.n_wall_modes = len(wall_shapes)
    modes.model = "voxel"
    return modes, walls


def _clamp_displacement(model: VX.VoxelModel, inputs: WorkInputs) -> np.ndarray:
    sup = inputs.supports
    if not inputs.clamped or not len(sup) or not np.any(sup.preload_N != 0.0):
        return np.zeros(model.n_dof)
    data = model.spring_data(sup.points, sup.k_locate)
    R = model.rigid_modes()
    S = R.T @ (model.pattern_matrix(data) @ R)
    lam, Q = np.linalg.eigh(0.5 * (S + S.T))
    free_dirs = Q[:, lam <= 1e-9 * max(lam[-1], 1e-300)] if lam[-1] > 0.0 else np.eye(6)
    soft = free_dirs.shape[1] > 0
    if soft:
        Sh = R.T @ (model.spring_matrix(sup.points, sup.k_held) @ R)
        if np.linalg.eigvalsh(0.5 * (Sh + Sh.T))[0] <= 1e-9 * max(np.abs(Sh).max(), 1e-300):
            return np.zeros(model.n_dof)
        data = data + inputs.settings.soft_support_ratio * model.spring_data(sup.points, sup.k_locked)
    system = VX.System(model, data, None, inputs.settings)
    u = system.solve(model.point_loads(sup.points, sup.preload_N))
    if soft:
        Rn = R @ free_dirs
        MR = model.M @ Rn
        u = u - Rn @ np.linalg.solve(Rn.T @ MR, MR.T @ u)
    return u


def _residual_stress(model: VX.VoxelModel, inputs: WorkInputs) -> np.ndarray:
    """Cell-wise initial stress [n_cells, 6] of the as-supplied stock."""
    mat = inputs.material
    if inputs.cylinder:
        radius = 0.5 * float(max(inputs.stock_hi[0] - inputs.stock_lo[0], inputs.stock_hi[1] - inputs.stock_lo[1]))
        return DS.bar_stress(mat, inputs.form, inputs.grid, model.cells, radius)
    sigma, _ = DS.plate_stress(mat, inputs.form, inputs.grid, model.cells, inputs.stock_lo, inputs.stock_hi)
    return sigma


def residual_reference(inputs: WorkInputs) -> np.ndarray:
    """Node lattice response of the as-supplied stock (same kind of boundary conditions): d_residual is relative
    to it, so the stock starts undistorted."""
    g = inputs.grid
    fill = np.clip(np.asarray(inputs.stock_geometry.fill_fraction(inputs.workpiece, g.origin, g.cell_m, g.shape),
                              dtype=float), 0.0, 1.0)
    active = fill > inputs.settings.fill_threshold
    active, _ = VX.keep_connected(active, None)
    mat = inputs.material
    model = VX.VoxelModel(g, active, mat.E_Pa, mat.nu, mat.rho_kg_m3)
    sigma = _residual_stress(model, inputs)
    if not np.any(sigma != 0.0):
        return np.zeros(g.node_shape + (3,))
    system, _, _ = _held_system(model, inputs)
    return model.to_lattice(system.solve(model.initial_stress_load(sigma)))


def _publish_displacement(grid: VX.VoxelGrid, ok: np.ndarray, thermal, clamp, residual, clamped: bool) -> Displacement:
    th = fill_nodes(thermal, ok)
    cl = fill_nodes(clamp, ok)
    rs = fill_nodes(residual, ok)
    d = Displacement(origin_S_m=grid.origin.copy(), cell_m=float(grid.cell_m), total_m=th + cl + rs, thermal_m=th,
                     clamp_m=cl, residual_m=rs)
    d.clamped = bool(clamped)
    return d


def _reusable(prev: WorkSolution | None, key: str) -> bool:
    """Whether the latest solution solved the same model (model_key): its eigenpairs, clamping and residual-stress
    displacements are then this job's too, exactly (the same matrices and loads)."""
    return (prev is not None and prev.kind == "voxel" and prev.stats.get("model_key") == key
            and prev.vectors.shape[0] > 0)


def compute_voxel(inputs: WorkInputs, prev: WorkSolution | None, reference: np.ndarray | None,
                  cache: WorkCache | None = None) -> WorkSolution:
    """One voxel job (module doc). When the model is the latest solution's (``model_key``: the same cells,
    supports, clamped state and material) the modes, the clamping and the residual-stress displacements are that
    solution's and only the thermal displacement is solved (temperatures are the only input that changed); the
    held system comes from ``cache`` when it holds that model (docs/models/structure.md 3.5)."""
    stats: dict = {}
    active, fill, dropped = _active_cells(inputs)
    key = model_key(inputs, active)
    model, held = cache.take(key) if cache is not None else (None, None)
    if model is None:
        mat = inputs.material
        model = VX.VoxelModel(inputs.grid, active, mat.E_Pa, mat.nu, mat.rho_kg_m3)
    stats.update({"dof": model.n_dof, "cells": int(len(model.cells)), "dropped_cells": dropped,
                  "cell_m": float(inputs.grid.cell_m), "model_key": key})
    dig = digest(inputs)
    if held is None:
        held = _held_system(model, inputs)
    system, Ks, fixed = held
    if cache is not None:
        cache.keep(key, model, held)
    stats["solver"] = system.solver.kind
    iterations0 = int(system.solver.iterations)
    reuse = _reusable(prev, key)
    if reuse:
        lam = np.asarray(prev.lam, float).copy()
        V = np.stack([model.from_lattice(prev.vectors[k]) for k in range(prev.vectors.shape[0])], axis=1)
        stats["solve"] = "reuse"
    else:
        lam, V = _modes(model, system, inputs, prev, dig, stats)
    modes, walls = _voxel_modes_on_grid(model, lam, V, Ks, inputs, fill)
    stats["walls"] = [w.to_dict() for w in walls]
    T = DS.cell_temperatures(inputs.grid, model.cells, *(inputs.thermal if inputs.thermal is not None else
                                                         (np.zeros(3), 0.0, None)))
    strain = DS.thermal_strain(inputs.material, T)
    u_th = system.solve(model.thermal_load(strain)) if np.any(strain != 0.0) else np.zeros(model.n_dof)
    if reuse:
        # the same model and supports: the clamping and residual-stress solves would repeat the latest solution's
        # (published lattices: re-filling them below gives the same values)
        clamp_lattice = prev.displacement.clamp_m
        u_rs = prev.displacement.residual_m
    else:
        clamp_lattice = model.to_lattice(_clamp_displacement(model, inputs))
        sigma = _residual_stress(model, inputs)
        if np.any(sigma != 0.0):
            u = system.solve(model.initial_stress_load(sigma))
            if reference is not None:
                u = u - model.from_lattice(reference)
                if fixed is not None:
                    # free state: the reference's own 3-2-1 support differs; restore the current one (rigid motion)
                    R = model.rigid_modes()
                    u = u - R @ np.linalg.solve(R[fixed], u[fixed])
            u_rs = model.to_lattice(u)
        else:
            u_rs = np.zeros(inputs.grid.node_shape + (3,))
    stats["cg_iterations"] = int(system.solver.iterations) - iterations0
    stats["max_T_K"] = float(np.max(T)) if len(T) else 0.0
    disp = _publish_displacement(inputs.grid, model.node_ok, model.to_lattice(u_th), clamp_lattice, u_rs,
                                 inputs.clamped)
    if reuse:
        vectors = prev.vectors
    else:
        vectors = np.stack([model.to_lattice(V[:, k]) for k in range(V.shape[1])]) if V.shape[1] else \
            np.zeros((0,) + inputs.grid.node_shape + (3,))
    return WorkSolution("voxel", np.asarray(lam, float), vectors, model.active.copy(), modes, disp, dig, stats)


# ------------------------------------------------------------------------------- turned parts (beam)
def compute_beam(inputs: WorkInputs, prev: WorkSolution | None) -> WorkSolution:
    st = inputs.settings
    g = inputs.grid
    mat = inputs.material
    model = LA.build_lathe(inputs.geometry, inputs.workpiece, inputs.stock_lo, inputs.stock_hi, mat, inputs.lathe, st)
    n_all = int(st.work_modes + st.work_residual_modes)
    lam, V = LA.lathe_modes(model, n_all)
    n_pub = int(min(st.work_modes, len(lam)))
    centres = g.cell_centres().reshape(-1, 3)
    shapes = np.zeros((n_pub,) + g.shape + (3,))
    for k in range(n_pub):
        shapes[k] = LA.field_at(model, V[:, k], centres).reshape(g.shape + (3,))
    C6 = LA.static_compliance6(model)
    nn = len(model.node_z_m)
    modal6 = np.zeros((nn, 6, 6))
    for k in range(n_pub):
        phi = V[:, k].reshape(nn, 6)
        modal6 += np.einsum("ni,nj->nij", phi, phi) / lam[k]
    comp = LA.compliance_at(model, C6 - modal6, centres).reshape(g.shape + (3, 3))
    Ks = model.K_springs
    s = np.einsum("ij,ij->j", V[:, :n_pub], Ks @ V[:, :n_pub]) / np.where(lam[:n_pub] > 0, lam[:n_pub], 1.0)
    zeta = 0.5 * P.p(P.WORK, "work_loss_factor") * (1.0 - s) + 0.5 * P.p(P.WORK, "support_loss_factor") * s
    modes = WorkModes(freq_Hz=MO.freq_Hz(lam[:n_pub]), zeta=zeta, origin_S_m=g.origin.copy(), cell_m=float(g.cell_m),
                      shape=shapes, static_compliance=0.5 * (comp + np.swapaxes(comp, -1, -2)))
    modes.n_wall_modes = 0
    modes.model = "beam"
    # displacements on the node lattice
    nodes = g.node_positions(np.argwhere(np.ones(g.node_shape, bool)))
    sl = model.slices
    zero = np.zeros(g.node_shape + (3,))
    thermal = zero.copy()
    if inputs.thermal is not None:
        cells_f = np.argwhere(sl.fill >= 0.0)
        fine_grid = VX.VoxelGrid(tuple(sl.origin), sl.cell_m, sl.fill.shape)
        T = DS.cell_temperatures(fine_grid, cells_f, *inputs.thermal).reshape(sl.fill.shape)
        eps = DS.thermal_strain(mat, T)
        e0, kx, ky = LA.slice_eigenstrains(model, eps)
        u = LA.solve_held(model, LA.eigen_loads(model, e0, kx, ky))
        thermal = LA.field_at(model, u, nodes)
        # radial expansion of the section with its mean temperature
        mean_eps = np.array([float(np.average(eps[:, :, idx], weights=np.maximum(sl.fill[:, :, idx], 1e-300)))
                             for idx in model.element_slices])
        z_el = 0.5 * (model.node_z_m[:-1] + model.node_z_m[1:])
        e_r = np.interp(nodes[:, 2], z_el, mean_eps)
        r = nodes - np.array([model.axis_xy_m[0], model.axis_xy_m[1], 0.0])
        r[:, 2] = 0.0
        thermal = (thermal + e_r[:, None] * r).reshape(g.node_shape + (3,))
    residual = zero.copy()
    radius = 0.5 * float(max(inputs.stock_hi[0] - inputs.stock_lo[0], inputs.stock_hi[1] - inputs.stock_lo[1]))
    cells_f = np.argwhere(np.ones(sl.fill.shape, bool))
    fine_grid = VX.VoxelGrid(tuple(sl.origin), sl.cell_m, sl.fill.shape)
    sigma = DS.bar_stress(mat, inputs.form, fine_grid, cells_f, radius)
    if np.any(sigma[:, 2] != 0.0):
        eps = (-sigma[:, 2] / mat.E_Pa).reshape(sl.fill.shape)
        e0, kx, ky = LA.slice_eigenstrains(model, eps)
        u = LA.solve_held(model, LA.eigen_loads(model, e0, kx, ky))
        residual = LA.field_at(model, u, nodes).reshape(g.node_shape + (3,))
    disp = Displacement(origin_S_m=g.origin.copy(), cell_m=float(g.cell_m), total_m=thermal + residual,
                        thermal_m=thermal, clamp_m=zero.copy(), residual_m=residual)
    disp.clamped = bool(inputs.clamped)
    stats = {"dof": int(len(model.free)), "elements": len(model.elements), "solve": "dense", "notes": model.notes}
    return WorkSolution("beam", np.asarray(lam, float), V.T.copy(), np.zeros((0, 0, 0), bool), modes, disp,
                        digest(inputs), stats)


def compute(inputs: WorkInputs, prev: WorkSolution | None, reference: np.ndarray | None,
            cache: WorkCache | None = None) -> WorkSolution:
    if inputs.kind == "beam":
        return compute_beam(inputs, prev)
    if inputs.kind != "voxel":
        raise InputError(f"structure: unknown workpiece model {inputs.kind!r}")
    return compute_voxel(inputs, prev, reference, cache)


__all__ = ["WorkInputs", "WorkSolution", "WorkCache", "compute", "compute_voxel", "compute_beam", "digest",
           "model_key", "fill_nodes", "residual_reference"]
