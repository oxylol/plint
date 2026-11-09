"""Modal model of one system: the active tool of a controller path against its workpiece (L2, docs/models/dynamics.md
section 1).

Sources of the modes (all mass-normalised, ``u = phi q``):

* **tool modes** - structure's ToolModes of the slot (tool + holder + spindle, 6-component shapes at its stations
  in T0), mapped onto the EdgeModel's stations by linear interpolation in z (rigid extrapolation with the end
  station's rotation outside structure's range); its residual flexibility mapped the same way;
* **fallback tool modes** - until structure publishes the slot's modes: the machine's spindle FRF modes (rotating
  tools; direction in the spindle-nose frame) or turret modes (static tools; direction in M), the tool rigid on
  them (``phi = d / sqrt(m)``, ``m = k / omega^2``);
* **machine modes** - machine.structure.modes and table_modes (relative tool-workpiece translations along a
  direction in M) and rotary_modes (torsion of a rotary axis: relative translation ``-/+ (e x r) / sqrt(I)`` at
  each station, ``I = k_theta / omega^2``, the axis line placed by the product of exponentials of the upstream
  axes at the step's axis values);
* **workpiece modes** - structure's WorkModes on its grid (shapes and residual compliance per cell, S),
  evaluated at the contact point (refreshed by cnctwin.dynamics.kernels.refresh_work_shapes).

Base excitation (machine and fallback modes, which have a modal mass): modal force ``-sqrt(m) d . a_rel`` with
``a_rel`` the rigid-body acceleration of the tool side relative to the workpiece side (M), and ``-sqrt(I) alpha``
for rotary modes. Unbalance: ``P_k = conj(phi_c) U_c + i conj(psi_c) C_c`` at the unbalance height.

:func:`project_basis` maps the modal state onto a new basis at the active contact points (tracked modes keep their
coordinates; the rest by least squares, minimum energy where underdetermined) and gives the energy difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from . import discretize as DZ
from .abi import (BASE_LINEAR, KIND_MACHINE, KIND_TOOL, KIND_WORK, MAX_ROTARY, MI_HAS_TOOL_RES, MI_IPO_TICKS,
                  MI_L0_TICKS, MI_N_NONWORK, MI_REF_STATION, MI_RELAX, MI_ROTATING, MS_DT_L0_S, MS_DT_TICK_S,
                  MS_TICK_S, N_BASE, DynSystem, WorkGrid, empty_dyn_accum, empty_dyn_model, empty_dyn_state,
                  empty_deflection, empty_modal_model, empty_modal_state, N_SCRATCH, all_stations)

AXIS_LETTERS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class Mode:
    """One mode of a system."""

    kind: int                               # KIND_TOOL / KIND_MACHINE / KIND_WORK
    omega_rad_s: float
    zeta: float
    source: str                             # structure | spindle_frf | turret | frame | table | rotary | work
    label: str
    mass_kg: float = 0.0                    # modal mass (kg or kg m2) of machine / fallback modes, 0 otherwise
    stations: np.ndarray | None = None      # [nst, 6] T0 (tool and machine modes)
    at_unbalance: np.ndarray | None = None  # [6] T0 at the unbalance height
    work_row: int = -1                      # row of the WorkGrid (work modes)
    base: np.ndarray = field(default_factory=lambda: np.zeros(N_BASE))
    key: tuple = ()                         # identity for basis tracking

    @property
    def freq_Hz(self) -> float:
        return self.omega_rad_s / (2.0 * math.pi)


def _vec(value) -> np.ndarray:
    if value is None:
        return np.zeros(3)
    if isinstance(value, str):
        if value.lower() not in AXIS_LETTERS:
            raise InputError(f"mode direction {value!r} is not x, y, z or a vector")
        return np.array(AXIS_LETTERS[value.lower()])
    v = np.array([float(c) for c in value], dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise InputError("mode direction is the zero vector")
    return v / n


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        try:
            value = obj[name]
        except (KeyError, TypeError, IndexError):
            value = getattr(obj, name, default)
    return default if value is None else value


def _num(obj, name, default=None):
    value = _get(obj, name, None)
    if value is None:
        return default
    return float(value)


# ------------------------------------------------------------------------------- station mapping
def station_map(z_from: np.ndarray, z_to: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation weights W [n_to, n_from] and the extrapolation offsets dz [n_to] (z_to - z_end for
    stations outside [z_from[0], z_from[-1]], 0 inside) of a map from structure's stations to the edge stations."""
    zf = np.asarray(z_from, float)
    zt = np.asarray(z_to, float)
    order = np.argsort(zf, kind="stable")
    zs = zf[order]
    W = np.zeros((len(zt), len(zf)))
    dz = np.zeros(len(zt))
    for i, z in enumerate(zt):
        if len(zs) == 1:
            W[i, order[0]] = 1.0
            dz[i] = z - zs[0]
            continue
        if z <= zs[0]:
            W[i, order[0]] = 1.0
            dz[i] = z - zs[0]
        elif z >= zs[-1]:
            W[i, order[-1]] = 1.0
            dz[i] = z - zs[-1]
        else:
            j = int(np.searchsorted(zs, z, side="right")) - 1
            j = min(max(j, 0), len(zs) - 2)
            t = (z - zs[j]) / (zs[j + 1] - zs[j]) if zs[j + 1] > zs[j] else 0.0
            W[i, order[j]] = 1.0 - t
            W[i, order[j + 1]] = t
    return W, dz


def map_shape(shape: np.ndarray, W: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """[n_from, 6] -> [n_to, 6]: interpolated, then outside the range moved rigidly with the end rotation
    (dx += dz ry, dy -= dz rx: a rotation r_y moves a point dz higher by +dz in x, r_x by -dz in y)."""
    out = W @ np.asarray(shape, float)
    out[:, 0] += dz * out[:, 4]
    out[:, 1] -= dz * out[:, 3]
    return out


def map_residual(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Residual flexibility [3 n_from, 3 n_from] -> [3 n_to, 3 n_to] (translations, nearest end outside)."""
    P = np.kron(W, np.eye(3))
    return P @ np.asarray(R, float) @ P.T


# ------------------------------------------------------------------------------- kinematics for rotary modes
def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rotary_axis_line(machine, axis_name: str, axis_values: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(point [3], direction [3] in M, side) of a rotary axis at the given axis values: the chains' joints are
    screws in the base frame at home (product of exponentials, Murray, Li & Sastry 1994); the axis line is moved by
    the joints upstream of it. side = -1 for the workpiece chain, +1 for the tool chain."""
    kin = _get(machine, "kinematics")
    for chain, side in (("workpiece_chain", -1), ("tool_chain", 1)):
        links = _get(kin, chain) or []
        T = np.eye(4)
        for link in links:
            name = _get(link, "axis")
            direction = _get(link, "direction")
            if name == axis_name:
                e = _vec(direction)
                p = np.asarray([float(c) for c in (_get(link, "axis_point_m") or (0.0, 0.0, 0.0))])
                return T[:3, :3] @ p + T[:3, 3], T[:3, :3] @ e, side
            if name is None or direction is None:
                continue
            value = float(axis_values.get(name, 0.0))
            e = _vec(direction)
            point = _get(link, "axis_point_m")
            axis = (_get(machine, "axes") or {})
            kind = _get(_get(axis, name), "type", "linear")
            G = np.eye(4)
            if kind == "rotary":
                p = np.asarray([float(c) for c in (point or (0.0, 0.0, 0.0))])
                R = _rot(e, value)
                G[:3, :3] = R
                G[:3, 3] = p - R @ p
            else:
                G[:3, 3] = e * value
            T = T @ G
    raise InputError(f"rotary mode of axis {axis_name!r}: the axis is in neither kinematic chain of the machine")


# ------------------------------------------------------------------------------- mode sources
def tool_modes_from_structure(tm, edge_z: np.ndarray, z_unbalance: float, version: int):
    """Modes and residual from a bus ToolModes (structure)."""
    freq = np.asarray(_get(tm, "freq_Hz", np.zeros(0)), float)
    zeta = np.asarray(_get(tm, "zeta", np.zeros(0)), float)
    shape = np.asarray(_get(tm, "shape", np.zeros((0, 0, 6))), float)
    z_s = np.asarray(_get(tm, "station_z_m", np.zeros(0)), float)
    if len(freq) == 0:
        return [], None
    if shape.ndim != 3 or shape.shape[0] != len(freq) or shape.shape[2] != 6 or shape.shape[1] != len(z_s):
        raise InputError(f"structure ToolModes: shape {shape.shape} does not match {len(freq)} modes x "
                         f"{len(z_s)} stations x 6")
    W, dz = station_map(z_s, edge_z)
    Wu, dzu = station_map(z_s, np.array([z_unbalance]))
    modes = []
    for k in range(len(freq)):
        w = 2.0 * math.pi * float(freq[k])
        DZ.check_mode(w, float(zeta[k]), f"tool mode {k}")
        st = map_shape(shape[k], W, dz)
        at_u = map_shape(shape[k], Wu, dzu)[0]
        modes.append(Mode(KIND_TOOL, w, float(zeta[k]), "structure", f"tool {k + 1} ({freq[k]:.0f} Hz)",
                          stations=st, at_unbalance=at_u, key=("structure", int(version), k)))
    R = np.asarray(_get(tm, "static_residual_m_per_N", np.zeros((0, 0))), float)
    residual = map_residual(R, W) if R.size and R.shape == (3 * len(z_s), 3 * len(z_s)) else None
    return modes, residual


def _rigid_modes(entries, source: str, nst: int, R_M_T0: np.ndarray, frame: str):
    """Modes of a rigid tool on flange / turret modes: phi = d / sqrt(m) at every station."""
    modes = []
    for i, md in enumerate(entries or []):
        fn = _num(md, "fn_Hz")
        zeta = _num(md, "zeta")
        k = _num(md, "k_N_per_m")
        if fn is None or zeta is None or k is None:
            raise InputError(f"{source} mode {i}: needs fn_Hz, zeta and k_N_um")
        w = 2.0 * math.pi * fn
        DZ.check_mode(w, zeta, f"{source} mode {i}")
        m = k / (w * w)
        d = _vec(_get(md, "direction", "x"))
        d_T0 = d if frame == "T0" else R_M_T0.T @ d
        d_M = R_M_T0 @ d_T0
        st = np.zeros((nst, 6))
        st[:, :3] = d_T0 / math.sqrt(m)
        base = np.zeros(N_BASE)
        base[:3] = -math.sqrt(m) * d_M
        modes.append(Mode(KIND_TOOL, w, zeta, source, f"{source} {i + 1} ({fn:.0f} Hz)", mass_kg=m, stations=st,
                          at_unbalance=st[0].copy(), base=base, key=(source, i)))
    return modes


def fallback_tool_modes(machine, spindle_id: str, rotating: bool, nst: int, R_M_T0: np.ndarray):
    """Spindle FRF modes (rotating tools, directions in the spindle-nose frame) or turret modes (static tools,
    directions in M) carrying a rigid tool."""
    if rotating:
        for sp in _get(machine, "spindles") or []:
            if str(_get(sp, "id", "")) == str(spindle_id):
                return _rigid_modes(_get(sp, "frf_modes"), "spindle_frf", nst, R_M_T0, "T0")
        spindles = _get(machine, "spindles") or []
        return _rigid_modes(_get(spindles[0], "frf_modes") if spindles else [], "spindle_frf", nst, R_M_T0, "T0")
    return _rigid_modes(_get(_get(machine, "structure"), "turret_modes"), "turret", nst, R_M_T0, "M")


def machine_modes(machine, station_M: np.ndarray, R_M_T0: np.ndarray, axis_values: dict, rotary_columns: dict):
    """Frame, table and rotary modes of the machine file at the step's pose (station_M [nst, 3]: station axis
    points in M)."""
    structure = _get(machine, "structure")
    if structure is None:
        return []
    nst = station_M.shape[0]
    modes = []
    for group, source in (("modes", "frame"), ("table_modes", "table")):
        for i, md in enumerate(_get(structure, group) or []):
            fn = _num(md, "fn_Hz")
            zeta = _num(md, "zeta")
            k = _num(md, "k_N_per_m")
            if fn is None or zeta is None or k is None:
                raise InputError(f"machine structure.{group}[{i}]: needs fn_Hz, zeta and k_N_um")
            w = 2.0 * math.pi * fn
            DZ.check_mode(w, zeta, f"machine structure.{group}[{i}]")
            m = k / (w * w)
            d = _vec(_get(md, "direction", "x"))
            st = np.zeros((nst, 6))
            st[:, :3] = (R_M_T0.T @ d) / math.sqrt(m)
            base = np.zeros(N_BASE)
            base[:3] = -math.sqrt(m) * d
            modes.append(Mode(KIND_MACHINE, w, zeta, source, f"{source} {i + 1} ({fn:.0f} Hz)", mass_kg=m,
                              stations=st, at_unbalance=st[0].copy(), base=base, key=(source, i)))
    for i, md in enumerate(_get(structure, "rotary_modes") or []):
        fn = _num(md, "fn_Hz")
        zeta = _num(md, "zeta")
        k = _num(md, "k_Nm_per_rad")
        name = _get(md, "axis")
        if fn is None or zeta is None or k is None or name is None:
            raise InputError(f"machine structure.rotary_modes[{i}]: needs axis, fn_Hz, zeta and k_Nm_rad")
        w = 2.0 * math.pi * fn
        DZ.check_mode(w, zeta, f"machine structure.rotary_modes[{i}]")
        inertia = k / (w * w)
        p, e, side = rotary_axis_line(machine, str(name), axis_values)
        st = np.zeros((nst, 6))
        for s in range(nst):
            st[s, :3] = R_M_T0.T @ (side * np.cross(e, station_M[s] - p)) / math.sqrt(inertia)
            st[s, 3:] = R_M_T0.T @ (side * e) / math.sqrt(inertia)
        base = np.zeros(N_BASE)
        col = rotary_columns.get(str(name), -1)
        if 0 <= col < MAX_ROTARY:
            base[BASE_LINEAR + col] = -math.sqrt(inertia)
        modes.append(Mode(KIND_MACHINE, w, zeta, "rotary", f"rotary {name} ({fn:.0f} Hz)", mass_kg=inertia,
                          stations=st, at_unbalance=st[0].copy(), base=base, key=("rotary", i)))
    return modes


def work_modes(wm, version: int, R_S_T0: np.ndarray, extra_S: np.ndarray | None):
    """Work modes and their WorkGrid from a bus WorkModes (None: no modes, only extra_S)."""
    freq = np.asarray(_get(wm, "freq_Hz", np.zeros(0)), float) if wm is not None else np.zeros(0)
    extra = np.zeros((3, 3)) if extra_S is None else np.asarray(extra_S, float)
    if len(freq) == 0:
        grid = WorkGrid(np.zeros((1, 1, 1, 1, 3)), np.zeros((1, 1, 1, 3, 3)), np.zeros(3), np.zeros(1),
                        np.ascontiguousarray(R_S_T0, float), np.ascontiguousarray(extra))
        return [], grid
    zeta = np.asarray(_get(wm, "zeta"), float)
    shape = np.ascontiguousarray(np.asarray(_get(wm, "shape"), float))
    if shape.ndim != 5 or shape.shape[0] != len(freq) or shape.shape[4] != 3:
        raise InputError(f"structure WorkModes: shape {shape.shape} does not match {len(freq)} modes x nx x ny x nz x 3")
    comp = np.asarray(_get(wm, "static_compliance", np.zeros((0, 0, 0, 3, 3))), float)
    if comp.shape[:3] != shape.shape[1:4] or comp.shape[3:] != (3, 3):
        comp = np.zeros((1, 1, 1, 3, 3))
    grid = WorkGrid(shape, np.ascontiguousarray(comp), np.asarray(_get(wm, "origin_S_m"), float).copy(),
                    np.array([float(_get(wm, "cell_m", 0.0))]), np.ascontiguousarray(R_S_T0, float),
                    np.ascontiguousarray(extra))
    modes = []
    for k in range(len(freq)):
        w = 2.0 * math.pi * float(freq[k])
        DZ.check_mode(w, float(zeta[k]), f"workpiece mode {k}")
        modes.append(Mode(KIND_WORK, w, float(zeta[k]), "work", f"work {k + 1} ({freq[k]:.0f} Hz)", work_row=k,
                          key=("work", int(version), k)))
    return modes, grid


# ------------------------------------------------------------------------------- the system arrays
@dataclass
class Layout:
    """Static data of a built system (kept by the domain; arrays live in the DynSystem)."""

    modes: list
    nst: int
    ref_station: int
    rotating: bool
    grid: WorkGrid
    l0_ticks: int
    ipo_ticks: int


def order_modes(modes: list) -> list:
    """Tool modes, then machine modes, then workpiece modes (the tool_shape row convention)."""
    return [m for m in modes if m.kind == KIND_TOOL] + [m for m in modes if m.kind == KIND_MACHINE] + \
        [m for m in modes if m.kind == KIND_WORK]


def unbalance_coefficients(modes: list, U_T_kg_m, C_T_kg_m2) -> np.ndarray:
    """[nm, 2] Re, Im of P_k = conj(phi_c) U_c + i conj(psi_c) C_c (shapes at the unbalance height, T0)."""
    U = np.asarray(U_T_kg_m, float).reshape(3)
    C = np.asarray(C_T_kg_m2, float).reshape(3)
    Uc = complex(U[0], U[1])
    Cc = complex(C[0], C[1])
    out = np.zeros((len(modes), 2))
    for k, m in enumerate(modes):
        if m.kind == KIND_WORK or m.at_unbalance is None:
            continue
        a = m.at_unbalance
        P = complex(a[0], -a[1]) * Uc + 1j * complex(a[3], -a[4]) * Cc
        out[k] = (P.real, P.imag)
    return out


def contact_shapes(modes: list, ref_station: int, grid: WorkGrid, work_shape: np.ndarray) -> np.ndarray:
    """[nm, 3] relative displacement shape at the reference station (T0)."""
    out = np.zeros((len(modes), 3))
    for k, m in enumerate(modes):
        if m.kind == KIND_WORK:
            out[k] = -(grid.R_S_T0.T @ work_shape[m.work_row])
        else:
            out[k] = m.stations[ref_station, :3]
    return out


def relaxation(modes: list, contact_shape: np.ndarray, G1q: np.ndarray, K_c_T0: np.ndarray | None) -> np.ndarray:
    """(I + K_m diag(G1[:, 0]))^-1 with the modal cutting stiffness K_m = -Psi^T K_c Psi (Psi = contact shapes,
    K_c = d F_tool / d u_rel, T0); the identity without an estimate."""
    nm = len(modes)
    if K_c_T0 is None or nm == 0:
        return np.eye(max(nm, 1))
    Psi = np.asarray(contact_shape, float).T          # [3, nm]
    Km = -Psi.T @ np.asarray(K_c_T0, float) @ Psi
    A = np.eye(nm) + Km @ np.diag(np.asarray(G1q, float))
    return np.ascontiguousarray(np.linalg.inv(A))


def build_system(modes: list, *, nst: int, ref_station: int, residual: np.ndarray | None, grid: WorkGrid,
                 point_S: np.ndarray, l0_ticks: int, ipo_ticks: int, tick_s: float, n_ticks: int, capacity: int,
                 rotating: bool, unbalance_T: tuple | None, K_c_T0: np.ndarray | None,
                 R_S_T0: np.ndarray, discretise=None) -> tuple[DynSystem, Layout]:
    """All kernel arrays of one system for the coming macro step (state zero; the caller copies the carried
    state in). ``discretise(omega, zeta, dt) -> (Phi, G0, G1, gram_D, gram_W)`` (default: discretize.modal_discretisation;
    the domain passes a cached one)."""
    modes = order_modes(modes)
    nm = len(modes)
    nm_alloc = max(nm, 1)
    nonwork = [m for m in modes if m.kind != KIND_WORK]
    work = [m for m in modes if m.kind == KIND_WORK]
    nmw = max(int(grid.shape.shape[0]), 1)
    model = empty_modal_model(nm_alloc, nst, nmw)
    state = empty_modal_state(nm_alloc)
    dm = empty_dyn_model(nm_alloc)
    dt = l0_ticks * tick_s
    omega = np.array([m.omega_rad_s for m in modes]) if nm else np.array([1.0])
    zeta = np.array([m.zeta for m in modes]) if nm else np.array([0.5])
    Phi, G0, G1, gD, gW = (discretise or DZ.modal_discretisation)(omega, zeta, dt)
    if nm == 0:
        Phi[:] = 0.0
        G0[:] = 0.0
        G1[:] = 0.0
        gD[:] = 0.0
        gW[:] = 0.0
    model.Phi[:] = Phi
    model.G0[:] = G0
    model.G1[:] = G1
    model.dt_s[0] = dt
    rows = np.full(nm_alloc, -1, np.int32)
    tool_shape = np.zeros((max(len(nonwork), 1), nst, 6))
    r_tool = 0
    for k, m in enumerate(modes):
        if m.kind == KIND_WORK:
            model.is_work[k] = 1
            model.work_index[k] = m.work_row
            rows[k] = m.work_row
        else:
            model.is_work[k] = 2 if m.kind == KIND_MACHINE else 0
            tool_shape[r_tool] = m.stations
            rows[k] = r_tool
            r_tool += 1
    model = model._replace(tool_shape=np.ascontiguousarray(tool_shape))
    if residual is not None and residual.shape == (3 * nst, 3 * nst):
        model.static_residual[:] = residual
    dm.omega_rad_s[:] = omega
    dm.zeta[:] = zeta
    dm.gram_D[:] = gD
    dm.gram_W[:] = gW
    dm.row[:] = rows
    dm.R_S_T0[:] = R_S_T0
    for k, m in enumerate(modes):
        dm.base_proj[k] = m.base
    if unbalance_T is not None and rotating and nm:
        dm.unb[:] = unbalance_coefficients(modes, unbalance_T[0], unbalance_T[1])
    dm.iscal[MI_REF_STATION] = int(ref_station)
    dm.iscal[MI_N_NONWORK] = len(nonwork)
    dm.iscal[MI_L0_TICKS] = int(l0_ticks)
    dm.iscal[MI_IPO_TICKS] = int(ipo_ticks)
    dm.iscal[MI_ROTATING] = 1 if rotating else 0
    dm.iscal[MI_HAS_TOOL_RES] = 1 if (residual is not None and np.any(residual != 0.0)) else 0
    dm.scal[MS_DT_L0_S] = dt
    dm.scal[MS_DT_TICK_S] = ipo_ticks * tick_s
    dm.scal[MS_TICK_S] = tick_s
    nm_ = nm_alloc
    sys = DynSystem(model, state, dm, empty_dyn_state(nm_, nst, capacity), empty_dyn_accum(n_ticks),
                    empty_deflection(nst), np.zeros((nm_, 2)), np.zeros((2, nm_)), np.zeros(nm_), np.zeros(N_SCRATCH),
                    all_stations())
    from .reference import refresh_work_shapes_ref
    if work:
        refresh_work_shapes_ref(sys, grid, np.asarray(point_S, float))
    else:
        dm.res_work_S[:] = grid.extra_S
        from .abi import MI_HAS_WORK_RES
        dm.iscal[MI_HAS_WORK_RES] = int(np.any(grid.extra_S != 0.0))
    dm.contact_shape[:] = 0.0
    if nm:
        dm.contact_shape[:nm] = contact_shapes(modes, ref_station, grid, sys.model.work_shape_at_contact)
        relax = relaxation(modes, dm.contact_shape[:nm], G1[:, 0], K_c_T0)
        dm.relax[:] = relax
        dm.iscal[MI_RELAX] = 0 if np.array_equal(relax, np.eye(nm)) else 1
    layout = Layout(modes, nst, int(ref_station), bool(rotating), grid, int(l0_ticks), int(ipo_ticks))
    return sys, layout


# ------------------------------------------------------------------------------- tip compliance and the L0 rule
def tip_compliance(modes: list, ref_station: int, grid: WorkGrid | None = None,
                   work_shape: np.ndarray | None = None) -> np.ndarray:
    """Static compliance (m/N, trace of the 3x3 driving-point block) each mode carries at the reference station."""
    out = np.zeros(len(modes))
    for k, m in enumerate(modes):
        if m.kind == KIND_WORK:
            if work_shape is None:
                continue
            phi = work_shape[m.work_row]
        else:
            phi = m.stations[ref_station, :3]
        out[k] = float(phi @ phi) / (m.omega_rad_s ** 2)
    return out


def f_max_resolved_Hz(modes: list, ref_station: int, share: float, work_shape: np.ndarray | None = None) -> float:
    """Highest natural frequency among the modes carrying at least ``share`` of the tip compliance (architecture
    5.1: the L0 stride keeps 2 pi f_max dt <= 0.3 for them)."""
    if not modes:
        return 0.0
    c = tip_compliance(modes, ref_station, None, work_shape)
    total = float(np.sum(c))
    if total <= 0.0:
        return 0.0
    keep = [m.freq_Hz for m, ck in zip(modes, c) if ck >= share * total]
    return float(max(keep)) if keep else 0.0


# ------------------------------------------------------------------------------- basis change
def physical_dofs(modes: list, ref_station: int, work_shape: np.ndarray | None, length_m: float) -> np.ndarray:
    """[n_dof, nm] map from modal coordinates to the physical DOFs at the active contact points: every station's
    translations and rotations (rotations times length_m, so all rows are metres) for the tool side, the
    contact point's translations for the workpiece side."""
    nonwork = [m for m in modes if m.kind != KIND_WORK]
    nst = nonwork[0].stations.shape[0] if nonwork else 0
    n_tool = 6 * nst
    A = np.zeros((n_tool + 3, len(modes)))
    for k, m in enumerate(modes):
        if m.kind == KIND_WORK:
            if work_shape is not None:
                A[n_tool:, k] = work_shape[m.work_row]
        else:
            s = np.array(m.stations, float).copy()
            s[:, 3:] *= length_m
            A[:n_tool, k] = s.reshape(-1)
    return A


def _min_energy_solve(A: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """argmin |A q - b|, minimum sum (weights q)^2 among the minimisers (weighted minimum norm)."""
    if A.shape[1] == 0:
        return np.zeros(0)
    Wi = 1.0 / np.asarray(weights, float)
    p, *_ = np.linalg.lstsq(A * Wi[None, :], b, rcond=1e-12)
    return Wi * p


def project_basis(old_modes: list, old_x: np.ndarray, new_modes: list, *, ref_station: int,
                  old_work_shape: np.ndarray | None, new_work_shape: np.ndarray | None, length_m: float,
                  mac_min: float, freq_tol: float) -> tuple[np.ndarray, float, float, int]:
    """New modal state for a new basis: modes whose identity, frequency and shape match keep their coordinates
    (sign by the shape product); the physical displacement and velocity of the others at the contact points are
    projected onto the unmatched new modes by least squares (minimum strain energy / kinetic energy where the
    problem is underdetermined). Returns (x_new [nm_new, 2], E_old, E_new, tracked count)."""
    old_x = np.asarray(old_x, float).reshape(-1, 2)[:len(old_modes)]
    w_old = np.array([m.omega_rad_s for m in old_modes])
    w_new = np.array([m.omega_rad_s for m in new_modes])
    E_old = DZ.modal_energy(old_x, w_old) if len(old_modes) else 0.0
    x_new = np.zeros((len(new_modes), 2))
    if not new_modes:
        return x_new, E_old, 0.0, 0
    A_old = physical_dofs(old_modes, ref_station, old_work_shape, length_m) if old_modes else None
    A_new = physical_dofs(new_modes, ref_station, new_work_shape, length_m)
    matched_new = np.zeros(len(new_modes), bool)
    matched_old = np.zeros(len(old_modes), bool)
    tracked = 0
    for i, mo in enumerate(old_modes):
        for j, mn in enumerate(new_modes):
            if matched_new[j] or mo.kind != mn.kind or mo.source != mn.source:
                continue
            if abs(mn.omega_rad_s - mo.omega_rad_s) > freq_tol * mo.omega_rad_s:
                continue
            a = A_old[:, i]
            b = A_new[:, j]
            na = float(a @ a)
            nb = float(b @ b)
            if na <= 0.0 or nb <= 0.0:
                continue
            mac = float(a @ b) ** 2 / (na * nb)
            if mac >= mac_min and abs(math.sqrt(nb / na) - 1.0) <= 1.0 - mac_min + freq_tol:
                # least-squares coefficient of the old shape on the new one (1 for the same mode, sign included)
                x_new[j] = (float(a @ b) / nb) * old_x[i]
                matched_new[j] = True
                matched_old[i] = True
                tracked += 1
                break
    rest_old = np.nonzero(~matched_old)[0]
    rest_new = np.nonzero(~matched_new)[0]
    if len(rest_old) and len(rest_new):
        u = A_old[:, rest_old] @ old_x[rest_old, 0]
        v = A_old[:, rest_old] @ old_x[rest_old, 1]
        An = A_new[:, rest_new]
        x_new[rest_new, 0] = _min_energy_solve(An, u, w_new[rest_new])
        x_new[rest_new, 1] = _min_energy_solve(An, v, np.ones(len(rest_new)))
    E_new = DZ.modal_energy(x_new, w_new)
    return x_new, E_old, E_new, tracked


__all__ = ["Mode", "Layout", "station_map", "map_shape", "map_residual", "rotary_axis_line",
           "tool_modes_from_structure", "fallback_tool_modes", "machine_modes", "work_modes", "order_modes",
           "unbalance_coefficients", "contact_shapes", "relaxation", "build_system", "tip_compliance",
           "f_max_resolved_Hz", "physical_dofs", "project_basis"]
