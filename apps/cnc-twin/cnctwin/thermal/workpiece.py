"""Workpiece temperatures: cut-cell finite volumes co-registered with the stock lattice, the
fixture as a lumped body, and the two-scale local temperature at the contact
(docs/models/thermal.md "Workpiece").

Per finite-volume step (every ``physics.thermal.every_n_macro`` macro steps, energy accumulated
exactly in between): (1) new material fractions in the box removal changed - removed material
leaves with its enthalpy; (2) the accumulated heat rows are deposited into the remaining material;
(3) one Douglas-Gunn ADI step (kernels.work_step) with convection, radiation and fixture contact;
(4) the fixture node takes the contact heat. Each solver books its energy ledger.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from ..db.models import params
from ..units import STEFAN_BOLTZMANN_W_M2K4
from . import convection as cv
from . import kernels as K
from . import props
from .abi import (B_ALL_WET, B_COOLANT_ON, B_EPS, B_FMIN, B_H_CONTACT, B_H_WET, B_SIGMA, B_T_AIR, B_T_FIX,
                  B_T_FLUID, B_T_RAD, FACE_NORMALS, L_AIR, L_CONTACT, L_COOL, L_LOST, L_MASS, L_RAD, L_SRC,
                  L_TSURF, L_U0, L_U1, N_L, MatTable, WorkBoundary, WorkGrid, new_work_boundary,
                  new_work_grid, new_work_scratch)
from .ledger import SolverLedger
from .response import gauss_legendre


@functools.lru_cache(maxsize=1)
def _p() -> dict:
    return dict(params("thermal-work"))


# positions (in half lengths from the source centre, behind it) where the surface maximum of a moving
# band is sought: the trailing edge at high Peclet numbers, further inside at low ones
BAND_SAMPLES = (-1.0, -0.95, -0.85, -0.65, -0.35, 0.0)


def grid_shape(lo: np.ndarray, hi: np.ndarray, cell_m: float) -> tuple[int, int, int]:
    span = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    return tuple(max(1, int(math.ceil(s / cell_m - 1e-9))) for s in span)


def default_contact(fill: np.ndarray, down_S: np.ndarray, fmin: float) -> np.ndarray:
    """Faces resting on the table/parallels: the face direction most aligned with gravity, on the
    lowest material layer along it (a part held only by friction on its base)."""
    dots = FACE_NORMALS @ np.asarray(down_S, dtype=float)
    face = int(np.argmax(dots))
    axis = face // 2
    positive = face % 2 == 1
    mask = np.zeros(fill.shape, dtype=np.uint8)
    material = fill > fmin
    if not material.any():
        return mask
    idx = np.nonzero(material.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    layer = int(idx.max() if positive else idx.min())
    sl = [slice(None)] * 3
    sl[axis] = layer
    layer_mat = material[tuple(sl)]
    face_bits = np.uint8(1 << face)
    target = mask[tuple(sl)]
    target[layer_mat] = face_bits
    mask[tuple(sl)] = target
    return mask


@dataclass
class FixtureNode:
    """Workholding as one lumped body: heat from the part's contact faces and from chips resting on
    it; free convection to air and contact conduction into the table (machine structure)."""

    mass_kg: float
    mat: MatTable
    area_m2: float
    h_surface_W_m2K: float
    g_table_W_K: float
    T_K: float
    h_J_kg: float = 0.0
    ledger: SolverLedger = field(default_factory=lambda: SolverLedger("fixture"))

    def __post_init__(self):
        self.h_J_kg = props.h_of_T(self.mat, self.T_K)

    def energy(self) -> float:
        return self.mass_kg * self.h_J_kg

    def step(self, dt: float, heat_in_J: float, T_air: float, T_table: float, tol: float, time_s: float) -> dict:
        cp = float(K.prop_interp(self.mat.T_K, self.mat.cp_J_kgK, self.T_K))
        cap = self.mass_kg * cp
        g_air = self.h_surface_W_m2K * self.area_m2
        t_lin = (cap * self.T_K + heat_in_J + dt * (g_air * T_air + self.g_table_W_K * T_table)) / \
            (cap + dt * (g_air + self.g_table_W_K))
        loss_air = dt * g_air * (t_lin - T_air)
        loss_table = dt * self.g_table_W_K * (t_lin - T_table)
        u0 = self.energy()
        self.h_J_kg = self.h_J_kg + (heat_in_J - loss_air - loss_table) / self.mass_kg
        self.T_K = props.T_of_h(self.mat, self.h_J_kg)
        self.ledger.book(u0, self.energy(), heat_in_J, loss_air + loss_table, 0.0, tol, time_s)
        return {"air_J": loss_air, "table_J": loss_table}


@dataclass
class ContactSurface:
    """Two-scale temperatures at one contact (K) and what they rest on."""

    surface_K: float
    preheat_K: float
    background_K: float
    feed_rise_K: float
    flash_rise_K: float
    cell_mean_K: float
    point_S_m: np.ndarray


class WorkThermal:
    """Finite-volume thermal model of one workpiece."""

    def __init__(self, workpiece: int, geometry, mat: MatTable, settings, temp0_K: float, family: str | None,
                 time_s: float, fixture_mass_kg: float | None = None, contact=None):
        self.workpiece = workpiece
        self.geometry = geometry
        self.contact_provider = contact
        self.mat = mat
        self.settings = settings
        self.family = family
        self.emissivity = cv.emissivity(family)
        lo, hi = geometry.stock_bounds_S(workpiece)
        cell = settings.cell_m
        self.shape = grid_shape(lo, hi, cell)
        self.grid = new_work_grid(self.shape, lo, cell)
        self.scratch = new_work_scratch(self.shape)
        self.time_s = float(time_s)
        self.temp0_K = float(temp0_K)
        self.pending: list[np.ndarray] = []
        self.fixture_heat_J = 0.0              # chips' heat on the fixture, for the next step
        self.fixture_to_table_J = 0.0          # fixture's loss into the table, not yet given to the structure
        self.down_S = np.array([0.0, 0.0, -1.0])
        self.up_S = np.array([0.0, 0.0, 1.0])
        self.surface_mean_K = float(temp0_K)
        self.last = np.full(N_L, np.nan)
        self.fv_count = 0
        self.unplaced_J = 0.0
        self.clamped = True
        self.ledger = SolverLedger(f"work.{workpiece}")
        self.reset_material(temp0_K)
        p = _p()
        fmat = props.material_table_by_id(str(p["fixture_material"]))
        mass = float(fixture_mass_kg) if fixture_mass_kg else float(p["fixture_default_mass_kg"])
        volume = mass / fmat.rho_kg_m3[0]
        self.fixture = FixtureNode(mass, fmat, p["fixture_area_factor"] * volume ** (2.0 / 3.0),
                                   p["fixture_surface_htc_W_m2K"],
                                   p["fixture_contact_conductance_W_m2K"] * p["fixture_footprint_factor"]
                                   * volume ** (2.0 / 3.0), float(temp0_K),
                                   ledger=SolverLedger(f"fixture.{workpiece}"))
        self.fixture.ledger.u_J = self.fixture.energy()
        self.fixture.ledger.u0_J = self.fixture.ledger.u_J

    # ------------------------------------------------------------------ geometry
    def reset_material(self, temp_K: float) -> None:
        """Fresh stock at a uniform temperature (start of the run, next part of a production run)."""
        g = self.grid
        fill = np.clip(np.asarray(self.geometry.fill_fraction(self.workpiece, g.geom[:3].copy(), g.geom[3],
                                                              self.shape), dtype=float), 0.0, 1.0)
        fill[fill <= self.settings.min_fraction] = 0.0
        g.f[:] = fill
        h0 = props.h_of_T(self.mat, temp_K)
        g.h[:] = np.where(fill > 0.0, h0, 0.0)
        g.T[:] = np.where(fill > 0.0, float(temp_K), np.nan)
        self.geometry.changed_box_S(self.workpiece)        # the fresh fill is current
        self.update_contact()
        energy = np.zeros(4)
        K.work_energy(g, self.mat, self.settings.min_fraction, energy)
        self.ledger.u_J = float(energy[0])
        self.ledger.u0_J = self.ledger.u_J
        self.mass_kg = float(energy[1])
        self.extent = self.material_extent()
        self.pending = []

    def update_contact(self) -> None:
        g = self.grid
        if self.contact_provider is not None:
            mask = np.asarray(self.contact_provider.contact_faces(self.workpiece, g.geom[:3].copy(), g.geom[3], g.f),
                              dtype=np.uint8)
        else:
            mask = default_contact(g.f, self.down_S, self.settings.min_fraction)
        if not self.clamped:
            mask = np.zeros_like(mask)
        g.contact[:] = mask

    def set_gravity(self, down_S: np.ndarray) -> None:
        down = np.asarray(down_S, dtype=float)
        norm = np.linalg.norm(down)
        if norm == 0.0:
            return
        down = down / norm
        if not np.allclose(down, self.down_S, atol=1e-6):
            self.down_S = down
            self.up_S = -down
            self.update_contact()

    def apply_fill_changes(self) -> tuple[float, float]:
        """(removed enthalpy, added enthalpy) of the material changes since the last call."""
        box = self.geometry.changed_box_S(self.workpiece)
        if box is None:
            return 0.0, 0.0
        g = self.grid
        cell = g.geom[3]
        lo_i = np.floor((np.asarray(box[0]) - g.geom[:3]) / cell).astype(int) - 1
        hi_i = np.ceil((np.asarray(box[1]) - g.geom[:3]) / cell).astype(int) + 1
        lo_i = np.clip(lo_i, 0, np.array(self.shape))
        hi_i = np.clip(hi_i, 0, np.array(self.shape))
        if np.any(hi_i <= lo_i):
            return 0.0, 0.0
        sub_shape = tuple(int(v) for v in hi_i - lo_i)
        origin = g.geom[:3] + lo_i * cell
        fnew = np.ascontiguousarray(np.asarray(self.geometry.fill_fraction(self.workpiece, origin, cell, sub_shape),
                                               dtype=float))
        out = np.zeros(5)
        h_amb = props.h_of_T(self.mat, self.temp0_K)
        K.fill_update(g, self.mat, fnew, int(lo_i[0]), int(lo_i[1]), int(lo_i[2]), self.settings.min_fraction, h_amb,
                      out)
        if self.contact_provider is None and out[4] > 0:
            self.update_contact()
        return float(out[0]), float(out[1])

    # ------------------------------------------------------------------ heat
    def add_rows(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=float)
        if rows.size:
            self.pending.append(np.ascontiguousarray(rows.reshape(-1, 8)))

    def pending_energy(self) -> float:
        return float(sum(r[:, 6].sum() for r in self.pending))

    def boundary(self, env) -> WorkBoundary:
        """Boundary description for the coming step (``env``: WorkEnvironment)."""
        near = np.asarray(env.near, dtype=float).reshape(-1, 5) if env.coolant_on else np.zeros((0, 5))
        wet = np.asarray(env.wet, dtype=np.int8) if env.wet is not None else np.zeros((0, 0), np.int8)
        bnd = new_work_boundary(near.shape[0], wet.shape if wet.ndim == 2 else (0, 0))
        bnd.near[:] = near
        if wet.ndim == 2 and wet.size:
            bnd.wet[:] = wet
        bnd.wet_geom[:] = [env.wet_origin_S_m[0], env.wet_origin_S_m[1], env.wet_cell_m if env.wet_cell_m > 0 else 1.0]
        bnd.h_air[:] = self.free_convection(env.T_air_K)
        sc = bnd.scal
        sc[B_T_AIR] = env.T_air_K
        sc[B_T_FLUID] = env.T_fluid_K
        sc[B_H_WET] = env.h_far_W_m2K
        sc[B_ALL_WET] = 1.0 if (env.coolant_on and (wet.ndim != 2 or wet.size == 0) and env.h_far_W_m2K > 0.0) else 0.0
        sc[B_EPS] = self.emissivity
        sc[B_T_RAD] = env.T_air_K
        sc[B_H_CONTACT] = _p()["fixture_contact_conductance_W_m2K"] if self.clamped else 0.0
        sc[B_T_FIX] = self.fixture.T_K
        sc[B_FMIN] = self.settings.min_fraction
        sc[B_COOLANT_ON] = 1.0 if env.coolant_on else 0.0
        sc[B_SIGMA] = STEFAN_BOLTZMANN_W_M2K4
        return bnd

    def material_extent(self) -> np.ndarray:
        g = self.grid
        material = g.f > self.settings.min_fraction
        extent = np.zeros(3)
        for axis in range(3):
            idx = np.nonzero(material.any(axis=tuple(a for a in range(3) if a != axis)))[0]
            extent[axis] = (idx.max() - idx.min() + 1) * g.geom[3] if idx.size else g.geom[3]
        return extent

    def free_convection(self, T_air: float) -> np.ndarray:
        """Free-convection h per face direction from the mean exposed temperature and the part size."""
        extent = self.extent
        up = self.up_S
        height = float(np.sum(np.abs(up) * extent))
        out = np.zeros(6)
        t_s = self.surface_mean_K if np.isfinite(self.surface_mean_K) else T_air
        for face in range(6):
            axis = face // 2
            others = [extent[a] for a in range(3) if a != axis]
            plate = others[0] * others[1] / (2.0 * (others[0] + others[1]))
            c = float(FACE_NORMALS[face] @ up)
            if abs(t_s - T_air) < 1e-9:
                out[face] = 0.0
                continue
            out[face] = cv.h_face(c, t_s, T_air, max(height, 1e-4), max(plate, 1e-4))
        return out

    def fv_step(self, dt: float, env, time_s: float) -> dict:
        """One finite-volume step: fill changes, deposits, ADI, fixture. Returns the step's terms."""
        u_before = self.ledger.u_J
        removed, added = self.apply_fill_changes()
        self.extent = self.material_extent()
        g = self.grid
        s = self.scratch
        s.src[:] = 0.0
        dep = np.zeros(2)
        if self.pending:
            rows = np.ascontiguousarray(np.concatenate(self.pending, axis=0))
            K.deposit(g.f, s.src, g.geom, rows, self.settings.min_fraction, self.settings.deposit_search_cells, dep)
        self.pending = []
        bnd = self.boundary(env)
        out = np.zeros(N_L)
        K.work_step(g, s, self.mat, bnd, float(dt), out)
        self.last = out
        if np.isfinite(out[L_TSURF]):
            self.surface_mean_K = float(out[L_TSURF])
        losses = out[L_AIR] + out[L_COOL] + out[L_RAD] + out[L_CONTACT]
        self.ledger.book(u_before, out[L_U1], out[L_SRC], losses, removed - added, self.settings.energy_tol, time_s)
        self.mass_kg = float(out[L_MASS])
        fixture = self.fixture.step(dt, out[L_CONTACT] + self.fixture_heat_J, env.T_air_K, env.T_table_K,
                                    self.settings.energy_tol, time_s)
        self.fixture_heat_J = 0.0
        self.fixture_to_table_J += fixture["table_J"]
        self.time_s = time_s
        self.fv_count += 1
        self.unplaced_J += float(dep[0]) + float(out[L_LOST])
        return {"deposited_J": float(out[L_SRC]), "unplaced_J": float(dep[0]) + float(out[L_LOST]),
                "removed_J": removed, "added_J": added, "air_J": float(out[L_AIR]), "coolant_J": float(out[L_COOL]),
                "radiation_J": float(out[L_RAD]), "contact_J": float(out[L_CONTACT]),
                "fixture_air_J": fixture["air_J"], "fixture_table_J": fixture["table_J"],
                "u_before_J": u_before, "u_after_J": float(out[L_U1]), "u_kernel0_J": float(out[L_U0])}

    # ------------------------------------------------------------------ queries
    def cell_of(self, point_S: np.ndarray, search: int = 2) -> tuple[int, int, int] | None:
        """The material cell containing a point, or the nearest material cell (centre distance, ties
        by index order) within ``search`` cells."""
        g = self.grid
        idx = np.floor((np.asarray(point_S, dtype=float) - g.geom[:3]) / g.geom[3]).astype(int)
        fmin = self.settings.min_fraction
        if all(0 <= idx[a] < self.shape[a] for a in range(3)) and g.f[tuple(idx)] > fmin:
            return (int(idx[0]), int(idx[1]), int(idx[2]))
        lo = np.maximum(idx - search, 0)
        hi = np.minimum(idx + search + 1, np.array(self.shape))
        if np.any(hi <= lo):
            return None
        block = g.f[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] > fmin
        if not block.any():
            return None
        ii, jj, kk = np.nonzero(block)
        cells = np.stack([ii + lo[0], jj + lo[1], kk + lo[2]], axis=1)
        centres = g.geom[:3] + (cells + 0.5) * g.geom[3]
        d2 = np.sum((centres - np.asarray(point_S, dtype=float)) ** 2, axis=1)
        best = int(np.argmin(d2))
        return (int(cells[best, 0]), int(cells[best, 1]), int(cells[best, 2]))

    def cell_temp(self, cell) -> float:
        return float(self.grid.T[cell]) if cell is not None else float("nan")

    def bulk(self) -> tuple[float, float]:
        """(mass-weighted mean, max) temperature of the material (K)."""
        out = np.zeros(4)
        K.work_energy(self.grid, self.mat, self.settings.min_fraction, out)
        return float(out[2]), float(out[3])

    def published_grid(self) -> np.ndarray:
        """Temperatures [nx, ny, nz] with NaN outside material (a new array, never modified later)."""
        return self.grid.T.copy()

    # ------------------------------------------------------------------ two-scale local temperature
    def contact_surface(self, src: "ContactSource", fv_step_now: bool) -> ContactSurface:
        """T_local = T_FV(cell) + [theta_feed - <theta_feed>_cell] + theta_flash at the trailing edge of
        the contact on the generated surface; preheat at its leading edge (docs/models/thermal.md)."""
        settings = self.settings
        glx, glw = gauss_legendre(settings.jaeger_points)
        cell = self.cell_of(src.point_S_m)
        t_bg = self.cell_temp(cell)
        if not np.isfinite(t_bg):
            t_bg = self.temp0_K
        mat = self.mat
        k = float(K.prop_interp(mat.T_K, mat.k_W_mK, t_bg))
        cp = float(K.prop_interp(mat.T_K, mat.cp_J_kgK, t_bg))
        alpha = k / (mat.rho_kg_m3[0] * cp)
        feed = 0.0
        lead = 0.0
        if src.power_W > 0.0 and src.feed_half_length_m > 0.0 and src.width_m > 0.0 and src.window_s > 0.0:
            q = src.power_W / (4.0 * src.feed_half_length_m * 0.5 * src.width_m)
            l = src.feed_half_length_m
            w = 0.5 * src.width_m
            feed = max(float(K.moving_rect_rise(q, l, w, src.feed_speed_m_s, k, alpha, src.window_s, x * l, 0.0, 0.0,
                                                glx, glw)) for x in BAND_SAMPLES)
            lead = float(K.moving_rect_rise(q, l, w, src.feed_speed_m_s, k, alpha, src.window_s, l, 0.0, 0.0, glx, glw))
        flash = 0.0
        if src.flash_flux_W_m2 > 0.0 and src.flash_half_length_m > 0.0 and src.flash_speed_m_s > 0.0:
            lf = src.flash_half_length_m
            flash = max(float(K.moving_rect_rise(src.flash_flux_W_m2, lf, 0.5 * src.width_m, src.flash_speed_m_s, k,
                                                 alpha, src.flash_duration_s, x * lf, 0.0, 0.0, glx, glw))
                        for x in BAND_SAMPLES)
        cell_mean = 0.0
        if fv_step_now and cell is not None:
            cap = mat.rho_kg_m3[0] * self.grid.geom[3] ** 3 * float(self.grid.f[cell]) * cp
            cell_mean = src.energy_in_cell_J(self.grid.geom[3]) / cap if cap > 0.0 else 0.0
            cell_mean = min(cell_mean, feed)
        ahead_cell = self.cell_of(src.point_S_m + src.feed_dir_S * (src.feed_half_length_m + self.grid.geom[3]))
        t_ahead = self.cell_temp(ahead_cell)
        if not np.isfinite(t_ahead):
            t_ahead = t_bg
        preheat = t_ahead + lead - (cell_mean if ahead_cell == cell else 0.0)
        surface = t_bg + feed - cell_mean + flash
        return ContactSurface(surface, preheat, t_bg, feed, flash, cell_mean, np.asarray(src.point_S_m, dtype=float))

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        g = self.grid
        arrays = {f"{prefix}f": g.f.copy(), f"{prefix}h": g.h.copy(), f"{prefix}T": g.T.copy(),
                  f"{prefix}contact": g.contact.copy(), f"{prefix}geom": g.geom.copy(),
                  f"{prefix}pending": (np.concatenate(self.pending, axis=0) if self.pending else np.zeros((0, 8))),
                  f"{prefix}down_S": self.down_S.copy(), f"{prefix}last": self.last.copy()}
        for name in self.mat._fields:
            arrays[f"{prefix}mat.{name}"] = np.array(getattr(self.mat, name))
        for name in self.fixture.mat._fields:
            arrays[f"{prefix}fixture_mat.{name}"] = np.array(getattr(self.fixture.mat, name))
        meta = {"workpiece": self.workpiece, "family": self.family, "emissivity": self.emissivity,
                "time_s": self.time_s, "temp0_K": self.temp0_K, "fixture_heat_J": self.fixture_heat_J,
                "fixture_to_table_J": self.fixture_to_table_J,
                "surface_mean_K": self.surface_mean_K, "fv_count": self.fv_count, "unplaced_J": self.unplaced_J,
                "ledger": self.ledger.to_dict(), "mass_kg": self.mass_kg, "clamped": self.clamped,
                "fixture": {"mass_kg": self.fixture.mass_kg, "area_m2": self.fixture.area_m2,
                            "h_surface_W_m2K": self.fixture.h_surface_W_m2K, "g_table_W_K": self.fixture.g_table_W_K,
                            "T_K": self.fixture.T_K, "h_J_kg": self.fixture.h_J_kg,
                            "ledger": self.fixture.ledger.to_dict()}}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str, geometry, settings, contact=None) -> "WorkThermal":
        obj = cls.__new__(cls)
        obj.workpiece = int(meta["workpiece"])
        obj.geometry = geometry
        obj.contact_provider = contact
        obj.settings = settings
        obj.family = meta["family"]
        obj.emissivity = float(meta["emissivity"])
        obj.mat = MatTable(*(np.ascontiguousarray(arrays[f"{prefix}mat.{name}"]).copy() for name in MatTable._fields))
        f = np.ascontiguousarray(arrays[f"{prefix}f"]).copy()
        obj.shape = tuple(int(v) for v in f.shape)
        obj.grid = WorkGrid(f, np.ascontiguousarray(arrays[f"{prefix}h"]).copy(),
                            np.ascontiguousarray(arrays[f"{prefix}T"]).copy(),
                            np.ascontiguousarray(arrays[f"{prefix}contact"]).copy(),
                            np.ascontiguousarray(arrays[f"{prefix}geom"]).copy())
        obj.scratch = new_work_scratch(obj.shape)
        pending = np.ascontiguousarray(arrays[f"{prefix}pending"]).copy()
        obj.pending = [pending] if pending.size else []
        obj.down_S = np.asarray(arrays[f"{prefix}down_S"], dtype=float).copy()
        obj.up_S = -obj.down_S
        obj.last = np.asarray(arrays[f"{prefix}last"], dtype=float).copy()
        obj.time_s = float(meta["time_s"])
        obj.temp0_K = float(meta["temp0_K"])
        obj.fixture_heat_J = float(meta["fixture_heat_J"])
        obj.fixture_to_table_J = float(meta["fixture_to_table_J"])
        obj.surface_mean_K = float(meta["surface_mean_K"])
        obj.fv_count = int(meta["fv_count"])
        obj.unplaced_J = float(meta["unplaced_J"])
        obj.ledger = SolverLedger.from_dict(meta["ledger"])
        obj.mass_kg = float(meta["mass_kg"])
        obj.clamped = bool(meta["clamped"])
        fm = meta["fixture"]
        fmat = MatTable(*(np.ascontiguousarray(arrays[f"{prefix}fixture_mat.{name}"]).copy()
                          for name in MatTable._fields))
        obj.fixture = FixtureNode(float(fm["mass_kg"]), fmat, float(fm["area_m2"]), float(fm["h_surface_W_m2K"]),
                                  float(fm["g_table_W_K"]), float(fm["T_K"]), ledger=SolverLedger.from_dict(fm["ledger"]))
        obj.fixture.h_J_kg = float(fm["h_J_kg"])
        obj.extent = obj.material_extent()
        return obj


@dataclass
class WorkEnvironment:
    T_air_K: float = 293.15
    T_table_K: float = 293.15
    coolant_on: bool = False
    T_fluid_K: float = 293.15
    h_far_W_m2K: float = 0.0
    near: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    wet: np.ndarray | None = None
    wet_origin_S_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wet_cell_m: float = 0.0


@dataclass
class ContactSource:
    """The work-bound heat of one contact as the two moving sources of the two-scale model."""

    point_S_m: np.ndarray
    feed_dir_S: np.ndarray
    power_W: float                 # mean work-bound heat rate over the step
    feed_speed_m_s: float
    feed_half_length_m: float      # half extent of the engagement along the feed
    width_m: float                 # extent across the feed (ap)
    window_s: float                # time since the last finite-volume deposit (capped by the engagement time)
    flash_flux_W_m2: float         # flux of one edge passage on its band
    flash_half_length_m: float     # half of (shear-plane footprint + flank contact)
    flash_speed_m_s: float         # cutting speed
    flash_duration_s: float        # duration of one passage

    def energy_in_cell_J(self, cell_m: float) -> float:
        """Energy of the feed source deposited inside the contact's cell during the window: the part of
        the path (v t) and of the footprint (2l x b) that lies in one cell (docs/models/thermal.md)."""
        path = self.feed_speed_m_s * self.window_s
        path_share = min(1.0, cell_m / path) if path > 0.0 else 1.0
        area_share = min(1.0, cell_m / (2.0 * self.feed_half_length_m)) * min(1.0, cell_m / max(self.width_m, 1e-12))
        return self.power_W * self.window_s * path_share * area_share
