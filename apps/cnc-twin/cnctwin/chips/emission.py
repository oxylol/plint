"""From the hot loop's emission rows (and from chips set moving again) to representative particles
(docs/models/chips.md "Emission").

Mass rule (architecture 5.4): the rows of a contact carry rho x analytic chip volume; the parcels of
the contact are scaled so that their total is rho x the removed field volume of the same window
(``RemovalContact.removed_volume_m3``, i.e. the ``volume_consistency`` scaling), exactly up to
rounding. Consecutive rows of a contact are grouped by ``1 / particles_per_tooth``; each group
becomes one parcel carrying the group's mass with the position, velocity, temperature and shape of
one row drawn by counter-based hash (stream ``chips.select``), its departure direction and speed
scattered by hash normals (stream ``chips.scatter``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.bus import EmissionBatch
from ..core.errors import TwinError
from ..core.rng import stream_key
from . import abi, kernels
from .abi import EM
from .params import ChipParams, chip_shape, hl_coefficients
from .protocols import Formation, FormationQuery
from .state import ChipsState

SELECT_STREAM = "chips.select"
SCATTER_STREAM = "chips.scatter"
FLUTE_STREAM = "chips.flute"
FORM_RIBBON = 11
FORM_ARC_LOOSE = 62
FORM_ELEMENTAL = 70


@dataclass
class Batch:
    """New particles, columnar."""

    pos: np.ndarray
    vel: np.ndarray
    time: np.ndarray
    tick: np.ndarray
    mass: np.ndarray
    temp: np.ndarray
    thickness: np.ndarray
    width: np.ndarray
    length: np.ndarray
    curl: np.ndarray
    form: np.ndarray
    slot: np.ndarray
    colour: np.ndarray
    segmented: np.ndarray
    wet: np.ndarray

    def __len__(self) -> int:
        return len(self.mass)


def empty_batch() -> Batch:
    f = np.zeros(0)
    i = np.zeros(0, np.int64)
    return Batch(np.zeros((0, 3)), np.zeros((0, 3)), f, i, f, f, f, f, f, f, i, i, np.zeros((0, 3), np.uint8),
                 np.zeros(0, bool), i)


def concat(batches: list[Batch]) -> Batch:
    batches = [b for b in batches if len(b)]
    if not batches:
        return empty_batch()
    return Batch(*(np.concatenate([getattr(b, name) for b in batches]) for name in Batch.__dataclass_fields__))


def make_batch(pos, vel, time, tick, mass, temp, thickness, width, length, curl, form, slot, colour=None,
               segmented=None, wet=None) -> Batch:
    n = len(mass)
    return Batch(np.asarray(pos, dtype=float).reshape(n, 3), np.asarray(vel, dtype=float).reshape(n, 3),
                 np.asarray(time, dtype=float), np.asarray(tick, dtype=np.int64), np.asarray(mass, dtype=float),
                 np.asarray(temp, dtype=float), np.asarray(thickness, dtype=float), np.asarray(width, dtype=float),
                 np.asarray(length, dtype=float), np.asarray(curl, dtype=float), np.asarray(form, dtype=np.int64),
                 np.asarray(slot, dtype=np.int64),
                 np.zeros((n, 3), np.uint8) if colour is None else np.asarray(colour, dtype=np.uint8).reshape(n, 3),
                 np.zeros(n, bool) if segmented is None else np.asarray(segmented, dtype=bool),
                 np.zeros(n, np.int64) if wet is None else np.asarray(wet, dtype=np.int64))


def to_emission_batch(batch: Batch) -> EmissionBatch:
    return EmissionBatch(tick=batch.tick.copy(), tool_slot=batch.slot.astype(np.int32), mass_kg=batch.mass.copy(),
                         temp_K=batch.temp.copy(), pos_M_m=batch.pos.copy(), vel_M_m_s=batch.vel.copy(),
                         thickness_m=batch.thickness.copy(), width_m=batch.width.copy(), length_m=batch.length.copy(),
                         curl_radius_m=batch.curl.copy(), form_code=batch.form.astype(np.int16),
                         colour_rgb=batch.colour.copy(), segmented=batch.segmented.copy())


# ------------------------------------------------------------------ counter-based draws

def draws(seed: int, stream: str, keys: np.ndarray, b: int, n_uniform: int, n_normal: int):
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    uniform = np.zeros((len(keys), n_uniform))
    normal = np.zeros((len(keys), n_normal))
    if len(keys):
        kernels.hash_draws(int(seed), int(stream_key(stream)), keys, int(b), 0, uniform, normal)
    return uniform, normal


def scatter(vel: np.ndarray, normals: np.ndarray, sigma_angle: float, sigma_speed: float) -> np.ndarray:
    """Departure velocities turned by two normal angles about two axes normal to v and scaled in speed
    (tan of the angles clipped at 1.4 rad)."""
    vel = np.asarray(vel, dtype=float).reshape(-1, 3)
    out = vel.copy()
    speed = np.linalg.norm(vel, axis=1)
    ok = speed > 0.0
    if not np.any(ok):
        return out
    d = vel[ok] / speed[ok, None]
    k = np.argmin(np.abs(d), axis=1)
    e = np.zeros_like(d)
    e[np.arange(len(d)), k] = 1.0
    e1 = np.cross(d, e)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(d, e1)
    a1 = np.tan(np.clip(sigma_angle * normals[ok, 0], -1.4, 1.4))
    a2 = np.tan(np.clip(sigma_angle * normals[ok, 1], -1.4, 1.4))
    nd = d + a1[:, None] * e1 + a2[:, None] * e2
    nd /= np.linalg.norm(nd, axis=1)[:, None]
    out[ok] = nd * (speed[ok] * np.maximum(1.0 + sigma_speed * normals[ok, 2], 0.0))[:, None]
    return out


# ------------------------------------------------------------------ formation per contact

def mean_conditions(rows: np.ndarray, weights: np.ndarray) -> dict:
    w = weights if weights.sum() > 0 else np.ones(len(rows))
    w = w / w.sum()
    out = {name: float(np.dot(w, rows[:, EM[name]])) for name in ("h_m", "b_m", "l_m", "v_cut_m_s", "rake_rad",
                                                                    "incl_rad", "temp_K")}
    for name in ("edge_kind", "group"):
        codes = rows[:, EM[name]].astype(np.int64)
        codes = np.maximum(codes, 0)
        counts = np.bincount(codes, weights=w)
        out[name] = int(np.argmax(counts))
    return out


def coefficient_lookup(coeffs, group: int, h_m: float) -> tuple[float, float, float]:
    """(shear angle, chip ratio h/h_c, shear-zone temperature) of ProcessCoefficients at h (NaN if absent)."""
    if coeffs is None:
        return math.nan, math.nan, math.nan
    grid = np.asarray(getattr(coeffs, "h_grid_m", []), dtype=float)
    if grid.size == 0 or h_m <= 0.0:
        return math.nan, math.nan, math.nan

    def at(table):
        table = np.asarray(table, dtype=float)
        if table.ndim != 2 or table.shape[1] != grid.size or table.shape[0] == 0:
            return math.nan
        row = table[min(max(group, 0), table.shape[0] - 1)]
        return float(np.interp(math.log(h_m), np.log(grid), row))

    ratio = at(coeffs.chip_ratio)
    if math.isfinite(ratio) and ratio > 1.0:
        ratio = 1.0 / ratio        # given as the compression ratio h_c / h
    return at(coeffs.shear_angle_rad), ratio, at(coeffs.T_shear_K)


def query_for(contact: int, slot: int, rows: np.ndarray, coeffs, coolant_kind: str, continuous: bool
              ) -> FormationQuery:
    mean = mean_conditions(rows, np.maximum(rows[:, EM["mass_kg"]], 0.0))
    phi, ratio, t_shear = coefficient_lookup(coeffs, mean["group"], mean["h_m"])
    temp = mean["temp_K"] if mean["temp_K"] > 0.0 else t_shear
    return FormationQuery(contact=contact, tool_slot=slot, h_m=mean["h_m"], b_m=mean["b_m"], l_m=mean["l_m"],
                          v_m_s=mean["v_cut_m_s"], rake_rad=mean["rake_rad"], incl_rad=mean["incl_rad"],
                          edge_kind=mean["edge_kind"], temp_K=temp, shear_angle_rad=phi, chip_ratio=ratio,
                          coolant_kind=coolant_kind, continuous=continuous)


# ------------------------------------------------------------------ parcels from rows

@dataclass
class ContactRelease:
    contact: int
    slot: int
    formation: Formation | None
    query: FormationQuery | None
    batch: Batch                 # particles (after dust and flute shares)
    made_kg: float               # total mass booked for the contact
    dust_kg: float
    dust_J: float
    flute_kg: float
    flute_J: float
    flute_dims: np.ndarray       # mass-weighted sums [thick, width, length] of the flute share
    scaled: bool
    kinetic_J: float             # 1/2 m v^2 of all the booked chips at separation (M)
    particle_J: float            # enthalpy of the particle share


def parcels_from_rows(contact: int, slot: int, rows: np.ndarray, target_kg: float | None, formation: Formation,
                      query: FormationQuery, ppt: float, seed: int, group0: int, params: ChipParams,
                      dust_fraction: float, confined: float, rho: float, enthalpy_of, tick_s: float
                      ) -> tuple[ContactRelease, int]:
    """``enthalpy_of(temps) -> J/kg`` (vectorised)."""
    """Parcels of one contact's rows. Returns the release and the number of groups used."""
    rows = rows[np.lexsort((rows[:, EM["tooth"]], rows[:, EM["tick"]]))]
    mass = np.maximum(rows[:, EM["mass_kg"]], 0.0)
    total = float(mass.sum())
    if target_kg is None:
        target_kg = total
        scaled = False
    else:
        scaled = True
    k = max(1, int(round(1.0 / ppt)))
    n_groups = (len(rows) + k - 1) // k
    keys = np.arange(group0, group0 + n_groups, dtype=np.int64)
    uni, _ = draws(seed, SELECT_STREAM, keys, contact, 1, 0)
    _, nor = draws(seed, SCATTER_STREAM, keys, contact, 0, 3)
    sel = []
    gmass = []
    for g in range(n_groups):
        lo, hi = g * k, min((g + 1) * k, len(rows))
        sel.append(lo + min(int(uni[g, 0] * (hi - lo)), hi - lo - 1))
        gm = float(mass[lo:hi].sum())
        gmass.append(target_kg * gm / total if total > 0.0 else target_kg * (hi - lo) / len(rows))
    sel = np.asarray(sel, dtype=np.int64)
    gmass = np.asarray(gmass)
    rep = rows[sel]
    vel = scatter(rep[:, [EM["vx_M_m_s"], EM["vy_M_m_s"], EM["vz_M_m_s"]]], nor,
                  float(params.aero["scatter_angle_rad"]), float(params.aero["scatter_speed_fraction"]))
    h_rows = rep[:, EM["h_m"]]
    ratio = formation.chip_ratio if formation.chip_ratio > 0.0 else 1.0
    thickness = (formation.thickness_m * h_rows / query.h_m if query.h_m > 0.0
                 else np.full(len(rep), formation.thickness_m))
    thickness = np.where(thickness > 0.0, thickness, h_rows / ratio)
    width = np.full(len(rep), formation.width_m) if formation.width_m > 0.0 else rep[:, EM["b_m"]]
    length = (np.full(len(rep), formation.length_m) if formation.length_m > 0.0
              else rep[:, EM["l_m"]] * ratio)
    temp = np.where(rep[:, EM["temp_K"]] > 0.0, rep[:, EM["temp_K"]], formation.chip_temp_K)
    dust = gmass * dust_fraction
    flute = (gmass - dust) * confined
    part = gmass - dust - flute
    h_spec = enthalpy_of(temp)
    n = len(rep)
    batch = make_batch(rep[:, [EM["x_M_m"], EM["y_M_m"], EM["z_M_m"]]], vel, rep[:, EM["tick"]] * tick_s,
                       rep[:, EM["tick"]].astype(np.int64), part, temp, thickness, width, length,
                       np.full(n, formation.curl_radius_m), np.full(n, formation.form_code), np.full(n, slot),
                       np.tile(np.asarray(formation.colour_rgb, dtype=np.uint8), (n, 1)), np.full(n, formation.segmented))
    keep = part > 0.0
    batch = Batch(*(getattr(batch, name)[keep] for name in Batch.__dataclass_fields__))
    flute_dims = np.array([float(np.dot(flute, thickness)), float(np.dot(flute, width)), float(np.dot(flute, length))])
    kinetic = 0.5 * float(np.dot(gmass, np.einsum("ij,ij->i", vel, vel)))
    release = ContactRelease(contact, slot, formation, query, batch, float(gmass.sum()), float(dust.sum()),
                             float(np.dot(dust, h_spec)), float(flute.sum()), float(np.dot(flute, h_spec)), flute_dims,
                             scaled, kinetic, float(np.dot(part[keep], h_spec[keep])))
    return release, n_groups


# ------------------------------------------------------------------ particles from the pile (mobilised rows)

def batch_from_mob(rows: np.ndarray, W: abi.Window, rho: float, temperature_of, slot_of_row=None) -> Batch:
    """Particles from flush / sweep / recut rows (abi.MOB_COLUMNS): mean chip of the cell, released at the
    row's point (G) with the row's velocity (M) at the row's time."""
    if len(rows) == 0:
        return empty_batch()
    c = {name: i for i, name in enumerate(abi.MOB_COLUMNS)}
    mass = rows[:, c["mass_kg"]]
    ok = mass > 0.0
    rows = rows[ok]
    mass = mass[ok]
    if len(rows) == 0:
        return empty_batch()
    n_t = W.grid_origin_M.shape[0] - 1
    t0 = float(W.t0_s[0])
    dtk = float(W.dt_tick_s[0])
    times = np.where(rows[:, c["time_s"]] >= t0, rows[:, c["time_s"]], t0)
    times = np.minimum(times, float(W.t_end_s[0]))
    q = np.clip((times - t0) / dtk, 0.0, n_t)
    k = np.minimum(np.floor(q).astype(np.int64), n_t - 1)
    frac = (q - k)[:, None]
    origin = W.grid_origin_M[k] + frac * (W.grid_origin_M[k + 1] - W.grid_origin_M[k])
    uvw = rows[:, [c["u_m"], c["v_m"], c["w_m"]]]
    pos = origin + uvw @ W.R_M_G.T
    temp = np.array([temperature_of(h) for h in rows[:, c["enthalpy_J"]] / mass])
    thick = rows[:, c["thick_mass"]] / mass
    width = rows[:, c["width_mass"]] / mass
    length = rows[:, c["length_mass"]] / mass
    long_frac = rows[:, c["long_kg"]] / mass
    kind = rows[:, c["kind"]].astype(np.int64)
    recut = kind == abi.MOB_RECUT
    length = np.where(recut, np.minimum(length, width), length)
    form = np.where(long_frac >= 0.5, FORM_RIBBON, np.where(length > 2.0 * thick, FORM_ARC_LOOSE, FORM_ELEMENTAL))
    form = np.where(recut, np.where(length > 2.0 * thick, FORM_ARC_LOOSE, FORM_ELEMENTAL), form)
    owner = rows[:, c["owner"]].astype(np.int64)
    slot = owner if slot_of_row is None else slot_of_row
    ticks = np.zeros(len(rows), np.int64)
    return make_batch(pos, rows[:, [c["vx_M_m_s"], c["vy_M_m_s"], c["vz_M_m_s"]]], times, ticks, mass, temp, thick,
                      width, length, np.zeros(len(rows)), form, slot)


# ------------------------------------------------------------------ insertion into the particle set

def insert(state: ChipsState, batch: Batch, params: ChipParams, rho: float, max_particles: int,
           specific_enthalpy, temperature_of) -> np.ndarray:
    """Insert new particles into free slots (increasing index), growing the set up to max_particles.
    Beyond the cap the remaining parcels merge into the last inserted parcel of the call (or, when nothing
    fits, into the flying particle of highest index), mass and enthalpy conserved. Returns the slots used."""
    n_new = len(batch)
    if n_new == 0:
        return np.zeros(0, np.int64)
    P = state.particles
    n_used = int(P.n[0])
    free = np.nonzero(P.status[:n_used] == abi.FREE)[0]
    need = n_new - len(free)
    if need > 0:
        target = min(n_used + need, max_particles)
        state.ensure_capacity(target)
        P = state.particles
        free = np.concatenate([free, np.arange(n_used, target, dtype=np.int64)])
    slots = free[:n_new]
    n_fit = len(slots)
    m_chip, dv, area, surface, psi = chip_shape(batch.thickness, batch.width, batch.length, rho)
    a, b, c, d = hl_coefficients(params.aero, psi)
    idx = slots
    sl = slice(0, n_fit)
    P.pos_M_m[idx] = batch.pos[sl]
    P.vel_M_m_s[idx] = batch.vel[sl]
    P.time_s[idx] = batch.time[sl]
    P.mass_kg[idx] = batch.mass[sl]
    P.temp_K[idx] = batch.temp[sl]
    P.thickness_m[idx] = np.maximum(batch.thickness[sl], 1e-9)
    P.width_m[idx] = np.maximum(batch.width[sl], P.thickness_m[idx])
    P.length_m[idx] = np.maximum(batch.length[sl], P.thickness_m[idx])
    P.curl_radius_m[idx] = batch.curl[sl]
    P.dv_m[idx] = dv[sl]
    P.chip_mass_kg[idx] = m_chip[sl]
    P.area_m2[idx] = area[sl]
    P.surface_m2[idx] = surface[sl]
    P.hl_a[idx] = a[sl]
    P.hl_b[idx] = b[sl]
    P.hl_c[idx] = c[sl]
    P.hl_d[idx] = d[sl]
    P.form[idx] = batch.form[sl]
    P.status[idx] = abi.FLYING
    P.owner[idx] = -1
    P.slot[idx] = batch.slot[sl]
    P.pid[idx] = np.arange(state.next_pid, state.next_pid + n_fit, dtype=np.int64)
    state.next_pid += n_fit
    P.bounces[idx] = 0
    P.wet[idx] = batch.wet[sl]
    state.colour[idx] = batch.colour[sl]
    state.segmented[idx] = batch.segmented[sl]
    state.attach_rel[idx] = 0.0
    if n_fit:
        P.n[0] = max(int(P.n[0]), int(slots.max()) + 1)
    if n_fit < n_new:
        if n_fit:
            target = int(idx[-1])
        else:
            flying = np.nonzero(P.status[:int(P.n[0])] == abi.FLYING)[0]
            if len(flying) == 0:
                raise TwinError("chips: the particle set is full of attached chips (raise physics.chips.max_particles)")
            target = int(flying[-1])
        for j in range(n_fit, n_new):
            m_old = float(P.mass_kg[target])
            m_add = float(batch.mass[j])
            h = (m_old * specific_enthalpy(float(P.temp_K[target])) + m_add * specific_enthalpy(float(batch.temp[j])))
            P.mass_kg[target] = m_old + m_add
            P.temp_K[target] = temperature_of(h / (m_old + m_add))
    return slots
