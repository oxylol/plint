"""Distribution of the cutting heat (single owner, docs/architecture.md 12).

The hot loop reports the heat generated per zone and contact (shear, rake, flank, ploughing);
ProcessCoefficients give, per element group and chip thickness, the fractions shear -> chip,
rake -> tool, flank -> tool. This module splits each zone once:

    chip = sum_e Qs_e f_sc + Qr_e (1 - f_rt)
    tool = sum_e Qr_e f_rt + (Qf_e + Qp_e) f_ft
    work = sum_e Qs_e (1 - f_sc) + (Qf_e + Qp_e)(1 - f_ft)

Ploughing (rubbing below h_min and edge forces) happens at the edge-flank contact and is split
with the flank fraction. The element shares Q*_e of a zone follow the per-element integrals of
the step (docs/models/thermal.md "Partition"): shear ~ int_h_dt db r (removed volume), rake ~
int_sigma_v_rake h zeta^n db (friction power over the contact area), flank and ploughing ~
int_sigma_v_flank l_f db; when an integral is zero everywhere the next proxy is used (sliding
distances, then in-cut time, then equal shares). The three destinations are computed
independently; their sum equals the generation to round-off (checked by the tests).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EdgeModel
from ..db.models import params


@functools.lru_cache(maxsize=1)
def fallback_fractions() -> tuple[float, float, float]:
    p = params("thermal-partition")
    return float(p["shear_to_chip"]), float(p["rake_to_tool"]), float(p["flank_to_tool"])


@functools.lru_cache(maxsize=1)
def _tool_params() -> dict:
    return dict(params("thermal-tool"))


@dataclass
class FractionTable:
    """The fraction tables of one tool slot as copied from ProcessCoefficients."""

    h_grid_m: np.ndarray
    shear_to_chip: np.ndarray        # [n_group, n_h]
    rake_to_tool: np.ndarray
    flank_to_tool: np.ndarray
    chip_ratio: np.ndarray           # [n_group, n_h] (may be empty)
    shear_angle_rad: np.ndarray      # [n_group, n_h] (may be empty)
    version: int = 0

    @classmethod
    def from_coefficients(cls, coeff) -> "FractionTable | None":
        if coeff is None:
            return None
        h = np.asarray(coeff.h_grid_m, dtype=float)
        tabs = [np.asarray(getattr(coeff, name), dtype=float) for name in
                ("frac_shear_to_chip", "frac_rake_to_tool", "frac_flank_to_tool")]
        if h.size == 0 or any(t.ndim != 2 or t.shape[1] != h.size or t.shape[0] == 0 for t in tabs):
            return None
        ratio = np.asarray(coeff.chip_ratio, dtype=float)
        phi = np.asarray(coeff.shear_angle_rad, dtype=float)
        return cls(h.copy(), tabs[0].copy(), tabs[1].copy(), tabs[2].copy(),
                   ratio.copy() if ratio.shape == tabs[0].shape else np.zeros((0, 0)),
                   phi.copy() if phi.shape == tabs[0].shape else np.zeros((0, 0)), int(coeff.version))

    def weights(self, group: np.ndarray, h: np.ndarray):
        """(group index, lower node, weight) of each element on the log-h grid (clamped), reused for
        every table of the slot."""
        groups = np.clip(np.asarray(group, dtype=np.int64), 0, self.shear_to_chip.shape[0] - 1)
        log_grid = np.log(np.maximum(self.h_grid_m, 1e-300))
        log_h = np.clip(np.log(np.maximum(h, 1e-300)), log_grid[0], log_grid[-1])
        if log_grid.size == 1:
            return groups, np.zeros(h.shape, dtype=np.int64), np.zeros(h.shape)
        lower = np.clip(np.searchsorted(log_grid, log_h, side="right") - 1, 0, log_grid.size - 2)
        weight = (log_h - log_grid[lower]) / (log_grid[lower + 1] - log_grid[lower])
        return groups, lower, weight

    def at(self, table: np.ndarray, where) -> np.ndarray:
        """Interpolate a [n_group, n_h] table with :meth:`weights` (linear in log h)."""
        groups, lower, weight = where
        if table.shape[1] == 1:
            return table[groups, 0]
        return table[groups, lower] * (1.0 - weight) + table[groups, lower + 1] * weight


@dataclass
class ContactHeat:
    """Where one contact's heat of the step goes (J), and the per-element tool inputs."""

    contact: int
    tool_slot: int
    workpiece: int
    generated_J: float = 0.0
    chip_J: float = 0.0
    tool_J: float = 0.0
    work_J: float = 0.0
    tool_rake_elem_J: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tool_flank_elem_J: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    work_elem_J: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    h_mean_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    compression: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    shear_angle_rad: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    fallback: bool = False


def _shares(energy: float, candidates: list[np.ndarray], valid: np.ndarray) -> np.ndarray:
    """Split ``energy`` with the first candidate weight array that is positive somewhere."""
    for weights in candidates:
        w = np.where(valid, np.maximum(weights, 0.0), 0.0)
        total = w.sum()
        if total > 0.0:
            return energy * w / total
    count = int(valid.sum())
    if count == 0:
        return np.zeros(valid.shape)
    return np.where(valid, energy / count, 0.0)


def split_contact(cut, removal_workpiece: int, table: FractionTable | None, edges: EdgeModel | None,
                  vb_m: np.ndarray | None) -> ContactHeat:
    """Partition one CuttingContact's zone heat (bus record of the step)."""
    out = ContactHeat(int(cut.contact), int(cut.tool_slot), int(removal_workpiece))
    q_shear = float(cut.heat_shear_J)
    q_rake = float(cut.heat_rake_J)
    q_flank = float(cut.heat_flank_J)
    q_plough = float(cut.ploughing_J)
    out.generated_J = q_shear + q_rake + q_flank + q_plough
    in_cut = np.asarray(cut.in_cut_time_s, dtype=float)
    n_elem = in_cut.size
    if edges is not None and n_elem and tuple(np.asarray(edges.elem_r_m).shape) == in_cut.shape:
        valid = (np.asarray(edges.elem_valid) != 0)
        db = np.asarray(edges.elem_db_m, dtype=float)
        radius = np.asarray(edges.elem_r_m, dtype=float)
        group = np.asarray(edges.elem_group)
        edge_radius = np.asarray(edges.elem_edge_radius_m, dtype=float)
    else:
        shape = in_cut.shape if n_elem else (1, 1)
        valid = np.ones(shape, dtype=bool)
        db = np.ones(shape)
        radius = np.ones(shape)
        group = np.zeros(shape, dtype=np.int32)
        edge_radius = np.zeros(shape)
        if not n_elem:
            in_cut = np.zeros(shape)
    shape = valid.shape
    int_h = _per_elem(cut.int_h_dt, shape)
    h_mean = np.where(in_cut > 0.0, int_h / np.where(in_cut > 0.0, in_cut, 1.0), 0.0)
    p = _tool_params()
    if table is not None:
        where = table.weights(group, h_mean)
        f_sc = table.at(table.shear_to_chip, where)
        f_rt = table.at(table.rake_to_tool, where)
        f_ft = table.at(table.flank_to_tool, where)
        ratio = table.at(table.chip_ratio, where) if table.chip_ratio.size else \
            np.full(shape, p["default_compression_ratio"])
        phi = table.at(table.shear_angle_rad, where) if table.shear_angle_rad.size else np.zeros(shape)
    else:
        sc, rt, ft = fallback_fractions()
        f_sc = np.full(shape, sc)
        f_rt = np.full(shape, rt)
        f_ft = np.full(shape, ft)
        ratio = np.full(shape, p["default_compression_ratio"])
        phi = np.zeros(shape)
        out.fallback = out.generated_J > 0.0
    f_sc = np.clip(f_sc, 0.0, 1.0)
    f_rt = np.clip(f_rt, 0.0, 1.0)
    f_ft = np.clip(f_ft, 0.0, 1.0)
    ratio = np.where(ratio > 0.0, ratio, p["default_compression_ratio"])
    zeta = np.where(ratio < 1.0, 1.0 / ratio, ratio)      # chip ratio h/h_c or compression h_c/h
    contact_factor = np.minimum(zeta ** p["poletika_exponent"], p["max_contact_ratio"])
    vb = np.zeros(shape) if vb_m is None or np.asarray(vb_m).shape != shape else np.asarray(vb_m, dtype=float)
    flank_width = vb + p["flank_contact_edge_factor"] * edge_radius
    rake_sig = _per_elem(cut.int_sigma_v_rake, shape)
    flank_sig = _per_elem(cut.int_sigma_v_flank, shape)
    rake_slide = _per_elem(cut.rake_sliding_m, shape)
    flank_slide = _per_elem(cut.flank_sliding_m, shape)
    shear_w = int_h * db * radius
    time_w = in_cut * db * radius
    qs = _shares(q_shear, [shear_w, time_w], valid)
    qr = _shares(q_rake, [rake_sig * h_mean * contact_factor * db, rake_slide * h_mean * contact_factor * db,
                          shear_w, time_w], valid)
    qf = _shares(q_flank, [flank_sig * flank_width * db, flank_slide * flank_width * db, time_w], valid)
    qp = _shares(q_plough, [flank_sig * flank_width * db, flank_slide * flank_width * db, time_w], valid)
    out.chip_J = float(np.sum(qs * f_sc) + np.sum(qr * (1.0 - f_rt)))
    out.tool_J = float(np.sum(qr * f_rt) + np.sum((qf + qp) * f_ft))
    out.work_J = float(np.sum(qs * (1.0 - f_sc)) + np.sum((qf + qp) * (1.0 - f_ft)))
    out.tool_rake_elem_J = qr * f_rt
    out.tool_flank_elem_J = (qf + qp) * f_ft
    out.work_elem_J = qs * (1.0 - f_sc) + (qf + qp) * (1.0 - f_ft)
    out.h_mean_m = h_mean
    out.compression = zeta
    out.shear_angle_rad = phi
    return out


def _per_elem(values, shape) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        return np.zeros(shape)
    return array
