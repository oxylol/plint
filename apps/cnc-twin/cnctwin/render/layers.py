"""Colour layers: a physical quantity of the run painted onto the workpiece and the tool with a colour map and
a colour bar (docs/models/render.md "Layers").

========== ============================================================ ================= ======
layer      part (per IPW vertex)                                        tool               unit
========== ============================================================ ================= ======
temperature keyframe ``work_temp_K`` (3-D) or ``surface_temp_K`` (top)  thermal edge temp  degC
chip_pile  - (the pile height field is drawn and coloured)              -                  mm
coolant_reach wetted cells (0 or 100 %)                                  -                  %
surface_finish ``Ra_m`` of the final brick attributes                    -                  um
deviation  ``deviation_m`` of the final brick attributes (diverging)     -                  um
deflection -                                                            keyframe tool_deflection_m  um
wear       ``generating_VB_m`` (VB of the edge that made the surface)   VB of the teeth    um
========== ============================================================ ================= ======

Values are SI inside; the colour bar shows display units (degC, mm, um, %). Colour bar limits for a video
are fixed over the whole run (:func:`run_range`), so the colours of one frame compare with the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import ZERO_DEGC_K
from . import colour as C
from . import params as P
from . import shapes as SH
from .rundir import RunData

LAYERS = ("temperature", "chip_pile", "coolant_reach", "surface_finish", "deviation", "deflection", "wear")


@dataclass(frozen=True)
class LayerSpec:
    name: str
    title: str
    unit: str
    scale: float          # display = SI * scale + offset
    offset: float
    symmetric: bool

    def display(self, values):
        return np.asarray(values, dtype=np.float64) * self.scale + self.offset

    def cmap(self) -> str:
        return str(P.mapping(P.VIEW, "layer_colormaps")[self.name])


SPECS = {
    "temperature": LayerSpec("temperature", "Temperature", "degC", 1.0, -ZERO_DEGC_K, False),
    "chip_pile": LayerSpec("chip_pile", "Chip pile height", "mm", 1e3, 0.0, False),
    "coolant_reach": LayerSpec("coolant_reach", "Coolant reach", "%", 100.0, 0.0, False),
    "surface_finish": LayerSpec("surface_finish", "Surface finish Ra", "um", 1e6, 0.0, False),
    "deviation": LayerSpec("deviation", "Deviation from nominal", "um", 1e6, 0.0, True),
    "deflection": LayerSpec("deflection", "Tool deflection", "um", 1e6, 0.0, False),
    "wear": LayerSpec("wear", "Flank wear VB", "um", 1e6, 0.0, False),
}

TOOL_TEMP_CHANNELS = ("thermal.tool_edge_max_K", "thermal.edge_temp_K")
WEAR_CHANNELS = ("wear.VB_m",)


def spec(name: str) -> LayerSpec:
    if name not in SPECS:
        raise ValueError(f"unknown layer {name!r} (one of {', '.join(LAYERS)})")
    return SPECS[name]


# ----------------------------------------------------------------------------- grids of a keyframe
def surface_temperature(fields: dict, points: np.ndarray) -> np.ndarray | None:
    """Workpiece temperature (K) at ``points`` from the keyframe fields, or None."""
    if "work_temp_K" in fields and "work_temp_origin_S_m" in fields and "work_temp_cell_m" in fields:
        grid = np.asarray(fields["work_temp_K"], dtype=np.float64)
        if grid.ndim == 3 and grid.size:
            return SH.trilinear_grid(grid, fields["work_temp_origin_S_m"], float(np.ravel(fields["work_temp_cell_m"])[0]),
                                     points)
    if "surface_temp_K" in fields and "surface_origin_S_m" in fields and "surface_cell_m" in fields:
        grid = np.asarray(fields["surface_temp_K"], dtype=np.float64)
        if grid.ndim == 2 and grid.size:
            # [ny, nx] on the top face: rows along y, columns along x
            return SH.bilinear_grid(grid.T, fields["surface_origin_S_m"], float(np.ravel(fields["surface_cell_m"])[0]),
                                    points)
    return None


@dataclass
class PileGrid:
    origin: np.ndarray        # corner of cell (0, 0) at w = 0 (S)
    axes: np.ndarray          # columns e_u, e_v, e_w (S)
    cell_m: float
    height_m: np.ndarray      # [nu, nv]


def pile_grid(fields: dict, chips: dict) -> PileGrid | None:
    """Settled chip heights: the chips file's pile grid, else the keyframe field on the surface grid."""
    if "pile_height_m" in chips and "pile_origin_S_m" in chips and "pile_cell_m" in chips:
        h = np.asarray(chips["pile_height_m"], dtype=np.float64)
        axes = np.asarray(chips.get("pile_axes_S", np.eye(3)), dtype=np.float64).reshape(3, 3)
        if h.ndim == 2 and h.size:
            return PileGrid(np.asarray(chips["pile_origin_S_m"], dtype=np.float64).reshape(3), axes,
                            float(np.ravel(chips["pile_cell_m"])[0]), h)
    if "pile_height_m" in fields and "surface_origin_S_m" in fields and "surface_cell_m" in fields:
        h = np.asarray(fields["pile_height_m"], dtype=np.float64)
        if h.ndim == 2 and h.size:
            origin = np.asarray(fields["surface_origin_S_m"], dtype=np.float64).reshape(3).copy()
            origin[2] = 0.0
            return PileGrid(origin, np.eye(3), float(np.ravel(fields["surface_cell_m"])[0]), h.T)
    return None


def pile_height_at(grid: PileGrid, points: np.ndarray) -> np.ndarray:
    """Pile height under ``points`` (nearest cell along e_u, e_v; 0 outside the grid)."""
    rel = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - grid.origin) @ grid.axes
    iu = np.floor(rel[:, 0] / grid.cell_m).astype(np.int64)
    iv = np.floor(rel[:, 1] / grid.cell_m).astype(np.int64)
    nu, nv = grid.height_m.shape
    inside = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    out = np.zeros(len(rel))
    out[inside] = np.nan_to_num(grid.height_m[iu[inside], iv[inside]], nan=0.0)
    return out


def wetted_at(fields: dict, points: np.ndarray) -> np.ndarray | None:
    """1 where the coolant wets the surface under the point, 0 where dry; None without a wetted grid."""
    if "wetted" not in fields:
        return None
    grid = np.asarray(fields["wetted"]).astype(np.float64)
    if grid.ndim != 2 or not grid.size:
        return None
    if "wetted_origin_S_m" in fields and "wetted_cell_m" in fields:
        origin = np.asarray(fields["wetted_origin_S_m"], dtype=np.float64).reshape(3)
        cell = float(np.ravel(fields["wetted_cell_m"])[0])
        g = grid
    elif "surface_origin_S_m" in fields and "surface_cell_m" in fields:
        origin = np.asarray(fields["surface_origin_S_m"], dtype=np.float64).reshape(3)
        cell = float(np.ravel(fields["surface_cell_m"])[0])
        g = grid.T                       # [ny, nx] on the surface grid
    else:
        return None
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    i = np.clip(np.floor((p[:, 0] - origin[0]) / cell).astype(np.int64), 0, g.shape[0] - 1)
    j = np.clip(np.floor((p[:, 1] - origin[1]) / cell).astype(np.int64), 0, g.shape[1] - 1)
    return g[i, j]


# ----------------------------------------------------------------------------- tool values
def _series_at(run: RunData, columns: list[str], tick: int) -> float | None:
    """Largest finite value over ``columns`` (teeth) of the last macro row at or before ``tick``."""
    values = [v for v in run.series_value_at("macro", tuple(columns), tick).values() if np.isfinite(v)]
    return float(max(values)) if values else None


def tool_temperature(run: RunData, tick: int) -> float | None:
    for name in TOOL_TEMP_CHANNELS:
        columns = run.channel_columns(name)
        if columns:
            value = _series_at(run, columns, tick)
            if value is not None:
                return value
    return None


def tool_wear(run: RunData, tick: int) -> float | None:
    for name in WEAR_CHANNELS:
        columns = run.channel_columns(name)
        if columns:
            value = _series_at(run, columns, tick)
            if value is not None:
                return value
    return None


def tool_deflection(fields: dict) -> float | None:
    if "tool_deflection_m" not in fields:
        return None
    d = np.asarray(fields["tool_deflection_m"], dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(d[0])) if len(d) else None


# ----------------------------------------------------------------------------- per frame
@dataclass
class LayerValues:
    spec: LayerSpec
    part: np.ndarray | None        # SI per IPW vertex (NaN: no value)
    tool: float | None             # SI
    chips: np.ndarray | None       # SI per chip (temperature only)


def evaluate(run: RunData, name: str, k: int | None, tick: int, vertices: np.ndarray, attrs: dict,
             fields: dict, chips: dict) -> LayerValues:
    s = spec(name)
    part = tool = chip_values = None
    if name == "temperature":
        part = surface_temperature(fields, vertices) if len(vertices) else None
        tool = tool_temperature(run, tick)
        if "temp_K" in chips:
            chip_values = np.asarray(chips["temp_K"], dtype=np.float64).reshape(-1)
    elif name == "chip_pile":
        part = None                      # the pile itself carries the colours (compose._add_pile)
    elif name == "coolant_reach":
        part = wetted_at(fields, vertices) if len(vertices) else None
    elif name == "surface_finish":
        part = attrs.get("Ra_m")
    elif name == "deviation":
        part = attrs.get("deviation_m")
    elif name == "deflection":
        tool = tool_deflection(fields)
    elif name == "wear":
        part = attrs.get("generating_VB_m")
        tool = tool_wear(run, tick)
    return LayerValues(s, part, tool, chip_values)


def run_range(run: RunData, name: str) -> tuple[float, float]:
    """Display-unit colour bar limits of a layer over the whole run (all keyframes, the final attributes and the
    tool series): ``render.range_percentiles`` of the values, symmetric for deviation."""
    s = spec(name)
    values = []
    if name in ("surface_finish", "deviation", "wear") and run.keyframes:
        ipw = run.ipw(len(run.keyframes) - 1)
        key = {"surface_finish": "Ra_m", "deviation": "deviation_m", "wear": "generating_VB_m"}[name]
        if key in ipw.attrs:
            values.append(ipw.attrs[key])
    for k in range(len(run.keyframes)):
        fields = run.fields(k)
        if name == "temperature":
            for key in ("work_temp_K", "surface_temp_K"):
                if key in fields:
                    values.append(np.asarray(fields[key], dtype=np.float64).reshape(-1))
                    break
            chips = run.chips(k)
            if "temp_K" in chips:
                values.append(np.asarray(chips["temp_K"], dtype=np.float64).reshape(-1))
        elif name == "chip_pile":
            grid = pile_grid(fields, run.chips(k))
            if grid is not None:
                values.append(grid.height_m.reshape(-1))
        elif name == "coolant_reach":
            values.append(np.array([0.0, 1.0]))
        elif name == "deflection":
            d = tool_deflection(fields)
            if d is not None:
                values.append(np.array([0.0, d]))
    if name in ("temperature", "wear"):
        columns = [c for ch in (TOOL_TEMP_CHANNELS if name == "temperature" else WEAR_CHANNELS)
                   for c in run.channel_columns(ch)]
        if columns:
            data = run.series_all("macro", tuple(columns))
            for c in columns:
                if c in data:
                    values.append(data[c])
    if not values:
        return (0.0, 1.0) if not s.symmetric else (-1.0, 1.0)
    v = s.display(np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in values]))
    if name == "coolant_reach":
        return 0.0, 100.0
    return C.robust_range(v, symmetric=s.symmetric)


def colours(values: np.ndarray | None, spec_: LayerSpec, vmin: float, vmax: float, fallback_srgb,
            n: int) -> np.ndarray:
    """sRGB per item: the colour map where a value exists, the fall-back colour elsewhere."""
    out = np.broadcast_to(np.asarray(fallback_srgb, dtype=np.float64), (n, 3)).copy()
    if values is None:
        return out
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    ok = np.isfinite(v)
    if np.any(ok):
        out[ok] = C.apply_colormap(spec_.display(v[ok]), vmin, vmax, spec_.cmap())
    return out
