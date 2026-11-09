"""Tool, holder and cutting-edge temperatures of one tool slot (docs/models/thermal.md "Tool").

Body: axial 1-D finite volumes of the tool (fluted part, neck, shank) and of its holder (nose to
gauge line, plus a lumped taper node in the spindle), coupled radially over the clamp length by a
contact conductance; lateral convection (rotating cylinder in air, coolant on the fluted part,
through-tool channels), radiation to the enclosure, taper contact to the spindle nose. Turning
tools: insert and holder bar, seat contact, the clamped part of the bar to the turret. Implicit
(backward Euler), banded, exactly conservative (kernels.ladder_step).

Edge: temperature at the cutting edge, over the rake contact (l_c = h zeta^n) and over the flank
contact (the partition's: wear land + the edge arc below the stagnation point) per element = body temperature at the
element + the local field of the periodic heating per revolution (Stephenson & Ali 1992; kernels.edge_temperatures),
at Gauss points of the in-cut interval (ToolHeat.q_weights) and the cycle amplitude. The periodic heating (flux, pass
time, period, cycles) comes from each element's heat history in run time (running sums of its passes, the period of a
rotating tool's revolution), so it does not depend on the macro step.

Cutting region (optional, :meth:`ToolThermal.enable_field`): a 3-D finite-volume field of the
insert or of the tooth tip (toolfield.py), solved with the ladder as one system; the elements it
carries take their edge temperatures from it (cycle mean) plus the periodic deviation of the edge
kernel, and :meth:`ToolThermal.temperature_at` gives the tool temperature at any point.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EdgeModel
from ..core.bus import ToolHeat
from ..core.errors import ConservationError, InputError
from ..db.models import params
from . import convection as cv
from . import kernels as K
from . import props
from .abi import (EDGE_CONTINUOUS, EDGE_INTERRUPTED, EDGE_NONE, EDGE_OUT_DEVIATION, EDGE_OUT_TEMPERATURE, FD_AIR,
                  FD_COOL, FD_DU, FD_ITER, FD_RES, FD_SP, FD_SRC, FD_U0, FD_U1, FS_TOL, LD_AIR, LD_COOL, LD_DU, LD_SP,
                  LD_SRC, LD_U0, LD_U1, N_LD, Ladder, MatTable, f64, new_edge_input, new_edge_output)
from .ledger import SolverLedger
from .response import edge_response, gauss_legendre, heating_phase_nodes
from .toolfield import ToolField, unroll_elements

TURNING_TYPES = ("turning_insert", "grooving_insert", "parting_insert", "threading_insert", "boring_bar",
                 "form_tool")

RISE_NAMES = ("rake", "peak", "flank", "amp", "rake_entry", "flank_entry")     # ToolThermal.rise
DEV_NAMES = ("rake", "peak", "flank", "rake_entry", "flank_entry")             # ToolThermal.dev

NODE_TOOL_EXPOSED = 0
NODE_TOOL_INSIDE = 1
NODE_HOLDER = 2
NODE_END = 3          # taper in the spindle, or the clamped bar of a turning tool


@functools.lru_cache(maxsize=1)
def _p() -> dict:
    return dict(params("thermal-tool"))


def _get(obj, *names, default=None):
    """First non-None attribute/key along a path of names (SIMap or mapping)."""
    cur = obj
    for name in names:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(name)
        else:
            cur = getattr(cur, name, None)
    return default if cur is None else cur


def _num(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ToolSpec:
    """Plain geometry of one tool instance and its holder (SI)."""

    kind: str = "rotating"                 # rotating | turning
    diameter_m: float = 0.01
    flutes: int = 2
    flute_length_m: float = 0.02
    overall_length_m: float = 0.07
    shank_diameter_m: float = 0.01
    neck_diameter_m: float = 0.0
    neck_length_m: float = 0.0
    fluted_area_m2: float = 0.0
    fluted_perimeter_m: float = 0.0
    core_diameter_m: float = 0.0           # web / core of the fluted section (0: not given)
    coolant_holes: int = 0
    coolant_hole_diameter_m: float = 0.0
    stickout_m: float = 0.035
    tool_material_id: str = "carbide-submicron-10co"
    insert_ic_m: float = 0.0127
    insert_thickness_m: float = 0.00476
    bar_h_m: float = 0.025
    bar_b_m: float = 0.025
    bar_overhang_m: float = 0.03
    holder_gauge_length_m: float = 0.08
    holder_profile_m: list = field(default_factory=list)     # [[z from nose, diameter], ...]
    holder_nose_diameter_m: float = 0.0
    holder_clamp_length_m: float = 0.0
    holder_mass_kg: float = 0.0
    taper: str = ""
    holder_material_id: str = ""


def tool_spec(entry, tool_item=None, holder_item=None) -> ToolSpec:
    """ToolSpec of a project tools.list entry (its resolved ``tool`` and ``holder`` items)."""
    p = _p()
    tool_item = tool_item if tool_item is not None else _get(entry, "tool")
    holder_item = holder_item if holder_item is not None else _get(entry, "holder")
    geo = _get(tool_item, "geometry")
    kind = "turning" if str(_get(tool_item, "type", default="")) in TURNING_TYPES else "rotating"
    spec = ToolSpec(kind=kind)
    tm = _get(tool_item, "tool_material")
    tm_id = _get(tm, "id")
    if tm_id is None:
        cls = str(_get(tm, "class", default="carbide"))
        names = list(p["class_names"])
        tm_id = list(p["class_materials"])[names.index(cls)] if cls in names else list(p["class_materials"])[0]
    spec.tool_material_id = str(tm_id)
    spec.holder_material_id = str(p["holder_material"])
    if kind == "turning":
        insert = _get(geo, "insert")
        bar = _get(geo, "holder")
        spec.insert_ic_m = _num(_get(insert, "ic_m"), p["default_insert_ic_m"])
        spec.insert_thickness_m = _num(_get(insert, "thickness_m"), p["default_insert_thickness_m"])
        shank = _get(bar, "shank_m")
        if isinstance(shank, (list, tuple)) or hasattr(shank, "__len__"):
            values = [float(v) for v in shank]
            spec.bar_h_m, spec.bar_b_m = values[0], values[-1]
        else:
            spec.bar_h_m = spec.bar_b_m = p["default_bar_section_m"]
        spec.bar_overhang_m = _num(_get(bar, "overhang_m"), _num(_get(entry, "stickout_m"), p["default_bar_overhang_m"]))
        return spec
    d = _num(_get(geo, "diameter_m"), None)
    if d is None or d <= 0.0:
        raise InputError("thermal: tool geometry needs diameter_mm")
    spec.diameter_m = d
    spec.flutes = int(_num(_get(geo, "flutes"), 2))
    spec.flute_length_m = _num(_get(geo, "flute_length_m"), _num(_get(geo, "cutting_length_m"), 2.0 * d))
    spec.shank_diameter_m = _num(_get(geo, "shank", "diameter_m"), d)
    spec.neck_diameter_m = _num(_get(geo, "neck", "diameter_m"), 0.0)
    spec.neck_length_m = _num(_get(geo, "neck", "length_m"), 0.0)
    spec.stickout_m = _num(_get(entry, "stickout_m"), None) or 0.0
    clamp = _num(_get(holder_item, "clamp_length_m"), None)
    if clamp is None or clamp <= 0.0:
        clamp = p["holder_clamp_ratio"] * spec.shank_diameter_m
    spec.holder_clamp_length_m = clamp
    overall = _num(_get(geo, "overall_length_m"), None)
    if overall is None or overall <= 0.0:
        overall = spec.stickout_m + clamp
    spec.overall_length_m = overall
    if spec.stickout_m <= 0.0 or spec.stickout_m > overall:
        spec.stickout_m = max(overall - clamp, 0.5 * overall)
    spec.core_diameter_m = _num(_get(geo, "core_diameter_m"), 0.0) or 0.0
    gullet = _num(_get(geo, "gullet_area_m2"), None)
    full = math.pi * d * d / 4.0
    if gullet is not None and gullet > 0.0:
        spec.fluted_area_m2 = max(full - spec.flutes * gullet, 0.2 * full)
    else:
        spec.fluted_area_m2 = p["fluted_solid_fraction"] * full
    spec.fluted_perimeter_m = p["fluted_perimeter_factor"] * math.pi * d
    holes = _get(geo, "coolant_holes")
    if holes is not None:
        spec.coolant_holes = int(_num(_get(holes, "count"), 0))
        spec.coolant_hole_diameter_m = _num(_get(holes, "diameter_m"), 0.0)
    spec.holder_gauge_length_m = _num(_get(holder_item, "gauge_length_m"), 2.0 * clamp)
    profile = _get(holder_item, "profile_m")
    if profile is not None:
        rows = [[float(r[0]), float(r[1])] for r in list(profile)]
        spec.holder_profile_m = sorted(rows)
    spec.holder_nose_diameter_m = _num(_get(holder_item, "nose_diameter_m"),
                                       p["holder_nose_diameter_ratio"] * spec.shank_diameter_m)
    spec.holder_mass_kg = _num(_get(holder_item, "mass_kg"), 0.0)
    spec.taper = str(_get(holder_item, "taper", default=""))
    return spec


def taper_dims(taper: str) -> tuple[float, float, float]:
    """(contact area m2, frustum volume m3, mean diameter m) of a spindle taper designation."""
    p = _p()
    text = taper.upper()
    sizes = re.findall(r"\d+", text)
    d1, length, ratio = None, None, None
    if "HSK" in text and sizes:
        size = int(sizes[0])
        if size in [int(s) for s in p["hsk_sizes"]]:
            idx = [int(s) for s in p["hsk_sizes"]].index(size)
            d1, length, ratio = p["hsk_gauge_diameter_m"][idx], p["hsk_length_m"][idx], 0.1
    elif sizes:
        size = int(sizes[-1])
        if size in [int(s) for s in p["steep_taper_sizes"]]:
            idx = [int(s) for s in p["steep_taper_sizes"]].index(size)
            d1, length, ratio = p["steep_taper_gauge_diameter_m"][idx], p["steep_taper_length_m"][idx], 7.0 / 24.0
    if d1 is None:               # unrecognised: the size-40 steep taper
        idx = [int(s) for s in p["steep_taper_sizes"]].index(40)
        d1, length, ratio = p["steep_taper_gauge_diameter_m"][idx], p["steep_taper_length_m"][idx], 7.0 / 24.0
    d2 = d1 - ratio * length
    mean = 0.5 * (d1 + d2)
    area = math.pi * mean * length
    volume = math.pi * length * (d1 * d1 + d1 * d2 + d2 * d2) / 12.0
    return area, volume, mean


def strain_from(mat: MatTable, temp_K, ref_K: float):
    """Free thermal strain between ref_K and temp_K from a mean-CTE (from 20 degC) table (arrays)."""
    from ..units import REFERENCE_TEMP_K
    t = np.asarray(temp_K, dtype=float)
    return (np.interp(t, mat.T_K, mat.alpha_1_K) * (t - REFERENCE_TEMP_K)
            - float(np.interp(ref_K, mat.T_K, mat.alpha_1_K)) * (ref_K - REFERENCE_TEMP_K))


@dataclass
class LadderLayout:
    z_m: np.ndarray                # node centre along the tool axis from the tip (rotating) / along insert+bar
    kind: np.ndarray               # NODE_*
    diameter_m: np.ndarray         # outer diameter for the convection correlation
    perimeter_m: np.ndarray        # lateral perimeter (0: not exposed)
    fluted: np.ndarray             # int8: wetted by the coolant jet
    channel: np.ndarray            # int8: carries a through-tool coolant channel
    tool_nodes: np.ndarray         # tool-strand node indices, tip first
    holder_nodes: np.ndarray       # holder-strand node indices (nose first), end node last
    stick_nodes: np.ndarray        # exposed tool nodes
    holder_strand: np.ndarray      # holder nodes between nose and gauge line
    end_node: int
    end_conductance_W_K: float     # taper-spindle or bar-turret contact conductance
    turning: bool


def _profile_diameter(spec: ToolSpec, z_from_nose: float) -> float:
    rows = spec.holder_profile_m
    if rows:
        zs = [r[0] for r in rows]
        ds = [r[1] for r in rows]
        return float(np.interp(z_from_nose, zs, ds))
    return spec.holder_nose_diameter_m


def build_ladder(spec: ToolSpec, mat_tool: MatTable, mat_holder: MatTable, cell_m: float,
                 temp0_K: float) -> tuple[Ladder, LadderLayout]:
    """Nodes, links and masses of the tool/holder ladder at a uniform initial temperature."""
    p = _p()
    nodes = []          # dicts: z, dz, area, perim, diam, kind, fluted, channel, mat
    links = []          # (a, b, kind, g)
    h_contact = p["shank_contact_conductance_W_m2K"]
    if spec.kind == "turning":
        ic = spec.insert_ic_m
        n_ins = max(1, int(round(ic / cell_m)))
        dz = ic / n_ins
        for i in range(n_ins):
            nodes.append(dict(z=(i + 0.5) * dz, dz=dz, area=ic * spec.insert_thickness_m,
                              perim=2.0 * (ic + spec.insert_thickness_m), diam=ic, kind=NODE_TOOL_EXPOSED, fluted=1,
                              channel=0, mat=0))
        clamp = p["bar_clamp_length_ratio"] * max(spec.bar_h_m, spec.bar_b_m)
        n_over = max(1, int(round(spec.bar_overhang_m / cell_m)))
        n_cl = max(1, int(round(clamp / cell_m)))
        section = spec.bar_h_m * spec.bar_b_m
        perim = 2.0 * (spec.bar_h_m + spec.bar_b_m)
        z0 = ic
        for i in range(n_over):
            dzb = spec.bar_overhang_m / n_over
            nodes.append(dict(z=z0 + (i + 0.5) * dzb, dz=dzb, area=section, perim=perim, diam=max(spec.bar_h_m, spec.bar_b_m),
                              kind=NODE_HOLDER, fluted=0, channel=0, mat=1))
        z0 += spec.bar_overhang_m
        for i in range(n_cl):
            dzc = clamp / n_cl
            nodes.append(dict(z=z0 + (i + 0.5) * dzc, dz=dzc, area=section, perim=0.0, diam=max(spec.bar_h_m, spec.bar_b_m),
                              kind=NODE_END, fluted=0, channel=0, mat=1))
        for a in range(len(nodes) - 1):
            if a == n_ins - 1:
                links.append((a, a + 1, 1, h_contact * p["insert_seat_area_ratio"] * ic * ic))
            else:
                links.append((a, a + 1, 0, 0.0))
        end_g = h_contact * perim * clamp
        layout_nodes = nodes
        turning = True
    else:
        turning = False
        length = spec.overall_length_m
        stick = min(spec.stickout_m, length)
        inside = min(length - stick, spec.holder_gauge_length_m)
        n_stick = max(1, int(round(stick / cell_m)))
        dz1 = stick / n_stick
        n_in = int(round(inside / cell_m))
        if inside > 0.0 and n_in == 0:
            n_in = 1
        dz2 = inside / n_in if n_in else 0.0
        rest = max(spec.holder_gauge_length_m - inside, 0.0)
        n_rest = max(1, int(round(rest / cell_m))) if rest > 0.0 else 0
        dz3 = rest / n_rest if n_rest else 0.0

        def tool_node(zc, dzn, kind):
            if zc <= spec.flute_length_m:
                area, perim, diam, fluted = spec.fluted_area_m2, spec.fluted_perimeter_m, spec.diameter_m, 1
            elif spec.neck_diameter_m > 0.0 and zc <= spec.flute_length_m + spec.neck_length_m:
                d = spec.neck_diameter_m
                area, perim, diam, fluted = math.pi * d * d / 4.0, math.pi * d, d, 0
            else:
                d = spec.shank_diameter_m
                area, perim, diam, fluted = math.pi * d * d / 4.0, math.pi * d, d, 0
            if kind != NODE_TOOL_EXPOSED:
                perim = 0.0
            channel = 1 if spec.coolant_holes > 0 and spec.coolant_hole_diameter_m > 0.0 else 0
            if channel:
                area = max(area - spec.coolant_holes * math.pi * spec.coolant_hole_diameter_m ** 2 / 4.0, 0.1 * area)
            return dict(z=zc, dz=dzn, area=area, perim=perim, diam=diam, kind=kind, fluted=fluted, channel=channel,
                        mat=0)

        def holder_node(zc, dzn):
            z_nose = zc - stick
            d = _profile_diameter(spec, z_nose)
            bore_depth = max(spec.holder_clamp_length_m, inside)
            area = math.pi * d * d / 4.0
            if z_nose < bore_depth:
                area -= math.pi * spec.shank_diameter_m ** 2 / 4.0
            area = max(area, 0.05 * math.pi * d * d / 4.0)
            return dict(z=zc, dz=dzn, area=area, perim=math.pi * d, diam=d, kind=NODE_HOLDER, fluted=0, channel=0,
                        mat=1)

        layout_nodes = []
        for i in range(n_stick):
            layout_nodes.append(tool_node((i + 0.5) * dz1, dz1, NODE_TOOL_EXPOSED))
        tool_in_idx = []
        holder_idx = []
        for i in range(n_in):
            zc = stick + (i + 0.5) * dz2
            tool_in_idx.append(len(layout_nodes))
            layout_nodes.append(tool_node(zc, dz2, NODE_TOOL_INSIDE))
            holder_idx.append(len(layout_nodes))
            layout_nodes.append(holder_node(zc, dz2))
        for i in range(n_rest):
            holder_idx.append(len(layout_nodes))
            layout_nodes.append(holder_node(stick + inside + (i + 0.5) * dz3, dz3))
        area_t, volume_t, mean_t = taper_dims(spec.taper)
        strand_mass = sum(mat_holder.rho_kg_m3[0] * layout_nodes[h]["area"] * layout_nodes[h]["dz"] for h in holder_idx)
        taper_mass = max(spec.holder_mass_kg - strand_mass, mat_holder.rho_kg_m3[0] * volume_t)
        taper_len = volume_t / (math.pi * mean_t * mean_t / 4.0)
        end_index = len(layout_nodes)
        layout_nodes.append(dict(z=stick + spec.holder_gauge_length_m + 0.5 * taper_len, dz=taper_len,
                                 area=math.pi * mean_t * mean_t / 4.0, perim=0.0, diam=mean_t, kind=NODE_END,
                                 fluted=0, channel=0, mat=1, mass=taper_mass))
        tool_seq = list(range(n_stick)) + tool_in_idx
        for a, b in zip(tool_seq, tool_seq[1:]):
            links.append((a, b, 0, 0.0))
        holder_seq = holder_idx + [end_index]
        for a, b in zip(holder_seq, holder_seq[1:]):
            links.append((a, b, 0, 0.0))
        for t_idx, h_idx in zip(tool_in_idx, holder_idx):
            z_nose = layout_nodes[t_idx]["z"] - stick
            dzn = layout_nodes[t_idx]["dz"]
            within = min(max(spec.holder_clamp_length_m - (z_nose - 0.5 * dzn), 0.0), dzn)
            if within > 0.0:
                links.append((t_idx, h_idx, 1, h_contact * math.pi * spec.shank_diameter_m * within))
        if not tool_in_idx:
            first_holder = holder_seq[0]
            links.append((tool_seq[-1], first_holder, 1,
                          h_contact * math.pi * spec.shank_diameter_m * min(spec.holder_clamp_length_m, dz1)))
        end_g = p["taper_contact_conductance_W_m2K"] * area_t
    n = len(layout_nodes)
    m = np.zeros(n)
    for idx, nd in enumerate(layout_nodes):
        rho = (mat_tool if nd["mat"] == 0 else mat_holder).rho_kg_m3[0]
        m[idx] = nd.get("mass", rho * nd["area"] * nd["dz"])
    link_a = np.array([a for a, _, _, _ in links], dtype=np.int32)
    link_b = np.array([b for _, b, _, _ in links], dtype=np.int32)
    bw = int(np.max(np.abs(link_b - link_a))) if links else 1
    if bw > 2:
        raise InputError("thermal: tool ladder bandwidth exceeds 2 (layout error)")
    mat_flag = np.array([nd["mat"] for nd in layout_nodes], dtype=np.int8)
    h0 = np.array([props.h_of_T(mat_tool if f == 0 else mat_holder, temp0_K) for f in mat_flag])
    ladder = Ladder(m=m, h=h0, T=np.full(n, float(temp0_K)), mat=mat_flag,
                    area=np.array([nd["area"] for nd in layout_nodes]), dz=np.array([nd["dz"] for nd in layout_nodes]),
                    link_a=link_a, link_b=link_b, link_kind=np.array([k for _, _, k, _ in links], dtype=np.int8),
                    link_g=np.array([g for _, _, _, g in links]), g_air=f64(n), g_cool=f64(n), g_sp=f64(n),
                    src=f64(n), band=f64((n, 3)), rhs=f64(n), scal=f64(4))
    kinds = np.array([nd["kind"] for nd in layout_nodes], dtype=np.int8)
    tool_nodes = np.array([i for i in range(n) if mat_flag[i] == 0], dtype=np.int64)
    tool_nodes = tool_nodes[np.argsort([layout_nodes[i]["z"] for i in tool_nodes], kind="stable")]
    holder_nodes = np.array([i for i in range(n) if mat_flag[i] == 1], dtype=np.int64)
    layout = LadderLayout(
        z_m=np.array([nd["z"] for nd in layout_nodes]), kind=kinds,
        diameter_m=np.array([nd["diam"] for nd in layout_nodes]),
        perimeter_m=np.array([nd["perim"] for nd in layout_nodes]),
        fluted=np.array([nd["fluted"] for nd in layout_nodes], dtype=np.int8),
        channel=np.array([nd["channel"] for nd in layout_nodes], dtype=np.int8),
        tool_nodes=tool_nodes, holder_nodes=holder_nodes,
        stick_nodes=np.array([i for i in range(n) if kinds[i] == NODE_TOOL_EXPOSED], dtype=np.int64),
        holder_strand=np.array([i for i in range(n) if kinds[i] == NODE_HOLDER], dtype=np.int64),
        end_node=int(n - 1), end_conductance_W_K=float(end_g), turning=turning)
    return ladder, layout


@dataclass
class ToolEnvironment:
    """What surrounds the tool during one step."""

    in_spindle: bool = False
    omega_rad_s: float = 0.0               # |spindle speed| (0 for turning tools and stopped spindles)
    T_air_K: float = 293.15
    T_end_K: float = 293.15                # spindle nose (in the spindle) / turret (turning)
    coolant_on: bool = False
    T_fluid_K: float = 293.15
    h_coolant_W_m2K: float = 0.0           # on the wetted (fluted / insert) surface
    reach: float = 0.0                     # fraction of the wetted surface the coolant reaches
    channel_h_W_m2K: float = 0.0           # through-tool channel wall


# EdgeState.acc rows: running sums of an element's steps (exponential forgetting in run time, edge_window_periods)
ACC_RAKE_J, ACC_FLANK_J, ACC_IN_CUT_S, ACC_ENTRIES, ACC_RAKE_LC, ACC_FLANK_LF, ACC_TIME_S = range(7)
N_ACC = 7


@dataclass
class EdgeState:
    n_cyc: np.ndarray                      # [nt, ne] passes (entries) since the element's engagement began
    t_eng: np.ndarray                      # [nt, ne] contact time of a continuous engagement (s)
    last_in_s: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # [nt, ne] run time at the end of the
    #                                        last step with in-cut time (-inf: never)
    acc: np.ndarray = field(default_factory=lambda: np.zeros((N_ACC, 0, 0)))  # [N_ACC, nt, ne] running sums

    def ensure(self, shape) -> None:
        """Arrays of the EdgeModel's shape (a new or changed edge model starts every element afresh)."""
        if self.n_cyc.shape != tuple(shape):
            self.n_cyc = np.zeros(shape)
            self.t_eng = np.zeros(shape)
        if self.last_in_s.shape != tuple(shape):
            self.last_in_s = np.full(shape, -np.inf)
        if self.acc.shape != (N_ACC,) + tuple(shape):
            self.acc = np.zeros((N_ACC,) + tuple(shape))


class ToolThermal:
    """Thermal state and model of one tool slot."""

    def __init__(self, slot: int, spec: ToolSpec, mat_tool: MatTable, mat_holder: MatTable, cell_m: float,
                 temp0_K: float, time_s: float):
        self.slot = slot
        self.spec = spec
        self.mat_tool = mat_tool
        self.mat_holder = mat_holder
        self.ladder, self.layout = build_ladder(spec, mat_tool, mat_holder, cell_m, temp0_K)
        self.temp_ref_K = float(temp0_K)
        self.time_s = float(time_s)
        self.edge = EdgeState(np.zeros((0, 0)), np.zeros((0, 0)))
        self.ledger = SolverLedger(f"tool.{slot}", u_J=float(K.ladder_energy(self.ladder)))
        self.ledger.u0_J = self.ledger.u_J
        self.last_heat = ToolHeat(body_temp_K=temp0_K, holder_temp_K=temp0_K)
        self.emissivity_tool = cv.emissivity("tool_steel" if spec.tool_material_id.startswith("hss") else None)
        self.emissivity_holder = cv.emissivity("tool_steel")
        self.to_spindle_J = 0.0            # heat into the spindle nose / turret in the last step
        self.to_coolant_J = 0.0
        self.rise = None                   # last local edge rise (rake, peak, flank [nt, ne, nq], amp, rake and
        #                                    flank at the entry into the cut [nt, ne])
        self.edge_calls = 0
        self.edge_next_s = 0.0             # run time of the next edge evaluation (physics.thermal.edge_interval_ms)
        self.field: ToolField | None = None   # 3-D cutting-region field (enable_field), solved with the ladder
        self.field_requested = False
        self.field_settings: dict = {}
        self.field_pending_dt = 0.0        # time accumulated since the last field solve
        self.field_pending_steps = 0
        self.field_every_n = 1             # field solved on every n-th step (field_interval_s = 0)
        self.field_interval_s = 0.0        # field solved at every multiple of this run time (> 0)
        self.field_next_s = 0.0            # run time of the next solve (field_interval_s > 0)
        self.field_pend = None             # element heat accumulated since the last solve (_field_pending)
        self.field_means = None            # [3, nE] rake, flank, edge temperatures of the field's elements
        self.dev = None                    # last periodic deviation of the field's elements (rake, peak, flank,
        #                                    rake and flank at the entry into the cut)

    # ------------------------------------------------------------------ body
    def set_environment(self, env: ToolEnvironment) -> None:
        """Lateral (air incl. radiation, coolant, channel) and end conductances for the coming step."""
        lad = self.ladder
        lay = self.layout
        exposed = lay.perimeter_m > 0.0
        area = lay.perimeter_m * lad.dz
        temp = lad.T
        diam = np.where(exposed, lay.diameter_m, 1.0)
        if env.in_spindle and not lay.turning and env.omega_rad_s > 0.0:
            h_conv = cv.h_rotating_cylinder(env.omega_rad_s, diam, temp, env.T_air_K)
        else:
            h_conv = cv.h_horizontal_cylinder(temp, env.T_air_K, diam)
        eps = np.where(lad.mat == 0, self.emissivity_tool, self.emissivity_holder)
        h_air = h_conv + cv.h_radiation(eps, temp, env.T_air_K)
        wet = exposed & (lay.fluted == 1) & env.coolant_on & (env.h_coolant_W_m2K > 0.0)
        reach = np.where(wet, min(max(env.reach, 0.0), 1.0), 0.0)
        lad.g_air[:] = np.where(exposed, (1.0 - reach) * h_air * area, 0.0)
        lad.g_cool[:] = np.where(exposed, reach * env.h_coolant_W_m2K * area, 0.0)
        if env.coolant_on and env.channel_h_W_m2K > 0.0:
            lad.g_cool[:] += np.where(lay.channel == 1, env.channel_h_W_m2K * self.spec.coolant_holes * math.pi
                                      * self.spec.coolant_hole_diameter_m * lad.dz, 0.0)
        lad.g_sp[:] = 0.0
        if lay.turning:
            ends = lay.kind == NODE_END
            lad.g_sp[:] = np.where(ends, lay.end_conductance_W_K * lad.dz / lad.dz[ends].sum(), 0.0)
        elif env.in_spindle:
            lad.g_sp[lay.end_node] = lay.end_conductance_W_K
        lad.scal[0] = env.T_air_K
        lad.scal[1] = env.T_fluid_K
        lad.scal[2] = env.T_end_K

    def deposit(self, z_m: np.ndarray, energy_J: np.ndarray) -> float:
        """Add element heats at positions z (from the tip) to the tool nodes (linear weights); with the
        cutting-region field, into the field at the edge where it covers z (no contact geometry: half
        on each face over the innermost rings) and into the ladder elsewhere."""
        if self.field is not None:
            return self._deposit_field_at(np.ravel(np.asarray(z_m, dtype=float)), np.ravel(np.asarray(energy_J,
                                                                                                    dtype=float)))
        return self._deposit_ladder(z_m, energy_J)

    def _deposit_ladder(self, z_m, energy_J) -> float:
        lad = self.ladder
        nodes = self.layout.tool_nodes
        z = np.ravel(np.asarray(z_m, dtype=float))
        q = np.ravel(np.asarray(energy_J, dtype=float))
        if self.layout.turning or nodes.size == 1:
            lad.src[nodes[0]] += q.sum()
            return float(q.sum())
        zs = self.layout.z_m[nodes]
        zc = np.clip(z, zs[0], zs[-1])
        i = np.clip(np.searchsorted(zs, zc, side="right") - 1, 0, zs.size - 2)
        t = (zc - zs[i]) / (zs[i + 1] - zs[i])
        np.add.at(lad.src, nodes[i], q * (1.0 - t))
        np.add.at(lad.src, nodes[i + 1], q * t)
        return float(q.sum())

    def _deposit_field_at(self, z: np.ndarray, q: np.ndarray) -> float:
        """Heat without contact geometry into the field: at the edge, half rake / half flank over the
        innermost rings, at the slab of each z (turning: the corner slab); a rotating tool's heat
        beyond the field's height range goes to the ladder."""
        f = self.field
        if q.size == 0:
            return 0.0
        strip = float(f.section.r_edges[min(3, f.section.r_edges.size - 1)])
        ye = f.y_edges
        if self.layout.turning:
            inside = np.ones(q.size, dtype=bool)
            y = np.full(q.size, 0.5 * (ye[0] + ye[1]))
        else:
            zf = f.frames.get("z_mid")
            yf = f.frames.get("y_mid")
            if zf is None or zf.size == 0:
                inside = z <= ye[-1]
                y = np.clip(z, ye[0], ye[-1])
            else:
                order = np.argsort(zf, kind="stable")
                inside = (z >= zf.min() - 1e-12) & (z <= zf.max() + 1e-12)
                y = np.interp(z, zf[order], yf[order])
        total = 0.0
        if np.any(inside):
            slab = np.clip(np.searchsorted(ye, y[inside], side="right") - 1, 0, ye.size - 2)
            e = q[inside]
            total += f.deposit_strips(0.5 * e, 0.5 * e, np.full(e.size, strip), np.full(e.size, strip), ye[slab],
                                      ye[slab + 1], np.zeros(e.size, dtype=np.int64))
        if np.any(~inside):
            total += self._deposit_ladder(z[~inside], q[~inside])
        return total

    def deposit_uniform(self, length_m: float, energy_J: float) -> None:
        """Spread heat over the tool nodes within ``length_m`` of the tip (no element geometry)."""
        nodes = self.layout.tool_nodes
        zs = self.layout.z_m[nodes]
        chosen = nodes[zs <= max(length_m, zs[0])]
        if self.field is not None:
            z = self.layout.z_m[chosen]
            self._deposit_field_at(np.asarray(z, dtype=float), np.full(chosen.size, energy_J / chosen.size))
            return
        for idx in chosen:
            self.ladder.src[idx] += energy_J / chosen.size

    def contact_sizes(self, edges: EdgeModel, heat, vb_m) -> tuple[np.ndarray, np.ndarray]:
        """Rake contact length l_c = h zeta^n (at least the edge radius) and flank contact width per element: the
        partition's flank contact (ContactHeat.flank_length_m: the wear land plus the edge arc below the stagnation
        point, the contact the flank share was computed for), else VB + factor x edge radius (docs/models/thermal.md
        2.2)."""
        p = _p()
        edge_r = np.asarray(edges.elem_edge_radius_m, dtype=float)
        shape = edge_r.shape
        vb = np.asarray(vb_m, dtype=float) if vb_m is not None and np.shape(vb_m) == shape else np.zeros(shape)
        h_mean = heat.h_mean_m if heat is not None and np.shape(heat.h_mean_m) == shape else np.zeros(shape)
        zeta = heat.compression if heat is not None and np.shape(heat.compression) == shape else \
            np.full(shape, p["default_compression_ratio"])
        contact_factor = np.minimum(np.maximum(zeta, 1.0) ** p["poletika_exponent"], p["max_contact_ratio"])
        l_c = np.maximum(h_mean * contact_factor, np.maximum(edge_r, 1e-6))
        flank = getattr(heat, "flank_length_m", None) if heat is not None else None
        if flank is not None and np.shape(flank) == shape and np.any(np.asarray(flank) > 0.0):
            l_f = np.maximum(np.asarray(flank, dtype=float), 1e-6)
        else:
            l_f = np.maximum(vb + p["flank_contact_edge_factor"] * edge_r, 1e-6)
        return l_c, l_f

    def deposit_elements(self, edges: EdgeModel, heat, vb_m=None) -> float:
        """Per-element tool heat of a contact (partition.ContactHeat: tool_rake_elem_J,
        tool_flank_elem_J) into the cutting-region field - the elements it carries, as uniform fluxes
        over their rake and flank contacts (the teeth of a rotating tool together, contact sizes
        heat-weighted over the teeth) - and into the ladder at their height (the others)."""
        q_r = np.asarray(heat.tool_rake_elem_J, dtype=float)
        q_f = np.asarray(heat.tool_flank_elem_J, dtype=float)
        if self.field is None or q_r.shape != np.shape(edges.elem_z_m):
            return self.deposit(edges.elem_z_m, q_r + q_f)
        el = self.field.elements
        if el is None or el.in_field.shape != q_r.shape:
            return self._deposit_field_at(np.ravel(np.asarray(edges.elem_z_m, dtype=float)), np.ravel(q_r + q_f))
        l_c, l_f = self.contact_sizes(edges, heat, vb_m)
        inf = el.in_field
        qr = np.where(inf, q_r, 0.0)[:, el.elem]
        qf = np.where(inf, q_f, 0.0)[:, el.elem]
        # accumulated until the field is solved (one mapping onto the faces per solve, with the heat-weighted
        # contact sizes of the interval: the mean flux over the mean contact)
        pend = self._field_pending(el.elem.size)
        pend[0] += qr.sum(axis=0)
        pend[1] += qf.sum(axis=0)
        pend[2] += (qr * l_c[:, el.elem]).sum(axis=0)
        pend[3] += (qf * l_f[:, el.elem]).sum(axis=0)
        pend[4] = np.minimum(l_c[0, el.elem], l_f[0, el.elem])
        total = float(qr.sum() + qf.sum())
        rest = ~inf & ((q_r != 0.0) | (q_f != 0.0))
        if np.any(rest):
            z = np.asarray(edges.elem_z_m, dtype=float)[rest]
            q = (q_r + q_f)[rest]
            total += self._deposit_field_at(z, q) if self.layout.turning else self._deposit_ladder(z, q)
        return total

    def deposit_contact(self, rake_J: float, flank_J: float, rake_length_m: float, flank_length_m: float,
                        width_m: float, start_m: float = 0.0, *, wedge_angle_rad: float | None = None,
                        edge_radius_m: float | None = None) -> float:
        """Heat of one contact patch into the cutting-region field: the rake heat as a uniform flux over
        [0, rake_length] from the edge x [start, start + width] along the edge, the flank heat over
        [0, flank_length] down the flank face (J; for a rotating tool all teeth together). A field that
        was requested without geometry is built on the first call that gives the wedge angle."""
        if self.field is None and self.field_requested and wedge_angle_rad is not None:
            self._build_field(self._settings(), None, float(wedge_angle_rad), edge_radius_m,
                              (start_m, start_m + width_m))
        if self.field is None:
            raise InputError("thermal: deposit_contact needs the cutting-region field (ToolThermal.enable_field)")
        return self.field.deposit_strips([rake_J], [flank_J], [rake_length_m], [flank_length_m], [start_m],
                                         [start_m + width_m], [0])

    def step(self, dt_s: float, env: ToolEnvironment, energy_tol: float, force: bool = False) -> dict:
        """Advance the ladder (and the cutting-region field) by dt with the heat deposited; books the ledger.

        With the field, field and ladder are solved together every ``every_n_steps`` (enable_field) calls
        over the accumulated time (the heat deposited in between waits in the field's sources; the
        returned losses are those of the solve, zero on the calls in between); ``force`` solves now."""
        self.set_environment(env)
        if self.field is not None:
            self.field_pending_dt += float(dt_s)
            self.field_pending_steps += 1
            interval = self.field_interval_s
            if interval > 0.0:
                # at every multiple of the interval in run time: the same instants whatever the macro step
                due = self.time_s + dt_s >= self.field_next_s - 1e-9 * interval
            else:
                due = self.field_pending_steps >= self.field_every_n
            if not force and not due:
                self.time_s += dt_s
                return {"source_J": 0.0, "air_J": 0.0, "coolant_J": 0.0, "spindle_J": 0.0, "du_J": 0.0}
            if interval > 0.0:
                self.field_next_s = (math.floor((self.time_s + dt_s) / interval + 1e-9) + 1.0) * interval
            span = self.field_pending_dt
            self.field_pending_dt = 0.0
            self.field_pending_steps = 0
            sizes = self._place_element_heat()
            self.field.set_environment(env, self.emissivity_tool)
            out = self.field.step(self.ladder, self.mat_tool, self.mat_holder, span)
            tol = float(self.field.sys.scal[FS_TOL])
            if not np.all(np.isfinite(out[:FD_RES + 1])) or out[FD_RES] > max(1e3 * tol, 1e-8):
                raise ConservationError(f"thermal energy (tool.{self.slot} field solve, residual {out[FD_RES]:.3g} "
                                        f"after {out[FD_ITER]:.0f} iterations)", float(out[FD_SRC]),
                                        float(out[FD_U1] - out[FD_U0]), tol, self.time_s + dt_s)
            self._update_field_means(sizes)
            self.time_s += dt_s
            losses = out[FD_AIR] + out[FD_COOL] + out[FD_SP]
            self.ledger.book(out[FD_U0], out[FD_U1], out[FD_SRC], losses, 0.0, energy_tol, self.time_s)
            self.ladder.src[:] = 0.0
            self.to_spindle_J = float(out[FD_SP])
            self.to_coolant_J = float(out[FD_COOL])
            return {"source_J": float(out[FD_SRC]), "air_J": float(out[FD_AIR]), "coolant_J": float(out[FD_COOL]),
                    "spindle_J": float(out[FD_SP]), "du_J": float(out[FD_DU])}
        out = np.zeros(N_LD)
        K.ladder_step(self.ladder, self.mat_tool, self.mat_holder, float(dt_s), out)
        self.time_s += dt_s
        losses = out[LD_AIR] + out[LD_COOL] + out[LD_SP]
        self.ledger.book(out[LD_U0], out[LD_U1], out[LD_SRC], losses, 0.0, energy_tol, self.time_s)
        self.ladder.src[:] = 0.0
        self.to_spindle_J = float(out[LD_SP])
        self.to_coolant_J = float(out[LD_COOL])
        return {"source_J": float(out[LD_SRC]), "air_J": float(out[LD_AIR]), "coolant_J": float(out[LD_COOL]),
                "spindle_J": float(out[LD_SP]), "du_J": float(out[LD_DU])}

    # ------------------------------------------------------------------ cutting-region field
    def _field_pending(self, n: int) -> np.ndarray:
        """[5, nE] heat accumulated per field element since the last solve: rake J, flank J, sum of rake J x
        l_c, sum of flank J x l_f, and the last contact sizes (for elements without heat)."""
        if self.field_pend is None or self.field_pend.shape[1] != n:
            self.field_pend = np.zeros((5, n))
        return self.field_pend

    def _place_element_heat(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Map the accumulated element heat onto the field's faces; returns the contact sizes used."""
        el = self.field.elements
        if el is None or el.elem.size == 0:
            return None
        pend = self._field_pending(el.elem.size)
        floor = pend[4] if np.any(pend[4] > 0.0) else np.full(el.elem.size, float(self.field.section.r_edges[1]))
        lc = np.where(pend[0] > 0.0, pend[2] / np.where(pend[0] > 0.0, pend[0], 1.0), np.maximum(floor, 1e-9))
        lf = np.where(pend[1] > 0.0, pend[3] / np.where(pend[1] > 0.0, pend[1], 1.0), np.maximum(floor, 1e-9))
        if np.any(pend[0] != 0.0) or np.any(pend[1] != 0.0):
            self.field.deposit_strips(pend[0], pend[1], lc, lf, el.lo, el.hi, el.side)
        pend[:4] = 0.0
        return lc, lf

    def _update_field_means(self, sizes) -> None:
        """Cycle-mean rake, flank and edge temperatures of the field's elements after a solve."""
        el = self.field.elements
        if sizes is None or el is None:
            self.field_means = None
            return
        lc, lf = sizes
        self.field_means = np.stack(self.field.strip_means(lc, lf, el.lo, el.hi, el.side))

    def flush_field(self, env: ToolEnvironment, energy_tol: float) -> dict:
        """Solve the field now over the time accumulated since its last solve (no-op without one)."""
        if self.field is None or self.field_pending_steps == 0:
            return {"source_J": 0.0, "air_J": 0.0, "coolant_J": 0.0, "spindle_J": 0.0, "du_J": 0.0}
        self.field_pending_steps -= 1          # the forced call below counts itself again
        return self.step(0.0, env, energy_tol, force=True)

    def _settings(self):
        from .schema import ThermalSettings
        return ThermalSettings(**self.field_settings) if self.field_settings else ThermalSettings()

    def enable_field(self, settings=None, edges: EdgeModel | None = None, *, wedge_angle_rad: float | None = None,
                     edge_radius_m: float | None = None, contact_edges_m=(), every_n_steps: int = 1,
                     interval_s: float = 0.0) -> bool:
        """Solve the 3-D cutting-region field (docs/models/thermal.md 2.4) with the ladder from now on.

        With an EdgeModel the field follows its edge (wedge angle, edge radius, the elements along the
        unrolled edge); without one it is a straight edge of ``wedge_angle_rad`` from the corner (inserts)
        or the tip (solid tools), fed by :meth:`deposit_contact` (``contact_edges_m``: positions along the
        edge where contacts start or end, kept as cell boundaries); without either it is built by the
        first deposit_contact that gives the wedge angle. ``every_n_steps``: field and ladder are solved on
        every n-th call of :meth:`step` over the accumulated time; ``interval_s`` > 0 instead: at the first call
        ending at or after every multiple of it in run time (the domain's cadence, physics.thermal.
        tool_field_interval_ms: the same solve instants whatever the macro step). Returns whether the field exists
        now."""
        from .schema import ThermalSettings
        if self.field is not None:
            return True
        st = settings if settings is not None else ThermalSettings()
        self.field_requested = True
        self.field_settings = dict(st.__dict__)
        self.field_every_n = max(1, int(every_n_steps))
        self.field_interval_s = max(float(interval_s), 0.0)
        if self.field_interval_s > 0.0:
            self.field_next_s = (math.floor(self.time_s / self.field_interval_s + 1e-9) + 1.0) * self.field_interval_s
        if edges is None and wedge_angle_rad is None:
            return False
        self._build_field(st, edges, wedge_angle_rad, edge_radius_m, contact_edges_m)
        return self.field is not None

    def _build_field(self, st, edges: EdgeModel | None, wedge_angle_rad: float | None,
                     edge_radius_m: float | None, contact_edges_m=()) -> None:
        p = _p()
        lay = self.layout
        spec = self.spec
        turning = lay.turning
        lo_w, hi_w = float(p["min_wedge_angle_rad"]), float(p["max_wedge_angle_rad"])
        elements = None
        elem_z = None
        frames: dict = {}
        if edges is not None:
            length = spec.insert_ic_m if turning else st.tool_field_length_m
            elements = unroll_elements(edges, turning, length)
            if elements.elem.size == 0:
                return
            e = elements.elem
            wedge = 0.5 * math.pi - np.asarray(edges.elem_rake_rad, dtype=float)[0, e] \
                - np.asarray(edges.elem_clear_rad, dtype=float)[0, e]
            beta = float(np.median(np.clip(wedge, lo_w, hi_w)))
            radius = float(np.min(np.asarray(edges.elem_edge_radius_m, dtype=float)[0, e])) \
                if edge_radius_m is None else float(edge_radius_m)
            elem_z = np.asarray(edges.elem_z_m, dtype=float)
            mult = 1.0 if turning else float(np.shape(edges.elem_valid)[0])
            frames = edge_frames(edges, elements, turning)
            field_length = length if turning else min(length, float(elements.hi.max()))
        else:
            beta = float(min(max(float(wedge_angle_rad), lo_w), hi_w))
            radius = float(edge_radius_m) if edge_radius_m else 0.0
            mult = 1.0 if turning else float(max(spec.flutes, 1))
            field_length = spec.insert_ic_m if turning else st.tool_field_length_m
        ring1 = st.tool_field_ring_m if radius <= 0.0 else min(st.tool_field_ring_m, 0.25 * radius)
        if turning:
            ic = spec.insert_ic_m
            seat = {"node": int(lay.holder_nodes[0]), "conductance_W_m2K": float(p["shank_contact_conductance_W_m2K"]),
                    "contact_area_m2": float(p["insert_seat_area_ratio"]) * ic * ic,
                    "bar_height_m": float(spec.bar_h_m)}
            kind, back, thick, root, ref_d = "insert", ic, spec.insert_thickness_m, 0.0, ic
        else:
            d = spec.diameter_m
            if spec.core_diameter_m > 0.0:
                root = 0.5 * (d - spec.core_diameter_m)
            else:
                root = 0.5 * d * (1.0 - math.sqrt(min(max(spec.fluted_area_m2 / (math.pi * d * d / 4.0), 0.0), 1.0)))
            root = max(root, 8.0 * ring1)
            kind, back, thick, seat, ref_d = "tooth", 0.0, 0.0, None, d
        self.field = ToolField.build(kind=kind, beta=beta, ring1_m=ring1, growth=st.tool_field_growth,
                                     sector_rad=math.radians(st.tool_field_sector_deg), dy_max_m=st.tool_field_dy_m,
                                     length_m=field_length, section_back_m=back, thickness_m=thick, root_radius_m=root,
                                     mult=mult, mat_tool=self.mat_tool, mat_holder=self.mat_holder, ladder=self.ladder,
                                     layout=lay, elements=elements, elem_z_m=elem_z, seat=seat, ref_diameter_m=ref_d,
                                     tol=st.tool_field_tol, max_iter=st.tool_field_max_iter, frames=frames,
                                     contact_edges_m=contact_edges_m)
        u = self.energy_J()
        self.ledger.u_J = u
        if self.ledger.steps == 0:
            self.ledger.u0_J = u

    def energy_J(self) -> float:
        """Internal energy of the tool's solver (J rel. 20 degC): the ladder, or the field and the
        ladder nodes it does not replace."""
        if self.field is None:
            return float(K.ladder_energy(self.ladder))
        lad = self.ladder
        act = self.field.sys.active != 0
        return math.fsum((self.field.mesh.m * self.field.sys.h).tolist() + (lad.m[act] * lad.h[act]).tolist())

    def temperature_at(self, points_E, outside: str = "nan") -> np.ndarray:
        """Tool temperatures (K) at points (x, y, z) of the cutting region's edge frame E: x along the
        rake face from the edge into the tool, y along the edge from the field's start (inserts: the
        corner), z the rake face's outward normal (the tool at z < 0); outside the field's solid NaN, or
        with ``outside="nearest"`` the temperature at the nearest point of the solid."""
        if self.field is None:
            raise InputError("thermal: temperature_at needs the cutting-region field (ToolThermal.enable_field)")
        return self.field.temperature_at(points_E, outside)

    def contact_temperatures(self, rake_length_m: float, flank_length_m: float, width_m: float,
                             start_m: float = 0.0) -> tuple[float, float, float]:
        """(rake contact mean, flank contact mean, cutting-edge line) face temperatures (K) of the field over
        a contact patch [0, rake_length] x [start, start + width] on the rake face and [0, flank_length]
        down the flank (the cycle mean of the last solve)."""
        if self.field is None:
            raise InputError("thermal: contact_temperatures needs the cutting-region field (ToolThermal.enable_field)")
        rake, flank, edge = self.field.strip_means([rake_length_m], [flank_length_m], [start_m],
                                                   [start_m + width_m], [0])
        return float(rake[0]), float(flank[0]), float(edge[0])

    def temperature_at_T(self, points_T) -> np.ndarray:
        """Tool temperatures (K) at points of the tool frame T (rotating tools at spindle angle 0).

        Inside the field: its 3-D temperature, the point taken into the edge frame of the nearest field
        element (the representative tooth's field for every tooth); elsewhere the ladder's body
        temperature at the point's height z_T (turning tools: the insert's first node)."""
        pts = np.atleast_2d(np.asarray(points_T, dtype=float))
        body = self.body_temp_at(pts[:, 2]) if not self.layout.turning else \
            np.full(pts.shape[0], float(self.ladder.T[self.layout.tool_nodes[0]]))
        if self.field is None or "pos" not in self.field.frames:
            return np.asarray(body, dtype=float)
        pe = points_to_edge_frame(self.field.frames, pts)
        inside = self.field.temperature_at(pe)
        return np.where(np.isfinite(inside), inside, body)

    def advance_idle(self, until_s: float, env: ToolEnvironment, steps: int, energy_tol: float) -> dict:
        """Bring an idle tool up to ``until_s`` in ``steps`` implicit steps (no heat input)."""
        span = until_s - self.time_s
        totals = {"air_J": 0.0, "coolant_J": 0.0, "spindle_J": 0.0}
        if span <= 0.0:
            return totals
        for _ in range(max(1, steps)):
            result = self.step(span / max(1, steps), env, energy_tol, force=True)
            for key in totals:
                totals[key] += result[key]
        self.time_s = float(until_s)
        self.edge.n_cyc[:] = 0.0
        self.edge.t_eng[:] = 0.0
        return totals

    # ------------------------------------------------------------------ outputs
    def body_temp_at(self, z_m: np.ndarray) -> np.ndarray:
        nodes = self.layout.tool_nodes
        if self.layout.turning:
            return np.full(np.shape(z_m), self.ladder.T[nodes[0]])
        return np.interp(z_m, self.layout.z_m[nodes], self.ladder.T[nodes])

    def strain(self, nodes: np.ndarray) -> np.ndarray:
        """Free thermal strain of nodes relative to the initial temperature (mean-CTE tables)."""
        out = np.zeros(nodes.size)
        temp = self.ladder.T[nodes]
        for flag, mat in ((0, self.mat_tool), (1, self.mat_holder)):
            sel = self.ladder.mat[nodes] == flag
            if sel.any():
                out[sel] = strain_from(mat, temp[sel], self.temp_ref_K)
        return out

    def growths(self) -> tuple[float, float]:
        """(tool length growth, holder growth) in m (relative to the initial temperature)."""
        stick = self.layout.stick_nodes
        if self.layout.turning:
            stick = np.array([i for i in range(self.ladder.m.size) if self.layout.kind[i] in
                              (NODE_TOOL_EXPOSED, NODE_HOLDER)], dtype=np.int64)
            return float(np.sum(self.strain(stick) * self.ladder.dz[stick])), 0.0
        strand = self.layout.holder_strand
        tool = float(np.sum(self.strain(stick) * self.ladder.dz[stick]))
        holder = float(np.sum(self.strain(strand) * self.ladder.dz[strand])) if strand.size else 0.0
        return tool, holder

    def mean_temp(self, nodes: np.ndarray) -> float:
        if nodes.size == 0:
            return float("nan")
        m = self.ladder.m[nodes]
        return float(np.sum(m * self.ladder.T[nodes]) / np.sum(m))

    def edge_heat(self, edges: EdgeModel | None, heat, in_cut_s: np.ndarray | None, entries: np.ndarray | None,
                  vb_m: np.ndarray | None, dt_s: float, settings, omega_rad_s: float | None = None) -> ToolHeat:
        """ToolHeat of the step: edge temperatures at the quadrature points, growths, temperatures.

        Call after :meth:`step` of the same step (the tool's clock is then at the step's end). ``omega_rad_s``: the
        spindle speed of the step (|omega|); a rotating tool's element passes the cut once per revolution, so the
        period of its heating is 2 pi / |omega| (docs/models/thermal.md 2.2). Without it (turning tools, callers
        without a spindle speed) the period is the element's mean time between passes."""
        xi, weights = heating_phase_nodes(settings.edge_quadrature)
        tool_len, holder_len = self.growths()
        body_mean = self.mean_temp(self.layout.stick_nodes if self.layout.stick_nodes.size else self.layout.tool_nodes)
        holder_nodes = self.layout.holder_nodes
        result = ToolHeat(q_weights=weights.copy(), body_temp_K=body_mean, holder_temp_K=self.mean_temp(holder_nodes),
                          length_growth_m=tool_len, holder_growth_m=holder_len)
        if edges is None:
            self.last_heat = result
            return result
        z = np.asarray(edges.elem_z_m, dtype=float)
        shape = z.shape
        self.edge.ensure(shape)
        t_body = self.body_temp_at(z)
        inp = new_edge_input(shape, xi.size, EDGE_OUT_TEMPERATURE)
        inp.xi[:] = xi
        inp.scal[0] = settings.sa_direct_cycles
        inp.scal[1] = settings.window_points
        inp.T_body[:] = t_body
        tab = self.mat_tool
        k = np.interp(t_body, tab.T_K, tab.k_W_mK)
        cp = np.interp(t_body, tab.T_K, tab.cp_J_kgK)
        inp.k[:] = k
        inp.alpha[:] = k / (tab.rho_kg_m3[0] * cp)
        if heat is not None and in_cut_s is not None and np.shape(in_cut_s) == shape:
            self._fill_edge_input(inp, edges, heat, np.asarray(in_cut_s, dtype=float),
                                  np.asarray(entries, dtype=float) if entries is not None and np.shape(entries) == shape
                                  else np.zeros(shape), vb_m, dt_s, omega_rad_s, settings)
        else:
            self._fill_edge_input(inp, edges, None, np.zeros(shape), np.zeros(shape), vb_m, dt_s, omega_rad_s,
                                  settings)
        out = new_edge_output(shape, xi.size)
        field = self.field if (self.field is not None and self.field.elements is not None
                               and self.field.elements.in_field.shape == shape) else None
        interval = float(getattr(settings, "edge_interval_s", 0.0) or 0.0)
        if interval > 0.0:
            # at every multiple of the interval in run time: the same instants whatever the macro step
            due = self.time_s >= self.edge_next_s - 1e-9 * interval
        else:
            due = self.edge_calls % max(1, settings.edge_every_n_macro) == 0
        fresh = (self.rise is None or self.rise[0].shape != out.rake_q.shape
                 or (field is not None and (self.dev is None or self.dev[0].shape != out.rake_q.shape)) or due)
        self.edge_calls += 1
        if fresh and interval > 0.0:
            self.edge_next_s = (math.floor(self.time_s / interval + 1e-9) + 1.0) * interval
        cutting = (inp.mode != EDGE_NONE)[:, :, None]
        if fresh:
            glx, glw = gauss_legendre(settings.window_points)
            if field is None:
                K.edge_temperatures(inp, edge_response(), glx, glw, out)
            else:
                # elements the field carries: it holds their cycle mean, the edge kernel gives the periodic
                # deviation around it; the others: body temperature + local rise as without the field
                carried = field.elements.in_field
                modes = inp.mode.copy()
                inp.mode[:] = np.where(carried, EDGE_NONE, modes)
                K.edge_temperatures(inp, edge_response(), glx, glw, out)
                inp.mode[:] = np.where(carried, modes, EDGE_NONE)
                inp.scal[2] = EDGE_OUT_DEVIATION
                dev = new_edge_output(shape, xi.size)
                K.edge_temperatures(inp, edge_response(), glx, glw, dev)
                inp.mode[:] = modes
                out.amp[:] = np.where(carried, dev.amp, out.amp)
                self.dev = (dev.rake_q, dev.peak_q, dev.flank_q, dev.rake_entry, dev.flank_entry)
            base = t_body[:, :, None]
            self.rise = (out.rake_q - base, out.peak_q - base, out.flank_q - base, out.amp.copy(),
                         out.rake_entry - t_body, out.flank_entry - t_body)
        else:
            # between evaluations: the last local rise on the current body temperature, for the elements
            # still in cut (others: body temperature)
            base = t_body[:, :, None]
            out.rake_q[:] = base + np.where(cutting, self.rise[0], 0.0)
            out.peak_q[:] = base + np.where(cutting, self.rise[1], 0.0)
            out.flank_q[:] = base + np.where(cutting, self.rise[2], 0.0)
            out.amp[:] = np.where(cutting[:, :, 0], self.rise[3], 0.0)
            out.rake_entry[:] = t_body + np.where(cutting[:, :, 0], self.rise[4], 0.0)
            out.flank_entry[:] = t_body + np.where(cutting[:, :, 0], self.rise[5], 0.0)
        if field is not None:
            rake_m, flank_m, edge_m = self._field_means(field, edges, heat, vb_m, t_body)
            carried = field.elements.in_field[:, :, None]
            dev_r, dev_p, dev_f = (np.where(cutting, d, 0.0) for d in self.dev[:3])
            ent_r, ent_f = (np.where(cutting[:, :, 0], d, 0.0) for d in self.dev[3:])
            out.rake_q[:] = np.where(carried, rake_m[:, :, None] + dev_r, out.rake_q)
            out.flank_q[:] = np.where(carried, flank_m[:, :, None] + dev_f, out.flank_q)
            out.peak_q[:] = np.where(carried, edge_m[:, :, None] + dev_p, out.peak_q)
            out.rake_entry[:] = np.where(carried[:, :, 0], rake_m + ent_r, out.rake_entry)
            out.flank_entry[:] = np.where(carried[:, :, 0], flank_m + ent_f, out.flank_entry)
        valid = np.asarray(edges.elem_valid) != 0
        strain = strain_from(self.mat_tool, t_body, self.temp_ref_K)
        result.edge_temp_q_K = out.rake_q
        result.flank_temp_q_K = out.flank_q
        result.edge_peak_q_K = out.peak_q
        result.edge_cycle_amp_K = out.amp
        # the coefficient pipeline's tool temperature: the rake contact when the edge enters the cut (mechanics'
        # stationary-source partition adds the rise of the engagement's own heat inside ProcessCoefficients,
        # architecture 5.3); the in-cut mean is published as rake_contact_mean_K
        result.rake_temp_mean_K = out.rake_entry.copy()
        result.flank_entry_K = out.flank_entry.copy()
        result.rake_contact_mean_K = np.einsum("ijq,q->ij", out.rake_q, weights)
        result.body_temp_at_elem_K = t_body
        result.radial_growth_m = np.where(valid, np.asarray(edges.elem_r_m, dtype=float) * strain, 0.0) \
            if not self.layout.turning else np.zeros(shape)
        self.last_heat = result
        return result

    def _field_means(self, field: ToolField, edges: EdgeModel, heat, vb_m, t_body: np.ndarray):
        """[nt, ne] rake-contact mean, flank-contact mean and edge-line temperature of the field's
        elements (the field's cycle-mean face temperatures over their contact strips). A rotating
        tool's representative tooth carries the teeth's mean heat; tooth j's rise above the core is the
        representative's times its heat ratio (linear superposition per element)."""
        el = field.elements
        shape = np.shape(edges.elem_z_m)
        if self.edge.acc.shape == (N_ACC,) + tuple(shape):
            # the teeth's heat rates from their running sums: heat per unit in-cut time (the teeth at one position
            # along the edge pass the same arc once per revolution, they differ in their chips). A step shorter than a
            # revolution holds the passes of some teeth only, the running sums weight a tooth that just passed more than
            # one that passed earlier, and a pass across a step boundary has its entry in one step and part of its heat
            # in the next: the step's heat, the sums or the heat per entry would make the split follow the step.
            acc = self.edge.acc
            heat_sum = acc[ACC_RAKE_J] + acc[ACC_FLANK_J]
            t_in = acc[ACC_IN_CUT_S]
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(t_in > 0.0, heat_sum / np.where(t_in > 0.0, t_in, 1.0), 0.0)
        elif heat is not None and np.shape(heat.tool_rake_elem_J) == shape:
            q = np.asarray(heat.tool_rake_elem_J, dtype=float) + np.asarray(heat.tool_flank_elem_J, dtype=float)
        else:
            q = np.zeros(shape)
        if self.field_means is None or self.field_means.shape[1] != el.elem.size:
            l_c, l_f = self.contact_sizes(edges, heat, vb_m)
            self.field_means = np.stack(field.strip_means(l_c[0, el.elem], l_f[0, el.elem], el.lo, el.hi, el.side))
        rake, flank, edge = self.field_means
        out = [np.full(shape, np.nan) for _ in range(3)]
        qe = np.where(el.in_field[:, el.elem], q[:, el.elem], 0.0)
        n_on = np.maximum(el.in_field[:, el.elem].sum(axis=0), 1)
        mean_q = qe.sum(axis=0) / n_on
        ratio = np.where(mean_q[None, :] > 0.0, qe / np.where(mean_q > 0.0, mean_q, 1.0)[None, :], 1.0) \
            if shape[0] > 1 else np.ones_like(qe)
        base = t_body[:, el.elem]
        for target, rep in zip(out, (rake, flank, edge)):
            target[:, el.elem] = base + ratio * (rep[None, :] - base) if shape[0] > 1 else rep[None, :]
        return out[0], out[1], out[2]

    def _fill_edge_input(self, inp, edges, heat, in_cut, entries, vb_m, dt_s, omega_rad_s=None,
                         settings=None) -> None:
        """Periodic-heating inputs of the edge kernel per element, from the element's heat history in run time
        (docs/models/thermal.md 2.2), not from one macro step's counts.

        An element is engaged from its first pass until it misses one: no in-cut time for longer than one period
        after the step that held its last pass. While engaged it keeps running sums of its steps (tool heat, in-cut
        time, passes, heat-weighted contact sizes) that forget with the time constant edge_window_periods x period:
        the flux while in cut is sum(heat) / (contact x sum(in-cut time)), the heating time per pass sum(in-cut time)
        / sum(passes), the period 2 pi / |omega| for a rotating tool (one pass per revolution), else the mean time
        between passes. A macro step without a pass of the element (a step shorter than the revolution) leaves its
        engagement and these values as they are, so the temperatures it publishes do not alternate with the step.
        Interrupted: at least two passes since the engagement began; continuous: one entry, in cut since (its
        contact time advances with its in-cut time)."""
        p = _p()
        shape = in_cut.shape
        st = self.edge
        st.ensure(shape)
        valid = np.asarray(edges.elem_valid) != 0
        db = np.maximum(np.asarray(edges.elem_db_m, dtype=float), 1e-6)
        rake = np.asarray(edges.elem_rake_rad, dtype=float)
        clear = np.asarray(edges.elem_clear_rad, dtype=float)
        radius = np.asarray(edges.elem_r_m, dtype=float)
        if heat is not None:
            q_rake = heat.tool_rake_elem_J if heat.tool_rake_elem_J.shape == shape else np.zeros(shape)
            q_flank = heat.tool_flank_elem_J if heat.tool_flank_elem_J.shape == shape else np.zeros(shape)
        else:
            q_rake = np.zeros(shape)
            q_flank = np.zeros(shape)
        l_c, l_f = self.contact_sizes(edges, heat, vb_m)
        wedge = np.clip(0.5 * math.pi - rake - clear, p["min_wedge_angle_rad"], p["max_wedge_angle_rad"])
        now = self.time_s
        dt = max(float(dt_s), 0.0)
        omega = abs(float(omega_rad_s)) if omega_rad_s is not None else 0.0
        rotating = omega > 0.0 and not self.layout.turning
        acc = st.acc
        # the period of an element's passes: a revolution of a rotating tool, else the mean time between its passes
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_gap = np.where(acc[ACC_ENTRIES] > 0.0, acc[ACC_TIME_S] / np.where(acc[ACC_ENTRIES] > 0.0,
                                                                                  acc[ACC_ENTRIES], 1.0), dt)
        period_prev = np.full(shape, 2.0 * math.pi / omega) if rotating else np.maximum(mean_gap, dt)
        cut_now = valid & (in_cut > 0.0) & ((q_rake > 0.0) | (q_flank > 0.0))
        # still engaged: no pass missed since the step that held the last one
        kept = valid & np.isfinite(st.last_in_s) & ((now - st.last_in_s) <= period_prev + dt + 1e-12)
        engaged = cut_now | kept
        fresh = cut_now & ~kept
        window_periods = float(getattr(settings, "edge_window_periods", 4.0) or 4.0) if settings is not None else 4.0
        tau = window_periods * period_prev
        decay = np.where(tau > 0.0, np.exp(-dt / np.where(tau > 0.0, tau, 1.0)), 0.0)
        decay = np.where(fresh | ~engaged, 0.0, decay)          # a new engagement starts its sums afresh
        acc *= decay[None, :, :]
        acc[ACC_RAKE_J] += np.where(engaged, q_rake, 0.0)
        acc[ACC_FLANK_J] += np.where(engaged, q_flank, 0.0)
        acc[ACC_IN_CUT_S] += np.where(engaged, in_cut, 0.0)
        acc[ACC_ENTRIES] += np.where(engaged, entries, 0.0)
        acc[ACC_RAKE_LC] += np.where(engaged, q_rake * l_c, 0.0)
        acc[ACC_FLANK_LF] += np.where(engaged, q_flank * l_f, 0.0)
        acc[ACC_TIME_S] += np.where(engaged, dt, 0.0)
        st.last_in_s[:] = np.where(cut_now, now, np.where(engaged, st.last_in_s, -np.inf))
        st.n_cyc[:] = np.where(engaged, np.where(fresh, 0.0, st.n_cyc) + entries, 0.0)
        st.t_eng[:] = np.where(engaged, np.where(fresh, 0.0, st.t_eng) + in_cut, 0.0)
        heat_r = acc[ACC_RAKE_J]
        heat_f = acc[ACC_FLANK_J]
        t_in = acc[ACC_IN_CUT_S]
        active = engaged & (t_in > 0.0) & ((heat_r > 0.0) | (heat_f > 0.0))
        safe_t = np.where(active, t_in, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lc_w = np.where(heat_r > 0.0, acc[ACC_RAKE_LC] / np.where(heat_r > 0.0, heat_r, 1.0), l_c)
            lf_w = np.where(heat_f > 0.0, acc[ACC_FLANK_LF] / np.where(heat_f > 0.0, heat_f, 1.0), l_f)
            mean_gap = np.where(acc[ACC_ENTRIES] > 0.0, acc[ACC_TIME_S] / np.where(acc[ACC_ENTRIES] > 0.0,
                                                                                  acc[ACC_ENTRIES], 1.0), dt)
            t_pass = np.where(acc[ACC_ENTRIES] > 0.0, t_in / np.where(acc[ACC_ENTRIES] > 0.0, acc[ACC_ENTRIES], 1.0),
                              t_in)
            if rotating and shape[0] > 1:
                # the teeth at one position along the edge pass the same arc: their pass time from all their passes
                # (a pass across a step boundary has its entry in one step and part of its in-cut time in the next;
                # the teeth's boundary passes fall in different steps)
                cols_in = np.where(engaged, t_in, 0.0).sum(axis=0, keepdims=True)
                cols_n = np.where(engaged, acc[ACC_ENTRIES], 0.0).sum(axis=0, keepdims=True)
                t_cols = np.where(cols_n > 0.0, cols_in / np.where(cols_n > 0.0, cols_n, 1.0), 0.0)
                t_pass = np.where((acc[ACC_ENTRIES] > 0.0) & (cols_n > 0.0), np.broadcast_to(t_cols, shape), t_pass)
        period = np.full(shape, 2.0 * math.pi / omega) if rotating else np.maximum(mean_gap, 1e-12)
        lc_w = np.maximum(lc_w, 1e-9)
        lf_w = np.maximum(lf_w, 1e-9)
        interrupted = active & (st.n_cyc >= 2.0) & (t_pass < period * (1.0 - 1e-6))
        continuous = active & ~interrupted
        inp.mode[:] = np.where(interrupted, EDGE_INTERRUPTED, np.where(continuous, EDGE_CONTINUOUS, EDGE_NONE))
        inp.a_r[:] = lc_w
        inp.a_f[:] = lf_w
        inp.b[:] = 0.5 * db
        inp.q_r[:] = np.where(active, heat_r / (lc_w * db * safe_t), 0.0)
        inp.q_f[:] = np.where(active, heat_f / (lf_w * db * safe_t), 0.0)
        inp.wedge[:] = 0.5 * math.pi / wedge
        inp.r_far[:] = np.maximum(p["far_field_radius_ratio"] * radius, 2.0 * np.maximum(np.maximum(lc_w, lf_w), db))
        inp.period[:] = np.where(interrupted, period, 0.0)
        inp.t_h[:] = np.where(interrupted, np.minimum(t_pass, period), 0.0)
        inp.n_cyc[:] = np.where(interrupted, st.n_cyc, 0.0)
        inp.t_eng[:] = np.where(continuous, np.maximum(st.t_eng, np.where(active, t_pass, 0.0)), 0.0)

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        lad = self.ladder
        arrays = {f"{prefix}{name}": np.array(getattr(lad, name)) for name in lad._fields}
        for name in ("z_m", "kind", "diameter_m", "perimeter_m", "fluted", "channel", "tool_nodes", "holder_nodes",
                     "stick_nodes", "holder_strand"):
            arrays[f"{prefix}layout.{name}"] = np.array(getattr(self.layout, name))
        for name in self.mat_tool._fields:
            arrays[f"{prefix}mat_tool.{name}"] = np.array(getattr(self.mat_tool, name))
            arrays[f"{prefix}mat_holder.{name}"] = np.array(getattr(self.mat_holder, name))
        arrays[f"{prefix}edge.n_cyc"] = self.edge.n_cyc.copy()
        arrays[f"{prefix}edge.t_eng"] = self.edge.t_eng.copy()
        arrays[f"{prefix}edge.last_in_s"] = self.edge.last_in_s.copy()
        arrays[f"{prefix}edge.acc"] = self.edge.acc.copy()
        if self.rise is not None:
            for i, name in enumerate(RISE_NAMES):
                arrays[f"{prefix}rise.{name}"] = self.rise[i].copy()
        if self.dev is not None:
            for i, name in enumerate(DEV_NAMES):
                arrays[f"{prefix}dev.{name}"] = self.dev[i].copy()
        if self.field_pend is not None:
            arrays[f"{prefix}field_pend"] = self.field_pend.copy()
        if self.field_means is not None:
            arrays[f"{prefix}field_means"] = self.field_means.copy()
        field_meta = None
        if self.field is not None:
            fa, field_meta = self.field.to_arrays(f"{prefix}field.")
            arrays.update(fa)
        meta = {"slot": self.slot, "spec": dict(self.spec.__dict__), "temp_ref_K": self.temp_ref_K,
                "edge_calls": self.edge_calls, "has_rise": self.rise is not None, "has_dev": self.dev is not None,
                "field": field_meta, "field_requested": self.field_requested,
                "field_settings": dict(self.field_settings), "field_pending_dt": self.field_pending_dt,
                "field_every_n": self.field_every_n, "field_interval_s": self.field_interval_s,
                "field_next_s": self.field_next_s, "edge_next_s": self.edge_next_s,
                "field_pending_steps": self.field_pending_steps,
                "time_s": self.time_s, "ledger": self.ledger.to_dict(),
                "end_node": self.layout.end_node, "end_conductance_W_K": self.layout.end_conductance_W_K,
                "turning": self.layout.turning, "emissivity_tool": self.emissivity_tool,
                "emissivity_holder": self.emissivity_holder}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "ToolThermal":
        obj = cls.__new__(cls)
        obj.slot = int(meta["slot"])
        obj.spec = ToolSpec(**meta["spec"])
        obj.ladder = Ladder(*(np.ascontiguousarray(arrays[f"{prefix}{name}"]).copy() for name in Ladder._fields))
        obj.layout = LadderLayout(
            *(np.ascontiguousarray(arrays[f"{prefix}layout.{name}"]).copy() for name in
              ("z_m", "kind", "diameter_m", "perimeter_m", "fluted", "channel", "tool_nodes", "holder_nodes",
               "stick_nodes", "holder_strand")),
            end_node=int(meta["end_node"]), end_conductance_W_K=float(meta["end_conductance_W_K"]),
            turning=bool(meta["turning"]))
        obj.mat_tool = MatTable(*(np.ascontiguousarray(arrays[f"{prefix}mat_tool.{name}"]).copy()
                                  for name in MatTable._fields))
        obj.mat_holder = MatTable(*(np.ascontiguousarray(arrays[f"{prefix}mat_holder.{name}"]).copy()
                                    for name in MatTable._fields))
        obj.edge = EdgeState(arrays[f"{prefix}edge.n_cyc"].copy(), arrays[f"{prefix}edge.t_eng"].copy())
        if f"{prefix}edge.last_in_s" in arrays:
            obj.edge.last_in_s = np.array(arrays[f"{prefix}edge.last_in_s"])
        if f"{prefix}edge.acc" in arrays:
            obj.edge.acc = np.array(arrays[f"{prefix}edge.acc"])
        obj.edge.ensure(obj.edge.n_cyc.shape)
        obj.temp_ref_K = float(meta["temp_ref_K"])
        obj.time_s = float(meta["time_s"])
        obj.ledger = SolverLedger.from_dict(meta["ledger"])
        obj.emissivity_tool = float(meta["emissivity_tool"])
        obj.emissivity_holder = float(meta["emissivity_holder"])
        obj.last_heat = ToolHeat(body_temp_K=obj.mean_temp(obj.layout.tool_nodes),
                                 holder_temp_K=obj.mean_temp(obj.layout.holder_nodes))
        obj.to_spindle_J = 0.0
        obj.to_coolant_J = 0.0
        obj.edge_calls = int(meta["edge_calls"])
        obj.edge_next_s = float(meta.get("edge_next_s", 0.0))
        obj.rise = tuple(np.array(arrays[f"{prefix}rise.{name}"]) for name in RISE_NAMES) \
            if meta["has_rise"] and all(f"{prefix}rise.{name}" in arrays for name in RISE_NAMES) else None
        obj.dev = tuple(np.array(arrays[f"{prefix}dev.{name}"]) for name in DEV_NAMES) \
            if meta.get("has_dev") and all(f"{prefix}dev.{name}" in arrays for name in DEV_NAMES) else None
        obj.field = ToolField.from_arrays(arrays, meta["field"], f"{prefix}field.") if meta.get("field") else None
        obj.field_requested = bool(meta.get("field_requested", False))
        obj.field_settings = dict(meta.get("field_settings") or {})
        obj.field_pending_dt = float(meta.get("field_pending_dt", 0.0))
        obj.field_pending_steps = int(meta.get("field_pending_steps", 0))
        obj.field_every_n = int(meta.get("field_every_n", 1))
        obj.field_interval_s = float(meta.get("field_interval_s", 0.0))
        obj.field_next_s = float(meta.get("field_next_s", 0.0))
        obj.field_pend = np.array(arrays[f"{prefix}field_pend"]) if f"{prefix}field_pend" in arrays else None
        obj.field_means = np.array(arrays[f"{prefix}field_means"]) if f"{prefix}field_means" in arrays else None
        return obj


def edge_frames(edges: EdgeModel, elements, turning: bool) -> dict:
    """Unit vectors in T (spindle angle 0) of the field's elements on every tooth, for mapping points
    of the tool frame into the field's edge frame E (docs/models/thermal.md 2.4):

    * pos: element midpoint; tan: edge tangent along the field axis (differences of the midpoints);
    * x: rake direction into the tool, cos g (-n) - sin g m; z: rake outward normal, cos g m - sin g n
      (n, m: the chip-thickness normal sin(kappa) e_r - cos(kappa) e_z and the cutting direction,
      made orthonormal in the plane normal to the edge; g the normal rake; docs/models/tooling.md 2);
    * y_mid, z_mid: the element's position along the field axis and its height z_T (tooth 0)."""
    h = float(edges.hand_sign[0])
    el = elements
    e = el.elem
    alpha = h * (np.asarray(edges.tooth_phase_rad, dtype=float)[:, None] + np.asarray(edges.elem_lag_rad,
                                                                                         dtype=float)[:, e])
    r = np.asarray(edges.elem_r_m, dtype=float)[:, e]
    zz = np.asarray(edges.elem_z_m, dtype=float)[:, e]
    if turning:
        pos = np.stack([r, np.zeros_like(r), zz], axis=-1)
        e_r = np.broadcast_to(np.array([1.0, 0.0, 0.0]), pos.shape)
        cut = np.broadcast_to(np.array([0.0, h, 0.0]), pos.shape)
    else:
        pos = np.stack([r * np.cos(alpha), r * np.sin(alpha), zz], axis=-1)
        e_r = np.stack([np.cos(alpha), np.sin(alpha), np.zeros_like(alpha)], axis=-1)
        cut = h * np.stack([-np.sin(alpha), np.cos(alpha), np.zeros_like(alpha)], axis=-1)
    kappa = np.asarray(edges.elem_kappa_rad, dtype=float)[:, e]
    normal = np.sin(kappa)[..., None] * e_r - np.cos(kappa)[..., None] * np.array([0.0, 0.0, 1.0])
    tan = np.zeros_like(pos)
    for s in (0, 1):
        idx = np.flatnonzero(el.side == s)
        if idx.size == 0:
            continue
        p = pos[:, idx]
        if idx.size == 1:
            t = np.cos(kappa[:, idx])[..., None] * e_r[:, idx] + np.sin(kappa[:, idx])[..., None] * \
                np.array([0.0, 0.0, 1.0])
        else:
            t = np.gradient(p, axis=1)
        tan[:, idx] = t
    tan /= np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-300)
    m = cut - np.sum(cut * tan, axis=-1, keepdims=True) * tan
    m /= np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-300)
    n = normal - np.sum(normal * tan, axis=-1, keepdims=True) * tan - np.sum(normal * m, axis=-1, keepdims=True) * m
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-300)
    g = np.asarray(edges.elem_rake_rad, dtype=float)[:, e][..., None]
    x_dir = np.cos(g) * (-n) - np.sin(g) * m
    z_dir = np.cos(g) * m - np.sin(g) * n
    return {"pos": np.ascontiguousarray(pos), "tan": np.ascontiguousarray(tan), "x": np.ascontiguousarray(x_dir),
            "z": np.ascontiguousarray(z_dir), "side": np.asarray(el.side, dtype=np.int64).copy(),
            "y_mid": 0.5 * (el.lo + el.hi), "z_mid": np.asarray(edges.elem_z_m, dtype=float)[0, e].copy(),
            "valid": (np.asarray(edges.elem_valid)[:, e] != 0).astype(np.int8)}


def points_to_edge_frame(frames: dict, points_T: np.ndarray) -> np.ndarray:
    """Points of T (spindle angle 0) in the field's edge frame E through the frame of the nearest
    field element (any tooth); minor-edge elements (side 1) lie along x on the prism's end face."""
    pts = np.atleast_2d(np.asarray(points_T, dtype=float))
    pos = frames["pos"]
    valid = frames["valid"] != 0
    flat = pos.reshape(-1, 3)
    ok = valid.reshape(-1)
    d2 = np.sum((pts[:, None, :] - flat[None, :, :]) ** 2, axis=-1)
    d2[:, ~ok] = np.inf
    best = np.argmin(d2, axis=1)
    nE = pos.shape[1]
    j, i = best // nE, best % nE
    rel = pts - pos[j, i]
    xd = np.sum(rel * frames["x"][j, i], axis=-1)
    zd = np.sum(rel * frames["z"][j, i], axis=-1)
    td = np.sum(rel * frames["tan"][j, i], axis=-1)
    side = frames["side"][i]
    along = frames["y_mid"][i] + td
    return np.stack([np.where(side == 1, along, xd), np.where(side == 1, xd, along), zd], axis=-1)
