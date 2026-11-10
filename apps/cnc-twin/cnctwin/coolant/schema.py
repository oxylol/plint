"""``physics.coolant`` project settings (numerical settings of the coolant models).

The declaration :data:`FIELDS` is the single source: :func:`settings` reads a project's
``physics.coolant`` (an SI view, where ``step_max_mm`` reads as ``step_max_m``, or the raw mapping)
into :class:`CoolantSettings` and rejects out-of-range values; :func:`schema_node` builds the
:mod:`cnctwin.db.schema` node for the project schema (imported lazily).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

# name -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple[str, str, float, float, float, str]] = {
    "every_n_macro": ("every_n_macro", "count", 5, 1, 100000,
                      "occlusion job every n macro steps (supply, heat transfer, tank: every step)"),
    "latency_macro": ("latency_macro", "count", 1, 0, 1000,
                      "an occlusion job started at macro n is applied at n + k (architecture 5.1: k = 1)"),
    "rays_per_jet": ("rays_per_jet", "count", 64, 1, 4096, "flow-weighted rays across each nozzle jet"),
    "rays_through_tool": ("rays_through_tool", "count", 8, 1, 1024,
                          "rays per angular sample of each through-tool hole jet"),
    "through_tool_angles": ("through_tool_angles", "count", 4, 1, 256,
                            "spindle angles sampled per revolution for rotating through-tool jets"),
    "step_max_m": ("step_max_mm", "length", 10.0e-3, 1.0e-4, 0.2, "longest polyline segment of a jet"),
    "step_min_m": ("step_min_mm", "length", 0.5e-3, 1.0e-5, 0.05, "shortest polyline segment of a jet"),
    "sagitta_m": ("sagitta_mm", "length", 5.0e-5, 1.0e-7, 5.0e-3,
                  "largest distance between a polyline chord and the trajectory"),
    "trace_length_m": ("trace_length_mm", "length", 1.5, 0.01, 20.0, "longest occluded trace of a ray"),
    "aim_length_m": ("aim_length_mm", "length", 1.0, 0.01, 20.0,
                     "longest unobstructed continuation searched for cutting zones"),
    "max_points": ("max_points", "count", 128, 4, 100000, "polyline points stored per ray (mesh pass, UI)"),
    "profile_dz_m": ("profile_dz_mm", "length", 2.5e-4, 1.0e-6, 0.01, "radius-table spacing of tool envelopes"),
    "profile_nodes": ("profile_nodes", "count", 2048, 8, 1000000, "most radius-table nodes per tool"),
    "brick_voxel_m": ("brick_voxel_mm", "length", 2.0e-3, 1.0e-5, 0.1,
                      "voxel of the default stock bricks (when no workpiece field is wired)"),
    "brick_sub": ("brick_sub", "count", 4, 1, 64, "sub-cells per brick edge of the default stock bricks"),
    "wetted_cell_m": ("wetted_cell_mm", "length", 2.0e-3, 1.0e-4, 0.1,
                      "cell of the wetted / flush grid when chips publish no pile grid"),
    "supply_temp_step_dK": ("supply_temp_step_dK", "number", 0.5, 1.0e-3, 50.0,
                            "fluid temperature granularity of the supply operating point (re-solved per bucket)"),
    "solver_rel_tol": ("solver_rel_tol", "number", 1.0e-12, 1.0e-15, 1.0e-3,
                       "relative tolerance of the network, Colebrook, boiling and radiation solvers"),
    "solver_max_iter": ("solver_max_iter", "count", 200, 10, 100000, "iteration limit of those solvers"),
    "reach_scale": ("reach_scale", "number", 1.0, 0.0, 2.0,
                    "multiplier on every reach fraction (uncertainty studies: coolant_reach_rel)"),
    "idle_step_s": ("idle_step_s", "number", 60.0, 0.01, 3600.0,
                    "tank integration step for the time between operations"),
    "aim_hand_nozzles": ("aim_hand_nozzles", "count", 1, 0, 1,
                         "1: external nozzles on a tool-chain link that are not programmable (hand-set flex nozzles) "
                         "are aimed by the operator at the cutting zone of the chain's tool, re-aimed with every tool; "
                         "0: the machine file's fixed aim"),
}


@dataclass(frozen=True)
class CoolantSettings:
    every_n_macro: int = 5
    latency_macro: int = 1
    rays_per_jet: int = 64
    rays_through_tool: int = 8
    through_tool_angles: int = 4
    step_max_m: float = 10.0e-3
    step_min_m: float = 0.5e-3
    sagitta_m: float = 5.0e-5
    trace_length_m: float = 1.5
    aim_length_m: float = 1.0
    max_points: int = 128
    profile_dz_m: float = 2.5e-4
    profile_nodes: int = 2048
    brick_voxel_m: float = 2.0e-3
    brick_sub: int = 4
    wetted_cell_m: float = 2.0e-3
    supply_temp_step_dK: float = 0.5
    solver_rel_tol: float = 1.0e-12
    solver_max_iter: int = 200
    reach_scale: float = 1.0
    idle_step_s: float = 60.0
    aim_hand_nozzles: int = 1


def _lookup(section, name: str, file_key: str, kind: str):
    if section is None:
        return None
    for key in (name, file_key):
        value = section.get(key) if isinstance(section, Mapping) else getattr(section, key, None)
        if value is None:
            continue
        number = float(value)
        if key == file_key and file_key.endswith("_mm") and kind == "length":
            number *= 1e-3
        return number
    return None


def settings(section) -> CoolantSettings:
    """Settings from ``physics.coolant`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(CoolantSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        value = _lookup(section, item.name, file_key, kind)
        if value is None:
            value = default
        if not lo <= value <= hi:
            raise InputError(f"physics.coolant.{file_key} = {value} is outside [{lo}, {hi}] (SI)")
        values[item.name] = int(round(value)) if kind == "count" else float(value)
    if values["step_min_m"] > values["step_max_m"]:
        raise InputError("physics.coolant.step_min_mm must not exceed step_max_mm")
    return CoolantSettings(**values)


def schema_node():
    """The ``physics.coolant`` node of the project schema (cnctwin.db.schema)."""
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc)
    return Struct(nodes, doc="coolant model settings (docs/models/coolant.md)")
