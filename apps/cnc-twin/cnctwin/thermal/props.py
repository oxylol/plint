"""Thermal properties from the databases as kernel tables (cnctwin.thermal.abi.MatTable).

A property may be a database :class:`~cnctwin.db.simap.Table` (points over kelvin), a
:class:`~cnctwin.db.simap.Sourced` or plain number, or the raw YAML forms ``{table: [[T_degC, v]]}``
and ``{value: v}``. All properties of one material are put on the union of their breakpoints,
which keeps every piecewise-linear interpolant exact; the enthalpy at the points is the exact
integral of the piecewise-linear cp from 20 degC. Database items that the resolved project does
not carry (holder, fixture and screw steels named in db/models/thermal-*.yaml, tool materials)
are read from ``db/<folder>/<id>.yaml`` with ``extends`` merged.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import numpy as np

from ..core.errors import InputError
from ..db.merge import deep_merge
from ..db.simap import SIMap, Sourced, Table
from ..db.yamlio import load_file
from ..paths import DB
from ..units import REFERENCE_TEMP_K, ZERO_DEGC_K
from . import kernels as K
from .abi import MatTable, new_mat_table


def table_points(prop, what: str = "property") -> tuple[np.ndarray, np.ndarray]:
    """(T_K, values) of a property in any of the accepted forms."""
    if prop is None:
        raise InputError(f"thermal: missing {what}")
    if isinstance(prop, Table):
        return np.asarray(prop.points_K, dtype=float), np.asarray(prop.values, dtype=float)
    if isinstance(prop, Sourced):
        if prop.value is None:
            raise InputError(f"thermal: {what} has no value")
        return np.array([REFERENCE_TEMP_K]), np.array([float(prop)])
    if isinstance(prop, Mapping) and "table" in prop:
        rows = prop["table"]
        return (np.array([float(r[0]) + ZERO_DEGC_K for r in rows]), np.array([float(r[1]) for r in rows]))
    if isinstance(prop, Mapping) and "value" in prop:
        return np.array([REFERENCE_TEMP_K]), np.array([float(prop["value"])])
    try:
        return np.array([REFERENCE_TEMP_K]), np.array([float(prop)])
    except (TypeError, ValueError):
        raise InputError(f"thermal: {what} is not a number or a temperature table: {prop!r}") from None


def enthalpy_points(T_K: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """Exact integral of the piecewise-linear cp (end values held) from 20 degC to each point."""
    n = T_K.size
    cumulative = np.zeros(n)
    for i in range(1, n):
        cumulative[i] = cumulative[i - 1] + 0.5 * (cp[i] + cp[i - 1]) * (T_K[i] - T_K[i - 1])
    ref = REFERENCE_TEMP_K
    if ref <= T_K[0]:
        at_ref = cp[0] * (ref - T_K[0])
    elif ref >= T_K[-1]:
        at_ref = cumulative[-1] + cp[-1] * (ref - T_K[-1])
    else:
        i = int(np.searchsorted(T_K, ref, side="right") - 1)
        d = ref - T_K[i]
        slope = (cp[i + 1] - cp[i]) / (T_K[i + 1] - T_K[i])
        at_ref = cumulative[i] + cp[i] * d + 0.5 * slope * d * d
    return cumulative - at_ref


def mat_table(rho, k, cp, alpha, name: str = "material") -> MatTable:
    """Kernel table of one material from its density, conductivity, cp and mean CTE."""
    parts = {"conductivity": table_points(k, f"{name} conductivity"),
             "specific heat": table_points(cp, f"{name} specific heat"),
             "expansion": table_points(alpha, f"{name} expansion")}
    grid = np.unique(np.concatenate([pts for pts, _ in parts.values()]))
    values = {}
    for key, (pts, vals) in parts.items():
        order = np.argsort(pts, kind="stable")
        values[key] = np.interp(grid, pts[order], vals[order])
    if np.any(values["conductivity"] <= 0.0) or np.any(values["specific heat"] <= 0.0):
        raise InputError(f"thermal: {name} has a non-positive conductivity or specific heat")
    rho_pts, rho_vals = table_points(rho, f"{name} density")
    rho_value = float(np.interp(REFERENCE_TEMP_K, rho_pts, rho_vals))       # material configuration: 20 degC
    if not rho_value > 0.0:
        raise InputError(f"thermal: {name} density must be positive")
    h_pts = enthalpy_points(grid, values["specific heat"])
    return new_mat_table(grid, values["conductivity"], values["specific heat"], h_pts, values["expansion"],
                         rho_value)


def _get(obj, name):
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def workpiece_table(material) -> MatTable:
    """Table of a workpiece material (``physical`` section of a db/materials item)."""
    physical = _get(material, "physical")
    name = str(_get(material, "id") or "workpiece")
    return mat_table(_get(physical, "density_kg_m3"), _get(physical, "conductivity_W_mK"),
                     _get(physical, "specific_heat_J_kgK"), _get(physical, "expansion_1_K"), name)


def solid_table(item) -> MatTable:
    """Table of a material whose properties are top-level keys (tool materials)."""
    name = str(_get(item, "id") or "tool material")
    return mat_table(_get(item, "density_kg_m3"), _get(item, "conductivity_W_mK"),
                     _get(item, "specific_heat_J_kgK"), _get(item, "expansion_1_K"), name)


@functools.lru_cache(maxsize=None)
def _raw_item(folder: str, item_id: str) -> dict:
    path = DB / folder / f"{item_id}.yaml"
    if not path.is_file():
        raise InputError(f"thermal: no database item {folder}/{item_id}", str(path))
    data = load_file(path).data
    if not isinstance(data, Mapping):
        raise InputError("thermal: database item is not a mapping", str(path))
    base_id = data.get("extends")
    if isinstance(base_id, str) and base_id:
        merged = deep_merge(_raw_item(folder, base_id), {k: v for k, v in data.items() if k != "extends"})
        return merged
    return dict(data)


THERMAL_KEYS = ("id", "family", "density_kg_m3", "conductivity_W_mK", "specific_heat_J_kgK", "expansion_1_K",
                "max_service_temp_degC", "solidus_degC", "liquidus_degC")


def db_raw(folder: str, item_id: str) -> dict:
    """``db/<folder>/<item_id>.yaml`` as plain data with ``extends`` merged (a copy)."""
    return dict(_raw_item(folder, str(item_id)))


def thermal_view(raw: Mapping, section: str | None = None) -> SIMap:
    """SI view of the thermal keys of a raw item (of one section of it when ``section`` is given),
    so unrelated parts of the file never affect the thermal model."""
    source = raw.get(section) if section else raw
    if not isinstance(source, Mapping):
        raise InputError(f"thermal: database item has no section {section!r}")
    picked = {key: source[key] for key in THERMAL_KEYS if key in source}
    if "src" in source:
        picked["src"] = source["src"]
    return SIMap(picked)


def db_item(folder: str, item_id: str, section: str | None = None) -> SIMap:
    """SI view of the thermal keys of ``db/<folder>/<item_id>.yaml`` (or of one of its sections)."""
    return thermal_view(_raw_item(folder, str(item_id)), section)


def material_table_by_id(item_id: str) -> MatTable:
    """Table of a db/materials item (its ``physical`` section)."""
    view = db_item("materials", item_id, "physical")
    return mat_table(view.density_kg_m3, view.conductivity_W_mK, view.specific_heat_J_kgK, view.expansion_1_K,
                     item_id)


def tool_material_table_by_id(item_id: str) -> MatTable:
    """Table of a db/tool_materials item."""
    return solid_table(db_item("tool_materials", item_id))


def h_of_T(mat: MatTable, temp_K: float) -> float:
    return float(K.enthalpy(mat, float(temp_K)))


def T_of_h(mat: MatTable, h_J_kg: float) -> float:
    return float(K.temperature(mat, float(h_J_kg)))


def mean_expansion(mat: MatTable, temp_K: float) -> float:
    """Mean CTE from 20 degC at temp_K."""
    return float(np.interp(float(temp_K), mat.T_K, mat.alpha_1_K))


def thermal_strain(mat: MatTable, temp_K: float, ref_K: float) -> float:
    """Free thermal strain between ref_K and temp_K from the mean CTE (from 20 degC) table."""
    return (mean_expansion(mat, temp_K) * (temp_K - REFERENCE_TEMP_K)
            - mean_expansion(mat, ref_K) * (ref_K - REFERENCE_TEMP_K))
