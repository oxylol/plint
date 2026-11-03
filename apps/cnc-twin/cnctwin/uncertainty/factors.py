"""The varied factors of a study: what the project states as a tolerance or a batch spread.

A project states its tolerances in ``stock.tolerance_mm``, ``material.batch`` (1-sigma spreads of
the material) and ``uncertainty.vary``. Every stated tolerance and batch spread is varied;
``uncertainty.vary`` (and the study options' ``vary`` on top) adds the spreads only it can state
(tool offsets, runout, clamping, coolant reach, any other key), replaces a batch spread, or switches
a factor off (``stock_size: false``, a spread of 0 or false). Distributions (JCGM 100 4.3.7,
JCGM 101 6.4):

| factor | value | distribution | applied in the variant |
|---|---|---|---|
| ``stock_size[i]`` | stock size i (m) | uniform size +- tolerance | ``stock.size_mm[i]`` |
| ``length_offset[T<n>]``, ``radius_offset[T<n>]`` | offset error (m) | uniform +- tool_offsets | ``length_offset_mm`` / ``radius_offset_mm`` (else the wear offset) |
| ``runout[T<n>]`` | runout TIR (m) | uniform runout +- spread, >= 0 | ``runout_um`` |
| ``kc_scale`` | specific-force scale | truncated normal 1 +- sigma | inline material: flow stress, Kienzle, mechanistic, indentation |
| ``hardness_scale`` | hardness scale | truncated normal | inline material: ``physical.hardness`` |
| ``flow_stress_scale`` | flow-stress scale | truncated normal | inline material: stresses of ``flow_stress`` |
| ``conductivity_scale`` | conductivity scale | truncated normal | inline material: ``physical.conductivity_W_mK`` |
| ``clamping_force[<op>.<j>]`` | clamping scale | uniform 1 +- rel | ``input_torque_Nm`` or ``pressure_bar`` (else options) |
| ``coolant_reach_scale`` | coolant reach scale | uniform 1 +- rel | options only |
| any other ``vary`` key | ``*_sigma(_pct)``: truncated normal scale; ``*_rel``: uniform scale; else uniform +- value | | options only |

Truncation: +- ``normal_truncation_sigma`` and never below ``min_scale_factor`` (db/models/uncertainty.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..db import keyunits
from .sampling import TruncatedNormal, Uniform

KNOWN_VARY = ("stock_size", "tool_offsets_um", "runout_um", "kc_scale_sigma", "hardness_sigma_pct",
              "flow_stress_sigma_pct", "conductivity_sigma_pct", "clamping_force_rel", "coolant_reach_rel")
BATCH_KEYS = {"kc_scale_sigma": "kc_scale", "hardness_sigma_pct": "hardness_scale",
              "flow_stress_sigma_pct": "flow_stress_scale", "conductivity_sigma_pct": "conductivity_scale"}


@dataclass(frozen=True)
class Factor:
    name: str
    label: str
    unit: str                       # SI unit of the sampled value ("m", "-")
    nominal: float
    distribution: object            # Uniform | TruncatedNormal
    applies: str                    # project | material | options
    target: str                     # where the value goes
    stated_by: str                  # the project key that states the spread
    index: dict = field(default_factory=dict)   # locators used by the variant writer

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "unit": self.unit, "nominal": self.nominal,
                "distribution": self.distribution.describe(), "applies": self.applies, "target": self.target,
                "stated_by": self.stated_by, "index": dict(self.index)}


def _si(key: str, value) -> float:
    _, number = keyunits.to_si(key, value)
    return float(number)


def _number(value):
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _scale(sigma: float, params: dict) -> TruncatedNormal:
    width = float(params["normal_truncation_sigma"]) * sigma
    return TruncatedNormal(1.0, sigma, max(float(params["min_scale_factor"]), 1.0 - width), 1.0 + width)


def _relative(rel: float, params: dict) -> Uniform:
    return Uniform(max(float(params["min_scale_factor"]), 1.0 - rel), 1.0 + rel)


def spreads(data: dict, options: dict | None = None) -> tuple[dict, dict]:
    """(what is varied and by how much, as written; where each setting comes from).

    The stated tolerances are varied by default: the stock tolerance (``stock_size: true``) and the
    material batch spreads; ``uncertainty.vary`` (then the study options' ``vary``) adds settings,
    replaces a spread or switches one off (``false`` or 0).
    """
    settings: dict = {"stock_size": True}
    origin: dict = {"stock_size": "stock.tolerance_mm"}
    for key, value in ((data.get("material") or {}).get("batch") or {}).items():
        settings[key], origin[key] = value, "material.batch"
    section = data.get("uncertainty") or {}
    for label, vary in (("uncertainty.vary", (section.get("vary") if isinstance(section, dict) else None) or {}),
                        ("options.vary", (options or {}).get("vary") or {})):
        for key, value in vary.items():
            settings[key], origin[key] = value, label
    return settings, origin


def tool_entries(data: dict, project_dir) -> list[tuple[str, int, dict]]:
    """(source, index in the variant's tools.list, entry) of every tool: the tools.list entries, then the
    rows of a CSV tool library (which the variant writer appends to tools.list in this order)."""
    listed = (data.get("tools") or {}).get("list") or []
    out = [("list", i, entry) for i, entry in enumerate(listed) if isinstance(entry, dict)]
    library = (data.get("tools") or {}).get("library")
    if library and str(library).lower().endswith(".csv"):
        from pathlib import Path

        from ..project.toollib import read_csv
        path = Path(library) if Path(library).is_absolute() else Path(project_dir) / library
        entries, _, _ = read_csv(path)
        out += [("library", len(listed) + i, entry) for i, entry in enumerate(entries)]
    return out


def tool_label(entry: dict) -> str:
    path = entry.get("path")
    return f"T{entry.get('number')}" + (f"@{path}" if path not in (None, 1) else "")


def factors_for(data: dict, project_dir, params: dict, options: dict | None = None) -> tuple[list[Factor], list[dict]]:
    """(factors, stated spreads that cannot be varied with the reason)."""
    vary, origin = spreads(data, options)
    out: list[Factor] = []
    skipped: list[dict] = []
    if vary.get("stock_size"):
        _stock(data, out, skipped)
    if "tool_offsets_um" in vary and _number(vary["tool_offsets_um"]):
        spread = _si("tool_offsets_um", _number(vary["tool_offsets_um"]))
        for where, i, entry in tool_entries(data, project_dir):
            label = tool_label(entry)
            for kind in ("length", "radius"):
                key = f"{kind}_offset_mm"
                target = f"tools.list[{i}].{key}" if _number(entry.get(key)) is not None else \
                    f"tools.list[{i}].wear_offsets_mm.{kind}"
                out.append(Factor(f"{kind}_offset[{label}]", f"{label} {kind} offset error", "m", 0.0,
                                  Uniform(-spread, spread), "project", target,
                                  f"{origin['tool_offsets_um']}.tool_offsets_um = {_number(vary['tool_offsets_um'])}",
                                  {"where": where, "tool": i, "kind": kind}))
    if "runout_um" in vary and _number(vary["runout_um"]):
        spread = _si("runout_um", _number(vary["runout_um"]))
        for where, i, entry in tool_entries(data, project_dir):
            label = tool_label(entry)
            nominal = _si("runout_um", _number(entry.get("runout_um")) or 0.0)
            out.append(Factor(f"runout[{label}]", f"{label} runout (TIR)", "m", nominal,
                              Uniform(max(0.0, nominal - spread), nominal + spread), "project",
                              f"tools.list[{i}].runout_um", f"{origin['runout_um']}.runout_um = {_number(vary['runout_um'])}",
                              {"where": where, "tool": i}))
    for key, name in BATCH_KEYS.items():
        value = _number(vary.get(key)) if vary.get(key) is not False else None
        if not value:
            continue
        out.append(Factor(name, name.replace("_", " "), "-", 1.0, _scale(_si(key, value), params), "material",
                          _MATERIAL_TARGET[name], f"{origin.get(key, 'uncertainty.vary')}.{key} = {value}"))
    if "clamping_force_rel" in vary and _number(vary["clamping_force_rel"]):
        rel = _number(vary["clamping_force_rel"])
        for o, op in enumerate(data.get("operations") or []):
            for j, entry in enumerate(((op or {}).get("setup") or {}).get("workholding") or []):
                if not isinstance(entry, dict):
                    continue
                key = "input_torque_Nm" if _number(entry.get("input_torque_Nm")) else \
                    "pressure_bar" if _number(entry.get("pressure_bar")) else None
                name = f"clamping_force[{op.get('name', o)}.{j}]"
                target = f"operations[{o}].setup.workholding[{j}].{key}" if key else \
                    "options.uncertainty.factors (the project gives no input torque or pressure for this clamp)"
                out.append(Factor(name, f"clamping force scale of {op.get('name', o)} clamp {j + 1}", "-", 1.0,
                                  _relative(rel, params), "project" if key else "options", target,
                                  f"{origin['clamping_force_rel']}.clamping_force_rel = {rel}",
                                  {"op": o, "entry": j, "key": key}))
    if "coolant_reach_rel" in vary and _number(vary["coolant_reach_rel"]):
        rel = _number(vary["coolant_reach_rel"])
        out.append(Factor("coolant_reach_scale", "coolant reach scale", "-", 1.0, _relative(rel, params), "options",
                          "options.uncertainty.factors (no project key holds a coolant reach)",
                          f"{origin['coolant_reach_rel']}.coolant_reach_rel = {rel}"))
    for key in sorted(k for k in vary if k not in KNOWN_VARY):
        factor = _generic(key, vary[key], params, origin.get(key, "uncertainty.vary"))
        if factor is None:
            if vary[key] not in (False, 0, None):
                skipped.append({"key": f"{origin.get(key, 'uncertainty.vary')}.{key}",
                                "reason": f"{vary[key]!r} is not a number to vary by"})
        else:
            out.append(factor)
    return sorted(out, key=lambda f: f.name), skipped


_MATERIAL_TARGET = {
    "kc_scale": "material (inline, extends the database material): flow stress, Kienzle and mechanistic "
                "force coefficients, indentation coefficient",
    "hardness_scale": "material (inline): physical.hardness",
    "flow_stress_scale": "material (inline): the stresses of flow_stress",
    "conductivity_scale": "material (inline): physical.conductivity_W_mK"}


def _stock(data: dict, out: list, skipped: list) -> None:
    stock = data.get("stock") or {}
    if stock.get("kind") == "mesh":
        skipped.append({"key": "stock.tolerance_mm", "reason": "a mesh stock has no size to vary"})
        return
    sizes = stock.get("size_mm")
    tolerance = stock.get("tolerance_mm")
    if not isinstance(sizes, list) or tolerance is None:
        skipped.append({"key": "stock.tolerance_mm", "reason": "the stock states no tolerance"})
        return
    tolerances = tolerance if isinstance(tolerance, list) else [tolerance] * len(sizes)
    if len(tolerances) != len(sizes):
        skipped.append({"key": "stock.tolerance_mm", "reason": f"{len(tolerances)} tolerances for {len(sizes)} sizes"})
        return
    for i, (size, tol) in enumerate(zip(sizes, tolerances)):
        size_m, tol_m = _si("size_mm", _number(size)), _si("tolerance_mm", _number(tol) or 0.0)
        if tol_m <= 0.0:
            continue
        out.append(Factor(f"stock_size[{i}]", f"stock size {('x', 'y', 'z', 'w')[i] if i < 4 else i}", "m", size_m,
                          Uniform(size_m - tol_m, size_m + tol_m), "project", f"stock.size_mm[{i}]",
                          f"stock.tolerance_mm[{i}] = {_number(tol)}", {"i": i}))


def _generic(key: str, value, params: dict, origin: str) -> Factor | None:
    number = _number(value)
    if number is None or isinstance(value, bool):
        return None
    stem, unit = keyunits.parse_key(key)
    si_value = keyunits.convert(number, unit) if unit is not None else number
    stated = f"{origin}.{key} = {number}"
    target = "options.uncertainty.factors (not a key this build applies to the project file)"
    if stem.endswith("_sigma"):
        return Factor(stem.removesuffix("_sigma") + "_scale", stem.replace("_", " "), "-", 1.0,
                      _scale(float(si_value), params), "options", target, stated)
    if key.endswith("_rel"):
        return Factor(key.removesuffix("_rel") + "_scale", key.replace("_", " "), "-", 1.0,
                      _relative(float(si_value), params), "options", target, stated)
    return Factor(stem, stem.replace("_", " "), unit.si_token if unit else "-", 0.0,
                  Uniform(-abs(float(si_value)), abs(float(si_value))), "options", target, stated)


__all__ = ["Factor", "factors_for", "spreads", "tool_entries", "tool_label", "KNOWN_VARY", "BATCH_KEYS"]
