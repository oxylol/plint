"""Report sections about the part and the process physics: inspection (CMM table, deviation
statistics, GD&T, threads, error budget), surface finish and burrs, chips, thermal state and the
part temperature at unload, coolant reach and shadows, chatter and stability lobes, tool deflection
and its share of each dimensional error, and the conservation residuals.

docs/report.md "Inputs" lists every key read; a characteristic or record whose kind the report does
not know is still shown, in a generic table with all its fields.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.bus import CHIP_FORMS
from ..units import ZERO_DEGC_K
from . import fmt, plots
from .context import Context
from .html import Section, figure, join, kv, note, para, pass_badge, table, ul
from .rundir import split_column

ONE_SIDED_KINDS = ("flatness", "straightness", "roundness", "circularity", "cylindricity", "profile",
                   "profile_of_a_line", "profile_of_a_surface", "perpendicularity", "parallelism", "angularity",
                   "position", "symmetry", "concentricity", "coaxiality", "runout", "circular_runout",
                   "total_runout")
SURFACE_KINDS = ("roughness", "finish", "surface_finish")


# ----------------------------------------------------------------------------- inspection

def limits(tolerance: dict) -> tuple[float | None, float | None, float | None]:
    """(nominal, lower, upper) of a characteristic in SI; one-sided zones have lower 0 and no nominal."""
    nominal = fmt.as_float(tolerance.get("nominal_m"))
    lower = fmt.as_float(tolerance.get("lower_m"))
    upper = fmt.as_float(tolerance.get("upper_m"))
    if nominal is not None and (lower is None or upper is None):
        plus = fmt.as_float(tolerance.get("plus_m"))
        minus = fmt.as_float(tolerance.get("minus_m"))
        if upper is None and plus is not None:
            upper = nominal + plus
        if lower is None and minus is not None:
            lower = nominal - abs(minus)
    zone = fmt.as_float(tolerance.get("tolerance_m"))
    if zone is not None and nominal is None and upper is None:
        return None, 0.0, zone
    return nominal, lower, upper


def utilisation(tolerance: dict) -> float | None:
    """Share of the tolerance used: |deviation| / allowed on that side (two-sided), measured / zone."""
    nominal, lower, upper = limits(tolerance)
    measured = fmt.as_float(tolerance.get("measured_m"))
    if measured is None:
        return None
    if nominal is None:
        return measured / upper if upper else None
    if measured >= nominal:
        return (measured - nominal) / (upper - nominal) if upper is not None and upper > nominal else None
    return (nominal - measured) / (nominal - lower) if lower is not None and nominal > lower else None


def _judged(tolerance: dict):
    given = tolerance.get("pass")
    if isinstance(given, bool):
        return given
    nominal, lower, upper = limits(tolerance)
    measured = fmt.as_float(tolerance.get("measured_m"))
    if measured is None or (lower is None and upper is None):
        return None
    return (lower is None or measured >= lower) and (upper is None or measured <= upper)


def inspection_section(ctx: Context) -> Section:
    inspection = ctx.run.inspection
    if not inspection:
        return Section("inspection", "Inspection", note("inspection.json is not in the run directory: no "
                                                        "measured characteristics."))
    tolerances = [t for t in inspection.get("tolerances", []) if isinstance(t, dict)]
    dimensional = [t for t in tolerances if str(t.get("kind", "")) not in SURFACE_KINDS + ("thread",)
                   and ("measured_m" in t)]
    threads = [t for t in tolerances if str(t.get("kind", "")) == "thread"]
    other = [t for t in tolerances if t not in dimensional and t not in threads
             and str(t.get("kind", "")) not in SURFACE_KINDS]
    temperature = fmt.as_float(inspection.get("temperature_K"))
    body = para(f"Measured at {fmt.degc(temperature)} °C on the unclamped part (ISO 1 reference "
                "temperature: 20 °C)." if temperature is not None else
                "inspection.json does not state the inspection temperature.")
    body += _cmm_table(ctx, dimensional)
    body += _point_statistics(dimensional) + _surface_deviation(ctx)
    body += _threads_table(threads)
    body += _generic_table(other, "Other characteristics (kinds the report does not interpret)")
    body += _error_budget(dimensional)
    summary = inspection.get("summary") or {}
    if summary:
        body += note(f"inspection.json summary: {summary.get('checked', fmt.DASH)} checked, "
                     f"{summary.get('failed', fmt.DASH)} failed.")
    return Section("inspection", "Inspection", body,
                   lead="Every inspected characteristic as a CMM report: nominal, limits, measured, deviation.")


def _cmm_table(ctx: Context, items: list[dict]) -> str:
    rows, classes = [], []
    for item in items:
        nominal, lower, upper = limits(item)
        measured = fmt.as_float(item.get("measured_m"))
        deviation = fmt.as_float(item.get("deviation_m"))
        if deviation is None and measured is not None and nominal is not None:
            deviation = measured - nominal
        passed = _judged(item)
        excess = None
        if measured is not None:
            if upper is not None and measured > upper:
                excess = measured - upper
            elif lower is not None and measured < lower:
                excess = lower - measured
            else:
                excess = 0.0
        used = utilisation(item)
        datums = " | ".join(str(d) for d in item.get("datums") or [])
        size = item.get("size") if isinstance(item.get("size"), dict) else {}
        fit = item.get("fit") or size.get("fit")
        lines = item.get("lines") or ([item["line"]] if item.get("line") else [])
        op = str(item.get("op") or "")
        where = ctx.line_link(op, item.get("program") or _program_of(ctx, op), int(lines[0])) if lines else fmt.DASH
        kind = str(item.get("kind", "")) + (f" {fit}" if fit else "") + (f" | {datums}" if datums else "")
        rows.append([item.get("id", ""), kind, item.get("feature", fmt.DASH), fmt.mm(nominal, 4),
                     f"{fmt.mm(lower, 4)} … {fmt.mm(upper, 4)}", fmt.mm(measured, 4), fmt.signed_um(deviation),
                     fmt.um(excess), fmt.pct(used, 0), pass_badge(passed), where])
        classes.append("fail" if passed is False else "")
    return "<h3>Characteristics</h3>" + table(
        ["characteristic", "kind | datums", "feature", "nominal (mm)", "limits (mm)", "measured (mm)",
         "deviation (µm)", "out of tol. (µm)", "tol. used (%)", "result", "made at"], rows,
        numeric={3, 4, 5, 6, 7, 8}, row_classes=classes,
        caption="Two-sided sizes: nominal ± limits; GD&T zones (flatness, position ...): lower limit 0, "
                "upper the zone. 'tol. used' is the deviation as a share of the tolerance on its side.",
        empty="inspection.json has no dimensional characteristics.")


def _program_of(ctx: Context, op: str) -> str:
    """The (first) program file of an operation, or of the first operation when ``op`` is not given."""
    for name, program, _ in ctx.run.programs():
        if not op or name == op:
            return program
    return ""


def _point_statistics(items: list[dict]) -> str:
    rows = []
    for item in items:
        points = [p for p in item.get("points") or [] if isinstance(p, dict)]
        deviations = np.array([fmt.as_float(p.get("deviation_m")) for p in points
                               if fmt.as_float(p.get("deviation_m")) is not None])
        if not len(deviations):
            continue
        rows.append([item.get("id", ""), len(deviations), fmt.signed_um(float(deviations.mean())),
                     fmt.um(float(deviations.std(ddof=1)) if len(deviations) > 1 else None),
                     fmt.signed_um(float(deviations.min())), fmt.signed_um(float(deviations.max())),
                     fmt.um(float(deviations.max() - deviations.min()))])
    if not rows:
        return ""
    return "<h4>Probing points</h4>" + table(
        ["characteristic", "points", "mean (µm)", "std (µm)", "min (µm)", "max (µm)",
         "range (µm)"], rows, numeric={1, 2, 3, 4, 5, 6},
        caption="Deviation of each probed point from the nominal surface (the range is the form error seen by "
                "the probe).")


def _surface_deviation(ctx: Context) -> str:
    attributes = ctx.run.attributes
    deviation = attributes.get("deviation_m")
    if deviation is None:
        return ""
    values = deviation[np.isfinite(deviation)].astype(np.float64)
    if not len(values):
        return ""
    stats = [("vertices", fmt.integer(len(values))), ("mean", f"{fmt.signed_um(values.mean())} µm"),
             ("standard deviation", f"{fmt.um(values.std())} µm"),
             ("RMS", f"{fmt.um(math.sqrt(float(np.mean(values ** 2))))} µm"),
             ("minimum", f"{fmt.signed_um(values.min())} µm"), ("maximum", f"{fmt.signed_um(values.max())} µm"),
             ("1 % / 99 % quantiles", f"{fmt.signed_um(np.quantile(values, 0.01))} / "
                                      f"{fmt.signed_um(np.quantile(values, 0.99))} µm")]
    png = plots.histogram(values * fmt.UM_PER_M, "µm", "Deviation of the machined surface from the nominal")
    return ("<h4>Deviation of the whole surface</h4>" + kv(stats)
            + figure(png, "surface deviation histogram",
                     "Per mesh vertex of the final part (keyframes/attr), unweighted by area."))


def _threads_table(items: list[dict]) -> str:
    if not items:
        return ""
    rows, classes = [], []
    for item in items:
        diameters = []
        for name in ("pitch_diameter_m", "major_diameter_m", "minor_diameter_m"):
            value = item.get(name)
            if isinstance(value, dict):
                diameters.append(f"{name.split('_')[0]}: {fmt.mm(value.get('measured'), 4)} "
                                 f"[{fmt.mm(value.get('min'), 4)}, {fmt.mm(value.get('max'), 4)}]")
            elif fmt.as_float(value) is not None:
                diameters.append(f"{name.split('_')[0]}: {fmt.mm(value, 4)}")
        gauges = ", ".join(f"{k}: {'accepts' if v else 'rejects'}" for k, v in sorted((item.get("gauges") or {}).items()))
        passed = _judged(item) if "pass" in item else None
        rows.append([item.get("id", ""), item.get("designation", fmt.DASH), item.get("class", fmt.DASH),
                     "; ".join(diameters) or fmt.DASH, gauges or fmt.DASH, pass_badge(passed)])
        classes.append("fail" if passed is False else "")
    return "<h3>Threads</h3>" + table(["thread", "designation", "class", "diameters measured [limits] (mm)",
                                       "gauges", "result"], rows, row_classes=classes,
                                      caption="Tolerance classes per ISO 965-1 (internal H/G, external h/g/f/e).")


def _generic_table(items: list[dict], title: str) -> str:
    if not items:
        return ""
    keys = sorted({k for item in items for k in item})
    rows = [[_plain(item.get(k)) for k in keys] for item in items]
    return f"<h4>{title}</h4>" + table(keys, rows)


def _plain(value) -> str:
    if isinstance(value, float):
        return fmt.num(value)
    if isinstance(value, (list, dict)):
        import json
        return json.dumps(value, sort_keys=True)
    return fmt.DASH if value is None else str(value)


def _contributions(item: dict) -> list[tuple[str, float]]:
    out = []
    for entry in item.get("contributions_m") or []:
        if not isinstance(entry, dict):
            continue
        value = fmt.as_float(entry.get("value_m"))
        if value is None:
            continue
        source = str(entry.get("source", "other"))
        if entry.get("tool") is not None:
            source += f" T{entry['tool']}"
        out.append((source, value))
    return out


def _error_budget(items: list[dict]) -> str:
    budgets = [(item, _contributions(item)) for item in items]
    budgets = [(item, parts) for item, parts in budgets if parts]
    if not budgets:
        return note("inspection.json gives no error budget (contributions_m) for the characteristics.")
    sources = sorted({source for _, parts in budgets for source, _ in parts})
    values = np.zeros((len(budgets), len(sources)))
    rows, totals = [], []
    for i, (item, parts) in enumerate(budgets):
        for source, value in parts:
            values[i, sources.index(source)] += value
        total = _explained_total(item)
        totals.append(total if total is not None else 0.0)
        listed = "; ".join(f"{s} {fmt.signed_um(v)}" for s, v in sorted(zip(sources, values[i]), key=lambda x: -abs(x[1]))
                           if v)
        rows.append([item.get("id", ""), listed, fmt.signed_um(float(values[i].sum())), fmt.signed_um(total)])
    png = plots.signed_stack([item.get("id", "") for item, _ in budgets], sources, values * fmt.UM_PER_M,
                             "µm", "Error budget: contributions to each deviation", np.array(totals) * fmt.UM_PER_M)
    return ("<h3>Error budget</h3>" + table(["characteristic", "contributions, largest first (µm)", "sum (µm)",
                                             "deviation or form error (µm)"], rows, numeric={2, 3})
            + figure(png, "error budget", "What each physical cause adds to each deviation (inspection.json "
                                          "contributions_m); the black tick is the total deviation (the measured "
                                          "form error for a tolerance zone)."))


def _explained_total(item: dict) -> float | None:
    """What the contributions explain: the deviation of a size, the measured value of a zone."""
    deviation = fmt.as_float(item.get("deviation_m"))
    if deviation is not None:
        return deviation
    nominal, _, _ = limits(item)
    return fmt.as_float(item.get("measured_m")) if nominal is None else None


# ----------------------------------------------------------------------------- surface and burrs

def surface_section(ctx: Context) -> Section:
    surface = ctx.run.surface
    body = ""
    if not surface:
        body += note("surface.json is not in the run directory: no finish or burr results.")
    else:
        rows, classes = [], []
        for feature in surface.get("features") or []:
            passed = _judged(feature) if "pass" in feature else None
            rows.append([feature.get("id", ""), feature.get("op", fmt.DASH), fmt.um(feature.get("Ra_m"), 2),
                         fmt.um(feature.get("Rz_m"), 2), fmt.um(feature.get("spec_Ra_m"), 2),
                         fmt.um(feature.get("spec_Rz_m"), 2),
                         f"T{feature['tool']}" if feature.get("tool") is not None else fmt.DASH, pass_badge(passed)])
            classes.append("fail" if passed is False else "")
        body += "<h3>Surface finish</h3>" + table(["feature", "op", "Ra (µm)", "Rz (µm)", "spec Ra (µm)",
                                                   "spec Rz (µm)", "tool", "result"], rows,
                                                  numeric={2, 3, 4, 5}, row_classes=classes,
                                                  caption="Ra and Rz per ISO 4287.",
                                                  empty="surface.json lists no finished features.")
        burrs = [b for b in surface.get("burrs") or [] if isinstance(b, dict)]
        burr_rows = [[b.get("edge", fmt.DASH), b.get("kind", fmt.DASH), fmt.um(b.get("height_m"), 1),
                      fmt.um(b.get("root_thickness_m"), 1), b.get("op", fmt.DASH),
                      ctx.line_link(b.get("op", ""), b.get("program") or _program_of(ctx, str(b.get("op", ""))),
                                    int(b.get("line") or 0)) if b.get("line") else fmt.DASH,
                      " ".join(fmt.mm(v, 2) for v in b.get("location_S_m") or []) or fmt.DASH] for b in burrs]
        body += "<h3>Burrs</h3>" + table(["edge", "kind", "height (µm)", "root (µm)", "op", "line",
                                          "where in S (mm)"], burr_rows, numeric={2, 3},
                                         empty="surface.json lists no burrs.")
    roughness = ctx.run.attributes.get("Ra_m")
    if roughness is not None and np.isfinite(roughness).any():
        values = roughness[np.isfinite(roughness)].astype(np.float64)
        body += kv([("Ra over the surface: median", f"{fmt.um(np.median(values), 2)} µm"),
                    ("Ra: 95 % quantile", f"{fmt.um(np.quantile(values, 0.95), 2)} µm"),
                    ("Ra: maximum", f"{fmt.um(values.max(), 2)} µm")])
    flagged = [f for f in ctx.run.findings if f.get("category") in ("finish_out_of_spec", "burr")]
    if flagged:
        body += ul([join([ctx.finding_link(f["id"]), " ", f.get("message", "")]) for f in flagged])
    return Section("surface", "Surface finish and burrs", body)


# ----------------------------------------------------------------------------- chips

def _last_with(ctx: Context, reader, key: str):
    for entry in reversed(ctx.run.keyframes):
        arrays = reader(entry)
        if arrays and key in arrays:
            return entry, arrays
    return None, None


def _grid_extent(arrays: dict, grid: np.ndarray, prefix: str) -> tuple[float, float, float, float] | None:
    origin = arrays.get(f"{prefix}_origin_S_m")
    cell = arrays.get(f"{prefix}_cell_m")
    if origin is None or cell is None:
        return None
    x0, y0 = float(np.asarray(origin)[0]), float(np.asarray(origin)[1])
    step = float(np.asarray(cell))
    ny, nx = grid.shape[:2]
    return tuple(v * fmt.MM_PER_M for v in (x0, x0 + nx * step, y0, y0 + ny * step))


def _extent(arrays: dict, grid: np.ndarray, *prefixes: str):
    for prefix in prefixes:
        found = _grid_extent(arrays, grid, prefix)
        if found is not None:
            return found, ""
    ny, nx = grid.shape[:2]
    return (0.0, float(nx), 0.0, float(ny)), "grid origin not recorded: axes in cells"


def chips_section(ctx: Context) -> Section:
    run = ctx.run
    production = run.production or {}
    rows = [[int(p.get("part", 0)) + 1, fmt.grams(p.get("removed_mass_kg"))] for p in production.get("per_part") or []]
    body = table(["part", "chip mass (g)"], rows, numeric={0, 1},
                 caption="Chip mass per part = removed material (mass balance, architecture 12).",
                 empty="production.json gives no chip mass per part.")
    if run.summary.get("removed_mass_kg") is not None:
        body += para(f"Removed in this run: {fmt.grams(run.summary.get('removed_mass_kg'))} g "
                     f"({fmt.num((run.summary.get('removed_volume_m3') or 0) * fmt.MM3_PER_M3, 5)} mm³).")
    entry, chips = _last_with(ctx, run.keyframe_chips, "mass_kg")
    if chips is not None:
        pairs = [("keyframe", f"{entry['k']} at {fmt.seconds(entry.get('time_s'))} s"),
                 ("chips in flight", fmt.integer(len(chips.get("mass_kg", []))))]
        for key in sorted(k for k in chips if k.endswith("_kg") and np.asarray(chips[k]).ndim == 0):
            pairs.append((key.replace("_kg", "").replace("_", " "), f"{fmt.grams(float(chips[key]))} g"))
        if "pile_height_max_m" in chips:
            pairs.append(("highest pile", f"{fmt.mm(float(chips['pile_height_max_m']), 2)} mm"))
        body += "<h4>Chips at the last keyframe</h4>" + kv(pairs) + _chip_forms(chips)
    fields_entry, fields = _last_with(ctx, run.keyframe_fields, "pile_height_m")
    if fields is not None:
        grid = np.asarray(fields["pile_height_m"], float)
        extent, caveat = _extent(fields, grid, "pile", "surface")
        png = plots.heatmap(grid * fmt.MM_PER_M, extent, "pile height (mm)",
                            f"Chip accumulation at {fmt.seconds(fields_entry.get('time_s'))} s")
        body += figure(png, "chip accumulation map", "Settled chip height over the part and fixture"
                       + (f" ({caveat})." if caveat else "."))
    else:
        body += note("No keyframe records a chip pile height field.")
    flagged = [f for f in run.findings if f.get("category") in ("chip_packing", "chip_recutting", "chip_hazard")]
    if flagged:
        body += "<h4>Chip findings</h4>" + ul([join([ctx.finding_link(f["id"]), " ", f.get("message", "")])
                                               for f in flagged])
    return Section("chips", "Chips", body, lead="Chip mass per part, where the chips end up, chip forms.")


def _chip_forms(chips: dict) -> str:
    codes = np.asarray(chips.get("form_code", []), np.int64)
    masses = np.asarray(chips.get("mass_kg", []), float)
    if not len(codes) or len(codes) != len(masses):
        return ""
    rows = []
    total = float(masses.sum())
    for code_value in sorted(set(codes.tolist())):
        mass = float(masses[codes == code_value].sum())
        rows.append([code_value, CHIP_FORMS.get(int(code_value), "unknown code"), fmt.grams(mass, 4),
                     fmt.pct(mass / total if total else None)])
    return table(["code", "chip form (ISO 3685)", "mass in flight (g)", "share (%)"], rows, numeric={0, 2, 3})


# ----------------------------------------------------------------------------- thermal

def _temperature_field(arrays: dict):
    names = sorted(k for k in arrays if k.endswith("temp_K") and np.asarray(arrays[k]).ndim in (2, 3))
    for name in names:
        grid = np.asarray(arrays[name], float)
        if grid.ndim == 3:
            grid = np.nanmax(grid, axis=2)
        return name, grid
    return None, None


def thermal_section(ctx: Context) -> Section:
    run = ctx.run
    body = ""
    maxima = []
    last = None
    for entry in run.keyframes:
        arrays = run.keyframe_fields(entry)
        if not arrays:
            continue
        name, grid = _temperature_field(arrays)
        if grid is None or not np.isfinite(grid).any():
            continue
        maxima.append((entry, name, float(np.nanmax(grid))))
        last = (entry, arrays, name, grid)
    if last is not None:
        entry, arrays, name, grid = last
        extent, caveat = _extent(arrays, grid, "surface", "thermal", "work_temp")
        png = plots.heatmap(grid - ZERO_DEGC_K, extent, "temperature (°C)",
                            f"{name} at {fmt.seconds(entry.get('time_s'))} s")
        body += figure(png, "thermal map", "Workpiece temperature at the last keyframe"
                       + (f" ({caveat})" if caveat else "") + "; a 3-D field is shown as its maximum through z.")
        body += table(["keyframe", "t (s)", "field", "max (°C)"],
                      [[e["k"], fmt.seconds(e.get("time_s")), n, fmt.degc(v)] for e, n, v in maxima],
                      numeric={0, 1, 3})
    else:
        body += note("No keyframe records a workpiece temperature field.")
    body += _unload(ctx)
    tools = (run.tools or {}).get("tools") or []
    hot = [[f"T{t.get('number')}", fmt.degc(t.get("max_edge_temp_K"))] for t in tools
           if t.get("max_edge_temp_K") is not None]
    if hot:
        body += "<h4>Cutting edges</h4>" + table(["tool", "hottest edge (°C)"], hot, numeric={1})
    flagged = [f for f in run.findings if f.get("category") in ("thermal_damage", "thermal_error")]
    if flagged:
        body += ul([join([ctx.finding_link(f["id"]), " ", f.get("message", "")]) for f in flagged])
    return Section("thermal", "Thermal state", body,
                   lead="Temperature map of the part and its temperature when it is unloaded.")


def _unload(ctx: Context) -> str:
    records = (ctx.run.inspection or {}).get("unload") or []
    rows = [[r.get("op", fmt.DASH), fmt.seconds(r.get("time_s")), fmt.degc(r.get("mean_temp_K")),
             fmt.degc(r.get("max_temp_K")), "inspection.json"] for r in records if isinstance(r, dict)]
    if not rows:
        macro = ctx.tier("macro")
        for name in ("thermal.work_unload_mean_K", "thermal.work_bulk_mean_K"):
            if macro is not None and macro.at_end(name) is not None:
                peak = macro.at_end("thermal.work_bulk_max_K") if "thermal.work_bulk_max_K" in macro.columns else None
                rows.append([fmt.DASH, fmt.seconds(macro.tick1 * macro.tick_s), fmt.degc(macro.at_end(name)),
                             fmt.degc(peak), f"{name} at the end of the run"])
                break
    if not rows:
        return note("Neither inspection.json (unload) nor the macro series give the part temperature at unload.")
    return "<h4>Part temperature at unload</h4>" + table(["op", "t (s)", "mean (°C)", "max (°C)", "source"],
                                                         rows, numeric={1, 2, 3})


# ----------------------------------------------------------------------------- coolant

def coolant_section(ctx: Context) -> Section:
    run = ctx.run
    body = ""
    macro = ctx.tier("macro")
    stats = macro.columns.get("coolant.reach_fraction") if macro is not None else None
    if stats is not None and stats.finite:
        curve = macro.curve("coolant.reach_fraction")
        on = curve[1] > 0.0
        pairs = [("share of the run with coolant reaching the cut", f"{fmt.pct(on.mean())} %")]
        if on.any():
            pairs += [("reach while on: mean", f"{fmt.pct(float(curve[1][on].mean()))} %"),
                      ("reach while on: lowest", f"{fmt.pct(float(curve[2][on].min()))} %")]
        body += kv(pairs)
    else:
        body += note("No coolant.reach_fraction series in the run directory.")
    body += _jets(ctx) + _wetted(ctx) + _shadows(ctx)
    flagged = [f for f in run.findings if f.get("category") == "coolant_shadowing"]
    body += ("<h4>Coolant shadow findings</h4>" + ul([join([ctx.finding_link(f["id"]), " ", f.get("message", "")])
                                                      for f in flagged])) if flagged else \
        note("No coolant-shadowing finding was raised.")
    return Section("coolant", "Coolant reach and shadows", body,
                   lead="How much of the coolant reaches the cut, per nozzle, and what blocks it.")


def _jets(ctx: Context) -> str:
    per_nozzle: dict[str, dict] = {}
    for entry in ctx.run.keyframes:
        arrays = ctx.run.keyframe_fields(entry)
        if not arrays or "jet_nozzle_id" not in arrays:
            continue
        ids = [str(v) for v in np.asarray(arrays["jet_nozzle_id"]).reshape(-1)]
        for i, nozzle in enumerate(ids):
            record = per_nozzle.setdefault(nozzle, {"kind": "", "flow": [], "speed": [], "reach": [], "frames": 0})
            record["frames"] += 1
            if "jet_kind" in arrays:
                record["kind"] = str(np.asarray(arrays["jet_kind"]).reshape(-1)[i])
            for key, name in (("flow", "jet_flow_m3_s"), ("speed", "jet_speed_m_s"), ("reach", "jet_reach_fraction")):
                if name in arrays:
                    record[key].append(float(np.asarray(arrays[name]).reshape(-1)[i]))
    if not per_nozzle:
        return note("No keyframe records coolant jets.")
    rows = []
    for nozzle, record in sorted(per_nozzle.items()):
        flow = np.mean(record["flow"]) if record["flow"] else None
        rows.append([nozzle, record["kind"], record["frames"],
                     fmt.num(None if flow is None else flow * fmt.L_MIN_PER_M3_S, 4),
                     fmt.num(np.mean(record["speed"]) if record["speed"] else None, 3),
                     fmt.pct(np.mean(record["reach"]) if record["reach"] else None),
                     fmt.pct(np.min(record["reach"]) if record["reach"] else None)])
    return "<h4>Jets</h4>" + table(["nozzle", "kind", "keyframes on", "flow (l/min)", "speed (m/s)",
                                    "reach mean (%)", "reach lowest (%)"], rows, numeric={2, 3, 4, 5, 6})


def _wetted(ctx: Context) -> str:
    for entry in reversed(ctx.run.keyframes):
        arrays = ctx.run.keyframe_fields(entry)
        if not arrays or "wetted" not in arrays:
            continue
        grid = np.asarray(arrays["wetted"], float)
        if grid.ndim != 2 or not grid.any():
            continue
        extent, caveat = _extent(arrays, grid, "wetted", "surface")
        png = plots.heatmap(grid, extent, "wetted (1) / dry (0)", f"Wetted surface at {fmt.seconds(entry.get('time_s'))} s",
                            vmin=0.0, vmax=1.0)
        return figure(png, "wetted map", f"Cells the coolant reaches (dry cells near the cut are shadows)"
                      + (f"; {caveat}" if caveat else "") + ".")
    return note("No keyframe records a wetted-surface field with coolant on.")


def _shadows(ctx: Context) -> str:
    rows = []
    for entry in ctx.run.keyframes:
        arrays = ctx.run.keyframe_fields(entry)
        if not arrays:
            continue
        ids = arrays.get("shadow_id")
        fractions = arrays.get("shadow_fraction")
        if ids is None or fractions is None:
            continue
        for occluder, fraction in zip(np.asarray(ids).reshape(-1), np.asarray(fractions).reshape(-1)):
            rows.append([entry["k"], fmt.seconds(entry.get("time_s")), str(occluder), fmt.pct(float(fraction))])
    if not rows:
        return note("No keyframe records coolant shadows (shadow_id / shadow_fraction).")
    return "<h4>Shadows</h4>" + table(["keyframe", "t (s)", "occluder", "flow blocked (%)"], rows, numeric={0, 1, 3})


# ----------------------------------------------------------------------------- chatter and stability

def chatter_section(ctx: Context) -> Section:
    run = ctx.run
    body = ""
    for name, document in sorted(run.sld.items()):
        speeds = np.asarray(document.get("speed_rad_s") or [], float) * fmt.RPM_PER_RAD_S
        depth = np.asarray(document.get("depth_limit_m") or [], float) * fmt.MM_PER_M
        points = []
        rows = []
        for point in document.get("operating_points") or []:
            rpm = abs(float(point.get("speed_rad_s", 0.0))) * fmt.RPM_PER_RAD_S
            ap = float(point.get("ap_m", 0.0)) * fmt.MM_PER_M
            label = f"{point.get('op', '')} line {point.get('line', '')}".strip()
            points.append({"rpm": rpm, "ap_mm": ap, "stable": bool(point.get("stable")), "label": label})
            op = str(point.get("op", ""))
            rows.append([op, ctx.line_link(op, point.get("program") or _program_of(ctx, op), int(point.get("line") or 0)),
                         fmt.fixed(rpm, 0), fmt.mm(point.get("ap_m"), 2),
                         "stable" if point.get("stable") else "UNSTABLE (chatter)"])
        title = f"{name}: stability lobes" + (f", ae {fmt.mm(document.get('radial_immersion_m'), 1)} mm"
                                              if document.get("radial_immersion_m") is not None else "")
        body += f"<h3>{name}</h3>" + figure(plots.stability_lobes(speeds, depth, points, title), f"{name} lobes",
                                            "Below the curve the cut is stable; markers are the program's operating "
                                            "points (circle stable, cross unstable).")
        body += table(["op", "line", "speed (rpm)", "depth (mm)", "prediction"], rows, numeric={2, 3},
                      empty="No operating points recorded.")
    if not run.sld:
        body += note("No stability lobe diagram (sld/) in the run directory.")
    flagged = [f for f in run.findings if f.get("category") == "chatter"]
    rows = [[ctx.finding_link(f["id"]), ctx.line_link(f.get("op", ""), f.get("program", ""), int(f.get("line") or 0)),
             fmt.seconds(f.get("time_s")), fmt.num((f.get("quantities") or {}).get("chatter_freq_Hz"), 4),
             fmt.num((f.get("quantities") or {}).get("chatter_index"), 3), f.get("message", "")] for f in flagged]
    body += "<h3>Chatter findings</h3>" + table(["finding", "line", "t (s)", "frequency (Hz)", "index", "message"],
                                                 rows, numeric={2, 3, 4}, empty="The run predicts no chatter.")
    return Section("chatter", "Chatter and stability", body)


# ----------------------------------------------------------------------------- deflection

def deflection_channels(ctx: Context) -> list[dict]:
    return [c for c in ctx.run.channels if c.get("unit") == "m" and c.get("domain") == "dynamics"
            and any(word in c["name"] for word in ("deflection", "displacement"))]


def deflection_section(ctx: Context) -> Section:
    run = ctx.run
    channels = deflection_channels(ctx)
    body = ""
    if not channels:
        body += note("channels.json has no tool deflection channel (dynamics.*deflection* or *displacement*, m).")
    tick = ctx.tier("tick")
    for channel in channels:
        tier = channel.get("tier")
        columns = []
        if tier == "fast" and tick is not None:
            columns = sorted(c for c in tick.columns if split_column(c)[0] == channel["name"]
                             and split_column(c)[2] in ("min", "max"))
            source = "tick"
        else:
            source = tier
            summary = ctx.tier(tier) if tier else None
            columns = sorted(c for c in (summary.columns if summary else {}) if split_column(c)[0] == channel["name"])
        per_tool = run.per_tool(source, columns) if columns else {}
        rows = []
        for tool in sorted(per_tool, key=lambda t: (t is None, t or 0)):
            stats = per_tool[tool]
            peaks = {}
            for column, (low, high, _, _) in stats.items():
                index = split_column(column)[1]
                peaks[index] = max(peaks.get(index, 0.0), abs(low), abs(high))
            bound = math.sqrt(sum(v * v for v in peaks.values()))
            rows.append(["no tool" if tool is None else f"T{tool}", len(peaks),
                         ", ".join(f"{k or '[0]'} {fmt.um(v, 1)}" for k, v in sorted(peaks.items())), fmt.um(bound, 1)])
        body += f"<h4>{channel['name']}</h4>" + table(
            ["tool", "components", "peak |u| per component (µm)", "peak |u| bound (µm)"], rows,
            numeric={1, 3}, caption=f"{channel.get('doc', '')}. Peaks over the {source} tier rows of each tool's blocks; "
                                    "the bound combines the component peaks (they need not occur together).",
            empty="No samples of this channel fall in a tool's blocks.")
    body += _deflection_share(ctx)
    return Section("deflection", "Tool deflection and its share of the errors", body,
                   lead="How far each tool bends and how much of each dimensional error that explains.")


def _deflection_share(ctx: Context) -> str:
    rows = []
    for item in [t for t in (ctx.run.inspection or {}).get("tolerances", []) if isinstance(t, dict)]:
        deviation = _explained_total(item)
        for entry in item.get("contributions_m") or []:
            if not isinstance(entry, dict) or "deflection" not in str(entry.get("source", "")):
                continue
            value = fmt.as_float(entry.get("value_m"))
            share = value / deviation if value is not None and deviation else None
            rows.append([item.get("id", ""), f"T{entry['tool']}" if entry.get("tool") is not None else fmt.DASH,
                         fmt.signed_um(value), fmt.signed_um(deviation), fmt.pct(share, 0)])
    if not rows:
        return note("inspection.json gives no deflection contributions to the characteristics.")
    return "<h4>Contribution to each dimensional error</h4>" + table(
        ["characteristic", "tool", "deflection (µm)", "deviation (µm)", "share (%)"], rows,
        numeric={2, 3, 4})


# ----------------------------------------------------------------------------- conservation

def conservation_section(ctx: Context) -> Section:
    run = ctx.run
    block = run.summary.get("conservation") or run.run.get("conservation")
    rows = []
    if isinstance(block, dict):
        for name, entry in sorted(block.items()):
            if not isinstance(entry, dict):
                continue
            residual = entry.get("max_residual_rel", entry.get("residual_rel"))
            tolerance = entry.get("tolerance_rel")
            ok = entry.get("ok")
            if ok is None and fmt.as_float(residual) is not None and fmt.as_float(tolerance) is not None:
                ok = abs(float(residual)) <= float(tolerance)
            exact = entry.get("residual_ticks")
            rows.append([name, fmt.num(residual, 3) if exact is None else f"{exact} ticks",
                         fmt.num(entry.get("final_residual_rel"), 3), fmt.num(tolerance, 3),
                         pass_badge(ok if isinstance(ok, bool) else None), entry.get("note", "")])
    body = table(["balance", "largest residual (relative)", "at the end", "tolerance", "result", "note"], rows,
                 numeric={1, 2, 3}, caption="Mass, energy and time balances, each term computed independently "
                                            "(architecture 12); a run above tolerance fails.",
                 empty="run.json has no conservation block: the balances were checked by the loop but their "
                       "residuals are not recorded.")
    macro = ctx.tier("macro")
    if macro is not None:
        series_rows = []
        for column in sorted(c for c in macro.columns if c.startswith("ledger.") or "consistency" in c
                             or "residual" in c):
            stats = macro.columns[column]
            series_rows.append([column, fmt.num(stats.min, 6), fmt.num(stats.max, 6), fmt.num(stats.mean, 6)])
        if series_rows:
            body += "<h4>Balance channels</h4>" + table(["channel", "min", "max", "mean"], series_rows,
                                                         numeric={1, 2, 3})
    body += _time_balance(ctx)
    return Section("conservation", "Conservation residuals", body)


def _time_balance(ctx: Context) -> str:
    blocks = [b for b in ctx.run.timeline if int(b.get("path", 0)) == 0]
    if not blocks:
        return ""
    ordered = sorted(blocks, key=lambda b: int(b["tick0"]))
    total = sum(int(b["tick1"]) - int(b["tick0"]) for b in ordered)
    span = int(ordered[-1]["tick1"]) - int(ordered[0]["tick0"])
    gaps = span - total
    return para(f"Time: path 0 runs {len(ordered)} blocks, {fmt.integer(total)} ticks of block time in a span of "
                f"{fmt.integer(span)} ticks ({'exact, no gaps' if gaps == 0 else f'{gaps} ticks of gaps or waits'}).")


__all__ = ["inspection_section", "surface_section", "chips_section", "thermal_section", "coolant_section",
           "chatter_section", "deflection_section", "conservation_section", "limits", "utilisation"]
