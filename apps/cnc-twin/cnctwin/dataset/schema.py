"""The columns of the dataset tables (docs/ml.md "Columns"): name, type, unit, provenance and meaning of each.

Two tables:

* ``runs``: one row per run (a sweep point or a run directory): identity, the sweep's parameters, the run's
  options, and the run-level results;
* ``blocks``: one row per executed NC block of every run (timeline.json): features (tool, material, spindle
  speed, feed per tooth, cutting speed, engagement, removal rate, path kind, coolant) and labels (forces, torque,
  power, temperatures, wear increment, deflection, vibration, chatter, roughness, burrs, findings).

Types: ``float64`` (NaN when the run does not have the value), ``int64``, ``int8`` (flags: 1 true, 0 false; -1
not known) and ``string``. Units are SI (the name's last token, as everywhere in the twin; ``-`` for
dimensionless). Some columns exist per tool, characteristic, feature or category; their names carry the item in
brackets (``VB_max_m[T1]``, ``deviation_m[pocket-depth]``, ``findings[chatter]``) and their schema entries are made
from the data (:func:`tool_columns` ...). Parameters of a sweep are ``param.<name>`` with the unit the spec
declares.
"""

from __future__ import annotations

from dataclasses import dataclass

FORMAT = "cnctwin-dataset-schema"
VERSION = 1
DTYPES = ("float64", "int64", "int8", "string")


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    unit: str
    provenance: str
    doc: str

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "unit": self.unit, "provenance": self.provenance,
                "doc": self.doc}


def C(name, dtype, unit, provenance, doc) -> Column:
    assert dtype in DTYPES, dtype
    return Column(name, dtype, unit, provenance, doc)


TICK_FORCE = "series/tick cutting.force_x_N, force_y_N, force_z_N [path]"
TICK_TORQUE = "series/tick cutting.torque_Nm [path]"
TICK_DEFLECTION = "series/tick dynamics.tool_deflection_x_m, tool_deflection_y_m [path]"
TICK_SPEED = "series/tick motion.path_speed_m_s [path]"
TICK_OMEGA = "series/tick motion.spindle_omega_rad_s [spindle of the path]"

# ----------------------------------------------------------------------------- runs: fixed columns

RUN_IDENTITY = [
    C("sweep", "string", "-", "sweep.json name", "name of the sweep the run belongs to ('' for a run given directly)"),
    C("point", "int64", "-", "plan.json points[].index", "index of the sweep point (-1 for a run given directly)"),
    C("run_id", "string", "-", "run.json run_id", "the run's id"),
    C("cache_key", "string", "-", "points/p<n>/result.json key", "run cache key of the point ('' when unknown)"),
    C("project_name", "string", "-", "run.json project.name", "project name (a sweep variant's carries its values)"),
    C("project_hash", "string", "-", "run.json project_hash", "sha256 of the project's content and inputs"),
    C("build", "string", "-", "run.json versions.build", "build id of the code that made the run"),
    C("seed", "int64", "-", "run.json seed", "the project's seed (stochastic models)"),
    C("machine", "string", "-", "run.json project.machine.id", "machine id"),
    C("material", "string", "-", "run.json project.material.db_id (else .id)", "workpiece material id"),
    C("complete", "int8", "-", "run.json complete", "1: the run simulated the whole program (0: max_sim_s cut it)"),
    C("safe", "int8", "-", "run.json verdict.safe", "1: no finding of severity error or fatal (architecture 11)"),
]
RUN_OPTIONS = [
    C("feed_override", "float64", "-", "run.json options.operator.feed_override['*']",
      "feed override of every tool not named per tool (1 = 100 %, cnctwin.sweep.operator)"),
    C("spindle_override", "float64", "-", "run.json options.operator.spindle_override['*']",
      "spindle override of every tool not named per tool (1 = 100 %)"),
    C("rapid_override", "float64", "-", "run.json options.operator.rapid_override", "rapid override (1 = 100 %)"),
]
RUN_RESULTS = [
    C("cycle_time_s", "float64", "s", "run.json summary.cycle_time_s", "machine cycle time (longest path)"),
    C("sim_time_s", "float64", "s", "run.json summary.sim_time_s", "simulated time"),
    C("takt_s", "float64", "s", "production.json per_part_s", "time per part including the operator model"),
    C("removed_mass_kg", "float64", "kg", "run.json summary.removed_mass_kg", "removed material (exact field volume x rho)"),
    C("removed_volume_m3", "float64", "m3", "run.json summary.removed_volume_m3", "removed material volume"),
    C("chip_mass_kg", "float64", "kg", "run.json summary.chips_kg (sum of the six inventories)",
      "chip mass booked by the chips domain"),
    C("cutting_work_J", "float64", "J", "run.json summary.shaft_work_J + feed_work_J",
      "work of the cut: spindle shaft work plus feed work"),
    C("energy_per_part_J", "float64", "J", "production.json per_part[simulated].energy.total_J (mean)",
      "machine energy per part (spindle, axes, pumps, base load, air)"),
    C("tool_cost_per_part_eur", "float64", "eur", "production.json per_part[simulated].tool_cost_eur (mean)",
      "tool cost per part from the wear model and tool prices"),
    C("force_peak_max_N", "float64", "N", "series/macro cutting.force_peak_N [path]: maximum",
      "largest instantaneous tool force (L0 peak of every macro step)"),
    C("force_max_N", "float64", "N", f"{TICK_FORCE}: |F| per tick, maximum", "largest tick-mean tool force"),
    C("force_mean_N", "float64", "N", f"{TICK_FORCE}: |F| per tick, mean over the ticks in cut (|F| > 0)",
      "mean tool force while cutting"),
    C("torque_max_Nm", "float64", "N m", f"{TICK_TORQUE}: |T| maximum", "largest tick-mean cutting torque"),
    C("torque_mean_Nm", "float64", "N m", f"{TICK_TORQUE}: |T| mean over the ticks in cut", "mean cutting torque while cutting"),
    C("spindle_power_max_W", "float64", "W", "series/macro cutting.spindle_power_W [path]: maximum",
      "largest step-mean spindle cutting power"),
    C("spindle_power_mean_W", "float64", "W", "series/macro cutting.spindle_power_W [path]: mean over the steps in cut "
      "(cutting.in_cut_fraction > 0)", "mean spindle cutting power while cutting"),
    C("spindle_load_max", "float64", "-", "series/macro cutting.spindle_load_fraction [path]: maximum",
      "largest cutting torque over the available torque at speed and gear"),
    C("spindle_thermal_load_max", "float64", "-", "series/macro cutting.spindle_thermal_load [path]: maximum",
      "largest spindle motor thermal load (1 = S1 limit)"),
    C("mrr_mean_m3_s", "float64", "m3/s", "series/macro cutting.chip_volume_rate_m3_s [path]: mean over the steps in cut",
      "mean material removal rate while cutting"),
    C("tool_deflection_max_m", "float64", "m", f"{TICK_DEFLECTION}: lateral magnitude, maximum",
      "largest tool deflection at the reference station"),
    C("vibration_pp_max_m", "float64", "m", "series/macro dynamics.vibration_pp_path_m [path]: maximum",
      "largest peak-to-peak relative tool-work vibration of a step"),
    C("chatter_index_max", "float64", "-", "series/macro dynamics.chatter_index_path [path]: maximum",
      "largest non-forced share of the vibration energy (chatter)"),
    C("stability_margin_min", "float64", "-", "series/macro dynamics.stability_margin [path]: minimum (finite, > 0)",
      "smallest depth limit over depth of cut at the operating point (below 1: chatter predicted)"),
    C("tool_edge_temp_max_K", "float64", "K", "series/macro thermal.tool_edge_max_K: maximum",
      "hottest cutting edge"),
    C("tool_flank_temp_max_K", "float64", "K", "series/macro thermal.tool_flank_max_K: maximum", "hottest flank contact"),
    C("work_temp_max_K", "float64", "K", "series/macro thermal.work_bulk_max_K [workpiece]: maximum",
      "hottest finite volume of the workpiece"),
    C("work_surface_temp_max_K", "float64", "K", "series/macro thermal.work_surface_K [workpiece]: maximum",
      "hottest surface temperature at the contact"),
    C("spindle_temp_max_K", "float64", "K", "series/macro thermal.spindle_temp_K [spindle]: maximum", "hottest spindle"),
    C("chip_temp_max_K", "float64", "K", "series/macro chips.max_temp_K: maximum", "hottest chip"),
    C("coolant_tank_temp_max_K", "float64", "K", "series/macro coolant.tank_temp_K: maximum", "warmest coolant tank"),
    C("holder_margin_min", "float64", "-", "series/macro holding.tool_margin: minimum (finite)",
      "smallest holder friction capacity over peak demand (below 1: the tool slips or pulls out)"),
    C("part_slip_max_m", "float64", "m", "series/macro holding.part_slip_m [workpiece]: maximum",
      "largest persistent slip of a workpiece in its setup"),
    C("burr_max_m", "float64", "m", "surface.json burrs[].height_m: maximum", "highest exit burr"),
    C("burr_count", "int64", "-", "surface.json burrs", "exit burrs recorded"),
    C("face_Ra_max_m", "float64", "m", "surface.json faces[].Ra_m: maximum",
      "roughest machined face not named by a roughness tolerance"),
    C("face_Ra_mean_m", "float64", "m", "surface.json faces[].Ra_m: mean", "mean Ra of those faces"),
    C("mass_balance_ok", "int8", "-", "run.json summary.conservation.mass.ok", "1: the mass balance held every step"),
    C("energy_balance_ok", "int8", "-", "run.json summary.conservation.energy.ok", "1: the energy balance held every step"),
    C("findings_count", "int64", "-", "findings.json", "findings of the run"),
]
SEVERITIES = ("info", "warning", "error", "fatal")
RUN_SEVERITY = [C(f"findings[{s}]", "int64", "-", f"findings.json severity == {s}", f"findings of severity {s}")
                for s in SEVERITIES]


def category_columns(categories) -> list[Column]:
    return [C(f"findings[{c}]", "int64", "-", f"findings.json category == {c}", f"findings of category {c}")
            for c in categories]


def parameter_columns(parameters: list[dict]) -> list[Column]:
    out = []
    for p in parameters:
        categorical = "values" in p and any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in p["values"])
        doc = p.get("doc") or f"sweep parameter {p['name']} ({p['kind']}"
        if not p.get("doc"):
            detail = p.get("target") or p.get("path") or p.get("placeholder") or p.get("argument") or ""
            doc += f": {detail})" if detail else ")"
        out.append(C(f"param.{p['name']}", "string" if categorical else "float64", str(p.get("unit") or "-"),
                     "plan.json points[].values", doc))
    return out


def tool_columns(numbers) -> list[Column]:
    out = []
    for n in numbers:
        out += [C(f"VB_max_m[{n}]", "float64", "m", f"tools.json tools[number={n}].VB_max_m", f"largest flank wear of {n}"),
                C(f"KT_max_m[{n}]", "float64", "m", f"tools.json tools[number={n}].KT_max_m", f"deepest crater of {n}"),
                C(f"life_used_fraction[{n}]", "float64", "-", f"tools.json tools[number={n}].life_used_fraction",
                  f"life used of {n} (first criterion)"),
                C(f"cut_time_s[{n}]", "float64", "s", f"tools.json tools[number={n}].cut_time_s", f"cutting time of {n}"),
                C(f"edge_temp_max_K[{n}]", "float64", "K", f"tools.json tools[number={n}].max_edge_temp_K",
                  f"hottest edge of {n}"),
                C(f"broken[{n}]", "int8", "-", f"tools.json tools[number={n}].broken", f"1: {n} broke")]
    return out


def inspection_columns(ids) -> list[Column]:
    out = []
    for i in ids:
        out += [C(f"measured_m[{i}]", "float64", "m", f"inspection.json tolerances[id={i}].measured_m",
                  f"value of characteristic {i} as the virtual CMM measures it (20 degC, unclamped)"),
                C(f"deviation_m[{i}]", "float64", "m", f"inspection.json tolerances[id={i}].deviation_m",
                  f"deviation of {i} from its nominal"),
                C(f"pass[{i}]", "int8", "-", f"inspection.json tolerances[id={i}].pass", f"1: {i} is in tolerance")]
    return out


def surface_columns(ids) -> list[Column]:
    out = []
    for i in ids:
        out += [C(f"Ra_m[{i}]", "float64", "m", f"surface.json features[id={i}].Ra_m", f"Ra of feature {i}"),
                C(f"Rz_m[{i}]", "float64", "m", f"surface.json features[id={i}].Rz_m", f"Rz of feature {i}")]
    return out


# ----------------------------------------------------------------------------- blocks

BLOCK_COLUMNS = [
    C("sweep", "string", "-", "sweep.json name", "sweep of the run ('' for a run given directly)"),
    C("point", "int64", "-", "plan.json points[].index", "sweep point (-1 for a run given directly)"),
    C("run_id", "string", "-", "run.json run_id", "the run's id"),
    C("block", "int64", "-", "timeline.json blocks[].index", "position of the block in the run's timeline"),
    C("op", "string", "-", "timeline.json blocks[].op", "operation"),
    C("program", "string", "-", "timeline.json blocks[].program (file name)", "program file name"),
    C("line", "int64", "-", "timeline.json blocks[].line", "line of the block in its program"),
    C("path", "int64", "-", "timeline.json blocks[].path", "controller path (0-based)"),
    C("part", "int64", "-", "timeline.json blocks[].part", "part of the batch (0-based)"),
    C("kind", "string", "-", "timeline.json blocks[].kind", "path kind: rapid, linear, arc, dwell, tool_change, ..."),
    C("tool", "int64", "-", "timeline.json blocks[].tool", "T number active (-1: none)"),
    C("t0_s", "float64", "s", "timeline.json blocks[].tick0 x tick_s", "start of the block"),
    C("duration_s", "float64", "s", "timeline.json blocks[] (tick1 - tick0) x tick_s", "duration of the block"),
    C("cutting_s", "float64", "s", "timeline.json blocks[].ticks.cutting x tick_s", "time in cut"),
    C("air_cutting_s", "float64", "s", "timeline.json blocks[].ticks.air_cutting x tick_s", "feed time out of contact"),
    # features
    C("tool_type", "string", "-", "run.json project.tools.list[number].tool.type", "tool type (database template)"),
    C("tool_diameter_m", "float64", "m", "run.json project.tools.list[number].tool.geometry.diameter_m", "cutting diameter"),
    C("tool_flutes", "int64", "-", "run.json project.tools.list[number].tool.geometry.flutes", "flutes (teeth)"),
    C("tool_corner_radius_m", "float64", "m", "run.json project.tools.list[number].tool.geometry.corner_radius_m",
      "corner radius"),
    C("tool_helix_rad", "float64", "rad", "run.json project.tools.list[number].tool.geometry.helix_rad", "helix angle"),
    C("tool_rake_rad", "float64", "rad", "run.json project.tools.list[number].tool.geometry.rake_rad", "rake angle"),
    C("tool_material_class", "string", "-", "run.json project.tools.list[number].tool.tool_material.class",
      "tool substrate class"),
    C("tool_coating", "string", "-", "run.json project.tools.list[number].tool.coating.id", "coating id"),
    C("tool_stickout_m", "float64", "m", "run.json project.tools.list[number].stickout_m", "stick-out from the holder"),
    C("tool_runout_m", "float64", "m", "run.json project.tools.list[number].runout_m", "runout (TIR)"),
    C("material", "string", "-", "run.json project.material.db_id (else .id)", "workpiece material id"),
    C("material_hardness_HB", "float64", "HB", "run.json project.material.physical.hardness.HB", "Brinell hardness"),
    C("feed_override", "float64", "-", "run.json options.operator.feed_override [the block's tool, else '*']",
      "feed override of the block's tool (1 = 100 %)"),
    C("spindle_override", "float64", "-", "run.json options.operator.spindle_override [the block's tool, else '*']",
      "spindle override of the block's tool (1 = 100 %)"),
    C("spindle_speed_rad_s", "float64", "rad/s", f"{TICK_OMEGA}: |omega|, mean over the block's ticks",
      "spindle speed (actual, from the spindle servo)"),
    C("feed_speed_m_s", "float64", "m/s", f"{TICK_SPEED}: mean over the block's ticks", "mean tool-tip path speed"),
    C("feed_speed_cut_m_s", "float64", "m/s", f"{TICK_SPEED}: mean over the block's ticks in cut (|F| > 0)",
      "mean path speed while cutting"),
    C("feed_per_tooth_m", "float64", "m", "feed_speed_cut_m_s / (spindle_speed_rad_s / 2 pi x tool_flutes)",
      "feed per tooth while cutting"),
    C("cutting_speed_m_s", "float64", "m/s", "spindle_speed_rad_s x tool_diameter_m / 2", "cutting speed at the diameter"),
    C("ae_m", "float64", "m", "series/macro removal.ae_m [path]: maximum over the steps overlapping the block",
      "radial engagement"),
    C("ap_m", "float64", "m", "series/macro removal.ap_m [path]: maximum over the steps overlapping the block",
      "axial engagement"),
    C("engaged_angle_rad", "float64", "rad", "series/macro removal.engaged_angle_rad [path]: maximum", "engaged angle"),
    C("mrr_m3_s", "float64", "m3/s", "series/macro cutting.chip_volume_rate_m3_s [path]: mean over the overlapping steps "
      "in cut", "material removal rate"),
    C("coolant_on", "float64", "-", "series/macro coolant.on: mean over the overlapping steps",
      "share of the block with coolant, MQL or air at the outlets"),
    C("coolant_flow_m3_s", "float64", "m3/s", "series/macro coolant.flow_m3_s: mean", "coolant delivered"),
    # labels
    C("force_mean_N", "float64", "N", f"{TICK_FORCE}: |F| per tick, mean over the block's ticks in cut",
      "mean tool force while cutting"),
    C("force_max_N", "float64", "N", f"{TICK_FORCE}: |F| per tick, maximum over the block", "largest tick-mean force"),
    C("force_peak_N", "float64", "N", "series/macro cutting.force_peak_N [path]: maximum over the overlapping steps",
      "largest instantaneous force"),
    C("torque_mean_Nm", "float64", "N m", f"{TICK_TORQUE}: |T| mean over the block's ticks in cut", "mean cutting torque"),
    C("torque_max_Nm", "float64", "N m", f"{TICK_TORQUE}: |T| maximum over the block", "largest cutting torque"),
    C("spindle_power_mean_W", "float64", "W", "series/macro cutting.spindle_power_W [path]: mean over the overlapping "
      "steps in cut", "spindle cutting power"),
    C("spindle_power_max_W", "float64", "W", "series/macro cutting.spindle_power_W [path]: maximum", "largest spindle power"),
    C("spindle_load_max", "float64", "-", "series/macro cutting.spindle_load_fraction [path]: maximum",
      "largest spindle load"),
    C("tool_edge_temp_max_K", "float64", "K", "series/macro thermal.tool_edge_max_K: maximum over the overlapping steps",
      "hottest edge"),
    C("work_temp_max_K", "float64", "K", "series/macro thermal.work_bulk_max_K [workpiece]: maximum", "hottest workpiece"),
    C("VB_increment_m", "float64", "m", "series/macro wear.VB_max_m: maximum - minimum over the overlapping steps",
      "flank wear grown during the block (steps shared with neighbouring blocks included)"),
    C("life_used_increment", "float64", "-", "series/macro wear.life_used_slot [the tool's slot]: maximum - minimum",
      "tool life used during the block"),
    C("deflection_max_m", "float64", "m", f"{TICK_DEFLECTION}: lateral magnitude, maximum over the block",
      "largest tool deflection"),
    C("vibration_pp_max_m", "float64", "m", "series/macro dynamics.vibration_pp_path_m [path]: maximum",
      "largest peak-to-peak vibration"),
    C("chatter_index_max", "float64", "-", "series/macro dynamics.chatter_index_path [path]: maximum", "chatter index"),
    C("stability_margin_min", "float64", "-", "series/macro dynamics.stability_margin [path]: minimum (finite, > 0)",
      "stability margin at the operating point"),
    C("Ra_m", "float64", "m", "surface.json features[] whose op and lines name the block: maximum Ra_m",
      "roughness of the named surfaces this block generated"),
    C("burr_max_m", "float64", "m", "surface.json burrs[block == block].height_m: maximum", "highest burr the block left"),
    C("findings_count", "int64", "-", "findings.json: tick in [tick0, tick1), same op and line",
      "findings raised in the block"),
    C("findings_severe_count", "int64", "-", "findings.json as findings_count, severity error or fatal",
      "errors and fatal findings raised in the block"),
    C("findings_categories", "string", "-", "findings.json as findings_count: categories, sorted, ';'-separated",
      "categories of the block's findings"),
]


def schema_document(tables: dict[str, list[Column]]) -> dict:
    return {"format": FORMAT, "version": VERSION,
            "tables": {name: [c.to_dict() for c in columns] for name, columns in tables.items()},
            "dtypes": {"float64": "IEEE double; NaN (empty in CSV) when the run does not have the value",
                       "int64": "integer", "int8": "flag: 1 true, 0 false, -1 not known", "string": "UTF-8 text"}}


__all__ = ["Column", "RUN_IDENTITY", "RUN_OPTIONS", "RUN_RESULTS", "RUN_SEVERITY", "BLOCK_COLUMNS", "SEVERITIES",
           "category_columns", "parameter_columns", "tool_columns", "inspection_columns", "surface_columns",
           "schema_document", "FORMAT", "VERSION"]
