# File formats

Every file the simulator reads or writes is plain text or a documented binary, carries a
`format` name and a `version`, and is migrated forward by the loader when an older version
is read.

## Databases

(owner: `cnctwin/db`; templates and the authors' rules: `db/_templates/*.yaml`, `db/_templates/README.md`)

### Files, kinds and versions

Every database file is YAML 1.2 (core schema: `1e-9` is a float, `yes`/`no`/dates stay text,
a mapping may not repeat a key, merge keys `<<:` work), UTF-8, with a `format` and a `version`.

| folder (`db/<folder>/**/*.yaml`) | format | kind of object | id |
|---|---|---|---|
| `materials` | `cnctwin-material` | material | `id` = file name |
| `tools` | `cnctwin-tool` | tool series: `items:` are tools | `series_id` = file name; each item `id` |
| `tool_materials` | `cnctwin-tool-material` | tool substrate | `id` = file name |
| `coatings` | `cnctwin-coating` | coating | `id` = file name |
| `holders` | `cnctwin-holder` | holder series: `items:` are holders | `series_id`; item `id` |
| `machines` | `cnctwin-machine` | machine | `id` = file name |
| `coolants` | `cnctwin-coolant` | cutting fluid | `id` = file name |
| `workholding` | `cnctwin-workholding` | workholding series: `items:` | `series_id`; item `id` |
| `controllers` | `cnctwin-controller` | control reference (alarms, words, timing) | `id` = file name |
| `models` | `cnctwin-model-params` | model coefficients | `model` = file name |
| `bibliography` | `cnctwin-bibliography` | sources (`entries:`) | entry key |
| `validation/cases/<quantity>/` | `cnctwin-validation-case` | validation case | `id` |

The current version of every format is 1. Ids are lower case (`[a-z0-9][a-z0-9._+-]*`); an id may
exist once per kind (series items are flattened: two series cannot define the same tool id).
Other folders under `db/` (and `_templates`) are not database files (a folder that is not a kind
is reported as a warning). `cnctwin.db.Database.load(root)` reads everything; `twin db check`
(`python -m cnctwin.db.check [--root DIR] [--strict] [--estimates] [--migrations] [--rules] [--json]`)
prints every problem as `file:line: key.path: message` and exits 1 if there is an error.

### Units in keys

Contract v2 (architecture 3): **the unit is the last token of the key** (`travel_mm`,
`kc11_N_mm2`, `max_rpm`, `growth_at_max_speed_um`, `min_abs_pressure_kPa`); minimum/maximum are
prefixes; `_min` at the end always means minutes; temperature differences are `_dK` (no offset;
`_degC` is only an absolute temperature); a per-length unit is `_per_mm`; hardness scales
(`_HV`, `_HB`, `_HRC` ...), currencies (`_eur`, `_usd`) and `_count` are labels, not converted.
Tokens are matched on an underscore boundary, the longest first (`max_feed_mm_min` is mm/min,
`time_constant_min` minutes, `index_90deg_s` seconds, `kv_1_s` 1/s); only `rpm` may stand alone
as a key (the span of a gear range; its SI key is `rad_s`).

The token table is `cnctwin.units.SUFFIXES` plus (`cnctwin/db/keyunits.py`, `EXTRA_UNITS`):
`rad_s`, `deg_s` (rotary speed), `1_s`, `mm2_s`/`m2_s` (kinematic viscosity), `kJ_kg`/`J_kg`,
`kg_mm`/`kg_m` (mass moment), `mJ`, `V`, `VA`, `kVA`, `N_per_Nm`, `per_MPa`/`per_Pa` (Usui-type
constants), `per_m`, `l_h`, `ml_min`, `J_m2`, `kJ_m2`, `um_per_min`, `s_per_rev`, and the labels
`HB HV HRC HRB HRA HK eur usd eur_per_l count`. The SI key replaces the unit token by its SI
token: `travel_mm` -> `travel_m`, `max_rpm` -> `max_rad_s`, `time_constant_min` ->
`time_constant_s`, `concentration_pct` -> `concentration_frac`, `ambient_degC` -> `ambient_K`.

**The schemas declare every field's unit** (`cnctwin/db/schemas.py`): a field's unit is its key's
last token, or given explicitly where a template key carries it elsewhere (`max_deg_per_step`,
`max_accel_rad_s2_per_gear`, whose SI keys are `max_rad_per_step` and the key itself). A file key
with the stem of a declared field but another unit is an error that names the declared unit
(`travel_in: the declared unit of 'travel' is mm, not in; write 'travel_mm'`); older key forms
are renamed only by the migration table below, never accepted as synonyms. A unit-less key under
a unit key inherits its unit (`fz_mm: {min, start, max}`, `rapid_mm_min: {X: .., Y: ..}`,
`home_mm: {X: 0}`, `body_mm: {outer_diameter, length}`). Unknown keys are errors except `note`
and `notes` (free text, anywhere) and, in materials, the `extensions:` section (data outside the
template, kept as written, read by no model; its numbers still need sources).

### Value forms

* a number, a list (`[lo, hi]` windows, vectors, per-flute lists), text, true/false, or `null`
  (on an optional field: **not published**; a project that needs the value says so, see
  "Project file");
* the sourced form `{value: x, src: <key>|est, note: "...", range: [lo, hi], page: "...",
  digitised: true, uncertainty_pct: n}` (`range` is the source's spread around the value used,
  `digitised` needs `uncertainty_pct`; declared extras such as `at_handle_torque_Nm` sit beside);
* a table `{table: [[x, v], ...], x: <unit>}`: x is declared per field (temperature-dependent
  properties allow only `degC`, which is also the default when `x:` is absent; a magnetic chuck's
  `holding_pressure_N_cm2` allows `x: mm`, which must then be written); rows sorted by x;
* a two-column curve declared by its schema path (`Pairs`): `spindles[].gear_ranges[].power_kW`
  and `.torque_Nm` and `additional_ratings[]` are `[[rpm, kW|Nm], ...]`; `residual_stress[].profile_MPa`
  is `[[z/thickness, MPa]]`; `coolant.pumps[].pressure_flow_bar_l_min` `[[bar, l/min]]`;
  workholding `centrifugal_loss_model.gripping_force_curve_kN` `[[rpm, kN]]` and
  `clamp_force_per_roller_vs_pressure_kN` `[[bar, kN]]`;
* a pose `{translate_mm: [x, y, z], rotate_deg: [a, b, c], order: xyz}` (kinematic `fixed` links,
  mesh primitives, project setups): `p_parent = R p_child + t`, `R = R(order[0]) R(order[1]) R(order[2])`
  with the angles indexed by axis (a about x, b about y, c about z): for `xyz` first about x, then
  about the new y, then the new z (intrinsic; three.js `Euler(a, b, c, 'XYZ')`). `order` is one of
  `xyz xzy yxz yzx zxy zyx`, default `xyz`.

The SI view (`cnctwin.db.simap`): an object is a `SIMap` (attribute and item access by SI key,
absent optional keys are `None` through attributes, `.raw` is the migrated data, `.to_dict()`
plain SI values). A sourced value is a `Sourced` (`float(x)` SI, `.src`, `.note`, `.estimated`,
`.range` SI, extras as attributes; it acts as a number or list); a table is a `Table` (call it
with the SI x, kelvin for temperature tables; linear interpolation, the end value held outside
with the clamps counted in `.clamped`; `float(table)` is the 20 degC value of a temperature
table); a curve is a `Curve` (a list of `[x_SI, y_SI]` rows, `.x`, `.y`, callable, end values held);
a pose is a `PoseMap` (`.matrix()` is the 4x4). `SIMap(raw)` without a schema infers units from the
key suffixes and curves from the curve keys the schemas declare (ad-hoc data and tests only).

Accuracy: unit conversions are exact up to IEEE-754 double rounding (factors from
`cnctwin.units`: SI definitions, CODATA 2018 / ISO 80000; relative error below 1e-15). Tables and
curves are interpolated linearly between the written points and hold their end values outside:
the error between points is the resolution of the source table itself (no smoothing is added),
and every evaluation outside the table is counted (`Table.clamped`) so a report can say a property
was extrapolated. A value is never rounded, scaled or invented by the loader: migrated keys keep
their numbers, added reference values (`h_ref_mm: 1`, `T_ref_min: 1`) are the definitions the older
template implied.

### Sources and estimates

Checked on each file **as written** (before migration, so paths and lines are the file's own):
every number needs an effective `src` (its own, or the `src` of the nearest enclosing mapping;
the series `src` covers its items). A `src` is a key of the merged bibliography or `est`. An `est`
needs a `note` saying how it was derived: its own, or the note written with the `src: est` of the
section it inherits from. A missing note is a **warning** (the value is still listed as an
estimate, with an empty note) and an error with `--strict` / `Database.load(strict=True)`: the
research files have about 700 such values, which would otherwise make every machine unusable;
`twin db check` lists each declaration once. A `src` that resolves to nothing, or a number with no
source at all, is an error (`SourceError` with file, line and key path when the object is read).
Definitional values need no source: the values of a material `skin` of `type: none`.
`Database.estimates()` lists every number whose effective source is `est` (file, line, key path,
value as written, note); `twin db check --estimates` prints them.

**Bibliography** (`db/bibliography/*.yaml`, `entries: {key: {type, authors, title, venue, year,
doi, url, accessed, pages, ...}}`; types article, book, chapter, conference, handbook, catalogue,
standard, manual, datasheet, web, thesis, report, patent, software, dataset; dated types need
`year`, the others `year` or `accessed`). All files are merged. A key may be defined in several
files if it names the same work: title and year, and authors, edition, doi, isbn, issn where both
give them, must agree (case-, space- and punctuation-insensitive), else it is an error; other
descriptive fields written differently are a warning (the first definition, files in name order,
is used); `pages`, `note` and `accessed` are merged (`a | b`).

### extends, series and references

* `extends: <id>` (materials, tool materials, coatings, machines, coolants, controllers; also series
  items and inline project objects): the object is deep-merged onto its base (chains allowed; a
  missing base or a cycle is an error). Mappings merge key by key; a value, list, sourced value or
  table replaces the base value whole (an extending machine that changes one spindle repeats the
  spindle list). Before merging, every mapping of both sides is given the `src` in force where it
  was written, so values keep their own sources. The extending file alone is checked for wrong keys
  and values; the merged object is checked completely. `dmg-dmu50-3rd-siemens` extends
  `dmg-dmu50-3rd` (its `same_mechanics_as` is migrated).
* Series files: every item is an object of its own with the series `maker`, `series`, `series_id`
  and `src` (unless it sets them) and the series `defaults:` mapping deep-merged under it.
* References checked: tool `tool_material.id`, `coating.id` (or `none`), tool `cutting_data[].material`,
  machine `controller.alarms` (a missing controller is a warning: the machine is usable, its alarm
  texts are not), validation `reference.src` and `setup.material.db`. A dangling reference is an
  error of the object that names it.

### Kind notes

* **machine**: `kinematics.workpiece_chain` / `tool_chain` are the main chains; further chains by
  name in `workpiece_chains: {<name>: [...]}` (sub spindle) and `tool_chains: {<name>: [...]}`
  (back tool post, second turret); every axis must be in some chain or turret; `home_mm` holds
  linear axes, `home_deg` rotary ones. A turret `chain` may list links by name (links of a chain).
  `turret_index_axis` is an axis name or the index axis direction. `capacity:` is the lathe working
  envelope. A spindle without `gear_ranges` is a warning (no power/torque limits).
* **material**: `flow_stress.model` is `johnson_cook` (required for metals), `temperature_table`
  (polymers: `tensile_strength_MPa` table, `compressive_yield_MPa`, `T_soft_degC`) or `none`
  (no constitutive model: optional `yield_MPa`, `compressive_flow_MPa`, `shear_yield_MPa`,
  `T_soft_degC`, `T_melt_degC`). Kienzle entries carry `h_ref_mm` and `b_ref_mm`, Taylor entries
  `T_ref_min`. Usui constants (`tool_life_usui[]`, tool material `usui[]`) give exactly one of
  `A_per_MPa` / `A_per_Pa` with `rate_unit` one of `"mm/min with sigma_n in MPa, V_s in m/min"`
  (A_SI = 1e-9 A), `"mm/s with sigma_n in MPa, V_s in mm/s"` (1e-6 A), `"m/s with sigma_n in Pa,
  V_s in m/s"` (A); the SI view's `A_per_Pa` is always in SI rates.
* **workholding**: `centrifugal_loss_model.method` is `vdi3106`, `measured`
  (`gripping_force_curve_kN`) or `quadratic_in_speed` (`loss_kN` at `at_rpm`, loss proportional
  to n^2); `body_mm` is a box `[[min], [max]]` or a cylinder `{outer_diameter, length, ...}`.
* **controller**: `alarms[]` (`code`, `text`, `trigger`, `cause`, `remedy`), `timing`, and the
  reference sections `messages`, `words`, `variables`, checked as free-form data.
* **model** (`db/models/<model>.yaml`): `params: {<name>: {value, unit, src, note, range}}`;
  `unit` is the unit the value is written in (a key unit token or plain text: `mm`, `1/s`,
  `W/(m K)`, `pct`; `-` dimensionless); the name carries the SI unit of the value
  `cnctwin.db.models.params(model)` returns (`arc_alarm_tolerance_m: {value: 0.02, unit: mm}`
  reads 2e-5); a name with a non-SI unit token, or a converted unit without a unit in the name,
  is an error. `range` (written unit) must contain the value. Values are cached per process
  (`cnctwin.db.models.clear_cache()` for tests).
* **validation case**: the `reference.src` must be a bibliography key; the setup is free-form
  data (values not stated by the source are `{value: .., assumed: true}`).

### Migration table (older key forms, applied at load time)

The research files were partly written against the first templates. `cnctwin.db.migrate`
rewrites a copy of each file into the current form (the file is not changed); each change is
recorded (`Database.migrations()`, `twin db check --migrations`: file, line, old and new key
path, rule). Values moved to `notes` keep their text (`key: value`); values moved elsewhere keep
the source they were written with. A value that cannot be migrated mechanically is moved to
`notes` and reported as a `migration` warning: it is not used until the file is corrected.

| rule | kinds | at | change | why |
|---|---|---|---|---|
| `pose-6-vector` | all database kinds | `**` | pose 6 vectors | a pose is {translate_mm, rotate_deg, order}: <name>_mm_deg [x, y, z, a, b, c] is <name> with order xyz (mechanical when at most one angle is non-zero) |
| `m-spindle-growth` | machine | `spindles[].thermal` | growth_um_at_max_rpm -> growth_at_max_speed_um | unit last: growth_um_at_max_rpm is growth_at_max_speed_um |
| `m-spindle-bearing-heat` | machine | `spindles[].thermal` | bearing_heat_W_at_max_rpm -> bearing_heat_at_max_speed_W | unit last: bearing_heat_W_at_max_rpm is bearing_heat_at_max_speed_W |
| `m-same-mechanics` | machine | `(file root)` | same_mechanics_as -> extends | same_mechanics_as: <id> is extends: <id> (the file's own keys override) |
| `m-rotation-90` | machine | `axes.*` | rotation_90deg_s -> index_90deg_s | rotation_90deg_s is the 90 deg index time index_90deg_s |
| `m-drive-power` | machine | `spindles[]` | drive_kW -> drive_power_kW | drive (text) and drive_kW share a stem: the rating is drive_power_kW |
| `m-guide-bush-switch` | machine | `guide_bushing` | switchable_to_non_guide_bush -> switchable | switchable_to_non_guide_bush is switchable |
| `m-axis-text` | machine | `axes.*` | description, travel_note, implementation, indexing -> notes | descriptive text kept in notes |
| `m-spindle-text` | machine | `spindles[]` | optional, max_torque_published, max_rpm_option, duty, tabulated, motor_kW -> notes | descriptive or duplicated catalogue data (the curves hold the ratings) kept in notes |
| `m-probe-alt-trigger` | machine | `probes.*` | trigger_force_high_N -> notes | an alternative trigger-force setting (the selected one is trigger_force_N) kept in notes |
| `m-probe-extra` | machine | `probes` | merge extra | <probe>_extra merged into <probe> with the source it was written with |
| `m-calibration-text` | machine | `probes` | tool text note | a calibration ball given only as text kept in notes |
| `m-coolant-text` | machine | `coolant` | mist_separator, pcool -> notes | descriptive text kept in notes |
| `m-chiller-text` | machine | `coolant` | chiller text | a chiller given as {value: text} is {kind: text} |
| `m-machine-air` | machine | `coolant` | machine air | the machine's air consumption is power.air_consumption_l_min |
| `m-nozzle-geometry` | machine | `coolant.nozzles[]` | nozzle geometry | nozzle geometry written on the nozzle (position_mm, diameter_mm) and its programmable sweep (step_deg, home_aim) |
| `m-thermal-text` | machine | `thermal` | mrzp_drift -> notes | descriptive text kept in notes |
| `m-envelope-text` | machine | `workholding_interface` | envelope_note -> notes | descriptive text kept in notes |
| `m-tooling-text` | machine | `(file root)` | tooling -> notes | tool-post capacity summary (not modelled: the project's tool list holds the tools) kept in notes |
| `m-chains` | machine | `kinematics` | kinematic chains | workpiece_chain_<name> is workpiece_chains.<name>, <name>_tool_chain is tool_chains.<name> |
| `m-rotary-home` | machine | `kinematics` | rotary homes | rotary axis homes belong in home_deg |
| `m-positioning` | machine | `axes.*` | positioning flat | positioning_<x> is positioning: {<x>} |
| `m-two-path` | machine | `controller` | two path | two_path: true is paths: 2 |
| `m-tcpc-dwo` | machine | `controller` | tcpc dwo | tcpc_dwo is five_axis: {tcpc, dwo} |
| `mat-profile` | material | `residual_stress[]` | profile -> profile_MPa | v1 profile was [[z/thickness, MPa]]: the unit goes in the key |
| `mat-kienzle-ref` | material | `cutting.kienzle[]` | kienzle refs | kc11 is by definition the specific force at h = 1 mm and b = 1 mm: the v1 entries carry h_ref_mm: 1, b_ref_mm: 1 implicitly |
| `mat-taylor-ref` | material | `tool_life[]` | taylor ref | v1 Taylor constants were for T in minutes: T_ref_min: 1 |
| `mat-usui-in-tool-life` | material | `(file root)` | usui in tool life | Usui constants under a tool_life entry go to tool_life_usui |
| `mat-usui-plain-a` | material | `tool_life_usui[]` | usui plain a | v1 Usui A was SI (dW/dt in m/s, sigma_n in Pa, V_s in m/s): A is A_per_Pa with that rate unit |
| `tm-usui-plain-a` | tool_material | `usui[]` | usui plain a | v1 Usui A was SI (dW/dt in m/s, sigma_n in Pa, V_s in m/s): A is A_per_Pa with that rate unit |
| `tm-thermal-shock-dk` | tool_material | `(file root)` | thermal_shock_deltaT_degC -> thermal_shock_dK | temperature differences are _dK (a difference in degC equals one in K) |
| `co-text` | coating | `(file root)` | process_detail -> notes | descriptive text kept in notes |
| `cl-noack-max` | coolant | `(file root)` | noack_loss_pct_max -> max_noack_loss_pct | minimum/maximum are prefixes |
| `cl-air-min` | coolant | `(file root)` | air_pressure_bar_min -> min_air_pressure_bar | minimum is a prefix (_min at the end means minutes) |
| `h-series-text` | holder_series | `(file root)` | collet, dimension_key, feature -> notes | series-level description kept in notes |
| `h-axial-float` | holder_series | `items[]` | axial_compensation_mm -> axial_float_mm | axial compensation travel of a tapping holder is axial_float_mm |
| `h-coolant-pressure` | holder_series | `items[]` | coolant_bar -> max_coolant_pressure_bar | the coolant pressure a holder takes is max_coolant_pressure_bar |
| `h-drawing-lengths` | holder_series | `items[]` | holder letter lengths | catalogue drawing lengths length_<X>_mm are dimensions_mm.<X> |
| `w-series-text` | workholding_series | `(file root)` | basis, dimension_key, friction_example, speed_rule, system, usage, force_vs_torque -> notes | series-level description kept in notes |
| `w-series-defaults` | workholding_series | `(file root)` | series defaults | series-level item values are defaults every item inherits |
| `w-centrifugal-general` | workholding_series | `(file root)` | centrifugal general | the series' centrifugal-loss coefficient is an item default |
| `w-min-abs-pressure` | workholding_series | `items[]` | pressure_kPa_abs_min -> min_abs_pressure_kPa | minimum is a prefix: min_abs_pressure_kPa |
| `w-max-holding-pressure` | workholding_series | `items[]` | holding_pressure_max_N_cm2 -> max_holding_pressure_N_cm2 | maximum is a prefix |
| `w-inertia` | workholding_series | `items[]` | moment_of_inertia_kg_m2 -> moment_of_inertia_kgm2 | kg_m2 reads as kg/m2 (as kg_m3 is kg/m3): a moment of inertia is kgm2 |
| `w-max-torque` | workholding_series | `items[]` | max_torque_Nm -> max_input_torque_Nm | a clamp's maximum screw torque is max_input_torque_Nm |
| `w-centrifugal-method` | workholding_series | `items[].centrifugal_loss_model` | type -> method | type is method |
| `w-gripping-curve` | workholding_series | `items[].centrifugal_loss_model` | gripping_force_curve -> gripping_force_curve_kN | unit last: [[rpm, kN]] is gripping_force_curve_kN |
| `w-body-letters` | workholding_series | `items[]` | body letters | catalogue drawing letters dropped from body_mm keys |
| `w-soft-jaw-dims` | workholding_series | `items[]` | soft jaw dims | the dimensions of a soft jaw are jaw_length/width/height_mm |
| `w-holding-curve` | workholding_series | `items[]` | holding pressure curve | holding pressure over part thickness is the table holding_pressure_N_cm2 with x: mm |
| `w-roller-force` | workholding_series | `items[]` | roller force at pressure | a roller force at a stated pressure is a point of clamp_force_per_roller_vs_pressure_kN |
| `range-window` | any | `any [lo, hi] field` | {range: w} -> {value: w} | a window written as range: without value: is the window itself |
| `text-value` | any | `any text field` | {text: t} -> {value: t} | a sourced text is {value: .., src: ..} |
| `text-in-number` | any | `any numeric field` | text -> notes (+ warning) | cannot be migrated mechanically |
| `note-text` | any | `any note/notes` | {value: t, src: s} -> "t (src s)" | a note is text; its source is named in the text |

State of the research files at the time of writing (2026-11-03; `twin db check` shows the current
one): every file under `db/` and `validation/cases/` loads with no error. Warnings that need the
authors: `dmg-dmu50-3rd.yaml` and `dmg-dmu50-3rd-siemens.yaml` `workholding_interface.clamping_area_mm`
is the text "diameter 630 x 500" (not migrated, not used); `dmg-ntx1000` and `star-sr20riv` name the
controller `fanuc-31i-b`, which has no file; `haas-st20y` spindle `sub` has no gear ranges; four
bibliography keys are described differently by two files (`iso-6983-1-2009`,
`jaeger-1942-moving-sources`, `carslaw-jaeger-1959-conduction`, `blok-1937-flash-temperature`);
about 690 `src: est` declarations have no note (errors with `--strict`).

## Project file

(owner: `cnctwin/project`; template: `db/_templates/project.yaml`)

### Format (version 1)

`project.yaml` (format `cnctwin-project`, version 1) follows the template; every key is declared
with its unit in `cnctwin/project/schema.py` (same rules as the databases: unit last, unknown keys
and wrong units are errors, `note`/`notes` free). Required: `format`, `version`, `name`, `machine`,
`material`, `stock`, `tools`, `coolant`, `operations`.

* `machine`, `material.id`, `tools.list[].tool`, `.holder`, `coolant.fluid`,
  `operations[].setup.workholding[].device` are **database references**: an id
  (`haas-vf2ss`), a path to a database file relative to the project (`machines/my-vmc.yaml`;
  any reference containing `/` or ending in `.yaml`; the file is checked as a database file:
  format, sources, migration, schema, `extends`), or **an inline object in place of the id**,
  checked with the inline schema of its kind (no header, `id` optional, `extends: <id>` starts from
  a database or tool-library object). Inline values need no sources; their SI view reports
  `src: project`.
* `controller_overrides`: any keys of the machine's `controller` section, deep-merged over it (the
  merged machine is checked).
* `material.batch` (`kc_scale_sigma`, `hardness_sigma_pct`, `flow_stress_sigma_pct`,
  `conductivity_sigma_pct`): 1-sigma batch variation for uncertainty runs.
* `environment` (`ambient_degC`, `coolant_degC`, `machine_state`: cold | warm |
  `{spindle_degC, axes_degC, structure_degC}`, `time_between_ops_min`); `stock` (`kind` block |
  cylinder | tube | mesh, `size_mm` with 3 / 2 / 3 values, `tolerance_mm`, `non_squareness_deg`,
  `form`, `mesh`); `cad` (`path`, `units`, `placement` pose, `datums`, `tolerances`: distance,
  flatness, position, roughness, diameter ... with `feature` selectors `{face, normal, axis,
  near_mm}`).
* `tools`: `library` (a CSV tool list or an ISO 13399 file/directory, below) and `list` of
  `{number, tool, holder, path, spindle, turret, station, stickout_mm, length_offset_mm,
  radius_offset_mm, wear_offsets_mm: {length, radius}, runout_um, runout_phase_deg, tilt_urad,
  life: {group, limit_min, used_min, limit_parts, sister_of}, initial_wear: {VB_mm, KT_um,
  edge_radius_um}, comment}`; a tool number is unique per controller path.
* `coolant` (`fluid`, `concentration_pct`, `tank_fill_fraction`); `operations[]` (`name` unique,
  `program` or `programs: {<path>: file}`, `setup: {workholding[], stock_pose, work_offsets_mm,
  fixture_pose}`, `from_previous: {op, flip, transfer}` naming an earlier operation,
  `nozzle_positions: {<T>: <position>}` for listed tools); `production`, `recording`, `outputs`,
  `uncertainty`, `machine_data`.
* `physics`: `tick_s` (fixed at 1e-6 s), `max_deg_per_step` (SI view `max_rad_per_step`),
  `steps_per_tooth`, `macro_ipo`, `lag_macro` (1 or 2), `lattice_mm`, `refine[]`, `edge_element_mm`,
  `memory_bins`, `polar_theta_bins`, `particles_per_tooth`, `tol_mass`, `tol_energy`; one section
  per domain (`controller motion tooling removal cutting chips thermal coolant wear holding
  structure dynamics surface`) with `every_n_macro` (architecture 5.1). A domain declares its
  other settings as `PHYSICS_SCHEMA` (a `cnctwin.db.schema.Struct`) in `cnctwin/<domain>/schema.py`
  and is then checked strictly; without one its section is checked generically (identifier keys;
  numbers, text, booleans, lists, mappings; SI by key suffix in the view). Settings a project does
  not write come from the template's `physics` section (the documented defaults: 0.5 deg per
  step, 96 steps per tooth, 10 L1 ticks per macro step ...).
* `physics.resolved` (written by `twin check --write`, read by a run): the recorded L0 strides
  `{ipo_ticks, macro_ipo, operations: {<op>: [{tool, gear, path, l0_ticks, max_rpm, f_max_Hz,
  teeth, rule}]}}`: one entry per tool / spindle-range segment; `l0_ticks` divides `ipo_ticks`;
  every operation named exists.

All file paths are relative to the project file and must exist (`program`, `programs`,
`cad.path`, `stock.mesh`, `tools.library`). Every problem names `file:line` and the key path;
`load_project` and `resolve` raise one `InputError` listing every error.

### Resolution (`cnctwin.project.resolve(path, db)` -> `ResolvedProject`)

The SI view of the project file (a `SIMap`) with the database objects in place of their ids:
`project.machine` (with `controller_overrides`), `project.material` (the material, plus the
project's `batch` and `db_id`), `project.tools.list[i].tool` / `.holder`, `project.coolant.fluid`,
`project.operations[i].setup.workholding[j].device`, and per operation `program_text`,
`program_path` (absolute), `program_texts` (per path), `stock_pose_W_S` and `fixture_pose_F_H`
(4x4); `cad.cad_pose_S_C`. File paths in the view are absolute (`.raw` as written).
`project.stock_pose_W_S` is the first operation's (findings use it). Frames: the workpiece frame
W coincides with the workholding frame at setup (`part_pose` is the identity until a part slips),
so `stock_pose_W_S` is the setup's `stock_pose` (S to W) and `fixture_pose_F_H` its `fixture_pose`.

A used holder must give `gauge_length_mm`, `nose_diameter_mm` and `profile_mm`; when the database
item has them as `null` (not published) the project is refused with the inline form to add them
(`holder: {extends: <id>, profile_mm: [...]}`).

`project.input_hashes` is the sha256 of every input file (`project.yaml`, programs, CAD, stock
mesh, tool library files, and the database files of every object used: their `extends` chains,
tool materials, coatings, controllers); `project.input_files` their paths.
`project_hash(project)` is the sha256 of the canonical JSON of the migrated project data (comments
and formatting do not count), the hashes of the other input files and of every `db/models/*.yaml`.

### Version 0 (migrated forward)

A file without `version` is version 0: one operation written flat at the top (`program`, `setup`,
`from_previous`, `nozzle_positions`, optional `operation_name`, default `OP10`), `tool_list` and
`tool_library` beside each other, `material` and `coolant` as bare ids:

```yaml
name: "old flat project"
machine: haas-vf2ss
material: al6061-t6            # v1: material: {id: al6061-t6}
coolant: emulsion-8pct         # v1: coolant: {fluid: emulsion-8pct}
tool_list: [...]               # v1: tools: {list: [...]}
tool_library: tools.csv        # v1: tools.library
program: op10.nc               # v1: operations: [{name: OP10, program: op10.nc, setup: ...}]
setup: {...}
```

`cnctwin.project.migrate(raw)` returns the version 1 form; the loader also moves the line numbers,
so problems still name the line of the version 0 file. A version newer than this build is refused.

### Tool libraries

**CSV** (`tools.library: tools.csv`; UTF-8, comma separated, one header row, `#` comment lines):
each row is a `tools.list` entry; empty cells are "not given"; an unknown column, a bad cell or a
row without `number` is an error with its CSV line. List columns: `number` (required), `tool`,
`holder`, `path`, `spindle`, `turret`, `station`, `stickout_mm`, `length_offset_mm`,
`radius_offset_mm`, `wear_length_mm`, `wear_radius_mm`, `runout_um`, `runout_phase_deg`,
`tilt_urad`, `life_group`, `life_limit_min`, `life_used_min`, `life_limit_parts`, `sister_of`,
`initial_VB_mm`, `comment`. Tool columns (an inline tool: with `tool` empty a complete item, with
`tool` given an `extends` of it): `type`, `hand`, `tool_material`, `tool_material_class`, `grade`,
`coating`, `coating_thickness_um`, `order_code`, `maker`, `diameter_mm`, `flutes`, `helix_deg`,
`rake_deg`, `corner_radius_mm`, `cutting_length_mm`, `flute_length_mm`, `overall_length_mm`,
`shank_diameter_mm`, `shank_length_mm`, `neck_diameter_mm`, `neck_length_mm`, `edge_radius_um`,
`point_angle_deg`, `tip_angle_deg`, `thread_pitch_mm`, `centre_cutting`, `max_rpm`, `mass_g`.
Rows join `tools.list` (a number already listed is an error).

**ISO 13399** (`tools.library: iso13399/` or one `.p21`/`.stp`/`.step` file): ISO 10303-21 exchange
files (`cnctwin.project.iso13399.parse_p21`: strings with `\X2\..\X0\` and `\X\hh` encodings,
numbers, enumerations, references, `$`, lists, typed measures, complex instances). Property
symbols are found in three encodings: an instance whose first string is the symbol followed by a
number or measure (`PROPERTY_VALUE('DC',LENGTH_MEASURE(10.),#unit)`), the symbol with
description, number and unit text (`PROPERTY('OAL','overall length',72.,'mm')`), or an ISO 10303
property chain (`PROPERTY_DEFINITION('GAMF',..)` <- `PROPERTY_DEFINITION_REPRESENTATION` ->
`REPRESENTATION` -> `MEASURE_REPRESENTATION_ITEM`). Units: a unit text, else the unit instance the
value references (SI units by prefix, conversion-based units such as INCH or DEGREE through their
measure-with-unit), else the `GLOBAL_UNIT_ASSIGNED_CONTEXT`, else mm and degrees. One file is one
item: id = the `PRODUCT` id lower-cased to the id form (else the file name), `order_code` the
`PRODUCT` id, `type` from the product name/description words (ball, torus/corner radius -> by
RE against DC/2, chamfer, spot, drill, tap, thread mill, reamer, face mill, insert ...).

| symbol | field | | symbol | field |
|---|---|---|---|---|
| DC (DCX if no DC) | `geometry.diameter_mm` | | ZEFP (NOF) | `geometry.flutes` (indexable: `inserts.count`) |
| DCN | `geometry.tip_diameter_mm` | | RE | `geometry.corner_radius_mm` (inserts: `nose_radius_mm`) |
| APMX | `geometry.cutting_length_mm` | | CHW | `geometry.corner_chamfer_mm` |
| LCF | `geometry.flute_length_mm` | | KAPR (90 - PSIR) | `geometry.inserts.lead_angle_deg` (turning: `holder.approach_angle_deg`) |
| OAL (LF if no OAL) | `geometry.overall_length_mm` | | GAMF | `geometry.rake_deg` |
| LU (with DN) | `geometry.neck.length_mm` | | GAMP | `geometry.inserts.axial_rake_deg` |
| DN | `geometry.neck.diameter_mm` | | FHA | `geometry.helix_deg` |
| DMM | `geometry.shank.diameter_mm` | | SIG | `geometry.point_angle_deg` |
| LS | `geometry.shank.length_mm` | | TP | `geometry.thread.pitch_mm` |
| CW | `geometry.width_mm` (grooving: `insert.width_mm`) | | IC | `geometry.insert.ic_mm` |
| S | `geometry.insert.thickness_mm` | | AN | `geometry.clearance_deg` (inserts: `insert.clearance_deg`) |
| EPSR | `geometry.insert.included_angle_deg` | | WT | `mass_g` (kg unless a unit is given) |
| RPMX | `max_rpm` | | BMC | `tool_material.class` (HW/HC/HF carbide, HT cermet, HS/HSS hss, C* ceramic, B* cbn, D* pcd) |
| HAND | `hand` (R, L, N) | | | |

Symbols not in the table are listed in the item's `notes`. ISO 13399 does not name this
database's tool materials or coatings: a list entry adds them inline
(`tool: {extends: <item id>, tool_material: {class, id}, coating: {id}}`), and the item is then
checked like any tool.

### `twin new` and `twin check --write`

`cnctwin.project.new_project(argv)` (`twin new <dir> --machine <id> [--material <id>]
[--coolant <id>] [--name ..] [--db DIR] [--force]`) writes `project.yaml` and `op10.<ext>` (the
shortest complete program of the machine's dialect: start, metric/absolute, end) and resolves
it; exit 0, 2 if the project exists (without `--force`), 1 on an error. Environment, stock,
production, recording, outputs and `physics` come from the template (architecture 5.1); the
material and coolant default to the template's (`al6061-t6`, `emulsion-8pct`) when present, else
the first valid id; the tool list and workholding are empty.

`cnctwin.project.write_resolved_physics(project, resolved)` stores `physics.resolved`: the block is
checked (schema, `l0_ticks` divides `ipo_ticks`, operations exist), only the `resolved` entry of
`physics` is replaced (added after the last `physics` key, or a `physics` section added at the
end, when absent), every other line stays as written (comments included); the edited text is
parsed again and compared with the expected data before it replaces the file (temporary file,
fsync, rename).

## Run directory

(written by the owner of `cnctwin/core/rundir.py`, `recorder.py`, `keyframes.py`, `checkpoint.py`)

## frames.bin

(written by the owner of `cnctwin/core/rundir.py`)

## Keyframes

(written by the owner of `cnctwin/core/keyframes.py`)
