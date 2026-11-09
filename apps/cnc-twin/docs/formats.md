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
  envelope. A spindle without `gear_ranges` is a warning (no power/torque limits). A spindle's
  thermal model (docs/models/thermal.md 4.1) is built from `drive_type` (built_in | coupled | belt |
  gear; else read from the `drive` text), `bearings: [{position: front | rear | middle, type:
  angular_contact | cylindrical_roller | tapered_roller | deep_groove | thrust_ball, count, bore_mm,
  outer_mm, contact_angle_deg, preload_N, static_load_rating_kN, lubrication: grease | oil_mist | oil_air |
  oil_bath | oil_jet, oil_viscosity_40C_mm2_s, oil_viscosity_100C_mm2_s, mass_kg, axially_fixed,
  designation}]` and `thermal: {shaft_diameter_mm, shaft_bore_mm, shaft_length_mm, bearing_span_mm,
  nose_diameter_mm, nose_to_fixed_bearing_mm, housing_mass_kg, housing_size_mm: [w, d, h],
  housing_growth_length_mm, housing_material, motor_no_load_loss_pct, motor_heat_to_head_pct, jacket:
  {flow_l_min, channel_diameter_mm, channels, area_cm2, inlet_degC | inlet_offset_dK}, model: network |
  lumped}`; absent keys take the model's defaults (db/models/thermal-machine.yaml, reported as estimated).
  The lumped keys (`growth_at_max_speed_um`, `time_constant_min`, `bearing_heat_at_max_speed_W`) select
  the first-order body when they are the only thermal data or `model: lumped` says so.
* **material**: `flow_stress.model` is `johnson_cook` (required for metals), `temperature_table`
  (polymers: `tensile_strength_MPa` table, `compressive_yield_MPa`, `T_soft_degC`) or `none`
  (no constitutive model: optional `yield_MPa`, `compressive_flow_MPa`, `shear_yield_MPa`,
  `T_soft_degC`, `T_melt_degC`). Kienzle entries carry `h_ref_mm` and `b_ref_mm`, Taylor entries
  `T_ref_min` and `life_basis`: what T counts, `cutting_time` (the tool feeding in the material: ISO 3685,
  ISO 8688 and catalogue lives; the default for milling, drilling, tapping, `any` and entries without
  a process) or `contact_time` (the edge in contact; the default for turning, boring, grooving,
  threading and orthogonal cuts, where it equals the cutting time). A milling cutting-time entry may
  give `engaged_arc_ref_deg`, the arc of each revolution the edge is in the material in its reference
  cut (side milling arccos(1 - 2 ae/D), slot 180); without it the ISO 8688-2 end-milling test arc
  (ae/D = 0.1, 36.87 deg) applies (docs/models/wear.md section 3). Usui constants (`tool_life_usui[]`, tool material `usui[]`) give exactly one of
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
  data. A setup input the source does not state is `{value: .., assumed: true, note: "how it was
  chosen"}` (a missing note is a warning); `{value: .., src: est, note: ..}` and a section's
  `assumed_inputs: [keys]` list mark assumed inputs too (cnctwin.validation.cases).

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

State of the databases (2026-11-03, after the cleanup; `twin db check` shows the current one): every
file under `db/` and `validation/cases/` loads with no error and no warning, also with `--strict`
(test `tests/unit/db/test_real_db.py::test_every_folder_is_clean_in_strict_mode`): every `src: est`
has a note saying how its value was derived (where a derivation could not be reconstructed the value
was re-derived from physics or a cited source and the note says `RE-DERIVED`), each bibliography key is
defined once (a file that cites a key defined elsewhere says where in its header), the DMU 50
`clamping_area_mm` is `[630, 500]` (diameter, width across the flats), the haas-st20y sub spindle has
an estimated gear range, and every coated catalogue tool states its coating thickness. The machines
that name the controller `fanuc-31i-b` resolve to db/controllers/fanuc-31i-b.yaml.

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
  radius_offset_mm, x_offset_mm, z_offset_mm, nose_radius_mm, tip, wear_offsets_mm: {length, radius,
  x, z}, runout_um, runout_phase_deg, tilt_urad, life: {group, limit_min, used_min, limit_parts,
  sister_of}, initial_wear: {VB_mm, KT_um, edge_radius_um}, comment}`; a tool number is unique per
  controller path. Lathe geometry offsets are written as set on the control's offset page:
  `x_offset_mm` (a diameter on a diameter-programmed lathe), `z_offset_mm`, `nose_radius_mm`, `tip`
  (imaginary tip number 0-9) and the wear `x`, `z`, `radius`.
  **Probes**: an entry with `kind: probe` is a spindle probe the program calls by its T number:
  `{number, kind: probe, probe, path, spindle, length_offset_mm, radius_offset_mm, wear_offsets_mm,
  comment}`, where `probe` names a probe of the machine's `probes` section (`spindle_probe`) or is
  an inline probe `{extends: <machine probe>, model, stylus_ball_mm, stylus_length_mm,
  stylus_diameter_mm, body: {diameter_mm, length_mm}, pretravel_um, repeatability_um,
  trigger_force_N, overtravel_z_mm}` (merged over the machine's probe; the stylus ball is required,
  from either). `length_offset_mm` reaches the ball's bottom (null: the program sets it),
  `radius_offset_mm` is the effective stylus radius the probing macros read (default: half the
  ball). A probe is not a cutting tool: it is not in the tool table (resolution below).
* `coolant` (`fluid`, `concentration_pct`, `tank_fill_fraction`); `operations[]` (`name` unique,
  `program` or `programs: {<path>: file}`, `setup: {workholding[], stock_pose, work_offsets_mm,
  fixture_pose}`, `from_previous: {op, flip, transfer}` naming an earlier operation,
  `nozzle_positions: {<T>: <position>}` for listed tools); `production`, `recording`, `outputs`,
  `uncertainty`, `machine_data`.
* `setup.work_offsets_mm`: `{<code>: [x, y, z, ...]}`, the work zero's machine coordinates as set on the
  machine, in the order X, Y, Z, then the path's other axes in the control's order, on every machine type
  (lathes included: `[x, 0, z]`); the controller and sanity read the same values.
* `setup.workholding[]`: `{device, pose, jaw_opening_mm, input_torque_Nm, grip_depth_mm,
  parallels_mm, pressure_bar, outline_mm, seat_height_mm, jaws, label}`. The device's frame D is its
  `pose` in the workholding frame; a vise's D has its origin on the seat (the parallels' top, else the
  vise bed: where the part's bottom rests), x along the jaw width, y the clamping direction, and
  `grip_depth_mm` is the jaw top above the seat. For devices whose database item gives no body:
  `outline_mm: [[x0, y0, z0], [x1, y1, z1]]` is the device's box in D (a strap clamp, a fixture),
  `seat_height_mm` the height of a vise's seat above its mounting face (the vise body below the
  seat).
* **`from_previous` and `stock_pose`** (one convention for every consumer): `from_previous` carries
  the part of the named earlier operation, as it left it, into this one. `flip: {axis, angle_deg}`
  turns it about that axis of the previous operation's stock frame S, `transfer: sub_spindle` half a
  turn about S y (gripped from the other end); `main_spindle`, `pallet`, `manual` (and no
  `from_previous`) keep its orientation. The turned stock box is placed by the stock conventions
  (a block's box starts at the S origin; a cylinder or tube whose axis stays on z keeps the axis on
  S z with z from 0): that is this operation's stock frame S, the frame the removal field carries
  the material in. `stock_pose` is then **only the extra placement**: it places this S in the
  workholding frame exactly as it would place a fresh blank, and must not repeat the turn. The
  resolver refuses a `stock_pose` whose rotation repeats or undoes the turn (the part would be
  turned twice, or not at all) and names the placement that was probably meant. Example
  (examples/two-setup-flip): an 80 x 60 x 25 blank, OP20 `flip: {axis: X, angle_deg: 180}`: OP20's
  S is (x, 60 - y, 25 - z) of the OP10 frame; `stock_pose: {translate_mm: [-40, -30, -2.5]}` centres
  it in the vise with the OP10 top (S z 2.5 after 2.5 mm were faced off) on the jaw steps.
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
so `stock_pose_W_S` is the setup's `stock_pose` (this operation's S to W) and `fixture_pose_F_H`
its `fixture_pose`. Per operation the resolver also gives the stock frame of `from_previous`
(above): `transfer_S_Sprev` (4x4, this S from the previous operation's S; the identity without a
turn), `pose_S_S0` (from the first operation's S, the frame `cad.placement` is written in: the CAD
in this S is `pose_S_S0 * cad_pose_S_C`), `stock_pose_W_S0 = stock_pose_W_S * pose_S_S0` (a point of
the first operation's stock frame in this W) and `stock_box_S_m` (`[[lo], [hi]]`, the stock box in
this S; a mesh stock's bounds are read only when a turn needs them).

`project.tools.list` holds the cutting tools only, in file order: its index is the tool-table slot
every domain uses. The `kind: probe` entries are `project.tools.probes` (each with `list_index`, its
place in the file's list, and `probe`: the machine's probe merged with the inline keys, in SI, with
`name` the machine probe it extends or null); the controller knows their T numbers (a tool change to
a probe carries `tool.probe`, a skip move `probe.stylus`).

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

(owner: `cnctwin/core/rundir.py`, `recorder.py`, `keyframes.py`, `checkpoint.py`, `ledger.py`; architecture 16)

```
runs/<run_id>/
  run.json            written at the end (complete, no volatile fields)    rundir.run_document
  status.json         volatile: status, progress, ETA                      the progress callback's owner (API worker, twin run)
  findings.json       rewritten when findings change                        RunDirectory.write_findings
  trace.jsonl         append-only, one TraceEvent per line, id = line       cnctwin.core.trace.TraceLog
  channels.json       written at the start                                  Recorder.write_channels
  timeline.json       rewritten (closed, final blocks)                      Timeline.write
  series/<tier>/part-<n>.parquet, series/index.jsonl                        Recorder
  scene.json, scene/<name>.glb                                              SceneWriter
  frames.bin          append-only, one record per L1 tick                   FramesWriter ("frames.bin" below)
  keyframes/..., chips/kf_<k>.npz                                           KeyframeWriter ("Keyframes" below)
  checkpoints/ck_<tick>.npz + ck_<tick>.json                                Checkpointer
  part_<op>.*, inspection.json, ... (operation ends, end of run)            registered with RunDirectory.register_output
```

**Run ids** (`run_id(project_hash, options, counter, slug)`): `<nnnnn>-<slug>-<h8>`, the counter as five
digits (one above the highest numbered folder, as the API numbers its runs), the project folder's slug,
and the first 8 hex digits of sha256(`<project hash>\n<options as canonical JSON>`) (sorted keys, no
spaces, `resume_from` left out: a resumed run is the same run). The same project, options and counter
always give the same id. `allocate_run_dir(runs_dir, project_hash, options, slug)` creates the folder.

**run.json** (`cnctwin-run`, version 1), written once at the end (`fsync`, rename); a resume removes it:
`run_id`, `complete`, `project` (the ResolvedProject as plain SI data, `operations[].program_sha256`
added), `project_hash`, `project_dir`, `project_file`, `inputs` ({file: {sha256, bytes}}), `options`
(without `resume_from`), `versions` (cnctwin, format, python, numpy, scipy, numba, llvmlite, pyarrow,
`build`), `numba_cache_key`, `threads` (OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS, PYTHONHASHSEED), `platform`
(machine, system, CPU model), `clock` (tick_s, ipo_ticks, macro_ipo, l0 strides; whatever the loop
passes), `seed`, `verdict` {safe, reasons[]} (`verdict(findings)`: safe when no finding is error or
fatal; a reason reads `<category> (<severity>) at <op> line <n>: <message>`), `summary` (the loop's
numbers; `summary.conservation` is `Ledger.summary()`), `outputs` [{path, bytes, sha256}] of every file
except run.json, status.json, checkpoints/ and dot files. JSON in insertion order, indent 1, UTF-8;
non-finite floats are written as null (JSON has no NaN).

**status.json** (`cnctwin-run-status`): `{status, progress {sim_time_s, est_total_s, wall_s, eta_s}, op,
phase, message, checkpoint {tick, file}, error, ...}`, status one of queued, running, cancelling, done,
failed, cancelled; written atomically (no fsync). Everything that depends on wall time is here and
only here. It is written by the owner of the progress callback: the API worker for jobs, `twin run`
(cnctwin.cli.Progress) for command-line runs; the loop reports through `progress_cb` only and never
writes it (`RunDirectory(path, own_status=False)` makes `write_status` a no-op and `prepare` keep the
file, for a loop that holds a RunDirectory under the worker).

**findings.json** (`cnctwin-findings`): `{findings: [Finding.to_dict()]}` in id order.

**channels.json** (`cnctwin-channels`): `{channels: [{name, unit, tier, domain, doc, max_shape, dtype,
storage}], fast: {mode, ring_rows, ring_s, l0_ticks_min, columns, window_revs, period_s, min_spacing_s,
max_windows_per_second, window_min_s, window_max_s}, tick_s, ipo_ticks, macro_ipo}`.

**timeline.json** (`cnctwin-timeline`): `{tick_s, blocks: [...]}`, one record per executed NC block of a
path, `[tick0, tick1)` in base ticks: `index` (position in the run's timeline, global over paths, the
`block_index` of `PathFrame`), `op, program, line, block, path, kind, tick0, tick1, part`, optional
`tool, ticks {category: ticks}, n_number, call_stack, stop, pumps_on, in_cut_ticks` (and any other key
its writer adds; the format cnctwin.production reads). Only closed records whose `final` is not false
are written (open records have `tick1 = -1`). `Timeline.open(...) -> index`, `.open_ref(LineRef, ...)`,
`.close(index, tick1, ticks=...)`, `.upsert(record)` (motion's own records), `.line_ref(index) ->
BlockRef(index, op, program, line, block, path, kind, tick0, n_number, call_stack)` for
`RunContext.finding`. Category ticks are non-negative integers; the ledger fails the run when a
block books more category ticks than it lasts, or when a path's block durations do not sum to its span.

**series/** (Parquet, zstd, no dictionary encoding, statistics on): one file per tier per checkpoint
interval, `series/<tier>/part-<n>.parquet` (n = 0, 1, ... per tier), written row group by row group into
`series/<tier>/.part-<n>.parquet.tmp` (a row group whenever a tier has buffered `series_buffer_max_bytes`,
4 MiB, of rows) and renamed into place when closed at each checkpoint and at the end; a part's
key-value metadata `cnctwin` = `{format: cnctwin-series, version, tier, part, tick_s}`. Columns: `tick`
(int64 base ticks, not null), `time_s` (float64, `tick / 10^6`), then every channel of the tier sorted
by name, each as `<name>` (scalar) or `<name>[i]` (C order over `max_shape`): float64/float32 as declared
(the fast tier in mode `all`: float32), int64/int32/int8 as declared, bool as int8 0/1. Padding beyond a
value's shape and values not given in a step are NaN (float columns) or null (integer columns). The
tick tier ends with `<fast column>.min`, `.mean`, `.max` (float64) of every fast column over the L0 rows
of each L1 tick `[t, t + ipo_ticks)` (finite values only; NaN where a tick had none). Rows: fast = L0
samples (in mode `windows` only those inside windows), tick = one per L1 tick (its first base tick),
macro = one per macro step (its end tick). `series/index.jsonl` (append-only, fsynced): one line per
closed part `{tier, part, file, tick0, tick1, t0_s, t1_s, rows}` (`replay: true` for a replay part).

Fast windows (`fast: windows`; settings in db/models/recording.yaml, project `recording:`): the ring
(`RingBuffer`, core/abi.py) holds `ring_buffer_s` (0.5 s) of L0 rows, fewer when `ring_max_bytes` (32 MiB)
would be exceeded, never fewer than one macro step plus one L1 tick. At each `end_step` the recorder
reads the rows pushed in the step (all of them: rows overwritten before being read fail the run) and
keeps the rows inside pending windows that come after the last row already written:

* engagement: a row more than one L1 tick after the previous one (the broad phase found no contact for
  a whole L1 tick) starts an engagement, or the loop calls `Recorder.engagement(tick, omega)`; window
  `[start, start + L)`, unless another engagement window started less than `fast_min_spacing_s` (0.25 s)
  before or the current whole second of the base clock already has `fast_max_windows_per_second` (2);
* finding: `[t_f - L/2, t_f + L/2)` around every finding raised (not capped; rows still in the ring);
* periodic: `[k P, k P + L)` for every `P = fast_period_s`;

with `L = fast_window_revs` revolutions at the step's largest |omega|, within `[fast_window_min_s,
fast_window_max_s]` (10 ms .. 0.25 s). Windows merge; a row is written once. A window whose start was
overwritten in the ring before it was asked for is counted (`lost_windows`); findings raised after
their rows left the ring (end-of-operation inspection) get their window by replay (`twin replay
--window T0 T1`: `Recorder(..., replay=(tick0, tick1))` writes every fast row of the window, float64,
to `series/fast/replay-<tick0>-<tick1>.parquet`, appended to the index; readers drop rows a replay
repeats). `fast: all` writes every row (float32), `fast: none` no fast tier (the tick tier still has
the envelopes).

Reading: `iter_series(run_dir, tier, columns, tick0, tick1)` yields `(ticks, {column: float64})` chunks in
time order (missing columns and nulls read NaN; the API's backend); `read_series(run_dir, tier, tokens,
t0_s, t1_s, max_points)` returns the rows as recorded up to `max_points`, else M4 (first, last, minimum,
maximum per bucket of `max_points / 4` buckets, every point a recorded sample; Jugel et al. 2014); a
token selects a column exactly or all columns of a channel (`name`, `name[i]`, `name.min`).
`twin export RUN [--tier tick] [--channels a,b] [--format csv|parquet] [--t0 S] [--t1 S] [--out FILE]`:
CSV `tick,time_s,<columns>` (header quoted, NaN as empty, shortest round-trip floats) or Parquet (zstd,
NaN as null).

**scene.json** (`cnctwin-scene`): `{units: "m", ...the loop's document (machine components and links,
chain, axes, fixture, stock with its brick layout, tool envelopes, nozzles), meshes {name:
"scene/<name>.glb"}, frames: <frames.bin header>, bricks: <keyframe brick layout>}`; each mesh is a
GLB written with cnctwin.geom's writer (`SceneWriter.add_mesh(name, mesh, color_rgba, extras,
quantize)`), names are file-safe ids.

**checkpoints/** (`Checkpointer`, architecture 9): `ck_<tick>.npz` is a zip of `__state__.json` (deflated)
and the arrays `a<n>.npy` (stored), written as `.ck_<tick>.npz.tmp`, fsynced, renamed; then
`ck_<tick>.json`, the commit marker, the same way, last. `__state__.json` = `{tree, arrays}`: the
encoded `{state: SimState, participants: {name: state}}` (generic codec, see cnctwin/core/checkpoint.py:
`{"$dc": "module:Class", "f": {...}}` dataclasses, `"$nt"` NamedTuples, `"$o"` objects, `"$cp"` states
with `to_arrays`/`from_arrays` (their arrays and meta), `"$d"` dicts with non-string keys, `"$t"` tuples,
`"$set"`, `"$dq"`, `"$e"` enums, `"$n"` NumPy scalars, `"$f"` non-finite floats by bit pattern, `"$a"`
arrays, `"#"`/`"$r"` shared objects) and per array index `{store: "a<n>"}` or `{view: "a<n>", offset,
shape, strides, dtype}` (a view of a stored base). The marker: `{format: cnctwin-checkpoint, version,
tick, time_s, macro, op, op_index, op_end, npz, npz_bytes, state_sha256, arrays, project_hash, build,
numba_cache_key, cnctwin, findings, trace {bytes, events}, cuts {participant: cut}, extra}` plus, for
readers, `trace_bytes`, `trace_events`, `frames_bytes`, `keyframes`, `timeline_blocks`, `series`
({tier: [closed files]}), `series_index_lines`, `series_index_bytes`. Cuts: frames `{bytes, records}`,
keyframes `{keyframes, index_bytes}`, recorder `{series, index_bytes, index_lines}`, timeline `{blocks,
closed}`, run directory `{outputs}`, ledger `{}`. Retention after each commit: the newest
`checkpoints_keep_last` (2) and every checkpoint written with `op_end=True`; a marker is removed before
its arrays. A checkpoint is refused (CheckpointError) when its project hash or build differs from the
run's. `resume` truncates every output to the cut (frames.bin and the index files to their byte
lengths, series parts and keyframes after the cut and temporary files removed, bricks.idx,
timeline.json and findings.json rewritten from the restored state, late outputs not registered before
the cut and run.json removed), reopens trace.jsonl at its cut (`Restored.trace`) and removes later
checkpoints. Arrays come back writable, native byte order, C-contiguous (views of a shared base stay
views; `Restored.converted_to_c` counts arrays that had another order).

**Participants** (what the loop passes to `Checkpointer.write(..., participants=...)` and `resume`):
objects with `checkpoint_cut() -> dict` (make outputs durable, return the cut), `checkpoint_state()`,
`restore_state(state)`, `truncate_outputs(cut)`: Recorder, FramesWriter, KeyframeWriter, Timeline,
RunDirectory, Ledger. The trace log is passed as `trace=ctx.trace`. Order of one macro step's
infrastructure calls (architecture 5.3): domains' `absorb` and `macro_step` write fresh records into
`bus_next`; findings rules; `Ledger.check(ctx, state, tick_end)`; `Recorder.record_tick`,
`.record_macro`, `.end_step(tick0, tick1, omega, state.findings)`; `FramesWriter.append_motion(
bus_next.motion)`; `Timeline.write()` and `RunDirectory.write_findings` when they changed;
`state.swap_bus()`, clock advanced; `KeyframeWriter.write` when `due`; `Checkpointer.write` when `due`
(the cut lies between two macro steps: a resume starts the next step at once).

**Conservation** (`cnctwin/core/ledger.py`, run.json `summary.conservation`, macro channels `ledger.*`;
architecture 12). `Ledger.check(ctx, state, tick_end)` runs every macro step after the slow domains. It
books the step's bus records into `SimState.ledger` (fields only the ledger writes: `removed_volume_m3`,
`removed_mass_kg`, `displaced_volume_m3`, `freed_body_mass_kg`, `shaft_work_J`, `feed_work_J`,
`heat_shear_J`, `heat_rake_J`, `heat_flank_J`, `ploughing_J`, `modal_energy_change_J`, `damping_J`,
`process_damping_J`, `chip_kinetic_J`, `block_time_ticks`, `cycle_time_ticks`) and checks, each term
computed by its owner, nothing closed by difference:

| balance | equation | read from | tolerance |
|---|---|---|---|
| mass (cumulative, kg) | `rho20 (V_removed + V_loose) = m_chips + rho20 V_displaced + m_loose` | `removal.contacts[*].removed_volume_m3, displaced_volume_m3, workpiece`; `removal.loose_bodies[*] {volume_m3, mass_kg, workpiece}` (bodies detached this step, not in `removed_volume_m3`); `m_chips` = `SimState.ledger.chips_in_flight_kg + chips_settled_kg + chips_conveyed_kg + chips_adhered_kg + chips_in_flutes_kg + dust_kg` (written by chips) valid at `chips.booked_until_tick`, the removal side taken at that step end; `rho20` = `project.material.physical.density_kg_m3` at 20 degC | `physics.tol_mass` (1e-9) relative to the larger side |
| energy (J, per interval) | `shaft + feed work = shear + rake + flank heat + ploughing + modal energy change + damping + process damping + chip kinetic + friction + stored cold work + unbalance work` | `cutting.contacts[*]` (`shaft_work_J, feed_work_J, heat_shear_J, heat_rake_J, heat_flank_J, ploughing_J`), `dynamics` (`modal_energy_change_J`, `damping_dissipation_J`, `process_damping_J`; `basis_change_J` is not work and is not booked), `chips.kinetic_energy_J` (booked once per new `booked_until_tick`), `SimState.ledger.friction_J, stored_cold_work_J, unbalance_work_J` (cumulative, written by their owners in the step whose window they cover) | `physics.tol_energy` (1e-6) relative to the sum of the magnitudes of the interval's terms |
| time (integer ticks) | per path: sum of block durations (open block up to now) = span (first tick0 to now, or to the last tick1 when no block is open); category ticks of a block <= its duration | `ctx.timeline` (Timeline records, or any `records` with path, tick0, tick1, ticks) | exact |
| thermal solvers | `dU = sources - losses - removed enthalpy` per solver | `thermal.ledger_residual_rel`; `Ledger.solver(name).book(u_before_J, u_after_J, sources_J, losses_J, removed_J, time_s)` | `physics.tol_energy` |
| heat partition (reported only) | heat generated by zone up to step n-1 = `heat_to_chip_J + heat_to_tool_J + heat_to_work_J` (thermal, step n) | cutting, `SimState.ledger` | not enforced |

Alignment: the hot-loop terms cover the step that wrote them; chips books its windows up to
`booked_until_tick` (one step behind when it reads `bus_prev`), so the energy interval ends at the
earlier of the two and the mass check compares chips' inventories with the removal totals at that
tick (a chips record more than `(chips.every_n_macro + lag_macro + 1)` macro steps behind is a
LedgerError). A balance whose owners are not among the run's active domains is reported as not
checked, never as passed. A violation adds a fatal finding (`mass_balance`, `energy_balance`,
`time_balance`) and raises ConservationError: the run fails. Contract: every macro step gives bus_next
fresh records (`state.bus_next = Bus()` before the hot loop, or each writer assigns new record objects);
a non-zero cutting, removal or dynamics record still present from two steps before is refused
(LedgerError) instead of being booked twice. Channels: `ledger.mass_in_kg`, `ledger.mass_out_kg`,
`ledger.mass_residual_rel`, `ledger.energy_in_J`, `ledger.energy_out_J`, `ledger.energy_residual_rel`,
`ledger.energy_residual_J`, `ledger.alignment_lag_s`, `ledger.time_residual_ticks`,
`ledger.thermal_residual_rel`, `ledger.partition_residual_rel` (NaN when not checked). Summary: `{mass,
energy: {max_residual_rel, final_residual_rel, tolerance_rel, checked_steps, ok, note}, mass.terms, time:
{residual_ticks, tolerance_rel: 0, checked_steps, ok, unclassified_ticks, note}, thermal: {solvers,
max_residual_rel, tolerance_rel}, partition: {..., enforced: false}}`. Floating-point bound of the checks
themselves: each residual is formed from per-step sums of at most a few thousand terms, so its rounding
error is below 1e-13 relative, four orders below the tightest tolerance (measured on the toy run of
tests/unit/core: mass residual 4e-16, energy 0).

**Settings** (db/models/recording.yaml, read with `cnctwin.db.models.params("recording")`; project
`recording:` over db/_templates/project.yaml for `fast`, `fast_window_revs`, `fast_period_s`,
`keyframe_period_s`):

| parameter | value | source |
|---|---|---|
| `ring_buffer_s` | 0.5 s | architecture 10 |
| `ring_max_bytes` | 32 MiB | est: within the 50 MB recording budget of architecture 10 |
| `series_buffer_max_bytes` | 4 MiB per tier | est: memory bound, row groups by count |
| `fast_window_min_s`, `fast_window_max_s` | 10 ms, 0.25 s | est: one macro step; half the ring |
| `fast_min_spacing_s`, `fast_max_windows_per_second` | 0.25 s, 2 | est: about 5 % of L0 steps at 10 000 rpm |
| `checkpoint_interval_s`, `checkpoints_keep_last` | 10 s, 2 | architecture 10, 16 |
| `keyframe_full_every`, `keyframe_field_max_cell_m` | 20, 8 mm | architecture 16 |

The run infrastructure draws no random numbers (no RNG streams).

## frames.bin

(owner: `cnctwin/core/rundir.py`, `FramesWriter`, `read_frames`)

Packed little-endian records, one per L1 tick in rising tick order, no file header; the header is
`scene.json["frames"]`:

```
{format: cnctwin-frames, version: 1, file: "frames.bin", record_bytes, byte_order: "little", tick_s, ipo_ticks,
 axes: [{name, kind: linear|rotary, unit: m|rad, origin}], paths: [{path, spindle}],
 fields: [{name, dtype, shape, offset, unit}], doc}
```

| field | dtype | shape | value |
|---|---|---|---|
| `tick` | `<i8` | | base tick of the L1 tick |
| `axis` | `<f4` | [n_axes] | linear: machine value minus `axes[a].origin` (the axis value at the stock origin), m; rotary: angle wrapped to [0, 2 pi), rad |
| `axis_rev` | `<i4` | [n_axes] | revolutions of a rotary axis (`angle = axis + 2 pi axis_rev`); 0 for linear axes |
| `spindle_angle_rad` | `<f4` | [n_paths] | spindle angle of the path, right-handed about +z_T, wrapped to [0, 2 pi) |
| `spindle_rev` | `<i4` | [n_paths] | its revolutions (`unwrapped = angle + 2 pi rev`) |
| `tip_S_m` | `<f4` | [n_paths, 3] | tool tip in the stock frame S |
| `axis_T_S` | `<f4` | [n_paths, 3] | tool axis +z_T in S (unit vector) |

`record_bytes = 8 + 8 n_axes + 8 n_paths + 24 n_paths`. Wrapping: `rev = floor(angle / 2 pi)`, `wrapped =
float32(angle - 2 pi rev)`, and a wrapped value that rounds to 2 pi in float32 is written as 0 with the
next revolution, so `0 <= wrapped < 2 pi` holds in float32. Precision: float32 keeps 24 bits, 6e-8 m at
0.5 m from the origin and 4.8e-7 rad near 2 pi (the reason linear axes are stored relative to the stock
origin and angles wrapped with integer revolutions); the UI never re-implements kinematics: tip and
tool axis come from the motion record. `FramesWriter.append_motion(MotionFrame)` takes axis values from
`PathFrame.axis_actual` by `axis_names`, the spindle from `spindle_angle_rad` + `spindle_rev`, the tip
from `tip_S_m` and the tool axis from `PathFrame.axis_T_S` when the record has it, else
`R_S_M @ axis_T_M` (exact while the stock does not rotate relative to M during the step).
`read_frames(run_dir, tick0, tick1) -> (records, {header, axis_abs, spindle_unwrapped_rad})` gives the
absolute axis values back (origin added, rotary axes unwrapped). The file is truncated to the
checkpoint's byte length at a resume.

## Keyframes

(owner: `cnctwin/core/keyframes.py`, `KeyframeWriter`, `KeyframeReader`)

A keyframe is written every `recording.keyframe_period_s` (0.5 s; the first at tick 0) as number k:

* `keyframes/kf_<k>.glb`: glTF 2.0 binary (asset generator `cnctwin keyframes`), one node per brick written
  in this keyframe, named `brick_<brick>` with `extras {brick}`; a brick with a surface has a mesh with
  one triangle primitive: POSITION unsigned short VEC3 (byte stride 8, KHR_mesh_quantization required),
  indices unsigned int, the node's `translation` = record origin and `scale` = record step (so viewers
  dequantise); an emptied brick is a node without mesh, `extras.empty: true`. JSON with sorted keys and
  fixed separators, chunks padded to 4 bytes: the same input gives the same bytes. Every
  `keyframe_full_every`-th keyframe (20; k % 20 == 0) is a full snapshot with every non-empty brick;
  the others hold the bricks that changed (`changed` = the field's dirty bricks; a dirty brick without a
  mesh is empty now; a brick whose record is byte-identical to its last version is skipped).
* the **brick record** (one contiguous byte range of the BIN chunk, 4-byte aligned): header `<4sIIffff` =
  `b"BRK1"`, n_vertices (uint32), n_indices (uint32), origin_S_m (3 float32), step_m (float32); then
  uint16 positions [n_vertices, 4] (x, y, z, 0) with `p = origin + q * step` in S, metres; then uint32
  indices [n_indices] (triangles, counter-clockwise seen from outside, as the removal meshing gives
  them). The step is `extent / 65535` of the brick's own vertex box (rounded up to float32, origin
  rounded down), so every vertex is within `step / 2` of its float64 position: 0.06 um for an 8 mm
  brick (lattice 0.25 mm), plus float32 rounding where a reader dequantises in float32.
* `keyframes/bricks.idx` (`cnctwin-bricks-index`, rewritten atomically when a brick gets a new version):
  `{cells: 32, lattice_m, grid, origin_S_m, record: <layout text>, bricks: {"<brick>": [[k, file, offset,
  length], ...]}}`, entries sorted by k, file relative to keyframes/. A full snapshot adds no entry for a
  brick whose record did not change, so the index grows with the changes only. Brick ids: the removal
  field's brick number on workpiece 0, `w<w>.<brick>` on workpiece w.
* `keyframes/index.jsonl` (append-only): `{k, tick, time_s, glb, fields, chips, full, bricks [written],
  empty [emptied], ...extra}`.
* `keyframes/kf_<k>.npz` (deterministic zip, members sorted, dated 1980-01-01, read with
  `allow_pickle=False`): `tick` (int64, 0-d), `time_s`, the loop's fields as given (chip pile heights,
  wetted cells, tool deflection, part pose, coolant jets `jet_*`), every float field whose name ends in
  `_K` as float16 (0.25 K resolution at 300 K, 0.5 K at 1000 K), and for every grid (`FieldGrid(values,
  origin_S_m, cell_m)`, cell-centred, NaN outside material) `<name>` block-averaged over the largest
  integer factor f with `f cell <= keyframe_field_max_cell_m` (8 mm; finite values only, a block without
  one is NaN; padded with NaN to a multiple of f), float16 for `_K` grids else float32, with
  `<name>_origin_S_m` and `<name>_cell_m` (= f cell).
* `chips/kf_<k>.npz`: the chips' arrays (particles in flight, pile summary) and `tick`.
* `keyframes/attr/brick_<brick>.npz`: final per-vertex attributes of each brick's last record
  (deviation, finish, wear), written at the end of the run (`write_attributes`).

Reading: `KeyframeReader(run_dir).brick(brick, tick=None | time_s=None)` finds the brick's newest entry at
or before the time in bricks.idx and reads its record with one `os.pread` (the API's
`/bricks/{brick}?t=` does the same); `scene_at(t)` does it for every brick; `playback(t)` lists the
keyframes to load from the GLB files alone (the newest full snapshot and every keyframe after it);
`decode_record(bytes) -> (vertices, faces, header)`. At a resume every keyframe file with k at or
after the cut's count is removed, index.jsonl is truncated to its byte length, bricks.idx is rewritten
from the restored state and attr/ is removed.

## Sweep

(owner: `cnctwin/sweep`; guide: docs/ml.md "Sweeps")

A sweep spec is YAML (format `cnctwin-sweep`, version 1; `cnctwin.sweep.spec.load_spec`, every problem listed in
one InputError, unknown keys are errors, `note`/`notes` free):

| key | value |
|---|---|
| `format`, `version` | `cnctwin-sweep`, 1 |
| `name` | required; the default sweep folder is `DATA_DIR/sweeps/<slug of name>` |
| `description` | text |
| `base` | the base project (a `project.yaml` or its folder), relative to the spec file; the API replaces it by the project it is given |
| `seed` | integer >= 0 (default 0), seeds the design only (the simulation keeps the project's seed) |
| `design` | `{kind: grid \| lhs \| sobol \| random \| list, points, optimization: none \| random-cd \| lloyd (lhs), scramble: true (sobol), list: [{parameter: value}] (list)}`; `points` for lhs, sobol and random only |
| `parameters` | a list of at least one parameter (below) |
| `generators` | `{<name>: {script, function: main, call: constants \| kwargs, output, operation, constants: {}, timeout_s: 600}}` |
| `placeholders` | `{NAME: value}`: fixed values of template placeholders no parameter sets |
| `options` | run options of every point (cnctwin.core.loop.run_project; default `{outputs: false}`); `operator` is not allowed here (overrides are parameters) |
| `execution` | `{workers: 1, memory: 1536M, memory_mode: auto \| systemd \| rlimit \| none, isolation: process \| inline, runner: cnctwin.sweep.runner:run_project, retry_failed: false, verify: full \| quick}` |

**Parameters**: `name` (letters, digits, `_`, `.`, `-`; unique), `kind`, the values: `range: [lo, hi]` (lo < hi;
`scale: linear | log` (log needs lo > 0), `step` (multiples of step above lo, at most hi), `integer: true`,
`levels: n` (n >= 2: the grid's values, ends included)) or `values: [...]` (any YAML values: a categorical
parameter), `default` (a `list` point that omits the parameter), `unit` (text, default `-`; the dataset's unit of
`param.<name>`), `doc`; and per kind:

| kind | keys | written as |
|---|---|---|
| `override` | `target: feed \| spindle \| rapid`, `tool` (T number; none: every tool), `path` (controller path of the tool, default 1); `rapid` has no tool | the point's run option `operator` (below), values are fractions (1 = 100 %) |
| `program_template` | `placeholder` (default the name), `template` (a program file relative to the base project's folder; default the operation's own program), `operation` (default the first), `format` (Python format string, e.g. `"{:.0f}."`) | `{{NAME}}` (spaces inside the braces allowed) replaced in the template, which becomes `points/p<n>/<program file name>`; every placeholder of a template must be bound |
| `project` | `path`: a key path of the project file: keys separated by `.`, list items by `[index]` or `[field=value]` (exactly one item must match) | the value at that path of the variant (the last key may be new); the variant must validate |
| `generator` | `generator` (a name of `generators`), `argument` (the constant or keyword argument) | the generator's program for the operation (`operation` of the generator, default the first), written as `points/p<n>/<output>` |

**Generators** (`cnctwin.sweep.generate`): the script (in the base project's folder) runs in its own process in a
copy of that folder (working directory and first on `sys.path`), with `__name__` not `"__main__"`. `call:
constants`: every top-level assignment (`NAME = ...`, `NAME: type = ...`) of an argument or of a `constants` entry
gets the value (written as a Python literal) before the module runs; a name the script does not assign at top
level is an error; then `function()` is called. `call: kwargs`: `function(**values)` after the module ran
unchanged. The program is the call's return value when it is text, else the file `output` (default the
operation's program file name) in the copied folder. Its standard output is kept in
`points/p<n>/.generated-<name>.json` (`request_sha256` over the script folder's files, the call and the values;
an unchanged request reuses the program).

**Operator option** (`options.operator`, written by `override` parameters; cnctwin.sweep.operator):
`{feed_override: {"*": f, "T<n>": f, "T<n>@<p>": f}, spindle_override: {...}, rapid_override: r}`, normalised (an
entry equal to its default and a table at 100 % are dropped; nothing overridden: no `operator` option, the plain
run). Checked against the machine's `feed_override_range_pct`, `spindle_override_range_pct` and
`rapid_override_steps_pct` (plus 100 %) and the project's cutting tools (a spindle override of a tap or threading
insert is refused).

**The sweep folder** (`cnctwin.sweep.execute`):

| file | format | what |
|---|---|---|
| `sweep.json` | `cnctwin-sweep-run` | `name`, `spec` (resolved: absolute base, defaults filled in), `spec_sha256`, `spec_path`, `points`, `notes` (of the design) |
| `plan.json` | `cnctwin-sweep-plan` | `parameters` (as described), `points: [{index, values, units, project, options, project_sha256, programs}]` |
| `points/p<nnnnn>/project.yaml` | `cnctwin-project` | the variant (a comment header names the sweep, point and values) |
| `points/p<nnnnn>/result.json` | `cnctwin-sweep-point` | `index, values, project, project_sha256, options, status (done \| cached \| failed), key, run_dir, run_id, verdict, complete, registered, error {type, message, traceback}, notes, wall_s` |
| `points/p<nnnnn>/job.json`, `progress.json`, `outcome.json`, `log.txt` | | a point process's request, progress (`sim_time_s, est_total_s, op, phase, wall_s`), outcome `{status, outcome, error}` and output |
| `status.json` | `cnctwin-sweep-status` | volatile: `status (running \| done \| failed \| cancelled), total, finished, done, cached, failed, running, wall_s, eta_s, message, error` |
| `summary.json` | `cnctwin-sweep-summary` | at the end: `points, done, cached, failed, safe, unsafe, results [{index, status, key, run_dir, run_id, safe, error}]` (no wall times) |

## Run cache

(owner: `cnctwin/cache.py`; guide: docs/ml.md "Run cache")

`<root>/index.jsonl` (root `CNCTWIN_CACHE_DIR`, else `DATA_DIR/cache`), JSON lines with sorted keys, append-only:
`{op: format, format: cnctwin-cache-index, version: 1}` first, then `{op: put, key, run_dir, run_id, owned, digest,
files, result_bytes, bytes, build, project_hash, at_s, components}` (`components`: the key's project hash,
databases digest, build, runner, options, platform and environment), `{op: hit, key, run_dir, at_s}`, `{op: drop,
key, run_dir, reason, at_s}`; a compacted index carries `used_s` and `hits` in its puts. The state is the lines
replayed in order per (key, run_dir); a line without its newline (a torn append) is skipped.
`<root>/runs/<run_id>/` holds the runs the cache owns, `<root>/locks/<key>.lock` the key locks, `<root>/.lock`
the index lock. The digest is sha256 over `<relative path>\0<sha256 of the file>\n` of the result set
(cnctwin.core.rundir.result_set_files, sorted) without `series/fast/replay-*.parquet`, series/index.jsonl taken
without its `replay: true` lines.

## Dataset

(owner: `cnctwin/dataset`; guide and every column: docs/ml.md "Datasets", "Columns")

`runs.parquet` / `runs.csv`, `blocks.parquet` / `blocks.csv`: one table each, the columns of `schema.json`
(`cnctwin-dataset-schema`, version 1: `{tables: {runs: [{name, dtype, unit, provenance, doc}], blocks: [...]},
dtypes}`) in that order. Parquet: zstd, no dictionary encoding, statistics, non-nullable fields with
`{unit, provenance, doc}` field metadata and the schema metadata `cnctwin = {format: cnctwin-dataset-table,
version: 1, table}`. CSV: UTF-8, one header row, `\n`, floats as their shortest round-trip text, NaN as an empty
cell. `manifest.json` (`cnctwin-dataset`, version 1): `cnctwin, formats, runs, blocks, files, sources [{run_dir,
run_id, sweep, point, cache_key, build}], skipped [{source, point, reason}], builds, notes`.
