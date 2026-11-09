# How to add a machine

A machine is one file, `db/machines/<id>.yaml` (format `cnctwin-machine`, version 1). It describes the
control as the machine has it configured, the axes and their drives, the kinematic chains, the spindles, the
structure, errors and thermal behaviour, the tool changer, probes, coolant, power and the collision geometry.
The controller, motion, thermal, structure, coolant, production and render domains read it; a project names
it with `machine: <id>`. Format rules: [formats.md "Databases"](formats.md#databases); rules for authors:
[db/_templates/README.md](../db/_templates/README.md).

## 1. Collect the sources first

* The maker's specification sheet or product page (travels, feeds, rapids, spindle speed, power and torque
  curves, tool changer, tank), the layout drawing (table size, spindle nose to table, enclosure), the
  operator's manual of the control (look-ahead, smoothing codes, override keys, settings), and the data
  sheets of options you model (probes, tool setter, high-pressure coolant).
* Add one bibliography entry per source to `db/bibliography/machines.yaml` (or a new
  `db/bibliography/<area>.yaml` with `format: cnctwin-bibliography`, `version: 1`, `entries:`). The key is
  `author-year[-topic]`, lower case with dashes; a dated type needs `year`, a web page `accessed` and `url`;
  `pages` says where in the source the values are. A key may be defined in several files only if both
  describe the same work the same way.

## 2. Copy the template and fill it

```
cp db/_templates/machine.yaml db/machines/<id>.yaml      # the id is the file name: lower case, [a-z0-9._+-]
```

Sections (every key is declared with its unit in `cnctwin/db/schemas.py`; the template comments list the
allowed values):

| section | what to give | notes |
|---|---|---|
| header | `format`, `version: 1`, `id`, `maker`, `model`, `type` (`vertical_mill`, `horizontal_mill`, `five_axis_mill`, `lathe`, `mill_turn`, `swiss`), `src` | `type` selects the controller variant (FANUC/Haas M or T series) and the defaults |
| `controller` | `dialect` (`fanuc`, `haas`, `siemens840d`, `heidenhain_klartext`, `heidenhain_iso`), `model`, `alarms` (the id of `db/controllers/<id>.yaml`), `ipo_period_ms`, `block_processing_ms`, `lookahead_blocks`, least input increments, `max_feed_mm_min`, `rapid_mm_min` per axis, `smoothing` (codes and tolerances), `corner_default`, `in_position_mm`, override ranges and rapid steps | the IPO period sets the L1 tick (architecture 5.1): it must be a whole number of microseconds |
| `axes` | every NC axis: `type`, `travel_mm` (machine coordinates, as G53 sees them) or `travel_deg`, rapid and feed limits, `accel_m_s2`, `jerk_m_s3`, `servo` (`kv_1_s`, `feedforward`, `velocity_bw_Hz`), backlash, reversal spike, ballscrew, ISO 230-2 positioning, `motor_force_N`, `moving_mass_kg`; rotary axes add `continuous`, `max_rpm`, brake, `backlash_arcsec` | makers rarely publish accelerations, jerk and servo gains: estimate them with a note |
| `kinematics` | `workpiece_chain` (machine base to the fixture face) and `tool_chain` (base to the spindle nose), each a list of links: an axis link `{link, axis, direction}` (rotary: `axis_point_mm`), or a fixed link `{link, fixed: {translate_mm, rotate_deg}}`; `home_mm` (linear axes), `home_deg` (rotary), `tool_axis`, `singularities` | every axis must be in a chain or a turret; `direction` is the motion of that link in the base frame for a positive command (a table that moves left for +X has `[-1, 0, 0]`); further chains (sub spindle, second turret) go in `workpiece_chains` / `tool_chains` |
| `spindles` | `id`, `kind`, `nose`, `max_rpm`, `gear_ranges` (`rpm`, `power_kW` and `torque_Nm` as `[[rpm, value], ...]`, `rating` saying which duty), acceleration and orientation times, `position_loop`, inertia, efficiency, idle power, runout, `frf_modes` at the holder flange, `drive_type`, `bearings`, `thermal` | a spindle without gear ranges is a warning (no power or torque limit); the thermal model is the bearing network when `bearings` are given, the lumped body when only `growth_at_max_speed_um`, `time_constant_min`, `bearing_heat_at_max_speed_W` are (docs/models/thermal.md 4.1) |
| `structure` | `modes` (frame), `table_modes`, `rotary_modes`: `{fn_Hz, zeta, k_N_um, direction}` | relative tool-to-workpiece modes, not the spindle's |
| `error_map`, `thermal` | ISO 230-1 squareness, straightness, angular and pivot errors; ballscrew drift, structure sensitivity, warm-up | `structure_sensitivity_um_per_K` is the tool-to-work drift of the whole machine per K of a uniform temperature change, the spindle head included (thermal takes the spindle model's own share back, docs/models/thermal.md 4) |
| `tool_changer`, `turrets`, `pallet_changer`, `tailstock`, `steady_rest`, `guide_bushing`, `doors` | capacities, times, limits; lathes: turret stations, mount, index times, live stations and chain | `capacity:` holds the lathe working envelope |
| `probes` | `spindle_probe`, `tool_setter`, `laser` | a project's `kind: probe` tool entry names one of these |
| `coolant`, `chip_conveyor`, `enclosure`, `power` | tank, pumps (`kind`, pressure, flow, power), nozzles with position, aim, diameter, pump and programmable positions, air blast, filtration; conveyor; enclosure box; base and standby power, air | M codes switch pumps by kind (M08 flood, M07 mist/MQL, through-spindle codes per dialect) |
| `geometry` | collision bodies attached to links: `primitives` (`box`, `cylinder` ...), `collide` | in the frame of the link they are attached to, mm |

A machine that differs from an existing one only in some sections can **extend** it: `extends: <id>` and the
sections that differ (a list is replaced whole, so repeat a spindle list to change one spindle). Every value
keeps the source it was written with.

## 3. Sourcing rules

* Every number has a source: its own `src`, or the `src` of the nearest enclosing mapping (the file's `src`
  covers the catalogue values). `{value: 21200}` under a section with `src: haas-uk-vf2ss-2026` is sourced.
* A value no source gives is an **estimate**: `{value: 5.0, src: est, note: "how it was derived"}`. The note
  says exactly how: a calculation with its inputs, a typical range and where it comes from, "assumed" with the
  reason. `twin db check --strict` refuses an estimate without a note.
* Never attach a real source to an invented number. If a source gives a range, store the value used and
  `range: [lo, hi]`; a value read off a chart is `digitised: true` with `uncertainty_pct`.
* The unit is the last token of the key (`travel_mm`, `accel_m_s2`, `power_kW`, `max_rpm`,
  `time_constant_min`); minimum and maximum are prefixes; temperature differences are `_dK`. A key with the
  stem of a declared field and another unit is an error that names the declared unit.
* Write the current key forms of the template: the migration table (formats.md) renames older forms at load
  and lists them with `--migrations`, but a new file should need none.

## 4. Check

```
bin/twin db check --strict                                   # 0 errors, 0 warnings for the whole database
bin/twin db check --estimates | grep 'db/machines/<id>'      # every estimated value with its note: read them
bin/twin db check --migrations | grep 'db/machines/<id>'     # should print nothing for a new file
```

`twin db check` prints every problem as `file:line: key.path: message` and exits 1 on an error. Then build the
machine the way the controller and motion see it, and run the minimal project of its dialect:

```
bin/tsafe python -c "
from cnctwin.db import Database
from cnctwin.controller.machine import build_machine
from cnctwin.motion.machine import MachineModel
m = Database.load().get('machine', '<id>')
print(build_machine(m)); print(MachineModel(m))"
bin/twin new /tmp/new-machine --machine <id>                 # project.yaml + the shortest program of the dialect
bin/twin check /tmp/new-machine --out /tmp/new-machine/check # dry run, strides, cost estimate
```

## 5. Tests

* `tests/unit/db/test_real_db.py` loads, validates (also strictly) and builds every database file: a new
  machine file is covered without a new test. Run `bin/test unit/db`.
* Add a test when the machine brings something the tests do not cover yet: a new kinematic layout (a test in
  `tests/unit/motion/test_kinematics.py` with a forward/inverse round trip, as
  `test_fk_ik_round_trip_for_each_five_axis_type`), a new turret or sub-spindle arrangement
  (`tests/unit/controller/test_multipath.py`), a new probe layout (`tests/unit/controller/test_probe.py`).
  The controller tests use small inline machines (`tests/unit/controller/harness.py`), so a test states
  exactly what it depends on.
* There is no test yet that builds every database machine through both machine models: the snippet in
  section 4 is that check, run it by hand.

## 6. Worked example: `db/machines/haas-vf2ss.yaml`

The Haas VF-2SS file starts with a comment that fixes the frames, then the header whose `src` covers the
catalogue values:

```yaml
# Frames: base frame origin = centre of the table top with X = Y = 0 (home); +x right, +y toward the
# column (away from the operator), +z up. Haas home is the + end of every axis (travel is negative).
format: cnctwin-machine
version: 1
id: haas-vf2ss
maker: "Haas Automation"
model: "VF-2SS"
type: vertical_mill
src: haas-uk-vf2ss-2026            # db/bibliography/machines.yaml: the haas.co.uk specification tables
```

Published values carry no `src` of their own (`max_feed_mm_min: 21200`, `motor_force_N: 8874`); values Haas
does not publish are estimates whose notes give the reasoning:

```yaml
  ipo_period_ms: {value: 1.0, src: est, note: "Haas does not publish the interpolation period; 1 ms is typical of current servo-drive CNCs"}
  ...
    accel_m_s2: {value: 5.0, src: est, note: "Haas does not publish accelerations; 0.5 g is typical of ballscrew VMCs of this size and leaves >2x margin on the published thrust (8874 N) for the est. 300 kg table plus a 300 kg part"}
```

A value from another source names it, with the page:

```yaml
  rapid_override_steps_pct: {value: [5, 25, 50, 100], src: haas-ngc-mill-om-2020, page: "p. 48, Table 2.16: ..."}
```

The kinematics: the table carries the part (Y then X), the head carries the spindle; the spindle nose
position is part published (610 mm above the table) and part assumed, which its note says:

```yaml
kinematics:
  workpiece_chain:
    - {link: saddle, axis: Y, direction: [0, -1, 0]}       # saddle and table move toward the operator for +Y
    - {link: table, axis: X, direction: [-1, 0, 0]}        # table moves left for +X
    - {link: table_top, fixed: {translate_mm: [0, 0, 0]}}
  tool_chain:
    - {link: head, axis: Z, direction: [0, 0, 1]}
    - {link: spindle_nose, fixed: {translate_mm: [381, 203, 610]}, src: est, note: "z 610 = max spindle nose to table (published); x/y place the spindle over the table centre at mid-travel (X -381, Y -203), assumed"}
```

The spindle's power and torque curves were digitised from the maker's chart, so the gear range says so and
states its duty:

```yaml
    gear_ranges:
      - rpm: [0, 12000]
        rating: "peak = 200 % spindle load, available for 3 min (haas-torque-note-2026)"
        src: haas-torque-vf40t-2016
        digitised: true
        uncertainty_pct: 3
        power_kW: [[0, 0], [300, 3.9], ..., [12000, 26.0]]
        torque_Nm: [[0, 125], [600, 125], ..., [12000, 20.7]]
```

(Every database file is written in the current key forms: `twin db check --migrations` lists no migrated key.
The loader still migrates the older forms of the migration table, docs/formats.md, for files written against the
first templates, such as `growth_um_at_max_rpm` for `growth_at_max_speed_um`; write the new form.)

An **extending** machine: `db/machines/dmg-dmu50-3rd-siemens.yaml` is the DMU 50 with a SINUMERIK 840D sl
instead of the TNC 640. It writes only its identity (`id`, `model`), `extends: dmg-dmu50-3rd`, its `src` and the
section that differs, `controller`; everything else is the base's. `extends` deep-merges mappings, so a controller
key the base writes and the variant does not (here the TNC's `five_axis.tcpm` name) stays the base's: write such
a key in the variant when the base's value is wrong for it.
