# Production metrics (`cnctwin.production`)

Cycle time broken down exactly per block, per program line, per tool and per operation; operator
and takt time; batches of N parts with the progression of wear, offsets, tool changes and cycle
time; electrical energy per part by consumer; coolant, MQL oil and compressed air; tool cost per
part; the run's `production.json`. Pure functions over the run's data, and an aggregator the loop
calls once per part. No random draws (no RNG streams); no bus records (production is not a domain).

## 1. Use

```python
from cnctwin.production import ProductionAggregator, PartInputs, EnergyInputs, ToolState
agg = ProductionAggregator.from_project(ctx.project, tick_s=ctx.tick_s)   # resolved project
agg.add_part(PartInputs(part=k, blocks=blocks_of_part_k, energy=EnergyInputs(...),
                        removed_mass_kg=m, tools=[ToolState(...), ...]))  # after each part
agg.write(run_dir)                                                        # production.json

from cnctwin.production import production_from_run_dir
document = production_from_run_dir(run_dir, project=None, machine=None, coolant=None)
```

* `ProductionAggregator(*, machine, tick_s=1e-6, coolant=None, concentration_frac=None, tools=None,
  operator=None, parts_planned=1, dialect=None, payload_kg=None, initial_life_frac=None,
  life_limit_s=None, spindle_extra_inertia_kgm2=0.0)`; `.from_project(project, *, tick_s=None,
  machine=None, coolant=None, tools=None, parts_planned=None, payload_kg=None)`; `.add_part(inputs)
  -> dict` (the part record); `.result() -> dict`; `.write(run_dir) -> Path`.
* `PartInputs(part, blocks, energy=None, removed_mass_kg=None, tools=[])`;
  `EnergyInputs(dt_s, spindle_omega_rad_s, spindle_shaft_power_W, axis_vel_m_s=None, axis_names=(),
  axis_process_power_W=None, feed_work_J=0.0, omega_tolerance_rad_s=0.0, velocity_tolerance_m_s=0.0)`
  (one sample per entry of `dt_s`, each holding for its duration);
  `ToolState(number, slot=-1, tool_id="", life_used_frac=None, life_used_part_frac=None,
  cut_time_s=None, VB_max_m=None, wear_offsets_m={}, changes=0, broken=False)`.
* Pieces: `cycle_time(blocks, part) -> CycleTime`, `OperatorModel.from_project(project, machine)`,
  `spindle_energy`, `axis_energy`, `axis_models`, `pump_energy`, `conveyor_power_W`, `base_energy`,
  `air_specific_energy_J_m3`, `process_feed_energy`, `coolant_states`, `coolant_use`, `air_use`,
  `tool_cost`, `read_timeline`, `read_frames`, `read_series_column`, `energy_inputs`, `tool_states`.

## 2. Inputs

**timeline.json** `{format: cnctwin-timeline, version, tick_s, blocks: [...]}`; a block runs
`[tick0, tick1)` in base ticks: `index, op, program, line, block, path, kind, tick0, tick1`
(required), and additively `part` (0-based, default 0), `tool` (T number; else taken from the last
tool-change block's T word on the same path; lathe turret blocks of kind `turret_index` read
T`ttoo`), `ticks` (`{category: ticks}`), `in_cut_ticks` (feed ticks in contact, when `ticks` is not
given), `pumps_on` (pump ids running during the block), `stop` (M00/M01/M30; else read from the
block text). The loop passes the same records as `TimelineBlock`s (`block_from_record`).

**frames.bin** with its header `scene.json["frames"]` (`record_bytes`, `fields [{name, dtype,
shape, offset}]`, `axes [{name, kind}]`, `tick_s`): fields `tick`, `axis` (values relative to the
stock origin), `spindle_angle_rad` (wrapped to [0, 2 pi)) and `spindle_rev` per L1 tick. Spindle
speed and axis velocities are forward differences over each tick of the unwrapped angle
(`2 pi rev + angle`) and of the positions; changes within twice the float32 resolution of the
recorded angle (and of the largest position) per tick are held (`hold_within`): otherwise the
quantisation noise adds up as accelerations (about 4e-4 of the run-up energy on the synthetic run).

**series** tick tier, channel `cutting.spindle_power_W` (shaft power of cutting); absent: taken as
0 and listed in `assumptions`. **tools.json** `{tools: [{slot, number, tool_id, life_used_fraction,
cut_time_s, VB_max_m, broken, wear_offsets_m?, changes?, per_part?: [{part, ...the same}]}]}`
(`life_used_fraction` is the life used of the edge in use; with several parts `per_part` is
required). **run.json** `clock.tick_s`, `summary.removed_mass_kg` (or `summary.parts[i]`),
`summary.feed_work_J`, `project` (when the caller does not pass the resolved project, machine and
coolant fluid: they must be the database objects, not ids).

From the resolved project (architecture 18): `machine` (spindles, axes with `moving_mass_kg`,
kinematic chains, coolant pumps, tank, air blast, chip conveyor, doors, pallet changer, power),
`coolant.fluid` and `coolant.concentration_pct`, `tools.list[i]` (`tool.price_eur`,
`tool.regrinds`, `life.limit_min`, `life.used_min`), `production.parts`, `production.operator`.

## 3. Cycle time

Categories (`CATEGORIES`): `cutting` (feed in contact), `air_cutting` (feed out of contact),
`feed` (feed, contact not recorded), `rapid`, `acceleration` (time lost to acceleration and
corner slow-down), `dwell`, `tool_change`, `spindle_ramp`, `coolant_delay`, `probing`,
`program_stop`, `auxiliary` (other M functions, doors, pallets, conveyor, messages), `path_wait`
(waits for another path and gaps in the timeline), `unclassified` (what a block's `ticks` leave
uncovered); a writer's own category name is kept as it is. Without `ticks` a block goes whole to
the category of its kind (`rapid`, `dwell`, `tool_change`/`turret_index`, `spindle`->`spindle_ramp`,
`coolant`->`coolant_delay`, `probe`/`tool_setter`->`probing`, `stop`->`program_stop`,
`sync`->`path_wait`, the rest `auxiliary`; `linear`/`arc` to `cutting`/`air_cutting` by
`in_cut_ticks`, else `feed`).

Sums are integer ticks (architecture 12, "Time"): the categories of a block sum to `tick1 - tick0`
(booking more raises `InputError`); per path the blocks plus the gaps between them (`path_wait`)
sum to the path's span; blocks overlapping on one path raise `InputError`; an operation's machine
time is its longest path's span and its breakdown that path's; the part's machine time is the sum
of its operations (time between operations, re-clamping, is operator time). Per tool: each block,
and the gap after it, go to the tool active in the block; per line: every execution of
a program line (subprograms, loops) adds up, with the execution count. Seconds are formed only at
the output (`ticks * tick_s`): the breakdown is exact, its accuracy that of the timeline.

## 4. Operator and takt

Per part and operation: unload, clean, load and two door movements (`load_s + unload_s + clean_s
+ 2 door`, the door movement `doors.open_time_s` of an automatic door, else `door_s`); per M00 and
per M01 when `optional_stop` is on: two door movements and `stop_s`; inspection `inspect_s` on the
first part (when `first_article_inspection`) and every `inspect_every`-th part; `tool_replace_s`
per tool or edge change in the part. Without a pallet changer the machine waits for all of it;
with two or more pallets the handling overlaps the machining of the other pallet and the machine
waits `pallet_change_s` per operation plus what the handling takes beyond the machine time; with
`inspect_parallel` the inspection overlaps the next part's machining likewise.
`takt_s = machine time + machine waiting time`, the time between finished parts the cell keeps up;
`parts_per_hour = 3600 n / sum(takt)`.

Project keys `production.operator`: `load_s, unload_s, door_s, clean_s, inspect_every, inspect_s`
(template) and additively `stop_s, optional_stop, inspect_parallel, first_article_inspection,
tool_replace_s`; a key not given takes db/models/production-operator.yaml (30, 20, 4, 15, 0, 60 s
as in the template; stop 30 s; optional stop off; inspection with the machine waiting; first
article inspected; tool replacement 120 s; all `est`). `operator_model.sources` in production.json
says which values came from the project.

## 5. Batch

The aggregator keeps one record per simulated part. When `production.parts` exceeds the parts
simulated, the remaining parts are projected: each tool's life used per part continues at its mean
over the simulated parts; a tool (edge) change happens on the part where the cumulative life used
passes a whole edge life (the first change of a tool that started at life `L0` comes after
`(1 - L0)/rate` parts); cycle, energy and consumables are the last simulated part's (base load,
standby and the machine's own air scaled to the projected takt); operator time is recomputed per
part (inspection pattern, tool replacements). Life used per part: `life_used_part_frac` when given,
else the difference of `life_used_frac` from the previous part (a part with `changes` > 0 uses
`changes - L_before + L_after`), else `cut_time_s / life.limit_min` of the project; the first part
starts from `life.used_min / life.limit_min`. Progression arrays over the batch: machine time,
takt, energy, tool cost, and per tool life used, life used per part, VB_max (simulated parts),
changes and wear offsets (held at the last simulated value on projected parts).

## 6. Energy

Electrical energy per part, the sum of (sources: gutowski-2006 and balogun-mativenga-2013 for the
split into basic, ready and cutting power):

* **spindle**: `E = (E_shaft + E_kin+) / eta + sum(P_idle(omega) dt) - E_recovered` with
  `E_shaft = sum(max(P_shaft, 0) dt)`, `E_kin+` the sum of the positive changes of
  `J omega^2 / 2` (J = `rotor_inertia_kgm2` + what the caller adds for tool and holder),
  `P_idle(omega) = P_idle,ref (|omega| / omega_max)^x`, `x = palmgren_speed_exponent = 5/3`
  (Palmgren's load-independent friction torque `M0 = 1e-7 f0 (nu n)^(2/3) dm^3`, palmgren-1959, so
  the loss power `M0 omega ~ n^(5/3)`; valid for `nu n >= 2000` mm2/s rpm, i.e. above a few hundred
  rpm with spindle oils), `P_idle,ref = idle_power_W` and `eta = efficiency` of the machine file
  (electrical to shaft at rated load), `omega_max` its `max_rpm`; braking energy (and negative shaft
  power) recovered at `spindle_regen_recovered_frac * eta` (default 0: braking resistor).
* **axes** (each linear axis of the machine file): per sample the mechanical work
  `W = m g_z v dt + (mu N + F_seal) |v| dt + P_process dt + Delta(m v^2 / 2)` (kinetic energy
  change booked at the end of the sample: exact for piecewise-constant velocity), `m` the axis'
  `moving_mass_kg` plus the payload the caller passes for the table axes (stock, fixture), `g_z` the
  upward component of the moving body's +motion from the kinematic chain times g, `N = m g` on the
  guides of a horizontal axis (0 for a vertical one), `mu = axis_guide_friction_coeff`,
  `F_seal = axis_seal_drag_N`; electrical `W / (eta_screw eta_drive)` when `W > 0`,
  `W axis_regen_recovered_frac eta` when negative. The process feed power, when known only as the
  ledger's total feed work, is `feed_work_J / (eta_screw eta_drive)`. Plus `servo_standby_W` for
  the whole takt.
* **pumps**: `power_kW * pump_load_factor / pump_motor_efficiency` while each pump runs. Pump time
  comes from the blocks' `pumps_on`, else from the program's coolant codes (the tables of
  cnctwin.sanity.scan: M08 flood, M88 through-spindle, M07 mist/shower, M83/M73 air; a tool change
  and the program end stop coolant), per path; with an MQL fluid the codes run the MQL unit (no
  pump), with air the air blast.
* **chip conveyor**: `conveyor_power_W[type]` while commanded (Haas M31/M33), or for the whole
  machine time when the program never commands it.
* **base load**: `power.base_W` for the whole takt.
* **compressed air**: free-air volume (section 7) times
  `e = p0 ln((p0 + p_g)/p0) / compressor_isothermal_efficiency` (isothermal compression work of an
  ideal gas, cengel-boles-2015; `p_g` the air blast pressure of the machine, else
  `air_supply_gauge_pressure_Pa`): 0.379 MJ/m3 at 6.9 bar(g), i.e. 0.105 kWh/m3, the range of
  industrial screw compressors (doe-compressed-air-2003).

Parameters (db/models/production-energy.yaml): `palmgren_speed_exponent` 5/3 (palmgren-1959),
`spindle_regen_recovered_frac` 0, `axis_guide_friction_coeff` 0.003 (0.002-0.005),
`axis_seal_drag_N` 30 N, `ballscrew_efficiency` 0.9, `axis_drive_efficiency` 0.85,
`axis_regen_recovered_frac` 0, `pump_motor_efficiency` 0.8 (IE2 class near 0.8 at 0.75-1.1 kW,
iec-60034-30-1-2014), `pump_load_factor` 1, `conveyor_power_W` by type (auger 200 W, belt and
scraper 370 W, drum filter 750 W), `compressor_isothermal_efficiency` 0.55 (0.4-0.7),
`air_supply_gauge_pressure_Pa` 6e5. All but the exponent are estimates (`src: est`, with notes).

**Error bounds** (from the parameter ranges, propagated term by term; the machine-file values are
estimates themselves and bound the result as much): spindle cutting energy +0/-10 % (the rated-load
efficiency used at part load overstates the efficiency by up to 5-10 points at 25-50 % load);
spindle idle +-30 % (exponent and the idle power at maximum speed); run-up energy exact for the
given inertia, converted at eta (+-10 %); axis friction +-60 % (mu and seal drag ranges) but
typically below 1 % of the part's energy; gravity and inertia as good as the masses (+-20 %);
pumps +-25 % (motor efficiency and duty point); compressed air +-25 % (isothermal efficiency
0.45-0.65); base load and standby as the machine file (+-30 %). With operator waits the base
load and the air usually dominate the energy per part: the total is then good to about +-25 %.

## 7. Coolant, MQL, air

Water-mixed fluids (emulsion, semi-synthetic, synthetic): mixture lost
`V = drag_out_ml_per_kg_chips * m_chips + mist_loss_fraction * V_pumped` (`V_pumped` = pump flow x
time), concentrate `c V` (`coolant.concentration_pct` of the project, else the fluid's), water
`(1 - c) V` plus evaporation `evaporation_l_per_h_per_m2 * (tank_l / tank_depth_m) * takt`
(tank_depth_m 0.35 m, est). Straight oil: `V` is oil. MQL: total loss
`dosage_default_ml_h` (fluid file, else `mql_dosage_m3_s` = 20 ml/h; DGUV 5-50 ml/h,
dguv-209-025-mql-2010) times the MQL running time, plus `mql_air_m3_s` (100 l/min) of carrier air.
Compressed air: `power.air_consumption_l_min` for the takt, `coolant.air_blast.flow_l_min` while
the air blast runs, the MQL carrier air. Cost: concentrate or oil volume times `cost_eur_per_l`.
Chip mass: `removed_mass_kg` of the part (ledger; from the run summary). Error: the drag-out of the
fluid files is a range (20-80 ml/kg for emulsions): +-50 %. (The MQL values are written in m3/s
with unit "-" in the model file because cnctwin.db.keyunits has no `m3_s` key token.)

## 8. Tool cost

`cost per part = price_eur / (regrinds + 1) * life used per part` (one new edge plus each
regrind), `parts per edge = 1 / life used per part`; null with the reason in `missing` when price
or life is unknown. Regrinding service and holder costs are not included (the cost is a lower
bound by the regrind service price).

## 9. production.json

`{format: cnctwin-production, version: 1, tick_s, categories, parts_planned, parts_simulated,
parts_projected, first_part: {cycle, lines}, per_part: [...], batch: {...}, progression: {...},
operator_model: {..., sources}, assumptions: [...], parts, cycle_time_s, cut_time_s, air_time_s,
tool_change_time_s, operator_s, per_part_s, removed_mass_kg, energy_J}` (the last keys are the
summary of the first run-directory layout, same meanings: machine time, cutting time, the rest of
the machine time, tool-change time, the operator times, takt and energy of the first part).

* `first_part.cycle`: `machine_ticks`, `machine_s`, `breakdown {total_ticks, total_s, ticks{},
  s{}}`, `operations [{op, span_ticks, critical_path, breakdown, paths [...]}]`,
  `tools [{tool, breakdown}]`, `stops`; `first_part.lines`: per program line `{op, program, line,
  block, executions, breakdown}`.
* `per_part[k]`: `part, simulated, machine_ticks, machine_s, breakdown, operations, operator
  {components_s, operator_s, machine_wait_s, takt_s, m00, m01, inspected}, takt_s,
  removed_mass_kg, energy {spindle{shaft_J, accel_J, brake_J, motor_loss_J, idle_J, recovered_J,
  electrical_J}, axes{axes{X: {inertial_J, gravity_J, friction_J, process_J, mechanical_J,
  electrical_J}, ...}, mechanical_J, electrical_J}, base_J, servo_standby_J, pumps{id: {on_s,
  power_W, electrical_J}}, conveyor{on_s, power_W, electrical_J, commanded}, air_specific_J_m3,
  air_J, total_J, total_kWh, missing[]}, consumables {coolant{...}, air{machine_m3, air_blast_m3,
  mql_m3, total_m3}, pump_on_s, air_blast_on_s, mql_on_s}, tools [{number, slot, tool_id,
  life_used_part_frac, life_used_frac, cut_time_s, VB_max_m, wear_offsets_m, changes, broken,
  cost{price_eur, regrinds, edge_lives, cost_per_edge_eur, parts_per_edge, cost_per_part_eur,
  missing}}], tool_cost_eur`.
* `batch`: `parts, machine_s, total_s, takt_mean_s, parts_per_hour, machine_utilisation_frac,
  energy_J, energy_kWh, energy_per_part_J, energy_by_consumer_J{spindle, axes, servo_standby, pumps,
  conveyor, base, compressed_air}, coolant{...}, air{...}, tools [{number, tool_id, life_used_frac,
  changes, cost_eur, cost_per_part_eur, parts_per_edge, edge_lives}], tool_cost_eur,
  tool_cost_per_part_eur, tool_changes`.
* Written with sorted keys and fixed separators (`write_production`): the same results give the
  same bytes (architecture 9); non-finite numbers are written as null.

## 10. Assumptions and limits

Listed in `assumptions` of every production.json: an operation's machine time is its longest
path; base load, standby and the machine's air run for the whole takt; the conveyor runs for the
cycle unless commanded; spindle idle scales with speed^(5/3); braking energy is dissipated; coolant
follows the M codes when the timeline has no `pumps_on`; tool cost without the regrind service. The
cycle time is exact in ticks; everything else is as good as the run's signals and the machine,
fluid and tool data (machine files publish few energy figures: section 6). Several parts of one
batch running at the same time on several pallets are not modelled beyond the pallet overlap of
the operator time.

## References

palmgren-1959, gutowski-2006, balogun-mativenga-2013, cengel-boles-2015, iec-60034-30-1-2014,
doe-compressed-air-2003 (db/bibliography/models-production.yaml), dguv-209-025-mql-2010.
