# Validation suite (`cnctwin.validation`, `twin validate`)

The suite runs every published experiment in `validation/cases/` through a **harness** that simulates
it with the domain libraries, compares each reference value with the simulated one, applies the error
budgets of `validation/budgets.yaml` and writes `docs/validation-report.md` and
`validation/results/<quantity>.json`. It never runs the full simulation loop: each harness calls the
libraries directly at the level the experiment measured (an orthogonal cut, a tool body, a spindle, a
controller with its planner).

## 1. Running it

```
twin validate                         # all cases; writes docs/validation-report.md and validation/results/*.json
twin validate --quantity chip_ratio   # one quantity (repeatable); writes its results/<quantity>.json, no report
twin validate --case <id>             # one case (repeatable); prints only
twin validate --report out.md         # the markdown report to a path (any selection)
twin validate --results DIR           # the JSON results to a directory
twin validate --json                  # print the JSON documents instead of the summary table
twin validate --no-write --list       # list cases and harnesses
```

Exit code 0 when no budgeted quantity fails, 1 when one fails or a case could not be simulated (the
error is in the report), 2 on a usage error. `twin validate` is `cnctwin.validation.main(argv)`.
Run Python through `bin/tsafe` (memory cap, single-threaded, keyed Numba cache). The output is
deterministic: cases in (quantity, id) order, no time stamps, floats rounded to 6 significant digits in
the JSON.

Tests: `tests/unit/validation/` (fast: framework, metrics, budgets, provenance, report, real case
loading) and `tests/validation/` (marked `slow`: every case through its harness; a quantity whose
independent cases miss their budget **fails**; quantities without a harness or a budget are skipped
with their statistics): `bin/tsafe python -m pytest -m slow tests/validation`.

## 2. Package

| module | content |
|---|---|
| `cases.py` | `Case` (loaded through `cnctwin.db`: `Database.validation_case`, `ids("validation_case")`), setup leaves with their sources, assumed inputs, per-test tables, reference units |
| `registry.py` | `Prediction`, `HarnessOutput`, `Assumption`, `Provenance`; `@harness(quantity)`, `@discussion(quantity)` |
| `metrics.py` | comparison of every value and curve point; case statistics |
| `budgets.py` | `validation/budgets.yaml` and its rule |
| `provenance.py` | which bibliography keys are one work; calibration dependence |
| `runner.py` | runs cases, aggregates per quantity, verdicts |
| `report.py` | markdown and JSON |
| `harnesses/` | `orthogonal` (chip_ratio, shear_angle), `temperature` (chip_temperature, tool_temperature), `taylor` (taylor_life), `spindle` (spindle_growth), `coolant` (coolant_reach), `chipform` (chip_form), `cycletime` (cycle_time), `cutforce` (cutting_force, torque), `drilling` (drilling_thrust, drilling_torque), `roughness` (roughness), `stability` (stability_lobe), `wallerror` (wall_error); helpers `common`, `contact`, `toolheat`, `cutsim` (prescribed cuts through the cutting kernels and coefficient pipeline), `assumptions` |

## 3. Cases

A case follows `db/_templates/validation_case.yaml` and is validated by the `validation_case` schema at
database load. The suite reads it as written (`Case.raw`) with these conventions:

* **Setup leaves** are plain values or `{value, src, note, assumed}`. A leaf is **assumed** when its
  `src` is `est`, `assumed` is true, or its key is in the section's `assumed_inputs` list; the report
  lists every assumed leaf of a case next to the harness's own assumptions.
* **Units** are the last token of the key (`vc_m_min`, `edge_radius_um`) and are converted by
  `cnctwin.db.keyunits` (`Case.si(section, key)`); reference units are the free texts of the case
  (`um`, `m/min`, `degC`, `index` = a label), converted by `keyunits.parse_unit_text`.
* **Per-test lists**: keys `<name>_per_test_<unit>` (or `_per_curve_`) in `conditions`, `tool`,
  `stock` or `coolant` give one value per test, named by `conditions.test_ids`; scalars are shared
  (`per_test_table(case)`). A reference value finds its test by a test id among the tokens of its name
  (`V0072`, `test 3`, `A1`), else by its `x` against the per-test values (`find_test`).
* **Bounds**: a reference value whose note says "lower bound" ("upper bound") is one-sided: a simulated
  value on the allowed side has zero error (e.g. tempering colours at the top of the colour table).
* Temperatures are compared as **rises above the ambient** the harness used (`HarnessOutput.ambient_K`:
  the case's `ambient_temp_degC`, else 20 degC, listed as an assumption).

### Adding a case

Write `validation/cases/<quantity>/<id>.yaml` from the template, run `twin db check --validation`
(schema, bibliography keys), then `twin validate --case <id>`. If the case's quantity has a harness, the
harness must be able to read everything it needs from the case: use the key forms above (per-test lists
with units, `test_ids`, reference names that carry the test id or an `x`). If it needs geometry the
template has no key for (a sensor position), add the case to the harness's table (for example
`harnesses/temperature.SENSORS`) with the case text the value comes from.

## 4. Harnesses

```python
from cnctwin.validation.registry import harness, discussion, HarnessOutput, Prediction, no_prediction, not_compared

@harness("cutting_force", models=("cnctwin.mechanics.coefficients",), description="one paragraph for the report")
def cutting_force(case, db) -> HarnessOutput:
    ...
    return HarnessOutput(values=[...], curves=[[...], ...], model_chain=[...], settings={...},
                         assumptions=col.assumptions, provenance=col.provenance, notes=col.notes, ambient_K=None)

@discussion("cutting_force")
def _why(quantity_result) -> dict:          # optional: "Failures and likely causes" text, computed from results
    return {"summary": "...", "<case id>": "..."}
```

Contract:

* `values[i]` is the `Prediction` of `case.reference_values[i]`; `curves[c][k]` of point k of
  `case.reference_curves[c]`, **in the unit of that reference**. Status `compared` (a value),
  `no_prediction` (the model gives none where the experiment measured one: counts as a miss) or
  `not_compared` (outside what this harness validates, with the reason: a duplicate of a counted value
  in another unit, a point the case itself excludes; listed, not counted).
* `kind="class"` with `reference_class` for classifications (chip forms, grades: error 0 on a match, 1
  otherwise). `bound="lower"` / `"upper"` makes a reference one-sided for this prediction when the case note
  does not say so: the reference bounds the simulated quantity from below (above), and only the error on the
  wrong side counts. The stability harness uses it for test points: a stable test at depth a says the limit is
  at least a, a chatter test that it is below a, so the simulated limit is compared with the tested depth as a
  bound and the percentage budget applies to the limit (a border test stays two-sided). `component` names the budget component when the quantity's budget
  has components (`cutting_force`: `main_force_pct`, `thrust_force_pct`, `milling_peak_force_pct`).
  `uncertainty` replaces the reference's uncertainty when the harness propagates it (a derived chip
  ratio from its measured thickness).
* `model_chain`, `settings`: what was run with which settings (shown per case).
* `assumptions`: every input the case does not state, with the value used and where it comes from
  (database value, model default, documented assumption in `harnesses/assumptions.py`).
* `provenance`: every database value the model chain used, with its `src` and note; the runner decides
  calibration dependence from it (section 7). Use `common.Collector` (`assume`, `used`, `note`) and
  `common.material_provenance`.

Rules (the integrity rules of the suite):

1. Call the domain libraries directly, at the level the experiment measured; never the run loop.
2. Take every model parameter from the databases and the case's stated inputs. Never tune a model,
   a coefficient or a budget to a reference; a miss is reported with its likely cause.
3. An input the case does not state is a database value or a documented assumption, listed in the
   output. Do not edit other packages; if a domain API lacks something, work around it inside the
   harness and say so in the report.
4. No `@njit` / `@numba` text in validation modules (it would change the kernel cache key of
   `bin/tsafe`); deterministic code only.

Helpers: `common` (work material and tool side from a case, tool-material and coating mapping, edge
radius / rake / clearance / speed / thickness readers, `Collector`), `contact.ContactModel` (the
coefficient pipeline on the case's own speeds and thicknesses, heat split as thermal's partition),
`toolheat` (thermal's insert ladder and the local edge field at a point), `cutsim` (the cutting domain's
coefficient pipeline (`MechanicsModel`, with the calibration provenance recorded) and its force kernels over a
prescribed cut: `mill` (steady straight milling with analytic engaged intervals), `drill` (steady
full-diameter drilling), `tube_turning` (orthogonal tube turning on a polar field); tools from inline items
through `tooling.build_tool`).

## 5. Metrics (`metrics.py`)

Per compared value: relative error e = (sim - ref) / |ref|; temperatures relative to the rise above
ambient; one-sided for bounds; 0/1 for classes. Curves are compared point by point at the reference x
(the harness simulates at those x, so x is always matched); the report adds each curve's RMS and maximum
|e|. Uncertainty of a value: the prediction's propagated one, else the value's or the curve's
`uncertainty`, else the case's `uncertainty_pct` of |reference| (as written); "within uncertainty" is
|sim - ref| <= uncertainty.

Aggregation per quantity, over the counted values of the **independent** cases: median and 90th
percentile of |e| (values without a prediction are counted separately), the share of values within their
uncertainty, cases within budget, the largest miss in tolerances.

## 6. Budgets (`budgets.py`)

`validation/budgets.yaml` defines budgets for every quantity: part 1 for cutting_force (three
components), torque, drilling_thrust, drilling_torque, tool_temperature, chip_temperature, chip_ratio and
shear_angle; part 2 for taylor_life (50 %), spindle_growth (30 %), coolant_reach (15 %), chip_form (25 % for
window boundaries, classes must match), cycle_time (5 %), roughness (30 %), stability_lobe (25 % of the limit)
and wall_error (25 %). Each entry's note and evidence give the justification (scatter of the experiments in the
cases, the sources' own model comparisons, error propagation); for taylor_life, spindle_growth, coolant_reach,
chip_form and cycle_time the file says that a first run of their harnesses had been seen when the budget was set
and that the justification does not use those results. Its header states the rule, applied exactly:

* tolerance of a value = max(budget_pct / 100 x |reference|, 2 x uncertainty of that value), the
  percentage of the rise above ambient for temperatures;
* a case passes when every compared value is within its tolerance (a value without a prediction is not);
* a quantity passes when at least 80 % of its (independent) cases pass and no case misses by more than
  twice its tolerance;
* a classification is within tolerance only on a match; a one-sided reference counts only the error on
  the wrong side.

The two numbers of the quantity rule are written only in the header comment; `budgets.py` holds them as
constants and a unit test checks the header still states them. A quantity the file has no budget for is
reported with its errors and "not judged"; the suite adds no budget of its own. The cutting_force milling
values (means, peaks, extremes, RMS of milling force components) use the `milling_peak_force_pct` component;
the orthogonal and turning values `main_force_pct` (Fc) and `thrust_force_pct` (feed, thrust, back force).

## 7. Calibration dependence (`provenance.py`)

A case is **calibration-dependent** when a database value its model chain used was built from the same
work as its reference (the same paper supplies the Taylor constants, a Johnson-Cook set chosen for its
agreement with the same experiments, a friction coefficient identified from the same tests). Two
bibliography keys are one work when their DOI or URL is equal, their normalised titles and years are
equal, a bibliography note says one is the same work / document / a version of the other, or they are
listed together in `provenance.SAME_DATA` (data sets the bibliography does not link: the ETH Zurich
orthogonal database published as an arXiv report and a journal article; the sections of one maker's
catalogue). A used value depends on the reference when its `src` is in the reference's group, or when
it is an estimate (`src: est`) whose note names a key of the group ("fitted here" to that source); the
note of a sourced value names alternatives and does not count. Calibration-dependent cases are reported
separately and never enter a verdict.

## 8. Reports

`docs/validation-report.md`: a plain summary table first (quantity, cases, independent and calibration-dependent counts, independent cases within budget, median error, verdict), then the detailed summary table (cases, independent and calibration-dependent counts,
independent cases within budget, median and p90 error, values within uncertainty, budget, verdict) and a
plain list of passing, failing, unjudged and harness-less quantities; per quantity the budget, the
verdict with its reason, the harness, the statistics and the "failures and likely causes"; per case the
source and location, measured or benchmark, the model chain and settings, the assumed inputs (case and
harness), the database values used, a table of every reference value (reference, simulated, error,
tolerance, within or not, note) and every curve with its RMS/max error.
`validation/results/<quantity>.json` (`format: cnctwin-validation-results`) holds the same data.

## 9. The harnesses of this suite

| quantity | what is simulated | main open inputs (listed per case) |
|---|---|---|
| chip_ratio | `oxley.solve` at t1 = h - h_min per test, tool and work at ambient; ratio h / t2 against the measured h / h_chip | edge radius where not stated (tooling default), tool material by grade |
| shear_angle | as chip_ratio; phi from the simulated ratio of the kind the source used (h / t2 or V_chip / V) with tan phi = r cos a / (1 - r sin a) | as chip_ratio |
| chip_temperature | Oxley's mean chip temperature, oxide exposure during contact and air cooling (`mechanics.colour`), the colour-equivalent temperature capped at the table's top | chip cooling in still air (convection + radiation) |
| tool_temperature | contact through the coefficient pipeline coupled to thermal's insert ladder with its 3-D cutting-region field (the rake and flank heat deposited as contact patches, `ToolThermal.deposit_contact`); natural thermocouple = rake + flank contact mean; IR window/spot and embedded thermocouple = the field's temperature at the location (`ToolThermal.temperature_at`) | cutting time (5 s when not stated), sensor positions from the case text, insert/bar geometry defaults |
| taylor_life | the wear law as the wear domain builds it (published Usui, its A anchored to the pairing's database Taylor entry when there is one; else calibrated to that entry; the flank land softened at its own temperature), integrated from a fresh edge to the VB/VN criterion; VB(t) curves | the Taylor entry used, coolant as a friction factor |
| spindle_growth | thermal's spindle network (`cnctwin.thermal.spindle`: Palmgren-Harris bearing friction with the lubricant viscosity at the bearing temperature, motor losses, shaft / bearing sets / housing, growth from their temperatures and lengths) at the case's speeds from a cold start, minus the lift of the machine structure (`structure_model`, heated by the ambient and the head) = the measured tool-to-table loop; the ambient per batch, rising linearly by the stated rise | the whole spindle construction (none described in the papers): the model's 40-taper defaults, no cooling; the structure sensitivity of the database VMC |
| coolant_reach | the case's jet (bore, distance, pressure, flow, direction) traced through a minimal occlusion scene of the case (insert, holder, bar; rotating cutter, block) to the cut; then the lubrication friction factor at that reach, the jet's convective cooling of the tool faces (`cnctwin.coolant.engagement.face_h`) in thermal's tool ladder over the pass (milling: contact temperature at the body temperature, pass-averaged), and the hydraulic wedge (`engagement.contact_length_ratio`) on the rake friction force and the chip's natural curl (turning) | fluid mapped to the closest database fluid, the jet's reference line (beta) and the milling jet's aim, the cut length and stickout |
| chip_form | Oxley chip state + `mechanics.chipform` per tested feed/depth (or a feed-depth grid for a chip-control window), mapped to the case's classes | chip-breaker geometry when the insert is not in the database |
| cycle_time | the case's path in ISO syntax through `cnctwin.controller` + `cnctwin.motion` (MotionRunner, no cutting) with the stated IPO, acceleration, jerk, feed and tolerance | drive data and look-ahead from the database machine of the same control family, G01 polygon of the CAM path |
| cutting_force | the cutting domain's force model at the case's engagement: rows of `cutting.coefficients.MechanicsModel` (Oxley / Johnson-Cook, calibrated to the material's measured Kienzle or mechanistic data of the pairing where the database has them, else the mechanics alone), linearised to the kernel table; orthogonal and tube turning through `turn_step` (straight edge at kappa_r 90 deg across the width, polar field; the closed form (K h + Ke) b listed as a check); milling through `cut_step` (tooling-built cutter, steady straight path, analytic engaged intervals of the straight cut, prefilled memory), each reference statistic (mean, peak, extreme, RMS, mean FXY) from the simulated force on the tool in Altintas' frame; per-case readers (`cutforce.MILLING`) state frames, statistics and which values are fresh-tool references | tool and work at ambient, rake / edge radius / entering angle where the source is silent (tooling defaults or `assumptions.py`), runout 0, source frames whose orientation is not defined compared as magnitudes |
| torque | the process torque of the same milling simulation | as cutting_force |
| drilling_thrust, drilling_torque | the case's twist drill built by tooling (lips with the helicoid rake of the helix, chisel from the web or the stated chisel length, margins), rows for the process drilling (indentation at the chisel and the slow centre), steady full-diameter `cut_step` (all stations engaged over the full circle, a pilot hole removes the inner ones); mean thrust and torque over a revolution | web, chisel angle, helix, point angle and edge radius where not stated (tooling defaults), the coolant as a friction factor |
| roughness | `cnctwin.surface.kernels.envelope_profile` (lower envelope with the minimum chip thickness, Brammertz, and side flow of the ploughed layer) of the generating edge (tool moving along +u: major edge leading, minor edge trailing) (insert nose and edges with kappa_r and kappa_r' from the ISO designations, cutter corner and end-edge dish, ball scallops and feed marks) at the case's feed or step-over, evaluated by `surface.roughness.evaluate` (ISO 4287 / 4288 / 16610-21) | edge radius (tooling default of the edge kind), the measured surface where not stated, ideal cutters (no runout) |
| stability_lobe | `dynamics.sld.limit_at_speed` (ZOA) and `dynamics.semidisc.limit_at_speed` (SD, the domain's m and points per mode period; the source's m = 40 for its own benchmark) of the linearised cut (straight teeth, or the tooling-built helical cutter through `dynamics.operating.cut_model`) with the case's modes and linear coefficients; limit = min(ZOA, SD); tests compared one-sided; `dynamics.drive.run_benchmark` (time domain) at each test listed | process damping (none stated), helix where est.; SD skipped above 120 intervals per tooth period (long delays, ZOA accurate) |
| wall_error | Budak: the cutting simulation's station forces at the surface-generating instants through a cantilever of the stated gauge length with tooling's fluted section and the stated collet stiffness (overcut positive), the maximum normal force, the static tip stiffness k = kc kb / (kc + kb); Insperger: the steady forced vibration of the case's modes through `dynamics.drive.run_benchmark`, SLE at the generating instants (material left positive) | rake and clearance of the end mills (`assumptions.py`), straight teeth in the SLE run |

Inputs of the experiment's own characterisation (modal parameters, the linear cutting coefficients of the
stability and SLE cases) are case inputs like the geometry; database values (flow stress, friction, Kienzle and
mechanistic entries, tool-material moduli) are recorded with `Collector.used` and decide calibration dependence.

Known gaps the harnesses report instead of hiding: tooling has no cylindrical margin on end mills (the Zylka
MAR 40-110 um curves are listed, not counted), no taper on ball end mills (Engin taper ball as a cylinder flank),
no individual insert offsets (Nowakowski: ideal cutter), and complex-residue FRFs get ZOA only.

## 10. What the second round of harnesses found (for the domain owners)

The per-case causes are in docs/validation-report.md ("Failures and likely causes"); the patterns behind them:

* **Mechanistic calibration across geometries** (`mechanics.coefficients.build_calibration`): the factors
  measured / model are formed at the entry's reference geometry and applied unchanged at the tool's own. The
  computed Ti6Al4V MAL entry (rake 10 deg, helix 30 deg, Krc 263 MPa) gives a radial factor of about 3.5, which
  at the rake-0 round inserts of the Engin cases makes the in-plane peaks 2-4 times the measured ones.
* **Flank wear land in milling** (Smithey et al. plastic land, `cutting` flank forces): with VB 0.11-0.21 mm the
  simulated mean normal force of the Wang 2025 side cut rises about tenfold, the measured one by a third to two
  thirds: a fully plastic land over the whole engaged arc does not describe a thin milling chip.
* **Edge (ploughing) force**: thrust and feed forces are high where the chip is thin against the edge radius
  (orthogonal thrust +35 to +100 %, the axial force of ball-end slotting at small feed), as docs/models/mechanics.md
  4 states for the (2 + pi) k contact.
* **Drilling**: the full-drill thrust is 70-120 % high in three independent cases (Gok 1045 HSS, Craciun Al,
  Sklenicka C45 carbide: the chisel and slow-centre indentation dominate it), the outer-lip thrust of Ti6Al4V with
  pilot holes about 90 % low (tooling's default sharp drill edge, the emulsion friction factor), the full-drill
  thrust and torque of 7075 about half (a calibration-dependent case); torques range from a median 11 % (Craciun)
  to -36 % (Krivokapic, HSS at 3-8 m/min, where Oxley falls back to Merchant at most map nodes and the map build
  costs about 4 minutes per drill in `mechanics.oxley.continuation_reference`).
* **Kinematic roughness with the minimum chip thickness** (`surface.kernels.envelope_profile`, Brammertz): where
  f^2 < r h_min (small feeds, the nearly flat minor edge of a face-mill insert, the ball's feed marks) the uncut
  layers accumulate to about h_min before a pass removes them; with the estimated edge radii this makes the
  profile an h_min sawtooth well above the measured roughness. Where the feed marks dominate (hard turning at
  0.14-0.2 mm/rev) the profile is within about 10-30 %. *Addressed 2026-11-09*: the ploughed layer is now
  pressed to the pass and flows sideways into the free space under it (volume conserved; Brammertz's R_t kept),
  so a trailing flat edge levels the feed marks (Nowakowski: no more 10x scatter); the harness's corner template
  had the major edge trailing and its slope as cot kappa_r (fixed). Below f = sqrt(2 r h_min) any rule that
  generates the surface where the chip reaches h_min gives R_t = h_min (also the per-pass reading of Brammertz):
  the remaining low-feed misses follow the unstated edge radii.
* **Stability**: the deepest trough of every analytical benchmark is reproduced within its digitisation
  uncertainty (the limits run 3-12 % below the charts in the troughs, SD alone 3-10 %), and 153 of 211 measured
  tests lie on the right side of the simulated limit (Engin 2/2, Insperger 2006 56/69, Mann 27/31 and 26/31, Weck
  20/40 and 22/38); the misses sit on steep flanks and in the flip-lobe speed ranges, and in Weck's slotting
  tests the simulated limit (0.78-0.96 mm) is below the stable 1.0 mm cuts.
* **Wall error**: in Budak's up-milling 7075 case the profile near the tool tip is within about 20 % (49 of 94
  points inside the budget) and falls off faster toward the shank than measured (a rotational clamp compliance
  would add the linear part); the down-milling case is about 1.7 times and the AISI 1040 case 2-5 times the
  measured error, following the simulated normal force (444 against 166 N maximum in the 1040 case): there the
  thin-chip normal force of the force model, not the deflection model, sets the miss. The static tip stiffness of
  the cantilever-on-collet model with tooling's fluted section is within 18 % of both measured tools. The
  surface location error of Insperger's low-immersion cut has the measured peaks at the resonant speeds, but
  between them the simulated steady SLE is positive where the measured one is negative.
* **Spindle growth** (third round): the physical spindle network replaced the lumped defaults. Run on the model's
  generic 40-taper construction (the papers describe no spindle) it reproduces the measured trends (growth rising
  less than proportionally with speed, more growth in cold ambient through the lubricant viscosity, a fast shaft
  and a slow housing response, the structure's slow lift) but over-predicts the plateaus about twofold: the
  bearing friction of Palmgren's grease f0, the housing's exposed area and growth length are estimates; a case
  with a described spindle is needed to test the construction data (spindle_growth/GAP.md).
* **Coolant** (third round): the jet is traced in each case's scene. The vertical jet of Struzikiewicz is half
  blocked by the uncut shoulder ahead of the corner, which raises its cutting force like the measured one (+15 %
  against the shallow jets), but the dry force level of the mechanics is about twice the measured; the hydraulic
  wedge shortens the contact by 2-7 % only and leaves the chip grades unchanged (the jet's push on the chip's back,
  the measured chip-breaking mechanism at alpha 60-90 deg, is not modelled). The mechanics has no input for an
  imposed (jet-shortened) contact length: the harness scales the friction factor and the natural curl's contact
  length instead (an input of `oxley.solve` / `group_coefficients` for the contact length ratio would do it
  properly). In Sultana's milling the jet keeps the tool body near ambient while the dry body warms by 140-165 K
  over the 100 mm pass; the dry contact temperature is already over-predicted by the contact model (flank
  contact 1000-1500 degC under a 50-100 um chip), which sets the miss.
* **Cutting force**: the best independent agreement is the high-feed face mill (Reznicek, in-plane peaks within
  6 %, axial -40 to -58 %), the IN718 tube (Wood, 14 %) and Ti6Al4V turning (Slusarczyk, median 19 %); the
  round-insert face mill (Plodzien, small ap) and the Ti6Al4V slot cases are high by factors 2-4.
