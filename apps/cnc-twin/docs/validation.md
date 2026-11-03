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
| `harnesses/` | `orthogonal` (chip_ratio, shear_angle), `temperature` (chip_temperature, tool_temperature), `taylor` (taylor_life), `spindle` (spindle_growth), `coolant` (coolant_reach), `chipform` (chip_form), `cycletime` (cycle_time); helpers `common`, `contact`, `toolheat`, `assumptions` |

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
* `kind="class"` with `reference_class` for classifications (chip forms, grades, stability points:
  error 0 on a match, 1 otherwise). `component` names the budget component when the quantity's budget
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
`toolheat` (thermal's insert ladder and the local edge field at a point).

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

`validation/budgets.yaml` defines budgets for cutting_force (three components), torque,
drilling_thrust, drilling_torque, tool_temperature, chip_temperature, chip_ratio and shear_angle. Its
header states the rule, applied exactly:

* tolerance of a value = max(budget_pct / 100 x |reference|, 2 x uncertainty of that value), the
  percentage of the rise above ambient for temperatures;
* a case passes when every compared value is within its tolerance (a value without a prediction is not);
* a quantity passes when at least 80 % of its (independent) cases pass and no case misses by more than
  twice its tolerance.

The two numbers of the quantity rule are written only in the header comment; `budgets.py` holds them as
constants and a unit test checks the header still states them. Quantities the file has no budget for
(taylor_life, spindle_growth, coolant_reach, chip_form, cycle_time, ...) are reported with their errors
and "not judged"; the suite adds no budget of its own.

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

`docs/validation-report.md`: a summary table (cases, independent and calibration-dependent counts,
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
| tool_temperature | contact through the coefficient pipeline coupled to thermal's insert ladder; natural thermocouple = rake + flank contact mean; IR window/spot and embedded thermocouple = body + local edge field at the location | cutting time (5 s when not stated), sensor positions from the case text, insert/bar geometry defaults |
| taylor_life | the wear law as the wear domain builds it (published Usui, else calibrated to the database Taylor entry), integrated from a fresh edge to the VB/VN criterion; VB(t) curves | the Taylor entry used, coolant as a friction factor |
| spindle_growth | thermal's first-order spindle body at the case's speeds from a cold start | every spindle parameter (no spindle data in the papers): model defaults, the database VMC's maximum speed |
| coolant_reach | the coolant's lubrication friction factor (reach 1) in the contact temperature of a milling engagement, or in the chip form and cutting force of a turning cut | fluid mapped to the closest database fluid |
| chip_form | Oxley chip state + `mechanics.chipform` per tested feed/depth (or a feed-depth grid for a chip-control window), mapped to the case's classes | chip-breaker geometry when the insert is not in the database |
| cycle_time | the case's path in ISO syntax through `cnctwin.controller` + `cnctwin.motion` (MotionRunner, no cutting) with the stated IPO, acceleration, jerk, feed and tolerance | drive data and look-ahead from the database machine of the same control family, G01 polygon of the CAM path |

The remaining quantities (cutting_force, drilling_thrust, drilling_torque, roughness, stability_lobe,
wall_error, torque) get their harnesses from the removal/cutting/dynamics/structure/surface work; add a
module to `harnesses/`, list it in `registry.HARNESS_MODULES` and follow section 4.
