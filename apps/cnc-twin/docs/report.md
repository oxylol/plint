# Reports, setup sheets and the comparison of two runs (`cnctwin.report`)

The report of a run (`report.html`, `report.pdf`), its setup sheet (`setup_sheet.html`,
`setup_sheet.pdf`) and the comparison of two runs (`compare.html`, `compare.json`) are built from
**run directories only** (architecture 16): nothing is simulated or recomputed from the project, and
nothing outside the run directory is read except `docs/approximations.md` of the build (when
run.json does not list the approximations itself). The HTML is one self-contained file (inline CSS,
every picture embedded as a data URI, no scripts, no external resources) that reads on a 390 px phone
and prints to A4; the PDF is the same HTML printed by WeasyPrint. The same run directory gives the
same bytes (section 6).

The uncertainty study (`cnctwin.uncertainty`, docs/models/uncertainty.md) has its own page and a
section of the report.

## 1. Use

```python
from cnctwin.report import build_report, report_html, setup_sheet_html, RunData

result = build_report(run_dir, out_dir=None, formats=("html", "pdf"), setup_sheet=True, study=None)
# -> {"files": [...], "report_html", "report_pdf", "report_pdf_pages", "setup_sheet_html", "setup_sheet_pdf",
#     "setup_sheet_pdf_pages", "verdict": {safe, reasons}, "missing": [{item, reason}], "run_dir"}
text, run = report_html(run_dir, study=None)        # the report as text and the RunData it was read with
sheet = setup_sheet_html(run_dir)

from cnctwin.report.compare import compare, compare_html, write_comparison, compare_runs
doc = compare(run_a_dir, run_b_dir, params=None)    # the comparison document (JSON-able)
doc = write_comparison(run_a_dir, run_b_dir, out_dir)   # + compare.html, compare.json; doc["files"]
```

* `out_dir` defaults to the run directory (where architecture 16 puts the report). `study` is a
  study document, a path to a `study.json`, or `None` (then `<run_dir>/uncertainty/study.json` when
  it exists). `formats` is any of `html`, `pdf`; an unknown one raises `ValueError`, a folder that
  does not exist `FileNotFoundError`.
* Command line: `python -m cnctwin.report RUN_DIR [--out DIR] [--format html,pdf] [--no-setup-sheet]
  [--study study.json] [--json]` (exit 0; 1 for a missing folder or an unknown format) and
  `python -m cnctwin.report.compare RUN_A RUN_B [--out DIR] [--json]` (exit 0; 1 when a folder is
  missing). Both print the verdict or the headline changes.
* API job entry (docs/api.md "Entry points"): `cnctwin.report.compare:compare_runs(request, out_dir,
  progress_cb) -> dict` with `request = {run_a, run_b, run_a_dir, run_b_dir, options}`; it writes
  `compare.html` and `compare.json` into `out_dir` and returns the comparison document (with
  `files`). `options` may override the three thresholds of section 8.
* Importing `cnctwin.report` is light; Matplotlib (Agg, object API, no pyplot) and WeasyPrint load
  when a report is built. For the synthetic run the command takes about 6 s with HTML only and 11 s
  with both PDFs on this box (Python start included), at a peak of about 360 MB.

## 2. What the report shows, in order

1. **Verdict first**: SAFE TO RUN or NOT SAFE TO RUN (architecture 11: safe when no finding is an
   error or fatal), and why: every reason of run.json linked to its finding. A run without run.json
   (not finished) says so and gives the verdict of the findings so far. Headline numbers: cycle time,
   finding counts, takt, energy and tool cost per part, characteristics passed.
2. **Findings**: a table by category and severity (every finding counted), the findings worst first
   (severity, category, time, op and program line linked to the listing, block, message, quantities in
   display units, tool tip in W, a link to the chain of cause), the **chains of cause** (trace.jsonl,
   breadth first like `TraceLog.chain`, each event with time, domain, line, quantities, parents), and
   the **program listing** with the lines that have findings highlighted by their worst severity and
   linked back to their findings.
3. **Cycle time**: the exact breakdown by category in integer ticks (and whether the categories sum to
   the machine time exactly), per operation, per tool, the longest program lines, takt, parts per hour,
   utilisation and the operator times; program stops.
4. **Energy, consumables and cost per part**: electrical energy by consumer (table and bars), the
   spindle terms, coolant (pumped, drag-out, mist, evaporation, concentrate, water, oil, cost),
   compressed air, pump times, chip mass, tool cost per edge, parts per edge and per part, consumables
   cost per part, the batch, and the assumptions of the production metrics. Electricity cost is not
   given: the run directory holds no electricity price.
5. **Tool wear and remaining life**: VB max and limit, life used and left, cutting time and parts left
   on the edge, broken, hottest edge; VB per tooth (bars) and every wear channel over time; tool
   findings.
6. **Inspection**: the CMM table of every characteristic (kind with fit class and datums, feature,
   nominal, limits, measured, deviation, amount out of tolerance, share of the tolerance used, result,
   the program line that made it); the probed points of each characteristic (mean, standard deviation,
   extremes, range = form error seen by the probe); the deviation of the whole machined surface
   (per-vertex statistics and histogram); threads (designation, ISO 965-1 class, measured diameters
   against their limits, gauges); characteristics of kinds the report does not interpret, with all
   their fields; the **error budget** (what each cause adds to each deviation: table and signed stacked
   bars with the total).
7. **Surface finish and burrs**: Ra and Rz against their specification (ISO 4287), the Ra
   distribution over the surface, burrs (edge, kind, height, root, line, position).
8. **Chips**: chip mass per part, removed mass and volume, the chips at the last keyframe (in flight,
   piled, conveyed, highest pile), chip forms by mass (ISO 3685 codes of `cnctwin.core.bus.CHIP_FORMS`),
   the **accumulation map** (pile height), chip findings.
9. **Thermal state**: the temperature map of the part at the last keyframe (a 3-D field as its maximum
   through z), the maximum per keyframe, the **part temperature at unload**, the hottest cutting edges,
   thermal findings.
10. **Coolant reach and shadows**: the share of the run with coolant at the cut, the reach while on
    (mean, lowest), per nozzle (keyframes on, flow, speed, reach), the wetted map, shadows (occluder,
    flow blocked), coolant-shadowing findings.
11. **Chatter and stability**: the stability lobe diagram of every tool with the program's operating
    points (stable or not) and the chatter findings (frequency, index).
12. **Tool deflection and its share of the errors**: per tool the peak deflection per component and
    their combined bound, and the deflection contribution to each characteristic's deviation.
13. **Conservation residuals**: mass, energy and time balances (largest and final residual, tolerance,
    result), the balance channels (`ledger.*`, `*consistency*`, `*residual*`), and the time check of
    path 0 (block time against span, in ticks).
14. **Model approximations** relevant to the run (section 4: the rows of the domains the run exercised that
    state an error bound, largest error first, at most 40, and a link to the full table) and 15. the
    **estimated database values** used (file, line, key, value, note).
16. **Pictures**: the tool path drawn from frames.bin (top and side views over the stock with the work
    zero; feed moves in the first palette colour, rapids grey) and every image of `images/`
    (cnctwin.render) embedded.
17. **Time series**: every channel of channels.json plotted, grouped by domain (small multiples, one
    y-scale each, at most 6 per image): each column a line averaged per plot bucket with the band from
    the bucket minimum to maximum (peaks stay visible); fast channels from their tick-tier envelopes and,
    for the recorded fast windows, the window with the highest peak; the **channel table** (tier, SI
    unit, columns with data against declared, minimum, mean, maximum in display units, what it is) is
    the plots' table view. Tick- and macro-tier columns that channels.json does not declare are
    plotted too and marked undeclared.
18. **Uncertainty** (when the run has a study), 19. **Run identity** (run, project, seed, format,
    clock, L0 strides, numba cache key, versions, thread settings, every input file with bytes and
    sha256), 20. **Not in this run directory**: every file or record the report looked for and did not
    find, and what that means for the report (nothing is left out silently).

## 3. Inputs

The report reads these files and keys; anything absent becomes an entry of section 20 and the
section concerned says what it could not show. Keys carry SI units (architecture 3); for the resolved
project in run.json the keys as written in project files (`size_mm`, `rotate_deg`, `used_min`) are
accepted too and converted with the unit table of `cnctwin.db.keyunits`; a sourced value
`{value, src}` counts as its value.

| file | keys read |
|---|---|
| run.json | `verdict {safe, reasons}`; `summary {cycle_time_s, removed_mass_kg, removed_volume_m3, conservation {<balance>: {max_residual_rel, final_residual_rel, tolerance_rel, ok, residual_ticks, note}}}` (or a top-level `conservation`); `project` (below); `inputs {file: {sha256, bytes}}`, `versions`, `threads`, `numba_cache_key`, `clock {tick_s, ipo_ticks, macro_ipo, l0_ticks}`, `seed`, `outputs`, `project_file`, `format`, `version`; `estimates [{file, line, path, value, note, obj}]` (`cnctwin.db.Estimate`; or `db_estimates`, or a file `estimates.json`); `approximations [{domain, approximation, error}]` (optional) |
| run.json `project` | `name, description, notes, seed, controller_overrides`; `machine {id, name, maker, model, type, controller {dialect, model, name}, spindles [{id, max_rad_s}], axes {<axis>: {travel_m}}}`; `material {id, name, condition, physical.hardness}`; `stock {kind, size_m, tolerance_m, form, mesh}`; `tools.list [{number, path, tool {id, type, geometry {diameter_m, flutes}}, holder {id}, stickout_m, length_offset_m, radius_offset_m, wear_offsets_m {length, radius}, runout_m, life {limit_s, used_s}, comment}]`; `coolant {fluid {id}, concentration_frac, tank_fill_fraction}`; `operations [{name, description, program, programs, program_text, program_texts, program_sha256, setup {workholding [{device {id, type}, pose, label, jaws, jaw_opening_m, input_torque_Nm, grip_depth_m, parallels_m, pressure_Pa}], stock_pose, fixture_pose, work_offsets_m}, from_previous, nozzle_positions}]`; poses are `{translate_m, rotate_rad, order}` |
| status.json | `status` (a run without run.json) |
| findings.json | `findings` (or a bare list): `Finding.to_dict()` |
| trace.jsonl | `TraceEvent` lines (`id` = line number); a line still being written is ignored |
| timeline.json | `blocks [{index, op, program, line, block, path, kind, tick0, tick1, part, tool, ticks}]`, `tick_s`; read with `cnctwin.production.timeline` for the tool of each block |
| channels.json | `channels [{name, unit, tier, domain, doc, max_shape}]` |
| series | `series/index.jsonl {tier, part, file, tick0, tick1, rows}` and `series/<tier>/part-<n>.parquet` (`tick` and the channel columns `<name>`, `<name>[i]`, tick-tier `.min/.mean/.max`) |
| scene.json | `frames` (the frames.bin header), `stock {kind, size_m}`, `work_zero_S_m` |
| frames.bin | `tick`, `tip_S_m` of path 0 (at most 20 000 evenly spaced records are read) |
| keyframes | `keyframes/index.jsonl {k, tick, time_s, fields, chips}`; `kf_<k>.npz`: a 2-D or 3-D `*temp_K` field with `surface_origin_S_m`/`surface_cell_m` (or `thermal_`, `work_temp_` prefixes), `pile_height_m` (origin and cell `pile_` or `surface_`), `wetted` (`wetted_` or `surface_`), the coolant jets `jet_*`, shadows `shadow_id` with `shadow_fraction`; `keyframes/attr/*.npz`: `deviation_m`, `Ra_m` per vertex |
| chips/kf_<k>.npz | `mass_kg`, `form_code` per particle; scalar `*_kg` (pile, conveyed ...), `pile_height_max_m` |
| inspection.json | `temperature_K`; `tolerances [{id, kind, nominal_m, plus_m, minus_m` or `lower_m, upper_m` (a size) or `tolerance_m` (a zone, lower bound 0)`, measured_m, deviation_m, pass, feature, op, program, lines` or `line, tools, datums, fit` or `size.fit, method, points [{x_S_m, y_S_m, z_S_m, deviation_m}], contributions_m [{source, tool, value_m}]}]`; kind `thread`: `designation, class, pitch_diameter_m, major_diameter_m, minor_diameter_m` (a number or `{measured, min, max}`), `gauges {name: accepts}`, `pass`; `summary {checked, failed}`; `unload [{op, time_s, mean_temp_K, max_temp_K}]` |
| surface.json | `features [{id, op, tool, lines, Ra_m, Rz_m, spec_Ra_m, spec_Rz_m, pass}]`, `burrs [{edge, kind, height_m, root_thickness_m, op, program, line, location_S_m}]` |
| tools.json | `tools [{slot, number, tool_id, holder_id, cut_time_s, VB_m [teeth], VB_max_m, VB_limit_m, life_used_fraction, remaining_life_s, broken, max_edge_temp_K}]` |
| production.json | docs/models/production.md section 9: `tick_s, first_part {cycle {breakdown, operations, tools, stops}, lines}, per_part [...]` (energy, consumables, tools with cost, tool cost, removed mass, operator), `batch, parts_planned, parts_simulated, parts_projected, assumptions` |
| sld/<tool>.json | `speed_rad_s [n], depth_limit_m [n], radial_immersion_m, operating_points [{op, program, line, speed_rad_s, ap_m, stable}]` |
| images/ | `*.png *.jpg *.jpeg *.svg *.gif *.webp` (another file type is listed as not embedded) |
| uncertainty/study.json | a `cnctwin-uncertainty` document (docs/models/uncertainty.md) |

The keys of inspection.json, surface.json, tools.json and of the keyframe fields beyond the first
layouts are this module's reading of the result files, which the synthetic run directory
(tests/fixtures/make_synthetic_run.py, `report_extras`, `report_run_extras`) writes; the domains that
own those files write the same keys or say how theirs map (section 9).

## 4. Rules

* **Display units**: lengths in mm (deviations, wear, roughness in µm), temperatures in °C, energy in
  kJ and kWh, power in W or kW, speeds in rpm, feeds in mm/min, volumes in l (coolant, air), mm³ or
  cm³, mass in g; a channel's display unit follows its SI unit and the size of its values
  (`fmt.display_unit`); a named quantity (`deviation_m`, `chatter_freq_Hz`) is shown by the unit its
  name ends with (`fmt.quantity`). Factors come from `cnctwin.db.keyunits` (one unit table).
  Numbers: 4 significant figures, plain between 1e-4 and 1e9, the minus sign U+2212, a dash for a
  missing value, costs below 1 EUR with 3 significant figures.
* **Characteristics**: a size has a nominal and limits (`nominal_m` with `plus_m`/`minus_m`, or
  `lower_m`/`upper_m`); a tolerance zone (`tolerance_m`, no nominal: flatness, perpendicularity,
  position ...) has the natural lower bound 0. *Tolerance used* = deviation over the allowed deviation
  on its side (a size) or measured over the zone. A characteristic's result is inspection.json's
  `pass`; without one it is judged against the limits. The error budget's total is the deviation of a
  size and the measured value of a zone.
* **Remaining tool life**: `remaining_life_s` of tools.json when the wear model gives it, else the edge
  life left `(1 - L)` at the rate of life use of the first part, `(1 - L) cut_time_part / dL_part`;
  parts left `(1 - L) / dL_part` (production.json `life_used_part_frac`). A new edge or regrind
  restarts it (production.json `edge_lives`, `parts_per_edge`).
* **Part temperature at unload**: inspection.json `unload`, else the macro channel
  `thermal.work_unload_mean_K`, else `thermal.work_bulk_mean_K`, at the end of the run.
* **Chip mass per part**: production.json `per_part[].removed_mass_kg` (the mass balance of
  architecture 12 makes it the chips plus displaced and freed material, which the ledger books).
* **Deflection per tool**: the channels of domain `dynamics` with unit m whose name has `deflection`
  or `displacement`; their rows are attributed to the tool of the timeline block they fall in (path 0;
  tools assigned as cnctwin.production assigns them); the bound combines the component peaks, which
  need not occur at the same instant (an upper bound of the vector magnitude).
* **Approximations** (docs/approximations.md, when run.json lists none; `cnctwin/report/approximations.py`):
  the rows of the generated table (`domain | approximation | consequence | error bound | evidence |
  source`) that concern what the run exercised and state an error bound, sorted by impact, at most 40
  (`MAX_ROWS`), with a link to the full table (`/api/docs/approximations.md`, the build's document as the
  API serves it). A domain was exercised when it shows activity in the run: a recorded channel of it
  whose value changed (macro and tick tiers; a domain declares its channels whether or not it acted, so
  a force that stayed 0 or a temperature that stayed at ambient does not count), a finding or a trace
  event of it, or its result with content (production.json, an inspection with tolerances, remaining
  material or gouges, rendered pictures, sound, an attached uncertainty study, a calibration overlay
  the project included). Table domains without activity of their own follow another
  (`mechanics` with `cutting`, `geom` with `inspect`, `hotloop` with itself or `cutting`); `controller`,
  `motion` and `loop` concern every run; `scope`, `performance` and `gpu` none (scope decisions and CPU
  cost, not errors of a result). Impact is the largest relative error the bound states, in percent
  (percentages and ranges, a validation row's p90, a factor f as (f - 1) x 100 %, an order of magnitude
  as 900 %); bounds without one (exactness, ulps, absolute errors) come after, in document order. A
  document that is not in the generated form is listed by sections instead: those whose heading or
  text names an active domain or whose heading names a topic of every run (`general, scope, core,
  loop, clock, frames, conservation, recording, determinism, numerics, all runs`), rendered from a
  small Markdown subset (paragraphs, bullets, tables, code, bold), escaped.
* **Bounds** (presentation, `cnctwin/report/context.py`, `rundir.py`, `sections_*.py`): the findings
  table shows the worst 400 findings with their chains (the table by category counts all of them,
  findings.json holds them); a program is listed whole up to 400 lines, else as excerpts of 3 lines
  around every line with a finding; the cycle-time section lists the 30 longest lines; time series
  use 600 plot buckets; a fast window keeps the 4 000 samples around its peak; a panel shows at most
  8 columns (more fold into one band from the lowest minimum to the highest maximum, with the mean).
* **Palette** (`cnctwin/report/plots.py`): 8 categorical colours in a fixed order validated for
  colour-vision deficiency on adjacent pairs, one-hue blue ramp for magnitudes (heat maps), blue/red
  with a grey midpoint for signed values (sensitivity), status colours only next to a text label
  (PASS/FAIL, severities), text in ink colours, one y-scale per plot, legends for two or more series,
  every chart beside a table of the same numbers.

## 5. The setup sheet

Built from run.json's resolved project, timeline.json, findings.json, frames.bin and images/:

* **Machine** (id, name, maker, model, type, control, spindle maximum speeds, axis travels,
  controller overrides); **stock and material** (kind, size, tolerance, form, mesh, material, name,
  condition, hardness);
* per **operation**: the workholding with its position (pose translation in mm and rotation in degrees
  with the order) and settings (jaw opening, input torque, grip depth, parallels, pressure), the stock
  and fixture poses, the operation it comes from, the **work offsets** (machine coordinates of the work
  zero as set on the machine), the **program** (file, lines, title from its first comment, sha256),
  and the **operator notes**: the project and operation descriptions and notes, every program stop
  (M00/M01) and message line, and the findings that ask for attention at the machine (chip packing,
  chip hazards, coolant shadows, part slip or free part, mislocation, tool end of life, pull-out,
  unbalance, collision, overtravel, burr, thermal error) with their line and time;
* the **tool list** (number, id, type, diameter, flutes, holder, stick-out from the holder nose,
  length and radius offsets, wear offsets, runout, life used of limit, operations using it, comment);
* **coolant** (fluid, concentration, tank fill, nozzle positions per tool); **pictures** (the tool path
  over the stock with the work zero, and the run's images, `images/*setup*` first).

## 6. Determinism

Same run directory, same build, same bytes of report.html, report.pdf, setup_sheet.html,
setup_sheet.pdf, compare.html (tested). No clock is read; every list is sorted by a key of the data;
Matplotlib runs with `rcdefaults()` plus a fixed style inside an `rc_context`, DejaVu fonts, PNG
written with `metadata={"Software": None}` (no version text; PNG output has no date); WeasyPrint
writes no dates (the HTML carries no `dcterms.created`) and no random identifier. The bytes depend on
the versions of Matplotlib, FreeType, WeasyPrint and the fonts (another build may render
differently, as architecture 9 allows for reports). A comparison re-rendered from its compare.json
gives the same compare.html.

## 7. Phone and print

* Phone (390 px): a viewport meta tag, body text 14-15 px, headline tiles in a fluid grid, tables in a
  box that scrolls sideways (tables with more than 5 columns keep 720 px, more than 9 keep 980 px, so
  narrow columns do not collapse into one word per line), figures 4.8 in wide at 150 dpi (8 pt text
  stays about 9 CSS px when the image is shown 390 px wide; sharp at 2x pixel density), no
  hover-only content, details/summary for the chains (no script). Checked with Chromium at 390 px:
  the page never scrolls sideways as a whole.
* Print (WeasyPrint, A4): 15/12/16/12 mm margins, the run name and verdict in the running header, page
  numbers, 9.5 pt text, 7.8 pt tables (6.8 pt above 5 columns), long words in cells may break, figures
  and table rows are not split across pages, figures at most 120 mm high. A test renders the report
  and checks that no table is wider than the page.

## 8. Comparing two runs

`compare(a, b)` returns `{format: cnctwin-compare, version: 1, a, b, verdict {a_safe, b_safe, changed},
summary [headlines], findings {matched [...], only_a [...], only_b [...], counts}, dimensions [...],
cycle {total, categories, lines, tools}, tools [...], energy {total_J, consumers, tool_cost_eur,
coolant_cost_eur, takt_s, parts_per_hour}, surface [...], inputs [...], parameters, missing {a, b}}`.

* **Findings** are matched in three passes, each finding at most once: (1) the same category and the
  same execution of the same program line (the k-th time the line runs, from timeline.json), which
  does not depend on feeds or speeds; (2) the same category and the same line, or for findings without
  a line the same inspected feature (a message that starts with a characteristic id of either run),
  or the same operation, paired by program progress (time over each run's cycle time) within
  `finding_match_progress_frac` by an optimal assignment (Hungarian method,
  `scipy.optimize.linear_sum_assignment`); (3) the rest is resolved in B (only in A) or new in B
  (only in B). A matched pair carries the time difference, the severity change (worse, better, same)
  and the change of every quantity both carry.
* **Characteristics** by id: both measured values, B - A, the share of the tolerance used, pass/fail;
  *worse* when B fails and A passes or the change exceeds `dimension_change_tolerance_frac` of the
  tolerance band with more of the tolerance used, *better* the other way, else *same*; only in A or B.
* **Cycle time**: total, per category, per program line (largest change first), per tool, from
  production.json (or the timeline); **tools**: VB max, life used, cutting time, hottest edge, cost per
  part; **energy and cost per part**: total, per consumer, tool and coolant cost, takt, parts per hour;
  **surface**: Ra and result per feature; **inputs**: which input files differ (run.json `inputs`
  sha256). A change below `relative_change_significant` of the larger value is *same*.
* Thresholds (db/models/report-compare.yaml): `finding_match_progress_frac` 0.02 (est),
  `dimension_change_tolerance_frac` 0.1 (the 10:1 rule of measuring-system resolution, aiag-msa-2010),
  `relative_change_significant` 0.01 (est).

## 9. Decisions and their consequences

* **The run directory is the only input** (architecture 16): a report can be rebuilt anywhere from a
  copied run folder and says exactly what was simulated. Consequence: what a run does not record cannot
  be reported (it is listed in section 20), and the setup sheet needs the resolved project in run.json.
* **Tolerant reading, explicit absence**: every reader returns nothing rather than failing, and every
  absence is listed with its meaning. Consequence: a broken or partial run still gives a report, which
  says it is partial; a test asserts that an empty folder renders.
* **PNG plots, not SVG**: deterministic, small for long series, and rendered the same by browsers and
  WeasyPrint. Consequence: plots do not zoom without loss and have no hover tooltips; every plot has a
  table with its numbers (channel table, breakdowns, statistics) for exact values.
* **Bounded sizes for long runs**: series are reduced to 600 buckets per column part by part, fast
  windows read only the columns needed, frames.bin is sampled, the findings table and the program
  listing are bounded (section 4). Consequence: the report of a long run stays a few MB and builds in
  bounded memory; the full data stay in the run directory and the API.
* **No electricity price**: the run directory has none and a report must not invent one; energy is
  given in kWh.
* **The keys of the result files** beyond the first layouts (section 3) are this module's
  specification, written by the synthetic run; a domain writing other names must map them or extend
  the readers (`cnctwin/report/sections_part.py`, `sections_production.py`).

## 10. Limits

* The report shows what the domains wrote: an inspection without probed points or an error budget, a
  run without keyframe fields or sld files, gives the corresponding sections as "not recorded".
* Deflection per tool is attributed through path 0's timeline: on a multi-path machine the deflection
  channel of another path's tool is attributed to the tool path 0 has in that block, unless the
  channel carries its own tool index.
* The fast window of a channel is chosen by its highest |value| over all its columns; a window split
  by a series part boundary counts as two windows.
* Surface deviation statistics are per mesh vertex, not weighted by area.
* Thread limits are shown as inspection.json gives them; the report does not compute ISO 965-1 limits
  from a designation.

## References

aiag-msa-2010 (db/bibliography/models-report.yaml) for the change threshold; iso-1101-2017,
iso-286-1-2010, iso-965-1-2013, iso-1-2016, iso-4287-1997 for the inspection conventions displayed;
docs/models/production.md for production.json; docs/models/uncertainty.md for the uncertainty section.
