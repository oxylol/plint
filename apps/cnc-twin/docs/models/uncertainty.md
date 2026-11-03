# Uncertainty studies (`cnctwin.uncertainty`)

How the results of a run scatter over the tolerances its project states: the stock size, tool
offsets, runout, the material batch (specific cutting force, hardness, flow stress, conductivity),
clamping force, and any other `uncertainty.vary` key. A study draws a sampling plan (Latin
hypercube or Monte Carlo), writes one project variant per sample, runs every variant through the
simulation (or any runner with the same call), and aggregates: the distribution of every inspected
dimension with its capability against the tolerance, the probability of every finding by category
and program location, the probability that the run is not safe, the scalar results (cycle time,
energy, wear, cost) and the sensitivity ranking of the factors by rank correlation. Not a domain: no
bus records, no state; seeded streams only (section 9).

## 1. Use

```python
from cnctwin.uncertainty import study, run_study, plan_study, aggregate, html_section, study_page, main

doc = study(project_path, out_dir, options=None, runner=None, progress_cb=None, base_run_dir=None, db=None)
plan = plan_study(project_path, out_dir, options=None, db=None)   # draws + variants only (plan.json)
result = run_study(request, out_dir, progress_cb)                   # API job entry (docs/api.md)
html = html_section(doc, heading_level=2)                            # the report's section
```

* `options` (the API's request options; they override the project's `uncertainty` section, which
  overrides the template's): `runs`, `method` (`lhs` | `monte_carlo`), `seed` (default the project's),
  `vary` (merged over `uncertainty.vary`), `vary_seed` (default false), `runner`
  (`module:function` or `/path/file.py:function`), `simulate_options` (passed to every run),
  `keep_runs` (default true), `attach` (default true: a copy of study.json and uncertainty.html goes
  to `<base_run_dir>/uncertainty/`, where the run's report finds it), `plan_only`.
* Runner: `run_project(project_path, run_dir, options, progress_cb)` (docs/api.md "Entry points");
  `options["runner"]`, else the API's `CNCTWIN_ENTRY_SIMULATE`, else `cnctwin.core.loop:run_project`.
  A runner that is not in the build raises `RunnerMissing` (a `TwinError`) after the plan is written.
* Command line: `python -m cnctwin.uncertainty PROJECT --out DIR [--run RUN_DIR] [--runs N]
  [--method lhs|monte_carlo] [--seed S] [--runner SPEC] [--vary-seed] [--plan-only] [--json]`
  (exit 0; 1 on an input error or a missing runner).
* API: `POST /api/runs/{id}/uncertainty {options}` runs `cnctwin.uncertainty:run_study` with
  `request = {run_id, run_dir, project_id, project_path, project_dir, options}`; it returns
  `{format, study, page, samples_ok, samples_failed, verdict, factors}` and writes the study folder.

## 2. Factors and their distributions

The stated tolerances are varied by default: the stock tolerance (`stock.tolerance_mm`) and the
material batch spreads (`material.batch`); `uncertainty.vary` (and the options' `vary` on top) adds the
spreads only it states, replaces a batch spread, or switches a factor off (`stock_size: false`, a
spread of 0 or false). Distributions follow the information a spread carries (JCGM 100:2008 4.3.7,
JCGM 101:2008 6.4): a **tolerance** (bounds only) is rectangular, a **1-sigma spread** is normal,
truncated at `normal_truncation_sigma` (3) and never below `min_scale_factor` (0.1).

| factor | value (SI) | distribution | stated by | applied in the variant |
|---|---|---|---|---|
| `stock_size[i]` | stock size i (m) | U(size - t, size + t) | `stock.tolerance_mm[i]` | `stock.size_mm[i]` |
| `length_offset[T<n>]`, `radius_offset[T<n>]` | offset error (m) | U(-t, t) | `vary.tool_offsets_um` | `length_offset_mm`/`radius_offset_mm` + error (else the wear offset) |
| `runout[T<n>]` | TIR (m) | U(max(0, r - t), r + t), r the project's (0 if none) | `vary.runout_um` | `runout_um` |
| `kc_scale` | specific-force scale | N(1, s) on [max(0.1, 1 - 3s), 1 + 3s] | `kc_scale_sigma` | inline material (section 4) |
| `hardness_scale` | hardness scale | as above, s = pct / 100 | `hardness_sigma_pct` | inline material |
| `flow_stress_scale` | flow-stress scale | as above | `flow_stress_sigma_pct` | inline material |
| `conductivity_scale` | conductivity scale | as above | `conductivity_sigma_pct` | inline material |
| `clamping_force[<op>.<j>]` | clamping scale | U(max(0.1, 1 - rel), 1 + rel) | `vary.clamping_force_rel` | `input_torque_Nm` or `pressure_bar` times the scale; else options |
| `coolant_reach_scale` | reach scale | U(max(0.1, 1 - rel), 1 + rel) | `vary.coolant_reach_rel` | options only |
| another `vary` key | `*_sigma[_pct]`: scale, normal; `*_rel`: scale, uniform; else U(-v, v) in its SI unit | | `vary.<key>` | options only |

Tool factors exist for every `tools.list` entry and every row of a CSV tool library (`T<n>@<path>` on
paths other than 1); clamping factors for every workholding entry of every operation (each clamp is
tightened independently). Factor values are SI; the variant writes them in the file's units.

Consequences: a tolerance is treated as a hard bound with every value inside equally likely, the
conservative reading of a tolerance without a stated distribution (a normal with the bounds at
3 sigma has 0.58 times its standard deviation); a batch spread beyond 3 sigma is not sampled
(0.27 % of the probability; the sampled standard deviation is 1.3 % below the stated sigma); factors
are independent (no stated correlation; a harder batch with a higher specific force must be stated
by the user as a larger kc spread).

## 3. Sampling plans

For n samples and each factor j (McKay, Beckman & Conover 1979; Helton & Davis 2003):

* **Latin hypercube**: `u_ij = (pi_j(i) + v_ij) / n` with `pi_j` a random permutation of 0..n-1 and
  `v_ij` uniform in (0, 1): exactly one sample in each of the n equal-probability strata of every
  factor; the strata of different factors are paired at random.
* **Monte Carlo** (JCGM 101 7): `u_ij` independent uniform in (0, 1).
* Values `x_ij = F_j^-1(u_ij)`: uniform `low + u (high - low)`; truncated normal
  `mu + sigma Phi^-1(Phi(a) + u (Phi(b) - Phi(a)))` with `a, b` the standardised bounds (scipy
  `ndtr`/`ndtri`).
* Unit draws are 53-bit `k / 2^53` moved to `(k + 1/2) / 2^53`: strictly inside (0, 1), so no inverse
  CDF is evaluated at 0 or 1.
* Each factor has its own stream (`cnctwin.core.rng.stream(seed, "uncertainty.<method>.<factor>")`,
  PCG64 with SeedSequence): the same seed gives the same plan on every platform, and adding or removing
  a factor leaves the other factors' draws unchanged.

**Error of the estimates** (Stein 1987; McKay et al. 1979): with Monte Carlo the standard error of a
mean is `sigma / sqrt(n)` and the relative standard error of a standard deviation about
`1 / sqrt(2 (n - 1))` for a normal output (16 % at n = 20, 7 % at n = 100); Latin hypercube sampling
removes the variance of the additive main effects from the mean asymptotically, so for the nearly
additive response of tolerance stacks its mean is far more precise (the tests see 5x less scatter of a
mean than random sampling at n = 200), and it is never worse than Monte Carlo for monotone responses.
Random pairing leaves spurious correlations between factors of order `1/sqrt(n)` (Helton & Davis
2003), which the partial rank correlation removes (section 6).

## 4. Variants

One project file per sample (`variants/v<nnnn>/project.yaml`), checked with
`cnctwin.project.load_project`, holding every sampled value the project format can hold, so the
simulation needs nothing special to honour them:

* file paths (programs, CAD, stock mesh, tool library) become absolute; a CSV tool library is
  inlined into `tools.list` (and the library key cleared) so its tools can be varied;
* stock sizes, tool offsets (the geometry offset when the project gives one, else the wear offset),
  runout, clamping input torque or pressure are replaced in place;
* the material batch makes the material an **inline object** that `extends` the database material
  (docs/formats.md "extends") and repeats only what is scaled: every stress of `flow_stress`
  (`*_MPa`, `*_Pa`, `*_N_mm2`, tables included) by kc x flow-stress scale; the Kienzle and mechanistic
  coefficients (stresses `*_N_mm2` and edge forces `*_N_mm`) and `cutting.indentation_coeff_N_mm3`
  by the kc scale; every number of `physical.hardness` by the hardness scale;
  `physical.conductivity_W_mK` by the conductivity scale. Lists (Kienzle, mechanistic) are repeated
  whole, mappings merge key by key; the scaled values carry a note naming the sample and the scales
  and no source (inline values report `src: project`). A material given as a path or inline object is
  scaled the same way.

  Why the flow stress: the twin's forces come from Oxley's model with the Johnson-Cook flow stress
  (cnctwin.mechanics); scaling A and B by s scales the whole flow-stress curve, and Oxley's
  forces at a fixed shear angle are proportional to the shear flow stress, so the specific cutting
  force scales by s to first order (the thermal softening of the hotter cut changes it by a fraction of
  the scale change). The Kienzle and mechanistic coefficients are scaled too, for every model that
  reads them.
* the project seed is kept in every sample (common random numbers: the stochastic models draw the
  same chipping, scatter and noise in every sample, so the spread and the sensitivities show the
  tolerances, not the model's own noise); `vary_seed: true` gives every sample its own seed from the
  stream `uncertainty.variant_seed`, which adds the model's scatter to the distribution;
* every sampled value, and separately the factors the file cannot hold, reach the run as
  `options["uncertainty"] = {sample, samples, method, seed, factors: {name: value},
  options_only: {name: value}}`.

**Options-only factors** (`coolant_reach_scale`, clamping of a clamp given without torque or pressure,
unknown `vary` keys) have an effect only when the simulation reads
`options["uncertainty"]["options_only"]`; the study lists them, and the report says so.

## 5. Running and the study folder

```
plan.json                       settings, factors (with distributions), every sample's draws and values
variants/v<nnnn>/project.yaml   the variant      variants/v<nnnn>/run/   its run directory
variants/v<nnnn>/result.json    what the run said (sha256 of the variant project + the sample result)
study.json  uncertainty.html    the results (format cnctwin-uncertainty) and their page
```

* A variant whose result.json names the same project sha256 is not run again: a cancelled study (the
  API's progress callback raises `Cancelled`, a BaseException that passes through) continues where it
  stopped, with the same results.
* A variant whose run raises an exception is a **failed sample**: recorded (index, error, run folder),
  excluded from the statistics, counted and listed in study.json and the page. A finished run that
  fails its checks (findings, a ConservationError recorded as a failed run) is a result, not a failure.
* Progress: `progress_cb(progress, phase="variant i/n", done, total, message)` before and during each
  variant (the run's own `sim_time_s`, `est_total_s`, `eta_s`, `op` pass through), then
  `phase="aggregating"`.
* A variant's results are read from its run directory with the report's reader (docs/report.md
  section 3): inspection.json characteristics (measured, deviation, limits, pass), findings
  (category, severity, op, program, line), the verdict, and the scalars cycle time, removed mass,
  energy per part, tool cost per part, `VB_max_m[T<n>]`, `life_used_fraction[T<n>]`, `Ra_m[<feature>]`.
* The run the study is made around (`base_run_dir`) is read the same way: its value of each dimension
  is shown with the distribution ("this run").

## 6. Statistics

With `c = confidence_level` (0.95) and `z = Phi^-1((1 + c) / 2)`:

* **Distribution** of each dimension (measured value) and scalar: n, mean, standard deviation (n - 1),
  minimum, maximum, the quantiles `percentile_levels` (0.135 %, 2.5 %, 5 %, 50 %, 95 %, 97.5 %,
  99.865 %) by linear interpolation between order statistics (Hyndman & Fan 1996, definition 7), and
  the interval of the mean `mean +- t((1 + c)/2, n - 1) s / sqrt(n)`.
* **Capability** (Kotz & Johnson 1993; AIAG SPC 2005; ISO 22514-2:2017), for a size with limits LSL
  and USL: `Cp = (USL - LSL) / (6 s)`, `Cpu = (USL - mean) / (3 s)`, `Cpl = (mean - LSL) / (3 s)`,
  `Cpk = min(Cpu, Cpl)`; for a tolerance zone (form, orientation, location, run-out: no nominal, the
  lower bound 0 is natural) only `Cpk = Cpu`. Intervals: `Cpk +- z sqrt(1/(9 n) + Cpk^2 / (2 (n - 1)))`
  (Bissell 1990), `Cp sqrt(chi2((1 - c)/2, n - 1) / (n - 1)) .. Cp sqrt(chi2((1 + c)/2, n - 1) / (n - 1))`.
  Quantile indices for any distribution (ISO 22514-2): `Cp_q = (USL - LSL) / (X99.865 - X0.135)`,
  `Cpk_q = min((USL - X50) / (X99.865 - X50), (X50 - LSL) / (X50 - X0.135))`. Expected fraction out of
  tolerance for a normal distribution `Phi((LSL - mean)/s) + 1 - Phi((USL - mean)/s)`. Capable when
  `Cpk >= capability_target` (1.33). Not reported below `min_samples_capability` (10) samples.
* **Probabilities** (a dimension out of tolerance, at least one finding of a category or at a program
  location, not safe to run): `k / n` with the Wilson score interval (Wilson 1927)
  `(p + z^2/(2n) +- z sqrt(p (1 - p)/n + z^2/(4 n^2))) / (1 + z^2/n)`, exactly 0 and 1 at the ends. A
  dimension is out of tolerance as inspection.json judges it (its `pass`), else against its limits.
* **Sensitivity** of each output (every dimension, every scalar, every finding category and the verdict
  whose probability is strictly between 0 and 1, as 0/1 outcomes): for each factor the Spearman rank
  correlation (Spearman 1904; average ranks for ties) with the two-sided p-value of
  `t = r sqrt((n - 2)/(1 - r^2))` on n - 2 degrees of freedom (Zar 1972), significant when
  `p < 1 - c`; and the partial rank correlation (Iman & Conover 1980; Marino et al. 2008): the Pearson
  correlation of the residuals of rank(x_j) and rank(y), each regressed (least squares with an
  intercept) on the ranks of all other varied factors, with its t test on `n - 2 - (k - 1)` degrees of
  freedom, reported when that is at least `prcc_min_dof` (3). Factors are ranked by |rank
  correlation|; a constant factor or output gives no correlation.

## 7. Validity and error bounds

* A study propagates the **stated input tolerances** through the simulation; the simulation's own
  model errors (docs/approximations.md and each model doc) are not part of the distribution. With
  `vary_seed` the stochastic model scatter is.
* Sample sizes: the probability of an event that never occurred in n samples is below
  `1 - (1 - c)^(1/n)` (the Wilson upper bound is close: 16 % for n = 20, 3.7 % for n = 100 at 95 %); a
  study of 20 runs cannot show a scrap rate below a few percent. Cp and Cpk assume a normal, stable
  distribution: their relative standard error is about `1/sqrt(2 (n - 1))` (16 % at n = 20, 7 % at
  n = 100); Bissell's interval is a large-sample normal approximation whose coverage falls below the
  nominal level for small samples, one reason why fewer than 10 samples give no index. The quantile indices need about
  `1 / 0.00135 = 740` samples for the 0.135 % quantile to be estimated rather than extrapolated: below
  that it lies between the smallest order statistics and the index is optimistic for heavy tails.
* Rank correlations measure monotone effects: an effect that changes direction inside the range (a
  stability lobe crossed by the speed scatter) can show a small rank correlation although it matters;
  the scatter plots of study.json's `factor_samples` against the sample results show it. Zar's t test
  is adequate for n >= 10. Partial rank correlations near +-1 for a factor whose rank correlation is
  small mean it acts in the shadow of a larger one (tested: a clamping effect 7x smaller than the
  runout effect has a rank correlation within noise and a partial rank correlation below -0.5).
* The sampled values are exact to the double precision of the inverse CDFs (scipy `ndtr`/`ndtri`);
  the truncated-normal mean and standard deviation of each factor are reported with scipy's
  `truncnorm` and checked against the samples in the tests.

## 8. Parameters (db/models/uncertainty.yaml)

| parameter | value | source |
|---|---|---|
| `normal_truncation_sigma` | 3 | est (JCGM 101 6.4.7 lets a truncated normal stand for a bounded quantity) |
| `min_scale_factor` | 0.1 | est |
| `confidence_level` | 0.95 | jcgm-100-2008 |
| `percentile_levels` | 0.00135, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99865 | iso-22514-2-2017 |
| `capability_target` | 1.33 | aiag-spc-2005 |
| `prcc_min_dof` | 3 | est (Marino et al. 2008 advise n well above k) |
| `min_samples_capability` | 10 | bissell-1990 |

Runs and method default to the project's `uncertainty` section, else db/_templates/project.yaml (20
runs, `lhs`).

## 9. RNG streams

`uncertainty.lhs.<factor>` and `uncertainty.monte_carlo.<factor>` (one per factor name, through
`cnctwin.core.rng.stream(seed, name)`), `uncertainty.variant_seed` (the per-sample seeds of
`vary_seed`). Nothing draws inside a simulation's macro loop.

## 10. study.json

`{format: cnctwin-uncertainty, version: 1, project {path, name, sha256}, base_run, settings {runs,
method, seed, vary_seed, keep_runs}, parameters, factors [{name, label, unit, nominal, distribution
{kind, low, high, mean, std, sigma, mean_param}, applies, target, stated_by, index}], not_varied
[{key, reason}], options_only [names], vary, vary_origin, samples [{index, units, values, project,
project_sha256, written, seed, options_factors}], results {samples_ok, samples_failed, failed, verdict
{unsafe {p, low, high, k, n}, safe_samples}, dimensions [{id, kind, nominal_m, lower_m, upper_m,
one_sided, measured, deviation, capability, out_of_tolerance, samples, nominal_run_m}], scalars [...],
findings {by_category, by_location}, sensitivity [{output, kind, ranking [{factor, rho, p, n,
significant, prcc, prcc_p}], prcc_dof}], factor_samples {names, values, indexes}}, nominal,
sample_results}`; JSON with sorted keys, no NaN (null), the same bytes for the same study (paths of
variants relative to the study folder).

## 11. What the simulation must provide

* `run_project(project_path, run_dir, options, progress_cb)` that runs a project file into a run
  directory (architecture 16) and lets `options` pass through; the study's own keys are under
  `options["uncertainty"]`.
* To vary what the project file cannot hold, read `options["uncertainty"]["options_only"]`:
  `coolant_reach_scale` (a factor on the coolant reach fraction), `clamping_force[<op>.<j>]` (a factor
  on the clamping force of workholding entry j of operation op, for a clamp given without torque or
  pressure), and any other key named there.
* inspection.json with `measured_m` and limits per characteristic (docs/report.md section 3); without it
  a study has findings and scalars but no dimensions.

## References

mckay-beckman-conover-1979, stein-1987, helton-davis-2003, iman-conover-1980, marino-2008,
spearman-1904, zar-1972, jcgm-100-2008, jcgm-101-2008, iso-22514-2-2017, aiag-spc-2005, bissell-1990,
wilson-1927, brown-cai-dasgupta-2001, hyndman-fan-1996, kotz-johnson-1993
(db/bibliography/models-uncertainty.yaml).
