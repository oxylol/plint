# Machine data import, alignment and calibration (`cnctwin/mdata`)

Recorded data of the real machine (spindle load and power, positions, following error, program
line, tool, alarms, temperatures) is imported into one format, aligned in time with a simulated
run of the same program, compared with it, and used to fit model coefficients by bounded least
squares. The result is a **calibration overlay** (`calibration.yaml`) the project includes. This
package is not a physics domain: it reads run directories (architecture 16) and machine data, and
writes into a job's output folder; it reads and writes no bus record and draws no random numbers
(no RNG streams).

Coefficients and settings, with their sources and the bounds of every fitted parameter: section
"Parameters".

| module | what |
|---|---|
| `log.py` | `MachineLog`: channels, events, units, times, files |
| `mtconnect.py` | MTConnectStreams files, agent client (`/current`, `/sample`, sequences) |
| `opcua.py` | asyncua client (history read, current read, browse), offline export |
| `focas.py` | FANUC FOCAS collector logs (CSV, JSON, JSON Lines) |
| `csvlog.py` | generic CSV with a column mapping |
| `importers.py` | `load_machine_data(spec, base_dir)`, `python -m cnctwin.mdata.importers` |
| `simrun.py` | a run directory as channels and block markers |
| `align.py` | time alignment, resampling, overlays, error metrics |
| `surrogates.py` | the analytic models a calibration fits |
| `fit.py` | bounded least squares, intervals from the Jacobian |
| `targets.py` | what is fitted from which measurement |
| `overlay.py` | the calibration overlay format and how it is applied |
| `calibrate.py` | the `calibrate` job and `python -m cnctwin.mdata.calibrate` |

## MachineLog

One format for every source: channels with a canonical name whose last token is the SI unit
(`spindle.load_frac`, `axis.X.pos_machine_m`, `axis.C.pos_machine_rad`, `temp.spindle_K`), a kind
(`sample`: continuous, interpolated linearly; `event`: holds until the next value) and observations
`(t_s, value)` in time order; text channels (block, program, execution state) hold `object` values;
alarms, messages and conditions are `MachineEvent(t_s, kind, code, text, severity, source)`.
`t_s` is float64 seconds since `epoch_ns` (integer ns since 1970-01-01T00:00:00Z; None for a source
with relative times). Float64 seconds keep 0.1 us resolution over 30 years of epoch offset, far below
any machine time stamp.

| channel | kind | meaning |
|---|---|---|
| `spindle.speed_rad_s`, `spindle.speed_cmd_rad_s` | sample, event | speed magnitude (unsigned), commanded |
| `spindle.load_frac` | sample | load meter: motor output over its reference output |
| `spindle.power_W`, `.torque_Nm`, `.current_A` | sample | electrical input power, torque, current |
| `spindle.growth_m` | sample | axial (tool tip) growth measured along the tool axis |
| `spindle.shaft_power_W` | sample | shaft power of cutting (simulated runs) |
| `spindle.override_frac`, `path.feed_override_frac`, `path.rapid_override_frac` | event | overrides |
| `path.feed_m_s`, `path.feed_cmd_m_s` | sample, event | path feed, programmed feed |
| `path.x_work_m`, `.y_work_m`, `.z_work_m` | sample | tool control point in work coordinates |
| `axis.<A>.pos_machine_m` / `_rad`, `pos_work`, `pos_relative`, `pos_cmd`, `pos_stock`, `distance_to_go` | sample | axis positions |
| `axis.<A>.following_error_m`, `load_frac`, `feed_m_s`, `force_N`, `torque_Nm`, `current_A` | sample | servo |
| `program.name`, `.main_name`, `.line`, `.sequence`, `.block_count`, `.block`, `.execution`, `.part_count` | event | program state |
| `controller.mode`, `controller.estop`, `tool.number`, `alarm.count` | event | control state |
| `temp.<where>_K`, `diag.<n>[.<axis>]`, `modal.<letter>` | sample, event | temperatures, raw diagnoses, modal words |

Absolute times are ISO 8601 (`YYYY-MM-DDTHH:MM:SS[.f up to 9 digits][Z|+hh:mm]`, iso-8601-1-2019);
a time without a zone is read as UTC and `log.meta["timezone_assumed"] = "UTC"` says so
(consequence: a local-time log is off by the zone offset, which the alignment absorbs as part of
its offset). Values convert to SI by their stated unit (`cnctwin.db.keyunits` tokens and unit texts
such as `%`, `mm/s`, `rpm`, `degC`, `kW`, `mA`); a unit that does not convert to the channel's SI
unit is an error, never guessed.

Files: `machine_log.json` (format `cnctwin-machine-log` version 1: source, epoch, channel list,
events, meta) and `machine_log.parquet` (long table `channel, t_s, value, text`, zstd); both
byte-identical for the same log. A log holds at most `max_observations` (mdata-import, 2e7, about
320 MB); a larger source is refused with a request to import a time window (`start`, `end`).

## Import

`load_machine_data(spec, base_dir)` with the project's `machine_data` section (or the API's
`machine_data` object): `{kind: mtconnect | opcua | focas | csv | log, path, mapping, start, end}`;
`paths: [...]` reads several files and merges them on one time base. `path` may be an MTConnect
agent (`http://host:5000/<device>`) or an OPC UA server (`opc.tcp://host:4840`). Mapping values are
text (the project schema declares `mapping` as text values): `"<column or node id> [unit]"`.

### MTConnect (ansi-mtc1-4-2018, mtconnect-2-2-2023)

Files: one MTConnectStreams document, several concatenated (a recorder appending responses), or a
directory of `*.xml`; observations are de-duplicated by `(instanceId, sequence)`. Documents with a
DTD or entities are refused (MTConnect uses neither; this closes the entity-expansion attack of
untrusted XML). Agent: `/current` gives the buffer state (`instanceId`, `firstSequence`,
`lastSequence`); `/sample?from=n&count=c` requests are chained by `nextSequence` until the
`lastSequence` seen at the start. A `from` that the buffer has overwritten (`OUT_OF_RANGE`, or a
response whose `firstSequence` is past `from`) is recorded in `meta["gaps"]` and the collection
continues at the oldest sequence; an agent restart (new `instanceId`) is recorded and the
collection continues from the new buffer. Requests: `mtconnect_count` per request, at most
`mtconnect_max_requests`, `http_timeout_s` each.

Mapping of types (values are in the standard units the agent reports): Linear `Position`
(ACTUAL / COMMANDED, MILLIMETER) -> `axis.<n>.pos_machine_m` / `pos_cmd_m`; Linear `Load` (PERCENT)
and `AxisFeedrate` (MILLIMETER/SECOND); Rotary `Angle` (DEGREE) -> `axis.<n>.pos_machine_rad`;
Rotary spindle `RotaryVelocity` / `SpindleSpeed` (REVOLUTION/MINUTE; COMMANDED and PROGRAMMED to
`spindle.speed_cmd_rad_s`), `Load`, `Wattage`, `Torque`, `Amperage`, `RotaryVelocityOverride`;
Path `PathFeedrate`, `PathPosition` (three values), `Line` / `LineNumber` -> `program.line`,
`LineLabel` -> `program.sequence`, `Block`, `Program`, `ToolNumber` / `ToolAssetId`, `Execution`,
`PartCount`, `PathFeedrateOverride` (RAPID: rapid override); Controller `ControllerMode`,
`EmergencyStop`, `Message` (event); `Temperature` of any component (CELSIUS); every `Condition`
(Normal, Warning, Fault, Unavailable) -> an event with `code` = `nativeCode` (else the condition
type). A Rotary is a spindle when it reports a rotary velocity or `RotaryMode` SPINDLE (the first is
`spindle`, further ones `spindle2`...; `mapping: {spindle: <component>}` chooses); the first Path is
`path` / `program` / `tool`, further ones are numbered. Roles are decided from all observations,
before a time window is cut. `UNAVAILABLE` is a gap. `...TimeSeries` observations are expanded with
`sampleRate`, the time stamp being that of the last sample (mtconnect-2-2-2023); a time series
without a rate is an error. `mapping: {<dataItemId or name>: "<channel> [unit]"}` maps anything
else; unmapped items are counted in `meta["unmapped"]` (nothing is dropped silently).

### OPC UA (opc-10000-11-2022, opc-10000-8-2022, asyncua-2-1-software)

`mapping: {<channel>: "<node id> [unit]"}`. `read_history(url, specs, start_ns, end_ns)` reads the
raw history of each node (HistoryRead, ReadRawModifiedDetails; asyncua follows continuation
points); `read_current` reads each node once; `browse(url, root)` lists the variables below a node
(depth `opcua_browse_max_depth`, at most `opcua_browse_max_nodes`) with node id, browse path, type,
value and engineering unit. Without a unit in the mapping the node's `EngineeringUnits` property
(EUInformation, unitId = the UNECE common code packed into bytes, unece-rec20-2021) gives it;
codes this importer converts: MMT MTR CMT INH CEL KEL WTT KWT P1 C62 RPM NEW KN NU MTS C16 2X AMP
4K VLT SEC C26 MIN HUR DD C81 HTZ KHZ BAR PAL KPA MPA JOU; another code without a stated unit is an
error naming the node. The time of a value is its SourceTimestamp (ServerTimestamp when there is
none, picoseconds included); a value whose status is not Good is a gap, counted per node in
`meta["bad_status"]`.

Offline export (written by `write_export`, read by `read_export`):
`{"format": "cnctwin-opcua-history", "version": 1, "server": url, "nodes": [{"channel", "node_id",
"unit", "values": [[time, value, status], ...]}]}` (status a StatusCode name, `Good...` being good,
or a number whose two top bits are 0); or a long CSV `node_id, time, value[, status][, channel]`
with the mapping giving each node's channel and unit.

### FANUC FOCAS collector logs (fanuc-focas2-reference)

FOCAS is FANUC's proprietary library and is **not used**; collectors built on it write what its
functions return, and these records are read. CSV (UTF-8, header row, `#` comments, comma or
semicolon), one row per returned element:

| column | content |
|---|---|
| `time` | ISO 8601, or a number with `mapping: {focas.time_unit: s / ms / us / ns / epoch_s / epoch_ms / epoch_us / epoch_ns}` (default s, relative) |
| `function` | the FOCAS function name |
| `item` | output of the function: `load` / `speed` (cnc_rdspmeter), `absolute` / `machine` / `relative` / `distance` (cnc_rdposition), `run` / `aut` (cnc_statinfo), the address letter (cnc_modal), the diagnosis number (cnc_diagnoss), `main` (cnc_rdprgnum) |
| `index` | 1-based axis / spindle number |
| `name` | axis or spindle name (cnc_rdaxisname / cnc_rdspdlname) |
| `value` | as returned: an integer with `dec` decimals, or a decimal number |
| `dec` | decimal places (LOADELM / POSELM `dec`; empty: `value` is decimal) |
| `unit` | POSELM 0 mm, 1 inch, 2 deg; LOADELM 0 %, 1 rpm; or a unit text |
| `code`, `type`, `text` | alarm number (`alm_no`), alarm type, message; `type: poll` marks a poll with no alarm |

JSON: `{"format": "cnctwin-focas-log", "version": 1, "records": [...]}`, a list, or JSON Lines; a
record is flat (the CSV columns) or `{"time", "function", "data": [elements]}`; a cnc_rdalmmsg poll
with an empty `data` means no alarm is active.

| function | channel |
|---|---|
| `cnc_absolute`, `cnc_machine`, `cnc_relative`, `cnc_distance`, `cnc_rdposition` | `axis.<n>.pos_work_m`, `pos_machine_m`, `pos_relative_m`, `distance_to_go_m` (axes A, B, C: `_rad`) |
| `cnc_rdspload`, `cnc_rdspmeter` load | `spindle.load_frac` (spindle 2: `spindle2.`) |
| `cnc_rdspmeter` speed, `cnc_acts` | `spindle.speed_rad_s` |
| `cnc_actf` | `path.feed_m_s` (mm/min unless `unit`) |
| `cnc_rdsvmeter` | `axis.<n>.load_frac` |
| `cnc_srvdelay`, `cnc_diagnoss` 300 | `axis.<n>.following_error_m` (detection units: `dec` or `unit` required) |
| `cnc_rdseqnum`, `cnc_rdblkcount` | `program.sequence`, `program.block_count` |
| `cnc_rdprgnum` | `program.name` (`O%04d`), item `main`: `program.main_name` |
| `cnc_modal` | item T: `tool.number`; other letters `modal.<letter>` |
| `cnc_statinfo` | run (0 RESET, 1 STOPPED, 2 FEED_HOLD, 3 ACTIVE, 4 MSTR) -> `program.execution`; aut (0 MDI, 1 MEM, 3 EDIT, 4-9 manual modes, 10 remote) -> `controller.mode` |
| `cnc_rdalmmsg`, `cnc_rdalmmsg2` | events: an alarm raised when its code appears in a poll, cleared when it is gone; `alarm.count`. Codes are type prefix + number: 0 SW, 1 PW, 2 IO, 3 PS, 4 OT, 5 OH, 6 SV, 7 SR, 8 MC, 9 SP, 10 DS, 11 IE, 12 BG, 13 SN, 15 EX, 19 PC (`PS0010`) |
| `cnc_diagnoss` 308, 309, 403 | `temp.axis_<n>_motor_K`, `temp.axis_<n>_encoder_K`, `temp.spindle_motor_K` (degC); other numbers `diag.d<number>[.<axis>]` as written |

The diagnosis numbers are those of the 30i / 0i-F maintenance manuals; another control may number
them differently: `mapping: {"<function>[:<item>]": "<channel> [unit]"}` overrides any row.
Unknown functions are counted in `meta["unmapped"]`. Errors name the file and the CSV line.

### Generic CSV

`mapping`: `<channel>: "<column> [unit]"`, `time: "<column> [iso | s | ms | us | ns | min | h |
epoch_s | epoch_ms | epoch_us | epoch_ns]"` (default iso when the first cell is not a number, else
s), options `csv.delimiter` (default the most frequent of `,` `;` tab in the header), `csv.decimal`
(`.` or `,`), `csv.kind.<channel>`, `csv.text.<channel>: true`. Empty cells and `nan` are gaps;
unmapped columns are listed in `meta["unmapped_columns"]`.

## The simulated run (`SimulatedRun`)

From the run directory: `spindle.speed_rad_s` (forward difference of the unwrapped spindle angle of
path 0 per L1 tick; float32 angles give 4e-4 rad/s noise at 1 ms), `spindle.shaft_power_W`
(`cutting.spindle_power_W`), `spindle.load_frac`, `path.feed_m_s`, `axis.<A>.following_error_m`
(`motion.following_error_m[i]`, axis names from the frames header), `axis.<A>.pos_stock_m`
(frames.bin: axis values relative to the stock origin; their velocities are the axis velocities),
`program.line` / `.sequence` (the N word of the block text) / `.block` / `tool.number` at each
block start (timeline.json), and the block markers. Simulated time is `tick * tick_s` from the
start of the run.

## Alignment and its error

One affine map, `t_sim = offset + scale * t_meas` (`Alignment`; `scale` absorbs clock drift).

**Markers** (preferred). Measured markers are the changes of `program.line` (else
`program.sequence`, else the normalised `program.block` text); simulated markers the block starts.
1. *Offset vote*: every pair of equal keys votes for `T_sim - t_meas` (at most 2000 simulated
   occurrences per key); the densest window of `+- max(sim block gap, poll interval) / 2` wins. A
   global dynamic-programming alignment (needleman-wunsch-1970) would cost O(n m) for logs of
   1e4-1e5 blocks; voting is robust to repeated lines (loops, subprograms) and to missed polls.
2. *Matching* in time order: each measured marker is paired with the nearest unused later
   simulated block start of the same key within the window, the provisional `(offset, scale)`
   refitted every 8 matches once they span `scale_min_span_s`.
3. *Robust fit*: Huber-weighted least squares (`huber_k` 1.345, scale MAD x 1.4826; huber-1981)
   with rejection of residuals above `marker_outlier_sigma` (4) robust deviations, iterated. The
   residual scale is floored at `max(poll, sim tick) / sqrt(12)` (jcgm-100-2008): quantised times
   give a MAD of zero, which must not make every other marker an outlier. The scale is fitted only
   with `marker_min_for_scale` (3) markers spanning `scale_min_span_s` (30 s); over shorter spans ppm
   drift is far below the polling quantisation. `|scale - 1| > max_scale_deviation` (1 %) is refused.
4. *Polling correction*: a polled marker lags the true block start by U(0, dt_poll); when at least
   90 % of the marker times lie on the sampling grid of a periodic channel of the log, the offset is
   corrected by `+ scale dt_poll / 2` (the mean lag).

Error (established in tests/unit/mdata/test_mdata_align.py): with n matched markers polled at
dt_poll the offset error is about `dt_poll / sqrt(12 n)` (synthetic run: 18 markers at 10 ms, error
0.4 ms, test bound 1.5 ms); over 600 s with 250 ppm drift, 20 Hz polling and 10 % missed changes the
scale is recovered within 2e-6 and the offset within 10 ms.

**Program start** (no block markers but the execution state): the first ACTIVE edge is the start of
the first simulated block, scale 1, polling-corrected. The end of the cycle is deliberately not
used: the difference between measured and simulated cycle time is what `block_time` calibrates.

**Cross-correlation** (no markers): the measured spindle signal (power or load) and its prediction
with the nominal machine values (`spindle_signal`: shaft power through the spindle model, so the
spindle-on idle step is in both) are resampled on a grid of `max(min(dt_meas, dt_sim), span /
xcorr_max_points)` and the masked normalised cross-correlation is computed at every lag from six
FFT correlations (padfield-2012-masked-ncc), lags with less than `xcorr_min_overlap_frac` (0.5)
overlap excluded. The peak, refined by a parabola through its neighbours, is the delay
(knapp-carter-1976). A secondary peak outside the main lobe above `xcorr_ambiguity_ratio` (0.9)
marks the delay `ambiguous` (periodic passes); a peak below `xcorr_min_correlation` (0.3) is
refused. With span >= `scale_min_span_s` local delays in `xcorr_scale_windows` (4) windows give the
scale by weighted least squares. Error: about a quarter of a grid step for edged signals (synthetic
run at 10 ms sampling: 0.5 ms; test bound 3 ms); 400 ppm drift over 600 s recovered within 5e-5.

**Given**: `alignment: {offset_s, scale}`.

**Resampling** (`resample`): a measured channel at simulated times `t_meas = (t_sim - offset) /
scale`; samples interpolate linearly, events hold; gaps longer than `resample_max_gap_factor` (3) x
the median sampling interval stay gaps. **Overlays** (`align.overlay`, and `overlay.json` of a
calibration): series on one grid in simulated time (`measured`, `simulated`, `calibrated`),
reduced to at most `overlay_max_points` (2e4) by per-bucket first/last/min/max (peaks kept), with
**error metrics** per series: n, bias (mean of predicted - measured), RMSE, MAE, max |error|,
NRMSE (RMSE / range of the measured), R^2 and Pearson r; per matched block the simulated and
measured start and duration; the cycle time measured from the execution state.

## Calibration (`fit.py`)

Parameters theta (SI, bounds from mdata-calibration; strictly positive ones spanning decades fitted
in ln), a callback `simulate(theta) -> {observation: predicted array}` and observation sets
`y_i` with standard uncertainties `sigma_i`. The weighted residual
`r = [(simulate(theta)_i - y_i) / sigma_i]` (finite measured samples only) is minimised by
`scipy.optimize.least_squares` (method trf, branch-coleman-li-1999; `x_scale` from the values and
bounds; `diff_step` and tolerances `diff_step_surrogate` 1e-6 / `tol_surrogate` 1e-10 for analytic
surrogates, `diff_step_loop` 1e-3 / `tol_loop` 1e-6 for full simulations, whose outputs are
quantised by the base tick; `max_nfev_per_param` 100). Sets without a stated sigma are weighted by
their own scatter: after a fit each set's sigma becomes its RMS residual, floored at its resolution
/ sqrt(12) (the smallest non-zero step of its values, jcgm-100-2008), and the fit is repeated
(`reweight_passes` 2), so sets in different units weigh by their actual scatter.

Inference (bates-watts-1988 ch. 2.3, seber-wild-2003 ch. 5): with the weighted Jacobian at the
solution (central differences for surrogates, the solver's for loop runs), `J = U S V'`,
`s^2 = |r|^2 / (m - p)` and `cov = s^2 V S^-2 V'` (singular values below `sv_rel_cutoff` 1e-10 x
the largest dropped); interval `theta_k +- t(1 - alpha/2, m - p) sqrt(cov_kk)` at
`confidence_level` 0.95 (ln-space parameters: exponentiated, asymmetric). Flags: a parameter with a
loading above 0.1 on a dropped direction is **not identifiable** (no interval; it is not written to
the overlay, its prior stays); pairs with `|rho| > correlation_flag` (0.95) are **confounded**; a
parameter on a bound has `at_bound` (its interval is one-sided). The intervals are linearisation
(Wald) intervals: the cost is re-evaluated at both ends of each one and the ratio of the actual to
the linearised increase reported (`ci_check`); outside `[1/2, 2]` (`ci_check_ratio_max`) the
interval is flagged approximate (curvature, bates-watts-1988 ch. 7). For a linear model the
intervals equal ordinary least squares exactly (test_mdata_fit.py, relative 1e-4). Before/after
error metrics of every set and the cost before and after are reported.

## Targets and models: validity and error (`targets.py`, `surrogates.py`)

Each target builds its parameters (prior = the project's / database's value), observation sets
(measured values at their simulated times) and `observe(sim_log, theta, sim_effects)`. Default free
parameters are marked *free*; `target_options.<target>.free: [...]` replaces them and
`.fixed: {name: value}` fixes any at a value.

**spindle** (needs the run and `spindle.power_W` or `spindle.load_frac`). Electrical power, the
production energy model (palmgren-1959 for x = 5/3):
`P_el = s_kc P_shaft,sim / eta + P_idle,max (|w| / w_max)^x + max(J w dw/dt, 0) / eta`.
Load meter (motor output torque over its reference, harris-kotzalas-2007-bearings for the bearing
term): `lambda = (s_kc P_shaft,sim + P_b,max (|w| / w_max)^x_b + J w dw/dt) / P_ref(w)`, `P_ref` the
machine's spindle power curve at the speed (or `load_reference_W`). Optional meter low-pass
`y' = (u - y) / tau` (exact for piecewise-linear input). Parameters: `kc_scale` (*free*; written as
every Kienzle `kc11/kf11/kp11` and mechanistic `K*c/K*e` of the material times the factor),
`spindle_idle_power_W` (*free*, power) or `spindle_bearing_power_W` (*free*, load; written as
`thermal.bearing_heat_at_max_speed_W`), `spindle_efficiency` (fixed), `meter_tau_s` (fixed 0).
**Decision**: in power data `kc_scale` and `eta` enter only as `kc_scale / eta`; the default fits
`kc_scale` and keeps the machine file's efficiency, because handbook kc1.1 values scatter by
+-20 % against a few % for drive efficiencies and the forces drive deflection, chatter and
temperature, not only energy. Consequence: if the efficiency is what is wrong, the calibrated
forces carry its error; with a force measurement free `spindle_efficiency` and fix `kc_scale`.
Validity: `s_kc` multiplies the simulated shaft power exactly as long as the cut does not change
with the forces (deflection feedback on chip thickness is second order: dF/F ~ F/(k h) x dk/k);
`simulator: loop` removes that approximation. Test: kc_scale 1.12 and idle 612 W recovered inside
their 95 % intervals, kc_scale within 1 % (test_mdata_calibrate.py).

**servo** (needs `axis.<A>.following_error_m`, or `pos_cmd` and `pos`): proportional position loop
with velocity feed-forward and an ideal velocity loop (ellis-2012-control-design),
`e' = (1 - K_ff) v_cmd - K_v e`, integrated exactly for piecewise-linear `v_cmd` (from the
measured commanded positions, else the run's command = position + its following error).
Parameters `axis_<A>_kv_1_s`, `axis_<A>_feedforward` (*free*, written to
`machine.axes.<A>.servo.kv_1_s` / `feedforward`). Validity: velocity-loop bandwidth well above
K_v (error O(K_v / w_v) in transients); at constant feed only `(1 - K_ff) / K_v` is determined
(reported as confounded). Test: K_v 33 1/s and K_ff 0.7 recovered within 1 % / 0.005 from 0.5 um noise.

**axis_friction** (needs `axis.<A>.load_frac` and positions): the production axis model,
`F = m a + m g u_z + sign(v) (mu N + F_seal)` (armstrong-1994-friction for the Coulomb form),
`N = m g` for horizontal axes (|u_z| <= 0.5), referred to `motor_force_N` (or `load_reference_N`).
Parameters `axis_seal_drag_N`, `axis_friction_coeff` (*free*; written to
`models.production-energy.axis_seal_drag_N` / `axis_guide_friction_coeff`). Samples while the run
cuts are left out (the run records no cutting load per axis). Test: 55 N and 0.008 within 5 % / 10 %.

**block_time** (needs the run aligned by markers): the interpreter / look-ahead queue of the
controller model: `ready_k = max(ready_k-1, end_k-B) + t_block`, `start_k = max(end_k-1, ready_k)`,
`end_k = start_k + m_k` (B = `lookahead_blocks`, m_k the block's motion time: its timeline
duration minus its `data_starvation` ticks when the timeline books them). Observations: the
measured times between consecutive matched markers (scale applied), sigma floor
`sqrt(2) dt_poll / sqrt(12)`. Parameter `block_processing_s` (*free*, written to
`machine.controller.block_processing_ms`, also as `controller_overrides`). Consequence when the
timeline books no `data_starvation`: the simulated durations already contain the nominal block
time, so only blocks that run longer on the machine inform a larger value; simulate the reference
run with a small `block_processing_ms`. Test: 2.4 ms recovered within 1 % from every 7th block
polled at 1 ms.

**thermal_growth** (needs `spindle.growth_m` and the measured speed history): first-order growth
towards `G (|w| / w_max)^x` (x = thermal-machine `bearing_heat_speed_exponent`; bryan-1990,
mayr-2012), exact for piecewise-constant speed; observed `c delta + y0` with `c = 1` (physical
growth, default) or `comp_residual_fraction` (`growth_measured: compensated`). Parameters
`spindle_growth_max_m`, `spindle_thermal_tau_s` (*free*; written to
`thermal.growth_at_max_speed_um`, `time_constant_min`), `spindle_growth_initial_m` (*free*,
nuisance), `growth_offset_m` (free with `absolute: true`). Validity: one dominant thermal time
constant (the spindle head); multi-body heating appears as a structured residual. Test: 32 um and
1200 s within 3 % / 5 %.

**tool_life** (table `tool_life`: `vc_m_min`, `life_min`, optional `feed_mm`, `ap_mm`): extended
Taylor (taylor-1907), `vc (T / T_ref)^n (f / f_ref)^a (ap / ap_ref)^b = C`, residuals in `ln T`
(tool life scatters multiplicatively). Parameters `taylor_C_m_s`, `taylor_n` (*free*), `taylor_a`,
`taylor_b` (free when the table varies feed / depth); written to the material's `tool_life` entry
matching the project tool's class. C is an extrapolation to T = 1 min: its interval is wide and C
and n are confounded; predictions inside the data are good to the scatter (test: 12 % scatter,
predicted life within 10 %).

**usui** (table `wear`: `test`, `t_min`, `VB_mm`, `sigma_n_MPa`, `sliding_m_min`, `temp_degC`; the
conditions of a row hold over the interval ending at it): `VB = VB0 + (cot alpha - tan gamma) int A
sigma_n V_s exp(-B / theta) dt` (usui-1984-tool-wear). The geometric factor: a wear flat at normal
depth W below the edge meets the flank (clearance alpha) at W cot alpha and the rake face (rake
gamma) at W tan gamma, so VB = W (cot alpha - tan gamma) (the wear-land construction of
yen-2004-tool-wear-fem). Parameters `usui_A_per_Pa` (*free*, ln space), `usui_B_K` (*free*); written to
`tool_life_usui[tool_class=<class>,surface=flank]` with `rate_unit: m/s with sigma_n in Pa, V_s in
m/s` (the entry is created when absent). A and B separate only with tests at two or more
temperatures. Test: B within 5 %, A within 30 %.

**deflection** (table `part_measurements`: `feature`, `measured_mm`, `simulated_mm` (else the
run's inspection.json tolerance of that id), `runout_sensitivity`, `deflection_mm`): linear in
runout and compliance, `d = d_sim + c_r (r - r_sim) + d_defl (1 / s_k - 1)`, with `c_r` the
feature's sensitivity to the cutting radius (+2 for a slot width, +-1 for a wall) and `d_defl` the
simulated static deflection contribution to it. Parameters `tool_runout_m` (*free* when a
sensitivity is non-zero; written to `tools[<n>].runout_um`), `tool_stiffness_factor` (*free* when
a deflection is non-zero; written to `factors.tool_stiffness[<n>]`). Validity: deflections small
against the chip thickness (linear superposition). Test: 8 um and 0.8 within 0.5 um / 0.03.

**kienzle** / **mechanistic** (`cuts`: `line` (with the run) or `t0_s` / `t1_s`, `ae_mm`, `ap_mm`,
optional `rpm`, `feed_mm_min`, `diameter_mm`, `teeth`): mean shaft power of each cut from the
measured mean over the cut (`segment_trim_frac` 0.1 dropped at each end), shaft = `eta (P - P_idle)`
for power or `lambda P_ref - P_b` for load, with the spindle values calibrated earlier in the same
request (else the machine file's). Kienzle (kienzle-1952), straight-flute equivalent with
`h = fz sin(phi)` (martellotti-1941; altintas-2012-mfg-automation ch. 2):
`P = n z R ap [kc11 h_ref^mc fz^(1-mc) I(1 - mc) + Kte phi_e]`, `I(p) = int_0^phi_e sin^p phi dphi`
by the regularised incomplete beta function, `phi_e = arccos(1 - ae / R)`; mechanistic:
`P = Ktc ae ap vf + Kte z n ap R phi_e`. The mean over a revolution is exact for any helix at
constant immersion; the circular-path approximation errs by O(fz / 2R) (< 1 % for fz / D < 0.02);
runout changes the mean only to second order. Parameters `kc11_Pa`, `kc_mc` or `Ktc_Pa`,
`Kte_N_per_m` (*free*). mc (and Kte) are determined only by cuts at different chip thicknesses.
Tests: kc1.1 900 MPa within 3 %, mc 0.22 within 0.02; Ktc within 3 %, Kte within 15 %.

## Simulators

* `surrogate`: `observe(nominal run, theta, sim_effects=True)`: the parameters' effect on the run
  is applied analytically (exact for the observation-model parameters, first order for `kc_scale`,
  `K_v`, `K_ff`, `t_block`).
* `loop` (spindle, servo, block_time): every evaluation writes a calibrated project file with the
  run parameters (`kc_scale`, the servo gains, `block_processing_s`) through
  `overlay.calibrated_project`, simulates it with `cnctwin.core.loop.run_project(project, run_dir,
  options.loop_options, progress_cb)`, reads the run and applies only the observation model; runs
  are cached by the run parameters and deleted once read. The alignment found on the nominal run is
  kept. Cost: about `10 (p + 1)` simulations of the program.
* `auto` (default): the surrogate, then, when `cnctwin.core.loop` is in the build and a project
  and run are given, the loop started from the surrogate's solution (one or two iterations usually).
  Targets without a run-based prediction always use their analytic model.

## The calibration overlay (`overlay.py`)

`calibration.yaml`, format `cnctwin-calibration` version 1: `name`, `machine`, `material` (the
objects the values belong to), `inputs` (machine data source, run id and run.json sha256, project
sha256), `alignment`, `set`, `parameters` (every parameter: prior, value, interval, flags,
written or why not), `fit` (per target: cost before / after, observations, dof, s^2, simulator,
confounded pairs). `set` maps target paths to `{value, parameter, ci}` with the value in the unit
of the path's last key (`block_processing_ms` in ms, `growth_at_max_speed_um` in um,
`time_constant_min` in min, `C_m_min` in m/min). Paths:
`machine.<key>[<selector>]...`, `material...`, `tools[<T number>[@<path>]].<key>`,
`models.<model>.<parameter SI name>` (SI value), `factors.<name>[<key>]`; selectors are an index,
`key=value[,key=value]` (creates the item when none matches) or a bare id (`[main]`). Only free,
identifiable parameters are written; values at a bound are written and flagged.

Applying: `apply_to_raw(kind, raw, overlay)` (machine or material raw data: each value becomes
`{value, src: est, note: "calibrated (<name>), parameter <p>, interval [lo, hi]"}`: a calibrated
value is a statistical estimate, and the note names where it came from),
`controller_overrides(overlay)` (the `machine.controller.*` entries, which also go into the
project's `controller_overrides` so they win over any existing override),
`apply_to_project(project_doc, overlay)` (tool entries, controller overrides),
`model_overrides(overlay)` (`{model: {name: SI value}}`), `factors(overlay)`, and
`calibrated_project(project, overlay, out, machine_raw=, material_raw=)`, a standalone project file
using only existing project syntax (`machine: {extends: <id>, <changed sections>}`, lists repeated
whole as `extends` requires, `material: {id: {extends: <id>, ...}}`, absolute file paths). The
calibrated project resolves with the current loader (tested).

## Running a calibration

Job (`POST /api/calibrate`, docs/api.md): `cnctwin.mdata.calibrate.calibrate(request, out_dir,
progress_cb) -> dict`, request `{project_id, project_path, project_dir, run_id, run_dir,
machine_data, options}`. Options: `targets` (default: every target whose data are present;
a requested target that cannot be built is an error), `simulator` (`auto | surrogate | loop`),
`alignment` (`{method: auto | markers | xcorr | given | none, offset_s, scale, key,
poll_interval_s}`), `target_options.<target>` (`free`, `fixed`, `load_reference_W`,
`load_reference_N`, `command`, `growth_measured`, `absolute`, `tool`, `entry`, `clearance_deg`,
`rake_deg`), `spindle`, `name`, `loop_options`, and the tables `tool_life`, `wear`,
`part_measurements`, `cuts` (rows, or CSV files relative to the project folder, column names with
units). Outputs in `out_dir`: `machine_log.json/.parquet`, `alignment.json`, `calibration.yaml`,
`calibration.json` (format `cnctwin-calibration-result`: every target with parameters, intervals,
correlation, confounded pairs, metrics before/after, notes), `overlay.json` (format
`cnctwin-calibration-overlay`: measured / simulated / calibrated series per observation set with
metrics) and `project.calibrated.yaml`. The returned summary is the job's `result.json`.
Progress phases: project, import, run, align, `<target>: fit`, write. Every output is
deterministic (no wall-clock; same inputs give the same bytes, tested).

Command line: `python -m cnctwin.mdata.calibrate --project P --run RUN (--data spec.yaml | --kind K
--path P --map key=value ...) [--target T ...] [--simulator auto|surrogate|loop] [--offset S]
[--options options.yaml] --out DIR` (exit 0, 1 on an input error). Import alone:
`python -m cnctwin.mdata.importers --kind K --path P [--map ...] [--start T --end T] [--out DIR]`,
`--browse opc.tcp://...` lists an OPC UA server's variables.

## Parameters

Coefficients: `db/models/mdata-import.yaml`, `mdata-align.yaml`, `mdata-calibration.yaml`
(read through `cnctwin.db.models.params`), sources in `db/bibliography/models-mdata.yaml`. The
speed laws of the spindle models are read from the models that simulate them, so a calibration
and the simulator agree: `production-energy` `palmgren_speed_exponent` (idle power) and
`thermal-machine` `bearing_heat_speed_exponent` (bearing friction, spindle growth).

| file | parameters | source |
|---|---|---|
| mdata-align | Huber k 1.345, MAD to sigma 1.4826 | huber-1981 |
| | cross-correlation ambiguity ratio 0.9 | knapp-carter-1976 |
| | marker outliers at 4 robust deviations, scale fitted from 3 markers over 30 s, scale refused beyond 1 %, overlap >= 0.5, at most 262144 correlation points, 4 scale windows, minimum correlation 0.3, gaps beyond 3 sampling intervals, overlays of at most 2e4 points | est |
| mdata-calibration | confidence level 0.95 | jcgm-100-2008 |
| | at most 100 evaluations per parameter | virtanen-2020-scipy |
| | singular-value cut-off 1e-10, correlation flag 0.95 | seber-wild-2003 |
| | interval check ratio 2 | bates-watts-1988 |
| | bounds: mc 0-0.7; feed-forward 0-1; growth 0-500 um, thermal time constant 1-1000 min; Taylor n 0.05-1.5; Usui A 1e-24-1e-3 1/Pa, B 10-1e5 K; seal drag 0-5000 N, friction coefficient 0-0.3 | kienzle-1952, ellis-2012-control-design, mayr-2012-thermal-issues, taylor-1907, usui-1984-tool-wear, armstrong-1994-friction |
| | solver tolerances (1e-10 surrogate, 1e-6 loop), difference steps (1e-6, 1e-3), 2 reweighting passes, cut trim 0.1; bounds: efficiency 0.3-0.99, idle power 0-10 x prior (at most 50 kW), bearing power 0-10 x prior (at most 20 kW), kc_scale 0.3-3, kc1.1 and cutting coefficients 0.2-5 x prior, edge coefficients up to 300 N/mm, K_v 1-1000 1/s, block time 0.01-100 ms, growth offset +-5 mm, compensation residual 0-1, Taylor C 0.05-20 x prior, Taylor exponents -1-3, stiffness factor 0.1-10, runout 0-200 um, meter time constant 0-5 s | est |
| mdata-import | MTConnect 1000 samples per request, at most 1e5 requests, time-outs 10 s, OPC UA browse depth 4 and 5000 nodes, at most 2e7 observations | est (request sizes and memory caps) |

The bounds are the validity range of a calibration: a value at a bound is written and flagged `at_bound`
(its interval one-sided).

## Decisions and their consequences

* **Load meter = motor output over the power curve.** FANUC-type load meters show the motor output
  relative to its rated output; the machine file's curve may be another duty rating (S1, S6, peak).
  Consequence: a different reference scales `kc_scale` and the bearing power by the same factor;
  give `load_reference_W` when the meter's 100 % is known.
* **Unsigned spindle speed.** Machine data report the speed magnitude (and the direction as an M
  code); the simulation's signed omega is compared by magnitude.
* **One affine time map per log.** A log spanning a machine restart or a clock jump needs splitting
  (`start`/`end`) into windows aligned separately.
* **Fixed alignment during a fit.** The alignment is found once on the nominal run; a loop
  evaluation that changes block timing (block_time) is compared by durations, which do not depend
  on the offset.
* **Not identifiable means not written.** A parameter the data cannot determine keeps its prior and
  the report says why; a confounded pair is written with its correlation in the report.
* **Wald intervals with a curvature check** instead of profile-likelihood intervals (2p extra
  evaluations instead of tens per parameter); the check says when they are approximate.
* **The intervals cover random measurement error only.** Model error (a simplification of the
  surrogate, an alignment error against a signal that changes within a millisecond) and readings
  quantised coarser than the signal varies (an integer-percent load meter on a 33 kW spindle
  resolves 330 W, while a finishing cut draws 1 kW) make the residuals systematic; the intervals then
  under-cover. Check the residual metrics (bias, R^2) in `calibration.json`, prefer power meters or
  load meters with decimals, and calibrate on cuts that load the spindle.

## Integration (what other parts must do)

* Project schema and loader (`cnctwin/project`, done 2026-11-09): `calibration: <path>` (a
  `cnctwin-calibration` file relative to the project) is applied at resolve time: `apply_to_tools` to
  the tool list (after the library rows joined it), `apply_to_raw("machine", ...)` to the machine after
  the project's `controller_overrides` (a calibrated controller value wins and keeps its note),
  `apply_to_raw("material", ...)` to the material, each patched object checked; the overlay is hashed
  as an input (`input_hashes`, so the project hash, run ids and cache keys change with it). Its
  `models.*` and `factors.*` entries are listed in `ResolvedProject.calibration_overlay` and reported as
  `calibration` warnings while no domain applies them (docs/formats.md "Project file").
  `project.calibrated.yaml` remains the standalone form; for a project that includes an overlay it is
  written with both baked in (`combined`) and without the key.
* `cnctwin.core.loop.run_project`: a project with `calibration:` runs calibrated through the ordinary
  resolver; `options["calibration"]` (an overlay path given per run) is not read.
* `cnctwin.db.models.params` users (production energy): `ResolvedProject.calibration_overlay["models"]`
  (`overlay.model_overrides`) gives calibrated model parameters by model and SI name, to be merged over
  `params(name)` for a calibrated project (not done yet).
* Structure domain: `calibration_overlay["factors"]["tool_stiffness"][<T number>]` should multiply the
  static (and modal) stiffness of that tool's chain at the tip (not done yet: the exact change is in
  docs/models/loop.md section 15, the structure domain being another owner's).
* Motion domain: should record `data_starvation` ticks in the timeline's `ticks` and
  `cutting.axis_force_N[i]` per tick; the block-time and axis-friction targets then use them.

## References

ansi-mtc1-4-2018, mtconnect-2-2-2023, opc-10000-11-2022, opc-10000-8-2022, unece-rec20-2021,
asyncua-2-1-software, fanuc-focas2-reference, iso-8601-1-2019, knapp-carter-1976,
padfield-2012-masked-ncc, needleman-wunsch-1970, huber-1981, jcgm-100-2008, branch-coleman-li-1999,
virtanen-2020-scipy, bates-watts-1988, seber-wild-2003, kienzle-1952, martellotti-1941,
taylor-1907, usui-1984-tool-wear, yen-2004-tool-wear-fem, ellis-2012-control-design,
armstrong-1994-friction (db/bibliography/models-mdata.yaml); altintas-2012-mfg-automation,
palmgren-1959, harris-kotzalas-2007-bearings, bryan-1990-thermal-errors, mayr-2012-thermal-issues
(other bibliography files).
