# CNC Twin UI (`web/`)

The browser UI of CNC Twin: projects, runs and their results, played back in real time. It is
served by the API at `/` (docs/api.md) and reads nothing but the API; every page works offline
(no CDN, no build step: vanilla ES modules), on a desktop and on a 390 px phone, in light and dark
mode, standalone or behind Plint's Caddy theme injection.

Contents: [pages](#pages) · [the run page](#the-run-page) · [time](#time) · [the 3D view contract](#the-3d-view-contract) ·
[api.js](#apijs) · [plots](#plots) · [colour and units](#colour-and-units) · [phones](#phones) ·
[theme](#theme) · [files](#files) · [decisions](#decisions-and-their-consequences) · [bounds](#bounds) · [limits](#limits) ·
[tests](#tests) · [integration](#integration)

## Pages

Hash routes (the URL path stays `/`, so the API's index.html fallback is never needed and the UI
also works under a path prefix):

| route | page |
|---|---|
| `#/` | projects: the examples (read-only) and the uploads, and the upload of a zipped project |
| `#/projects/<id>` | a project: machine, material, operations (missing programs flagged), tools, every file (text files readable in place, project.yaml shown), **Check** (`twin check` as a job: summary, errors, warnings, problems, scalars, nested results), **Start a run** with options (a JSON object passed to the simulation unchanged, remembered per project), the project's runs |
| `#/runs?project=&status=` | runs, newest first: status, progress and time left while running, verdict, cycle time, findings; pick two finished runs to compare; refreshes every 2 s while a run is active |
| `#/runs/<id>/<tab>?t=<s>` | a run (below); the tab and the playback time are kept in the URL |
| `#/compare?a=<run>&b=<run>` | comparison of two finished runs (`GET /api/compare`, job followed until done) |

## The run page

Always on top: the title, status and actions (project, report, PDF, compare, cancel while queued or
running, resume when cancelled or failed), the **verdict banner** (SAFE TO RUN / NOT SAFE TO RUN with
every reason of run.json, each linked to its finding; while running "verdict so far" from the
findings; a failed run shows the error and its traceback) and, while queued or running, the
**progress** (simulated time of the expected total, percentage, time left from `eta_s` or from the
rate so far, wall time, operation, phase, message, jobs ahead, last checkpoint). Server-sent events
(`/api/runs/<id>/events`) drive the progress, add findings and keyframes as they arrive and, at
`done`, reload the run and rebuild the result tabs.

| tab | what it shows | read from |
|---|---|---|
| Overview | headline tiles (cycle time, findings by severity, characteristics passed, removed mass, energy and tool cost per part, takt), findings by category and severity, every finding (filters: minimum severity, category; time order or worst first; paged by 100) with line, block, quantities in display units, the tool tip in W and the **chain of cause** (expand: `/trace/<id>`, breadth first, each event clickable to its time), the run summary | run.json, findings, inspection, production |
| Playback | the 3D view, the transport, the layer controls, the **DRO**, the **program viewer**, the **plots** (sections below) | scene, frames, timeline, program, series |
| Inspection | the CMM table of every characteristic (kind, fit and datums, nominal, limits, measured, deviation, amount out of tolerance, tolerance used, result, the program lines that made it), per characteristic the probed points with their statistics (range = form seen by the probe), the error budget as signed bars, thread data and every field the table does not interpret; the deviation of the whole machined surface (per-vertex histogram of `keyframes/attr/*.npz`); the part temperature at unload | inspection.json, attributes |
| Surface | Ra and Rz of every feature against the specification, the Ra distribution over the surface, every burr (edge, kind, height, root, line, position) | surface.json, attributes |
| Tools | per tool: VB max against its limit and life used (meters), cutting time, life left, hottest edge, broken, cost per part and edge; VB per tooth; the wear channels over time; **tool deflection** per tool and component with the combined bound, and the deflection share of every characteristic's error budget; tool findings | tools.json, production.json, series, inspection.json |
| Production | machine time, takt, parts per hour, utilisation, energy and tool cost per part; the cycle-time breakdown by category (stacked bar and exact integer ticks; whether they sum to the machine time), per operation, per tool, the 30 longest lines (each a link to its first execution), operator times, program stops; electrical energy by consumer, spindle terms, axes, pumps, conveyor; coolant, air and tool cost per part; the batch, part by part, the assumptions | production.json |
| Stability | the stability lobe diagram of every tool (depth limit over rpm, stable region shaded) with the program's operating points (stable filled, unstable hollow), the operating points table, chatter findings | sld/, findings |
| Coolant | the share of the run with coolant at the cut, reach while on (mean, lowest), every nozzle (keyframes on, flow, speed, reach, origin), the jets at the playback time, the wetted map at a keyframe, shadows, coolant findings, the coolant channels | series, keyframe fields, scene |
| Chips | chip mass per part, removed mass and volume, the chips at a keyframe (in flight, piled, conveyed, highest pile), chip forms by mass (ISO 3685 codes), the **accumulation map**, chip findings and channels | production.json, chips/, keyframe fields |
| Thermal | the temperature map of the part at a keyframe (a 3-D field as its maximum through z), the highest temperature per keyframe, the part temperature at unload (inspection.json, else `thermal.work_unload_mean_K`, else `thermal.work_bulk_mean_K`), hottest cutting edges, thermal findings and channels | keyframe fields, inspection, tools, series |
| Checks | the conservation balances (largest and final residual, tolerance, steps, result), the time check of every path from timeline.json in integer ticks (span, block time, gaps, booked categories, over-booked blocks), the `ledger.*`, `*consistency*` and `*residual*` channels | run.json, timeline, series |
| Uncertainty | the study made around the run (from `uncertainty/study.json` of the run, else the newest uncertainty job of the run): P(not safe to run) with its interval, samples, plan, the factors and their distributions, every dimension (distribution, Cp, Cpk with intervals, capable, P(out of tolerance), histogram of the samples with the limits, nominal and this run), results per sample, findings by category and program location, the sensitivity ranking (rank correlation, partial correlation, significance) with a scatter of the strongest factor; **start a study** (samples, plan, seed, own seed per sample, more options) | study.json, `/api/runs/<id>/uncertainty`, jobs |
| Calibration | the calibration jobs of the run and of its project; of the chosen one the time alignment, every observation set **measured against simulated (prior) and calibrated** on the simulated time axis with its error metrics, the fitted parameters (prior, value, 95 % interval, standard error, flags, written), confounded pairs, what goes into the overlay, the job's files; **start a calibration** (machine data kind and path, channel mapping, targets, simulator, options) | `overlay.json`, `calibration.json` of calibrate jobs, `/api/calibrate` |
| Reports & files | report (HTML, PDF), setup sheet (HTML, PDF), the study page, the video and the sound of the cut (players, range requests), the pictures, the **tool path from above and from the front** (frames.bin over the stock, feed and rapid moves, work zero), the machined part files, full-resolution exports (CSV, Parquet, every column or chosen channels, per tier), every file of the run | files, frames |
| Run info | run identity (project, seed, format, hashes, numba cache key, job and memory cap), clock and L0 strides, versions, threads, platform, every input with sha256, options, the estimated database values used, model approximations listed by the run, the **setup** (machine, stock and material, tools, operations with workholding and work offsets, coolant) and the rest of the resolved project in display units, the channel table, run.json and status.json as written | run.json, channels |

Everything the headless report shows has a place here: report sections 1-2 Overview, 3-4
Production, 5 Tools, 6 Inspection, 7 Surface, 8 Chips, 9 Thermal, 10 Coolant, 11 Stability, 12
Tools (deflection), 13 Checks, 14-15 Run info (approximations listed by run.json; the report reads
docs/approximations.md of the build, which the API does not serve) and estimates, 16 Reports & files
(pictures and tool path), 17 Playback plots plus the channel table, 18 Uncertainty, 19 Run info, 20
every panel says what its run directory lacks ("not recorded", "written when the run ends").

### Playback

* **3D view** (web/js/view3d, its own module; [contract](#the-3d-view-contract)).
* **Transport** (sticky at the bottom of the screen while the playback is in view): play/pause,
  previous/next block, speed 0.01x to 100x, the time readout, and the **scrubber**: blocks coloured
  by their time category, keyframes as ticks, findings as triangles by severity; drag it with mouse,
  pen or finger, or use the arrow keys (shift: tenths), Home, End. Keys elsewhere on the page: space
  plays or pauses, `,` and `.` step a block.
* **Layer controls**: none, temperature, chips, coolant, finish (Ra), deviation, deflection, wear
  (layers the run does not record are disabled), the layer's colour bar and range with where it
  comes from, measure, follow the tool, reset camera, the section plane (normal to x, y or z, offset
  over the stock, flip), what is shown (machine, fixture, tool, chips, coolant, coolant shadows,
  deflected tool), and the picked point (position in mm, normal, the layer value in display units)
  and the measured distance.
* **DRO**: machine coordinates of every axis (mm, degrees; frames.bin values plus each axis's
  `origin`, rotary axes unwrapped with their revolutions; "relative to the stock origin" when the
  header has no origins), the tool tip in work coordinates (`tip_S - work_zero_S` turned into W by
  `stock.pose_W_S`, named by the operation's first work offset), the tool (number, id, diameter),
  spindle speed and direction (difference of the unwrapped spindle angle of neighbouring frames;
  M03 = negative = CW), cutting power, load (`cutting.spindle_load_fraction`), feed
  (`motion.feed_m_s`, else the tip's speed from frames), overrides (channels named `*override*`; "not
  recorded" otherwise), the executed block. Long runs: an overview of at most 20 000 frames plus
  full-resolution chunks of 4096 L1 ticks fetched around the time shown; channel values from an
  8000-point overview, refined around the time when playback pauses.
* **Program viewer**: the program as it was run (`/program/<op>`, else the project's file), line
  numbers, the time each line took over all its executions and the count, the worst finding on each
  line, the active line following playback (scrolling pauses the follow for 1.5 s), the selected
  line with its executions and findings; a click seeks to the line's next execution; go to line;
  operation picker; virtualised rows (tens of thousands of lines).
* **Plots** ([below](#plots)).

## Time

`panels/timebase.js` is the only clock. It advances t by wall-clock seconds since the last animation
frame times the speed (each frame's step capped at 0.25 s of wall time, so a stalled tab does not
jump), stops at the end, and emits `time` at most once per frame. Everything that shows time listens:
the 3D view (`view.setTime(t)`), the DRO, the program viewer, the plots' cursor and the scrubber.
Block stepping uses timeline.json of the chosen path: next goes to the start of the following
block; previous goes to the start of the current block, or of the one before when already there.
Finding links seek to the finding's time and select its line; timeline links (longest lines,
inspection and burr lines, uncertainty locations) seek to the line's first execution. The URL keeps
the time (`?t=`) after a seek or a pause.

## The 3D view contract

The shell imports `createView3D` from `web/js/view3d/index.js` (owned by the 3D view) lazily:

```js
import { createView3D } from './view3d/index.js'
const view = await createView3D(containerEl, { api, runId })   // api: the object of web/js/api.js
```

What the shell calls, and when:

| call | when | arguments |
|---|---|---|
| `view.setTime(t_s)` | after mounting, then on every time change (every animation frame while playing) | simulated seconds |
| `view.setLayer(name \| null)` | layer chip | `'temperature' 'chips' 'coolant' 'finish' 'deviation' 'deflection' 'wear'` or `null` |
| `view.setSection({axis, offset_m, flip})` | section controls | axis `'x' 'y' 'z'` or `null`; offset in metres along the **stock frame S** axis, from 0 to the stock size; flip boolean |
| `view.setMeasure(on)`, `view.followTool(on)`, `view.resetCamera()` | toggles, button | booleans |
| `view.setVisible({machine, fixture, chips, coolant, shadows, tool, deflectedTool})` | after mounting (all keys; shadows and deflectedTool off, the rest on) and on every toggle | booleans, always all seven keys |
| `view.on('pick', cb)`, `view.on('measure', cb)` | once | the shell shows `point_m` (S, metres → mm), `normal`, `layerValue` (SI: K for temperature, m for the rest, wetted 0/1 for coolant) and `distance_m` |
| `view.legend(layer)` (optional; also `layerLegend`, `getLegend`) | layer chip | if it returns `{min, max, unit, map}` (display units) or `{categories: [[colour, label], ...]}`, the colour bar shows that instead of the shell's own range |
| `view.destroy()` | leaving the run page | |

Every call is guarded: a missing method is skipped, a throwing one is reported once in the 3D
panel; the playback, DRO, program and plots never depend on the view. A missing module or a failing
`createView3D` shows a message in the view's place. For a run that has not written scene.json yet,
the view is mounted when it appears.

Colours: the shell's colour bars use `layerColormap(layer, mode)` of `web/js/plot/colormap.js`
(temperature: one-hue orange ramp; chips, finish, deflection, wear: blue ramp; deviation:
blue-grey-red diverging, symmetric about 0; coolant: categories wetted, dry, jet, shadowed flow).
The 3D view should import the same function, or provide `legend()`, so the bar matches the part.

## api.js

`createApi({base, fetch, EventSource})` returns the client; `api` (default export) is the page's
instance (`base` = the page's path prefix). Errors are `ApiError {status, detail, url, body}`
(status 0: the server cannot be reached). The 3D view receives the same object.

| method | endpoint / returns |
|---|---|
| `url(path, params)`, `request(path, {method, params, body, headers, signal})` | URL with query; fetch with error mapping |
| `json`, `text`, `arrayBuffer`, `binary` (`{buffer, headers}`), `post(path, body, {withStatus})`; aliases `get`, `getJSON`, `getJson`, `fetchJSON`, `getText`, `buffer`, `getArrayBuffer`, `fetchArrayBuffer` | generic |
| `events(path, {open, progress, finding, keyframe, done, error})` → `{close, source}` | EventSource, JSON-parsed, closed after `done` |
| `health()`, `projects()`, `project(id)`, `projectFileUrl(id, path)`, `projectFile(id, path)`, `uploadProject(file, name)`, `checkProject(id, options, {waitS})` → `{status, job, result}` | `/api/health`, `/api/projects*` |
| `runs({projectId, status, limit, offset})`, `run(id)`, `createRun(projectId, options)`, `cancelRun(id)`, `resumeRun(id)`, `runEvents(id, handlers)` | `/api/runs*` |
| `findings(id, {severity, category, op})`, `trace(id, eventId, {maxEvents})`, `channels(id, tier)` | results |
| `series(id, {tier, channels, t0, t1, maxPoints, signal})`, `exportUrl(id, {format, tier, channels, t0, t1})` | `/series` (M4 above maxPoints), `/export` |
| `program(id, op)`, `timeline(id, {op, t0, t1})`, `scene(id)` (cached per page) | |
| `frames(id, {t0, t1, maxFrames, signal})` → `{count, stride, header, tick, time_s, fields: {name: {dtype, shape, width, data}}}`; `framesJson(...)` | `/frames` binary decoded with scene.json's header (`decodeFrames`); pass `maxFrames` or `t0/t1` for long runs |
| `keyframes(id)`, `keyframeGlbUrl(id, k)`, `keyframeGlb(id, k)` → ArrayBuffer, `keyframeFields(id, k, names)` → `{k, time_s, file, arrays: {name: {dtype, shape, data}}}` | `/keyframes*` (`decodeArrayDoc` flattens an array) |
| `brick(id, brick, t)` → `{buffer, keyframe, timeS, sourceFile}`, `chips(id, {t, names})`, `jets(id, t)` | `/bricks`, `/chips`, `/jets` |
| `sldList(id)`, `sld(id, tool)`, `inspection(id)`, `surface(id)`, `tools(id)`, `production(id)` | result files |
| `files(id)`, `fileUrl(id, path)`, `file(id, path)` → ArrayBuffer, `fileText`, `fileJson`, `npz(id, path)` → `{name: {dtype, shape, data}}` | `/files` |
| `compare(a, b, {waitS, retry})` → `{status, job, result}`, `startUncertainty(id, options)`, `calibrate({projectId, runId, machineData, options})` | studies |
| `jobs({kind, state, limit})`, `job(id)`, `cancelJob(id)`, `jobEvents(id, handlers)`, `jobFiles(id)`, `jobFileUrl`, `jobFileJson`, `jobLogUrl`, `waitJob(id, {onProgress, signal})` → `{job, result}` (events, polling when they fail) | `/api/jobs*` |
| `dbKinds()`, `dbItems(kind)`, `dbItem(kind, id)` | `/api/db*` |
| aliases for the view: `getScene`, `getFrames`, `getKeyframes`, `fields`, `glb`, `runFileUrl` | |

Exported helpers: `decodeFrames(buffer, header)`, `parseNpy(bytes)`, `parseNpz(bytes)` (stored and
deflated members, zip64), `inflateRaw(bytes)` (DecompressionStream('deflate-raw'), else
`inflateRawSync`, an RFC 1951 decoder tested against zlib), `halfToFloat`, `decodeArrayDoc`,
`queryString`, `encodePath`, `pageBase`. 8-byte integers come back as Float64Array (exact to 2^53),
float16 as Float32Array, booleans as Uint8Array, strings as arrays; Fortran-ordered arrays are
returned in C order.

## Plots

`web/js/plot/timeplot.js` (own canvas code, no charting library):

* **Data**: the series API at `max_points = 4 x plot width` (at most 20 000), for the visible range
  (+2 %), re-fetched 0.22 s after a zoom or pan. Above that the API returns M4 (first, last, minimum
  and maximum of each bucket, Jugel et al. 2014): drawn as a 1 px line with one bucket per pixel
  column it rasterises like the full series, so peaks are never smoothed away. Below it the rows come
  as recorded (2 px lines, markers when fewer than 40 points are in view).
* **Envelopes**: a fast (L0) channel is drawn from its tick-tier `.min/.mean/.max` columns: the band
  from the minimum to the maximum of the L0 samples in each L1 tick, the line their mean. M4-reduced
  envelope columns (each column picks its own samples) are re-bucketed on one grid of the server's
  buckets: bucket minimum of `.min`, maximum of `.max`, mean of `.mean`. When the view spans 1 s or
  less, the fast tier's own L0 samples (recording windows) are drawn on top.
* **Gaps**: lines break at null/NaN and where the time step exceeds 3 x the median step (raw) or 2.5
  bucket widths (M4): fast windows and missing parts are not bridged.
* **Columns**: one colour per column index in the fixed palette order (colour follows the column,
  not its rank); labels from the channel doc's trailing "(x, y, z)" list, "tooth n" for per-tooth
  channels, else `[i]`; all-NaN padding columns are left out; more than eight columns fold into one
  band (lowest minimum to highest maximum, with the mean), as in the report. Display units by the
  channel's SI unit and size; `_fraction`/`_frac` channels in %.
* **Interaction** (the same on every plot; all plots share range and cursor): tap or click sets the
  time; drag or a one-finger sideways swipe pans (vertical swipes scroll the page: `touch-action:
  pan-y`); ctrl/cmd + wheel or a trackpad/two-finger pinch zooms around the pointer; shift + wheel
  pans; double-click resets; keyboard on a focused plot: arrows pan, + / - zoom, 0 resets; buttons
  zoom in, zoom out, whole run; "keep the cursor in view" pans with the playback while zoomed. A
  crosshair tooltip lists every series at that time (on touch for 2.5 s after a tap).
* **Table view**: per plot the minimum, mean, maximum and value at the cursor of every column.
* **Picker**: every channel of channels.json, grouped by domain, with unit, shape, tier and doc;
  defaults, all, none; remembered in localStorage.
* While a run runs, plots grow with it and re-fetch at most every 3 s.

Other charts (`plot/charts.js`, `plot/bars.js`): XY plots (stability lobes, scatter, tool path with
equal axes), histograms with labelled marks, heat maps of 2-D fields with their colour bar and a
per-cell tooltip, colour bars, bar lists (signed for error budgets and correlations), one stacked
bar with its legend, meters. Every chart has a table of its numbers.

## Colour and units

* **Categorical** (`plot/colormap.js`, `css/tokens.css`): the report's eight hues in their fixed
  order (#2a78d6, #eb6834, #1baf7a, #eda100, #e87ba4, #008300, #6250d6, #e34948), stepped for the dark
  surface in dark mode. Validated with the dataviz six-check validator against the Plint surfaces:
  light (#e4e9e5) adjacent CVD ΔE ≥ 9.1, normal-vision ≥ 19.6, four slots below 3:1 contrast (so
  every chart has a table view and a legend: the relief the check demands); dark (#1e2421) CVD ≥ 8.4,
  normal-vision ≥ 19.3, all ≥ 3:1. Time categories have fixed slots (cutting, air cutting, rapid,
  feed, tool change, acceleration, spindle ramp; the rest grey, listed by name).
* **Sequential**: the blue ramp of the report (13 steps), light to dark, reversed in dark mode so
  zero recedes into the surface; temperature takes the second slot's hue as its own one-hue ramp at
  the blue ramp's OKLab lightness and chroma (gamut-clipped). **Diverging**: blue - grey (#f0efec /
  #383835) - red, symmetric about zero. Interpolation in OKLab.
* **Severity and status** use Plint's tokens (accent, warn, bad, info) and always come with their
  word and an icon.
* **Display units** (`plot/units.js`, the report's rules of `cnctwin/report/fmt.py`): lengths in µm
  below 1 mm, mm below 1 m; °C; W/kW; J/kJ/kWh; rpm; mm/min (m/min from 1 m/s); mm³/cm³/l; mg/g/kg;
  Pa/kPa/MPa; l/min for flows; fractions in %; 4 significant figures, plain between 1e-4 and 1e9,
  U+2212 minus, a dash for a missing value; named quantities by their key's last token.

## Phones

At 390 px: one column; the page never scrolls sideways (tables and long program lines scroll inside
their own box; the test checks every page and tab); form fields at 16 px (no zoom on focus); 44 px
touch targets on coarse pointers; the transport stays at the bottom while scrolling the playback;
touch scrubbing and touch play/pause; plot values on tap; nothing is hover-only (tooltips repeat
values that the tables also give); safe-area insets respected.

## Theme

The Plint Soft UI: one ground colour, raised cards, pressed wells, sage accent, light/dark from the
device. `css/tokens.css` reads Plint's `--p-*` tokens with the same values as fallbacks, so the app
looks the same standalone and behind Caddy's injection of theme/plint.css. Fonts: Manrope and
JetBrains Mono when Plint provides them, else the system UI and monospace fonts (the app loads no
font files itself, so nothing 404s standalone). The Plint pill: the app sets
`window.Plint.mount = ['.app-head .pill-slot']` at start and on load, so plint.js places the pill at
the start of the app's header instead of floating it.

## Files

```
web/index.html                 shell markup, header, main, footer
web/css/tokens.css  app.css    tokens (Plint fallbacks, chart colours) · layout and components
web/js/app.js                  router, header state, Plint pill, footer (API health)
web/js/api.js                  API client and decoders
web/js/plot/                   units.js scale.js colormap.js canvas.js timeplot.js charts.js bars.js
web/js/panels/                 dom.js (elements, tables, figures) · timebase.js · rundata.js (run index, FrameStore,
                               ChannelSampler) · fields.js · viewhost.js · transport.js · dro.js · program.js ·
                               findings.js · plots.js · layers.js · playback.js · run.js · overview.js · results.js
                               (inspection, surface, tools) · production.js · stability.js · process.js (coolant,
                               chips, thermal) · checks.js (checks, run info) · files.js · uncertainty.js ·
                               calibration.js · projects.js · runs.js · compare.js
```

## Decisions and their consequences

* **Hash routes**: deep links survive any server or path prefix and need no server support.
  Consequence: the API's client-route fallback is unused; links look like `/#/runs/<id>/playback`.
* **No requests for files a run has not written**: the run's file list is read first and scene,
  channels and timeline are requested only when present. Consequence: no 404 in the console while a
  run starts; panels say "written when the run ends" instead.
* **The shell owns time; the 3D view only follows it** (the contract above). Consequence: playback,
  DRO, program and plots work without the view, and the view cannot drift from them.
* **Data in the API's own units (SI) everywhere; conversion only at display**, by the report's rules.
* **Keyframe field grids**: cnctwin.core.keyframes.FieldGrid documents `values[nx, ny(, nz)]`; the
  report and the synthetic run read 2-D fields as rows = y. The UI takes `[nx, ny]` unless only the
  `[ny, nx]` reading covers the stock with less than one cell of padding; 3-D fields are shown as
  their maximum through z. Consequence: both writers' grids show right side up; a grid over more
  than the stock (chip piles over the fixture) is read by the core contract.
* **Layer colour bars from the run's own data** (sampled: 12 keyframes for fields, 24 bricks for
  attributes) unless the view gives `legend()`. Consequence: the bar's range is the run's range; it
  matches the part when the view colours with `layerColormap` over the same range or provides its
  legend.
* **Tool deflection per tool** follows the report (channels of domain dynamics in m named
  deflection/displacement; samples attributed through path 0's timeline; the bound combines
  component peaks, an upper bound of the magnitude).
* **Charts on canvas, bar charts in HTML**: crisp text and accessible bars; canvases redraw on
  resize and on a light/dark switch. Every chart has a table view.
* **Uncertainty and calibration results are found, not configured**: the run's attached study, else
  the newest uncertainty job of the run; calibrations of the run and of its project. Consequence:
  studies made outside this server appear only when attached to the run directory.

## Bounds

Presentation settings (in the code, like the report's bounds in cnctwin/report; none is a model
coefficient):

| setting | value | where |
|---|---|---|
| points per plot request | 4 x the plot's width in CSS px, 400 to 20 000 | panels/plots.js |
| fast samples drawn over the envelopes | when the view spans 1 s or less | panels/plots.js `FAST_SPAN_S` |
| columns before folding into one band | 8 | panels/plots.js `MAX_COLUMNS` (as the report) |
| re-fetch while a run grows | at most every 3 s | panels/plots.js `REFRESH_S` |
| refetch after zoom or pan | 0.22 s after the last change | plot/timeplot.js |
| DRO frames | overview of 20 000 records, chunks of 4096 L1 ticks (12 kept) | panels/rundata.js |
| DRO channels | 8000-point overview, refined within ±2 L1 ticks (macro tier ±0.05 s) of the time on pause | panels/rundata.js, playback.js |
| longest wall-clock step of one playback frame | 0.5 s | panels/timebase.js `MAX_FRAME_S` |
| playback speeds | 0.01x to 100x | panels/timebase.js `SPEEDS` |
| findings per page | 100 | panels/findings.js |
| program viewer follow pause after a manual scroll | 1.5 s | panels/program.js |
| colour-bar sampling | 12 keyframes (fields), 24 bricks (attributes) | panels/layers.js |
| histograms of per-vertex values | up to 64 bricks | panels/results.js |
| keyframes read for nozzles / maxima | 48 / 40, evenly over the run | panels/process.js |
| tool-path pictures | 6000 frames | panels/files.js |
| table rows before "show all" | 50 (probed points), 100-300 (files, lines) | panels/*.js |

## Limits

* The 3D view is a separate module: without it the playback shows the DRO, program and plots, and a
  message in the view's place.
* Overrides are shown only when the run records channels named `*override*`; the DRO's work
  coordinates assume the work offset is a translation of the stock frame (true for the setups the
  project format describes; a rotated work coordinate system (G68, tilted planes) is not shown).
* Programs: the viewer shows the operation's program and project files; subprogram files that are
  neither are named but not shown.
* Plots list the channels of channels.json; columns a run writes without declaring them are not
  offered (the report plots them).
* Layer colour-bar ranges and per-vertex histograms are sampled on long runs (12 keyframes, 24/64
  bricks), as stated beside them.
* The comparison, uncertainty and calibration pages show what their documents give; their numbers
  are computed by cnctwin.report.compare, cnctwin.uncertainty and cnctwin.mdata, not by the UI.

## Tests

`tests/ui/` (commands in tests/ui/README.md):

* `unit_test.mjs` (node, 86 checks): units and formatting, scales, colour maps (OKLab, ramp
  monotonicity), the API client (query strings, paths, errors, every method the 3D view may use),
  `inflateRawSync` against zlib, `.npy` (float64/32/16, int64, bool, unicode, big endian, Fortran
  order) and `.npz` (stored and deflated), `decodeFrames`, series building (envelopes, M4
  re-bucketing, folding, padding, %), the run index (blocks by time and path, stepping, time per
  line, findings once each), the inspection judgement, grid orientation.
* `ui_test.mjs` (Playwright, Chromium, 1280 x 900) and `ui_phone_test.mjs` (390 x 844, touch), run
  in `mcr.microsoft.com/playwright:v1.48.2-jammy` against `serve_ui.py`: the real API on port 5995
  with synthetic runs (`ui_fixtures.py`) and fake jobs (`fake_jobs.py`; comparisons are real). They
  open every project (upload, check) and every run, start a run and follow it live to its verdict,
  open every tab, check every canvas on a screenshot (not blank), plot every channel, try every
  layer, the transport (1x and 5x against the wall clock, block steps), the 3D view contract (with
  `view3d_stub` when web/js/view3d is absent: every call and argument the shell makes), finding
  links, cause chains, files and exports, the comparison, an uncertainty study and a calibration,
  then the phone: no sideways overflow, 16 px fields, touch scrubbing, 44 px buttons; and no
  console errors anywhere.
* `test_ui_fixtures.py` (pytest): the test data's structure and consistency; with
  `CNCTWIN_UI_TESTS=1` it also runs the browser suite.

## Integration

* **Caddy** (outside this module): a site block `@<name> host <name>.{$DOMAIN}` proxying to the
  API's port with `import theme <name>` and `header_up Accept-Encoding identity` like the other
  themed apps (CLAUDE.md "Look"). plint.js always loads `/_plint/apps/<name>.js` and Caddy injects
  `apps/<name>.css`: add both to theme/apps/ (the app styles itself on the tokens and places the
  pill itself, so they need no rules beyond a comment), or the browser logs two 404s.
* **The 3D view**: implement the calls in the contract table; colour layers with
  `layerColormap(layer, mode)` or provide `legend(layer)`; take `api` methods from the table above
  (they exist; `tests/ui/unit_test.mjs` checks their names). The section offset is in the stock frame
  S, metres, 0 to the stock size. `tests/ui/view3d_stub/index.js` shows the expected use.
* **Run directory writers**: keyframe grids should follow one axis order (FieldGrid says
  `[nx, ny]`; the synthetic run writes `[ny, nx]`). The DRO wants `axes[].origin` in the frames
  header (machine coordinates; without it positions are shown relative to the stock origin) and
  `scene.work_zero_S_m`. The Tools tab's deflection table wants dynamics channels in m named
  `*deflection*`/`*displacement*`; the DRO's override wants channels named `*override*`.
* **API**: the UI uses only the endpoints of docs/api.md; new result files are reachable through
  `/files` and appear in the Reports & files list.
