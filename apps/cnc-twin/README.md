# CNC Twin

CNC Twin runs a CNC program on a physical twin of the machine before the real machine runs it, and says
whether the program is safe to run and why. It reads the program in the control's own dialect, executes it
the way the control and the machine would (look-ahead, block timing, axis drives, spindle), cuts the stock
with the real tool geometry, and follows what the cut does: forces, vibration and chatter, heat, tool wear,
chips, coolant, the workholding and the finished part's geometry against its drawing. Every finding names
the program line, the simulated time, the tool position and the chain of causes behind it. The verdict is
**SAFE TO RUN** when no finding has severity error or fatal (docs/architecture.md section 11).

Everything is SI inside, offline, deterministic (same project, build and hardware give bit-identical
results, docs/determinism.md), and every number in the databases carries a source or is marked as an
estimate with a note on how it was derived. How good the predictions are is stated per quantity in
[Limits](#limits): read it before acting on a number.

What a run gives: the verdict and its reasons, findings with cause chains, the machined part (meshes, a
virtual CMM against the CAD tolerances), surface finish and burrs, tool wear and life, cycle time and
production figures, time series of every channel, a 3-D playback, an HTML/PDF report and setup sheet, a
video and the sound of the cut. All of it is files in one run directory (docs/formats.md "Run directory").

## Quick start

**On Plint (the service).** The stack is `stacks/twin/compose.yaml`: one container (`cnc-twin`, image
`plint/cnc-twin:local` built from this folder), the API and the web UI on `127.0.0.1:5210`, projects and runs
in `data/cnc-twin`, one simulation at a time, 1.5 GB memory cap, 2 CPUs.

```
sudo install -d -o 1002 -g 1002 /opt/plint/data/cnc-twin   # once: the container runs as uid 1002
plint up twin                                              # builds the image the first time (it compiles every kernel)
plint logs cnc-twin
```

The UI is at `https://twin.<domain>` (Caddy proxies to port 5210; the home screen has a card). Open an
example or upload a zipped project folder, press **Check**, then **Start a run**, and follow it live; the
verdict, findings, playback, inspection, tools, report and files are tabs of the run page (docs/ui.md).
After a change to this folder, `plint sync` rebuilds the image.

**Command line.** `bin/twin` runs the CLI from the virtualenv `~/.venvs/cnc-twin` (`CNCTWIN_VENV`), or from
the Docker image when there is no virtualenv, with the deterministic environment either way.

```
cd /opt/plint/apps/cnc-twin
bin/twin warmup                                                 # compile every kernel once (about 12 CPU minutes)
bin/twin check examples/pocket-islands-al6061 --out /tmp/pocket-check   # timestep, cost estimate, program checks
bin/twin run examples/pocket-islands-al6061/project.yaml --set max_sim_s=8
bin/twin explain runs-data/runs/<run_id>                        # the findings; add a finding id for its causes
```

`twin run` prints the run directory, then `SAFE TO RUN` (exit 0) or `NOT SAFE TO RUN` with the reasons
(exit 3). Runs go to `DATA_DIR/runs/` (`DATA_DIR` defaults to `runs-data/` here, `/data` in the container)
and through the run cache: the same project, inputs, options and build are answered from the cache.
`max_sim_s` stops after that much simulated time; a whole program of minutes takes hours (see Limits).
Every command, option and file: [docs/howto-run.md](docs/howto-run.md).

## What is modelled

| domain | what it does (model doc) |
|---|---|
| controller | interprets FANUC 0i-F, Haas NGC, Siemens 840D sl and Heidenhain TNC 640 (Klartext, ISO) programs with macros, cycles, compensation, probing, alarms and the block-time model ([dialects.md](docs/dialects.md)) |
| motion | look-ahead planner, interpolation, feed drives and spindles as servo loops, kinematic chains with geometric and thermal errors, the timeline ([motion.md](docs/models/motion.md)) |
| tooling | cutting edges of every tool as discretised elements, envelopes, thread and insert profiles, tool properties ([tooling.md](docs/models/tooling.md)) |
| removal | the in-process workpiece as a tri-dexel field (polar field for turning), subtraction, gating, collisions, loose bodies ([removal.md](docs/models/removal.md)) |
| cutting | forces, torque and heat per edge element and step from the surface the previous tooth left, process damping, ploughing, the spindle ([cutting.md](docs/models/cutting.md)) |
| mechanics | Oxley / Johnson-Cook chip formation, edge and flank contact, heat partition, chip form and colour (library) ([mechanics.md](docs/models/mechanics.md)) |
| chips | chip flight, piles, flute packing, flushing, recutting and chip hazards ([chips.md](docs/models/chips.md)) |
| thermal | heat in tools, holders, spindles, axes, structure and workpiece; thermal growth and surface damage ([thermal.md](docs/models/thermal.md)) |
| coolant | pumps, jets and their occlusion, heat transfer and boiling, lubrication, the tank ([coolant.md](docs/models/coolant.md)) |
| wear | flank and crater wear, notch, built-up edge, chipping, cracking, breakage, tool life ([wear.md](docs/models/wear.md)) |
| holding | tool holder clamping and pull-out, balance, workholding forces, part slip, lift and location error ([holding.md](docs/models/holding.md)) |
| structure | tool-holder-spindle and workpiece modes and compliance, thin walls, displacement grids ([structure.md](docs/models/structure.md)) |
| dynamics | tool-workpiece vibration in and out of the cut, chatter detection, stability lobe diagrams ([dynamics.md](docs/models/dynamics.md)) |
| surface | roughness by kinematic profile simulation, marks, burrs, surface integrity, thread accuracy ([surface.md](docs/models/surface.md)) |
| inspect | the finished part against the CAD: deviation map, remaining material, gouges, GD&T by a virtual CMM ([inspect.md](docs/models/inspect.md)) |
| production | cycle time breakdown, operator and takt, energy, coolant and tool cost per part ([production.md](docs/models/production.md)) |
| sanity | static checks of the program before it runs: offsets, feeds, coolant, clearances, cutting data ([sanity.md](docs/models/sanity.md)) |
| report, render, sound | the report and setup sheet, pictures and video, the sound of the cut ([report.md](docs/report.md), [render.md](docs/models/render.md), [sound.md](docs/models/sound.md)) |
| studies | uncertainty over the stated tolerances, calibration from machine data, run comparison ([uncertainty.md](docs/models/uncertainty.md), [calibration.md](docs/models/calibration.md)) |

The loop that steps them all, the clock and the checkpoints: [docs/architecture.md](docs/architecture.md),
[docs/models/loop.md](docs/models/loop.md), [docs/models/hotloop.md](docs/models/hotloop.md).

## Examples

Each folder in `examples/` is a complete project (`project.yaml`, programs, `part.step`) with a README and an
`EXPECTED.md` of hand-calculated bands.

| example | machine and control | what it shows |
|---|---|---|
| `pocket-islands-al6061` | Haas VF-2SS, Haas NGC | probing the stock top, trochoidal roughing and finishing of a pocket with two islands in 6061-T6 |
| `thin-wall-al7075` | Haas VF-2SS | two 1 mm walls: wall and tool deflection, wall modes against the tooth frequency |
| `two-setup-flip` | Haas VF-2SS | two operations with the part turned over (`from_previous`), tolerances across the setups |
| `drill-tap-plate-1045` | Haas VF-2SS | spot, peck drilling with through-spindle coolant, chamfer and rigid tapping in AISI 1045 |
| `batch-50` | Haas VF-2SS | 50 brackets with probing every part and sister tools over the batch |
| `sanity-mistakes` | Haas VF-2SS | a program with one of each mistake the sanity checks know (not meant to run) |
| `blade-5axis-ti` | Haas UMC-750, G234 TCPC | simultaneous 5-axis flank milling of a Ti-6Al-4V blade |
| `die-h13-ballnose` | DMG DMU 50, Heidenhain TNC 640 | 3+2 hard milling of an H13 die cavity with ball-nose tools |
| `ti-slot-throughtool` | DMG DMU 50, Siemens 840D sl | a Ti-6Al-4V slot with 40 bar through-tool coolant |
| `lathe-shaft-4140` | Haas ST-20Y | a shaft between chuck and live centre: roughing, finishing, grooving, threading, part-off |

The examples' README "Notes and limits" sections were written when parts of the controller were missing;
where one says a dialect, a cycle provider or a project key is missing, check docs/dialects.md and
docs/formats.md first.

## Tests

```
bin/test                  # unit and integration tests, one pytest process per package, through bin/tsafe
bin/test --all            # also the regression projects and the validation suite (slow: hours)
bin/test --ui             # also the browser tests (needs docker)
bin/test unit/thermal     # one package
bin/twin validate         # the validation suite: docs/validation-report.md and validation/results/*.json
```

Python always runs through `bin/tsafe` (a memory-capped cgroup at nice 10, one thread, `PYTHONHASHSEED=0`,
and a Numba cache keyed by every kernel source file). The first test run after a change to a kernel file
compiles the kernels again.

## Where things are

```
bin/twin  bin/test  bin/tsafe    the CLI, the test runner, the memory-capped runner
cnctwin/                         the package: one folder per domain (controller ... surface), core/ (clock, bus,
                                 loop, recorder, checkpoints), project/, db/, api/, report/, sweep/, dataset/ ...
db/                              databases: machines, controllers, tools, holders, materials, tool materials,
                                 coatings, coolants, workholding, model coefficients, bibliography; _templates/
examples/                        example projects
validation/                      cases/ (published experiments), budgets.yaml, results/
web/                             the browser UI (no build step)
tests/                           unit/, integration/, regression/, validation/, ui/
docs/                            architecture, formats, models/, how-tos (index: docs/index.md)
Dockerfile                       the image; stacks/twin/compose.yaml in the Plint repo runs it
```

Start at [docs/index.md](docs/index.md). Adding data: [machine](docs/howto-add-machine.md),
[tool](docs/howto-add-tool.md), [material](docs/howto-add-material.md),
[controller dialect](docs/howto-add-dialect.md).

## Limits

**Accuracy.** The models take their parameters from published sources; nothing is tuned to the validation
references, and no validated quantity meets its error budget yet. Median absolute relative error over the
independent cases (validation/results/*.json as written by `twin validate` on 2026-11-10;
per case: [docs/validation-report.md](docs/validation-report.md)):

| quantity | independent cases (within budget) | median error | 90th percentile | budget |
|---|---|---|---|---|
| cycle time | 2 (0) | 7.0 % | 21.9 % | 5 % |
| stability limit (lobes) | 12 (1) | 12.0 % | 44.8 % | 25 % |
| chip temperature | 1 (0) | 20.2 % | 44.4 % | 20 % |
| drilling torque | 4 (0) | 24.4 % | 82.1 % | 15 % |
| coolant reach | 2 (0) | 25.8 % (10 of 18 classes) | 52.9 % | 15 % |
| chip compression ratio | 3 (0) | 26.4 % | 49.1 % | 20 % |
| cutting force | 17 (0) | 28.5 % | 91.6 % | 15 / 20 / 30 % (main, milling peak, thrust) |
| wall error | 6 (1) | 29.3 % | 58.8 % | 25 % |
| shear angle | 2 (0) | 35.1 % | 48.1 % | 20 % |
| drilling thrust | 4 (0) | 43.9 % | 91.7 % | 20 % |
| roughness | 9 (0) | 46.5 % | 78.1 % | 30 % |
| tool temperature | 5 (0) | 62.9 % | 191.4 % | 15 % |
| tool life (Taylor) | 5 (1) | 87.7 % | 117.3 % | 50 % |
| spindle growth | 4 (0) | 107.4 % | 147.3 % | 30 % |
| chip form | 2 (0) | 5 of 32 classes match | | 25 % |

With the pairing's own measured cutting data (calibration-dependent cases, reported apart): cutting force
16 %, tool life 16 % (2 of 2 within). In practice: geometry, kinematics, cycle time, collisions and removed
volume are the reliable outputs; forces, power and deflection are good for comparing candidates on the same
tool and material; tool life, temperatures, roughness, thermal growth and coolant reach are not a basis for
decisions without calibration against the machine (docs/models/calibration.md, docs/ml.md "Limits").

**Approximations.** Every approximation, its consequence, its error bound and what the bound rests on is in
[docs/approximations.md](docs/approximations.md) (generated from the model docs by
`bin/tsafe python -m cnctwin.docs_tools approximations`); what each model doc still lacks is in
[docs/doc-gaps.md](docs/doc-gaps.md).

**Cost.** With every domain on, a run costs tens to hundreds of CPU seconds per simulated second: 22 for the
first 15 s of examples/pocket-islands-al6061 (mostly probing and moves; the compiled hot loop 0.9 of it, the
slow domains 18), about 70 for examples/thin-wall-al7075 over 52 s, 120-170 in continuous cutting (pocket
helix, lathe roughing) and about 1100 for the spot drill in full engagement (docs/performance.md, docs/models/loop.md
13). A program of minutes takes hours to days on one core: simulate its critical part with `max_sim_s`, cache the
run and play it back. Peak memory 1.7-2.0 GB on the examples; about 15 CPU minutes of compilation after any
change to a kernel file (the image compiles them when it is built).

**Whole runs.** The loop's regression (docs/models/loop.md section 14) runs the first 8 simulated seconds of
each example: at its last recorded state several runs stopped at the first cut with a conservation error or
a model error, and none reached a second operation within the budget. Read that section for the current
state before relying on a full run.
