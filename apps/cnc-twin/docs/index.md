# CNC Twin documentation

Start with the [README](../README.md): what CNC Twin is, the quick start, the examples and the limits. This
page lists every document, grouped by what you want to do.

## Run it

| document | what is in it |
|---|---|
| [howto-run.md](howto-run.md) | every way to run the twin: the CLI commands, the API, the UI, the examples, checkpoints and resume, the run cache, sweeps, datasets, calibration, uncertainty studies, comparing runs |
| [ui.md](ui.md) | the browser UI: pages, the run page and its tabs, playback, plots, phones, tests |
| [ui-3d.md](ui-3d.md) | the 3-D view: frames, the in-process workpiece, layers, sections, picking, measuring |
| [api.md](api.md) | the HTTP API: endpoints, examples, server-sent events, series downsampling, the job manager; [openapi.json](openapi.json) is the machine-readable form |
| [ml.md](ml.md) | training data from the twin: the run cache, parameter sweeps, datasets and every column, and what the data is (not) good for |
| [report.md](report.md) | the report, the setup sheet and the comparison of two runs |
| [outputs.md](outputs.md) | what a finished run gets afterwards (pictures, video, sound, report) and the studies built on runs |

## Write projects and data

| document | what is in it |
|---|---|
| [formats.md](formats.md) | every file the twin reads or writes: format name, version, fields and migration rule (databases, project, sweep, run directory, frames.bin, keyframes, checkpoints, cache, datasets, calibration, uncertainty, validation, job files) |
| [howto-add-machine.md](howto-add-machine.md) | adding a machine: template, sourcing rules, checks, tests, a worked example |
| [howto-add-tool.md](howto-add-tool.md) | adding a cutting tool (and its holder, substrate and coating), or a project tool library |
| [howto-add-material.md](howto-add-material.md) | adding a workpiece material |
| [howto-add-dialect.md](howto-add-dialect.md) | adding a controller dialect |
| [dialects.md](dialects.md) | the controller: structure, block-time model, the FANUC, Haas, Siemens and Heidenhain dialects, cycles and compensation |
| [../db/_templates/README.md](../db/_templates/README.md) | the rules every database file follows, and the templates |

## Understand the models

| document | what is in it |
|---|---|
| [architecture.md](architecture.md) | the contract: stack, units and names, frames, the clock and the loop levels, the kernel ABI, the bus, determinism, conservation, the run directory |
| [approximations.md](approximations.md) | every approximation with its consequence, error bound and evidence, generated from the model docs, the validation results and the performance figures (`bin/tsafe python -m cnctwin.docs_tools approximations`) |
| [doc-gaps.md](doc-gaps.md) | which required items (equations, parameters and sources, assumptions, validity, error bound, references) each model doc lacks (`bin/tsafe python -m cnctwin.docs_tools check`) |
| [models/loop.md](models/loop.md) | the run loop: wiring of the domains, operations and setups, batches, checks, checkpoints, `twin check`, `twin warmup`, the regression state |
| [models/hotloop.md](models/hotloop.md) | the compiled L0/L1 loop: layouts, tooth passes, refill planes, turning, the driver |
| [models/motion.md](models/motion.md) | planner, interpolation, feed drives, spindles, kinematics, timing |
| [models/tooling.md](models/tooling.md) | tool geometry as the physics sees it: edge elements, envelopes, profiles, properties |
| [models/removal.md](models/removal.md) | the in-process workpiece: tri-dexel and polar fields, subtraction, gating, collisions, loose bodies |
| [models/cutting.md](models/cutting.md) | the force kernel, the coefficient pipeline, the surface memory, the spindle |
| [models/mechanics.md](models/mechanics.md) | flow stress, Oxley's theory and its fallback, edge and flank contact, heat partition, chip form and colour |
| [models/chips.md](models/chips.md) | chip flight, piles, flute packing, flushing, recutting, chip findings |
| [models/thermal.md](models/thermal.md) | heat in tool, holder, spindle, machine and workpiece; thermal damage |
| [models/coolant.md](models/coolant.md) | supply, jets, occlusion, heat transfer, boiling, lubrication, the tank |
| [models/wear.md](models/wear.md) | flank wear, calibration to Taylor life, crater, notch, BUE, chipping, cracking, breakage, life |
| [models/holding.md](models/holding.md) | tool holders, balance, workholding devices, slip, lift, location error |
| [models/structure.md](models/structure.md) | tool-holder-spindle and workpiece modes, thin walls, displacement grids, fixture compliance |
| [models/dynamics.md](models/dynamics.md) | vibration in and out of the cut, chatter detection, stability lobes |
| [models/surface.md](models/surface.md) | roughness, marks, burrs, surface integrity, threads |
| [models/inspect.md](models/inspect.md) | result geometry and inspection against the CAD: deviation, remaining material, GD&T, virtual CMM |
| [models/production.md](models/production.md) | cycle time, operator and takt, batches, energy, consumables, tool cost |
| [models/sanity.md](models/sanity.md) | the static program checks and their rules |
| [models/geom.md](models/geom.md) | the geometry library: transforms, meshes, BVH, STEP import with PMI |
| [models/render.md](models/render.md) | pictures and video of a run |
| [models/sound.md](models/sound.md) | the sound of the cut |
| [models/uncertainty.md](models/uncertainty.md) | uncertainty studies over the project's stated tolerances |
| [models/calibration.md](models/calibration.md) | machine data import, time alignment, calibration and the calibration overlay |
| [models/gpu.md](models/gpu.md) | the optional CUDA paths and how they equal the CPU kernels |

## Trust it

| document | what is in it |
|---|---|
| [validation.md](validation.md) | the validation suite: cases, harnesses, metrics, budgets, calibration dependence, findings for the domain owners |
| [validation-report.md](validation-report.md) | the latest results per quantity and per case (written by `twin validate`) |
| [determinism.md](determinism.md) | what is bit-identical, how it is kept, tolerances across platforms |
| [performance.md](performance.md) | where the time and memory go, per kernel, per level, per run |

## Not written yet

* `docs/convergence.md` (named in architecture.md section 5.3 for the comparison of `lag_macro` 1 and 2).
