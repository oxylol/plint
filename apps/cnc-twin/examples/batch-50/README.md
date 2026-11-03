# batch-50: 50 aluminium brackets, probed every part, with sister tools (Haas VF-2SS)

## Why not the Swiss-type pin

The task offered a turned and cross-drilled brass pin on the Star SR-20RIV (FANUC dialect, guide bush,
bar feed). The Swiss definition cannot express that job yet:

* **Tool positions.** The machine file has no gang tool post stations (`turrets: []`, `tool_changer:
  none`); the gang plate is only described in text (`tooling.turning_tools: front 5, rear 2`, gang pitch
  an estimate). A T call cannot place a tool, and the project format has no Y geometry offset either.
* **Tooling.** The tool database has no Swiss-size turning tools: every turning, grooving, parting and
  threading holder is 25 x 25 mm (PCLNR2525, SVJCR2525, CTER2525, CER2525, the QSP26 blade); the SR-20RIV
  takes 12 mm square shanks. The parting tool and its sister would have to be invented.
* **Workholding and bar feed.** `db/workholding` has no main-spindle collet or guide bush item (KLAMP
  DEL5C is a lathe spindle-nose collet chuck), and the machine file has no bar feeder, so bar loading
  and re-chucking per part cannot be described.

So, as the task allows, this example is the alternative: a small aluminium bracket on the
`haas-vf2ss` with spindle-probe compensation every part and sister tools.

## The job

L-section bracket 60 x 50 x 25 from 6061-T6 flat bar 50 x 25 sawn to 60 (`part.step`): an upright
flange 20 thick over the full height, a foot 8 thick, two d6.8 through holes in the foot. One cycle
(`bracket.nc`) makes one part; `production.parts: 50`.

1. **T20 spindle probe** (the machine's Renishaw OMP40-2): touches the top (Z), the left end face (X)
   and the front face (Y) with G31 and moves G54 to the measured corner (`#5221-#5223`), with M79 after
   each touch. The operator loads the blank by hand; the self-centring vise centres it in Y only.
2. **T1 D20 3-flute end mill** (Helical 59784): the 30 x 17 step along X in five climb passes at
   ae 5.94 / ap 17 (vc 640 = the maker's start value, fz 0.183), then a wall pass at ae 0.3
   (fz 0.0838).
3. **T2 d6.8 HSS-Co drill** (OSG EX-GDR): the two holes, G81, full diameter 2 mm below the foot.

Length offsets are called with `H#3026` (the tool actually in the spindle), so a sister tool uses its
own length.

## Machine, setup, tools

* `db/machines/haas-vf2ss.yaml`, CAT40 12 000 rpm, flood.
* LANG Quick-Point 52 grid plate 45600 on the table, LANG Makro-Grip 77 vise 48085-46 (height 65 mm per
  the LANG data sheet), blank pre-stamped and gripped 3 mm deep across its 50 mm width. Workholding frame
  at the jaw seat, 89 mm above the table, at the table centre: `fixture_pose` (-381, -203, -521) in
  machine coordinates (sibling-example convention, table top at Z -610).
* True G54 at the blank corner (-411.000, -228.000, -496.000); the project's G54 is what the setter left
  (+0.25 / -0.15 / +0.20 off): the probe corrects it on part 1.

| T | db id | holder | notes |
|---|---|---|---|
| 1 | `helical-59784-mh35al-025-32000` D20 3FL LOC 50 | `haimer-40-720-32` ER32 | 57.5 of 60 min life used, VB 0.15 |
| 11 | same | same | sister of T1 (`life.sister_of: 1`), new, set 0.024 longer |
| 2 | `osg-60568-ex-gdr-6.8` | `haimer-40-720-16` ER16 | 12 of 45 min used |
| 12 | same | same | sister of T2, never needed in this batch |
| 20 | machine spindle probe (`probes.spindle_probe`) | - | not in `tools.list`: the project format has no probe tool type; H20 set by G10 L10 |

Operator model (project `production.operator`): load 12 s, unload 6 s, clean 6 s, door 3 s, inspection
45 s every 10th part (and the first article).

## What the run is meant to show

* Probing as part of the cycle: three triggered moves, the work offset written by the program, the hand-set
  error removed on part 1 and nothing to correct afterwards.
* A real production roughing cut (13 kW, 90 % of the S1 rating, 50 um tool deflection) and the wall pass
  that cleans it up.
* Tool life management over a batch: T1 expires inside the batch by its life limit and the control loads
  the sister T11; the part made after the switch differs (no radial wear: the upright is ~0.03 mm
  thinner; the length offset of T11 is used), while T2 never expires.
* Cycle time, operator time, takt and the batch progression (life used, wear, offsets, tool changes per
  part) in production.json.

Numbers and bands: `EXPECTED.md` (calculations in `expected_calc.py`).

## Files

| file | what |
|---|---|
| `project.yaml` | the project (resolves with `cnctwin.project.resolve`) |
| `bracket.nc` | Haas NGC program, one part per cycle |
| `make_cad.py` -> `part.step` | part model |
| `expected_calc.py` | hand calculations behind EXPECTED.md |

## Notes and limits

* T20 is not in the tool list (no probe tool type in the project schema), so the controller warns
  that T20 is not in the project's tool list and the sanity rule `tool-unknown` reports it; the same
  holds for the sibling example `pocket-islands-al6061`.
* Tool life management: the program calls T1 and T2; the sister selection is the tool-life provider's
  (`cnctwin.controller.hooks.register_tool_life`), driven by `life.limit_min`, `used_min` and
  `sister_of`. On a real NGC this is Advanced Tool Management with a tool group; the group call (T1000 ...)
  is not used because the twin's tool table has no group numbers as tools.
* In this build G81 has no cycle provider (TW0001); with the cycle expanded by hand and the probe checks
  removed the program dry-runs to M30 in 29.33 s.
