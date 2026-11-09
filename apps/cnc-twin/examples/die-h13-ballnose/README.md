# die-h13-ballnose: 3+2 hard milling of an H13 (50 HRC) die cavity (DMU 50, TNC 640)

## The job

A die insert of hot-work steel H13 (1.2344) hardened and tempered to 50 HRC, 100 x 80 x 40, with an
elliptic-paraboloid cavity 48 x 32, 9.6 deep, and a convex R2 rim to the parting face. The cavity is
0.8 of the 60 x 40 x 12 of the brief, to keep the finishing time down. The insert was roughed and
semi-finished soft, leaving 0.3 mm on the cavity (`stock.step`: the same insert with a smaller
paraboloid and the same R2 rim), then vacuum hardened, and the parting face and the datum faces were
ground. This operation hard-mills the cavity with two Tungaloy TEB ball-nose end mills: roughing to
0.06 mm plus scallops of up to 0.04, then finishing with a new tool. Both run in two 3+2 halves.

Tolerances, entered by hand in `project.yaml` (A = bottom, B = front, C = left face): cavity profile
0.03 to A-B-C, rim profile 0.05, and Ra 1.6 / Rz 6.3 at the bottom and on the steep side wall
(the scallop sets these).

## Toolpath (`make_program.py`, from the CAD)

1. `part.step` is tessellated with a 2 um chord (cavity, rim and parting face). On the concave cavity
   the facets lie on the tool side of the surface, so the tessellation error can only leave material.
2. There are two halves. The x >= 0 half is cut under `PLANE SPATIAL SPB+15`, with the tool leaning
   towards +x. The x <= 0 half uses `SPB-15`, and the halves overlap by 1 mm. Because the tool always
   leans away from the half it is cutting, the contact sits 14-54 deg from the ball's axis and never
   at the dead centre. Tungaloy p. I054 advises tilting about 15 deg.
3. A ball drop-cutter (vertex, edge and facet tests) runs on the triangles in each tilted working plane.
   The program points are ball tips. Raster lines run along x', zig-zag. A line keeps going while the
   ball touches the cavity or the rim, and then another 0.3 mm onto the parting face, so the R2 rim is
   rolled over completely. The lines span y from one tangent line of the rim with the parting face to
   the other.
4. The stepover holds the scallop height constant: 40 um for roughing, 4.5 um for finishing. The drop
   cutter first samples ball-centre heights on a probe grid (0.25 x 0.5 mm). In the normal section across
   the lines, `h = dc^2 / (8 R) (1 + R k)`, where `dc` is the spacing of neighbouring ball centres
   across the path and `k` is the normal curvature of the centre surface (positive in the concave
   cavity, negative over the convex rim). This gives finishing stepovers of 0.27-0.42 mm. A flat-surface
   formula would leave 7 um scallops at the cavity bottom, where the section radius is 13.3 mm.
5. Points are spaced 0.25 mm along a line. A midpoint is added wherever the chord between two points
   would leave the ball-centre path by more than 0.3 um; over the convex rim such a chord would cut into
   the part. Coordinates have 4 decimals, the TNC 640's 0.1 um input resolution.
6. Spindle speed per tool: the maker's vc is placed at the 80th percentile of the effective cutting
   diameter `D sin(psi + kappa)` (psi = contact angle, kappa = engagement depth on the ball). The feed is
   F = S z fz.
7. Links: 0.5 mm up the tool axis and across at F1500, then down into the cut at the cutting feed. Inside
   the concave cavity and over the flat parting face, a straight link between two lifted positions is
   collision-free.

Program structure (Klartext `die_h13.h`):

* `BLK FORM`, then `CYCL DEF 247` to set preset 1 (centre of the parting face).
* Per tool: `TOOL CALL n Z S F` and `L Z-10 R0 FMAX M91`.
* Per half: `PLANE SPATIAL SPA+0 SPB+-15 SPC+0 TURN MB MAX FMAX SEQ+-` and `CYCL DEF 32`. Roughing
  uses T0.02 `HSC-MODE:1`, finishing uses T0.005 `HSC-MODE:0`. `M3` comes with the first positioning
  and `M8` with the approach, followed by the raster `L` blocks and `PLANE RESET TURN`.
* The end of the program: `M91` retract, `M9`, `M30`.

`check_program.py` is a separate check. It reads the program back, sweeps the balls and measures the
distance to every cavity and rim point of `part.step`. After finishing, the cavity has 0-5.4 um of
material left (p99 4.6) and no gouge (min -0.06 um). The rim band has -1.1 to +5.4 um.

## Machine, setup, tools

* `db/machines/dmg-dmu50-3rd.yaml`: HEIDENHAIN TNC 640, B -35/+110 swivel rotary table with C,
  SK40 speedMASTER spindle (15 000 rpm, 21 kW), optional coolant/air switch and air blast.
  `SPB+15` / `SPB-15` correspond to B +15 / -15 with C 0, both inside the B range.
* Workholding: LANG Quick-Point 52 grid plate 45600 (104 x 104 x 27) on the table centre. Two d16
  clamping studs on the 52 mm grid, screwed into the insert's bottom, pull it down; the stud holes are
  not drawn. The workholding frame is the plate top centre. `fixture_pose` (0, 0, -598) is given in
  machine coordinates at B0 C0, with the table top at Z -625 and X0 Y0 on the C axis. `stock_pose`
  (-50, -40, 0) is the insert's corner. `PRESET 1` (0, 0, -558) is the centre of the parting face.
* T1 and T2: Tungaloy TEB100E4L-20/30C10CFH72 AH710 (D10 ball, 4 flutes, variable pitch, 38 deg helix,
  relieved neck d 9.8 to 30 mm). Each sits in a HAIMER Power Shrink Chuck SK40 40.445.10.3. That chuck
  is entered inline in `project.yaml`: the database has SK40 shrink chucks for d 25 and d 32 only, so
  the data come from the HAIMER shop page with its URL. Stickout is 40 (32 mm of the 42 mm shank
  clamped, the minimum clamping length; a longer stickout runs Z past -475 in the tilted halves).
  L = 105.000. T1 has 6 min of use (VB 0.01), T2 is new.
* Cutting data come from the tool file's rows "hardened steel <= 49 HRC (e.g. h13-50hrc lower end)",
  p. I053:
  * Roughing (vc 70-80, fz 0.025-0.03, ap <= 0.8): vc 75 at the 80th percentile, fz 0.028,
    ap 0.13-0.24. S2590 F290.
  * Finishing (vc 100, fz 0.021-0.026, ap <= 0.1): vc 100 at the 80th percentile, fz 0.024,
    ap 0.06-0.10. S3640 F349.

  At 50 HRC the insert sits on the border between the "<= 49 HRC" rows and the "50-60 HRC" rows
  (30-50 m/min). The project uses the former, as the database's material hint suggests.
* Air blast instead of flood (`coolant.fluid: air`). Tungaloy recommends air blow for hard milling, and
  M8 sends air through the nozzles with the machine's coolant/air switch.

## What the run is meant to show

* 3+2 machining with `PLANE SPATIAL`: the table swivels and the same tilted working-plane coordinates
  apply. The machine file's B pivot errors shift the two halves by +-2.6 um along x, in opposite
  directions. That is at most 1.6 um normal to the surface at the x-ends of the cavity, and nothing at
  the joint, where dz/dx = 0.
* Ball-nose hard milling:
  * Chips are thin: h_max is 4-9 um, close to the 6 um edge radius, so ploughing matters.
  * Kienzle kc is 11.5-14 kN/mm2. Forces are small: mean tangential force is about 10 N in roughing and
    1-2 N in finishing.
  * Power is a few watts.
  * The cutting speed swings with the contact angle: 48-77 m/min in roughing, 57-104 m/min in
    finishing.
* A scallop-limited surface: 4.0-4.5 um across the lines (Ra 1.0-1.2), 0.2 um along them.
* A profile inside +-0.015 with a tool error budget of a few microns (EXPECTED.md).
* Wear: small. The finishing tool's edges are in contact 4 % of the time, so on the twin's wear basis
  VB grows by under 1 um. It would be about 10 um (a 1.4 um gradient on the part) if the Taylor life
  were read as in-cut time.

Numbers and bands are in `EXPECTED.md` (from `expected_calc.py`, `toolpath_stats.json` and
`check_program.py`).

## Files

| file | what |
|---|---|
| `project.yaml` | the project (resolves with `cnctwin.project.resolve`, no problems) |
| `die_h13.h` | HEIDENHAIN Klartext program (written by `make_program.py`) |
| `make_cad.py` -> `part.step`, `stock.step` | insert and hardened blank |
| `make_program.py` -> `die_h13.h`, `toolpath_stats.json` | toolpath from the CAD surface (about 5 min, 4 processes) |
| `check_program.py` | swept balls of the program against `part.step` |
| `expected_calc.py` | forces, deflection, wear, scallop and energy behind EXPECTED.md |

## Notes and limits

* The controller's HEIDENHAIN Klartext dialect (`heidenhain_klartext`, docs/dialects.md) runs the program:
  its ideal dry run (no acceleration) reaches `END PGM` with no alarm in 1155.0 s, 1137.5 s of it in feed
  (38 207 commands; checked 2026-11-09 with the controller's dry run under bin/tsafe). The static program
  checks (cnctwin.sanity) read only FANUC and Haas syntax themselves, so they report one `scan-limits`
  finding for this program and no other rule. The travel check in `make_program.py` uses the machine file's
  kinematics, where the B pivot is assumed to be 120 mm below the table top: X +-50.4, Y +-16.9,
  Z -468.9..-453.0 (limit -475).
* Machine time is about 19.5 min (333 s roughing and 804 s finishing in feed), well above the
  5-minute guidance. Hard finishing a die cavity to Ra 1.6 with a D10 ball takes this long. For a
  shorter run, simulate one tool or one half: the blocks from `TOOL CALL 2` are 13.6 min, and one
  finishing half is about 6.8 min.
* The cavity is scaled to 0.8 of the brief, and the material sits on the maker's hardness border
  (see above).
