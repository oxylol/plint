# Pocket with two islands, Al 6061-T6 (Haas VF-2SS)

A 3-axis job of the kind a VMC runs every day: an 84 x 64 x 10 mm pocket with two islands (a
rounded rectangle and an obround, both 18 x 32 mm with their tops 4 mm below the stock top) in a
100 x 80 x 30 mm block of 6061-T651 plate. The stock top is measured with the spindle probe
first, the pocket is roughed at full depth with trochoidal (constant-engagement) loops, and the
floor, island tops and walls are finished with a second end mill.

| | |
|---|---|
| machine | `haas-vf2ss` (Haas NGC dialect), 12 000 rpm, flood coolant through the two flex nozzles |
| material | `al6061-t6` (ISO N; Kienzle kc1.1 800 N/mm2, mc 0.25, carbide) |
| stock | 100 x 80 x 30 block, rolled plate |
| workholding | LANG Makro-Grip 77, 48120-46 (self-centring, 46 mm jaws, 0-120 mm), 80 mm across the jaws, 3 mm form-fit grip on pre-stamped flanks, 60 Nm (14 kN) |
| program | `op10.nc`, written by `make_program.py`; contains the probe macros O9810/O9811 after the main program |
| CAD | `part.step` (AP242, mm), written by `make_cad.py`; tolerances by hand in `project.yaml` (no PMI) |

## Setup

* Stock frame = CAD frame: origin at the lower-left-bottom corner of the block, z up, top at z 30.
* The vise sits on the table centre (workholding frame at machine X -381, Y -203.2, jaw seat at
  Z -510, i.e. 100 mm above the table top; the seat height is an assumption, LANG publishes no
  body height in the database item). The stock is centred in the vise (`stock_pose` -50, -40, 0).
* G54 X/Y are the stock's lower-left corner (-431.0, -243.2). G54 Z was touched off by hand
  0.20 mm too high (-479.800 instead of -480.000); the probe cycle in N10 measures the top and
  corrects it. Without the probe the pocket would come out 10.20 mm deep (out of tolerance).

## Tools

| T | item | holder | stickout / H | use |
|---|---|---|---|---|
| 20 | machine spindle probe (Renishaw OMP40-2, 6 mm ball, from `haas-vf2ss` `probes`) | - | H20 = 125.000 by G10 L10 | stock top |
| 1 | `helical-59766-mh35al-025-31000` D10 3FL LOC 25, uncoated | `haimer-40-720-32` ER32 CAT40 | 32 / 102.000 | trochoidal roughing |
| 2 | `helical-59744-mh35al-020-31200` D12 3FL LOC 24, uncoated | `haimer-40-720-32` | 34 / 104.000 | floor, island tops, walls |

Cutting data are Helical's wrought-aluminium values from the tool database: fz 0.0965 mm (D10,
side milling "Rgh") and 0.0686 mm (D12, "Slot"/"Fin"). Helical's starting speed is 640 m/min;
the VF-2SS tops out at 12 000 rpm, so both tools run spindle-limited at 377 and 452 m/min with the
chart chip loads (F 3474 and 2470 mm/min). Radial and axial engagements stay inside the maker's
limits: trochoids ap 9.8 mm (<= 2 D), equivalent ae <= 2.8 mm (<= 0.4 D); island tops ae <= 3.5 mm
(D10) and 4.0 mm (D12); walls ae 0.3 mm (<= 0.06 D), ap 10 mm (<= LOC).

## Program

1. **N10 probe** (T20): `G10 L10 P20` sets the probe length, protected positioning (`G65 P9810`,
   G31 with M78) to 5 mm above the top, `G65 P9811 Z0. S54.`: a fast touch (F600), back off
   1.5 mm, the measuring touch at F60, `#5223 = #5223 + (#5063 - Z)`. The macros are
   Renishaw-style (written here; Renishaw's own Inspection Plus macros are not reproduced).
2. **N20 roughing** (T1): helical entry at B (r 2.7 mm, pitch 0.79 mm, ramp 2.7 deg) to Z-9.8, then
   counter-clockwise trochoidal loops (r 2.7 mm, 1.0 mm advance per loop, two exact G03 semicircles
   per loop) along the channel centre lines B-C-D-E-F-A-B-E. Every channel is 16 mm wide, so the
   swept width 10 + 2 x 2.7 = 15.4 mm leaves 0.3 mm on the walls everywhere. A clockwise contour
   at the allowance around each island removes the corner wedges the loops leave; the island tops
   are taken down 3.8 mm by climb raster passes. 0.2 mm is left on the floor and the island tops.
3. **N30 finishing** (T2): the floor at Z-10 along the centre lines (a 12 mm slot 0.2 deep) with a
   short excursion into each pocket corner; island tops at Z-4 (climb raster); the walls as
   tool-centre paths with wear compensation (`G41 D02`, D02 = 0 in the offset table, so a wall can
   be trimmed with a radius wear value): pocket counter-clockwise, islands clockwise (climb with
   M03), tangent R2 lead-in/out arcs, ae 0.3 mm, ap 10 mm. The wall passes also take the 2 mm floor
   strips along the walls.

`make_program.py` checks the roughing on a 2-D material map (0.05 mm pixels): engagement angle
mean 38 deg, 95th percentile 60 deg, maximum 64 deg (ae equivalent 1.28 / 2.50 / 2.81 mm), no
pixel cut inside the 0.3 mm allowance, < 0.5 mm2 of material left outside it.

## What the run is meant to show

* the probe cycle as the control runs it: two G31 skips, the latched position, the work offset
  written by the macro, and the depth that follows from it;
* constant-engagement roughing: steady spindle power and forces along the loops (no full-slot
  spikes after the helix), chip load and chip thinning at low engagement, the 12 000 rpm limit;
* finishing accuracy: wall position from tool deflection and runout, floor and island-top depth
  from the corrected offset and the thermal state, island-top flatness, the floor finish of a
  square end mill (corner and end-dish kinematics);
* mass bookkeeping: 46 798 mm3 (126.4 g) of chips from the CAD difference.

## Files

`project.yaml` · `op10.nc` · `part.step` · `make_cad.py` · `make_program.py` · `EXPECTED.md`

Regenerate: `cd apps/cnc-twin && bin/tsafe python examples/pocket-islands-al6061/make_cad.py && bin/tsafe python examples/pocket-islands-al6061/make_program.py`
