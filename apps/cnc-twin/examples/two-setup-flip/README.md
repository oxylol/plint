# Two setups with a flip, Al 6061-T6 (Haas VF-2SS)

One part, two operations in the same self-centring vise: OP10 machines the top, the part is
turned over (180 deg about X) and OP20 machines the bottom, locating on the face OP10 made. The
tolerances that matter are the ones that cross the two setups: the 20.00 mm thickness, the
parallelism of the bottom to the top, and the counterbores made from below on holes drilled from
above.

| | |
|---|---|
| machine | `haas-vf2ss`, flood emulsion |
| material | `al6061-t6` |
| stock | 80 x 60 x 25 blank, rolled plate; the finished part is 80 x 60 x 20 (both faces machined, sides as sawn) |
| workholding | LANG Makro-Grip 48120-46 in both setups, 60 mm across the jaws, 3 mm form-fit grip (the blank is pre-stamped at both ends), 60 Nm |
| programs | `op10.nc` (G54), `op20.nc` (G55), written by hand |
| CAD | `part.step` (finished part in the OP10 stock frame, z 2.5..22.5), written by `make_cad.py`; hand tolerances |

## Setups

* **OP10**: blank centred in the vise (`stock_pose` -40, -30, 0); G54 X/Y at the blank's left
  front corner, Z0 at the finished top = blank top - 2.5 (machine Z -487.5). Datums: A = the
  OP10 top face, B = the left end (x stop in both setups), C = the OP10 front face.
* **OP20**: `from_previous: {op: OP10, flip: {axis: X, angle_deg: 180}}`; the part lies on its
  OP10 top face on the jaw steps. The flip turns the OP10 part about X and re-bases the turned
  80 x 60 x 25 blank box at the origin (docs/formats.md "Project file": OP20's stock frame is
  (x, 60 - y, 25 - z) of the OP10 frame); `stock_pose` is only the extra placement of that frame:
  translation (-40, -30, -2.5), no rotation (centred in the vise, the OP10 top, now 2.5 above the
  box bottom, on the jaw steps). Together: W = (x - 40, 30 - y, 22.5 - z) for a point (x, y, z) of
  the OP10 stock frame. G55 X0 = left end, Y0 = the face that was y 60 in OP10 (now in front),
  Z0 = finished bottom (W z 20, machine Z -490.0). An OP10 hole at (x, y) is at (x, 60 - y) in
  OP20; the four holes map onto the same four positions.
* Between the setups (`time_between_ops_min: 2`) the operator unloads, blows the jaw steps clean,
  checks the edge breaks and turns the part.

## Tools

| T | item | holder | stickout / H | data (maker's values for wrought Al) |
|---|---|---|---|---|
| 1 | `helical-59784-mh35al-025-32000` D20 3FL | `haimer-40-720-32` | 40 / 110 | S10200 (640 m/min), fz 0.183 ('Rgh'), ae 7 (0.35 D), ap 2.5: F5600 |
| 2 | `helical-59766-mh35al-025-31000` D10 3FL | `haimer-40-720-32` | 32 / 102 | S12000, fz 0.0965: F3474 (G13 radial step 3 mm) |
| 3 | `osg-62910-nc-lds-10x90` spot drill | `haimer-40-720-16` | 40 / 110 | 80 m/min (51-100), 0.25 mm/rev (0.22-0.36): S2550 F640 |
| 4 | `osg-60568-ex-gdr-6.8` HSS-Co drill, 130 deg | `haimer-40-720-16` | 55 / 125 | 80 m/min (63-100), 0.40 mm/rev (0.34-0.48): S3740 F1500 |
| 5 | `harvey-871045` D10 90 deg chamfer cutter, carbide | `haimer-40-720-16` | 35 / 105 | S12000 (Harvey's 305 m/min at D_eff 4.3 mm would need 22 600 rpm), fz 0.0381: F1830 |
| 6 | `helical-59726-mh35al-020-30600` D6 3FL | `haimer-40-720-16` | 25 / 95 | S12000, fz 0.0356 ('Slot'): F1280 |

## Programs

**OP10** (top): face 2.5 mm with the D20 (zig-zag, ae 7, Z0); circular pocket D30 x 6 at the
centre: helical entry r 1 (pitch 1) to Z-6, then Haas `G13 I6. K15. Q3. Z-6. D02` (climb,
counter-clockwise, radii 6-9-12-15); spot the four holes to a 7.8 mm 90 deg countersink (G82);
drill them 21 mm deep with G83 Q7 (the tip stays 1.5 mm above the blank bottom, inside the 2.5 mm
OP20 removes); 0.3 x 45 deg edge break round the top (tool tip at Z-1.55, centre 2 mm outside the
faces, clockwise).

**OP20** (bottom): face 2.5 mm to the finished bottom (Z0 of G55): this opens the holes; four
counterbores D11 x 6.4 (DIN 974 for M6 cap screws) by helical interpolation with the D6 (r 2.5,
pitch 1, counter-clockwise, then a flat turn) in a local subprogram (`M97 P100`), which also
removes the drill point cone the facing exposes; 0.3 x 45 deg edge break round the bottom.

R planes and approach heights stay above the blank's top (Z3 against Z2.5 of the unfaced blank),
because the programs are checked against the blank.

## What the run is meant to show

* the part carried from OP10 into OP20 with its machined state, turned over by `from_previous`
  and placed by the OP20 stock pose;
* seating on a machined face: the thickness and the parallelism come from what lies between the
  OP10 top face and the jaw steps (clean steps and edge breaks: microns; a chip or burr: its own
  height, see EXPECTED.md);
* a feature made from both sides: the drill point opened by the OP20 facing, the counterbores
  coaxial with holes drilled in the other setup, located through the self-centring vise;
* the effect of the blank size: positions referenced to the side face C move by half the blank
  width error, while the counterbore-to-hole coaxiality does not (centring vise).

## Files

`project.yaml` · `op10.nc` · `op20.nc` · `part.step` · `make_cad.py` · `EXPECTED.md`
