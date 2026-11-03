# blade-5axis-ti: 5-axis flank milling of a ruled Ti-6Al-4V blade (Haas UMC-750, G234)

## The job

A twisted, leaned blade of Ti-6Al-4V stands on a 60 x 32 x 20 platform. Its whole side (pressure side,
suction side, leading and trailing edge) is one ruled surface between the hub section (z 0, chord 40,
t/c 0.10, 4 % camber) and the tip section (z 20, chord 36, the same section scaled, turned 8 deg and
shifted 6 mm in -y). The ruled surface continues to the platform at z -7 (root band). `blade_geom.py`
defines the sections (smooth elliptic-type airfoil: LE radius 0.29, TE radius 0.13 at the hub);
`make_cad.py` builds `part.step` and the blank `stock.step` (the same blade with a 2.0 mm fin allowance,
the result of an earlier 3-axis roughing operation).

The blade is flank milled with the side of a D12 R1 end mill, the tool axis along the rulings, in climb,
in four loops: roughing to 1.0 and 0.3 mm stock (T1), finishing and a spring pass on the same path with a
fresh tool (T2). The profile tolerance is 0.08 mm to A-B-C (platform faces), Ra 0.8.

## Toolpath (`make_program.py`, from the CAD)

1. Reads the single B-spline face of `part.step`; it is ruled (degree 1 across the span), so the isolines
   across the span are the rulings; ruling end points at z 0 and z 20 and the surface normals there come
   from the surface itself (6000 rulings around the section).
2. Two-point offset (Bedi et al.): the tool axis passes through `P_hub + (R + s) n_hub` and
   `P_tip + (R + s) n_tip`, then is moved outwards by half of its mid-span overcut so the twist error is
   split (+8.6 um at the ends, -8.6 um at mid-span). The tool tip sits 1.2 mm below the hub contact
   along the axis, so the cylinder covers the whole span and the R1 corner never touches the flank.
3. Resampled to <= 0.3 mm and <= 0.5 deg per block; B and C for the trunnion from the machine file's
   kinematics: `C = atan2(a_y, a_x)`, `B = atan2(|a_xy|, a_z)`; travel, rotary rates (<= 19.4 deg/s) and the
   swept-cylinder deviation are checked; `blade.nc` and `toolpath_stats.json` are written.

Program: `G234 H01/H02` tool centre point control with X Y Z = tool tip in the part frame (G54 = part zero
at rotary zero, the Haas convention), B C axis positions, F the tip feed; `G49` cancels it before each tool
change; `G187 P1/P3` smoothing; approach and retract along the tool axis, 3 mm lead along the normal.

## Machine, setup, tools

* `db/machines/haas-umc750.yaml`: CAT40 8100 rpm, B -35..120 / C continuous trunnion, flood.
* LANG Quick-Point 52 plate 45600 on the platter, LANG Makro-Grip 77 vise 48085-46 (height 65 mm) gripping
  the platform 3 mm deep across its 32 mm width. Workholding frame at the jaw seat, 89 mm above the
  platter: `fixture_pose` (-330, -254, -521) in machine coordinates at B0 C0 (platter centre under the
  spindle at X -330 Y -254, platter top at Z -610). G54 = the stacking axis at the hub section: (-330, -254,
  -494).
* T1 and T2: `helical-59487-mhsv-020-41200-r100` (D12, R1, 4 flutes, LOC 24, Aplus) in
  `haimer-40-720-32` (ER32), stickout 45, L 115.000. Cutting data from the tool file's ISO S rows (Ti-6Al-4V):
  vc 54.9, fz 0.0711 roughing (ae <= 0.4 D), fz 0.0559 finishing (ae <= 0.06 D).

## What the run is meant to show

* Simultaneous 5-axis motion through TCPC: B and C changing every block, the tool tip on the path, the
  rotary axes inside their rates, and the effect of the rotary-axis pivot geometry on the linear axes.
* The geometric limit of flank milling a non-developable ruled surface (+-8.6 um after balancing).
* Thin-wall flank milling: 0.13-0.24 kN normal force on a blade whose tip yields 0.19 um/N, a 14 N/um
  tool: the finishing pass leaves the tip 35-63 um thick, the spring pass brings it to 3-12 um.
* Segmented titanium chips, small wear, flood coolant reach on a tilted tool.

Numbers and bands: `EXPECTED.md` (`expected_calc.py`, `toolpath_stats.json`).

## Files

| file | what |
|---|---|
| `project.yaml` | the project (resolves with `cnctwin.project.resolve`) |
| `blade.nc` | Haas NGC program (written by `make_program.py`) |
| `blade_geom.py` | section definition shared by the CAD script |
| `make_cad.py` -> `part.step`, `stock.step` | blade and blank |
| `make_program.py` -> `blade.nc`, `toolpath_stats.json` | toolpath from the CAD surface |
| `expected_calc.py` | forces and deflections behind EXPECTED.md |

## Notes and limits

* The controller dry-runs the whole program to M30 without alarms (G234 is implemented); its ideal time
  (176.7 s) spreads F over the combined machine axes, the G234 tip feed gives 82 s of loop feed.
* The root band (z -7..0) keeps the fin allowance outside the tool's reach; a root-fillet operation would
  follow. The CAD continues the ruled surface there, so the remaining-material check reports that band.
