# Expected physical behaviour: blade-5axis-ti

Numbers from `make_program.py` (toolpath_stats.json: geometry of the toolpath, exact for the model) and
`expected_calc.py` (forces and deflections; run both with `../../bin/tsafe python <script>` in this folder).
Force law: Kienzle **kc11 = 2110 N/mm2, mc = 0.22** (ISCAR group 37, `carbide, any coating`, the entry the
twin selects for the Aplus-coated tool); the measured uncoated value (1253) gives about 0.6 x the forces.
Radial/tangential ratio 0.4-0.9.

## 1. Kinematics (exact)

* S1455: vc = pi 12 1455 / 1000 = 54.9 m/min (Helical start value for Ti-6Al-4V, D12). Tip feed F414
  roughing (fz 0.0711), F325 finishing and spring pass (fz 0.0559).
* Tool axis along the rulings, two-point offset balanced at mid-span. Finishing pass: **B 8.5-25.8 deg,
  C -132.9 to -69.2 deg** (no pass through B = 0, so no C singularity); rotary rates at the programmed feed
  **<= 19.4 deg/s** (machine limit 50 deg/s). Machine travel used: X -408.9..-353.8, Y -280.7..-228.8,
  Z -401.9..-385.9 (all inside the UMC-750 travels).
* Contact band on the flute: 1.2-23.0 mm from the tip (cylindrical part from 1.0 to 24.0: the R1 corner
  and the shank never touch the flank); lowest point of the tool z -5.7 (platform top z -7).
* The controller's G234 transform reproduces the tool axis of every block in the part frame (dry run:
  line 21 B 8.607 C -129.729 -> axis (-0.0957, -0.1151, 0.9887)).

## 2. Geometric error of flank milling a non-developable ruled surface

Computed in make_program.py: distance of the ruled surface from the envelope of the cylinder (R 6) swept
over neighbouring positions, 21 heights x 600 rulings. Pure two-point offset: 0 at hub and tip, overcut
17.3 um at mid-span (largest on the leading edge, where the normals of hub and tip differ most). Balanced
(axis moved out by half of it): **+8.6 um at z 0 and z 20, -8.6 um at z 10** (band +-0.010 mm). This is
the error of the strategy alone, before deflection and wear.

## 3. Forces (per pass, mean over a revolution, contact band 21.8 mm)

| pass | ae | h_max | torque | power | force normal to the blade |
|---|---|---|---|---|---|
| rough 1 (stock 1.0) | ~0.95 | 0.038 | 4.5 Nm | 0.69 kW | 0.55-0.89 kN |
| rough 2 (stock 0.3) | 0.70 | 0.033 | 3.5 Nm | 0.53 kW | 0.39-0.67 kN |
| finish | 0.30 | 0.0175 | 1.3 Nm | 0.20 kW | 0.13-0.24 kN |
| spring | 0.01-0.06 (what finishing left) | < 0.008 | < 0.5 Nm | < 0.08 kW | 0.03-0.09 kN, edge forces dominate |

Spindle load well below the 122 Nm / 32 kW peak curve. Ti-6Al-4V chips segmented at all passes.

## 4. Deflection and the finished profile

* Tool-holder-spindle: D12 carbide, 45 mm stickout (fluted part as d 0.8 D), ER32 nut d50 over 70 mm,
  spindle modes 70 N/um: **13.7 N/um (band 9-20)**.
* Blade as a cantilever from the platform (z -7), bending about its weak axis (hub I_min = 148 mm4,
  section area 126 mm2, similar sections, E 114.5 GPa), load spread over z 0..20:
  compliance **0.019 um/N at z 0, 0.093 at z 10, 0.187 at z 20**.
* Finishing pass alone: tool 10-18 um + blade 2-5 um (z 0), 12-22 um (z 10), 25-45 um (z 20): the blade
  would stay thick, most at the tip (undercut 35-63 um there). The spring pass on the same path takes
  that leftover with 13 % of the force and leaves **3-12 um** (z 0 .. z 20).
* **Final profile deviation (spring pass + twist error): -0.008 .. +0.025 mm**, growing towards the tip:
  inside the 0.08 mm profile zone (+-0.04). Accept -0.015 .. +0.040.
* Torsion of the blade about the stacking axis and local plate bending at the 0.13 mm trailing edge are not
  in these numbers: they add undercut at the edges (no hand value; the band above leaves 15 um for it).

## 5. Surface

Peripheral flank finish, runout 5 um: per-tooth cusp `fz^2/(8R)` = 0.065 um; the largest-running tooth
leaves a mark every `z fz` = 0.224 mm, 1.04 um high: **Ra 0.2-0.5 um** (spec 0.8).

## 6. Mass, time, wear

* Removed volume stock - part = 45 267 - 41 126 = 4141 mm3 = **18.3 g** of Ti-6Al-4V (part of the fin
  allowance stays in the root band z -7..0, which this operation does not reach: the remaining-material
  check reports it there, nowhere above z 0).
* Tip path: roughing 253.9 mm, finishing 122.8 mm, spring 122.8 mm; at the programmed tip feed:
  **36.8 + 22.7 + 22.7 = 82.2 s of loop feed**. With approaches (F2000 along the tool axis, 1.05 s each),
  2 tool changes (chip-to-chip 3.6 s), rotary positioning and TCPC: **cycle 110-130 s**. The controller's
  ideal dry run gives 176.7 s because it spreads F over the combined machine axes (rotary degrees counted
  as millimetres) instead of the tool tip.
* Tool wear: Taylor for TiAlN carbide in ti6al4v (db, est.: C 170, n 0.3, a 0.3, b 0.1) at vc 54.9 m/min:
  T = 55 min (roughing, fz 0.0711, ap 21.8) and 70 min (finishing) of edge contact; each edge is in
  contact 9 % (rough) / 5 % (finish) of the loop time, i.e. about 0.1 % of its life in this part:
  T1 VB 0.02 -> **<= 0.03 mm**, T2 VB **<= 0.01 mm**.
