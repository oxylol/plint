# Drilled and tapped plate, AISI 1045 (Haas VF-2SS)

A hole-making job in steel: a 160 x 80 x 16 mm plate of 1045 with 6 x M8x1.25 on a 50 mm bolt
circle and 4 x M10x1.5, all tapped through. Spot drill (which also countersinks the M8 holes),
solid carbide drills with internal coolant in G83 peck cycles with through-spindle coolant, a
90 deg chamfer mill for the M10 countersinks, and rigid tapping (G84) with spiral-point taps.

| | |
|---|---|
| machine | `haas-vf2ss`; flood (M08) and the optional 300 psi through-spindle coolant (M88, 21 bar) that the machine file lists (`coolant.pumps[tsc]`, `option: true`) |
| material | `aisi1045` (ISO P, 190 HB; Kienzle kc1.1 1700 N/mm2 / mc 0.25 for coated carbide, 1672 / 0.32 uncoated) |
| stock | 160 x 80 x 16 flat bar (hot rolled), sawn |
| workholding | 4 x `carrlane-strap-m12` (M12 studs in the table T-slots, 47.5 Nm, 9.79 kN each) on the plate ends, the plate on two 25 mm parallels |
| program | `op10.nc`, written by hand (canned cycles) |
| CAD | `part.step` (AP242 with semantic PMI), written by `make_cad.py` with `cnctwin.geom.io.step.write_step_pmi` |

## Setup

* Stock frame = CAD frame: origin at the plate's lower-left-bottom corner, z up, top at z 16.
* The plate lies on two 25 mm parallels under y 2..10 and y 70..78 (every hole breaks through
  clear of them) and is clamped at its ends over the parallels by four straps whose noses sit 5 mm
  onto the plate (x 5 and x 155). The plate ends are kept free of holes, so no holder (the widest
  is the ER32 nut, 50 mm) passes over a strap.
* The workholding frame is the plate corner on the parallels: machine X -461, Y -243.2, Z -585
  (table top Z -610 + 25). G54 = X -461.0, Y -243.2, Z -569.0 (corner, top).
* The database has no sub-plate or parallel items and no strap geometry, so the straps go straight
  into the VF-2SS table T-slots (16 mm, 125 mm pitch) and their outline is not known to the twin
  (the sanity checks say so).

## Tools

| T | item | holder | stickout / H | data used (maker's range for 1045) |
|---|---|---|---|---|
| 1 | `osg-62910-nc-lds-10x90` HSSE spot drill D10 x 90 deg | `haimer-40-720-16` ER16 | 45 / 115 | vc 26 (22-30) m/min, f 0.12 (0.10-0.18) mm/rev: S830 F100 |
| 2 | `osg-8690680-ado-3d-6.8` carbide drill, internal coolant | `haimer-40-720-16` | 40 / 110 | vc 110 (80-160), f 0.18 (0.14-0.26): S5150 F927 |
| 3 | `osg-8690850-ado-3d-8.5` | `haimer-40-720-16` | 50 / 120 | vc 110 (80-160), f 0.20 (0.16-0.28): S4120 F824 |
| 4 | `harvey-18032` 1/2 in 90 deg pointed chamfer cutter, M-42 | `haimer-40-720-32` ER32 | 35 / 105 | Harvey steel 'full chamfer' at D_eff 9.75 mm: vc 24.4 m/min, fz 0.033 x 45 % for plunging = 0.0149: S800 F48 |
| 5 | `osg-8325107-a-pot-m8x1.25` spiral point tap | `haimer-40-720-16` | 45 / 115 | vc 20 (15-50) m/min: S800, F = 800 x 1.25 = 1000 |
| 6 | `osg-8325117-a-pot-m10x1.5` | `haimer-40-720-16` | 50 / 120 | vc 20 (15-50): S640, F = 640 x 1.5 = 960 |

The ADO-3D drills need a sealed ER16 collet for through-spindle coolant (not in the database).
Taps are held in plain ER16 collets for rigid tapping (no CAT40 synchro holder is in the database).

## Program

1. **T1** spots all ten holes with G82: the M8 holes to Z-4.5, i.e. a 9.0 mm 90 deg spot that
   is their countersink, the M10 holes to Z-3.0 (6.0 mm spot).
2. **T2 / T3** drill through with G83 (Q6, R2) under M88 (a 1 s dwell for the TSC pressure):
   6.8 to Z-18.5 (full diameter 1.26 mm below the bottom), 8.5 to Z-18.8.
3. **T4** countersinks the M10 holes to 11.0 mm (G82 Z-5.373 = (11.0 - 0.254 tip) / 2).
4. **T5 / T6** rigid-tap through with G84 (R5): M8 to Z-22.5 (5 x 1.25 = 6.25 mm of chamfer ends
   0.25 mm below the bottom: full thread over the whole 16 mm), M10 to Z-24.5 (1.0 mm below).
5. Each tool section starts with its own G90 G54, G43 H, spindle and coolant and ends with M09,
   M05, G53 G49 Z0.; the program ends with the table forward (G53 Y0.) and M30.

## Tolerances

PMI in `part.step` (read back by `cnctwin.geom.io.step.read_step`: 3 datums, 20 tolerances, none
unmapped): datums A (bottom), B (left end), C (front); per hole a position tolerance of diameter
0.10 to A|B|C and the drilled diameter with the ISO 965-2 6H minor-diameter limits (M8: 6.647-6.912,
M10: 8.376-8.676). By hand in `project.yaml`: the same positions with the thread class
(`diameter_mm: {nominal: 8 | 10, fit: 6H}`), because the PMI writer has no thread form.

## What the run is meant to show

* drilling thrust and torque in steel (point geometry, chip thickness, peck retracts), chips
  flushed by 21 bar through the drill, exit burrs on through holes;
* rigid tapping: the spindle-feed synchronisation at 800 / 640 rpm, the reversal at the bottom,
  tapping torque, thread formed by the swept tap profile;
* hole positions from the machine's positioning errors; hole sizes from runout;
* clamping by straps on parallels: no slip, negligible plate deflection under the drill thrust.

## Files

`project.yaml` · `op10.nc` · `part.step` · `make_cad.py` · `EXPECTED.md`
