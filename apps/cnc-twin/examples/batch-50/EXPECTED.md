# Expected physical behaviour: batch-50

Numbers from `expected_calc.py` (run: `bin/tsafe python examples/batch-50/expected_calc.py`) and the
controller's ideal dry run of `bracket.nc`. Force law: Kienzle **kc11 = 800 N/mm2, mc = 0.25**
(db/materials/al6061-t6, ISCAR, carbide any coating), radial/tangential ratio 0.5 (band 0.3-0.7).

## 1. Probing and the work offset (every part)

* The setter left G54 at (-410.75, -228.15, -495.80); the blank's front-left-top corner is at
  (-411.000, -228.000, -496.000) (fixture pose + stock pose). Part 1: the probe triggers at Z -0.200,
  X -0.250 - 2.999, Y +0.150 - 2.999 (work coordinates of the hand-set offset), so the program moves
  G54 by **dX -0.250, dY +0.150, dZ -0.200 (each +- 0.003)**: probe repeatability 1 um and pretravel 1 um
  (machine file), ball radius 2.999 programmed against the 3.000 drawn.
* Parts 2-50: the twin loads every blank at the same pose, so the corrections are **|d| <= 0.003** and
  G54 stays at the measured corner.
* M79 after each G31: no alarm (every probe move triggers, 2 mm before its end).

## 2. Milling with T1 (D20 3FL, S10186 = vc 640 m/min, the Helical start value)

* Roughing 5 passes, ae 5.94 (0.30 D), ap 17, fz 0.183: h_max 0.167 mm, torque 12.3 Nm,
  **power 13.1 kW (band 9-15 kW)**, i.e. about 90 % of the S1 rating at 10 186 rpm (half of the
  28.9 kW peak curve): no spindle overload. MRR 565 cm3/min.
* Mean wall-normal force in roughing 1.2 kN (band 0.8-1.6 kN) against a tool-holder-spindle stiffness of
  **24 N/um** (fluted part as d16 over 47 mm + ER32 holder + spindle modes): the tool yields
  **~50 um** away from the upright, so the wall pass finds 0.30-0.36 mm instead of 0.30.
* Wall pass ae 0.3, fz 0.0838: mean force 32 N, deflection ~1.3 um; peak summed tangential force
  0.73 kN.

## 3. Geometry per part, and the step at the sister switch

* Radial wear of T1 at VB 0.15 mm: `VB tan(alpha)` = 0.15 tan 12 deg = 32 um (primary clearance of the
  tool file). Upright thickness (front face to the finished wall):
  **parts with T1: 20.030-20.036 (accept 20.020-20.045); parts with T11 (new): 20.000-20.004
  (accept 19.995-20.010)** - a step of about -0.03 mm at the switch, all inside 20 +- 0.05.
* Foot thickness 8.000 +- 0.005 for both tools: the program uses `H#3026`, the length of the tool in the
  spindle, and T11 is set 0.024 longer than T1. If the length of T1 were applied to T11, the foot would
  drop to 7.976 from the switch part on (a regression check of the sister's offsets).
* Holes d6.8 (HSS drill, runout 8 um): oversize of a few hundredths, typical of twist drills (no hand value:
  accept d 6.80-6.90); position within 0.02 of (15, 37) and (45, 37) from the probed corner. Drill torque
  about 1.5 Nm, thrust 0.4-0.9 kN (Kienzle with the HSS entry, 920 N/mm2).

## 4. Surface

* Upright (peripheral wall pass, runout 5 um): per-tooth cusp 0.09 um, once-per-revolution mark at
  `z fz` = 0.251 mm, height 0.79 um: **Ra 0.15-0.4 um** with T11, rougher with the worn T1 (flank wear
  burnishing / smearing, no hand value).
* Foot top: cut by the end teeth during roughing: Ra <= 1.6 um (spec 3.2).

## 5. Tool life, sister switch, wear progression

* T1 starts with 150 s of its 60 min life left (used 57.5 min). Per part it is over the stock for
  **6.17 s** (5 x 80 mm at F5592 + 80 mm at F2561) and in G01 feed for 9.11 s.
  - Life counted by cutting time with the cutter in the material: T1 expires during **part 25**; the M06
    of **part 26** loads **T11** (accept 25-27).
  - Life counted by all feed (G01) time of the tool: expires in part 17, T11 from part 18.
  The life limit, not wear, decides the switch: VB grows only ~0.26 um per part (Taylor, est.), from
  0.150 to ~0.156 mm at part 25 (VB limit 0.3).
* T11 from part 26 to 50: life used 25 x 6.17 s = 2.6 min; VB ~0.01 mm (break-in) at part 50.
* T2 (drill, 33 min left, ~0.5 s in cut per part) **does not expire**: T12 is never loaded.
* Tool cost (production model `price / (regrinds + 1)` x life used per part): T1/T11 317.25 EUR / 3 per
  60 min edge life -> 0.18 EUR per part; T2 has no price in the database (cost reported missing).

## 6. Cycle time, operator, takt

* Ideal dry run (no acceleration, probe moves to the end of travel): **29.33 s**. With the probes
  triggering 2 mm early (-2.4 s) and spindle ramps, accelerations and the orient (+2-4 s):
  **cycle 29-32 s per part** (< 40 s). The switch part has the same cycle (same chip-to-chip time).
* Operator per part: load 12 + unload 6 + clean 6 + 2 x door 3 = 30 s, machine waiting (no pallet
  changer); inspection 45 s on part 1 (first article) and every 10th part (6 in all).
  **Takt ~59 s, batch ~54 min, ~56 parts per hour.**

## 7. Mass and energy

* Chips per part 31 181 mm3 = **84.2 g** (rho 2700); part 0.118 kg; batch 4.21 kg of chips. Mass ledger
  exact per part.
* Spindle energy per part dominated by roughing: 13.1 kW x 5 x 0.86 s ~ 56 kJ at the shaft
  (+ motor losses, idle and run-up), base load 900 W over the 59 s takt ~ 53 kJ.
