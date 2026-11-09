# Expected physical behaviour: lathe-shaft-4140

Numbers from `expected_calc.py` (run: `bin/tsafe python examples/lathe-shaft-4140/expected_calc.py`).
Force law: Kienzle `Fc = kc11 b h^(1-mc)`, **kc11 = 2000 N/mm2, mc = 0.25** (db/materials/aisi4140-qt30,
Walter, the only entry: carbide, any coating). Chip thickness `h = f sin(kappa_r)`, width `b = ap / sin(kappa_r)`;
for ap below the nose radius the effective kappa is half the nose engagement angle `acos(1 - ap/r)`.

## 1. Spindle speed and feed (exact from the program)

* T1 roughing G96 S200, clamp G50 S2500: 1415 rpm at d45, 1553 at d41, 1768 at d36, 2054 at d31,
  2449 at d26; the clamp holds below d25.5.
* T2 finishing vc 220 (clamped below d28), T3 groove vc 100, T4 thread G97 S1326 (vc 100 at d24),
  T5 part-off vc 90 down to d14.3, then the 2000 rpm clamp to the centre.
* Feeds per revolution: 0.3 (rough), 0.12 and 0.08 (finish), 0.08 (groove, part-off), 1.5 (thread).
  All within the Tungaloy rows for aisi4140-qt30 or, for VCMT/DGM/QGM, the first ISO P row
  (`cutting-data` reports nothing).

## 2. Roughing (G71 type I, D 2.5, U 0.6, W 0.1)

| pass radius | length | ap | n | Fc | Pc | of S1 |
|---|---|---|---|---|---|---|
| 20.5 | 141.9 | 2.0 | 1553 | 1.62 kN | 5.4 kW | 70 % |
| 18.0 | 141.9 | 2.5 | 1768 | 2.03 kN | 6.8 kW | 96 % |
| 15.5 | 61.9 | 2.5 | 2054 | 2.03 kN | 6.8 kW | 105 % |
| 13.0 | 24.9 | 2.5 | 2449 | 2.03 kN | 6.8 kW | 111 % |
| profile | ~190 | 0-1.2 | | | | |

* Spindle shaft power **5.4-6.8 kW (band 4.5-8 kW)** while cutting, i.e. 70-111 % of the S1 rating
  (half the 200 % peak curve of the machine file): the load meter goes past 100 % on the small
  diameters but stays inside the 10 min S2 rating (150 %); no overload finding.
* Spindle torque in pass 1: 1.62 kN x 21.5 mm (mean radius of the cut) = 35 Nm (band 25-42 Nm).
* Feed force (axial) 0.3-0.5 Fc = 0.6-1.0 kN, pushing the bar towards the chuck (the jaws hold it
  by friction and the bore step).
* Roughing feed time **63 s** (passes 42.4 s + profile pass 20.6 s).
* Chips: continuous below the 200 m/min segmentation onset of the database (vc = 200 is at the onset):
  long helical chips broken by the TM chip breaker at f 0.3; mill scale (0.05 mm, abrasiveness 2) is cut
  only on the first pass.

## 3. Finishing and the shaft as a beam

* Finishing force at ap 0.3, f 0.12, r 0.4: effective kappa 37.8 deg, h 0.073, b 0.49 -> **Fc 138 N**;
  passive (radial) force 0.6-1.0 Fc = 83-138 N. On the d30 seat (f 0.08): Fc 102 N, passive 61-102 N.
* Compliance at the cut: turret tool tip (machine file: 300 Hz mode, 40 N/um) 0.025 um/N plus the shaft,
  a beam fixed at the jaw face and elastically supported by the live centre (tailstock stiffness 30-60
  N/um assumed: not in the machine file or the Röhm item): 0.0097-0.0141 um/N at the d30 seat
  (S z 188), 0.0043-0.0050 at the d35 seat (S z 130), 0.0004 near the chuck.
* **Push-off on diameter**: d30 seat 4.3-8.0 um, d35 seat 4.9-8.3 um, collar < 6 um; the shaft grows
  less than 0.5 um in taper over the d30 seat.
* Finished sizes (20 degC, after the 1-2 um thermal shrink of section 6):
  **d30 g6: 29.985-29.991 (limits 29.980-29.993)**; drawn and programmed at 29.983 = 4 um below the
  middle for this push-off. **d35 h7: 34.991-34.995 (34.975-35.000)**. Collar d42 41.995-42.010.
* Cylindricity of d30 <= 0.003 mm (tolerance 0.008); runout of d35 and the hub face to A-B <= 0.006 mm
  (same clamping, spindle runout 3 um est.).

## 4. Surface (kinematic roughness of a round nose)

`Rt = f^2/(8 r)`, `Ra = f^2/(31.2 r)` (parabolic cusp):

* d30 seat, f 0.08, r 0.4: **Rt 2.0 um, Ra 0.51 um** (spec 0.8); accept Ra 0.45-0.8.
* d35, collar, thread OD, f 0.12: **Rt 4.5 um, Ra 1.15 um** (spec 1.6 / 3.2); accept 1.0-1.6.
  (Ploughing below the minimum chip thickness and side flow raise the real values by 10-40 %.)

## 5. Groove, thread, part-off

* Relief groove: plunge 2.2 mm at f 0.08 (1.3 s), Fc = 2000 x 3 x 0.08^0.75 = **0.90 kN** (band 0.7-1.1).
  Groove 3.00 wide (the insert width CW 3 +- 0.05), bottom d21.60, corners R0.2.
* Thread M24x1.5 G76 P1 (single edge, constant chip area), first depth D 0.3: **11 passes**, cumulative
  depths 0.300, 0.424, 0.520, 0.600, 0.671, 0.735, 0.794, 0.849, 0.900, 0.949, 0.971; 0.83 s in feed per
  pass, about 13 s. Results: root d22.058; crest topped by the full-profile insert to **d23.898
  (6g: 23.732-23.968)**; pitch diameter **22.924 +- 0.010 (6g: 22.854-22.994)**; the thread runs out
  into the relief groove. Per-revolution synchronisation of the start: every pass starts at the same
  spindle angle (no doubled flanks).
* Part-off at Z-180: Fc = **0.90 kN**, 1.35 kW at vc 90, 20 Nm at d45; feed time 11.3 s under CSS plus
  2.8 s at the 2000 rpm clamp = **14.1 s**. The tailstock is retracted (M22) before the part-off, so the
  part separates at X0 held by nothing: **a free part of 1.316 kg** (no parts catcher in the machine
  file); the twin must raise the free-part event at the separation tick. The 47 mm bar stub (0.587 kg)
  stays in the jaws.

## 6. Heat and dimensions

* Roughing energy about 6.8 kW x 45 s in cut = 0.31 MJ; 10-20 % into the workpiece = 31-61 kJ, which
  would heat the bar by 28-56 K adiabatically. Flood coolant (h 6000 W/m2K, db emulsion) on 0.025 m2 has a
  4 s time constant: the bar stays **3-9 K above the coolant** while cutting, so the finished diameters
  shrink by **1-3 um** (d30: 0.36 um/K) when the part cools to 20 degC, and the length by 6-20 um.
* Tool edge temperatures: no hand value; the coolant reaches the rake face only partially at 2 bar flood.

## 7. Tool wear

Taylor (db, CVD carbide in 4140 Q&T, est.): roughing at vc 200, f 0.3, ap 2.5: **T = 23.6 min**; 1.05 min in
cut -> 4.4 % of life: T1 VB grows from 0.05 to **0.06-0.07 mm**. The other tools change by less than
0.01 mm VB in this part.

## 8. Mass

Bar 365 715 mm3 (with the centre hole), part 167 680 mm3 = **1.316 kg**, stub 74 750 mm3 = 0.587 kg,
**chips 123 285 mm3 = 0.968 kg** (rho 7850). The mass ledger closes to `tol_mass`.

## 9. Cycle time

Feed times: rough 63.0 s, finish 51.8 s, groove 1.3 s, thread ~11 s in feed (+ returns), part-off 14.1 s
-> about **141 s of feed**. Turret indexes (0.5-1.2 s est.), G53 X0 retracts at 12 m/min (about 1 s
each way), spindle ramps between the CSS settings and the tailstock moves add 15-25 s:
**expected 155-170 s**. The controller's ideal dry run (no acceleration) gives 142.8 s of feed, the 141 s
above within the rounding of the hand calculation (which follows the CSS speeds), and 156.9 s in all, no
alarm: the lower bound the turret, retracts, ramps and tailstock add to.
