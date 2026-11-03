# Expected physical behaviour: drill-tap-plate-1045

Database values: `aisi1045` Kienzle kc1.1 = 1700 N/mm2, mc = 0.25 (carbide, any coating, Walter
P2) and 1672 / 0.32 (uncoated carbide, Klippel 2022 at 100 m/min); E = 206 GPa; rho = 7850 kg/m3.
Drill point 140 deg (kappa_r = 70 deg), LU 21 / 26 mm. TSC 21 bar. Straps 9.79 kN at 47.5 Nm,
friction 0.19 (`carrlane-strap-clamps`).

## 1. Drilling loads (T2, T3; Sandvik drilling formulas)

h = (f/2) sin kappa_r; kc = kc1.1 h^-mc; P_c = f vc D kc / 240 000 (kW); M_c = P_c / omega;
F_f = 0.5 kc (D/2) f sin kappa_r. The values below use the coated-carbide entry; the uncoated
entry gives +17 %. The chisel edge (R-thinned point) adds about 10-20 % to the thrust.

| drill | n (rpm) | f (mm/rev) | h (mm) | kc (N/mm2) | P_c (kW) | M_c (N m) | F_f (N) |
|---|---|---|---|---|---|---|---|
| 6.8 | 5150 | 0.18 | 0.0846 | 3152 | 1.77 | 3.28 | 906 |
| 8.5 | 4120 | 0.20 | 0.0940 | 3070 | 2.39 | 5.54 | 1226 |

| quantity | expected | acceptance band |
|---|---|---|
| drilling torque, 6.8 | 3.3 N m | 2.6-4.6 N m |
| drilling thrust, 6.8 | 0.91 kN | 0.8-1.4 kN |
| drilling torque, 8.5 | 5.5 N m | 4.4-7.8 N m |
| drilling thrust, 8.5 | 1.23 kN | 1.0-1.9 kN |
| spindle power | <= 2.8 kW (S1 at 4000-5200 rpm is 16-17 kW) | no overload finding |
| pecks | G83 Q6: 4 feed segments per hole (to Z-4, -10, -16, -18.5/-18.8 from R2) with full retracts | exact |
| thrust at breakthrough | falls to zero over the point length (1.24 / 1.55 mm) as the point exits Z-16 | qualitative |

## 2. Through-spindle coolant

Two outlets of 0.7 mm (6.8 drill; 0.85 mm for 8.5, both est. in the tool file): orifice flow
Q = Cd A sqrt(2 dp / rho) with Cd 0.6, dp 21 bar, rho 992 kg/m3: jet velocity sqrt(2 x 21e5 / 992)
= 65 m/s, Q = 0.6 x 7.70e-7 m2 x 65 = 1.8 L/min (6.8) and 2.7 L/min (8.5), below the pump's
11 L/min (est.), so the 21 bar setting holds. Acceptance: Q 1.0-3.5 L/min per drill; chips
leave the 16 mm holes without packing (flute fill < 1 at 2.4 D depth).

## 3. Tapping (T5, T6)

Groove area per pitch outside the drilled hole (ISO basic profile: groove 3P/4 wide at D1, P/8 at
D, depth 5H/8; minus the strip between D1 and the drill): M8 0.30 mm2, M10 0.46 mm2. Chip per
chamfer tooth: radial depth / (5 threads x 3 flutes) = 0.040 / 0.050 mm -> kc = 3801 / 3595 N/mm2.
Torque = kc A r_m (r_m = (D_drill + D)/4); cross-check with the catalogue rule M = P^2 D kc / 8000.

| tap | n (rpm) | F (mm/min) | torque kc A r_m | P^2 D kc / 8000 | acceptance |
|---|---|---|---|---|---|
| M8x1.25 | 800 | 1000 | 4.2 N m | 5.9 N m | 3-8 N m |
| M10x1.5 | 640 | 960 | 7.7 N m | 10.1 N m | 5-14 N m |

* feed per revolution = pitch exactly (rigid tapping): 1.250 / 1.500 mm/rev; spindle-Z
  synchronisation error below 1 % of the pitch at constant speed;
* spindle reversal at Z-22.5 / Z-24.5, retract at the same feed (no J word);
* thread over the full plate thickness: tap chamfer (5 P) ends 0.25 mm (M8) / 1.0 mm (M10)
  below the bottom;
* pitch diameter of the cut thread: basic D2 7.188 (M8), 9.026 (M10) plus the OSG tap limit
  (OH3 / OH4 in the tool file) and runout: inside 6H (7.188-7.348 / 9.026-9.206).

## 4. Hole size and position

* Drilled diameter: drill (h8) plus runout-driven oversize, 6.80-6.83 / 8.50-8.53 mm; inside the
  PMI minor-diameter limits 6.647-6.912 / 8.376-8.676. Acceptance: 6.79-6.85, 8.49-8.55.
* Position (diameter zone, A|B|C): machine positioning (est. 8 um X, 7 um Y accuracy in the
  machine file) and drill wander in a spotted start (< 10 um): diameter zone 2 sqrt(8^2 + 7^2) +
  10 = 31 um. Acceptance: every hole within a 0.05 mm diameter zone (tolerance 0.10).
* Countersinks: M8 9.0 mm (spot drill, 90 +-1 deg), M10 11.0 mm (chamfer mill), +-0.1 mm.

## 5. Clamping and plate deflection

Normal clamping force 4 x 9.79 = 39.2 kN; lateral capacity mu N = 0.19 x 39.2 = 7.4 kN. The
largest lateral load is the tapping torque reaction (10 N m over the 70 mm lever to the clamp
line: 0.15 kN) and the drill thrust presses the plate onto the parallels: no slip
(part_pose change < 1 um). Plate bending between the parallels (span 60 mm, effective width
40 mm, I = 40 x 16^3 / 12 = 13 653 mm4) under the 8.5 drill thrust at mid-span:
delta = F L^3 / (48 E I) = 1226 x 60^3 / (48 x 206 000 x 13 653) = 2 um. Acceptance < 5 um.

## 6. Removed material

CAD difference (drilled holes and countersinks, no threads): 160 x 80 x 16 - 197 504.4 =
**7295.6 mm3 (57.27 g)**. The threads remove about 0.40 of the annulus between the drill and the
major diameter over 16 mm: 6 x 89 + 4 x 143 = 1106 mm3 (8.7 g). Total 8400 mm3 = **65.9 g**.
Acceptance: total removed 8400 +-170 mm3 (thread estimate +-15 %); field volume to tol_mass.

## 7. Cycle time

Lower bound from the program (the static scan's expanded cycles: rapids at 35.6 m/min, feeds,
1.6 s tool changes): **140.7 s** (T1 40.0, T2 12.6, T3 10.6, T4 35.7, T5 23.2, T6 18.5 s). The
controller dry run needs the canned-cycle providers (G82/G83/G84; TW0001 in this build). The
real cycle adds the dwells (G82 P0.3 x 10, P0.2 x 4, G04 P1. x 2: 5.8 s), chip-to-chip time
(6 x 0.6 s), spindle run-ups and accelerations. Expected 150-175 s; acceptance 145-185 s.

## 8. Program sanity (cnctwin.sanity on the resolved project)

| line | rule | severity | why |
|---|---|---|---|
| 0 | retract-below-clamps | info | the strap clamps have no outline in the project format (WORKHOLDING_ENTRY has no outline_mm; the database items have no body) |
| 15 | scan-limits | info | `G53 G00 Z0.` before the first tool call: no tool geometry for that move |

No other finding: speeds and feeds are inside OSG's ranges, the tapping feed equals the pitch,
TSC is available and the drills have coolant holes, no tool cuts deeper than its flutes.
