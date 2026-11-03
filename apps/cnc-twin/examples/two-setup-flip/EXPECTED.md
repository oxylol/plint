# Expected physical behaviour: two-setup-flip

Database values: `al6061-t6` Kienzle 800 N/mm2 / 0.25 (carbide), 920 / 0.25 (HSS), rho 2700 kg/m3,
E 68.9 GPa; Makro-Grip 14 kN at 60 Nm; machine positioning repeatability 4 um (X, Y), squareness
15 urad (est., machine file).

## 1. Carry-over and pose (OP10 -> OP20)

* The OP20 part is the OP10 result (from_previous) placed by the OP20 stock pose: a point
  (x, y, z) of the OP10 stock frame is at W (x - 40, 30 - y, 22.5 - z). Checks: the OP10 top
  (z 22.5) lies on W z 0 (the jaw seat); the OP10 pocket floor (z 16.5) is at W z 6, facing down;
  the drilled hole tips (z 1.5) are at W z 21.0, i.e. 1.0 mm below the OP20 stock top (W z 22.5)
  and inside the 2.5 mm the OP20 facing removes.
* After the OP20 facing the holes break through the bottom with the drill-point cone: with the
  130 deg point (cone height 3.4 / tan 65 deg = 1.585 mm) and the tip at z 1.5, the opening in the
  faced plane z 2.5 is 6.8 x (2.5 - 1.5) / 1.585 = 4.29 mm across; the counterbores (to z 8.9)
  remove the whole cone, so the finished hole is 6.8 mm from z 8.9 to the top.

## 2. Thickness and parallelism (cross-setup)

OP20 faces at W z 20.000 above the seat; the part's OP10 top is on the seat, so

thickness = 20.000 + (seat flatness) - (anything between the top face and the seat) + (Z drift).

| contribution | magnitude | how |
|---|---|---|
| burr on the seating edges | 0 | the 0.3 mm edge break in OP10 removes the facing exit burr where the part sits on the jaw steps |
| chip of height h on one jaw step | thickness 20 - h on that side, parallelism error h over the 60 mm width | the part rotates about X by h / 60 mm; OP20 faces parallel to the table |
| clamping (14 kN across the 60 mm, 22.5 mm tall section) | elastic, < 2 um after unclamping | sigma = 14 kN / (22.5 x 46 mm2) = 13.5 MPa, far below yield |
| spindle growth difference between the two facings (warm machine, both at 10 200 rpm, about 2 min apart) | < 3 um | 25 um x 0.3 residual x (change over 2 min with tau 25 min) |
| D20 axial deflection in facing (axial force < 100 N, stiffness > 300 N/um) | < 0.3 um | |

Expected with the clean seat the operator model gives (`clean_s` 20 s, chips blown off):
thickness **20.000 +-0.005** (acceptance 19.992-20.008, tolerance +-0.02); bottom-parallelism to A
**<= 0.005** (acceptance <= 0.008, tolerance 0.02). If the chip model leaves debris of height h
under a locator, the run must show thickness 20.000 - h (local) and parallelism h, with the
finding naming the debris; an aluminium facing chip (h about the chip thickness 0.1-0.2 mm, fz
0.183 x sin(engagement)) would break both tolerances.

## 3. Positions across the setups

* X: both setups take X0 at the left end (B) against the stop: counterbore-to-hole offset in X
  = machine repeatability, <= 4 um.
* Y: the self-centring vise places the part about its centre plane in both setups, and the
  programs place the holes (OP10) and counterbores (OP20) about that plane; so the
  counterbore-to-hole coaxiality does not depend on the blank width. Expected coaxiality <= 0.008
  (2 x 4 um repeatability); acceptance <= 0.015.
* Positions to datum C (the side face) do depend on the blank width w: every hole and counterbore
  moves by (w - 60) / 2 from C. Nominal run: 0. In the uncertainty runs (stock tolerance +-0.2 in
  y) the position deviation to A|B|C reaches a 0.2 mm diameter zone, beyond the 0.10 tolerance:
  the expected result of locating a centred part from a side datum.
* Nominal position deviation (holes and counterbores): <= 0.010 diameter zone (positioning
  accuracy 7-8 um est.); acceptance <= 0.02.

## 4. Feature sizes and depths

| feature | expected | acceptance | basis |
|---|---|---|---|
| pocket diameter (G13, D02 = 5.000) | 30.000 + TIR 0.005 = 30.005 | 30.000-30.015 | final circle rho = 10 + tool radius, the largest-radius tooth cuts |
| pocket depth | 6.000 | 5.995-6.008 | same setup as the top face |
| hole diameter (HSS drill in Al) | 6.82-6.85 | 6.80-6.88 | HSS drills cut 0.3-0.7 % oversize in aluminium (rule of thumb), runout 5 um |
| counterbore diameter (D6 at r 2.5) | 11.000 + 0.005 | 11.000-11.015 | 2 x 2.5 + 6 + TIR |
| counterbore depth from the bottom | 6.400 | 6.390-6.410 | same setup as the bottom face |
| top flatness (D20 zig-zag, ae 7) | <= 0.003 | <= 0.006 | tram 15 urad x 20 mm = 0.3 um; pass steps from deflection < 2 um |

## 5. Loads

* Facing (D20, ap 2.5, ae 7, fz 0.183, 10 200 rpm): phi_e = acos(1 - 2 x 7/20) = 72.5 deg,
  h_m = 0.183 (1 - cos phi_e)/phi_e = 0.1065 mm, kc = 800 x 0.1065^-0.25 = 1405 N/mm2,
  P = kc ae ap vf = 1405 x 7 x 2.5 x 93.3 mm/s = **2.3 kW** (band 1.5-3.1 kW).
* Helical counterbore (D6, r 2.5, pitch 1 mm per turn, F1280 = 81.5 turns/min): about 450 mm3
  per counterbore (D11 x 6.4 less the drilled part) in 6.9 turns (5.1 s), 90 mm3/s, P = kc x Q
  = 1500 x 90 = 0.14 kW (band < 0.3 kW).
* HSS drill 6.8 (Sandvik): h = (0.40/2) sin 65 deg = 0.181 mm, kc = 920 x 0.181^-0.25 = 1415,
  P = 0.40 x 80 x 6.8 x 1415 / 240 000 = 1.28 kW, torque 3.3 N m, thrust 0.5 x 1415 x 3.4 x 0.40 x
  sin 65 deg = 0.87 kN (bands +-35 %).

## 6. Removed material and time

* Blank 120 000 mm3 - part 87 314 mm3 (OCC volume of `part.step`) = **32 686 mm3 = 88.25 g**
  (OP10 + OP20). Acceptance +-0.3 %.
* OP20: controller dry run 57.7 s (T1 14.8, T6 28.3, T5 14.3 s). OP10: the scan bound is 46.2 s
  plus the G13 loops (path radii 1, 4, 7, 10 mm: 2 pi x 22 mm + transitions = 160 mm at
  F3474 = 2.8 s): 49 s; the controller dry run stops at G82 in this build (TW0001, no canned-cycle
  provider). Expected OP10 50-60 s, OP20 58-68 s.

## 7. Program sanity (cnctwin.sanity on the resolved project)

| op | line | rule | severity | why |
|---|---|---|---|---|
| OP10 | 12 | scan-limits | info | `G53 G00 Z0.` before the first tool call |
| OP10 | 55 | scan-limits | info | G13 is not expanded by the static scan |
| OP20 | 11 | scan-limits | info | `G53 G00 Z0.` before the first tool call |
| OP20 | 14 | scan-limits | info | first move after the tool change: start position unknown to the scan |

The chamfer passes feed out to Z5 before the rapid: the sanity context builds the vise jaws in
the stock frame, so in the flipped OP20 it places them on top of the part (W z 19.5-22.5) and a
rapid out of the Z-1.55 chamfer depth would be reported as crossing a jaw (a sanity limitation,
see notes); a G01 retract avoids the false report and is harmless.
