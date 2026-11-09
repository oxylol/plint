# Expected physical behaviour: pocket-islands-al6061

Each item gives the quantity, how its magnitude follows (hand calculation or cited rule), the
expected value and the acceptance band a regression test should use. Database values: Kienzle
kc1.1 = 800 N/mm2, mc = 0.25 (`al6061-t6`, carbide, ISCAR group 22); E(carbide) = 530 GPa
(`carbide-submicron-10co`); rho(6061) = 2700 kg/m3; spindle mode 70 N/um (`haas-vf2ss`
frf_modes); probe repeatability 1.0 um (`probes.spindle_probe`). Program coordinates: Z0 = stock
top after the probe.

## 1. Probe Z set (N10, lines 11-20)

The setup gives G54 Z = -479.800; the stock top is at machine Z -480.000 (fixture_pose z -510 +
30 mm stock). O9811 measures the top at program Z -0.200 and adds the error to #5223.

| quantity | expected | band |
|---|---|---|
| G31 triggers | 4 G31 moves, 2 triggering (first touch F600, measuring touch F60) | exact |
| #135 (measured error) | -0.200 mm | +-0.003 mm (repeatability 1 um + pretravel 1 um est. + latch quantum: 60 mm/min x 1 ms IPO = 1 um) |
| G54 Z after N10 | -480.000 mm | +-0.003 mm |
| probe stylus load | trigger 0.5 N (XY) / 5.85 N (Z), no deflection of the part | < 10 N |

Without N10 every Z below would be 0.200 mm deeper (pocket depth 10.200: out of tolerance).

## 2. Cycle time

Dry run of the controller on the ideal machine (no acceleration limit, rapids at 35.6 m/min,
tool change 1.6 s): **167.6 s** (T20 11.2 s, T1 114.9 s, T2 41.5 s). The largest item is the
trochoid: 4957 mm of arcs at F3474 = 85.6 s (length / feed). The real cycle adds acceleration
(arcs of 8.5 mm at 58 mm/s: ~10 % at 5 m/s2), the chip-to-chip time (2.2 s vs 1.6 s) and three
spindle run-ups to 12 000 rpm (1.3 s each).
Expected 175-200 s; **acceptance 167.6 s <= t <= 215 s**, and >= the dry-run bound exactly.

## 3. Removed material (mass ledger)

Stock 240 000 mm3 - part 193 201.8 mm3 (OCC volume of `part.step`) = **46 798 mm3 = 126.36 g**
of chips. By hand: pocket floor area 4252.2 mm2 x 10 mm + island areas (562.3 + 506.5 mm2) x 4 mm
= 46 797 mm3.
Acceptance: removed field volume within +-0.3 % of 46 798 mm3 (lattice 0.075 mm; the field volume
is exact in integer units, the CAD comparison carries the dexel boundary error of about
lattice^2 x curvature / 8 on the R4/R8/R9 faces) and mass conservation to `tol_mass` 1e-9.

## 4. Roughing: constant engagement and loads (T1, N20)

Engagement from the 2-D material map of `make_program.py`: angle mean 38 deg, p95 60 deg, max
64 deg after the helix; radial equivalent ae = R (1 - cos phi) = 1.28 / 2.50 / 2.81 mm; ap 9.8 mm.

Mean chip thickness h_m = fz (1 - cos phi) / phi (phi in rad), Kienzle kc = 800 h_m^-0.25,
mean tangential force per revolution F_t = z ap fz kc (1 - cos phi) / (2 pi):

| engagement | h_m (mm) | kc (N/mm2) | F_t (N) | torque (N m) | cutting power (kW) |
|---|---|---|---|---|---|
| mean, 38 deg | 0.0309 | 1909 | 183 | 0.91 | 1.15 |
| max, 64 deg | 0.0485 | 1705 | 432 | 2.16 | 2.71 |

(power = torque x 1257 rad/s; cross-check: P = kc x ae x ap x vf = 1705 x 2.81 x 9.8 x 57.9 mm/s
= 2.72 kW at the loop front.) The time-averaged cutting power over the trochoid, from the volume
it removes (39 400 mm3 in 85.6 s, 460 mm3/s) times kc at the mean chip thickness, is 0.88 kW.

| quantity | expected | band |
|---|---|---|
| peak spindle cutting power in the loops | 2.7 kW | 1.8-3.6 kW |
| time-averaged cutting power, trochoid | 0.9 kW | 0.6-1.3 kW |
| peak of the per-revolution mean tangential force | 430 N | 290-580 N |
| spindle load at 12 000 rpm (S1 = 50 % of the 26.0 kW peak curve) | 21 % of S1 at the peak | no overload finding |
| helical entry: ramp angle | atan(0.792 / (2 pi 2.7)) = 2.67 deg | exact |
| no full-slot engagement after the helix | max engagement angle <= 64 deg | <= 70 deg |

The band is +-33 %: Kienzle from a catalogue kc1.1 (rake and speed effects not in the value), the
helix spreading each tooth's load over 78.6 deg of lag (2 ap tan 35 deg / D) smooths peaks but not
the per-revolution mean used here.

Tool deflection T1: stepped cantilever, fluted part (25 mm) with the Kops-Vo equivalent diameter
0.8 D (I = 201 mm4) or the 0.55 D core (I = 45 mm4), shank 7 mm (I = 491 mm4), E = 530 GPa:
k_tool = 14.1 N/um (0.8 D) to 4.2 N/um (core only), in series with the spindle (70 N/um): 3.9-11.7
N/um. With the radial load 0.3-0.5 F_t = 130-215 N at the peak: **11-55 um**, well inside the
0.3 mm allowance (no gouge of the finished walls).

## 5. Finishing loads and wall accuracy (T2, N30)

Walls: ae 0.3, ap 10, fz 0.0686, D12: phi_e = acos(1 - 2 x 0.3/12) = 18.2 deg, h_m = 0.0108 mm,
kc = 2482 N/mm2, F_t = 3 x 10 x 0.0686 x 2482 x 0.05 / (2 pi) = 41 N; wall-normal force 0.4-0.7 F_t
= 16-29 N. T2 stiffness (as above, LOC 24, stickout 34): 8-20 N/um -> deflection 1-4 um (the
tool bends away from the wall: material left). Runout 5 um TIR: the wall is cut by the
largest-radius tooth, r_eff = 6.0025 mm: pocket walls 2.5 um out, island walls 2.5 um in.

| tolerance | nominal | expected | acceptance (test band) | drawing tolerance |
|---|---|---|---|---|
| wall-position-x | 8.000 | 8.000 - 0.0025 + delta = 7.998-8.002 | 8.000 +-0.008 | +-0.03 |
| wall-position-y | 8.000 | same | 8.000 +-0.008 | +-0.03 |
| pocket-width | 84.000 | 84 + 2 (0.0025 - delta) = 83.997-84.003 | 84.000 +-0.012 | +-0.03 |
| wall taper over 10 mm | 0 | tool tip deflects most: wall 0-3 um thicker at the floor | <= 0.005 | - |

## 6. Depths and flatness

The axial stiffness of the tools (> 300 N/um) and the axial forces (< 30 N in finishing) give
< 0.1 um; the residual of the compensated spindle growth (25 um at 12 000 rpm x 0.3, warm machine,
time constant 25 min, 3 min of cutting) is at most 7.5 um and moves the floor and the island tops
alike, but not the measured stock top (the probe runs with the spindle stopped).

| tolerance | nominal | expected | acceptance | drawing |
|---|---|---|---|---|
| pocket-depth | 10.000 | 10.000 (+0 to +0.008 from spindle growth during cutting) | 10.000 +-0.010 | +-0.05 |
| island-top-depth | 4.000 | 4.000 (same growth) | 4.000 +-0.010 | +-0.05 |
| island-1-flatness | 0 | squareness XZ/YZ 15 urad x 32 mm = 0.5 um; raster pass-to-pass steps < 1 um | <= 0.005 | 0.02 |
| island-2-flatness | 0 | as island 1 | <= 0.005 | 0.02 |

## 7. Surface finish

Floor (square end mill, the outer corner generates the floor; end-dish 1 deg, db/models/tooling.yaml
`end_dish_rad`): R_max = fz / (cot kappa_r + cot kappa_r') with kappa_r = 90 deg, kappa_r' = 1 deg
(Boothroyd & Knight, sharp-tool kinematic roughness): R_t = 0.0686 x tan 1 deg = 1.20 um,
Ra = R_t / 4 (triangular profile) = **0.30 um**. Radial runout does not change the corner height;
edge rounding and ploughing raise it. Acceptance **Ra 0.2-0.8 um** (spec Ra 1.6, floor-finish).
Walls (peripheral marks, radius 6 mm): R_t = fz^2 / (8 r) = 0.098 um; with one tooth dominant
(runout 5 um > the 0.4 um per-tooth scallop) the marks are per revolution, f = 0.206 mm:
R_t = 0.88 um, Ra about 0.23 um. Acceptance wall Ra 0.05-0.4 um.

## 8. Workholding

Makro-Grip form-fit holding force 14 kN at 60 Nm (lang-2024-catalogue p. 123). Peak resultant
cutting force < 0.6 kN (section 4): safety factor > 20, no part slip (part_pose change < 1 um),
no jaw contact by any tool (jaw tops 3 mm above the stock bottom, deepest tool tip 10 mm below
the top = 20 mm above the stock bottom).

## 9. Program sanity (cnctwin.sanity on the resolved project)

| line | rule | severity | why |
|---|---|---|---|
| 10 | scan-limits | info | the move of T20 has no cutting geometry: T20 is the machine's OMP40-2, a `kind: probe` entry of the tool list (known to the controller by its T number, so no `tool-unknown`); the G65 P9810/P9811 probing calls run in the controller's dry run, which gives the scan their motion |
| 631 | cutting-data | warning | the engagement estimate gives ae 4.03 mm on two moves of the D10 slot passes, above the 4 mm Helical gives for slotting |
| 803 | cutting-data | warning | the engagement estimate gives ae 6.45 mm on two moves (the arcs at lines 803-805) of the D12 finishing passes, above the 4.8 mm Helical gives for slotting |

(As found by cnctwin.sanity with the controller assist on 2026-11-09, run through bin/tsafe; the program
dry-runs to M30 with no alarm, 165.2 s on the ideal machine.)

No other finding. In particular no rapid below the stock top, no G43/H, D or work-offset
problem, no cutting data outside Helical's limits on the straight passes.
