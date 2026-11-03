# Expected physical behaviour: thin-wall-al7075

Database values: Kienzle kc1.1 = 800 N/mm2, mc = 0.25 (`al7075-t6`, carbide; the ISCAR value is
for 100 HB, 7075-T6 is 150 HB, so forces may be under-predicted), E = 71.7 GPa, rho = 2800 kg/m3;
E(carbide) = 530 GPa; spindle mode 70 N/um. Numbers marked (M) are printed by `make_program.py`.

## 1. Roughing loads (T1)

Peel passes: ae 2.375, ap 12.5, fz 0.0965, D10: phi_e = acos(1 - 2 ae/D) = 58.3 deg,
h_m = fz (1 - cos phi_e)/phi_e = 0.0450 mm, kc = 800 h_m^-0.25 = 1737 N/mm2,
F_t = z ap fz kc (1 - cos phi_e) / (2 pi) = 475 N; P = kc ae ap vf = 1737 x 2.375 x 12.5 x 57.9 mm/s
= 2.99 kW. The trochoid between the fins (engagement mean 40 deg, max 57 deg (M)) peaks lower.

| quantity | expected | band |
|---|---|---|
| peel cutting power | 3.0 kW | 2.0-4.0 kW |
| peel mean tangential force | 475 N | 320-640 N |
| trochoid peak engagement | 57 deg (ae 2.28 mm) | <= 65 deg |
| spindle load (S1 13 kW at 12 000 rpm) | 23 % | no overload finding |
| rough fin blank (2.0 mm) deflection when the trochoid clears the band beside it (radial load 100-240 N at about 12.5 mm above its root; strip k = 3 E I / a^3, b_eff = 2a = 25 mm: 1.8 N/um) | 55-130 um, inside the 0.5 mm stock | no gouge of the final faces |

## 2. Finishing loads (T2, every ladder level)

ae 0.5, ap 5, fz 0.0584, D10: phi_e = 25.8 deg, h_m = 0.0130 mm, kc = 2372 N/mm2,
F_t = 3 x 5 x 0.0584 x 2372 x 0.1 / (2 pi) = **33 N** (band 22-45 N); wall-normal force
F_n = 0.3-0.75 F_t = **10-25 N** (16.5 N used below).

## 3. Deflection, thickness and taper

Wall stiffness at the cut (M): stepped clamped-free strip (2.0 mm rough root, 1.5-2.0 mm band,
1.0 mm finished above), delta = int_0^a (a - x)^2 12 / (E t(x)^3) dx per unit width, effective
width b_eff = 2 a (load height a above the base). Tool (M): D10 at 42 mm, flutes 25 mm as 0.8 D or
the 0.55 D core, in series with the spindle: 3.2-7.2 N/um.

| level | load height | k_wall (N/um) | wall deflection at 16.5 N | tool deflection |
|---|---|---|---|---|
| 1 (Z-5) | 22.5 mm | 0.57 | 29 um | 2.3-5.2 um |
| 2 (Z-10) | 17.5 mm | 0.93 | 18 um | 2.3-5.2 um |
| 3 (Z-15) | 12.5 mm | 1.8 | 9 um | 2.3-5.2 um |
| 4 (Z-20) | 7.5 mm | 4.9-5.1 | 3.3 um | 2.3-5.2 um |
| 5 (Z-25) | 2.5 mm | 19-46 | 0.4-0.9 um | 2.3-5.2 um |

Both the wall (pushed away from the cut) and the tool (bent away from the wall) leave material.
In down milling the surface is generated where the tooth leaves the cut; with the 35 deg helix
(lag 2 ap tan 35 / D = 40 deg over 5 mm) part of the flute still cuts then, so the surface error is
kappa = 0.3-1.0 of the mean deflection (Budak & Altintas 1995, static form errors in peripheral
milling of plates). Thickness excess of a band = 2 (kappa delta_wall + delta_tool):

| band (height above the base) | thickness excess | expected thickness | acceptance |
|---|---|---|---|
| 20-25 mm (level 1) | 22-68 um | 1.022-1.068 | 1.015-1.080 |
| 15-20 mm (level 2) | 15-46 um | 1.015-1.046 | 1.010-1.055 |
| 10-15 mm (level 3) | 10-29 um | 1.010-1.029 | 1.005-1.035 |
| 5-10 mm (level 4) | 7-17 um | 1.007-1.017 | 1.003-1.022 |
| 0-5 mm (level 5) | 5-12 um | 1.005-1.012 | 1.000-1.016 |

* the thickness grows monotonically from the root to the top (taper); the fin ends (free plate
  edges, about half the effective width) are up to twice as thick-excess as the middle at the
  upper levels;
* the top band may exceed the +0.05 mm tolerance (fin-1/2-thickness): expected behaviour of an
  unsupported 25:1 wall, not a fault of the model;
* witness steps at Z-5/-10/-15/-20 of 2-15 um (adjacent levels deflect differently);
* fin-1-perpendicularity (face tilt over 25 mm, half the excess difference top-to-root per face):
  0.008-0.030 (tolerance 0.05); fin-1/2 face flatness: 0.008-0.035 (tolerance 0.03, at risk);
* fin-1-position (datum C to the front face): 19.5 - (top excess)/2 = 19.466-19.497
  (acceptance 19.46-19.50); fin-gap: 19.0 - excess of the two facing faces = 18.932-18.989
  (acceptance 18.92-18.995; the lower end is outside the -0.05 tolerance).

## 4. Dynamics: wall modes, tooth harmonics and chatter risk

First wall mode per state (M, Euler-Bernoulli strip FE, clamped at the base): level 1 2616-2836 Hz,
level 2 3120-3263 Hz, level 3 3211-3327 Hz, level 4 2468-2711 Hz, level 5 1716-1885 Hz; finished
1.0 x 25 mm fin 1308 Hz (closed form 0.5596 sqrt(E t^2 / (12 rho)) / L^2 = 1308 Hz). Real
plate modes (free ends, 60 mm long) lie within -10 % / +20 % of the strip values.

* Tooth frequency 3 x 12 000 / 60 = 600 Hz; the force pulse lasts phi_e / omega = 0.36 ms, so
  harmonics up to about 2.8 kHz are strong. Ratios f_wall / f_tooth: 4.4-4.7 (level 1),
  5.2-5.4 (2), 5.4-5.5 (3), 4.1-4.5 (4), 2.9-3.1 (5).
* Absolute stability limit of the wall mode (1 DOF, k 0.57 N/um, damping est. 1 %, Ks 2.4e9 N/m2,
  directional factor 0.2, 0.22 teeth in cut): a_lim = 2 k zeta / (Ks N* alpha) = 0.05-0.2 mm,
  far below ap 5 mm. Stability at levels 1-4 therefore depends on the low-immersion lobes: stable
  pockets near integer ratios, period-doubling (flip) lobes near ratio k + 1/2.
* Expected: a high chatter risk at levels 1, 3 and 4 (ratio within 0.3 of 4.5 or 5.5): if the
  twin flags chatter, it must be at one of the upper four levels, with the chatter frequency at
  the wall mode of that level (2.3-3.6 kHz) and sidebands at multiples of 300 Hz (half the tooth
  frequency, flip type). No chatter at level 5 (k_wall >= 19 N/um, ratio about 3) or in roughing.
* Forced response: the wall amplitude at a level is at least 2x its static deflection where a
  tooth harmonic is within 10 % of the wall mode (levels 1, 4: 5th / 4th harmonic).

## 5. Removed material, time, clamping

* Removed volume: 90 x 60 x 32 - (90 x 60 x 7 + 2 x 60 x 1 x 25) = 172 800 - 40 800 =
  **132 000 mm3 = 369.6 g** (CAD: same). Acceptance +-0.2 % (lattice 0.1, refined 4x at the fins).
* Cycle time: controller dry run on the ideal machine **264.5 s** (T1 204.4 s, T2 60.1 s).
  Expected 270-300 s with accelerations and tool change; acceptance 264.5-310 s.
* Makro-Grip holding force 14 kN against peak roughing forces < 0.7 kN: no slip (part_pose change
  < 1 um); the jaws (3 mm high) are below the base top (Z-25 = 7 mm above the stock bottom).

## 6. Program sanity (cnctwin.sanity on the resolved project)

| line | rule | severity | why |
|---|---|---|---|
| 10 | scan-limits | info | `G53 G00 Z0.` before the first tool call has no tool geometry |

No other finding (cutting data inside Helical's limits on every pass, no rapid below the stock top,
compensation and offsets consistent).
