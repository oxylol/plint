# Expected physical behaviour: die-h13-ballnose

Numbers come from three sources:

* `expected_calc.py` (run: `../../bin/tsafe python expected_calc.py` in this folder);
* `toolpath_stats.json` (written by `make_program.py`);
* `check_program.py` (ideal swept balls against `part.step`).

The force law is Kienzle **kc11 = 3000 N/mm2, mc = 0.25** (db/materials/h13-50hrc, Walter H1
46-52 HRC, carbide any coating). Wear uses the file's extended Taylor law for TiAlN carbide
(est.: 15 min at vc 100, fz 0.1, ap 0.5). Sign convention: + = material left on the part (cavity
smaller), - = cut too deep.

## 1. Kinematics and travel

* `PLANE SPATIAL SPB+15 ... SEQ+` gives **B +15, C 0**, and `SPB-15 ... SEQ-` gives **B -15, C 0**.
  The machine file's B axis turns about -y, so B +15 lifts the +x side of the table and the tool leans
  towards +x relative to the part. Both halves stay at C 0, so the C pivot errors (8, 8 um) do not
  shift one half against the other. If a controller solved `SPB-15` with C 180 instead, the halves would
  be 16 um apart in x and y; watch for this.
* Travel used (machine coordinates, tool tips): **X -50.4..+50.4, Y -16.9..+16.9, Z -468.9..-453.0**.
  The Z limit is -475, so the margin is 6 mm. The swivels are 15 deg at 180 deg/s, after `MB MAX`.
* The rotary axes are clamped while cutting (3+2). No TCPM.

## 2. Cutting conditions (per tool)

| | T1 roughing | T2 finishing |
|---|---|---|
| S / F | 2590 / 290 | 3640 / 349 |
| fz | 0.028 | 0.024 |
| contact angle from the axis | 14-54 deg (p50 41) | 14-54 deg (p50 41) |
| vc p5 / p50 / p80 / max (m/min) | 48 / 70 / 75 / 77 | 57 / 91 / 100 / 104 |
| engagement normal to the surface | 0.13-0.24 (blank allowance 0.19-0.30, least near the rim) | 0.06-0.10 (0.06 + roughing scallop) |
| stepover | 0.81-1.27 (scallop 40 um) | 0.27-0.42 (scallop 4.5 um) |
| chip h_max | 9 um | 4-5 um |
| Kienzle kc | 11.5 kN/mm2 | 13.2-14.1 kN/mm2 |
| removal rate (mean) | 0.98 mm3/s (315 mm3) | 0.15 mm3/s (113 mm3) |
| cutting power | **~11 W** (band 5-25 W) | **~2 W** (band 1-5 W) |
| mean tangential force | **~10 N** (band 5-20 N) | **1-1.6 N** (band 0.5-4 N) |

* The chips are of the order of the edge radius (6 um, est.) and above the minimum chip thickness
  (0.25 x 6 = 1.5 um). Ploughing and the passive force are therefore large compared with Kienzle's
  tangential force. For hard milling with a -5 deg rake, expect the resultant at 1.5-2.5 x Ft.
* Spindle load is dominated by idle and bearing losses, well under 1 kW. Cutting torque is about
  0.04 Nm in roughing and below 0.01 Nm in finishing.
* Chips are segmented (saw-tooth) and come off hot. Air blast, no flood.

## 3. Deflection

* Static stiffness at the ball is **6.2 N/um** (0.162 um/N). This combines the stepped carbide tool
  (flutes d 8 equivalent, neck 9.8, shank 10) in the chuck, the HAIMER chuck (d 26.5 over 65 mm) and
  the spindle's softer flange mode (80 N/um, est.).
* Roughing: 10-20 N gives **1.6-3.2 um**. The tool yields away from the surface, so this adds to the
  finishing stock.
* Finishing: 1-4 N gives **0.2-0.6 um**. This is negligible against the scallop.
* No chatter is expected: the forces are tiny and the overhang is 4 x D with a variable-pitch tool.

## 4. Surface

* Across the raster lines, the scallop between neighbouring paths is predicted from the probe grid at
  **4.0 (p50) / 4.45 (p95) / 4.5 (max) um** (`toolpath_stats.json`). `check_program.py` measures
  material left of **p99 4.6, max 5.4 um** in the cavity; the extra is facet and sampling effects.
* For parabolic cusps, Ra = h / 3.9, which gives **Ra 1.0-1.15 um across the lay**. Along the lay:
  * The per-tooth cusp `fz^2 / 8R` is 0.014 um.
  * With runout 3 um TIR, one tooth makes the surface, leaving once-per-revolution marks every
    0.096 mm with a 0.23 um cusp.

  So **Ra along the lay is <= 0.2 um**.
* Expected at both roughness points (bottom and steep side): **Ra 0.9-1.3 (accept <= 1.6), Rz 3.8-5.5
  (accept <= 6.3)**. A flat-surface stepover would have given 7 um at the bottom, where the y-section
  radius is 13.3 mm and the curvature factor is 1.6; the curvature-corrected stepover avoids this.
* Over the convex R2 rim, the same stepovers leave about 1 um (factor 0.29).

## 5. Profile (cavity 0.03 = +-0.015, rim 0.05 = +-0.025 to A-B-C)

Error budget (normal to the surface, finishing tool):

| source | cavity | rim |
|---|---|---|
| program (ideal swept balls, `check_program.py`) | 0..+5.4 (scallop peaks), min -0.06 | -1.1..+5.4 |
| deflection | +0.2..+0.6 | +0.2..+0.6 |
| runout 3 um TIR (largest edge 1.5 um proud) | -1.5 | -1.5 |
| flank wear, first to last line (clearance 8 deg; section 6) | 0..+0.1 (in-cut basis: 0..+1.4) | 0..+0.1 |
| B pivot error (+-2.6 um along x, sign by half) | -1.6..+1.6 (x-ends), 0 at the joint | -1.6..+1.6 |
| tool length setting (uncertainty runs only: +-3 um) | +-3 cos(psi) | +-3 cos(psi) |

* **Cavity profile, nominal run: -0.004..+0.007 (accept -0.015..+0.015). Rim: -0.005..+0.007 (accept
  -0.025..+0.025).** The mean surface sits 1-2 um above the nominal (the scallop mean is h/3).
* The joint between the halves at x = 0 shows a step of **<= 2 um** (positioning repeatability and wear
  between the two finishing halves). The scallop ridges line up, because both halves use the same y
  positions.
* The twin measures the surface on its 0.05 mm refine lattice, which gives 6-9 samples per scallop, so
  the scallop peaks can read up to about 1 um low.

## 6. Tool wear and life

* The twin calibrates its wear law to the Taylor life as continuous edge contact
  (docs/models/wear.md section 17), so an edge wears only while it is in the material.
* T1 (life 394 min at p50 vc 70): 5.3 min in cut, with an edge engaged about 7.6 % of each revolution
  (radial depth ap / sin psi on the local circle of the ball). That is 0.4 min of edge contact,
  0.1 % of the life: **VB 0.010 -> about 0.0102 mm**.
* T2 (life 256 min at p50 vc 91): 13.0 min in cut, edge engaged 4.3 %, 0.56 min of contact, 0.22 % of
  the life: **VB 0 -> about 0.0004 mm**. The radial wear (0.06 um) is invisible on the part.
* Read as in-cut time instead, the usual basis of milling tool-life data, the same Taylor entry would
  give VB +2.7 um (T1) and +10 um (T2). The finishing gradient would then be 1.4 um, still inside the
  profile band.
* Life counters (time in cut, as the production model counts): T1 6 + 5.6 min, T2 13.4 min, both of
  30 min. No change by life.

## 7. Heat

* Energy at kc 10-13 kN/mm2: 4.3-5.6 kJ for the job. Even if all of it stayed in the 2.45 kg insert, the
  rise would be **<= 5 K**. With air blast most leaves with the chips, so expect a part warming of
  **0.5-2 K**. That is less than 2 um of thermal growth over the cavity, and no visible thermal error.
* In the warm machine at 2600-3640 rpm, the spindle growth stays within a few um over the 19 min.

## 8. Mass

* Hard-milled: **427.8 mm3 = 3.34 g** (rho 7800), roughing about 315 mm3 and finishing about 113 mm3.
  The mean allowance is 0.272 mm over a cavity and rim area of 1575 mm2. The mass ledger closes on
  `stock.step` minus `part.step`.

## 9. Time

* Feed time: **T1 333.7 s + T2 803.7 s = 1137 s**. Two tool changes take 2 x 5.5 s. There are four
  swivels with `MB MAX` retract, plus approaches and retracts, and the spindle run-up overlaps the
  positioning. **The cycle is about 1165-1185 s (19.5 min).**
* The blocks are 0.125-0.25 mm long and run at 290-349 mm/min, i.e. 25-50 blocks/s. That is far below
  the TNC's block rate, and at these feeds no acceleration or jerk limit slows the path, so the real
  time matches the programmed feed time.
