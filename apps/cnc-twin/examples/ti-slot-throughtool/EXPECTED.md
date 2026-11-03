# Expected physical behaviour: ti-slot-throughtool

Numbers come from `expected_calc.py` (closed-form laws with the database values; run it with
`bin/tsafe python examples/ti-slot-throughtool/expected_calc.py`). Each item gives the expected
magnitude and the band a regression test should accept. Where the twin may pick between database
entries the band covers both.

Force law used: Kienzle `Fc = kc11 b h^(1-mc)` per tooth, `h = fz sin(phi)` (90 deg lead), with
**kc11 = 2110 N/mm2, mc = 0.22** (ISCAR group 37, `tool_class: carbide, coating: any` in
`db/materials/ti6al4v.yaml`: for the TiAlN-coated AH3225 insert this entry outscores the uncoated measured
ones in `cnctwin.mechanics.coefficients.select_entries`). The measured Klippel et al. 2024 value at
40 m/min (kc11 1253, mc 0.22, uncoated) gives the lower end of each band (about 0.59 x). Radial/tangential
ratio Kr = 0.4-0.9 (Klippel kf/kc at h = 0.05 mm is 0.88-0.92 for a 0 deg rake, a positive insert is lower).

## 1. Kinematics (exact, from the program)

| quantity | slotting (N70-N170) | finishing (N180-N240) |
|---|---|---|
| spindle | S573, vc = pi 25 573 / 1000 = 45.0 m/min | S700, vc = 55.0 m/min |
| feed | F183 = 573 x 4 x 0.0799 mm | F336 = 700 x 4 x 0.12 mm |
| engagement | ae = 25 (full slot), ap = 3.9 per pass | ae = 0.512 per wall, ap = 7.8 wall + 0.2 floor |
| max chip thickness | h_max = fz = 0.080 mm | h_max = fz sin(acos(1 - 2 ae/D)) = 0.034 mm |

All inside Tungaloy's range for AH3225 in ISO S (vc 20-60, fz 0.04-0.15, ap <= 11.5, ae <= 1 D): the
`cutting-data` sanity rule must report nothing for T1.

## 2. Spindle load in slotting

Mean torque `T = z/(2 pi) int_0^pi Fc(phi) R dphi`, two teeth in cut on average.

* T = 19.6 Nm (measured coefficients: 11.7 Nm). **Band 10-24 Nm.**
* Cutting power P = T omega = 1.18 kW (0.70 kW). **Band 0.6-1.4 kW**, 4-9 % of the 16 kW S1 rating;
  the spindle torque is 14-28 % of the 85 Nm S1 plateau: no spindle-overload finding.
* Specific cutting energy P / MRR = 3.96 J/mm3 (2.35) at MRR = 3.9 x 25 x 183 = 17 840 mm3/min.

## 3. Forces and tool deflection

* Slotting mean side force (towards +y on the +X pass N150, towards -y on the -X pass N170):
  |Fy| = 1200 N (713 N). **Band 0.6-1.4 kN.** Mean feed-direction force -0.5 to -1.1 kN (Kr 0.4-0.9).
* Static compliance at the cut, tool tip in the shrink chuck in the spindle: steel body d 25 over 58 mm +
  holder d 45.5 over 75 mm (cantilevers) + the two spindle flange modes (80 and 220 N/um):
  0.0495 um/N, i.e. **k = 20 N/um (band 12-30 N/um)**; the spindle accounts for a third of the compliance.
* Rough slot: the tool deflects by Fy/k = 35-60 um, to +y in the upper band (z -3.9..0) and to -y in the
  lower band (-7.8..-3.9): **the rough walls show a step of 70-120 um between the two depth bands**, and
  the finishing allowance per wall varies between about 0.45 and 0.57 mm over the height.
* Finishing pass: peak single-tooth entry force at h = 0.034 mm over 7.8 mm is 1.2-1.6 kN (no helix
  smoothing; the 6.8 deg axial rake spreads the entry over about 4 deg of rotation), at 16 % of each
  tooth period. Mean wall-normal force 70-130 N; pass A (N220) also cuts the 0.2 mm floor layer over the
  whole width, adding 84 N towards +y.

## 4. Finished geometry (inspection)

* Slot width: programmed 2 x (12.5 + 0.512) = 26.024 mm. Each wall is left undercut by the deflection
  away from it: wall A (pass A, wall + floor force) 7.7-10.5 um, wall B 3.3-6.1 um.
  **Width 26.007-26.013 mm; accept 26.000-26.024** (inside 26 +0.05/0; narrower than programmed).
  Spindle thermal growth at 573-700 rpm (5 % of the maximum speed) and the expansion of the plate
  (9e-6/K, < 1 K bulk rise) are below 1 um.
* Slot depth 8.000 mm: axial force of the floor cut is below 300 N against an axial stiffness of the spindle
  much higher than the radial one: **depth 8.000 +- 0.010** (tolerance +- 0.05).
* Slot position (wall A from datum B): CAD 21.9875, programmed 21.988, plus the 7.7-10.5 um undercut:
  **21.996-21.999 mm (accept 21.988-22.005)**.
* Floor flatness and parallelism to A: the floor is cut in one level by pass A and the 1 mm strip of
  pass B; the axial force changes between them by < 100 N: **flatness <= 0.005 mm** (tolerance 0.02).

## 5. Surface

* Walls (peripheral edge): per-tooth cusp `fz^2/(8R)` = 0.14 um, but with a radial TIR of 12 um the most
  protruding insert leaves the surface once per revolution, mark spacing `z fz` = 0.48 mm and height
  `(z fz)^2 / (8 R)` = 2.30 um: **Rt 1.5-3 um, Ra 0.4-0.9 um** (Ra = Rt/3.9 for a parabolic cusp).
* Floor: the AVMT wiper flat (1.1 mm) exceeds the feed per revolution (0.48 mm), so the kinematic floor
  roughness is below 0.5 um; the floor shows the insert height scatter and vibration:
  **Ra <= 1.6 um** (spec 3.2).

## 6. Coolant (ICS 40 bar through the tool)

* Ideal jet velocity `sqrt(2 dp / rho)` = 89.8 m/s; with a discharge coefficient 0.6-0.8 the four
  1.0 mm holes pass **10-13.5 L/min** (below the 20 L/min pump estimate, so the pump runs at its 40 bar
  limit); the jets leave the flutes at 70-90 m/s.
* If all slotting heat went into the through-tool flow the coolant would warm by 1.7 K: the tank
  temperature rise over the job is **< 0.5 K**.
* The cut is wetted at every tooth entry (jets from the flutes into the slot): the coolant domain must report
  the cutting zone reached in both full-slot passes, where flood nozzles (n1-n3, unused here) would be
  shadowed by the slot walls (qualitative, no hand calculation).

## 7. Chips and mass

* Ti-6Al-4V segments at all speeds above 11 m/min (db `chip.segmentation`): **segmented chips** in all
  passes; chip thickness about 1.5-3 x h (no hand value).
* Removed volume = 100 x 26.024 x 8 - 2 x 100 x (0.8^2 - pi 0.8^2/4) = 20 792 mm3 -> **92.1 g**
  (rho 4430 kg/m3). Mass ledger exact to `tol_mass` (1e-9).

## 8. Tool wear

Taylor (db `tool_life`, carbide TiAlN, est.): slotting at vc 45, fz 0.08, ap 3.9 gives T = 168 min of edge
contact to VB 0.3 mm; finishing T = 46 min. Each edge is in contact 50 % of the slotting time (43 s) and
4.6 % of the finishing time (2.2 s): life used 0.4 %. **VB_max after the job <= 0.02 mm**; the tool table
life used is 2.3 min of the 30 min limit (cutting time, all edges).

## 9. Cycle time

Feed time (exact from the program): slotting 2 x 132 mm / 183 = 86.6 s, finishing 2 x 132 / 336 = 47.1 s,
air feeds at F2000 0.35 s, dwell G4 F2 = 2.0 s: **lower bound 136.0 s**. With the tool change
(chip-to-chip 5.5 s est.), spindle ramp (< 0.1 s to 573 rpm), rapids at 42 m/min and acceleration:
**expected 140-150 s**. In-cut time (cutter over the plate) about 108 s.

## 10. Conservation

Time: sum of block times = cycle time (integer ticks). Mass: removed field volume x rho = chip mass in all
states. Energy: spindle shaft work ~ 1.18 kW x 69 s in-cut slotting + 0.11 kW x 39 s finishing
(about 85 kJ, measured-coefficient end 50 kJ) balances heat + chip kinetic energy within `tol_energy`.
