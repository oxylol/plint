# lathe-shaft-4140: shaft between chuck and live centre on the Haas ST-20Y

## The job

A shaft of 42CrMo4+QT (AISI 4140, ~30 HRC, `db aisi4140-qt30`) is turned from a d45 bar sawn to 230 mm,
faced and centred (DIN 332-1 A3.15) beforehand:

| program Z | feature |
|---|---|
| 0 .. -20 | M24 x 1.5 - 6g thread (chamfer 1.5 x 45) |
| -20 .. -23 | relief groove 3 x d21.6 |
| -23 .. -60 | d30 g6 bearing seat, Ra 0.8 |
| -60 .. -140 | d35 h7 hub seat, Ra 1.6 |
| -140 .. -180 | d42 collar; part-off face at Z-180 |

Datums A = d30 seat, B = d35 seat; runout of d35 to A 0.015, of the hub face and the thread to A-B
0.02 / 0.05; cylindricity of d30 0.008; length 180 +- 0.1, shoulders 60 / 140 +- 0.05.

Order of work in `shaft.nc`: tailstock in (M21), T1 OD roughing G71 (D 2.5, f 0.3, vc 200), T2 finishing
G70 along the same contour (f 0.12, 0.08 on the d30 seat, vc 220), T3 relief groove (single plunge),
T4 thread G76 (11 passes, flank infeed P1, first depth 0.3 per Tungaloy E066), tailstock out (M22), T5
part-off. Flood emulsion throughout.

## Machine and setup

`db/machines/haas-st20y.yaml`: Haas ST-20Y, A2-6 spindle 4000 rpm / 14.9 kW (peak), 12-station BMT65
turret, tailstock option (MT4 live centre; MT4 is the taper of the ST-20 series tailstock).

* Chuck `kitagawa-b-208` (8 in, 86 kN max grip, hydraulic at 15 bar) with soft jaws `kitagawa-sj08b1`
  bored to d45.00 x 30 deep: chuck body 91 mm, jaw front 129 mm from the spindle face; the bar end
  sits on the jaw bore bottom at z 99; 200 mm stick out of the jaws.
* Live centre `roehm-hm-pro-mt4-108` (MT4, 3800 rpm max, 5 um concentricity) in the centre hole at the
  free end, 329 mm from the spindle face (tailstock range 302-836 mm).
* G54: X -426.0 (spindle centreline in machine X diameter of the turret's OD tool-tip reference),
  Z -333.0 (free end face: 329 - 662). The spindle limit G50 S2500 respects the live centre (3800 rpm)
  and the chuck (5000 rpm).

## Tools (BMT65 turret, stations 1-5)

| T | db id | use |
|---|---|---|
| 1 | `tungaloy-cnmg120408-tm-t9225-pclnr2525m12e` | OD roughing, RE 0.8, 95 deg |
| 2 | `tungaloy-vcmt160404-ps-t9215-svjcr2525m16` | OD finishing, RE 0.4, 93 deg |
| 3 | `tungaloy-dgm3-020-ah7025-cter2525-3t20` | relief groove, CW 3 |
| 4 | `tungaloy-16er15iso-ah725-cer2525m16dt` | thread M24x1.5, full profile |
| 5 | `tungaloy-qgm3-020-ah7025-qsp26-3d` | part-off, CW 3 (cut-off d up to 75) |

Cutting data from the Tungaloy rows in the tool files (aisi4140-qt30 rows where they exist, else the first
ISO P row): see the program header and EXPECTED.md section 1.

The contour is programmed for the imaginary tip of T2 (no G41/G42): the 45 deg chamfers are moved by
`r (2 - sqrt 2)` = 0.234 mm (radius) at both ends, which is what a CAM post does with "compensation in
computer"; cylinders and faces need no correction. T3 and T5 are programmed on their +Z corner.

## What the run is meant to show

* The spindle load of a real roughing cycle: 1.6-2.0 kN cutting force, 5.4-6.8 kW, 70-111 % of the S1
  rating as the CSS speed rises on the small diameters.
* Push-off: the turret and the shaft between chuck and live centre yield by 0.03-0.04 um/N, so the
  finishing pass leaves the seats 4-8 um over the programmed diameter; the g6 seat is drawn and
  programmed 4 um below the middle of its zone for that.
* Round-nose roughness Ra = f^2/(31.2 r) on the two seats (0.51 and 1.15 um).
* Thread cutting synchronised to the spindle over 11 passes with a full-profile insert that tops the
  crest, running out into the relief groove.
* Part-off of a 1.3 kg part with the tailstock retracted and no parts catcher: a free body at X0.
* Flood cooling that keeps the bar within a few kelvin, mass and time conservation.

## Files

| file | what |
|---|---|
| `project.yaml` | the project (resolves with `cnctwin.project.resolve`) |
| `shaft.nc` | Haas NGC lathe program |
| `make_cad.py` -> `part.step`, `stock.step` | finished shaft (with the helical thread) and the centred bar |
| `expected_calc.py` | hand calculations behind EXPECTED.md |

## Notes and limits

* The machine file has no parts catcher, so the part drops into the chip pan at part-off.
* In this build the controller has no provider for the lathe cycles G70/G71/G76 (TW0001), and the Haas
  dialect reads the `D` word of a G71/G76 block as an offset number, so `D2.5` raises alarm 311
  (decimal point). Haas writes D with a decimal point in these cycles, so the program keeps it. With the
  three cycle blocks replaced by plain moves the rest of the program dry-runs to M30.
* The project format cannot carry lathe X/Z tool geometry offsets (the controller reads `x_offset_m`,
  `z_offset_m`, `nose_radius_m`, `tip`, which the project schema does not declare), so the offsets are
  left to the presetter (`length_offset_mm: null`). The tailstock stiffness is not in the machine or the
  centre's database item; EXPECTED.md brackets it (30-60 N/um).
