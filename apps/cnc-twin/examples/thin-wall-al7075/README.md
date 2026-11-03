# Thin walls, Al 7075-T6 (Haas VF-2SS)

Two free-standing walls ("fins"), 1.0 mm thick, 25 mm tall and 60 mm long, on a 7 mm base, cut
from a 90 x 60 x 32 mm block of 7075-T651 plate. The job is about what thin walls do while they
are cut: they and the long tool bend away from the cut, so the walls come out thicker at the top
than at the root, and a wall whose first mode sits near a harmonic of the tooth frequency rings or
chatters.

| | |
|---|---|
| machine | `haas-vf2ss`, 12 000 rpm, flood emulsion |
| material | `al7075-t6` (ISO N, 150 HB, E 71.7 GPa; Kienzle kc1.1 800 N/mm2, mc 0.25, carbide) |
| stock | 90 x 60 x 32 block, rolled plate |
| workholding | LANG Makro-Grip 48120-46, 60 mm across the jaws (y), 3 mm form-fit grip, 60 Nm (14 kN) |
| program | `op10.nc`, written by `make_program.py` |
| CAD | `part.step` (AP242), written by `make_cad.py`; tolerances by hand in `project.yaml` (no PMI) |

## Setup

Stock frame = CAD frame (origin at the lower-left-bottom corner, top z 32). The vise sits on the
table centre (workholding frame at machine X -381, Y -203.2, jaw seat Z -510: 100 mm above the
table, assumed); the stock is centred (`stock_pose` -45, -30, 0); G54 = X -426.0, Y -233.2,
Z -478.0 (stock corner, top). Fins at y 19.5..20.5 and 39.5..40.5, x 15..75.

## Tools

| T | item | holder | stickout / H | use |
|---|---|---|---|---|
| 1 | `helical-59766-mh35al-025-31000` D10 3FL LOC 25 | `haimer-40-720-32` ER32 | 35 / 105 | roughing |
| 2 | `helical-59766-mh35al-025-31000` (a second one) | `haimer-40-720-16` ER16 (28 mm nut) | 42 / 112 | ladder finishing, L/D 4.2 |

The database has no reduced-neck (long-reach) square end mill; Helical's reduced-neck aluminium
line (H35AL-RN) exists but no metric item was read. The finisher is therefore the 25 mm LOC D10 at
a long stickout in the slim ER16 chuck: the full wall height is inside its flutes (LOC 25 = wall
height) and the 28 mm nut stays 17 mm above the fin tops.

Cutting data (Helical wrought aluminium, tool database): roughing fz 0.0965 mm ('Rgh'), ap 12.5 mm
(<= 2 D), ae 2.375 mm peel / <= 2.3 mm trochoid equivalent (<= 0.4 D); finishing fz 0.0584 mm
('Fin'), ae 0.5 mm (<= 0.06 D), ap 5 mm. Spindle-limited at 12 000 rpm (377 m/min against Helical's
640 m/min starting value): F 3474 and 2102 mm/min.

## Program

1. **N10 roughing** (T1), two levels Z-12.5 and Z-25.0, each: constant-engagement peel passes
   from the front stock face towards fin 1 (8 passes, moving -X, climb) and from the back face
   towards fin 2 (8 passes, moving +X), returning 1 mm off the new wall at depth (F8000, no rapid
   below the stock top); a trochoid (r 4.0, 1.0 mm per loop, counter-clockwise) along the 18 mm band
   between the fins; the 2 mm strips beyond the fin ends. 0.5 mm stays on every fin face and end;
   the base top (Z-25) is final. 2-D material check: trochoid engagement mean 40 deg, max 57 deg.
2. **N20 ladder finishing** (T2): five levels of 5 mm (Z-5 ... Z-25); at each level one
   clockwise (climb) contour 5 mm off each fin, so both faces of the band are finished while the
   2 mm rough wall below still carries it. Tool-centre path with wear compensation (G41 D02,
   D02 = 0); R5 tool-centre arcs round the fin ends; lead-in along the face from beyond the fin
   end, lead-out arc into the cleared band.

## What the run is meant to show

* wall deflection and tool deflection leaving material: the walls taper, thicker at the top
  (strip model in `make_program.py`: wall stiffness at the cut 0.57 N/um at the top level up to
  19-46 N/um at the root level; the long tool 3.2-7.2 N/um);
* the walls' first mode moving with the ladder (2.6-3.3 kHz while the rough root is still there,
  1.3 kHz for the finished fin) against the tooth frequency of 600 Hz and its harmonics: forced
  response and the chatter risk at the upper levels (EXPECTED.md section 4);
* witness lines at the ladder steps (Z-5, -10, -15, -20) from the different deflection of
  adjacent levels;
* constant-engagement roughing near the fins without overloading them.

## Files

`project.yaml` · `op10.nc` · `part.step` · `make_cad.py` · `make_program.py` · `EXPECTED.md`
