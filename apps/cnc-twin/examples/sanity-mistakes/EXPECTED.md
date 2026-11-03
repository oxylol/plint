# Expected behaviour: sanity-mistakes

## 1. Program sanity (the regression test)

`cnctwin.sanity.check_operation(resolve(project.yaml), 0)` returns exactly the 23 issues of
`expected_findings.yaml` (rule, line, severity), one per rule of `RULE_IDS` except `scan-limits`,
and nothing else; `report.safe()` is False (13 errors). The comparison is exact: sorted
`(rule, line, severity)` triples. Checked against this build: 23 = 23, no missing, no extra.

Where the values come from (so a change in the database or the rules can be told from a broken
check):

| rule | quantity in the finding | source |
|---|---|---|
| cutting-data (21) | ap 13 mm vs 12.5 mm = 1.25 D, ratio 1.04 < 1.5 -> warning | Helical slotting ADOC (tool db `ap_max_xD: 1.25`), the slot floor of the first level at Z-2 |
| feed-inherited (30) | F1000 from line 23; 1000 / 3000 rpm = 0.333 mm/rev, inside OSG's 0.22-0.36 | so that only feed-inherited fires, not cutting-data |
| retract-below-clamps (30) | R plane Z2 below the jaw tops Z4 = stock bottom + grip_depth 24 - 20 | `_vise_jaws` (docs/models/sanity.md, Clamps) |
| g28-absolute (25) | intermediate Z0 below the highest of stock top (Z0) and jaw tops (Z4) | |
| coolant-unavailable (37) | M7 -> needs "shower"; the VF-2SS has no shower nozzle or pump | `cnctwin.sanity.scan.COOLANT_ON["haas"]`, machine `coolant` |
| tap-depth (46) | tap end Z-14 vs drill tip Z-12: 2.000 mm | |
| part-outside-travel (57) | X = 10 + (-850) = -840 mm, travel [-762, 0], 78 mm beyond | machine `axes.X.travel_mm` |
| arc-endpoint (62) | radius 5.000 -> 5.006 (0.006 mm): above 2 x 0.001 mm (rounding allowance), below the 0.02 mm alarm value | `arc_accept_increments`, `arc_alarm_tolerance_m` (db/models/sanity-rules.yaml) |
| toolchange-clearance (56) | tip at Z2, needed max(Z0, Z4) + 25 = Z29 | `toolchange_clearance_m` |
| flute-length (53) | 14 mm deep vs 12 mm flute (MH35AL-020-30600 LOC) | |

## 2. The control

The controller stops at the first block it cannot execute: **Haas alarm 320 "No Feed Rate
Specified" at line 16** (`G01 Z-2.` before any F), after the tool change and the approach (dry run
of this build: alarm at line 16, 3.5 s into the program). Nothing is cut. `T7` at line 12 is a
valid pre-selection for the control (a carousel pocket), so it does not alarm.

## 3. What each mistake would do if the program ran past it

For the physics domains these are the consequences to expect if a variant of the program removes
the earlier stops; magnitudes from the setup:

* line 20-21: the rapid stops 1 mm above the slot floor (no contact); the plunge `G01 Z-15.` at
  F3474 gives the end teeth an axial chip of 0.0965 mm: kc = 800 x 0.0965^-0.25 = 1435 N/mm2,
  tangential force per end tooth kc fz R = 1435 x 0.0965 x 5 = 0.69 kN, torque about
  3 x 0.69 kN x 2.5 mm = 5.2 N m, and a thrust of the same order (kN), far above the slot's
  side-milling load;
* line 25: the tool goes to work Z0 at X30 Y39 (over the open slot) before going home: no contact
  here, a collision wherever a jaw or the part stands above Z0 under the tool;
* line 30: the spot drill cuts dry (aluminium adhesion); the R2 plane is below the jaw tops, but
  these two holes (y 68, tool edge at y 73) do not pass over a jaw (y >= 80);
* line 37: M07 is not a device of the VF-2SS: the twin's TW0010 (M code without its device);
* line 38: a right-hand drill turning M04 does not cut: thrust and heat rise until the drill breaks;
* line 46: the tap reaches the drill-point cone 2 mm below the hole bottom: torque spike, broken tap;
* lines 49-53: T5 runs with H04 = 115 mm while its own length is 95 mm, so its tip is 20 mm above
  the programmed path and the "14 mm deep" slot is cut in air; with the spindle stopped and G95,
  the per-revolution feed is zero: the axes stall (the dry run reports the stall);
* line 56: Haas M06 retracts Z before the change, so the VF-2SS itself is safe here; the rule
  guards against builders whose changer moves X/Y first;
* line 57: overtravel alarm (X -840 beyond the -762 limit) before the move;
* line 58: H00 leaves the 104 mm tool length out of Z;
* lines 61-62: D11 is 0 (no offset); the 0.006 mm radius mismatch is inside Haas's arc
  tolerance (0.0254 mm), the control blends it into a spiral without an alarm;
* line 65: G56 is all zeros: X10 Y10 is beyond the + end of the travel (overtravel);
* line 69: Haas M30 stops spindle and coolant anyway; the warning is about practice.
