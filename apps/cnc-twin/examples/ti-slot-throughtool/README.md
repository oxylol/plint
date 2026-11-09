# ti-slot-throughtool: Ti-6Al-4V slot with 40 bar through-tool coolant (DMU 50, SINUMERIK 840D sl)

## The job

An open slot 26 +0.05/0 mm wide and 8 mm deep is milled along the whole length of a Ti-6Al-4V
(grade 5, annealed) plate 100 x 70 x 30 mm: two full-width slotting passes of 3.9 mm with a 25 mm
indexable shoulder mill, then one finishing pass that climbs along wall A in +X and back along wall B
in -X at the final depth, taking 0.512 mm per wall and the last 0.2 mm of the floor. Coolant is the
machine's internal coolant supply (ICS) at 40 bar through the spindle and the cutter body.

The part model `part.step` (from `make_cad.py`) is drawn at the middle of the width tolerance
(26.025 mm) with the R0.8 floor corners the inserts leave, so the program, which is the tool-centre
path for R 12.500, aims at the middle of the zone.

## Machine and controller

`db/machines/dmg-dmu50-3rd-siemens.yaml`: DMG MORI DMU 50 3rd generation, SK40 speedMASTER spindle
15 000 rpm / 21 kW (S6), SINUMERIK 840D sl. The job runs at B0 C0. Program `ti_slot.mpf`:
`G17 G40 G71 G90 G94`, `G54`, `T1` / `M6` / `D1`, CYCLE832 high-speed settings with the machine
file's tolerance bands (`CYCLE832(0.05,_ROUGH,1)` for slotting, `CYCLE832(0.005,_FINISH,1)` for the
finishing pass, `CYCLE832()` to deselect), `M7` for the ICS pump, a 2 s dwell for the pressure to build,
and `G0 G153 Z0 D0` before `M30`.

The ICS pump is in the machine file (`coolant.pumps[id: ics]`, option, up to 40 bar). Which M code
switches it is a machine-builder choice; the project declares it in `physics.controller.m_codes`
(`M7: {kind: coolant, pump: ics}`), so it is in one place if the real machine uses another code.

## Setup

* LANG Quick-Point 52 grid plate 45600 (104 x 104 x 27) on the rotary table, LANG Makro-Grip 77 vise
  48085-46 (jaw width 46, clamping range 0-85, height 65 mm per the LANG data sheet) on it, centred on
  the C axis. The plate is pre-stamped and gripped 4 mm deep across its 70 mm width (jaws on y = 0 and
  y = 70), 14 kN holding force at 60 Nm.
* Workholding frame: centre of the jaw seat, 88 mm above the table (27 + 65 - 4). In machine
  coordinates (Z of the gauge line, table top at Z -625): `fixture_pose` (0, 0, -537).
* G54 = centre of the plate top: (0, 0, -507). The slot centreline is Y0, the plate spans X -50..50.

## Tool

| T | tool (db id) | holder | stickout | L1 / R |
|---|---|---|---|---|
| 1 | `tungaloy-epav12m025c250r04-avmt1204-ah3225`: TungForce-Rec EPAV12M025C25.0R04, 4 x AVMT120408PDER-MM AH3225, D25, RE 0.8, APMX 11.5, cylindrical shank 25 with coolant holes (`coolant_holes: 4 x 1.0`, est.) | `haimer-40-445-25-3` HAIMER Power Shrink SK40, bore 25, gauge 75 | 60 | 135.000 / 12.500 |

The tool database has no solid end mill with coolant holes; the TungForce-Rec body is the database tool with
through-coolant that fits, and slotting is one of its listed applications (506-G p. 26). Cutting data
from its ISO S rows (vc 20-60 m/min, fz 0.04-0.15 mm): slotting vc 45 / fz 0.08, finishing vc 55 / fz 0.12.

## What the run is meant to show

* Full-width slotting in titanium: a 1.2 kN side force that changes sign with the feed direction, a
  20 N/um tool-holder-spindle stack that deflects by 35-60 um, so the rough walls are stepped between
  the two depth bands and the finishing allowance varies over the height.
* The finishing pass in climb: small mean wall forces but 1.2-1.6 kN single-tooth entry impulses, the
  floor layer on the first wall, and the resulting undercut of each wall: the slot ends about 11-17 um
  narrower than programmed, still inside the tolerance because the model is at the middle of the zone.
* Through-tool high-pressure coolant in a closed slot: jet flow and velocity from the 40 bar supply and
  the four 1 mm holes, the cutting zone wetted where flood nozzles are shadowed, chips flushed out of the
  open slot ends.
* Segmented titanium chips, low tool wear over one part, mass conservation and an exact cycle time.

Numbers and acceptance bands: `EXPECTED.md` (calculations in `expected_calc.py`).

## Files

| file | what |
|---|---|
| `project.yaml` | the project (resolves with `cnctwin.project.resolve`) |
| `ti_slot.mpf` | SINUMERIK 840D sl program |
| `make_cad.py` -> `part.step` | part model (OCP, deterministic STEP header) |
| `expected_calc.py` | hand calculations behind EXPECTED.md |

## Notes and limits

* The controller's SINUMERIK 840D sl dialect (`siemens840d`, docs/dialects.md) runs the program on
  `dmg-dmu50-3rd-siemens` (the machine model builds; the machine file now extends `dmg-dmu50-3rd` and writes
  only its controller): the ideal dry run (no acceleration) reaches M30 with no alarm in 142.7 s, 134.0 s of
  it in feed, which is the feed time of EXPECTED.md section 9 (checked 2026-11-09 with the controller's dry
  run under bin/tsafe). The program uses only words listed in `db/controllers/siemens-840d-sl.yaml`. The
  static program checks (cnctwin.sanity) read only FANUC and Haas syntax themselves: they report one
  `scan-limits` finding for this program and no other rule.
* The vise height (65 mm) and the 4 mm grip are setup values for the work offset, not database values.
