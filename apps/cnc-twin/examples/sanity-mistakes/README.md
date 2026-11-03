# Sanity mistakes: one of each (Haas VF-2SS)

A small plate job (slot, spot, drill, tap, slot) whose program `op10.nc` contains every mistake
the program-sanity checks know, each exactly once and on a known line. It is the example version
of `tests/fixtures/sanity_all_mistakes.nc`: the same 23 rules (all of `cnctwin.sanity.RULE_IDS`
except `scan-limits`), but on the real `haas-vf2ss` with database tools, so it also tests the
context the checks build from a resolved project (machine coolant, maker cutting data, vise
jaws, work offsets). `expected_findings.yaml` lists line -> rule -> severity; nothing else may be
reported. The job is not meant to be run (the control stops at the first mistake, see
EXPECTED.md).

| | |
|---|---|
| machine | `haas-vf2ss`: flood, air blast, through-spindle option; **no shower coolant** |
| material | `al6061-t6` (ISO N: coolant required by the sanity tables) |
| stock | 100 x 80 x 20 plate |
| workholding | LANG Makro-Grip 48120-46; `grip_depth_mm: 24` puts the jaw tops 4 mm above the 20 mm plate (the plate sits low), which is what makes an R2 retract plane wrong |
| work offsets | G54 correct (stock corner, top); G55 = X -850 (puts the part outside the X travel); G56 not set |
| tools | T1 `helical-59766-mh35al-025-31000` D10, T2 `osg-62910-nc-lds-10x90` spot drill, T3 `osg-60568-ex-gdr-6.8` HSS drill, T4 `osg-8325107-a-pot-m8x1.25`, T5 `helical-59726-mh35al-020-30600` D6, T6 `helical-59744-mh35al-020-31200` D12 |

## The mistakes

| line | block | mistake | rule |
|---|---|---|---|
| 7 | `(T5 D12 3FL CARBIDE ENDMILL)` | header comment does not match the tool table (T5 is a D6) | tool-comment |
| 12 | `T7` | tool number not in the tool table | tool-unknown |
| 16 | `G01 Z-2.` | feed move without a feed rate | feed-missing |
| 20 | `Z-1.` | rapid below the stock top (into the slot) | rapid-below-stock |
| 21 | `G01 Z-15.` | slotting 13 mm deep: Helical's slotting ADOC limit is 1.25 D = 12.5 mm | cutting-data |
| 25 | `G28 Z0.` | G28 without G91: the intermediate point is work Z0, below the jaw tops | g28-absolute |
| 30 | `G99 G81 Z-3. R2.` | no coolant (M08 missing) for aluminium | coolant-required |
| 30 | same | the drill feed is inherited from T1 (F1000) | feed-inherited |
| 30 | same | R plane Z2 below the jaw tops Z4 | retract-below-clamps |
| 37 | `G43 H03 Z25. M07` | M07 shower coolant, which the machine does not have | coolant-unavailable |
| 38 | `G98 G81 ...` (spindle M04 at line 36) | right-hand drill turning M04 | spindle-direction |
| 46 | `G95 G98 G84 Z-14. ...` | tap deeper than the drilled hole (Z-14 in a Z-12 hole) | tap-depth |
| 49 | `N50 (...)` | restart at N50 runs in G94; the program reaches it in G95 | restart-modal |
| 52 | `Z5. M08` | T5 moves in Z with the G43 H4 of T4 (no G43 after M06) | missing-g43 |
| 53 | `G01 Z-14. F0.05` | D6 with 12 mm of flute plunging 14 mm | flute-length |
| 53 | same | cutting with the spindle stopped | spindle-stopped |
| 56 | `T6 M06` | tool change with the tool at Z2, over the part | toolchange-clearance |
| 57 | `G90 G94 G55 ...` | G55 puts the part outside the X travel (X -840 < -762) | part-outside-travel |
| 58 | `G43 H00 ...` | length offset H00 | offset-length |
| 61 | `G41 D11 X40.` | radius offset D11 not set | offset-radius |
| 62 | `G02 X50.006 Y10. I5. J0.` | arc end 0.006 mm off its radius (above the 0.002 mm rounding allowance, below the 0.02 mm alarm) | arc-endpoint |
| 65 | `G56 X10. Y10.` | work offset G56 not set | work-offset-unset |
| 69 | `M30` | program end with spindle and coolant still on | program-end |

**Restart line**: N50 (file line 49). Started there, the control is in its power-on G94, so
`F0.05` at line 53 is 0.05 mm/min instead of 0.05 mm/rev; the program itself reaches N50 in G95
(set at line 46 for the tap).

## Differences from the fixture

The fixture (`tests/fixtures/sanity_all_mistakes.nc` with `sanity_setup.yaml`) uses a synthetic
mill without through-spindle coolant, inline tools with min/max cutting data and strap clamps with
`outline_mm`. Here: the VF-2SS lists the TSC option, so the unavailable coolant is M07 (shower);
Helical publishes only starting values for speed and chip load, so the cutting-data mistake is a
depth (ADOC) beyond the maker's limit; a valid project cannot give a clamp outline
(`outline_mm` is not a project key), so the clamps are the vise jaws standing above the plate and
`retract-below-clamps` is the retract-plane warning rather than the crossing error.

## Files

`project.yaml` · `op10.nc` · `expected_findings.yaml` · `part.step` (the intended part, `make_cad.py`) · `EXPECTED.md`
