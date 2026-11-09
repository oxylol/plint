# Program sanity checks (`cnctwin.sanity`)

The machinist's checks a control would not catch, run on a program before (or alongside) the
simulation. Each problem is a finding of category `sanity` (cnctwin/core/findings.py) tied to its
program line and block, with the tool-tip position, and a chain of cause that points at the lines
the problem rests on (the block that set the inherited feed, the tool change, the work offset).
The checks are static: they do not run the physics; their geometric part uses a height-map
estimate of the engagement (section 5). There is no random draw (no RNG streams) and no bus record:
sanity is not a domain, it reads the program, the project and optionally the controller's command
stream, and writes findings.

## 1. Use

```python
from cnctwin.sanity import check_operation, check_program, SanityContext

report = check_operation(project, op_index, commands=None, restart_analysis=None,
                         trace=ctx.trace, timeline=None)          # resolved project (architecture 18)
report.add_to(state.findings)                                     # or report.findings / report.issues

ctx = SanityContext.from_project(project, op_index)               # or build one by hand
report = check_program(text, "op10.nc", ctx, commands=None, subprograms={"O2000": text2})
```

* `check_operation(project, op_index, *, program_text=None, commands=None, subprograms=None,
  restart_analysis=None, trace=None, timeline=None, project_dir=None, stock_box_M=None,
  clamps_M=None, use_controller=True) -> SanityReport` reads `operations[i].program_text` (else the
  file `project_dir/operations[i].program`); with `use_controller` (and no `commands`) it runs the
  controller's dry run of the operation for what the scan cannot evaluate (section 3a).
* `check_program(text, name, ctx, *, commands=None, subprograms=None, restart_analysis=None,
  trace=None, timeline=None, tick_s=1e-6, assist=None) -> SanityReport`.
* `SanityReport`: `issues` (rule id, severity, message, line, program, block, command index,
  quantities, causes, pos_W_m), `findings` (core `Finding`s, ids -1 until added to a run's
  `FindingList` with `add_to`), `program` (the view), `engagement`, `trace`, `context`, `assist`
  (the controller assist consulted, or None; `assist.note` says why there is none, `assist.alarm`
  the alarm the dry run stopped at), `safe()`.
* Findings: `category=sanity`, `domain="sanity"`, `message="<rule-id>: <text>"`, `line`, `block`,
  `op`, `program`, `pos_W_m` (intended tool tip, program coordinates of the active work offset),
  `pos_M_m` (that plus the work offset), `quantities` (SI, keys carry units), `cause_chain`: trace
  events `finding:sanity:<rule>` (the finding) with parents `sanity.cause:<why>` whose `line` is
  the cause line. With a `trace` the events go into the run's trace log; otherwise a private
  in-memory log is returned in the report. With a `timeline` (blocks with `program`, `line`,
  `start_tick`) a finding carries the tick at which its line first runs; static findings carry
  tick 0.

## 2. Program view

`cnctwin.sanity.program`: `lex(text)` gives one `Line` per physical line (1-based): address words
(`Word(letter, value, text, has_point)`), comments (parentheses; text after `;` is kept as a
comment: it is the end-of-block mark on Fanuc/Haas), `N` number, block delete `/`, `O` number,
`%`. A macro statement (`#100 = 2`, `IF`, `WHILE`, `GOTO`, any multi-letter keyword) or an address
whose value is an expression (`X#101`, `Z[#1 - 2]`) marks the line `macro`. `Program` holds the
lines of every program file read (`texts`), the commands (in execution order), the tool-tip
positions `tip_start_W_m`/`tip_end_W_m` of every command (NaN where unknown), the source
(`scan`, `controller` or `none`) and the scan result.

Tool comments (`parse_tool_comment`): T number (`T1`, `TOOL 3`), diameter (`D10`, `DIA 10`,
`Ø10`, `10MM`, `10 DIA`, a fraction `1/2` in inches, else a single bare number), kind (face mill,
ball, bull nose, flat, end mill/EM, thread mill, spot drill, centre drill, tap, reamer, chamfer,
boring, engraver, drill; `TAP DRILL` is a drill) and metric thread (`M8X1.25` on taps and thread
mills). A comment describes a tool when it gives a diameter, a thread or a T number.

## 3. Commands: the controller's stream or the own scan

The rules read `cnctwin.controller.ir.Command` objects with their `modal` snapshot. When the
controller's stream is given (`commands=`), it is used; positions come from `tip_target_m` for
`Space.TIP` commands, else from the machine-axis `target` of a 3-axis mill
(`x_W = X - offset_x`, `z_W = Z - offset_z - h`, `h` the active H register, 0 for G49 or H00).
Modal keys the controller's snapshot does not carry are filled from the own scan of the same
program line (controller values win), and the scan's holes, tool sections, G28 events and
program-set offsets are re-indexed to the controller's commands by line and kind. As in the scan,
the program starts at the machine home and a tool change without targets leaves Z at home.

**Modal keys read** (the own scan writes all of them; a controller snapshot should use the same
names): `units` (`mm`/`inch`), `distance` (`absolute`/`incremental`), `plane` (`XY`/`ZX`/`YZ`),
`motion`, `work_offset` (`G54`, `G54.1P3`, `G110`), `work_offset_line` (0 = power-on default),
`tool`, `tool_line`, `next_tool`, `section` (tool section index, 0 before the first tool change),
`length_comp` (bool), `length_comp_code` (`G43`/`G44`), `length_offset` (H), `length_line`,
`length_section`, `radius_comp` (`G40`/`G41`/`G42`), `radius_offset` (D), `feed_mode`
(`ir.FeedMode` value), `feed` (SI: m/s, m/rev or 1/s), `feed_line`, `feed_tool`, `feed_section`,
`spindle_dir` (`M3`/`M4`/`M5`), `spindle_rpm`, `spindle_rad_s` (signed, M03 < 0), `spindle_on`,
`spindle_line`, `css_m_s`, `max_rpm`, `coolant` (active codes), `cycle`, `cycle_retract`,
`cycle_info` (`{code, phase, r_W_m, z_W_m, initial_W_m, retract_W_m, retract_mode}` on the
commands of an expanded canned cycle), `machine` (`mill`/`lathe`), `dialect`, `call_depth`,
`arc_chord_too_long`. Coolant commands carry `coolant={code, needs, on}`, tool changes
`tool={number, ...}`, pre-selections `Kind.MODE` with `tool={number, preselect: True}`.

### Own modal scan (`cnctwin.sanity.scan`)

Fanuc and Haas word-address programs, mill and lathe (ISO 6983-1; the Fanuc 0i and Haas NGC
manuals' modal groups: haas-ngc-mill-om-2020, haas-ngc-lathe-om-2020). It follows M98/M99
(external `O` numbers given in `subprograms` or later in the same file, repeat `L`/`K` or the Fanuc
8-digit `P`), Haas M97 local subprograms (N labels), up to `max_call_depth` levels, and expands
canned cycles (G73/G74/G76/G81-G89, G98/G99, R/Z absolute or G91 incremental, `K`/`L` repeats)
into rapid-to-position, rapid-to-R, feed-to-Z, dwell and retract commands with a `Hole` record
each. It tracks G52/G92 shifts, G10 L2/L20 work offsets and L10-L13 tool offsets, G37 length
measurement, G28/G30 (with the intermediate point) and G53 moves. Lathe semantics: X in diameter,
U/W/V incremental, T`ttoo` tool and offset words (a value under 100 is tool = offset), G96/G97/G50,
G98/G99 feed modes, G18 default.

Order inside a block: unit codes, the other G codes (modal groups), F and S, the T word, the
"before motion" M codes (M03/M04/M06/M19/M29 and the coolant-on codes), one-shot G codes or the
motion, then the remaining M codes (M05, M09, stops, M30/M02, calls).

Assumptions (each a choice with its consequence):

* the program starts with the tool at the machine home (`kinematics.home_mm`), G49, G40, G80,
  G90, G94 (lathes G99), G17 (lathes G18) and the units of `default_units`; the tool in the
  spindle is `setup.initial_tool` (additive project key) or unknown;
* M06 stops spindle and coolant and retracts Z to the home height (the tool-change macros of most
  builders do; Haas M06 retracts Z); programs that rely on the machine keeping them on get
  `spindle-stopped`/`coolant-required` findings;
* positions are the *intended* tool tip: the programmed path as if the compensations were right
  (`missing-g43`, `offset-length`, `offset-radius` report the compensation problems themselves);
  cutter radius compensation is not applied to the path (the engagement error is up to R on
  G41/G42 contours);
* H00 and G49 apply no length; Fanuc coordinates without a decimal point are least input
  increments (machine `least_input_increment_mm`, `fanuc_inch_increment_m` in inch), Haas whole
  units; block-delete blocks are executed (switch off);
* not evaluated by the scan alone, and reported as `scan-limits` info findings: macro expressions
  and statements (positions after an unknown value are NaN), G65/G66 macro calls, G68 rotation,
  G51 scaling, G15/G16 polar, mirror, TCP (G43.4/G234), DWO, Haas pattern cycles
  (G12/G13/G150/G70-G72, G47), G30 positions, G29, unknown G/M codes, subprograms not available, the
  lathe multiple-repetitive cycles' passes (G70-G76 and G90/G92/G94 are checked as one feed move to
  their end point). With the controller assist (3a) most of these come from the controller.
* lathe G71/G72/G73 with P and Q continue after the shape's last block N(nf) (the shape blocks are
  not run after the cycle, as on the control); G70 P Q runs the blocks N(ns)..N(nf) and then returns
  to its start point at rapid, so the finishing contour is checked with the finishing tool;
* `,C` / `,R` corner words are read and the programmed corner is kept (the chamfer or fillet lies
  within its size of it); the Haas device and check M codes (brakes, gears, clamps, M16, M78/M79,
  lathe M14/M15, M21/M22, M23/M24, M36/M37, M146/M147) have no effect on the scan's geometry.

Dialects other than Fanuc and Haas (Siemens 840D, Heidenhain) need the controller's command
stream; without it one `scan-limits` info finding says so and no other rule runs.

### 3a. Controller assist (`cnctwin.sanity.assist`)

`check_operation` runs the controller's dry run of the operation (cnctwin.controller.dry_run: the
ideal machine, with `cnctwin.controller.domain.stock_probe`, an ideal spindle probe against the
stock as set up) with the executor's block log on, and the scan asks it about the lines it cannot
evaluate. A line's k-th visit by the scan is matched with the k-th execution of that program line by
the controller (without macro flow the two orders are the same; with GOTO/WHILE the scan's single
pass takes the first execution). From the matched execution the scan takes:

* the values of addresses written as macro expressions (`G43 H#3026`, `G41 D#7`, `X#101`,
  `Z[#1 - 2]`): the values the executor evaluated, in program units;
* the motion of lines it does not expand, as the controller ran them (tool-tip positions in program
  coordinates of the active work offset, the controller's machine targets and tool axis): moves
  under tool centre point control (G43.4/G43.5, Haas G234), dynamic work offsets (G254) and tilted
  planes (G68.2) until their cancel; the lathe cycles (G70-G76, G90/G92/G94, the Haas lathe drilling
  cycles) through their cycle providers; the Haas mill pattern cycles (G12/G13/G150, G47) and bolt
  hole patterns (G70-G72, whose holes are recorded with their cycle phases like the scan's own
  canned-cycle holes); G65 macro calls and subprograms the scan cannot read (the motion of every
  block they run); G31 skip moves, which end where the probe latched; G codes the scan does not
  interpret.

The scan keeps its own modal state, holes, sections and notes; mill canned cycles G73-G89 keep the
scan's own six-operation model (its R, Z and return planes are what the depth and clearance rules
read; the providers' peck re-approaches inside the hole are not checked as rapids). A move taken from
the controller carries `modal source: controller`: `part-outside-travel` reads its machine targets
(the controller's inverse kinematics) instead of tip + offset + length. Lines after a controller
alarm have no execution and the scan reads them alone (the alarm itself is the controller's
finding in a run). When the controller cannot be built or run for the operation, the scan runs
alone and `report.assist.note` says why.

## 4. Context (`SanityContext`)

`SanityContext.from_project(project, op_index, *, stock_box_M=None, clamps_M=None)` reads, in SI:
the machine (dialect, type, least input increment, linear-axis travel, home, coolant pump kinds and
ids, nozzle ids, air blast, programmable nozzle, `controller.arc_radius_tolerance_mm` when given),
the material (`id`, `family`, `iso_group`), the coolant fluid kind, the tool table (`ToolFacts` per
number: type, hand, tool-material class, diameter, flutes, corner radius, point angle, flute and
cutting length, stickout, length and radius offsets, coolant holes, thread pitch, chamfer threads,
maker's `cutting_data` per ISO group, project comment, price, regrinds), the work offsets of the
operation, the stock box and the clamps. A tool given only by a database id that the resolver did
not resolve has `resolved=False` (geometry unknown: reported, see `scan-limits`). A probe
(`project.tools.probes`, or a raw `kind: probe` entry of `tools.list`) is a known T number of type
`probe`: its diameter is the stylus ball, it does not rotate and never cuts (the engagement estimate
checks its rapids only); its comment may name it `PROBE` (bare model numbers such as `OMP40-2` are
not read as a diameter for a probe). The stock box is the operation's own (`stock_box_S_m` of the
resolved project: the turned box of a `from_previous` operation), else the stock's size.

**Setup geometry convention.** For a 3-axis mill the table frame (where `fixture_pose` and the
work offsets are written, db/_templates/project.yaml) is the machine coordinate frame: the stock
box in machine coordinates is `fixture_pose * stock_pose * box_S` (block from the stock origin
corner; cylinder and tube about +z_S from the bottom face centre), program coordinates of work
offset `G5x` are machine coordinates minus the offset. Poses use the project convention
(`cnctwin.db.simap.pose_matrix`). Callers that know the kinematics better pass `stock_box_M` and
`clamps_M` (`Box(lo, hi, name)`, machine coordinates).

**Clamps.** A device is fixed to the workholding frame W, not to the part: its frame D is the
entry's `pose` in W (the identity when none is written), so in a flipped setup the clamps stay
where the fixture holds them. A workholding entry with `outline_mm: [[x0, y0, z0], [x1, y1, z1]]`
(a box in D, docs/formats.md) is used as it is; a vise or soft jaws give two jaw boxes in D by the
holding convention (cnctwin.holding.devices.build_vise: D's origin on the seat, the parallels' top
else the bed, x along the jaw width, y the clamping direction): the jaw faces at the part's y faces
(the stock box as placed in D), `vise_jaw_thickness_m` thick, `jaw_width_mm` wide (default: the
part's length) centred on D (on the part when it does not lie there), their tops `grip_depth_mm`
above the seat (else jaw height - parallels, else the part top), rising from the bed (z =
-parallels), or from the vise's mounting face when `seat_height_mm` gives the body below the seat;
vacuum, magnetic, adhesive, fixture plates, tabs and lathe chucks have nothing above the part; any
other device without an outline is listed as unknown (an info finding: moves are not checked
against it).

## 5. Engagement estimate (`cnctwin.sanity.geometry`)

Mills (tool axis +z of the program frame): the stock box of each work offset is a height map
(cell `zmap_cell_m`, grown until at most `zmap_max_cells` cells; a cell holds material when its
centre lies inside the box, and a size that is a whole number of cells up to round-off gets no extra
row: until 2026-11-09 a 12 mm block read 12.5 mm on the 0.5 mm map, from 0.012 / 0.0005 =
24.000000000000004), carved by every move with the
tool's lower envelope `z_tip + lift(r)`: flat end mill `lift = 0`; torus/ball with corner radius
`rc` `lift = rc - sqrt(rc^2 - (r - (R - rc))^2)` beyond `R - rc`; drills `lift = r / tan(point/2)`.

* A straight move at constant Z is evaluated as a swept segment, exactly: the envelope at a cell is
  `lift(d)` with `d` the cell's distance from the segment; an arc in XY is split into chords with a
  sagitta of a quarter cell; a vertical plunge is the disc at its lower end; ramps and helices are
  sampled along the path (step `min(cell, R/2)`, at least half a cell) with the front half of the
  disc (the half the tool moves into) evaluated against the heights before the move.
* A move cuts when material stands more than 1 um above the envelope. Material lies between the
  stock bottom and the cell's height: a cell carved below the bottom (a through cut) holds none.
  `ap` is the depth of the engaged material: from the highest engaged material down to the tip, or
  to the stock bottom when the tip is below it (a through hole or a profile cut past the bottom);
  `reach` how far below that material top the tip reaches (through or not). `ae` is the extent of the
  engaged cells across the feed direction plus one cell, capped at the diameter (for a plunge the
  larger of the x and y extents); on an arc in XY whose engagement is a crescent on one side of the
  path it is the radial depth measured from the arc's centre (outward: rho + R minus the nearest
  engaged cell's distance; inward, around a boss: the farthest minus rho - R; plus half a cell), the
  radial depth of cut on a circular path, where the swept width across a chord would count the
  crescent's length.
* Rapids: `gap` = the smallest clearance between envelope and material (negative: into it).
* Not estimated: probe moves other than rapids (a probe touches, it never cuts) and moves whose tool
  axis is tilted more than `max_tool_tilt_rad` from +z (five-axis moves from the controller assist:
  the 3-axis model does not apply; reported under `scan-limits`).

Error: `ap` and `ae` within about one cell (0.5 mm by default) plus, on arcs, the chord sagitta
(cell/4); ramps add the sampling step. The map knows the stock only (no previous operation's
result, fixtures are separate boxes, the part's CAD is not used). Lathes: a feed move cuts when its
radius is inside the stock radius; ap = stock radius - radius. Cost on this machine: about 0.2 ms
per block (a 40 000-block zig-zag program in 8 s, scan and rules included), plus the controller's
dry run of the assist (about 70 us per command).

## 6. Rules

Each rule reports once per tool section where it repeats (count and worst value in the
quantities), at the first line where it applies; sections are numbered by tool change.

| id | condition | severity | applies to |
|---|---|---|---|
| `feed-missing` | feed move (G01/G02/G03, cycle feed, G31) with no feed rate active | error | all |
| `feed-inherited` | first cutting move of a tool section without an F of the section: it runs at the F of the previous tool (cause: the line that set it) | warning | all |
| `spindle-stopped` | a cutting move (engagement estimate) with the spindle stopped | error | all |
| `spindle-direction` | spindle turning against the tool's hand (right needs M03, left M04); G84/G74 against the tap's hand or with the wrong rotation | error | mills, rotating tools |
| `tool-unknown` | T word (pre-selection or change) not in the tool table; M06 without a T | error | all |
| `tool-comment` | a tool comment (header list or section comment) whose T number, diameter (within `diameter_match_tolerance_m`; taps: thread major), kind or pitch disagree with the tool table, or with the project's tool `comment` | warning | all |
| `offset-length` | G43 with H0 or no H; H register without a length in the tool table and not measured by the program (G37, G10 L10/L11; a G65 call in the section makes it a warning); H of another tool (error when the lengths differ); lathes: cutting with offset 00 | error / warning | all |
| `offset-radius` | G41/G42 with D0 or no D; D without a radius in the tool table (G10 L12/L13 counts); D of another tool | error / warning | mills |
| `work-offset-unset` | the active work offset has no value in the setup and the program does not set it; no work offset selected before the first move (power-on G54 used) | error / warning | all |
| `missing-g43` | after a tool change, a Z move or feed move before G43 H is programmed in the section | error | Fanuc/Haas mills |
| `tap-depth` | a tap with no smaller drilled hole at its X/Y (within `hole_match_tolerance_m`); tap end below the drill tip (error) or within the drill point cone plus `tap_bottom_margin_pitches` (warning); through holes pass | error / warning | mills |
| `flute-length` | the tool's reach below the top of the engaged material over its flute (else cutting) length (the shank behind the flutes is as wide as the cut), or over its stickout (the holder reaches the part); a tap counts the material its thread engages (`ap`: to the stock bottom on a through hole), its neck passing the tapped hole | error | mills |
| `coolant-unavailable` | a coolant code needing a pump/nozzle the machine does not have (M07 mist or shower, M08 flood, M88 through-spindle, M73 through-tool air, M83 air blast, M34/M35 programmable nozzle); through-spindle coolant on a tool without coolant holes (warning) | error / warning | all |
| `cutting-data` | vc, fz (side milling; plunges only against the maximum), f per rev (drills), ap, ae (fraction of D or mm) outside the maker's range for the material's ISO group (`cutting_data`, exact material id preferred, then operation); warning, error beyond `cutting_data_error_factor`; tapping feed per rev not the pitch (error, `tapping_feed_tolerance_frac`); no data for the group: info | warning / error / info | all (lathes: vc, f) |
| `coolant-required` | cutting dry where the maker's data say wet, the tool type needs lubrication (`coolant_required_tool_types`), drilling deeper than `deep_hole_depth_ratio` x D, or the ISO group requires coolant (`coolant_need_by_iso_group`); water-based fluid on a water-free family (error); coolant on a dry tool class (warning) | warning / error | all |
| `retract-below-clamps` | a rapid (or the move between holes at the retract level) passing over a clamp below its top (error); a cycle's retract plane (R for G99, initial for G98) below the clamp tops without such a crossing (warning); clamp geometry unknown (info) | error / warning / info | mills |
| `part-outside-travel` | a programmed position that, with its work offset and the active H, needs machine coordinates outside the axis travel (moves from the controller assist: its machine targets) | error | mills |
| `arc-endpoint` | I/J/K arc whose end-point radius differs from the start radius by more than `arc_accept_increments` least input increments (warning: the control blends a spiral) or by more than the control's tolerance (error: it alarms; machine `controller.arc_radius_tolerance_mm`, else `arc_alarm_tolerance_m`); R arc shorter than half its chord (error) | warning / error | all |
| `program-end` | no M30/M02/M99 reached (error); M30/M02 with the spindle or coolant still on (warning) | error / warning | all |
| `restart-modal` | the controller's restart analysis issues (records `{line, program, issues: [{group, message, severity}]}`); otherwise per tool section (restart at the N-numbered line before the tool change, else the tool change): a modal group used before the section writes it, whose value differs from the power-on default (distance, work offset, plane, units, cutter compensation, active cycle: error; feed mode: warning), and G90/G91 not stated before the first move even when it equals the default (warning; not on lathes) | error / warning | all |
| `g28-absolute` | G28/G30 in G90: the intermediate point is in work coordinates; error when its Z is below the stock or clamp top (lathes: X inside the stock radius), else warning | error / warning | all |
| `toolchange-clearance` | M06 with the tool tip less than `toolchange_clearance_m` above the highest of stock and clamps | warning | mills |
| `rapid-below-stock` | a rapid (not a pure retract) ending below the stock top over the stock: warning when it stays at least `rapid_min_clearance_m` above the material left by earlier cuts, error when it goes into it or closer; a retract only when it penetrates | warning / error | mills |
| `scan-limits` | what the static checks could not evaluate (section 3) and moves whose engagement is unknown (tool geometry or stock unknown; moves in an unset work offset are covered by `work-offset-unset`) | info | all |

Sources of the rules: program structure, safety blocks, G43 after a tool change, G91 G28, canned
cycles and tapping in smid-2008; tap-drill depth in oberg-2016; word-address format in
iso-6983-1-2009; ISO groups in iso-513-2012; Haas codes (M88, M07 shower, M73, M83, M34/M35, M97,
Setting 15 H/T agreement) in haas-ngc-mill-om-2020.

## 7. Parameters (`db/models/sanity-rules.yaml`)

| name | value | source |
|---|---|---|
| `default_units` | mm | est (metric power-on setting) |
| `default_least_increment_m` | 1e-6 (0.001 mm) | est (IS-B; when the machine gives none) |
| `fanuc_inch_increment_m` | 2.54e-6 (0.0001 in) | est (Fanuc IS-B) |
| `arc_accept_increments` | 2 | est (rounding of CAM output) |
| `arc_alarm_tolerance_m` | 2e-5 (0.02 mm) | est (machine setting, Fanuc parameter 3410) |
| `rapid_min_clearance_m` | 5e-4 | est |
| `toolchange_clearance_m` | 0.025 | est |
| `hole_match_tolerance_m` | 1e-4 | est |
| `tap_bottom_margin_pitches` | 1 | est (smid-2008, oberg-2016 practice) |
| `deep_hole_depth_ratio` | 3 | est |
| `cutting_data_error_factor` | 1.5 | est |
| `slotting_ae_fraction` | 0.95 | est (slotting vs side-milling data) |
| `diameter_match_tolerance_m` | 5e-5 | est |
| `tapping_feed_tolerance_frac` | 0.01 | est |
| `zmap_cell_m`, `zmap_max_cells` | 5e-4, 4e6 | numerical settings |
| `vise_jaw_thickness_m` | 0.025 | est |
| `max_tool_tilt_rad` | 0.1 deg | est (a D20 tool's lower edge moves 0.02 mm, below the map's cell) |
| `coolant_need_by_iso_group` | P recommended, M required, K optional, N required, S required, H optional, O optional | est (tool makers' practice; iso-513-2012 groups) |
| `coolant_required_tool_types` | taps, reamer, gun drill | est |
| `water_free_families` | magnesium | est |
| `dry_tool_classes` | ceramic, cbn | est |
| `max_call_depth` | 10 | est |

The thresholds are judgement values of practice, not measured constants; each is written with
how it was chosen in the model file.

## 8. Fixtures

`tests/fixtures/sanity_setup.yaml` (100 x 80 x 30 mm Al 6061-T6 plate, two strap clamps 15 mm above
it, six tools, a Haas-dialect mill with flood coolant and an air blast but no through-spindle
coolant), `tests/fixtures/sanity_all_mistakes.nc` (every rule's mistake once; the 23 findings it
must give, with the mistake each one stands for, are in `sanity_all_mistakes.expected.yaml`), and
`tests/fixtures/sanity_clean.nc` (the same features done right: no finding at all).
examples/sanity-mistakes reuses the program with the same setup.

## 9. Validity and limits

Valid for 3-axis milling with the tool axis along +z of the program frame (VMCs; HMCs at a fixed
B), 2-axis turning, Fanuc and Haas dialects; five-axis and TCP programs and tilted planes get their
tool-tip positions and machine targets from the controller assist (section 3a) but no engagement
estimate where the tool axis is tilted, rotated work planes (G68) and multi-path lathes are checked
only by the rules that do not need geometry, and say so (`scan-limits`). The checks are as good as
the setup data: a wrong fixture pose moves the stock box and the clamps (section 4); the stock box of
an operation carried over by `from_previous` still contains what the earlier operation removed. The
assist is only as right as the controller's dry run: its ideal probe triggers on the stock box as set
up. Cutting data are compared at the programmed speed (overrides ignored).
Everything is deterministic: same inputs, same findings in the same order (sorted by program, line,
rule order, message).

## References

smid-2008, oberg-2016, iso-6983-1-2009, iso-513-2012, haas-ngc-mill-om-2020, haas-ngc-lathe-om-2020
(db/bibliography/models-sanity.yaml; the Haas manuals in controllers.yaml, iso-6983-1-2009 in
models-controller.yaml).
