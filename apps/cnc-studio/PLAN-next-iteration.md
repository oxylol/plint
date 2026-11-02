# CNC Studio — next iteration (plan to check, 2026-11-02)

Draft for Filip. Seven parts; each can be handed over as a task of its own (A–D and G are the substance,
E–F the UI and small things). Phones are out of scope. Once a part is approved it becomes a `TASK-<name>.md` in the
style of `TASK-threads.md`.

## Decisions (Filip, 2026-11-02)

Approved as below with four additions: brief items point at the program lines they concern (B); the probe
part must detect when the work zero would be hard to set (D, the datum check); no light theme (E8 dropped);
and **everything in this iteration sits behind one switch in Settings** ("Iteration 2", default off),
so the app can fall back to today's behaviour at any time. Implemented on Plint directly, part by part,
each with tests and a commit.

### The switch

`settings.iteration2` (stored with the other settings on the server, so it applies on every device). Off
means today's behaviour exactly: the chat request carries `features`, and the server then sends the old
tool list and the old prompt; the Brief tab, the Measure mode, the deviation view, the cost pill and the
version history are hidden; the craft checks do not run; G31 is a plain feed move as today; the slim
context is not used. The node tests run the simulator parts in both modes. A probe tool in the library is
harmless either way (it only matters when a program uses it).

### Decisions as asked

1. **Edit tool shape (A).** A list of exact `old` → `new` replacements, the shape Claude is trained on. The
   alternative is line-range edits; I advise against them, line numbers shift after every edit.
2. **The brief (B).** Approved: hidden from the program, the export and the saved `.nc`, shown in a
   *Brief* tab. Claude maintains it through a `brief` field on `write_program` / `edit_program`. **Added:
   every item points at the program lines it concerns** (see B.8): clicking an item highlights its lines,
   the lines show which item they belong to, and the links survive edits.
3. **Probe scope (D).** Approved, all three layers, with the point Filip cares about made explicit:
   **the simulator must detect when it would be difficult to set the work zero** for a setup (D.10, the
   datum check): which surfaces the probe would need for X, Y and Z, whether the standard probe can reach
   them past the part, the vise and the overhangs, and what to use instead (a bore or boss from an earlier
   operation, a reference left on purpose).
4. **"Being able to measure".** Confirmed by Filip: this is for the user, in the 3D view, and has nothing
   to do with the probe: click to measure a side, the distance between points, a face or a hole, and the
   deviation from the target part. Part G.
5. **UI list (E).** Keep all but the light theme (E8 dropped).

---

## A. `edit_program`: change a program without rewriting it

### The problem

`write_program` always takes the complete program. That is deliberate (nothing to apply, nothing that can
fail to apply, trivial streaming and versions), but output tokens are the slow and expensive ones:

| Program | Output per write | Cost per write (Opus 5.5) | Streaming time |
|---|---|---|---|
| 70 lines | ~700 tokens | ~1.5 cent | 10–15 s |
| 400 lines | ~4 500 tokens | ~9 cent | 1–1.5 min |

per attempt, and the prompt allows three. Rewrites also drift (unasked changes elsewhere) and take the
editor away from the user while streaming. Anthropic's migration notes for Fable 5.1 say it rewrites whole
files where a targeted edit would do, and give a one-line prompt that restores targeted edits.

### What to build

1. Tool `edit_program` (server.js `TOOLS`, run in `app.js` like `write_program`): inputs `operation`
   (optional, same rules as write_program), `title`, `edits: [{ old, new }]`, `brief` (optional, part B).
   Apply in order to the operation's text (CRLF normalized to `\n` first, as write_program does). Each
   `old` must occur exactly once; if any does not, apply nothing and return an error (`is_error: true`)
   that says how many matches there were and quotes the closest lines, so Claude corrects the next call.
   `new` may be empty (delete). After applying: exactly what write_program does, `syncEditorToOp`,
   `saveOps`, a version snapshot (title from the call), simulate, return the full report, prefixed by one
   line: `Applied 3 edits: +6 −2 lines (lines 24–31, 58).`
2. No streaming into the editor for edits; apply at `block_stop`. The `beginWrite`/`prepareWrite`/
   `streamWrite` path stays for write_program only.
3. Editor: flash the changed lines (a CodeMirror line class that fades over ~2 s) and scroll the first
   change into view; keep the user's cursor and scroll otherwise.
4. Chat card: `Edited program — <title> · +6 −2 · ✓ clean · 0:45 · restore · report · diff`, where *diff*
   expands a unified diff (3 lines of context) built from the two snapshots.
5. Versions: a snapshot after every edit, as today. The 80-entry dropdown becomes a history list with
   time, source (Claude/user) and "diff vs previous" (E9).
6. `prompts/system.md`: edit when changing an existing program; write for a new program, a new operation,
   or when more than about a third of the lines change. Add Anthropic's line: "The number of tokens used
   to edit files is best minimized, all else being equal. Therefore, when it will not affect the end result,
   try to surgically edit a file rather than rewrite the entire thing."
7. Input validation in `claude.js runTool` (non-empty array of `{old, new}` strings, `old` non-empty),
   like the checks that exist for the other tools.
8. Tests: `test/edit.mjs` (apply, miss, duplicate match, delete, CRLF, the summary line, the snapshot),
   `test/real-path.mjs` (request shape with the new tool), `test/mock-server.mjs` (a scripted edit round).

### Done when

"Change the feed on T3 to F600" produces one `edit_program` call of a few lines; the editor shows the
change highlighted; the report arrives; restore brings back the pre-edit version.

---

## B. The brief: what the user fixed, what Claude chose

### The problem

The only memory of *why* a program looks the way it does is the chat, which New chat resets and which grows
long. Claude reads its own earlier choices back out of the program as if they were requirements (a tool it
picked, a stepover, an origin, a chamfer it skipped), then defends them ("you asked for…") or refuses to
improve them. The user cannot see which parts are negotiable either.

### What to build

1. A brief per operation: `op.brief = { items: [{ id, text, by, fixed, note }], updatedAt }` with
   `by` ∈ `user | drawing | model | claude` (who decided) and `fixed` (do not change without asking;
   default true for user/drawing, false for claude; the user can flip either in the UI). Examples:
   - `[user, fixed] Material: S355 steel`
   - `[user, fixed] Tool T6 chamfer mill for the edge break`
   - `[user, fixed] Part zero: centre of the back face`
   - `[drawing, fixed] Ø6 H7 through, 4×, on PCD 60`
   - `[model] Stock from the model + 1 mm per side`
   - `[claude] Face with T8 Ø50 in one pass, offset 12 mm from the centreline`
   - `[claude] Roughing Ø10: ae 4 mm, ap 8 mm, 0.3 mm finishing allowance; helical entry 3°`
2. Claude maintains it through the optional `brief` input on `write_program` and `edit_program`: the full
   list for that operation (short, so whole-list replace is fine; ids are kept when the text matches). The
   report confirms the stored brief. Prompt rules: record every requirement from the user or the drawing
   as `user`/`drawing`; record each own choice as `claude`; **never treat a `claude` item as a
   requirement**: change it freely when the result gets better and say so; ask before changing a fixed
   item; when a new message contradicts an item, update the item. A program opened from a file has no
   brief: treat its content as the user's until they ask for improvements.
3. Context: `<simulator_state>` carries each operation's brief before its program:
   `<brief op="OP20">` with one `- [user, fixed] …` / `- [claude] …` line per item.
4. UI: a **Brief** tab next to Check. Two groups, *Required* (user/drawing/model) and *Claude's choices*;
   each row has the text, a source badge, a lock toggle (fixed), ✕, and an *Add* field for a requirement
   typed by the user. Changes persist and reach Claude on the next turn.
5. Persistence with the operation: `saveOps` (store key `ops`), projects (`project.json`),
   `.cncjob.json` (format version 2; old files load with empty briefs), *Duplicate operation* copies it.
   Never in the NC text, never in the export, not in *Save .nc*.
6. New chat keeps the brief; New program clears it.
7. Tests: `test/brief.mjs` (round trips through store, project and job file; id merge rules; the context
   text; line anchors before and after edits).
8. **Lines.** Each item carries `lines`: the ranges of the program it concerns (`[[24, 31], [58, 58]]`),
   given by Claude in the `brief` input as line numbers of the program it just wrote or edited, and kept
   as anchors (the text of the first and last line of each range) so they are re-found after an edit by
   the user or by `edit_program`; an anchor that cannot be found makes the item show "lines moved" instead
   of a wrong range. In the Brief tab, hovering or clicking an item highlights its lines in the editor and
   scrolls to the first; in the editor, a thin coloured bar in the gutter marks lines that belong to a
   fixed item (user/drawing), and hovering it names the item. In the context the ranges are written after
   the item (`- [user, fixed] Part zero: centre of the back face (lines 6, 21)`), so Claude sees which
   code implements which requirement.

### Done when

After New chat, "make the face milling better" changes Claude's own face-milling choice without arguing,
and "use T6 for the chamfer" stays put through later changes until the user unlocks it.

---

## C. Machining craft: tool choice, face milling, rubbing

### The problem

The prompt covers cutting data, entry strategies and depth per pass, not the things a machinist checks by
habit: which tool to take when the user did not say, where the cutter sits relative to the part, and
whether the edge cuts or rubs. The simulator measures engagement but does not judge it. Cases: a Ø50 face
mill run down the centreline of a 40 mm part (the inserts cross the centreline, the radial force flips,
vibration); fz 0.02 mm with a Ø10 end mill at 10 % ae (chip thickness 0.006 mm, rubbing); ap 0.15 mm
with an R0.8 insert (the nose pushes instead of cutting); a face mill plunged; a ball mill finishing a
flat floor with its tip (zero surface speed at the centre).

### What to build

1. **Prompt rules** (`prompts/system.md`, a new *Craft* section, in the voice of the rest):
   - Tools: unless the user named one, choose freely from the library and record the choice in the brief
     as yours. Roughing: the largest tool that fits the smallest inside radius; finishing: the corner
     radius the drawing needs; the shortest tool whose flute length suffices. A tool the user named is
     fixed.
   - Face milling: a cutter 20–50 % wider than the surface takes it in one pass, run off-centre (axis about
     D/4 beside the part centreline, the inserts leaving the part on the climb side), never centred.
     Wider surfaces: passes of about 70 % of D, each with the same rule. Roll in and out; never stop in
     the cut.
   - Rubbing: keep the mean chip thickness above about 0.02 mm (carbide); raise fz when ae is small
     (chip thinning, hm = fz·√(ae/D) for ae < D/2); no finishing pass with ap below the corner radius on
     the mill or below about 2/3 of the nose radius on the lathe; lathe finishing feed not below about
     0.05 mm/rev; never plunge a face mill or a non-centre-cutting tool; do not finish floors with the tip
     of a ball mill; no dwelling in the cut.
   - Stability: lower ae with higher ap rather than full-width slotting with long tools.
   - Holes: spot → drill (tap drill from the table) → tap or ream; chamfer last.
2. **Simulator checks** in a new `public/js/sim/craft.js`, called from `job.js runOp` next to
   `extraIssues`, working on the engagement samples (`engagement.js`: ae, contact arc lower/upper, ap, f,
   plunge per sample) and the tool table. Each check yields an issue with a line number and a hint, so it
   shows in Check, in the gutter and in Claude's report:
   - `warning` **centred face mill**: FACE tool, ae 50–100 % of D, contact arc symmetric about the motion
     (|lower + upper| < 20°) over more than 2×D of path:
     `T8 Ø50 face mill runs centred over the 40 mm width (contact −53°..+53°) — offset the path about 12 mm so the inserts leave the part on one side`
   - `warning` **rubbing**: mean chip thickness below 0.02 mm over more than 10 mm of path. Needs the flute
     count: new optional tool field `flutes` (`Z4` in the TOOL comment, register import where known,
     defaults by type: flat/bull/ball 3, face 5, chamfer/spot 4, drill 2); fz = F/(S·z):
     `T2 Ø6: chip thickness 0.006 mm (fz 0.02 at 10 % ae) — rubbing; feed about F1200 or more ae`
   - `warning` **ap below the corner radius** (bull/ball finishing) and, on the lathe outside threading
     and grooving, **ap below 2/3 × nose radius** or **f below 0.05 mm/rev**:
     `T2 R0.4: ap 0.15 mm on the finishing pass — below the nose radius the insert pushes; take 0.3 mm or a smaller radius`
   - `error` **face mill plunging** (plunge sample deeper than 0.2 mm); `warning` **ball tip on a flat
     floor** (ball tool, contact only at the tip over a horizontal floor).
   - Thresholds in one table at the top of `craft.js`; nothing new in Settings.
3. **Report**: a `Craft:` line per flagged tool in the engagement block, so Claude does not have to dig
   the issue out of the list.
4. Tests: `test/craft.mjs` (centred vs offset face pass, low-fz contour, lathe finishing at ap 0.15, face
   mill plunge, ball mill floor); `test/examples.mjs` must stay clean (fix examples that trip the checks).

### Done when

The built-in examples pass; a centred Ø50 face pass gets the warning with the offset hint; asked to "face
the top", Claude writes an offset pass without being told.

---

## D. A standard probe: can it reach, what does it measure

### The problem

Real multi-setup parts are set up with a probe: find the top face, find a bore, set G54 for the next
setup, check a machined surface before finishing. Claude cannot write that today: `G31` is simulated as a
plain feed move (`interpreter.js` maps it to G01), nothing sets a skip position, no probe tool exists. And
nothing checks whether a probe can physically reach a surface: on a flipped or rotary part an overhang,
a wall or the vise blocks the approach, or hits the probe body long before the stylus touches.

### What to build

1. **Tool type `probe`** (mill, `tools.js`): `diameter` = stylus ball Ø, `stylusLength` (ball centre to
   body), `stylusDiameter` (shaft), `holderDiameter` = body Ø, `overallLength` = ball tip to holder face.
   Built-in library entry `T20 probe Ø6 ball, stylus 50, shaft Ø4.5, body Ø40 × 50 "Touch probe"`
   (OMP40-style; the user's register overrides). Comment form:
   `(TOOL T20 PROBE D6 SL50 BD40 "TOUCH PROBE")`. Drawn as ball + thin shaft + cylinder, no colour tint.
2. **Never cuts**: material contact of a probe tool outside a `G31` move is an `error`
   (`T20 probe touches the part on a G00/G01 move at …`), rapids included.
3. **`G31 X.. Y.. Z.. F..`** (skip): feed along the line, stop where the stylus ball first touches
   material (ball surface, not centre). The material is the simulated part as it stands at that line:
   everything cut so far in this operation and in the earlier ones, so the probe measures the machined
   result, never the raw stock. Set `#5061–#5063` (ball centre in work coordinates at the trigger,
   as a real control does; document this) and `#5021–#5023`. The move ends there; the next blocks continue
   from it. No contact by the end: `warning` `G31 line 42: no skip signal — the probe reached X.. Y.. Z.. without touching`.
   Shaft or body contact during the approach, or the ball touching while shaft/body also touch (overhang):
   `error` `Probe body hits the part at … before the stylus touches the floor — the Ø40 body does not fit under the 8 mm overhang`,
   with the first contact point. All three engines; overhangs exist only in the 3D (voxel) engine
   (multi-operation, rotary), the height-map engine checks walls, pockets and the vise jaws.
4. **Cycles built in** (recommended): Renishaw Inspection Plus-style macros as built-in subprograms, used
   when the job does not supply its own: `G65 P9810 X.. Y.. Z.. F..` protected positioning (a G31 that
   must NOT touch: touching is an error), `P9811` single surface (X, Y or Z target → `#135–#137`
   measured, `#139–#141` error; with `S<n>` updates work offset n), `P9812` web/pocket (size → `#138`,
   error `#142`), `P9814` bore/boss (`D` nominal, four touches, centre and size), `P9832`/`P9833` probe
   on/off (no-ops). Argument letters and output variables as in the Inspection Plus manual (verify there).
   Written in Macro B on top of G31, so the simulator runs them like any subprogram.
5. **Report**: a `Probing:` block per operation: every touch with the measured point and the surface it hit
   (top face, wall, pocket floor, and which tool cut it), and for cycles measured vs target:
   `P9811 Z: target Z0, measured Z-0.012 (face from OP10) → G54 Z −0.012`,
   `P9814 bore D20: measured Ø19.96 at X0.01 Y-0.02 (milled by T2 in this operation), target Ø20 → −0.04`.
   The values are the simulated geometry, so Claude can use them in Macro B (`#1 = 20 - #138`) to size a
   finishing pass from a measurement, and the report shows whether that worked. `inspect_part`'s own
   `probe` query stays what it is (Claude's virtual measurement); the machine probe is separate.
6. **Optional `(FIXTURE ERROR X0.1 Y-0.05 Z0.2)`** directive: shifts the part in the fixture for this
   operation so a probing routine has an error to find; the comparison with the target then shows whether
   the program compensates. Without it the stock sits exactly where programmed and probing returns nominal
   values, which still tests reachability and the real surfaces of a flipped part.
7. **Export**: probe cycles export as written (standard macro calls on a Renishaw-equipped machine). The
   export check warns when `P98xx` is called without a probe tool in the job.
8. **Prompt**: when to probe (after each re-clamp: Z and the XY datum before cutting; a critical surface
   before finishing when asked), how to approach (protected moves, from the open side, 2–5 mm overtravel,
   F300 probing feed), and that the simulator checks reachability.
9. Tests: `test/probe.mjs` (touch on a top face, a wall, a pocket floor; no skip; body collision under an
   overhang in the voxel engine; vise jaw in the way; cycle outputs; report lines; export unchanged; the
   datum check on the flipped example and on a part whose sides sit inside the jaws).
10. **Datum check** (automatic, per operation, mill): from the setup directives (ORIENT, ZERO, VISE,
    ROTARY) and the geometry at the start of the operation, work out how the work zero could be set with the
    standard probe and whether it can:
    - Z: the top face (or a reachable floor) under the ball with the body clear.
    - X and Y: opposite outside faces (web cycle) that the ball can touch above the jaw top and the
      parallels with the body clear, or a bore / boss / pocket from an earlier operation whose position
      relative to the new zero is known, or a single face for a corner zero.
    - For each axis: the reference it would use, whether it is a machined surface or raw stock (raw stock
      gives a poor reference and is said so), the free height or width available to the ball, and the
      verdict. Difficult means: no reachable reference, a reference reachable with less than 3 mm for the
      ball, or only raw stock after the first operation.
    - Report block `Setting the work zero for OP40:` with one line per axis, for example
      `Y: both Y sides are inside the vise jaws (part protrudes 1.7 mm above the jaw top) — not probeable; use the Ø6 bore at X0 Y-8.7 from OP10 (reachable, machined) or leave a reference`,
      and a `warning` issue when an axis has no good reference, so it reaches the Check tab and Claude.
    - The prompt tells Claude to plan each setup so the zero can be set, to read this block, and to change
      the ZERO, add a reference feature or re-order operations when it says difficult.

### Done when

On the current OP40 job, Claude can add a probing block that finds the back face and the M6 bore after the
flip, a P9814 on a bore milled earlier in the same operation reports the size the simulator actually cut,
the report shows the measured values, and approaching under the tongue gives the body-collision error
instead of a silent pass.

---

## E. UI (desktop)

1. Check tab, target comparison: each region clickable → the camera flies there and the region is outlined;
   a *Deviation* view toggle paints the part red (gouge) / blue (material left) with a legend. The samples
   exist already (`compareTarget`, 0.29 mm).
2. Issues with a position (rapid into material, holder, vise, probe body) get a marker in the 3D view.
3. Gutter markers: click shows the message inline under the line (line widget) instead of hover only.
4. Tools tab: one row per tool, name on its own line, numbers below, full type names in the dropdown.
5. Engagement strip merged with the timeline: the strip is the scrubber; the caption moves to the right
   end, off the bars.
6. Editor header: setup info on its own line under the file name instead of a truncated badge.
7. Composer: the model picker moves to Settings; effort stays.
9. Versions: a history list with time, source and "diff vs previous" (replaces the dropdown).

## F. Small things

1. Session cost in the status pill (`12.3k in · 4.1k out ≈ $0.21`; prices per model in `server.js`).
2. Slimmer context: only the active operation's program in full; the others as one line plus their brief;
   a `read_program(operation)` tool returns another one in full on demand. Multi-operation jobs stop
   re-sending every program on every turn.
3. Simulator gap: lathe `G92` thread cycle continuation lines with only X do not cut; only the first pass
   is simulated.

## G. Measuring the machined part in the 3D view

### The problem

The only way to get a number off the simulated part today is to read it out of the measured-geometry text
or to ask Claude. A machinist wants to click: how long is this side, how far apart are these two points,
where is this face, how deep is this pocket, how big is this hole, and how far is this spot from the model.

### What to build

1. A **Measure** mode in the view tools (a ruler button, `M` key, Esc leaves it). In this mode clicks pick
   on the simulated part, the stock and the target ghost (three.js raycasting on the height-map mesh, the
   surface-nets mesh and the lathe revolve). Picks snap within a few pixels to feature geometry the
   simulator already knows (`inspect.js`: corners and edges of outlines, hole and boss centres, pocket
   floors, shoulders on the lathe), so a click near a corner gives the corner, not a point on the fillet.
2. What a pick gives, in the operation's work coordinates, shown in a small panel in the viewport and drawn
   in the view until cleared:
   - one point: X Y Z (lathe: Ø and Z) and, with a model loaded, the signed deviation from the target
     there (`+0.12 material left` / `−0.05 gouge` / `on size`);
   - two points: the distance and ΔX ΔY ΔZ;
   - an edge (click near a straight edge): its length, its direction and the angle to the axes;
   - a face: its level (Z of a floor, X or Y of a wall, Ø of a cylinder), its extent, and the deviation
     from the model over the face (min / max);
   - a hole or boss: diameter, centre, depth;
   - two edges: the angle between them;
   - lathe: lengths along Z between shoulders, diameters, chamfer sizes.
3. The deviation values come from the existing comparison (`compareTarget` sampling); a point picked on the
   target ghost measures the model instead, so model and part can be compared point by point.
4. A list of the session's measurements (copy as text, clear all); double-clicking one re-centres the
   camera on it. Nothing is saved with the job.
5. Tests: `test/measure.mjs` for the snapping and the numbers on known parts (the pocket example: side
   lengths, corner radius, floor depth, a hole diameter; a lathe profile: a shoulder length and a
   diameter).

### Done when

On the pocket example, clicking two opposite pocket walls gives 40.00, clicking the floor gives Z-6.00, a
hole gives Ø6.80, and with the model loaded a point on the floor shows its deviation.

## Order and size

A small. B medium (serialization in four places + a tab). C medium (the checks need the flute count).
D large: tool type + G31 + collisions is one task, the cycles a second, the fixture error a third.
G medium (picking and snapping are the work; the numbers exist). E and F fit anywhere.
Suggested order: A, B, C, then G and D, then E/F.
