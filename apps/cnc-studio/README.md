# CNC Studio

A browser-based CNC simulator (3-axis mill with an optional A/B rotary table, and 2-axis lathe, Fanuc-style) with
Claude built in. Describe the toolpath you want — or attach a drawing or a 3D model of the part. Claude writes the
G-code straight into the editor, the simulator runs it and measures the result, Claude checks the numbers against
what you asked for, and it fixes its own mistakes. Parts that need several setups (flip in the vise, turn around in
the chuck, turn then mill) are a job of operations, each starting from what the previous one left.

## Run it

```bash
npm start
```

Then open <http://localhost:5178>. On Windows you can also double-click `start.bat`.

Claude needs an Anthropic API key. You can either:

- set `ANTHROPIC_API_KEY` before starting (`$env:ANTHROPIC_API_KEY="sk-ant-…"; npm start` in PowerShell), or
- paste the key in **Settings** (the gear icon). It is stored in your browser and sent only to this local server.

The server listens on `127.0.0.1` only. To host it on your own server under a domain (HTTPS, password, service),
see [DEPLOY.md](DEPLOY.md).

## What you can do

- **Ask Claude**, for example "Pocket 40×30, 6 deep, R5 corners, centred", "M16×1.5 thread 18 long with a relief groove",
  "machine the back side in OP20", or "explain this program". You watch the G-code being written live. Claude then
  reads the simulation report: errors, rapid moves into material, shank, holder, chuck and vise collisions, cycle time,
  the cut envelope per tool — and the **measured geometry** of the result (see below), which is how it checks its work.
  Every version Claude writes can be restored from the chat or the *Versions* menu.
- **Attach drawings and models.** 📎 *Attach*, drag-and-drop or paste onto the chat:
  - drawings (PNG, JPG, PDF) go to Claude as images / documents;
  - a 3D model of the finished part (STEP, IGES, STL, OBJ, 3MF — STEP/IGES are read by OpenCascade compiled to
    WebAssembly, in a worker) becomes the job's **target**. It is drawn as a green ghost, placed in the first
    operation's coordinates (Stock tab: axis, rotation, zero, units, shift), can size the stock (*Stock from model*),
    and every operation is compared with it: material still to remove and gouges, with location and volume.
- **Operations / setups.** The tabs above the editor are the job's operations (OP10, OP20, …). `+` adds the next one
  (continuing from the last result, flipped), opens a file as an operation, duplicates one, or saves the whole job
  (`.cncjob.json`, including the target model). Setup directives in the program say how the part is held:
  `(STOCK FROM OP10)`, `(ORIENT FLIP X)`, `(ORIENT ROT Y90)`, `(ZERO TOP CENTER)`, `(VISE GRIP 8)`, `(CHUCK GRIP 20)`.
  A lathe operation can continue a turned part turned around in the chuck, and a mill operation can take a turned part.
- **4th axis.** `(ROTARY A)` and `A`/`B` words (indexing or moving together with X/Y/Z); the part and its toolpath
  turn on screen and the A/B position shows in the DRO. ISO sense: `A90` brings the part's −Y face up;
  `(ROTARY A REVERSE)` for a table wired the other way.
- **Data instead of screenshots.** After every run the part is measured, in that operation's work coordinates:
  size and outline, pockets, holes (diameter along the depth — tapped tops, counterbores, drill points), slots, steps
  and islands with sizes, corner radii, depths and centres, seen from every side; lathe parts as a profile of
  cylinders, faces, chamfers, radii and threads. Claude also has an `inspect_part` tool for exact sections, depth maps
  seen from any side, point probes, and the comparison with the target. Pictures (`view_simulation`) are still
  available but rarely needed.
- **Edit by hand.** The editor has G-code highlighting and re-simulates as you type. Gutter markers show
  problems, and double-clicking a line jumps the simulation to it.
- **Simulate.** Material removal is shown in 3D: a heightmap for single-setup mill programs, a narrow-band 3D distance
  field (surface-nets mesh) for multi-operation and rotary work, and a revolved profile with an optional ¾ section on
  the lathe. Each tool tints the surfaces it cuts. You get playback, block stepping, a timeline, a DRO and
  iso / top / front / side views.
- **Check.** Cycle time, tool changes, removed volume, the comparison with the target model and an issue list.
  "Ask Claude to fix these" hands the list to Claude. Threads are judged as threads: the simulator cuts them as rings
  (lathe) or a plain bore (tap), and models usually draw them as a plain diameter or the tap drill, so each thread zone
  is left out of the surface comparison and gets one line instead — consistent, or MISMATCH (wrong pitch, size or place).
- **Tool engagement.** Every feed move is measured on the material as it is just before the cut: on the mill the radial
  width of cut (ae, mm and % of Ø), the arc of the cutter in contact and the axial depth (ap); on the lathe the depth of
  cut and feed per rev; plus the removal rate and the feed spent in air. It shows in the DRO while playing, as a strip
  above the timeline (tap it to go there) and per tool in the Check tab. Claude gets the per-tool summary in every report
  and the line-by-line table with `inspect_part` (query "engagement").
- **Projects.** *Projects* (top left) keeps jobs on the server under a name: the operations, subprograms and the target
  model with its placement. Save the job as a new project or back into the open one, and open, rename or delete any of
  them from any device; the button shows the open project's name, with a dot while it has unsaved changes. (Needs
  `DATA_DIR`, see DEPLOY.md; *+ → Save job* still writes a `.cncjob.json` file for taking a job elsewhere.)
- **Hide Claude.** Settings → *Hide Claude* takes everything Claude off the screen (the chat tab, the status pill, the
  API key field, *Ask Claude to fix these*) — CNC Studio is then a plain simulator. Separately, ☰ (top right) or × in
  the panel's tab row hides the whole side panel (Check, Tools, Stock and the chat) so the 3D view gets the room; ☰
  brings it back. Both choices are remembered.
- **Export for the machine.** *Export* builds one NC program with standard content only: CNC Studio's directives
  (STOCK, TOOL, SETUP, ORIENT, ZERO, VISE, CHUCK, ROTARY, `$Set…`) are removed, comments become plain upper-case ASCII,
  M98 subprograms are built in (appended as O programs when they jump), and several operations of one machine are joined
  with M00 and a re-clamp note. The export is simulated again and compared move by move with the original.
- **Tools.** On first start, CNC Studio finds your CncSimulator5 installation and imports its tool registers
  (`MillTools.millreg`, `TurnTools.ini`). It looks in the same places, in the same order, as CncSimulator5:
  `%APPDATA%\CamModul6\Tools`, then `%APPDATA%\CncSimulator5UserFiles\Tools`, then the install folder.
  You can re-import or pick a file in the Tools tab at any time.
  - **Lathe tools keep their real geometry.** That's the insert outline, the holder outline, the groove-blade
    reference corner (`ToolOrigin`), and the drill length and point angle. They are used for cutting, for drawing,
    and for **holder collision** checks. Built-in and hand-typed lathe tools get generated outlines.
  - **Mill tools** have a flute length, a stick-out and a holder diameter, so you get shank-rub warnings and
    holder collision errors.
  - **Program comments override the library**, e.g. `(TOOL T1 FLAT D10 L30 SL45)` or CAM headers like
    `(T6 D=10. CR=0. - FLAT END MILL)`.
- **Stock.** The stock comes from `(STOCK …)` or `($SetPartSize(…))`, is auto-fitted, or is set manually.
- **Files.** Open/Save `.nc`. When you open several files at once, the extra ones become subprograms for `M98`. Older
  Latin-1 files load correctly.

## Supported G-code

Fanuc mill and lathe (Group A), including:

- G00–G03 with helical arcs, `,R`/`,C` corner rounding, and G41/G42 cutter and nose radius compensation
- G28/G53/G54–G59/G52/G92/G10, G15/G16 polar
- Mill canned cycles G73–G89 with R/Q/P/K and G98/G99
- Lathe cycles G70–G76 and G90/G92/G94, with G96/G97/G50 constant surface speed
- Subprograms M98/M97/M99 and G65 macro calls
- **Fanuc Macro B**: `#vars`, expressions, `IF`/`WHILE`/`GOTO`

CncSimulator's own directives `($SetPartSize(x, y, z, n))` and `($SetRadiusPrg(1))` are read too.

Rotary A (about X) and B (about Y) on the mill, through the work zero.

Not simulated: the lathe C axis / live tooling, G68 rotation, G51 scaling, and other dialects (Heidenhain conversational, Siemens).

## Reverse-engineering notes (CncSimulator5)

CNC Studio was designed from an analysis of the installed *CncSimulator5* by MicroTech StellaData:

- **Architecture.** `CncSimulator5.exe` is a native MFC renderer. Dialects are translated by managed .NET readers in
  `Tolk\NET\MachineSpecific.dll`: `FanucMillFormatReader`, `FanucTurnFormatReader`, `HeidenhainMillFormatReader` and
  `OkumaTurnFormatReader`. These feed interpreters and `MtsOldSimFileFormater`, which emits the simulator's legacy
  internal format.
- **Grammar.** The readers register trigger patterns such as `G2,G02,!R,!C[X#][Y#][Z#]`, `M98{P#}[L#]` and
  `G81{R#}{Z#}{F#}{K#}`. These revealed the supported words: U/V/W incremental moves, `,R`/`,C`, polar moves, K
  repeats, G70–G76 two-block lathe cycles, and the `$SetPartSize` / `$ToolDiameter` directives.
- **Cycles and checks.** `MillCycleDrill` implements G73/G74/G76/G81–G85 plus Heidenhain CYCL 200–203.
  `FanucCycle` implements the G71/G72/G73 roughing logic (`MakeRoughTurning`, `OffsetContour`, `ReverseContour`).
- **Tool registers**, decompiled from the register loader `FUN_00432a20` in `CncSimulator5.exe`:
  - **`TurnTools.ini`:** `CutterN`/`HolderN` points are `dz dx` (Z offset, then radial offset) from the tool's
    reference point, and `NoseOffsetX/Y` is the nose centre (Z, X). `ToolType=Normal` tools carry their own polygons.
  - **Groove tools** (`Groov*`) are generated as a `Width × Height` blade starting at the `ToolOrigin` corner
    (1 bottom-left … 8 left-middle). `GroovVertDown` holders hang from the blade's top centre, and `GroovHoriLeft`
    holders from the blade's far end.
  - **Drills** (`Drill*`): `Width` = length, `Height` = diameter, and the point angle is `TipAngleDrill`. The apex is
    the reference point, and `ToolOrigin` is ignored.
  - **`MillTools.millreg`** rows are `"%d %lf %lf %lf %lf %d %lf %s"` = slot, diameter, cutting length, feed,
    spindle, tool number, overall length, name. The column names come from string table entries T240–T243.
  - **Duplicate tool numbers:** the first one wins, as in CncSimulator (string T259).
- **What CNC Studio adds.** The original's checks ("spindle start / feed / S before G01", "M30 present") are kept.
  CNC Studio adds material-removal collision detection (rapids into material, shank, holder, chuck), Macro B, a full
  G41/G42 implementation and the Claude integration.

## Development

```bash
npm test                      # interpreter over sample programs
node test/geometry.mjs        # exact-answer checks: G41/G42 offsets, ,R fillets, G71 levels, G83, macros, helix
node test/sim.mjs             # material removal accuracy + real programs
node test/examples.mjs        # built-in examples (programs and jobs) must simulate clean
node test/job.mjs             # multi-setup jobs: flips, stock hand-off, lathe→mill, rotary, vise collisions
node test/voxel.mjs           # 3D material engine: cutting, volumes, meshing, lathe/heightmap conversion
node test/inspect.mjs         # measured geometry Claude gets (features, holes, profiles, sections, probes)
node test/model.mjs           # STL/OBJ/3MF/STEP readers, target placement, stock-vs-target comparison
node test/threads.mjs         # threads vs target models: cosmetic, modeled and wrong threads, tapped holes
node test/engagement.mjs      # tool engagement of known cuts (slot, stepover, plunge, air, lathe depth of cut)
node test/export.mjs          # export: standard content only, subprograms built in, same toolpath
node test/store.mjs           # the work kept on the server (DATA_DIR)
node test/projects.mjs        # projects kept on the server (DATA_DIR/projects)
node test/real-path.mjs       # server ↔ Anthropic SDK request shape, against a local fake API
node test/registers.mjs       # CncSimulator5 register import (needs CncSimulator5 installed)
node test/register-sim.mjs    # the original's samples simulated with your own register
# http://localhost:5178/dev/tools.html  draws every imported lathe tool outline
node test/mock-server.mjs     # run the app with a scripted Claude (no API key) on :5179
```

| Path | What it holds |
|---|---|
| `server.js` | Static server plus the `/api/chat` streaming proxy (official `@anthropic-ai/sdk`) and tool definitions |
| `prompts/system.md` | Claude's system prompt: the dialect contract between Claude and the simulator |
| `public/js/gcode/` | Parser, interpreter, contour/compensation, lathe cycles, tools, stock (pure, testable in Node) |
| `public/js/gcode/setup.js` | Setup directives and part → work transforms |
| `public/js/sim/` | Material removal (mill heightmap, 3D voxel field, lathe profile), playback runner, jobs (`job.js`), measured geometry (`inspect.js`) |
| `public/js/model/` | 3D model readers (STL, OBJ, 3MF, STEP/IGES via `occt-import-js`), target placement and comparison |
| `public/js/view/` | three.js scene, surface-nets mesher |
| `public/js/claude.js` | Browser-side agent loop (streaming, tool execution, chat rendering) |
| `public/js/app.js` | UI glue |

The Claude integration uses Claude Opus 5.5 by default, with adaptive thinking (summaries shown in the chat), a
selectable effort level, the server-side refusal fallback (`fallbacks: "default"`), prompt caching of the system
prompt and conversation, and tool input that streams eagerly so the program appears in the editor as it is written.
