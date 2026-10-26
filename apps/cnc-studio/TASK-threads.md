# Task: make the target-model comparison thread-aware

You are working on CNC Studio, a browser CNC simulator (Fanuc mill + lathe) with Claude built in. The app is the
folder this file is in (on the server usually `/opt/cnc-studio`, run by the `cnc-studio` systemd service). Read
`README.md` for the overview. This task changes simulator code only; no new dependencies.

## The problem

Users load a 3D model of the finished part (STEP/STL/…) as the "target". Every operation's simulated result is
compared with it (`compareTarget` in `public/js/model/model.js`), and the comparison goes into the report Claude reads.
Threads break this:

1. **Most CAD exports have cosmetic threads.** The thread is not in the geometry: an external thread is a plain
   cylinder at the major diameter, a tapped hole is a plain hole (usually at tap-drill size, e.g. Ø6.8 for M8, sometimes
   at the minor or nominal diameter).
2. **Some models have real helical threads.**
3. **The simulator does not make helixes.** On the lathe, G76/G92/G32 cut axisymmetric V-grooves at the pitch (the
   simulation is a turned profile, so the thread is a set of rings at the program's own start position). On the mill, a
   tap (`G84`/`G74` with a `tap` tool) cuts a plain cylinder at the tap's nominal diameter.

So today a correctly threaded part is reported as damaged. Measured with `test/threads.mjs`:

| Case | Today's report |
|---|---|
| Lathe M20×1.5 (shaft example), model has a plain Ø19.8 cylinder | 12 × "GOUGE … 0.82 mm", one per thread groove |
| Lathe, model has a real helix starting at another angle than the program | alternating "material still to remove" / "GOUGE" every half pitch |
| Mill M8 tapped hole, model hole at tap drill Ø6.8 | "GOUGE … 137 mm³" around the hole |
| Mill M8 tapped hole, model has a real M8 helix | "GOUGE … 77 mm³" |
| Measured geometry of the tapped hole | "through hole Ø7.96" — nothing says it is tapped |

Two related problems found while testing, also to fix:

- **False excess along outside edges.** A part that matches the model exactly still gets "Material still to remove"
  along every convex outside edge (10.8 mm³ on a 40×40×10 plate on the height-map engine, 2.1 mm³ on the 3D model).
  The model's distance field rounds sharp convex edges slightly; the samples there are slivers, not material to remove.
- **Claude is not told** that model threads are usually cosmetic, so it may "fix" a correct thread (drop G76 passes,
  pick a smaller tap) to make a reported gouge go away.

## What to build

1. **Find the threads the program cuts**, per operation, from the simulation result (`out.result.moves`, see
   `public/js/sim/job.js` → `runOp`):
   - Lathe: feed moves with `move.thread` (the lead) — cycles `G76 thread`, `G92 thread cycle`, `G32 thread`. Positions
     are work coordinates with X as a **radius**. Get the Z range, pitch, root (minor) diameter, and major diameter
     (the stock/finished diameter there), external vs internal (bore).
   - Mill: moves of a tool with `type === 'tap'` (cycles `G84 tap` / `G74 left-hand tap`): hole position(s), depth,
     nominal diameter (tool `diameter`), pitch (tool `pitch`, else F ÷ S of the tapping moves). Thread mills
     (`type === 'threadmill'`, helical moves) the same way if practical.
2. **Judge each thread as a thread** in `compareMill` / `compareLathe`:
   - Leave the thread zone out of the surface comparison (lathe: its Z range ± about one pitch, between minor and
     major radius; mill: around the hole axis within the tapped depth, out to the major radius plus tolerance), so it
     produces no GOUGE / excess entries.
   - Instead check the model in that zone and report one line per thread, for example:
     - `Thread M20×1.5 external (T4, G76) Z-1.5..Z-19.5, Ø19.8 / root Ø18.16: the model has a plain Ø19.8 cylinder there (cosmetic thread) — consistent.`
     - `Tapped hole M8×1.25 (T2, G84) at X0 Y0 to Z-11: the model has a plain Ø6.8 hole there (tap drill for M8 — cosmetic thread) — consistent.`
     - `… the model has a modeled thread of pitch 1.0 — MISMATCH: the program cuts pitch 1.5.`
     - `… the model has a Ø5 hole (tap drill for M6) — MISMATCH: the program taps M8.`
   - Consistent means: a plain diameter at the major diameter (external) or at the tap drill / minor / nominal
     diameter (internal) within a sensible tolerance, or a modeled thread with the same pitch (for a helix in the model,
     detect the pitch from the mesh, e.g. the oscillating radius along Z — the lathe profile code in
     `public/js/sim/inspect.js` has `findThreads`, and `latheProfileArrays` in `model.js` cuts the mesh exactly).
     Anything else is a mismatch and must stand out to Claude.
   - Real problems outside thread zones must still be reported exactly as now.
3. **Label threads in the measured geometry** (`partSummary` in `public/js/sim/inspect.js`): a hole that was tapped
   says so (`through hole Ø7.96 — tapped M8×1.25 by T2 to Z-11`); lathe thread rows say which tool/cycle cut them.
4. **Remove the edge false alarm** in `compareMill`: do not report excess/gouge slivers along convex edges (e.g. only
   count samples that are off by more than the tolerance on a solid neighbourhood — an erosion / thickness test — or
   drop regions thinner than about twice the tolerance). Real excess and gouges in `test/model.mjs` must still be found.
5. **Tell Claude** in `prompts/system.md` (Precision or Operations section): model threads are usually cosmetic (plain
   cylinder at the major Ø, or a hole at tap-drill size); the thread spec comes from the drawing or the user; the
   comparison checks threads separately and reports consistent / MISMATCH; do not change a correct thread to silence a
   report.

## Done when

- `node test/threads.mjs` passes (it encodes the cases above, including that real gouges are still caught). Do not
  weaken the test; if you believe a check in it is wrong, explain why in your report.
- `npm run check` still passes (all the existing tests). Then add `node test/threads.mjs` to the `check` script in
  `package.json`.
- The service is restarted (`sudo systemctl restart cnc-studio`) and the site loads.
- Keep the existing code style: small functions, comments only where the why is not obvious, same naming.

Report back: what you changed (files and functions), an example of the new report lines for a lathe thread and a
tapped hole, and the test results.
