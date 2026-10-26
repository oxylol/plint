// A job = ordered operations (programs). Each operation may start from the material
// another operation left ((STOCK FROM OP10)), re-clamped in a new orientation.
import { parseProgram } from '../gcode/parser.js';
import { interpret } from '../gcode/interpreter.js';
import { parseSetup, setupTransform, isIdentity } from '../gcode/setup.js';
import { IDENTITY, VoxelSim, applyX, mulXform } from './voxelsim.js';
import { MillSim } from './millsim.js';
import { LatheSim } from './lathesim.js';
import { SimRunner } from './simrunner.js';
import { EngagementRecorder } from './engagement.js';

export function opLabel(i) { return `OP${(i + 1) * 10}`; }

// name an operation is known by: (SETUP name) or its position
export function opNames(ops) {
  return ops.map((op, i) => parseSetup(parseProgram(op.text).comments).name || opLabel(i));
}

function boundsUnder(X, b) {
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const cx of [b.min[0], b.max[0]]) for (const cy of [b.min[1], b.max[1]]) for (const cz of [b.min[2], b.max[2]]) {
    const p = applyX(X, [cx, cy, cz]);
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  }
  return { min: mn.map(r3), max: mx.map(r3) };
}
const r3 = (v) => Math.round(v * 1000) / 1000;

function findSource(outs, ref) {
  const want = ref.trim().toLowerCase();
  const base = (s) => (s || '').toLowerCase().replace(/\.[a-z0-9]+$/, '');
  return outs.find((o) => o.name.toLowerCase() === want) || outs.find((o) => (o.op.fileName || '').toLowerCase() === want || base(o.op.fileName) === base(want));
}

// per-tool cut envelopes (feed moves, work coordinates)
export function toolEnvelopes(result) {
  const perTool = {};
  for (const m of result.moves) {
    if (m.kind !== 'feed') continue;
    const t = (perTool[m.tool] ||= { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity], length: 0, time: 0 });
    for (const p of m.pts) for (let k = 0; k < 3; k++) { if (p[k] < t.min[k]) t.min[k] = p[k]; if (p[k] > t.max[k]) t.max[k] = p[k]; }
    t.length += m.len;
    t.time += m.t1 - m.t0;
  }
  return perTool;
}

// vise jaws as boxes in the work frame, from the part's work-frame bounds
export function viseBoxes(vise, wb) {
  const jawT = 25, cx = (wb.min[0] + wb.max[0]) / 2, W = Math.max(vise.width, wb.max[0] - wb.min[0] + 20);
  const z0 = wb.min[2] - vise.jaw, z1 = wb.min[2] + vise.grip;
  return [
    { name: 'vise jaw (−Y)', min: [cx - W / 2, wb.min[1] - jawT, z0], max: [cx + W / 2, wb.min[1], z1] },
    { name: 'vise jaw (+Y)', min: [cx - W / 2, wb.max[1], z0], max: [cx + W / 2, wb.max[1] + jawT, z1] },
  ];
}

export function simulateJob(ops, upto, opts, cache = new Map()) {
  const outs = [];
  const names = opNames(ops);
  const jobMode = ops.length > 1;
  for (let i = 0; i <= Math.min(upto, ops.length - 1); i++) {
    const op = ops[i];
    const setup = parseSetup(parseProgram(op.text).comments);
    const name = names[i];
    let src = null, srcError = null;
    if (setup.stockFrom) {
      src = findSource(outs, setup.stockFrom);
      if (!src) srcError = `STOCK FROM ${setup.stockFrom}: no earlier operation with that name or file (operations run in tab order)`;
    }
    const dupe = names.slice(0, i).some((n) => n.toUpperCase() === name.toUpperCase());
    const key = JSON.stringify([op.text, op.machine, opts.key, src ? src.key : null, jobMode, dupe]);
    const hit = cache.get(op.id);
    if (hit && hit.key === key) { outs.push(hit); continue; }
    const out = runOp(op, name, setup, src, srcError, opts, jobMode);
    if (dupe) out.issues.unshift({ level: 'warning', text: `An earlier operation is also called ${name} — (STOCK FROM ${name}) uses the first one; give this one its own (SETUP name)`, line: null, file: 'main', callLine: null });
    out.key = key;
    cache.set(op.id, out);
    outs.push(out);
  }
  return outs;
}

function runOp(op, name, setup, src, srcError, opts, jobMode) {
  const machine = op.machine;
  const pre = [];
  if (srcError) pre.push({ level: 'error', text: srcError, line: null, file: 'main', callLine: null });
  let X = IDENTITY, start = null, engine = null, stockOverride = null;
  // frames: the job's part frame is the first operation's simulation frame; simFromPart maps it
  // to this operation's simulation frame (lathe flips rebuild the bar in a new frame)
  let simFromPart = IDENTITY;
  const budget = opts.voxelBudget || 5e6;

  if (src) {
    if (machine === 'lathe') {
      if (src.engine !== 'lathe') {
        pre.push({ level: 'error', text: `STOCK FROM ${src.name}: a milled part cannot be chucked in the lathe simulation — starting from fresh stock`, line: null, file: 'main', callLine: null });
      } else {
        const b = src.final.materialBounds();
        X = setupTransform({ ...setup, orient: setup.orient.length ? [{ axis: 'x', deg: 180 }] : [] }, b, 'lathe');
        if (setup.orient.some((o) => o.axis !== 'x' || Math.abs(o.deg) !== 180)) pre.push({ level: 'warning', text: 'On the lathe only (ORIENT FLIP X) — turning the part around in the chuck — is possible; treated as a flip', line: null, file: 'main', callLine: null });
        start = LatheSim.fromProfile(src.final, X, setup.chuck?.grip ?? 15);
        stockOverride = { ...start.stock, source: `result of ${src.name}` };
        engine = 'lathe';
        simFromPart = mulXform(X, src.simFromPart || IDENTITY);
      }
    } else {
      let base = null;
      if (src.engine === 'voxel') base = src.final;
      else if (src.engine === 'lathe') base = VoxelSim.fromLathe(src.final, {}, { budget });
      else if (src.engine === 'height') base = VoxelSim.fromHeightmap(src.final, {}, { budget });
      const mb = base.materialBounds() || base.bounds;
      const pb = { min: mb.min.map(r3), max: mb.max.map(r3) };
      X = setupTransform(setup, pb, 'mill');
      start = src.engine === 'voxel' ? base.clone() : base;
      stockOverride = { kind: 'box', ...boundsUnder(X, pb), source: `result of ${src.name}` };
      engine = 'voxel';
      simFromPart = src.simFromPart || IDENTITY;
    }
  }

  const interp = { ...(typeof opts.interp === 'function' ? opts.interp(machine) : opts.interp), machine };
  if (stockOverride) interp.stockOverride = stockOverride;
  else if (opts.manualStock?.[machine]) interp.stockOverride = { ...opts.manualStock[machine], source: 'manual' };
  const result = interpret(op.text, interp);

  if (!start) {
    if (machine === 'lathe') {
      start = new LatheSim({ ...result.stock, grip: setup.chuck?.grip ?? 15 }, result.tools);
      engine = 'lathe';
    } else if (jobMode || setup.orient.length || result.usesRotary || setup.vise) {
      const st = result.stock;
      start = VoxelSim.fromBox(st.min, st.max, result.tools, { budget });
      engine = 'voxel';
      if (setup.orient.length || setup.zero) X = setupTransform(setup, { min: st.min, max: st.max }, 'mill');
    } else {
      start = new MillSim(result.stock, result.tools, { resolution: opts.resolution || 420 });
      engine = 'height';
    }
  }
  start.tools = result.tools;
  // removed volume counts this operation only
  if (engine === 'voxel' && src) start.initialVolume = start.materialVolume();

  let sim;
  if (engine === 'voxel') {
    sim = start.clone();
    sim.pristine = start;
    sim.xform = X;
  } else if (engine === 'lathe') {
    sim = start.clone ? start.clone() : start;
    if (!sim.pristine) sim.pristine = { outer: sim.outer.slice(), inner: sim.inner.slice(), toolOuter: sim.toolOuter.slice(), toolInner: sim.toolInner.slice() };
  } else {
    sim = start;
  }
  const runner = new SimRunner(result, sim, { xform: X, rotReverse: !!setup.rotary?.reverse });
  // fixtures (work frame)
  const fixtures = [];
  if (setup.vise && machine === 'mill') {
    const wb = engine === 'voxel' ? boundsUnder(X, sim.materialBounds() || sim.bounds) : { min: result.stock.min, max: result.stock.max };
    fixtures.push(...viseBoxes(setup.vise, wb));
  }
  runner.fixtures = fixtures;
  // only the full run records tool engagement; playback re-runs the same moves
  runner.engagement = new EngagementRecorder(result, sim);
  const t0 = performance.now();
  runner.runAll();
  const simMs = performance.now() - t0;
  const engagement = runner.engagement.finish();
  runner.engagement = null;
  const final = sim.clone();
  const issues = [...pre, ...result.messages, ...runner.finalIssues(), ...extraIssues(result, sim, engine)];
  const workFromPart = mulXform(engine === 'voxel' ? X : IDENTITY, simFromPart);
  return { op, name, setup, engine, X, result, sim, runner, final, start, issues, src: src?.name || null, fixtures, simMs, simFromPart, workFromPart, engagement };
}

function extraIssues(result, sim, engine) {
  const out = [];
  if (engine === 'lathe' && sim.partedOff !== null) {
    out.push({ level: 'info', text: `The part is cut off at Z${sim.partedOff.toFixed(2)} — everything beyond falls away`, line: null, file: 'main', callLine: null, source: 'sim' });
  }
  if (engine === 'height') {
    let through = 0;
    for (let k = 0; k < sim.heights.length; k++) if (sim.heights[k] < sim.bottom) through++;
    if (through) out.push({ level: 'info', text: `Cuts through the stock bottom (${((through * sim.cx * sim.cy) / 100).toFixed(1)} cm² area) — make sure the part is on parallels / a sacrificial plate`, line: null, file: 'main', callLine: null, source: 'sim' });
  }
  return out;
}

export { isIdentity };
