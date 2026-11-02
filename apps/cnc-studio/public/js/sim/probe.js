// The touch probe (Iteration 2): where a G31 skip move stops, what the stylus and body hit on the
// way, and whether a setup's work zero can be set with the probe at all (the datum check).
import { invertX, applyX, VoxelSim } from './voxelsim.js';
import { MillSim } from './millsim.js';
import { SimRunner } from './simrunner.js';
import { materialField, depthGrid, viewFeatures, VIEWS, footprint } from './inspect.js';
import { setupTransform } from '../gcode/setup.js';
import { probeAlong } from './probegeom.js';
export { materialTest, probeContact, probeAlong } from './probegeom.js';

const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
const f3 = (v) => (Math.abs(v) < 0.0005 ? 0 : v).toFixed(3);

// The oracle the interpreter asks at every G31: it keeps its own simulation of the operation,
// advanced over the moves emitted so far, so the probe meets the material as it is at that line.
// ctx: { start, engine, X, setup, jobMode, budget, resolution, rotReverse, fixturesFor }
export function makeProbeOracle(ctx) {
  let sim = null, runner = null, engine = ctx.engine, X = ctx.X, failed = null;
  const ensure = (interp) => {
    if (sim || failed) return;
    if (ctx.start) {
      sim = engine === 'voxel' ? ctx.start.clone() : ctx.start;
    } else {
      const st = interp.stock;
      if (!st || !st.min) { failed = 'the stock is auto-fitted: give the program a (STOCK …) line so the probe has material to find'; return; }
      if (ctx.jobMode || ctx.setup.orient.length || interp.usesRotary || ctx.setup.vise) {
        sim = VoxelSim.fromBox(st.min, st.max, interp.tools, { budget: ctx.budget });
        engine = 'voxel';
        if (ctx.setup.orient.length || ctx.setup.zero) X = setupTransform(ctx.setup, { min: st.min, max: st.max }, 'mill');
      } else {
        sim = new MillSim(st, interp.tools, { resolution: ctx.resolution || 420 });
        engine = 'height';
      }
    }
    if (engine === 'voxel') sim.xform = X;
    const result = { moves: interp.moves, tools: interp.tools, machine: 'mill', stock: interp.stock || { min: [0, 0, 0], max: [0, 0, 0] }, home: [0, 0, 0] };
    runner = new SimRunner(result, sim, { xform: X, rotReverse: ctx.rotReverse });
    runner.fixtures = ctx.fixturesFor ? ctx.fixturesFor(sim, engine, X) : [];
    runner.quiet = true; // its issues are reported by the real run
  };
  return {
    touch(interp, a, b, tool) {
      ensure(interp);
      if (!sim) return { error: failed };
      const last = interp.moves[interp.moves.length - 1];
      runner.advanceTo((last ? last.t1 : 0) + 1);
      return probeAlong(sim, a, b, tool, X);
    },
  };
}

// ---------------------------------------------------------------- the datum check
// Can the work zero of this setup be set with the probe? Per axis: the reference it would use,
// whether that is machined or raw, how much of it the ball can reach, and the verdict.
export function datumCheck(out, probe) {
  if (!probe || out.result.machine !== 'mill' || !out.start) return null;
  const r = probe.diameter / 2, need = probe.diameter + 1; // the ball plus a little clearance
  const fresh = out.engine !== 'voxel';
  const field = materialField({ final: out.start, engine: fresh ? 'height' : 'voxel', X: out.X });
  const b = field.bounds;
  if (!b) return null;
  const top = b.max[2];
  const jaws = (out.fixtures || []).filter((f) => /jaw/.test(f.name));
  const jawTop = jaws.length ? Math.max(...jaws.map((f) => f.max[2])) : null;
  // machined or raw: a voxel part remembers which tool cut each cell
  const machinedAt = (p) => {
    if (fresh || !out.start.toolIdx) return false;
    const inv = invertX(out.X);
    const q = applyX(inv, p);
    const s = out.start;
    const i = Math.round((q[0] - s.origin[0]) / s.h), j = Math.round((q[1] - s.origin[1]) / s.h), k = Math.round((q[2] - s.origin[2]) / s.h);
    if (i < 0 || j < 0 || k < 0 || i >= s.nx || j >= s.ny || k >= s.nz) return false;
    return s.toolIdx[(k * s.ny + j) * s.nx + i] > 0;
  };
  const surfaceKind = (pts) => { const n = pts.filter(machinedAt).length; return n > pts.length / 2 ? 'machined' : 'raw stock'; };
  const outline = footprint(field, VIEWS.top);
  const grid = depthGrid(field, 'top', 160);
  const feats = viewFeatures(field, grid).feats;
  const lines = [`Setting the work zero for ${out.name} with the ${probe.name || 'touch probe'} (ball Ø${probe.diameter}, stylus ${probe.stylusLength} mm, body Ø${probe.holderDiameter}):`];
  const issues = [];
  // Z: the top face
  const topPts = [];
  for (const [fx, fy] of [[0.3, 0.3], [0.7, 0.3], [0.5, 0.7], [0.5, 0.5]]) topPts.push([b.min[0] + (b.max[0] - b.min[0]) * fx, b.min[1] + (b.max[1] - b.min[1]) * fy, top - 0.1]);
  lines.push(`  Z: top face at Z${f2(top)} (${surfaceKind(topPts)}) — probeable (G65 P9811 Z${f2(top)}).`);
  // X and Y: opposite outside faces, or a round feature
  const axes = [{ ax: 'X', k: 0, jawsOn: false }, { ax: 'Y', k: 1, jawsOn: jaws.length > 0 }];
  const round = feats.filter((f) => f.shape?.kind === 'circle' && (f.kind === 'hole' || f.kind === 'island') && (f.kind === 'island' || f.shape.d >= probe.diameter + 2) && (f.through || Math.abs((f.wallFrom ?? top) - (f.wallTo ?? top)) >= 3));
  const roundText = round.length ? round.slice(0, 3).map((f) => `${f.kind === 'hole' ? `Ø${f2(f.shape.d)} bore` : `Ø${f2(f.shape.d)} boss`} at X${f2(f.shape.cx)} Y${f2(f.shape.cy)} (G65 P9814 D${f2(f.shape.d)}${f.kind === 'island' ? ` Z${f2(Math.max(b.min[2], (f.wallTo ?? top) + 2))}` : ''})`).join(', ') : '';
  for (const { ax, k, jawsOn } of axes) {
    const straight = outline && (outline.kind === 'rect' || outline.kind === 'slot') && !(outline.rc > 2);
    const lo = b.min[k], hi = b.max[k];
    const free = jawsOn && jawTop !== null ? top - jawTop : top - b.min[2];
    const faces = `${ax}${f2(lo)} and ${ax}${f2(hi)}`;
    const mid = (a) => (a === 0 ? [(lo + hi) / 2, (b.min[1] + b.max[1]) / 2, top - Math.min(2, free / 2)] : [(b.min[0] + b.max[0]) / 2, (lo + hi) / 2, top - Math.min(2, free / 2)]);
    const facePts = k === 0 ? [[lo + 0.2, mid(0)[1], mid(0)[2]], [hi - 0.2, mid(0)[1], mid(0)[2]]] : [[mid(1)[0], lo + 0.2, mid(1)[2]], [mid(1)[0], hi - 0.2, mid(1)[2]]];
    const kind = surfaceKind(facePts);
    if (!straight) {
      if (round.length) lines.push(`  ${ax}: the outline is not a plain rectangle, so no flat sides to probe — use ${roundText}.`);
      else { lines.push(`  ${ax}: no flat outside faces and no bore or boss to probe — difficult; leave a reference feature or a flat pad, or set ${ax} from the fixture.`); issues.push({ level: 'warning', text: `Work zero ${ax} of ${out.name} cannot be set with the probe: no flat outside faces and no bore or boss reachable — plan a reference (a bore, a boss or a flat pad) or set it from the fixture`, line: null, file: 'main', callLine: null, source: 'datum' }); }
      continue;
    }
    if (free >= need) {
      const warn = kind === 'raw stock' && !fresh;
      lines.push(`  ${ax}: sides at ${faces} (${kind}), ${f2(free)} mm free above ${jawsOn && jawTop !== null ? 'the jaw top' : 'the bottom'} — probeable with the web cycle (G65 P9812 ${ax}${f2(hi - lo)} Z${f2(top - Math.min(free - r - 0.5, 3 + r))})${warn ? `; raw stock gives a poor reference — ${round.length ? `better: ${roundText}` : 'expect ±0.2 mm'}` : ''}.`);
      continue;
    }
    if (round.length) { lines.push(`  ${ax}: the sides at ${faces} are inside the jaws (${f2(free)} mm free, ${f2(need)} needed) — use ${roundText}.`); continue; }
    lines.push(`  ${ax}: the sides at ${faces} are inside the jaws (${f2(free)} mm free above the jaw top, ${f2(need)} needed for the ball) and no bore or boss is reachable — difficult; leave the part ${f2(need - free)} mm higher in the vise, or leave a reference, or set ${ax} from the fixed jaw.`);
    issues.push({ level: 'warning', text: `Work zero ${ax} of ${out.name} is difficult to set: the ${ax} sides are inside the vise jaws (${f2(free)} mm free, ${f2(need)} needed) and no bore or boss is reachable — raise the part, leave a reference, or take ${ax} from the fixed jaw`, line: null, file: 'main', callLine: null, source: 'datum' });
  }
  return { text: lines.join('\n'), issues };
}

// what the probe did in a run, for the report
export function probeReport(result) {
  const o = [];
  for (const p of result.probes || []) {
    const where = `X${f3(p.point[0])} Y${f3(p.point[1])} Z${f3(p.point[2])}`;
    if (p.hit === 'ball') o.push(`  G31 line ${p.callLine + 1}: touched at ${where} (ball centre; surface ${f3(p.radius)} mm further along the approach)`);
    else if (p.hit === 'stylus' || p.hit === 'body') o.push(`  G31 line ${p.callLine + 1}: the probe ${p.hit} hit the part at ${where} before the ball touched — collision`);
    else if (p.error) o.push(`  G31 line ${p.callLine + 1}: not simulated — ${p.error}`);
    else o.push(`  G31 line ${p.callLine + 1}: no skip signal — reached ${where} without touching`);
  }
  for (const c of result.probeCycles || []) {
    const v = c.vars;
    const got = [];
    if (c.axes.x) got.push(`X ${f3(v[135])} (error ${f3(v[139])})`);
    if (c.axes.y) got.push(`Y ${f3(v[136])} (error ${f3(v[140])})`);
    if (c.axes.z) got.push(`Z ${f3(v[137])} (error ${f3(v[141])})`);
    if (c.num === 9812 || c.num === 9814) got.push(`size ${f3(v[138])} (error ${f3(v[142])})`);
    o.push(`  G65 P${c.num} line ${c.callLine + 1}: ${got.join(', ') || 'done'}${c.offset ? ` → work offset ${c.offset} updated` : ''}`);
  }
  return o.length ? `Probing:\n${o.join('\n')}` : '';
}
