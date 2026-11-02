// The touch probe: G31 stops at the material, the stylus and body collide, the built-in cycles
// measure and set offsets, and the datum check says when a work zero is hard to set. node test/probe.mjs
import { simulateJob } from '../public/js/sim/job.js';
import { VoxelSim, IDENTITY } from '../public/js/sim/voxelsim.js';
import { probeAlong } from '../public/js/sim/probegeom.js';
import { normalizeTool } from '../public/js/gcode/tools.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const opts = { interp: (m) => ({ machine: m }), features: { probe: true, craft: false } };
const run = (ops) => simulateJob(ops.map((o, i) => ({ id: `p${i}${Math.random()}`, fileName: `p${i}.nc`, machine: 'mill', ...o })), ops.length - 1, opts);

// ---- geometry: an overhang and a narrow slot
const box = (p, mn, mx) => { const d = [0, 1, 2].map((k) => Math.max(mn[k] - p[k], p[k] - mx[k])); return Math.hypot(...d.map((v) => Math.max(v, 0))) + Math.min(0, Math.max(...d)); };
const over = VoxelSim.fromBox([-20, -20, -30], [20, 20, 0], {}, { budget: 3e5 });
over.fill((x, y, z) => Math.min(box([x, y, z], [-20, 0, -30], [20, 20, 0]), box([x, y, z], [-20, -20, -10], [20, 0, 0]), box([x, y, z], [-20, -20, -30], [20, 0, -20])));
const longP = normalizeTool({ num: 20, type: 'probe', diameter: 6, stylusLength: 50, shankDiameter: 4.5, holderDiameter: 40 }, 'mill');
const shortP = normalizeTool({ num: 21, type: 'probe', diameter: 6, stylusLength: 15, shankDiameter: 4.5, holderDiameter: 40 }, 'mill');
let r = probeAlong(over, [0, -10, 10], [0, -10, -5], longP, IDENTITY);
ok(r.hit === 'ball' && near(r.point[2], 3, 0.25), `ball touches the lip's top from above at Z${r.point[2].toFixed(2)} (ball centre, ~3)`);
r = probeAlong(over, [0, -30, -15], [0, 5, -15], longP, IDENTITY);
ok(r.hit === 'stylus' && near(r.point[1], -22.25, 0.6), `the recess under the lip: the stylus hits the lip at Y${r.point[1].toFixed(2)} before the ball reaches the back wall (${r.hit})`);
const slot = VoxelSim.fromBox([-20, -20, -30], [20, 20, 0], {}, { budget: 3e5 });
slot.fill((x, y, z) => Math.min(box([x, y, z], [-20, -20, -30], [20, -5, 0]), box([x, y, z], [-20, 5, -30], [20, 20, 0]), box([x, y, z], [-20, -20, -30], [20, 20, -25])));
r = probeAlong(slot, [0, 0, 10], [0, 0, -29], shortP, IDENTITY);
ok(r.hit === 'body' && near(r.point[2], -12, 0.6), `a 15 mm stylus down a 30 mm slot: the body hits the top at ball Z${r.point[2].toFixed(2)} (~-12)`);
r = probeAlong(slot, [0, 0, 10], [0, 0, -29], longP, IDENTITY);
ok(r.hit === 'ball' && near(r.point[2], -22, 0.4), `a 50 mm stylus reaches the slot floor: ball Z${r.point[2].toFixed(2)} (~-22)`);
r = probeAlong(slot, [0, 0, 10], [0, 0, 5], longP, IDENTITY);
ok(r.hit === null && r.point[2] === 5, 'nothing touched: the move runs to its end');

// ---- G31 in a program on the height map
const head = '%\nO1 (PROBE)\n(STOCK X-50:50 Y-40:40 Z-20:0)\nG90 G17 G21 G40 G49 G80 G94\nT20 M06\nG54 G00 X0. Y0.\nG43 Z10. H20\n';
let [out] = run([{ text: head + 'G31 Z-5. F300\nG01 Z10. F1000\nG00 X-60. Y0.\nG00 Z-5.\nG31 X0. F300\nG01 X-60. F1000\nG00 Z10.\nG31 Z20. F300\nG01 X0. Y0. Z-3. F1000\nG00 Z25.\nM30\n%' }]);
const pr = out.result.probes;
ok(pr.length === 3 && pr[0].hit === 'ball' && near(pr[0].point[2], 3, 0.05), `G31 Z stops with the ball centre at Z${pr[0]?.point[2].toFixed(3)} (3.000 on the top face)`);
ok(pr[1].hit === 'ball' && near(pr[1].point[0], -53, 0.3), `G31 X stops at X${pr[1]?.point[0].toFixed(2)} against the side (−53)`);
ok(pr[2].hit === null && near(pr[2].point[2], 20, 1e-6) && out.issues.some((i) => /no skip signal/.test(i.text)), 'G31 away from the part: no skip signal, noted');
ok(out.issues.some((i) => i.level === 'error' && /probe touches the part on a G01 move/.test(i.text)), `a probe into the part on G01 is an error: ${out.issues.find((i) => /G01 move/.test(i.text))?.text.slice(0, 70)}`);
ok(!out.issues.some((i) => /spindle/.test(i.text)) && out.final.removedVolume() < 1e-6, 'no spindle warnings for the probe, and it removed nothing');
const skipMove = out.result.moves.find((m) => m.skip);
ok(skipMove && skipMove.kind === 'feed' && near(skipMove.pts[skipMove.pts.length - 1][2], 3, 0.05), 'the move list holds the shortened G31 move');

// ---- the Z cycle finds a fixture error and corrects G54
[out] = run([{ text: '%\nO2 (DATUM Z)\n(STOCK X-50:50 Y-40:40 Z-20:0)\n(FIXTURE ERROR Z0.2)\n' + head.split('\n').slice(3).join('\n') + 'G65 P9811 Z0. S1 F300\nG00 Z25.\nT1 M06\nS6000 M03\nG00 X-60. Y0.\nG43 Z5. H01\nG01 Z-1. F300\nX60. F800\nG00 Z25.\nM30\n%' }]);
const cyc = out.result.probeCycles[0];
ok(cyc && cyc.num === 9811 && near(cyc.vars[137], 0.2, 0.03) && near(cyc.vars[141], 0.2, 0.03) && cyc.offset === 'G54', `P9811 Z measured Z${cyc?.vars[137]?.toFixed(3)} (error ${cyc?.vars[141]?.toFixed(3)}), ${cyc?.offset} updated`);
const cut = out.result.moves.filter((m) => m.kind === 'feed' && m.tool === 1).pop();
ok(cut && near(cut.pts[cut.pts.length - 1][2], -0.8, 0.03), `after the correction the Z-1 pass runs at machine Z${cut?.pts[cut.pts.length - 1][2].toFixed(3)} (−0.8), 1 mm into the shifted part`);
ok(!out.issues.some((i) => i.level === 'error'), `no errors in the probing program (${out.issues.map((i) => i.text.slice(0, 50)).join(' | ') || 'none'})`);
ok(/Probing:/.test(JSON.stringify(out.result.probes)) || out.result.probes.length >= 1, 'probe records exist for the report');

// ---- the datum check
const plate = (grip, extra = '') => `%\nO3 (SETUP)\n(STOCK X-50:50 Y-40:40 Z-20:0)\n(VISE GRIP ${grip})\n${extra}G90 G17 G21 G40 G49 G80 G94\nT1 M06\nS6000 M03\nG54 G00 X0. Y0.\nG43 Z25. H01\nM30\n%`;
[out] = run([{ text: plate(8) }]);
ok(out.datum && /Z: top face at Z0/.test(out.datum) && /X: sides at X-50.00 and X50.00 .*probeable/.test(out.datum) && /Y: sides at Y-40.00 and Y40.00 .*12.00 mm free above the jaw top .*probeable/.test(out.datum), `grip 8: ${out.datum?.split('\n').slice(1).map((l) => l.trim().slice(0, 60)).join(' / ')}`);
ok(!out.issues.some((i) => i.source === 'datum'), 'no datum warning when the sides are free');
[out] = run([{ text: plate(18) }]);
ok(/Y: the sides .* are inside the jaws \(2.00 mm free .*difficult/.test(out.datum) && out.issues.some((i) => i.source === 'datum' && i.level === 'warning' && /difficult to set/.test(i.text)), `grip 18: ${out.datum?.split('\n').find((l) => /Y:/.test(l))?.trim().slice(0, 110)}`);
// with a bore from the previous operation, Y can be set from it
const op10 = '%\nO10 (BORE)\n(STOCK X-50:50 Y-40:40 Z-20:0)\n(TOOL T5 DRILL D12 L60 "DRILL 12")\nG90 G17 G21 G40 G49 G80 G94\nT5 M06\nS2000 M03\nG54 G00 X20. Y0.\nG43 Z5. H05\nG81 Z-24. R2. F150\nG80\nG00 Z25.\nM30\n%';
const op20 = '%\nO20 (SECOND)\n(SETUP OP20)\n(STOCK FROM OP10)\n(ZERO SAME)\n(VISE GRIP 18)\nG90 G17 G21 G40 G49 G80 G94\nT1 M06\nS6000 M03\nG54 G00 X0. Y0.\nG43 Z25. H01\nM30\n%';
const outs = run([{ text: op10 }, { text: op20 }]);
ok(/Y: the sides .* inside the jaws .* use Ø1[12]\.\d\d bore at X20\.00 Y0\.00 \(G65 P9814 D1[12]/.test(outs[1].datum || '') && !outs[1].issues.some((i) => i.source === 'datum'), `OP20 with a bore: ${outs[1].datum?.split('\n').find((l) => /Y:/.test(l))?.trim().slice(0, 110)}`);

console.log(fails ? `\n${fails} FAILED` : '\nall ok');
process.exit(fails ? 1 : 0);
