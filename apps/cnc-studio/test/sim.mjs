// Material-removal sanity checks + timing on real programs. node test/sim.mjs
import { readFileSync, existsSync } from 'node:fs';
import { interpret } from '../public/js/gcode/interpreter.js';
import { analyzeProgram } from '../public/js/sim/simrunner.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };

// rapid into stock must be reported
{
  const r = interpret(`(STOCK X0:50 Y0:50 Z-10:0)\nT1 M6\nS1000 M3\nG0 X25 Y25 Z5\nG0 Z-3\nG1 X30 F100\nM30`, { machine: 'mill' });
  const a = analyzeProgram(r);
  ok(a.issues.some((i) => /Rapid move/.test(i.text) && i.callLine === 4), 'rapid into stock flagged on line 5');
  // pocket depth: centre node should now be at -3
  const s = a.sim; const k = Math.round(25 / s.cy) * s.nx + Math.round(25 / s.cx);
  ok(Math.abs(s.heights[k] + 3) < 1e-3, `heightmap cut to Z-3 (got ${s.heights[k].toFixed(3)})`);
  const vol = a.removedVolume; // Ø10 hole 3 deep + slot 5x10x3
  const expect = Math.PI * 25 * 3 + 5 * 10 * 3;
  ok(Math.abs(vol - expect) / expect < 0.05, `removed volume ${vol.toFixed(0)} ≈ ${expect.toFixed(0)} mm³`);
}
// lathe: turn Ø40 -> Ø30 over 20 mm with R0.8 tip -> diameter 30 exactly
{
  const r = interpret(`(STOCK D40 L60)\n(T1 TURN R0.8)\nT0101\nG97 S1000 M3\nG0 X42 Z2\nG0 X30\nG1 Z-20 F0.2\nX42\nG0 Z2\nM30`, { machine: 'lathe' });
  const a = analyzeProgram(r);
  const s = a.sim;
  const at = (z) => s.outer[Math.round((z - s.zMin) / s.dz)] * 2;
  ok(Math.abs(at(-10) - 30) < 0.01, `turned to Ø30 at Z-10 (got Ø${at(-10).toFixed(3)})`);
  ok(Math.abs(at(-25) - 40) < 0.01, `untouched Ø40 at Z-25 (got Ø${at(-25).toFixed(3)})`);
  ok(a.issues.length === 0, 'no issues: ' + a.issues.map((i) => i.text).join('; '));
}
// ball mill groove cross-section must follow the ball (regression: profile was taken at the reach edge)
{
  const r = interpret('(STOCK X0:40 Y0:20 Z-10:0)\n(TOOL T1 BALL D6)\nT1 M6\nS9000 M3\nG0 X5 Y10 Z2\nG1 Z-2 F300\nG1 X35\nG0 Z5\nM30', { machine: 'mill' });
  const a = analyzeProgram(r, { resolution: 400 });
  const s = a.sim;
  const h = (x, y) => s.heights[Math.round((y - s.minY) / s.cy) * s.nx + Math.round((x - s.minX) / s.cx)];
  const exp = -2 + 3 - Math.sqrt(9 - 2.25);
  ok(Math.abs(h(20, 10) + 2) < 1e-3 && Math.abs(h(20, 11.5) - exp) < 0.01, `ball groove: centre Z${h(20, 10).toFixed(3)}, 1.5 off Z${h(20, 11.5).toFixed(3)} (expected ${exp.toFixed(3)})`);
  ok(a.issues.length === 0, 'ball groove: no false collisions on retract');
}
// lathe: groove blade plunging deeper than its 16 mm blade -> holder collision; G75 within depth -> none
{
  const prog = (depth) => `(STOCK D60 L60)\n(TOOL T3 GROOVE W3 H16)\nT0303\nG97 S800 M3\nG0 X64 Z-20\nG1 X${60 - 2 * depth} F0.05\nG0 X64\nM30`;
  const ok1 = analyzeProgram(interpret(prog(10), { machine: 'lathe' }));
  const bad = analyzeProgram(interpret(prog(20), { machine: 'lathe' }));
  ok(!ok1.issues.some((i) => /holder/.test(i.text)), 'groove 10 deep with a 16 mm blade: no holder hit');
  ok(bad.issues.some((i) => /holder of T3/.test(i.text)), 'groove 20 deep with a 16 mm blade: holder collision reported');
  const s = ok1.sim; const at = (z) => s.outer[Math.round((z - s.zMin) / s.dz)] * 2;
  ok(Math.abs(at(-18.5) - 40) < 0.02 && Math.abs(at(-21) - 60) < 0.02, `groove W3 from Z-20 toward +Z: Ø${at(-18.5).toFixed(2)} inside, Ø${at(-21).toFixed(2)} outside`);
}
// mill: Ø10 end mill, 30 flutes, 40 stick-out, Ø32 holder — 20 deep is fine, 45 deep rubs the shank and hits the holder
{
  const prog = (z) => `(STOCK X-50:50 Y-50:50 Z-60:0)\n(TOOL T1 FLAT D10 L30 SL40 HD32)\nT1 M6\nS6000 M3\nG0 X0 Y0 Z5\nG1 Z${z} F200\nG1 X20 F600\nG0 Z50\nM30`;
  const fine = analyzeProgram(interpret(prog(-20), { machine: 'mill' }));
  const deep = analyzeProgram(interpret(prog(-45), { machine: 'mill' }));
  ok(fine.issues.length === 0, 'mill 20 deep with 30 mm flutes / 40 stick-out: no issues');
  ok(deep.issues.some((i) => /shank rubs/.test(i.text)) && deep.issues.some((i) => /holder of T1 hits/.test(i.level === 'error' ? i.text : '')), 'mill 45 deep: shank rub warning + holder collision error');
}
// lathe: M20x1.5 thread shows crests at the pitch (not a smooth cylinder)
{
  const r = interpret('(STOCK D20 L40)\n(TOOL T4 THREAD A60)\nT0404\nG97 S800 M3\nG0 X22 Z3\nG76 P010060 Q50 R0.02\nG76 X18.16 Z-20 P920 Q300 F1.5\nG0 X40 Z40\nM30', { machine: 'lathe' });
  const s = analyzeProgram(r).sim;
  let lo = Infinity, hi = -Infinity;
  for (let z = -15; z <= -10; z += 0.05) { const d = s.outer[Math.round((z - s.zMin) / s.dz)] * 2; lo = Math.min(lo, d); hi = Math.max(hi, d); }
  ok(Math.abs(lo - 18.16) < 0.06 && hi > 19.5, `thread profile: root Ø${lo.toFixed(2)}, crest Ø${hi.toFixed(2)}`);
}
// real programs, when present on this machine
for (const [f, machine] of [['C:/Users/Filip/Downloads/1006.nc', 'mill'], ['C:/Users/Filip/Downloads/1007.nc', 'lathe']]) {
  if (!existsSync(f)) continue;
  const r = interpret(readFileSync(f, 'latin1'), { machine });
  const t0 = performance.now();
  const a = analyzeProgram(r);
  console.log(`     ${f.split('/').pop()}: analysis ${(performance.now() - t0).toFixed(0)} ms, removed ${(a.removedVolume / 1000).toFixed(1)} cm³, issues: ${a.issues.map((i) => `L${i.callLine + 1} ${i.text}`).join(' | ') || 'none'}`);
}
process.exitCode = fails ? 1 : 0;
