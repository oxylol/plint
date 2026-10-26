// Multi-setup jobs: flips, stock hand-off, lathe->mill, rotary, vise. node test/job.mjs
import { simulateJob } from '../public/js/sim/job.js';
import { applyX, invertX } from '../public/js/sim/voxelsim.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const errs = (o) => o.issues.filter((i) => i.level === 'error').map((i) => i.text);
const run = (ops) => simulateJob(ops.map((o, i) => ({ id: String(i), fileName: `p${i}.nc`, ...o })), ops.length - 1, { voxelBudget: 3e6, interp: {} });

// 1) mill: pocket on top, flip, face 1 mm + slot on the other side
{
  const op10 = `%
O10 (OP10)
(STOCK X-50:50 Y-40:40 Z-20:0)
(TOOL T1 FLAT D10 L30)
T1 M6
S6000 M3
G0 X-15 Y0 Z5
G1 Z-5 F300
G1 X15 F800
G0 Z25
M30
%`;
  const op20 = `%
O20 (OP20 - BACK SIDE)
(SETUP OP20)
(STOCK FROM OP10)
(ORIENT FLIP X)
(ZERO TOP CENTER)
(VISE GRIP 6)
(TOOL T1 FLAT D10 L30)
T1 M6
S6000 M3
G0 X-60 Y-30 Z5
G1 Z-1 F500
G1 X60 F1500
Y-20
X-60
Y-10
X60
Y0
X-60
Y10
X60
Y20
X-60
Y30
X60
G0 Z5
G0 X0 Y25
G1 Z-4 F300
G1 Y-25 F800
G0 Z25
M30
%`;
  const t0 = performance.now();
  const [a, b] = run([{ machine: 'mill', text: op10 }, { machine: 'mill', text: op20 }]);
  console.log(`     2-op mill job: ${(performance.now() - t0).toFixed(0)} ms (voxel ${a.sim.nx}×${a.sim.ny}×${a.sim.nz}, h=${a.sim.h.toFixed(2)})`);
  ok(a.engine === 'voxel' && b.engine === 'voxel', 'both operations on the 3D engine');
  ok(b.src === 'OP10' && b.result.stock.source === 'result of OP10', 'OP20 stock = result of OP10');
  ok(errs(a).length === 0 && errs(b).length === 0, 'no errors ' + [...errs(a), ...errs(b)].join('; '));
  const f = b.final;
  ok(!f.isMaterial(0, 0, -3) && f.isMaterial(0, 0, -6), 'OP10 pocket (top, 5 deep) still present');
  ok(!f.isMaterial(30, 0, -19.5) && f.isMaterial(30, 0, -18.5), 'OP20 faced the old bottom: part is now 19 thick');
  // OP20 slot along work Y at X0, 4 deep from the new top; work X0 = part X0, part z = -(4 + 1) below old top? new top = part z-19
  ok(!f.isMaterial(0, 10, -17) && f.isMaterial(0, 10, -14.5), 'OP20 slot cut into the back side (floor 4 below the faced surface)');
  ok(Math.abs(f.materialBounds().max[2] - 0) < 0.4, 'top face untouched by OP20');
}

// 2) lathe flip: turn the front, flip, face to length and turn the back
{
  const op10 = `%
O1 (OP10 FRONT)
(STOCK D40 L60)
(TOOL T1 TURN R0.4)
T0101
G97 S1500 M3
G99
G0 X42 Z2
G0 X30
G1 Z-25 F0.2
G0 X42
G0 Z2
M30
%`;
  const op20 = `%
O2 (OP20 BACK)
(STOCK FROM OP10)
(ORIENT FLIP X)
(CHUCK GRIP 20)
(TOOL T1 TURN R0.4)
T0101
G97 S1500 M3
G99
G0 X42 Z0
G1 X-1.6 F0.15
G0 Z2
G0 X34
G1 Z-20 F0.2
G0 X42
M30
%`;
  const [a, b] = run([{ machine: 'lathe', text: op10 }, { machine: 'lathe', text: op20 }]);
  ok(b.engine === 'lathe' && b.src === 'OP10', 'lathe OP20 continues from OP10');
  const s = b.final;
  const dia = (z) => s.outer[Math.round((z - s.zMin) / s.dz)] * 2;
  // OP20 Z0 = old back end (old Z-75: 60 out + 15 grip); turned Ø34 for 20; the old front Ø30 is now at the chuck end
  ok(Math.abs(dia(-10) - 34) < 0.02, `back end turned Ø34 (got Ø${dia(-10).toFixed(3)})`);
  ok(Math.abs(dia(-40) - 40) < 0.02, `middle still Ø40 (got Ø${dia(-40).toFixed(3)})`);
  ok(Math.abs(dia(s.zMin + 10) - 30) < 0.02, `old front Ø30 now in the chuck (got Ø${dia(s.zMin + 10).toFixed(3)})`);
  ok(errs(b).length === 0, 'no errors ' + errs(b).join('; '));
}

// 3) lathe -> mill: turned Ø30 shaft, then a flat milled on the mill (part axis vertical)
{
  const op10 = `%
(STOCK D30 L40)
(TOOL T1 TURN R0.4)
T0101
G97 S1500 M3
G99
G0 X32 Z2
G0 X24
G1 Z-15 F0.2
G0 X32
M30
%`;
  const op20 = `%
(SETUP OP20)
(STOCK FROM OP10)
(ORIENT ROT Y90)
(ZERO TOP CENTER)
(TOOL T2 FLAT D10 L30)
T2 M6
S6000 M3
G0 X-40 Y-20 Z5
G1 Z-3 F300
G1 Y20 F800
X-30
Y-20
X-20
Y20
G0 Z25
M30
%`;
  const [a, b] = run([{ machine: 'lathe', text: op10 }, { machine: 'mill', text: op20 }]);
  ok(b.engine === 'voxel' && errs(b).length === 0, 'lathe part milled on the mill ' + errs(b).join('; '));
  const top = b.result.stock.max[2];
  ok(Math.abs(top) < 0.5, `part lies horizontally with its top at the work zero (stock top Z${top})`);
  // bar (40 out + 15 in the chuck) lies along work X centred on X0: part Z-55..0 -> work X-27.5..27.5,
  // so the passes at X-40..-20 (D10) flatten the Ø30 end that sat in the chuck
  const inv = invertX(b.X);
  const probe = (x, y, z) => b.final.isMaterial(...applyX(inv, [x, y, z]));
  ok(!probe(-22, 0, -2) && probe(-22, 0, -4), 'flat milled 3 deep on top of the Ø30 end');
  ok(probe(5, 0, -2) && !probe(20, 0, -2), 'turned Ø24 end (part Z0..-15) untouched at the +X end');
}

// 4) rotary: slot on top at A0, then A90 and a slot on what was the -Y... side
{
  const op = `%
(STOCK X-30:30 Y-15:15 Z-15:15)
(ROTARY A)
(TOOL T1 FLAT D6 L30)
T1 M6
S6000 M3
G0 A0
G0 X-20 Y0 Z20
G1 Z13 F300
G1 X20 F800
G0 Z30
G0 A90
G0 X-20 Y0 Z20
G1 Z13 F300
G1 X20 F800
G0 Z30
M30
%`;
  const [a] = run([{ machine: 'mill', text: op }]);
  ok(a.engine === 'voxel' && a.result.usesRotary, 'rotary program uses the 3D engine');
  const f = a.final;
  ok(!f.isMaterial(0, 0, 14) && f.isMaterial(0, 0, 12), 'slot at A0 on top (+Z face)');
  // A90 turns the part +90° about X: part -Y side comes to the top
  ok(!f.isMaterial(0, -14, 0) && f.isMaterial(0, -12, 0), 'slot at A90 on the part\'s −Y face');
  ok(f.isMaterial(0, 14, 0) && f.isMaterial(0, 0, -14), 'other faces untouched');
}

// 5) vise: a pass too low next to the jaw is a collision
{
  const op = (z) => `%
(STOCK X-40:40 Y-20:20 Z-20:0)
(VISE GRIP 8)
(TOOL T1 FLAT D10 L30)
T1 M6
S6000 M3
G0 X-50 Y-24 Z5
G1 Z${z} F300
G1 X50 F800
G0 Z25
M30
%`;
  const [safe] = run([{ machine: 'mill', text: op(-5) }]);
  const [crash] = run([{ machine: 'mill', text: op(-15) }]);
  ok(!safe.issues.some((i) => /vise/.test(i.text)), 'side cut above the jaws: no vise collision');
  ok(crash.issues.some((i) => /hits the vise jaw/.test(i.text)), 'side cut below the jaw top: vise collision reported');
}
process.exitCode = fails ? 1 : 0;
