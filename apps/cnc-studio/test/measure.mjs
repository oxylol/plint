// Measuring in the view: clicks snap to what the simulator knows. node test/measure.mjs
import { simulateJob } from '../public/js/sim/job.js';
import { measureContext, pointItem, distanceItem, snapMill } from '../public/js/sim/measure.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const run = (machine, text) => simulateJob([{ id: `m${Math.random()}`, fileName: 'm.nc', machine, text }], 0, { interp: (m) => ({ machine: m }) })[0];

// a 100×80 plate with a 40×30 pocket 6 deep, R5 corners, and a Ø6.8 through hole at X30 Y20
const mill = `%
O1 (MEASURE)
(STOCK X-50:50 Y-40:40 Z-20:0)
(TOOL T1 FLAT D8 L25 "EM8")
(TOOL T2 DRILL D6.8 L40 "DRILL 6.8")
G90 G17 G21 G40 G49 G80 G94
T1 M06
S6000 M03
G54 G00 X0. Y0.
G43 Z5. H01
G01 Z-6. F300
G01 X-16. Y-11. F800
X16.
Y-5.5
X-16.
Y0.
X16.
Y5.5
X-16.
Y11.
X16.
Y-11.
X-16.
Y11.
G00 Z5.
T2 M06
S3000 M03
G00 X30. Y20.
G43 Z5. H02
G81 Z-23. R2. F200
G80
G00 Z25.
M30
%`;
const out = run('mill', mill);
const ctx = measureContext(out, null);
ok(ctx.feats.some((f) => f.kind === 'hole' && near(f.shape.d, 6.8, 0.3)) && ctx.feats.some((f) => f.kind === 'pocket' && near(f.shape.w, 40, 0.3)), `features found: ${ctx.feats.map((f) => `${f.kind} ${f.shape?.kind} ${f.shape?.w ? f.shape.w.toFixed(1) + '×' + f.shape.h.toFixed(1) : f.shape?.d?.toFixed(2) || ''}`).join(', ')}; levels ${ctx.levels.map((v) => v.toFixed(2)).join(', ')}`);

// a click near the hole snaps to its centre and knows the hole
let it = pointItem(ctx, [30.4, 19.7, 0], [0, 0, 1], 'part');
ok(near(it.p[0], 30, 0.1) && near(it.p[1], 20, 0.1) && /centre of Ø6\.[5-9]\d hole/.test(it.detail) && /through/.test(it.detail) && it.circle, `hole centre: ${it.text} — ${it.detail}`);
// near a pocket corner: the corner (R5 corners are not corners to snap to, so the walls)
it = pointItem(ctx, [-19.6, 14.7, -6], [0, 0, 1], 'part');
ok(/corner of pocket|X-(19|20)\.\d\d wall of pocket/.test(it.detail) && near(it.p[0], -20, 0.3) && near(it.p[1], 15, 0.3) && near(it.p[2], -6, 0.05), `pocket corner: ${it.text} — ${it.detail}`);
// the floor of the pocket by level, with the pocket named
it = pointItem(ctx, [3, 2, -5.9], [0, 0, 1], 'part');
ok(near(it.p[2], -6, 0.01) && /floor of pocket 40\.[0-3]\d×30\.[0-3]\d/.test(it.detail), `pocket floor: ${it.text} — ${it.detail}`);
// the part's corner and top face
it = pointItem(ctx, [49.6, -39.5, 0.1], [0, 0, 1], 'part');
ok(near(it.p[0], 50, 0.2) && near(it.p[1], -40, 0.2) && /corner of the part|X50\.00 side of the part, Y-40\.00 side/.test(it.detail), `part corner: ${it.text} — ${it.detail}`);
it = pointItem(ctx, [-30, 10, 0.02], [0, 0, 1], 'part');
ok(near(it.p[2], 0, 0.01) && /top face/.test(it.detail), `top face: ${it.text} — ${it.detail}`);
// a wall by its normal
it = pointItem(ctx, [-30, 25, -10], [1, 0, 0], 'part');
ok(/wall/.test(it.detail), `wall: ${it.text} — ${it.detail}`);
// distance across the pocket = 40
const a = pointItem(ctx, [-19.8, 0, -3], [1, 0, 0], 'part'), b = pointItem(ctx, [19.8, 0, -3], [-1, 0, 0], 'part');
const d = distanceItem(ctx, a, b);
ok(near(parseFloat(d.text), 40, 0.3) && /ΔX (39\.[89]|40\.[0-3])/.test(d.detail) && /ΔY 0\.000/.test(d.detail), `pocket width: ${d.text} (${d.detail})`);
// nothing to snap to: the point stays
const s = snapMill(ctx, [12.345, 23.456, -9.87]);
ok(near(s.p[0], 12.345, 1e-9) && near(s.p[1], 23.456, 1e-9) && near(s.p[2], -9.87, 1e-9) && !s.what.length, 'a point away from everything keeps its coordinates');

// lathe: shoulders and diameters
const lathe = `%
O2 (TURN)
(STOCK D40 L60 Z0)
(TOOL T1 TURN R0.4 A80 "CNMG")
G18 G21 G40 G99
T0101
G97 S1500 M03
G00 X42. Z2.
G00 X30.
G01 Z-20. F0.15
X40.
G00 Z5.
M30
%`;
const lo = run('lathe', lathe);
const lc = measureContext(lo, null);
it = pointItem(lc, [15.2, 0, -19.8], null, 'part');
ok(near(it.p[0], 15, 0.05) && near(it.p[2], -20, 0.15) && /shoulder/.test(it.detail) && /Ø30\.00/.test(it.text), `lathe shoulder: ${it.text} — ${it.detail}`);
const p1 = pointItem(lc, [15, 0, -5], null, 'part'), p2 = pointItem(lc, [20, 0, -40], null, 'part');
const ld = distanceItem(lc, p1, p2);
ok(/ΔZ 35\.000/.test(ld.detail) && /ΔØ (9\.99|10\.0)/.test(ld.detail), `lathe distance: ${ld.text} (${ld.detail})`);

console.log(fails ? `\n${fails} FAILED` : '\nall ok');
process.exit(fails ? 1 : 0);
