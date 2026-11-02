// Modal lathe cycles: a pass that names only X repeats the Z of the pass before. node test/lathe-modal.mjs
import { interpret } from '../public/js/gcode/interpreter.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const run = (body) => interpret(`%\nO1 (MODAL)\n(STOCK D40 L60 Z0)\n(TOOL T4 THREAD "THREAD 60")\n(TOOL T1 TURN R0.4 A80 "CNMG")\nG18 G21 G40 G99\n${body}\nM30\n%`, { machine: 'lathe' });

// G92: three passes, the second and third name only X
let r = run('T0404\nG97 S800 M03\nG00 X42. Z5.\nG92 X19.4 Z-20. F1.5\nX19.\nX18.7\nG00 X50. Z20.');
const threads = r.moves.filter((m) => m.kind === 'feed' && m.thread > 0);
ok(threads.length === 3, `G92 with two X-only passes cuts 3 thread passes (${threads.length})`);
ok(threads.every((m) => Math.abs(m.pts[m.pts.length - 1][2] + 20) < 1e-6), `every pass runs to Z-20 (${threads.map((m) => m.pts[m.pts.length - 1][2].toFixed(2)).join(', ')})`);
ok(Math.abs(threads[2].pts[threads[2].pts.length - 1][0] * 2 - 18.7) < 1e-6, 'the last pass is at X18.7');
// G90: box turning, X-only passes
r = run('T0101\nG97 S1500 M03\nG00 X42. Z2.\nG90 X36. Z-30. F0.2\nX32.\nX28.\nG00 X50. Z20.');
const cuts = r.moves.filter((m) => m.kind === 'feed' && Math.abs(m.pts[0][2] - m.pts[m.pts.length - 1][2]) > 20);
ok(cuts.length === 3 && cuts.every((m) => Math.abs(m.pts[m.pts.length - 1][2] + 30) < 1e-6), `G90 with X-only passes cuts 3 boxes to Z-30 (${cuts.length})`);
// G00 ends the cycle: an axis-only block after it is a rapid, not another pass
r = run('T0101\nG97 S1500 M03\nG00 X42. Z2.\nG90 X36. Z-30. F0.2\nG00 X50. Z20.\nX30.\nM05');
const last = r.moves.filter((m) => m.kind === 'rapid' || m.kind === 'feed').pop();
const boxFeeds = r.moves.filter((m) => m.kind === 'feed').length;
ok(last.kind === 'rapid' && Math.abs(last.pts[last.pts.length - 1][0] * 2 - 30) < 1e-6 && boxFeeds === 2, `G00 cancels the cycle: the X30. block is a rapid (${last.kind}), ${boxFeeds} feed moves from the one box`);
console.log(fails ? `\n${fails} FAILED` : '\nall ok');
process.exit(fails ? 1 : 0);
