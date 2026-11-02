// Craft checks: known bad habits are flagged, good practice is not. node test/craft.mjs
import { simulateJob } from '../public/js/sim/job.js';
import { EXAMPLES } from '../public/js/examples.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const run = (machine, text, extra = {}) => simulateJob([{ id: `c${Math.random()}`, fileName: 'c.nc', machine, text }], 0, { interp: (m) => ({ machine: m }), features: { craft: true }, ...extra })[0];
const craft = (out) => out.issues.filter((i) => i.source === 'craft');
const has = (out, re) => craft(out).some((i) => re.test(i.text));

const head = (stock, tool) => `%\nO1 (CRAFT)\n${stock}\n${tool}\nG90 G17 G21 G40 G49 G80 G94\nT1 M06\n`;
const tail = 'G00 Z25.\nM30\n%';

// a Ø50 face mill over a 40 mm wide part: centred is flagged, offset is fine
const face = (y) => head('(STOCK X-50:50 Y-20:20 Z-5:0)', '(TOOL T1 FACE D50 L6 Z5 "FACE MILL")') + `S1900 M03\nG00 X-85. Y${y}. Z5.\nG01 Z-1. F500\nX85. F900\n` + tail;
const centred = run('mill', face(0)), offset = run('mill', face(12));
ok(has(centred, /face mill runs centred over the 40 mm width/), `centred face pass flagged: ${craft(centred)[0]?.text.slice(0, 80)}`);
ok(/offset the path about 13 mm/.test(craft(centred)[0]?.text || ''), 'with the offset hint');
ok(!has(offset, /centred/), `offset by 12 mm passes (${craft(offset).length} craft issues)`);

// chip thickness: fz 0.011 at 10 % ae rubs, F1200 does not
const side = (F) => head('(STOCK X-50:50 Y-30:30 Z-20:0)', '(TOOL T1 FLAT D10 L30 Z3 "EM10")') + `S4000 M03\nG00 X-60. Y-34. Z5.\nG00 Z-5.\nG01 X60. F${F}\n` + tail;
const slow = run('mill', side(200)), fast = run('mill', side(1200));
const rub = craft(slow).find((i) => /rubbing/.test(i.text));
ok(rub && rub.level === 'warning' && /chip thickness 0\.00[4-6] mm \(fz 0\.017 mm, z3\)/.test(rub.text) && /feed about F\d+/.test(rub.text), `thin chip flagged: ${rub?.text.slice(0, 90)}`);
ok(!has(fast, /rubbing/), `F1200 at 10 % ae is fine (${craft(fast).map((i) => i.text.slice(0, 40)).join('; ') || 'no craft issues'})`);
ok(rub && rub.callLine === side(200).split('\n').findIndex((l) => l.startsWith('G01 X60.')), 'the finding points at the cutting line');

// a face mill plunged
const plunge = run('mill', head('(STOCK X-50:50 Y-30:30 Z-20:0)', '(TOOL T1 FACE D50 L6 "FACE MILL")') + 'S1900 M03\nG00 X0. Y0. Z5.\nG01 Z-1. F200\nX40. F900\n' + tail);
ok(craft(plunge).some((i) => i.level === 'error' && /face mill plunges/.test(i.text)), `face mill plunge is an error: ${craft(plunge).find((i) => /plunge/.test(i.text))?.text.slice(0, 70)}`);

// a ball mill finishing a flat top with its tip
const ball = run('mill', head('(STOCK X-50:50 Y-30:30 Z-20:0)', '(TOOL T1 BALL D6 L20 "BALL 6")') + 'S8000 M03\nG00 X-60. Y0. Z5.\nG00 Z-0.2\nG01 X60. F1500\n' + tail);
ok(has(ball, /ball mill cuts a flat area with its tip/), `ball tip on a floor flagged: ${craft(ball)[0]?.text.slice(0, 70)}`);

// lathe: a 0.15 mm pass with an R0.8 insert, and a 0.03 mm/rev feed
const turn = (x, f) => `%\nO2 (TURN)\n(STOCK D40 L60 Z0)\n(TOOL T1 TURN R0.8 A80 "CNMG")\nG18 G21 G40 G99\nT0101\nG97 S1500 M03\nG00 X42. Z2.\nG00 X${x}\nG01 Z-40. F${f}\nG00 X45.\nZ5.\nM30\n%`;
const thin = run('lathe', turn(39.8, 0.12)), slowF = run('lathe', turn(36, 0.03)), good = run('lathe', turn(36, 0.15)), fin = run('lathe', turn(39.4, 0.12));
ok(craft(thin).some((i) => i.level === 'info' && /depth of cut 0\.10 mm/.test(i.text) && /quarter of the nose radius/.test(i.text)), `lathe 0.10 mm pass with R0.8 noted: ${craft(thin)[0]?.text.slice(0, 70)}`);
ok(!craft(fin).length, 'a 0.3 mm finishing pass with R0.8 passes');
ok(craft(slowF).some((i) => i.level === 'warning' && /feed 0\.030 mm\/rev/.test(i.text)), `lathe 0.03 mm/rev flagged: ${craft(slowF).find((i) => /feed/.test(i.text))?.text.slice(0, 70)}`);
ok(!craft(good).length, 'lathe 2 mm at 0.15 mm/rev passes');

// the built-in examples with the checks on: no errors, and the warnings are listed for review
for (const ex of EXAMPLES) {
  const ops = (ex.ops || [{ fileName: `${ex.id}.nc`, machine: ex.machine, text: ex.text }]).map((o, i) => ({ id: `${ex.id}${i}`, ...o }));
  const outs = simulateJob(ops, ops.length - 1, { interp: (m) => ({ machine: m }), features: { craft: true } });
  const found = outs.flatMap((o, i) => craft(o).map((c) => `${ops.length > 1 ? `OP${(i + 1) * 10} ` : ''}${c.level}: ${c.text}`));
  ok(!found.some((f) => f.startsWith('error')), `example "${ex.name}": ${found.length ? found.map((f) => f.slice(0, 110)).join('\n      ') : 'no craft findings'}`);
}

console.log(fails ? `\n${fails} FAILED` : '\nall ok');
process.exit(fails ? 1 : 0);
