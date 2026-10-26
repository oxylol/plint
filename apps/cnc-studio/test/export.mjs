// Export: one standard program, same toolpath. node test/export.mjs
import { simulateJob, opNames } from '../public/js/sim/job.js';
import { EXAMPLES } from '../public/js/examples.js';
import { exportJob } from '../public/js/gcode/export.js';

let fails = 0;
const ok = (c, m, d = '') => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c && d) console.log(d.split('\n').map((l) => '     | ' + l).join('\n')); if (!c) fails++; };
const io = (m, subs = []) => ({ machine: m, subprograms: subs });
const job = (ops, subs = [], opts = {}) => {
  const list = ops.map((o, i) => ({ id: `x${i}${Math.random()}`, fileName: o.fileName || `p${i}.nc`, machine: o.machine || 'mill', text: o.text }));
  const outs = simulateJob(list, list.length - 1, { interp: (m) => io(m, subs) });
  const names = opNames(list);
  return exportJob({ ops: list.map((o, i) => ({ ...o, name: names[i], out: outs[i] })), subprograms: subs, interpOptions: (m) => io(m, subs), ...opts });
};
const allOk = (r) => r.verify.every((v) => v.ok);

// every example: verified, standard text only
for (const ex of EXAMPLES) {
  const r = job(ex.ops || [{ machine: ex.machine, text: ex.text }]);
  const lines = r.text.split('\r\n');
  ok(allOk(r), `${ex.id}: exported program gives the same toolpath (${r.verify.map((v) => v.text.split(' — ')[1]).join(', ')})`, r.verify.map((v) => v.text).join('\n'));
  ok(lines[0] === '%' && /^O\d{4}/.test(lines[1]) && lines.at(-2) === '%' && lines.at(-1) === '' && !/[^\x20-\x7e\r\n]/.test(r.text) && r.text === r.text.toUpperCase(), `${ex.id}: % … %, O number, upper-case ASCII, CRLF`);
  ok(!/\((STOCK|TOOL T|SETUP|ORIENT|ZERO (TOP|SAME|FACE|PART)|VISE|CHUCK|ROTARY [AB]|\$)/.test(r.text), `${ex.id}: no CNC Studio directives left`);
  ok((r.text.match(/\bM30\b/g) || []).length === 1, `${ex.id}: exactly one M30`);
}

// two setups in one program: stop and re-clamp note between them
{
  const ex = EXAMPLES.find((e) => e.id === 'plate2sides');
  const r = job(ex.ops);
  const i = r.text.indexOf('M00');
  ok(i > 0 && /\(OP20 - TURN OVER ABOUT X[^)]*\)\r\nM00/.test(r.text) && r.text.lastIndexOf('M30') > i, 'two operations: M00 with "OP20 - TURN OVER ABOUT X …" between them, M30 at the end', r.text.slice(Math.max(0, i - 200), i + 20));
}

// subprograms: built in when plain, appended when they jump
const main = (call) => `%
O1001 (SUBS ø TEST)
(STOCK X-50:50 Y-50:50 Z-10:0)
(TOOL T1 FLAT D6 L20 "EM6")
G90 G17 G21 G40 G49 G80 G94
T1 M06
S5000 M03
G54 G00 X0. Y0.
G43 Z5. H01
${call}
G00 Z25.
M30
%`;
const plainSub = { name: 'O1000.nc', text: '%\nO1000 (POCKET STEP)\nG91 G01 Z-1. F200\nG90 G01 X10. F600\nX0.\nM99\n%' };
const jumpSub = { name: '2000.nc', text: '%\nO2000\n#1=0\nN10 G91 G01 Z-1. F200\nG90 X10.\nX0.\n#1=#1+1\nIF [#1 LT 2] GOTO10\nM99\n%' };
{
  const r = job([{ text: main('M98 P1000 L3') }], [plainSub]);
  ok(allOk(r) && !/M98/.test(r.text) && (r.text.match(/G91 G01 Z-1\. F200/g) || []).length === 3 && !/M99/.test(r.text), 'M98 P1000 L3 is built in three times (no M98/M99 left), same toolpath', r.text);
  ok(/\(SUBS DIA TEST\)/.test(r.text), 'comments become plain ASCII (ø → DIA)');
  const r0 = job([{ text: main('M98 P31000') }], [plainSub]);
  ok(allOk(r0) && (r0.text.match(/G91 G01 Z-1\. F200/g) || []).length === 3, 'Fanuc 0 form M98 P31000 = three times O1000');
  const r2 = job([{ text: main('M98 P2000') }], [jumpSub]);
  ok(allOk(r2) && /M98 P2000/.test(r2.text) && /M30\r\nO2000\r\n[\s\S]*M99\r\n%/.test(r2.text) && /appended/.test(r2.notes.join(' ')), 'a subprogram with GOTO is appended after M30 as O2000, M98 kept, same toolpath', r2.text + '\n' + r2.notes.join('\n'));
  const r3 = job([{ text: main('M98 P1000 L2') }], [plainSub], { inlineSubs: false });
  ok(allOk(r3) && /M98 P1000 L2/.test(r3.text) && /\r\nO1000\r\n/.test(r3.text), 'subprograms can be appended by choice');
  const r4 = job([{ text: main('M98 P1000') }], [plainSub], { keepComments: false, toolList: false });
  ok(!/[()]/.test(r4.text.split('\r\n').slice(2).join('\n')), 'comments can be left out entirely');
}

// a check that really checks: an export that would move differently is flagged
{
  const ex = EXAMPLES.find((e) => e.id === 'plate');
  const list = [{ id: 'a', fileName: 'a.nc', machine: 'mill', text: ex.text }];
  const [out] = simulateJob(list, 0, { interp: (m) => io(m) });
  const other = ex.text.replace('X80.', 'X79.');
  const r = exportJob({ ops: [{ ...list[0], text: other, name: 'OP10', out }], subprograms: [], interpOptions: (m) => io(m) });
  ok(!r.verify[0].ok && /differs/.test(r.verify[0].text), `a different toolpath is reported: ${r.verify[0].text}`);
}

// lathe radius programming is said out loud
{
  const r = job([{ machine: 'lathe', text: '%\nO2002\n($SetRadiusPrg(1))\n(STOCK D40 L50)\n(TOOL T1 TURN R0.4 A80 "CNMG")\nG18 G21 G99\nT0101\nG97 S1000 M03\nG00 X21. Z2.\nG01 Z-20. F0.2\nG00 X25.\nM30\n%' }]);
  ok(allOk(r) && /RADIUS/.test(r.notes.join(' ')) && !/SETRADIUSPRG/i.test(r.text), 'radius programming: directive removed, a note says to set the control to radius mode, same toolpath', r.notes.join('\n') + '\n' + r.verify.map((v) => v.text).join('\n'));
}
process.exitCode = fails ? 1 : 0;
