// Tool engagement: known cuts must measure what they are. node test/engagement.mjs
import { simulateJob } from '../public/js/sim/job.js';
import { EXAMPLES } from '../public/js/examples.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const run = (machine, text, job = false) => {
  const ops = [{ id: `e${Math.random()}`, fileName: 'e.nc', machine, text }];
  if (job) ops.push({ id: 'e2', fileName: 'e2.nc', machine, text: '%\n(STOCK FROM OP10)\nM30\n%' });
  return simulateJob(ops, ops.length - 1, { interp: (m) => ({ machine: m }) })[0];
};
// the engagement of one program line (by its text)
const line = (out, re) => {
  const lines = out.op.text.split('\n');
  return out.engagement.lines.find((l) => l.file === 'main' && re.test(lines[l.callLine]));
};

const mill = `%
O1001 (ENGAGEMENT)
(STOCK X-50:50 Y-30:30 Z-20:0)
(TOOL T1 FLAT D10 L30 "EM10")
G90 G17 G21 G40 G49 G80 G94
T1 M06
S6000 M03
G54 G00 X-60. Y0.
G43 Z5. H01
G00 Z-3.
G01 X40. F800 (SLOT)
G00 Z5.
G00 X-60. Y2.
G00 Z-3.
G01 X30. F800 (SIDE 2 MM, ENDS BEFORE THE SLOT'S ROUND END)
G00 Z5.
G00 X0. Y-20.
G01 Z-4. F200 (PLUNGE)
G00 Z5.
G01 X-60. Y25. F800
G01 X40. (AIR)
M30
%`;
for (const [out, eng, grid] of [[run('mill', mill), 'height map', 0.15], [run('mill', mill, true), '3D model', 0.45]]) {
  const slot = line(out, /SLOT/), side = line(out, /SIDE/), pl = line(out, /PLUNGE/), air = line(out, /AIR/);
  ok(slot && near(slot.ae, 10, grid) && near(slot.angle, 180, 6) && near(slot.ap, 3, grid), `${eng}: full slot — ae ${slot?.ae.toFixed(2)} (Ø10), ${slot?.angle.toFixed(0)}°, ap ${slot?.ap.toFixed(2)} (3)`);
  ok(side && near(side.ae, 2, grid) && near(side.angle, 53.1, grid === 0.15 ? 4 : 12) && near(side.ap, 3, grid), `${eng}: 2 mm stepover — ae ${side?.ae.toFixed(2)} (2), ${side?.angle.toFixed(0)}° (53), ap ${side?.ap.toFixed(2)} (3)`);
  ok(pl && pl.plunge && near(pl.ap, 4, grid + 0.3), `${eng}: plunge 4 deep — ap ${pl?.ap.toFixed(2)}`);
  ok(air && air.cut === 0 && air.len > 90, `${eng}: a feed move above the part cuts air (${air?.len.toFixed(0)} mm)`);
  const T = out.engagement.perTool[1];
  ok(T && near(T.max.ae.v, 10, grid) && T.slot > 80 && T.slot < 115, `${eng}: tool summary — max ae ${T?.max.ae.v.toFixed(2)}, full-width slotting ${T?.slot.toFixed(0)} mm`);
}

const lathe = `%
O2001 (TURN)
(STOCK D40 L60 Z0)
(TOOL T1 TURN R0.4 A80 "CNMG")
G18 G21 G40 G99
T0101
G97 S1000 M03
G00 X42. Z2.
G00 X37.
G01 Z-30. F0.2 (DEPTH 1.5)
G00 X42. Z2.
G00 X34.
G01 Z-20. F0.2 (DEPTH 1.5 AGAIN)
G01 X38. (UP THE SHOULDER)
G00 X42. Z2.
M30
%`;
{
  const out = run('lathe', lathe);
  const d1 = line(out, /DEPTH 1\.5\)/), d2 = line(out, /AGAIN/);
  ok(d1 && near(d1.ap, 1.5, 0.06) && near(d1.f, 0.2, 1e-9), `lathe: turning Ø40 → Ø37 is a 1.5 mm depth of cut (${d1?.ap.toFixed(3)}), f 0.2`);
  ok(d2 && near(d2.ap, 1.5, 0.06), `lathe: Ø37 → Ø34 again 1.5 (${d2?.ap.toFixed(3)})`);
  ok(out.engagement.perTool[1].max.q.v > 0, `lathe: removal rate ${out.engagement.perTool[1].max.q.v.toFixed(1)} cm³/min`);
}

// every example still runs, and records something for the tools that cut
for (const ex of EXAMPLES) {
  const ops = ex.ops || [{ id: ex.id, fileName: `${ex.id}.nc`, machine: ex.machine, text: ex.text }];
  const outs = simulateJob(ops.map((o, i) => ({ id: `${ex.id}${i}`, fileName: o.fileName || `${ex.id}${i}.nc`, machine: o.machine || ex.machine, text: o.text })), ops.length - 1, { interp: (m) => ({ machine: m }) });
  ok(outs.every((o) => o.engagement && Object.values(o.engagement.perTool).some((T) => T.cut > 0)), `example ${ex.id}: engagement recorded`);
}
// Claude can ask for it: the browser's check of inspect_part queries allows what the tool schema offers
{
  const { readFileSync } = await import('node:fs');
  const list = (file, re) => JSON.parse(re.exec(readFileSync(new URL(file, import.meta.url), 'utf8'))[1].replace(/'/g, '"'));
  const server = list('../server.js', /query: \{ type: 'string', enum: (\[[^\]]*\])/), client = list('../public/js/claude.js', /const QUERIES = (\[[^\]]*\])/);
  ok(server.includes('engagement') && server.length === client.length && server.every((q) => client.includes(q)), `inspect_part queries: schema [${server}] = browser check [${client}]`);
}
process.exitCode = fails ? 1 : 0;
