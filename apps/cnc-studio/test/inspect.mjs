// Data inspection of simulated parts (what Claude gets instead of screenshots). node test/inspect.mjs
import { EXAMPLES } from '../public/js/examples.js';
import { simulateJob } from '../public/js/sim/job.js';
import { partSummary, section, probe, viewDetail } from '../public/js/sim/inspect.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const job = (id) => {
  const ex = EXAMPLES.find((e) => e.id === id);
  const ops = ex.ops ? ex.ops.map((o, i) => ({ id: id + i, ...o })) : [{ id, fileName: id + '.nc', machine: ex.machine, text: ex.text }];
  return simulateJob(ops, ops.length - 1, { interp: (m) => ({ machine: m }) });
};
const num = (re, txt) => { const m = re.exec(txt); return m ? Number(m[1]) : NaN; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// two-sided plate on the 3D engine
{
  const [op10, op20] = job('plate2sides');
  const t0 = performance.now();
  const s10 = partSummary(op10);
  console.log(`     OP10 summary ${(performance.now() - t0).toFixed(0)} ms`);
  const pocket = /pocket: rectangular ([\d.]+) \(X\) × ([\d.]+) \(Y\), corner R([\d.]+), centre X(-?[\d.]+) Y(-?[\d.]+); floor Z(-?[\d.]+)/.exec(s10);
  ok(pocket && near(+pocket[1], 50, 0.05) && near(+pocket[2], 30, 0.05) && near(+pocket[3], 5, 0.1) && near(+pocket[6], -6, 0.02), 'OP10: pocket 50 × 30 R5, floor Z-6');
  const holes = /4× through hole Ø([\d.]+); through from Z(-?[\d.]+) to Z(-?[\d.]+); centres X,Y: (.*)/.exec(s10);
  ok(holes && near(+holes[1], 6.8, 0.03), `OP10: 4× Ø6.8 through (${holes?.[1]})`);
  ok(holes && near(+holes[2], 0, 0.02) && near(+holes[3], -25, 0.02), 'OP10: holes go from the top face to the bottom');
  const centres = holes ? [...holes[4].matchAll(/\((-?[\d.]+), (-?[\d.]+)\)/g)].map((m) => [+m[1], +m[2]]) : [];
  ok(centres.length === 4 && centres.every(([x, y]) => near(Math.abs(x), 40, 0.02) && near(Math.abs(y), 20, 0.02)), 'OP10: hole centres at X±40 Y±20');
  ok(/Outline seen from above[^:]*: 100 × 60, square corners/.test(s10), 'OP10: outline 100 × 60, square corners (voxel edge rounding ignored)');
  const s20 = partSummary(op20);
  const slot = /open slot\/channel along X [^,]+, ([\d.]+) wide \(centre Y(-?[\d.]+)\): floor Z(-?[\d.]+), ([\d.]+) below Z(-?[\d.]+)/.exec(s20);
  ok(slot && near(+slot[1], 12, 0.03) && near(+slot[2], 0, 0.02) && near(+slot[3], -5, 0.02) && near(+slot[4], 4, 0.02), 'OP20: open slot 12 wide, 4 deep at Y0');
  ok(/Seen from below[\s\S]*pocket: rectangular 50 \(X\) × 30 \(Y\), corner R5(\.0\d)?, [^;]*; floor Z-19, 6 deep below Z-25/.test(s20), 'OP20: the OP10 pocket is seen from below after the flip');
  ok(!/Seen from the (front|back|left|right)/.test(s20), 'no noise from the plain side faces');
  const sec = section(op20, 'y', 0);
  ok(/1 loop[\s\S]*material, area 1699\.\d/.test(sec) || /1 loop[\s\S]*material, area 1700/.test(sec), 'section Y0: one material loop of the expected area (100×20 − 50×6 pocket)');
  const pr = probe(op20, [[0, 0, -4], [0, 0, -6], [40, 20, -10], [30, 25, -10]]);
  ok(/Z-4: air/.test(pr) && /Z-6: material/.test(pr) && /X40 Y20 Z-10: air/.test(pr) && /X30 Y25 Z-10: material/.test(pr), 'probes: slot air, floor material, hole air, web material');
}

// height-map engine: the original plate and bolt circle
{
  const [p] = job('plate');
  const s = partSummary(p);
  const isl = /island\/boss: rectangular ([\d.]+) \(X\) × ([\d.]+) \(Y\), corner R([\d.]+)[^;]*; top Z-0.5, standing ([\d.]+) above the Z(-?[\d.]+) floor/.exec(s);
  ok(isl && near(+isl[1], 90, 0.4) && near(+isl[2], 70, 0.4) && near(+isl[3], 5, 0.6) && near(+isl[5], -10, 0.01), `plate: 90 × 70 R5 island on the Z-10 floor (${isl?.slice(1).join(' ')})`);
  ok(/pocket: rectangular 40\.\d+ \(X\) × 30\.\d+ \(Y\), corner R4\.\d+[^;]*; floor Z-6/.test(s), 'plate: 40 × 30 pocket, floor Z-6');
  ok(/4× through hole Ø[\d.]+; through from Z-0.5 to Z-20; along the depth: Ø7\.\d+ Z-0\.\d+\.\.-1[78]/.test(s), 'plate: tapped holes show their Ø8 top and the drill below');
  const [b] = job('boltcircle');
  const sb = partSummary(b);
  ok(/round pocket\/blind hole: round Ø(29\.9\d+|30\.0\d+) at X0(\.0\d+)? Y0; floor Z-5/.test(sb), 'bolt circle: Ø30 pocket 5 deep');
  const bc = /8× through hole Ø([\d.]+); through from Z0 to Z-15; centres X,Y: (.*)/.exec(sb);
  const pts = bc ? [...bc[2].matchAll(/\((-?[\d.]+), (-?[\d.]+)\)/g)].map((m) => Math.hypot(+m[1], +m[2])) : [];
  ok(bc && near(+bc[1], 8.5, 0.3) && pts.length === 8 && pts.every((r) => near(r, 35, 0.05)), 'bolt circle: 8 holes on R35');
  ok(/circle Ø(29\.9\d+|30\.0\d+) centred X0(\.00\d)? Y0/.test(section(b, 'z', -2.5)), 'section Z-2.5 reports the pocket as a circle');
}

// lathe profiles
{
  const [s] = job('shaft');
  const t = partSummary(s);
  const th = /thread, pitch ([\d.]+), major Ø([\d.]+), minor Ø([\d.]+)/.exec(t);
  ok(th && near(+th[1], 1.5, 0.01) && near(+th[2], 19.8, 0.01), `shaft: M20×1.5 thread recognised (${th?.slice(1).join(' ')})`);
  ok(/Z-20(\.05)? → Z-22\.95: Ø17 /.test(t), 'shaft: Ø17 relief groove');
  ok(/Z-24 → Z-44\.85: Ø29\.999/.test(t), 'shaft: Ø30 section');
  const [, b] = job('shaft2ends');
  const t2 = partSummary(b);
  ok(/Z-1: front face \(Ø0 → Ø20\.079\)/.test(t2) && /Z-1 → Z-3: chamfer Ø20\.079 → Ø23\.999/.test(t2), 'flipped shaft: faced front, 2 mm chamfer');
  ok(/R0\.4 fillet Ø24 → Ø24\.419/.test(t2), 'flipped shaft: R0.4 nose-radius fillet at the shoulder');
  ok(/Z-60\.4 → Z-83: Ø30/.test(t2), 'flipped shaft: OP10 Ø30 end now in the chuck');
}

// rotary: each face seen from its side
{
  const [r] = job('rotary4');
  const s = partSummary(r);
  ok((s.match(/open slot\/channel along X[^,]*, 10(\.0\d)? wide/g) || []).length === 4, 'rotary: a 10 wide slot on all four faces');
  const v = viewDetail(r, 'front');
  ok(/Depth map/.test(v) && /#/.test(v), 'view detail includes a depth map');
}
process.exitCode = fails ? 1 : 0;
