// Threads vs target models: the comparison must judge threads as threads, not as surfaces.
// Acceptance test for the thread-aware comparison (see TASK-threads.md). node test/threads.mjs
import { readFileSync } from 'node:fs';
import { EXAMPLES } from '../public/js/examples.js';
import { simulateJob } from '../public/js/sim/job.js';
import { partSummary } from '../public/js/sim/inspect.js';
import { trisBounds, buildTarget, compareTarget } from '../public/js/model/model.js';

let fails = 0;
const ok = (c, m, detail = '') => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c && detail) console.log(detail.split('\n').map((l) => '     | ' + l).join('\n')); if (!c) fails++; };
const model = (tris, name) => ({ name, format: 'stl', tris, triCount: tris.length / 9, bounds: trisBounds(tris) });
const PL = { up: '+Z', rotZ: 0, align: 'model', offset: [0, 0, 0], scale: 1 };
const target = (tris, name, budget = 2e6) => buildTarget(model(tris, name), PL, { budget });
const surfaceProblem = (txt) => /GOUGE|Material still to remove/.test(txt);

// exact meshes ------------------------------------------------------------------------------
// closed solid of (near) revolution about Z: radius r(theta, z), z0..z1
function turnedSolid(rAt, z0, z1, { segs = 180, dz = 0.05 } = {}) {
  const nz = Math.round((z1 - z0) / dz), out = [];
  const P = (k, j) => { const th = (2 * Math.PI * k) / segs, z = z0 + j * dz, r = rAt(th, z); return [r * Math.cos(th), r * Math.sin(th), z]; };
  for (let j = 0; j < nz; j++) for (let k = 0; k < segs; k++) {
    const a = P(k, j), b = P((k + 1) % segs, j), c = P((k + 1) % segs, j + 1), d = P(k, j + 1);
    out.push(...a, ...b, ...c, ...a, ...c, ...d);
  }
  for (const [j, flip] of [[0, true], [nz, false]]) for (let k = 0; k < segs; k++) {
    const a = P(k, j), b = P((k + 1) % segs, j), c = [0, 0, z0 + j * dz];
    out.push(...(flip ? [...c, ...b, ...a] : [...c, ...a, ...b]));
  }
  return new Float32Array(out);
}
// 40 × 40 × 10 plate (Z-10..0) with a through hole of radius r(theta, z) at X0 Y0
function plateWithHole(rAt, { N = 256, dz = 0.05, half = 20, z0 = -10, z1 = 0 } = {}) {
  const out = [];
  const tri = (a, b, c) => out.push(...a, ...b, ...c);
  const th = (k) => (2 * Math.PI * k) / N;
  const sq = (k, z) => { const c = Math.cos(th(k)), s = Math.sin(th(k)), m = Math.max(Math.abs(c), Math.abs(s)); return [(half * c) / m, (half * s) / m, z]; };
  const ci = (k, z) => { const r = rAt(th(k), z); return [r * Math.cos(th(k)), r * Math.sin(th(k)), z]; };
  for (let k = 0; k < N; k++) {
    const k2 = (k + 1) % N;
    tri(ci(k, z1), sq(k, z1), sq(k2, z1)); tri(ci(k, z1), sq(k2, z1), ci(k2, z1));
    tri(ci(k, z0), sq(k2, z0), sq(k, z0)); tri(ci(k, z0), ci(k2, z0), sq(k2, z0));
    tri(sq(k, z0), sq(k2, z0), sq(k2, z1)); tri(sq(k, z0), sq(k2, z1), sq(k, z1));
  }
  const nz = Math.round((z1 - z0) / dz);
  for (let j = 0; j < nz; j++) for (let k = 0; k < N; k++) {
    const k2 = (k + 1) % N, za = z0 + j * dz, zb = z0 + (j + 1) * dz;
    tri(ci(k, za), ci(k2, zb), ci(k2, za)); tri(ci(k, za), ci(k, zb), ci(k2, zb));
  }
  return new Float32Array(out);
}
// ISO internal thread tooth: radius between minor and major, 60° V, given pitch, starting angle phase
const helixHole = (pitch, minor, major, phase = 0) => (th, z) => {
  const u = ((((z + pitch * (th / (2 * Math.PI) + phase)) % pitch) + pitch) % pitch) / pitch;
  return minor / 2 + ((major - minor) / 2) * (1 - Math.abs(2 * u - 1));
};

// lathe: the shaft example — M20×1.5 cut with G76 from Z-1.5 to Z-19.5 ---------------------
{
  const ex = EXAMPLES.find((e) => e.id === 'shaft');
  const [shaft] = simulateJob([{ id: 'shaft', fileName: 'shaft.nc', machine: 'lathe', text: ex.text }], 0, { interp: (m) => ({ machine: m }) });
  const S = shaft.final;
  const prof = (z) => { const i = Math.round((z - S.zMin) / S.dz); return i >= 0 && i < S.n && S.outer[i] - S.inner[i] > 1e-4 ? S.outer[i] : 0; };
  const inThread = (z) => z <= -1.5 && z >= -19.5;
  const solid = (rAt) => target(turnedSolid(rAt, -84.9, 0), 'shaft', 3e5);
  // the profile of the program, with the thread replaced by a helix of the given pitch starting at `phase` of a turn
  const helix = (pitch, phase) => (th, z) => (inThread(z) ? prof(-1.5 - ((((-1.5 - z) + pitch * (th / (2 * Math.PI) + phase)) % pitch) * 1.5) / pitch) : prof(z));

  const cosmetic = compareTarget(shaft, solid((th, z) => (inThread(z) ? 9.9 : prof(z))));
  ok(!surfaceProblem(cosmetic) && /thread/i.test(cosmetic) && /1\.5/.test(cosmetic), 'lathe, cosmetic thread (plain Ø19.8 in the model): the thread is reported as a thread, no gouges', cosmetic);

  const modeled = compareTarget(shaft, solid(helix(1.5, 0.5)));
  ok(!surfaceProblem(modeled) && /thread/i.test(modeled), 'lathe, modeled helix starting half a turn later: no gouge / excess pattern', modeled);

  const wrongPitch = compareTarget(shaft, solid(helix(1.0, 0)));
  ok(/thread/i.test(wrongPitch) && /pitch/i.test(wrongPitch) && /mismatch|does not match|differs|≠/i.test(wrongPitch), 'lathe, model has pitch 1.0 but the program cuts 1.5: reported as a thread mismatch', wrongPitch);

  const bigger = compareTarget(shaft, solid((th, z) => (inThread(z) ? 9.9 : z <= -24 && z >= -44.85 ? 15.25 : prof(z))));
  ok(/GOUGE[^\n]*\n?[^\n]*Z-2[45]/.test(bigger) || /Z-2[45][^\n]*GOUGE/.test(bigger), 'lathe, a real undersize elsewhere (Ø30 cut where the model has Ø30.5) is still a gouge', bigger);
}

// mill: 40 × 40 × 10 plate, Ø6.8 drilled through, tapped M8×1.25 ------------------------------
{
  const prog = (tap) => `%
O1001 (TAPPED HOLE)
(STOCK X-20:20 Y-20:20 Z-10:0)
(TOOL T1 DRILL D6.8 A118 L40 "DRILL 6.8")
(TOOL T2 TAP D8 L20 "TAP M8X1.25")
G90 G17 G21 G40 G49 G80 G94
T1 M06
S2000 M03
G54 G00 X0. Y0.
G43 Z5. H01
G81 Z-13. R2. F200
G80 G00 Z25.
${tap ? `T2 M06
S500 M03
G00 X0. Y0.
G43 Z5. H02
G84 Z-11. R2. F625
G80 G00 Z25.
` : ''}M30
%`;
  const run = (text, multi = false) => {
    const ops = [{ id: `p${multi}${text.length}`, fileName: 'p.nc', machine: 'mill', text }];
    if (multi) ops.push({ id: 'p2', fileName: 'p2.nc', machine: 'mill', text: '%\n(STOCK FROM OP10)\nM30\n%' });
    return simulateJob(ops, ops.length - 1, { interp: (m) => ({ machine: m }) })[0];
  };
  const tapped = run(prog(true)), tapped3d = run(prog(true), true), drilled = run(prog(false));
  ok(tapped.engine === 'height' && tapped3d.engine === 'voxel', 'the tapped plate runs on the height map and (in a 2-operation job) on the 3D model');

  const s = partSummary(tapped);
  ok(/tapped/i.test(s) && /M8|Ø8/.test(s) && /1\.25/.test(s), 'measured geometry labels the hole as tapped M8×1.25', s);

  for (const [out, eng] of [[tapped, 'height map'], [tapped3d, '3D model']]) {
    const tapDrill = compareTarget(out, target(plateWithHole(() => 3.4), 'tap drill'));
    ok(!surfaceProblem(tapDrill) && /tap|thread/i.test(tapDrill), `${eng}: model hole at tap-drill Ø6.8 (cosmetic thread) — consistent with the M8 tap, no gouge`, tapDrill);
    const nominal = compareTarget(out, target(plateWithHole(() => 4), 'nominal'));
    ok(!surfaceProblem(nominal), `${eng}: model hole at Ø8 — matches, and no false excess along the plate's outside edges`, nominal);
    const helical = compareTarget(out, target(plateWithHole(helixHole(1.25, 6.647, 8, 0.3)), 'helix'));
    ok(!surfaceProblem(helical) && /thread|tap/i.test(helical), `${eng}: modeled M8×1.25 helix — consistent, no gouge`, helical);
  }
  const wrongSize = compareTarget(tapped, target(plateWithHole(() => 2.5), 'M6 tap drill'));
  ok(/GOUGE|mismatch|does not match|differs/i.test(wrongSize), 'model has a Ø5 hole (M6 tap drill) but the program taps M8: flagged', wrongSize);
  const plain = compareTarget(drilled, target(plateWithHole(() => 3), 'Ø6'));
  ok(/GOUGE/.test(plain), 'no tap, Ø6.8 drilled where the model has Ø6: still a gouge', plain);
}

// Claude is told how threads appear in models
ok(/cosmetic/i.test(readFileSync(new URL('../prompts/system.md', import.meta.url), 'utf8')), 'system prompt explains cosmetic threads in models');
process.exitCode = fails ? 1 : 0;
