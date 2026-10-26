// 3D material engine checks. node test/voxel.mjs
import { VoxelSim, rotX, mulXform, translate, applyX, invertX, snapX } from '../public/js/sim/voxelsim.js';
import { meshChunk } from '../public/js/view/surfacenets.js';
import { normalizeTool } from '../public/js/gcode/tools.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const tools = { 1: normalizeTool({ num: 1, type: 'flat', diameter: 10, fluteLength: 30 }, 'mill'), 2: normalizeTool({ num: 2, type: 'ball', diameter: 6 }, 'mill') };
const mv = (tool) => ({ tool, kind: 'feed' });

// OP10: 100x80x20 block, top at Z0; slot 60 long at Z-5 with a Ø10 end mill
const t0 = performance.now();
const sim = VoxelSim.fromBox([-50, -40, -20], [50, 40, 0], tools, { budget: 4e6 });
console.log(`     grid ${sim.nx}×${sim.ny}×${sim.nz} cell ${sim.h.toFixed(3)} mm, init ${(performance.now() - t0).toFixed(0)} ms`);
ok(Math.abs(sim.materialVolume() - 160000) / 160000 < 0.01, `block volume ${sim.materialVolume().toFixed(0)} ≈ 160000 mm³`);
let t1 = performance.now();
sim.cutSegment([-30, 0, 5], [-30, 0, -5], mv(1));
const res = sim.cutSegment([-30, 0, -5], [30, 0, -5], mv(1));
console.log(`     slot cut ${(performance.now() - t1).toFixed(0)} ms`);
ok(sim.isMaterial(0, 0, -5.3) && !sim.isMaterial(0, 0, -4.7), 'slot floor at Z-5');
ok(!sim.isMaterial(0, 4.7, -2) && sim.isMaterial(0, 5.3, -2), 'slot wall at Y+5 (Ø10)');
const slotVol = 60 * 10 * 5 + Math.PI * 25 * 5;
ok(Math.abs(sim.removedVolume() - slotVol) / slotVol < 0.05, `slot volume ${sim.removedVolume().toFixed(0)} ≈ ${slotVol.toFixed(0)} mm³`);
ok(res.removed > 0, 'cut reports removal');

// OP20: part flipped about X (top face down), zero on the new top centre. In OP20 work coords
// the old bottom (part Z-20) is now Z0; cut a 4 mm deep slot along X at OP20 Y+20.
const flip = rotX(180);
const b = { min: [-50, -40, -20], max: [50, 40, 0] };
const rb = sim.workBounds(flip);
const X = snapX(mulXform(translate([-(rb.min[0] + rb.max[0]) / 2, -(rb.min[1] + rb.max[1]) / 2, -rb.max[2]]), flip));
const top = applyX(X, [0, 0, -20]);
ok(Math.abs(top[2]) < 1e-9, 'OP20 transform: old bottom is the new top (Z0)');
sim.cutSegment([-30, 20, 5], [-30, 20, -4], mv(1), X);
sim.cutSegment([-30, 20, -4], [30, 20, -4], mv(1), X);
// OP20 Y+20 is part Y-20 after flipping about X; depth 4 from the bottom -> part Z-16
const q = applyX(invertX(X), [0, 20, -4]);
ok(Math.abs(q[1] + 20) < 1e-9 && Math.abs(q[2] + 16) < 1e-9, `OP20 point maps to part (0, ${q[1]}, ${q[2]})`);
ok(!sim.isMaterial(0, -20, -17) && sim.isMaterial(0, -20, -15.3), 'OP20 slot cut into the bottom face (floor at part Z-16)');
ok(!sim.isMaterial(0, 0, -4.7) && sim.isMaterial(0, 0, -10), 'OP10 slot still there, middle untouched');

// ball nose groove profile
const s2 = VoxelSim.fromBox([0, 0, -10], [40, 20, 0], tools, { cell: 0.1 });
s2.cutSegment([5, 10, 2], [5, 10, -2], mv(2));
s2.cutSegment([5, 10, -2], [35, 10, -2], mv(2));
const exp = -2 + 3 - Math.sqrt(9 - 2.25);
const zAt = (x, y) => { let z = 0; while (z > -10 && !s2.isMaterial(x, y, z)) z -= 0.01; return z; };
ok(Math.abs(zAt(20, 10) + 2) < 0.06 && Math.abs(zAt(20, 11.5) - exp) < 0.06, `ball groove: centre ${zAt(20, 10).toFixed(2)}, 1.5 off ${zAt(20, 11.5).toFixed(2)} (expected ${exp.toFixed(2)})`);

// mesh: every chunk meshes, normals finite, the cut region produces triangles
t1 = performance.now();
const ct = new Float32Array(256 * 3).fill(0.7);
let tris = 0, bad = 0;
for (let c = 0; c < sim.dirty.length; c++) {
  const m = meshChunk(sim, c, ct);
  if (!m) continue;
  tris += m.index.length / 3;
  for (const v of m.normals) if (!Number.isFinite(v)) bad++;
}
console.log(`     mesh ${tris} triangles in ${(performance.now() - t1).toFixed(0)} ms`);
ok(tris > 1000 && bad === 0, 'surface nets mesh: triangles, finite normals');

// lathe part -> solid
import('../public/js/sim/lathesim.js').then(({ LatheSim }) => {
  const l = new LatheSim({ diameter: 40, length: 50, faceStock: 0, innerDiameter: 0 }, {});
  for (let i = 0; i < l.n; i++) { const z = l.zMin + i * l.dz; if (z > -20) l.outer[i] = 10; }
  const v = VoxelSim.fromLathe(l, tools, { budget: 2e6 });
  ok(v.isMaterial(9.5, 0, -10) && !v.isMaterial(10.5, 0, -10) && v.isMaterial(0, 19.5, -30) && !v.isMaterial(0, 20.5, -30), 'lathe profile → solid: Ø20 front, Ø40 back');
  const vol = Math.PI * (100 * 20 + 400 * 30 + 400 * l.grip); // incl. the length held in the chuck
  ok(Math.abs(v.materialVolume() - vol) / vol < 0.03, `turned solid volume ${v.materialVolume().toFixed(0)} ≈ ${vol.toFixed(0)}`);
  process.exitCode = fails ? 1 : 0;
});
