// Every example (single programs and jobs) must simulate without errors or warnings. node test/examples.mjs
import { EXAMPLES } from '../public/js/examples.js';
import { simulateJob } from '../public/js/sim/job.js';
let bad = 0;
for (const ex of EXAMPLES) {
  const ops = ex.ops || [{ fileName: `${ex.id}.nc`, machine: ex.machine, text: ex.text }];
  const outs = simulateJob(ops.map((o, i) => ({ id: `${ex.id}${i}`, ...o })), ops.length - 1, { interp: (m) => ({ machine: m }) });
  console.log(`\n${ex.id}${ex.ops ? ` (job, ${outs.length} operations)` : ''}`);
  for (const o of outs) {
    const r = o.result;
    console.log(`  ${o.name} (${o.engine}${o.src ? `, from ${o.src}` : ''}): ${r.moves.length} moves, ${Math.floor(r.totalTime / 60)}m${Math.round(r.totalTime % 60)}s, removed ${(o.final.removedVolume() / 1000).toFixed(1)} cm³, stock ${r.stock.source}`);
    for (const m of o.issues) console.log(`     ${m.level.toUpperCase()} L${m.callLine === null || m.callLine === undefined ? '-' : m.callLine + 1}: ${m.text}`);
    if (o.issues.some((m) => m.level === 'error' || m.level === 'warning')) bad++;
    const s = o.final;
    if (o.engine === 'lathe') {
      const at = (z) => { const i = Math.round((z - s.zMin) / s.dz); return i >= 0 && i < s.n ? (s.outer[i] * 2).toFixed(3) : '–'; };
      console.log(`     diameters: Z-10 Ø${at(-10)}  Z-21.5 Ø${at(-21.5)}  Z-30 Ø${at(-30)}  Z-47.5 Ø${at(-47.5)}  Z-55 Ø${at(-55)}`);
    } else if (o.engine === 'height') {
      const h = (x, y) => { const i = Math.round((x - s.minX) / s.cx), j = Math.round((y - s.minY) / s.cy); return i >= 0 && j >= 0 && i < s.nx && j < s.ny ? s.heights[j * s.nx + i].toFixed(3) : '–'; };
      console.log(`     heights: centre ${h(0, 0)}  (30,0) ${h(30, 0)}  (47,0) ${h(47, 0)}  (-35,-25) ${h(-35, -25)}`);
    }
  }
}
process.exitCode = bad ? 1 : 0;
