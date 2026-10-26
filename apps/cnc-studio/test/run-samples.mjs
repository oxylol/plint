// Runs the interpreter over real NC files and prints a summary per file.
// usage: node test/run-samples.mjs [-v] <file.nc>[:lathe|:mill] ...
// The machine defaults to lathe for paths containing "Turn", otherwise mill.
import { readFileSync, readdirSync } from 'node:fs';
import { basename, join, dirname } from 'node:path';
import { interpret } from '../public/js/gcode/interpreter.js';
import { describeStock } from '../public/js/gcode/stock.js';
import { describeTool } from '../public/js/gcode/tools.js';

const args = process.argv.slice(2);
const verbose = args.includes('-v');
const files = args.filter((a) => a !== '-v').map((a) => {
  const m = /^(.*):(lathe|mill)$/.exec(a);
  const path = m ? m[1] : a;
  return { path, machine: m ? m[2] : /turn/i.test(path) ? 'lathe' : 'mill' };
});

function loadSubs(path) {
  // sibling SubProgram folder (CncSimulator layout) or the file's own folder
  const subs = [];
  for (const dir of [join(dirname(path), '..', '..', 'SubProgram'), join(dirname(path), '..', '..', '..', 'SubProgram'), dirname(path)]) {
    try {
      for (const f of readdirSync(dir)) if (/^\d+\.nc$/i.test(f)) subs.push({ name: f, text: readFileSync(join(dir, f), 'latin1') });
    } catch { /* missing */ }
  }
  return subs;
}

let failures = 0;
for (const { path, machine } of files) {
  const text = readFileSync(path, 'latin1');
  const t0 = performance.now();
  let res;
  try {
    res = interpret(text, { machine, subprograms: loadSubs(path) });
  } catch (e) {
    failures++;
    console.log(`\n=== ${basename(path)} [${machine}] CRASH: ${e.stack}`);
    continue;
  }
  const ms = (performance.now() - t0).toFixed(1);
  const errs = res.messages.filter((m) => m.level === 'error');
  const warns = res.messages.filter((m) => m.level === 'warning');
  const mins = Math.floor(res.totalTime / 60), secs = Math.round(res.totalTime % 60);
  console.log(`\n=== ${basename(path)} [${machine}]  ${res.moves.length} moves, ${res.stats.blocks} blocks, cycle ${mins}m${secs}s, ${ms} ms`);
  console.log(`    stock: ${describeStock(res.stock)} (${res.stock.source})`);
  console.log(`    tools: ${res.usedTools.map((n) => describeTool(res.tools[n], machine)).join(' | ')}`);
  console.log(`    feed ${res.stats.feedLength.toFixed(0)} mm, rapid ${res.stats.rapidLength.toFixed(0)} mm, Z ${res.stats.minZ.toFixed(2)}..${res.stats.maxZ.toFixed(2)}`);
  for (const m of [...errs, ...warns].slice(0, verbose ? 100 : 8)) console.log(`    ${m.level.toUpperCase()} L${m.line === null ? '-' : m.line + 1}${m.file !== 'main' ? ' ' + m.file : ''}: ${m.text}`);
  if (errs.length + warns.length > 8 && !verbose) console.log(`    … ${errs.length} errors, ${warns.length} warnings total`);
  // sanity: no NaN in points
  let nan = 0;
  for (const mv of res.moves) for (const p of mv.pts) if (p.some((v) => !Number.isFinite(v))) nan++;
  if (nan) { failures++; console.log(`    !! ${nan} non-finite points`); }
}
process.exitCode = failures ? 1 : 0;
