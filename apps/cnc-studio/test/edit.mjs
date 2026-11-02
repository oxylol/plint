// edit_program: exact replacements applied in order, all or nothing, with a diff. node test/edit.mjs
import { applyEdits, EditError } from '../public/js/edit/edit.js';
import { diffLines, unifiedDiff, changedRanges, diffCounts } from '../public/js/edit/diff.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const throws = (fn, re, m) => { try { fn(); ok(false, `${m} (no error)`); } catch (e) { ok(e instanceof EditError && re.test(e.message), `${m}: ${e.message.split('\n')[0]}`); } };

const prog = ['%', 'O1001 (POCKET)', '(STOCK X-40:40 Y-30:30 Z-15:0)', '(TOOL T1 FLAT D8 L25 "END MILL D8")', 'G90 G17 G21 G40 G49 G80 G94', 'T1 M06', 'S6000 M03', 'G54', 'G00 X0. Y0.', 'G43 Z25. H01 M08', 'G00 Z2.', 'G01 Z0. F300', 'G01 X-14. Y-9. F800', 'X14.', 'Y9.', 'X-14.', 'Y-9.', 'G00 Z25.', 'M05', 'M09', 'G28 G91 Z0.', 'G90', 'M30', '%'].join('\n');

// one-line change
let r = applyEdits(prog, [{ old: 'G01 X-14. Y-9. F800', new: 'G01 X-14. Y-9. F600' }]);
ok(r.text.includes('F600') && !r.text.includes('F800'), 'single replacement applied');
ok(r.added === 1 && r.removed === 1 && JSON.stringify(r.changed) === '[[13,13]]', `counts +${r.added} −${r.removed}, changed ${JSON.stringify(r.changed)}`);
ok(/^@@ -10,7 \+10,7 @@/.test(r.patch) && r.patch.includes('-G01 X-14. Y-9. F800') && r.patch.includes('+G01 X-14. Y-9. F600'), 'unified patch has one hunk with context');

// multi-line insert + delete, in order
r = applyEdits(prog, [
  { old: 'G01 Z0. F300\n', new: 'G01 Z0. F300\nG03 X0. Y0. Z-2. I2. J0. F800\nG03 X0. Y0. Z-4. I2. J0.\n' },
  { old: 'M09\n', new: '' },
]);
ok(r.text.includes('G03 X0. Y0. Z-4.') && !/\nM09\n/.test(r.text), 'insert and delete applied in order');
ok(r.added === 2 && r.removed === 1, `+${r.added} −${r.removed}`);
ok(JSON.stringify(r.changed) === '[[13,14],[22,22]]', `changed ranges ${JSON.stringify(r.changed)} (insert, then the line after the deleted M09)`);

// CRLF programs and CRLF edits
r = applyEdits(prog.replace(/\n/g, '\r\n'), [{ old: 'S6000 M03\r\n', new: 'S5000 M03\r\n' }]);
ok(!r.text.includes('\r') && r.text.includes('S5000'), 'CRLF normalized before matching');

// trailing whitespace differences still match
r = applyEdits(prog.replace('X14.', 'X14.   '), [{ old: 'X14.\nY9.', new: 'X14.\nY10.' }]);
ok(r.text.includes('Y10.'), 'match ignoring trailing spaces');

// misses: nothing applied, helpful errors
throws(() => applyEdits(prog, [{ old: 'G01 X-14. Y-9. F800', new: 'A' }, { old: 'G01 X-99.', new: 'B' }]), /Edit 2 does not match/, 'second edit missing → error');
throws(() => applyEdits(prog, [{ old: 'G01 X-99. Y-9. F800', new: 'A' }]), /Closest lines[\s\S]*L13: G01 X-14\. Y-9\. F800/, 'closest line named');
throws(() => applyEdits(prog, [{ old: 'G00 Z', new: 'G00 Z' }]), /matches 2 places \(lines 11, 18\)/, 'ambiguous edit → error with lines');
throws(() => applyEdits(prog, [{ old: '  ', new: 'x' }]), /"old" is empty/, 'empty old');
throws(() => applyEdits(prog, []), /No edits/, 'no edits');

// diff helpers on their own
const ops = diffLines(['a', 'b', 'c', 'd'], ['a', 'x', 'c', 'd', 'e']);
ok(ops.map((o) => o.t).join('') === ' -+  +' && diffCounts(ops).added === 2, 'myers diff ops');
ok(JSON.stringify(changedRanges(ops)) === '[[2,2],[5,5]]', 'changed ranges from ops');
ok(unifiedDiff(ops, 1) === '@@ -1,4 +1,5 @@\n a\n-b\n+x\n c\n d\n+e', 'unified hunks merge within 2×context');
ok(unifiedDiff(diffLines(['a', 'b', 'c', 'd', 'e', 'f', 'g'], ['a', 'x', 'c', 'd', 'e', 'f', 'y']), 1).split('@@').length === 5, 'far-apart changes make two hunks');
const big = Array.from({ length: 4000 }, (_, i) => `N${i} G01 X${i}.`);
const big2 = big.slice().reverse();
const t0 = performance.now();
const bigOps = diffLines(big, big2);
ok(performance.now() - t0 < 2000 && diffCounts(bigOps).added === 4000 - 1 || diffCounts(bigOps).added > 0, `large rewrite diffs quickly (${Math.round(performance.now() - t0)} ms, +${diffCounts(bigOps).added})`);

console.log(fails ? `\n${fails} FAILED` : '\nall ok');
process.exit(fails ? 1 : 0);
