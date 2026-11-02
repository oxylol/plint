// The brief: requirements vs Claude's choices per operation, tied to program lines. node test/brief.mjs
import { parseLines, makeRanges, relocateRanges, normalizeBrief, briefText, addItem, setFixed, removeItem, sanitizeBrief, describeLines } from '../public/js/brief/brief.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };

ok(JSON.stringify(parseLines('6, 21-25')) === '[[6,6],[21,25]]' && JSON.stringify(parseLines([[3, 4], 9])) === '[[3,4],[9,9]]' && JSON.stringify(parseLines('L12–L14; x')) === '[[12,14]]', 'line specs parse');

const prog = ['%', 'O1 (PLATE)', '(STOCK X-50:50 Y-40:40 Z-20:0)', '(TOOL T8 FACE D50 L6)', '(MATERIAL S355)', 'T8 M06', 'S1200 M03', 'G00 X-80. Y12. Z2.', 'G01 Z-1. F300', 'X80. F600', 'G00 Z25.', 'M30', '%'].join('\n');
const b1 = normalizeBrief({ items: [
  { text: 'Material: S355 steel', by: 'user', lines: '5' },
  { text: 'Face with T8 Ø50 in one pass, offset 12 mm from the centreline', by: 'claude', lines: '6-10' },
  { text: 'Top face to Z-1', by: 'drawing' },
  { text: '  ', by: 'claude' },
  { text: 'Bad source', by: 'nobody', fixed: true },
] }, prog);
ok(b1.items.length === 4 && b1.items[0].fixed && !b1.items[1].fixed && b1.items[2].fixed && b1.items[3].by === 'claude' && b1.items[3].fixed, 'defaults: user/drawing fixed, claude free, unknown source → claude, blank dropped');
ok(b1.items[0].ranges[0].first === '(MATERIAL S355)' && b1.items[1].ranges[0].from === 6 && b1.items[1].ranges[0].to === 10 && b1.items[1].ranges[0].last === 'X80. F600', 'ranges carry their anchors');
const txt = briefText(b1, 'OP10');
ok(txt.startsWith('<brief op="OP10">') && txt.includes('- [user, fixed] Material: S355 steel (lines 5)') && txt.includes('- [claude] Face with T8') && txt.includes('(lines 6–10)') && txt.includes('- [drawing, fixed] Top face to Z-1\n'), 'context block');

// an edit above moves the lines; a changed anchor is found again; a lost one says so
const edited = prog.replace('(MATERIAL S355)', '(MATERIAL S355)\n(OPERATOR: DEBURR AFTER)').replace('X80. F600', 'X80. F800');
const r = relocateRanges(edited, b1.items[1].ranges);
ok(r[0].from === 7 && r[0].to === 11 && r[0].last === 'X80. F800', 'first anchor moved down one line, the edited last line keeps the span and re-anchors');
const lost = relocateRanges(edited, [{ from: 6, to: 10, first: 'GONE', last: 'ALSO GONE' }]);
ok(lost[0].from === null && describeLines(lost) === 'lines moved', 'both anchors gone → lines moved');
const r2 = relocateRanges(edited, b1.items[0].ranges);
ok(r2[0].from === 5 && r2[0].to === 5, 'unchanged line keeps its range');
ok(describeLines(r) === 'lines 7–11' && describeLines(r2) === 'lines 5', 'describeLines');
const r3 = relocateRanges(edited, makeRanges(prog, [[6, 9]]));
ok(r3[0].from === 7 && r3[0].to === 10, 'a whole range shifts with the insert');

// ids and the user's locks survive Claude's next brief
const b2 = normalizeBrief({ items: [{ text: 'Face with T8 Ø50 in one pass, offset 12 mm from the centreline', by: 'claude', lines: '6-10' }, { text: 'material: S355 STEEL', by: 'user' }] }, prog, b1);
ok(b2.items[0].id === b1.items[1].id && b2.items[1].id === b1.items[0].id, 'ids carry over by text (case and spacing ignored)');
setFixed(b1, b1.items[1].id, true);
const b3 = normalizeBrief({ items: [{ text: 'Face with T8 Ø50 in one pass, offset 12 mm from the centreline', by: 'claude', fixed: false }] }, prog, b1);
ok(b3.items[0].fixed === true && b3.items[0].fixedBy === 'user', 'a lock set by the user wins over Claude');
addItem(b3, { text: 'No chamfers on the back' });
const b4 = normalizeBrief({ items: [{ text: 'Something else', by: 'claude' }] }, prog, b3);
ok(b4.items.some((i) => i.text === 'No chamfers on the back' && i.by === 'user' && i.fixed), 'what the user typed stays when Claude leaves it out');
ok(removeItem(b4, b4.items[0].id) && b4.items.length === 1, 'remove');

// stored briefs are checked field by field
const s = sanitizeBrief({ items: [{ id: 'abc', text: 'x', by: 'user', fixed: 'yes', ranges: [{ from: 2, to: 'z', first: 'a', last: 'b' }] }, { text: '' }, 5], updatedAt: 3 });
ok(s.items.length === 1 && s.items[0].fixed === true && s.items[0].ranges[0].to === null && s.updatedAt === null, 'sanitize');
ok(sanitizeBrief(null) === null && sanitizeBrief({ items: 'no' }) === null, 'sanitize rejects junk');

console.log(fails ? `\n${fails} FAILED` : '\nall ok');
process.exit(fails ? 1 : 0);
