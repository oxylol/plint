// Program edits for Claude's edit_program tool: exact `old` → `new` replacements, each matching one
// place, applied in order, all or nothing. Pure: runs in Node for the tests.
import { diffLines, diffCounts, changedRanges, unifiedDiff } from './diff.js';

export class EditError extends Error {}

const norm = (s) => String(s ?? '').replace(/\r\n?/g, '\n');

function occurrences(text, needle) {
  const at = [];
  let i = 0;
  while ((i = text.indexOf(needle, i)) >= 0) { at.push(i); i += Math.max(1, needle.length); }
  return at;
}

const lineOf = (text, index) => text.slice(0, index).split('\n').length;

// the same lines ignoring trailing whitespace: [startIndex, endIndex] in `text`, or null
function trimmedMatches(text, needle) {
  const lines = text.split('\n'), want = needle.split('\n').map((l) => l.trimEnd());
  while (want.length > 1 && want[want.length - 1] === '') want.pop();
  const found = [];
  for (let i = 0; i + want.length <= lines.length; i++) {
    let ok = true;
    for (let k = 0; k < want.length; k++) if (lines[i + k].trimEnd() !== want[k]) { ok = false; break; }
    if (ok) found.push(i);
  }
  return found.map((i) => {
    const start = lines.slice(0, i).reduce((n, l) => n + l.length + 1, 0);
    const end = start + lines.slice(i, i + want.length).reduce((n, l) => n + l.length + 1, 0) - 1;
    return [start, end];
  });
}

// lines that look most like the edit's first line, to help the next attempt
function closest(text, needle) {
  const first = needle.split('\n').map((l) => l.trim()).find((l) => l) || '';
  const lines = text.split('\n');
  const score = (l) => {
    const t = l.trim();
    if (!t) return 0;
    let n = 0;
    while (n < t.length && n < first.length && t[n] === first[n]) n++;
    const words = first.split(/\s+/).filter((w) => w.length > 1 && t.includes(w)).length;
    return n * 3 + words;
  };
  const ranked = lines.map((l, i) => ({ i, s: score(l) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 3);
  if (!ranked.length) return '';
  const show = (i) => lines.slice(Math.max(0, i - 1), i + 2).map((l, k) => `  L${Math.max(1, i) + k}: ${l}`).join('\n');
  return `\nClosest lines in the program:\n${ranked.map((x) => show(x.i)).join('\n  …\n')}`;
}

// returns { text, added, removed, changed: [[from, to], …] (1-based lines of the new text), ops, patch }
export function applyEdits(text, edits) {
  const before = norm(text);
  let cur = before;
  if (!Array.isArray(edits) || !edits.length) throw new EditError('No edits given');
  edits.forEach((e, n) => {
    const old = norm(e?.old), nw = norm(e?.new);
    if (!old.trim()) throw new EditError(`Edit ${n + 1}: "old" is empty`);
    let hits = occurrences(cur, old).map((i) => [i, i + old.length]);
    if (!hits.length) hits = trimmedMatches(cur, old);
    if (!hits.length) throw new EditError(`Edit ${n + 1} does not match the program: no line reads\n${old.split('\n').map((l) => `  ${l}`).join('\n')}${closest(cur, old)}`);
    if (hits.length > 1) throw new EditError(`Edit ${n + 1} matches ${hits.length} places (lines ${hits.map(([i]) => lineOf(cur, i)).join(', ')}) — include a line before or after it so it matches once`);
    const [s0, s1] = hits[0];
    // deleting whole lines takes the line break with them
    const whole = nw === '' && (s0 === 0 || cur[s0 - 1] === '\n') && (s1 === cur.length || cur[s1] === '\n');
    cur = cur.slice(0, s0) + nw + cur.slice(whole && s1 < cur.length ? s1 + 1 : s1);
  });
  const ops = diffLines(before.split('\n'), cur.split('\n'));
  const { added, removed } = diffCounts(ops);
  return { text: cur, added, removed, changed: changedRanges(ops), ops, patch: unifiedDiff(ops) };
}
