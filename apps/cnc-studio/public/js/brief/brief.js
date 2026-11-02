// The brief of an operation: what the user, the drawing or the model require, and what Claude chose on
// its own — each item tied to the program lines it concerns. Lines are kept as anchors (the text of
// the first and last line of each range) so they survive edits. Pure: runs in Node for the tests.

export const SOURCES = ['user', 'drawing', 'model', 'claude'];
export const MAX_ITEMS = 60;

const uid = () => Math.random().toString(36).slice(2, 8);
const key = (s) => String(s || '').toLowerCase().replace(/\s+/g, ' ').trim();
const lineList = (text) => String(text || '').replace(/\r\n?/g, '\n').split('\n');

// "6, 21-25" | "6" | [[6, 6], [21, 25]] | [6, [21, 25]] → [[from, to], …] (1-based, inclusive)
export function parseLines(spec) {
  const out = [];
  const push = (a, b) => { a = Math.floor(Number(a)); b = Math.floor(Number(b ?? a)); if (a >= 1 && b >= a) out.push([a, b]); };
  if (Array.isArray(spec)) {
    for (const r of spec) { if (Array.isArray(r)) push(r[0], r[1]); else push(r, r); }
    return out;
  }
  for (const part of String(spec ?? '').split(/[,;]/)) {
    const m = /^\s*L?\s*(\d+)\s*(?:[-–—]\s*L?\s*(\d+))?\s*$/i.exec(part);
    if (m) push(m[1], m[2]);
  }
  return out;
}

// ranges with their anchors, clamped to the program
export function makeRanges(text, lineRanges) {
  const lines = lineList(text);
  const out = [];
  for (let [a, b] of lineRanges || []) {
    a = Math.max(1, Math.min(lines.length, a));
    b = Math.max(a, Math.min(lines.length, b));
    out.push({ from: a, to: b, first: lines[a - 1].trim(), last: lines[b - 1].trim() });
  }
  return out;
}

// find the anchored lines again in a changed program. A range whose one end was edited keeps its
// length and is re-anchored there; one whose both ends are gone gets from/to = null ("lines moved").
export function relocateRanges(text, ranges) {
  const lines = lineList(text).map((l) => l.trim());
  const n = lines.length;
  const nearest = (needle, around) => {
    let best = -1;
    for (let i = 0; i < n; i++) if (lines[i] === needle && (best < 0 || Math.abs(i - around) < Math.abs(best - around))) best = i;
    return best;
  };
  return (ranges || []).map((r) => {
    if (!r.first && !r.last) return { ...r, from: null, to: null };
    if (r.from && r.to && lines[r.from - 1] === r.first && lines[r.to - 1] === r.last) return { ...r };
    const span = Math.max(0, (r.to || r.from || 1) - (r.from || 1));
    const want = (r.from || 1) - 1;
    const i = r.first ? nearest(r.first, want) : -1;
    if (i >= 0) {
      let j = -1;
      if (span === 0) j = i;
      else for (let k = i + 1; k < n && k <= i + span + 30; k++) if (lines[k] === r.last) { j = k; break; }
      if (j < 0) j = Math.min(n - 1, i + span);
      return { from: i + 1, to: j + 1, first: lines[i], last: lines[j] };
    }
    const j = r.last ? nearest(r.last, want + span) : -1;
    if (j >= 0) { const i2 = Math.max(0, j - span); return { from: i2 + 1, to: j + 1, first: lines[i2], last: lines[j] }; }
    return { ...r, from: null, to: null };
  });
}

export function relocateBrief(brief, text) {
  if (!brief?.items) return brief;
  for (const it of brief.items) it.ranges = relocateRanges(text, it.ranges);
  return brief;
}

// Claude's `brief` input → the stored brief; ids and the user's own lock choices carry over
export function normalizeBrief(input, text, prev) {
  const items = [];
  const prevItems = prev?.items || [];
  const byKey = new Map(prevItems.map((p) => [key(p.text), p]));
  const used = new Set();
  for (const raw of (Array.isArray(input?.items) ? input.items : []).slice(0, MAX_ITEMS)) {
    const txt = String(raw?.text ?? '').replace(/\s+/g, ' ').trim().slice(0, 300);
    if (!txt) continue;
    const by = SOURCES.includes(raw?.by) ? raw.by : 'claude';
    let fixed = typeof raw?.fixed === 'boolean' ? raw.fixed : by === 'user' || by === 'drawing';
    const old = byKey.get(key(txt));
    let id = old && !used.has(old.id) ? old.id : uid();
    used.add(id);
    let fixedBy;
    if (old?.fixedBy === 'user') { fixed = old.fixed; fixedBy = 'user'; }
    const ranges = makeRanges(text, parseLines(raw?.lines));
    items.push({ id, text: txt, by, fixed, ...(fixedBy ? { fixedBy } : {}), ranges });
  }
  // what the user typed in by hand stays even when Claude left it out
  for (const p of prevItems) if (p.by === 'user' && p.addedBy === 'user' && !used.has(p.id)) items.push({ ...p, ranges: relocateRanges(text, p.ranges) });
  return { items: items.slice(0, MAX_ITEMS), updatedAt: new Date().toISOString() };
}

export function addItem(brief, { text, by = 'user', fixed = true, ranges = [] }) {
  const b = brief?.items ? brief : { items: [], updatedAt: null };
  const txt = String(text || '').replace(/\s+/g, ' ').trim().slice(0, 300);
  if (!txt) return b;
  b.items.push({ id: uid(), text: txt, by, fixed, fixedBy: 'user', addedBy: 'user', ranges });
  b.updatedAt = new Date().toISOString();
  return b;
}

export function setFixed(brief, id, fixed) {
  const it = brief?.items?.find((x) => x.id === id);
  if (!it) return false;
  it.fixed = !!fixed;
  it.fixedBy = 'user';
  brief.updatedAt = new Date().toISOString();
  return true;
}

export function removeItem(brief, id) {
  if (!brief?.items) return false;
  const n = brief.items.length;
  brief.items = brief.items.filter((x) => x.id !== id);
  brief.updatedAt = new Date().toISOString();
  return brief.items.length < n;
}

export function describeLines(ranges) {
  const live = (ranges || []).filter((r) => r.from);
  if (!live.length) return (ranges || []).length ? 'lines moved' : '';
  return `lines ${live.map((r) => (r.from === r.to ? `${r.from}` : `${r.from}–${r.to}`)).join(', ')}`;
}

// the block Claude sees before the program
export function briefText(brief, opName) {
  const items = brief?.items || [];
  if (!items.length) return '';
  const o = [`<brief op="${opName}">`];
  for (const it of items) {
    const tags = [it.by, it.fixed ? 'fixed' : null].filter(Boolean).join(', ');
    const ln = describeLines(it.ranges);
    o.push(`- [${tags}] ${it.text}${ln ? ` (${ln})` : ''}`);
  }
  o.push('</brief>');
  return o.join('\n');
}

// a stored brief from the server or a file, checked field by field
export function sanitizeBrief(raw) {
  if (!raw || typeof raw !== 'object' || !Array.isArray(raw.items)) return null;
  const items = [];
  for (const it of raw.items.slice(0, MAX_ITEMS)) {
    if (!it || typeof it !== 'object') continue;
    const text = String(it.text ?? '').slice(0, 300).trim();
    if (!text) continue;
    const ranges = Array.isArray(it.ranges) ? it.ranges.filter((r) => r && typeof r === 'object').map((r) => ({
      from: Number.isInteger(r.from) && r.from >= 1 ? r.from : null, to: Number.isInteger(r.to) && r.to >= 1 ? r.to : null,
      first: String(r.first ?? '').slice(0, 200), last: String(r.last ?? '').slice(0, 200),
    })).slice(0, 20) : [];
    items.push({
      id: /^[a-z0-9]{1,12}$/i.test(String(it.id)) ? String(it.id) : uid(), text, by: SOURCES.includes(it.by) ? it.by : 'claude', fixed: !!it.fixed,
      ...(it.fixedBy === 'user' ? { fixedBy: 'user' } : {}), ...(it.addedBy === 'user' ? { addedBy: 'user' } : {}), ranges,
    });
  }
  return { items, updatedAt: typeof raw.updatedAt === 'string' ? raw.updatedAt : null };
}
