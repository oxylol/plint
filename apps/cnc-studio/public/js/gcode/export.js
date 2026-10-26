// One NC program for the machine: CNC Studio's own directives taken out (STOCK, TOOL, SETUP,
// ORIENT, ZERO, VISE, CHUCK, ROTARY, CncSimulator $ directives), comments reduced to plain
// upper-case ASCII, M98 subprograms built in (or appended as O programs when that is not safe),
// several operations of one machine joined with an M00 stop and a re-clamp note, % … % and an
// O number. The result is run through the interpreter again and compared move by move with
// the original program.
import { interpret } from './interpreter.js';
import { parseSetup } from './setup.js';

const fx = (v, d = 3) => { const s = (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d); return s.includes('.') ? s.replace(/\.?0+$/, '') || '0' : s; };

// comments CNC Studio reads as directives (mirrors parseSetup, the STOCK/TOOL readers and $ directives)
const DIRECTIVES = [
  /^(STOCK|BLANK|RÄMNE|RAMNE|ÄMNE|AMNE|WORKPIECE|PART SIZE)\b/,
  /^\$/,
  /^TOOL\s+T\s*\d+/,
  /^(SETUP|OPERATION)\s*[:=]?\s*\S/,
  /^OP\s*\d+$/,
  /^ORIENT(ATION)?\s+/,
  /^ZERO\s+(SAME|TOP|FACE|PART)\b/,
  /^(VISE|CHUCK)\b/,
  /^ROTARY\s+[AB]\b/,
  /^(RADIUS|DIAMETER)\s+PROGRAMMING$/,
];
export const isDirective = (text) => { const t = text.toUpperCase().replace(/\s+/g, ' ').trim(); return DIRECTIVES.some((re) => re.test(t)); };

const CHARS = { 'Ø': 'DIA', 'ø': 'DIA', '×': 'X', '°': 'DEG', '±': '+-', 'Å': 'A', 'Ä': 'A', 'Ö': 'O', 'Ü': 'U', 'É': 'E', 'È': 'E', 'ß': 'SS', 'µ': 'U', '²': '2', '³': '3', '–': '-', '—': '-', '’': '', '"': '', "'": '' };
// what a Fanuc-style control accepts inside ( ): upper-case letters, digits, space and a few signs
export const plainComment = (text) => [...text.toUpperCase()].map((c) => CHARS[c] ?? c).join('').replace(/[^A-Z0-9 .,:=+\-*/#%_]/g, ' ').replace(/\s+/g, ' ').trim();

// a line split into its code and its comments ( (…) with nesting, and ; to the end of the line )
export function splitLine(raw) {
  let code = '', depth = 0, cur = '';
  const comments = [];
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (depth === 0 && c === ';') { comments.push(raw.slice(i + 1).trim()); break; }
    if (c === '(') { if (depth++ > 0) cur += c; continue; }
    if (c === ')' && depth > 0) { if (--depth === 0) { comments.push(cur.trim()); cur = ''; } else cur += c; continue; }
    if (depth > 0) cur += c; else code += c;
  }
  if (depth > 0) comments.push(cur.trim());
  return { code: code.replace(/\s+$/, ''), comments };
}

const words = (code) => code.toUpperCase().replace(/\s+/g, '');
const hasWord = (code, re) => re.test(words(code));
const isEnd = (code) => hasWord(code, /M(30|02|2)(?!\d)/);

// ---------------------------------------------------------------- subprograms
// O number → the lines of that program (after its O line, up to and including M99), from the
// main file (after M30) and the subprogram files — the same places the interpreter looks
function programTable(mainText, subprograms) {
  const table = new Map();
  const collect = (text, fallback) => {
    const lines = text.replace(/\r\n?/g, '\n').split('\n');
    let cur = null, found = false;
    for (const raw of lines) {
      const code = words(splitLine(raw).code);
      const o = /^O(\d+)/.exec(code) || /^:(\d+)/.exec(code);
      if (o) { cur = []; found = true; if (!table.has(+o[1])) table.set(+o[1], cur); continue; }
      if (cur) { cur.push(raw); if (/M99(?!\d)/.test(code)) cur = null; }
    }
    if (!found && fallback !== null && !table.has(fallback)) table.set(fallback, lines.filter((l) => l.trim() !== '%'));
  };
  collect(mainText, null);
  for (const sp of subprograms || []) { const m = /(\d+)/.exec(sp.name); collect(sp.text, m ? +m[1] : null); }
  return table;
}

// a subprogram can be built in when it is plain: no jumps (GOTO, M99 P) and no macro call of its own
const inlinable = (body) => !body.some((l) => { const c = words(splitLine(l).code); return /GOTO|M99P|G65/.test(c); });

// ---------------------------------------------------------------- one operation
// returns { lines, appended: Map(num → lines), notes } for one program text
function cleanProgram(text, { subprograms, keepComments, inlineSubs }, notes) {
  const table = programTable(text, subprograms);
  const appended = new Map();
  const out = [];
  const emit = (raw, depth) => {
    const { code, comments } = splitLine(raw);
    const c = code.trim();
    if (c === '%') return;
    const W = words(c);
    if (/^O\d+/.test(W) || /^:\d+/.test(W)) return; // the O number is written once, at the top
    const kept = keepComments ? comments.filter((t) => t && !isDirective(t)).map(plainComment).filter(Boolean) : [];
    const call = /M98/.test(W) ? callOf(W) : null;
    const macro = /G65/.test(W) ? /P(\d+)/.exec(W.slice(W.indexOf('G65'))) : null;
    if (macro) addAppended(+macro[1]);
    if (call && inlineSubs && table.has(call.num) && inlinable(table.get(call.num)) && depth < 6) {
      const rest = c.toUpperCase().replace(/M98\s*/, '').replace(/P\s*\d+/, '').replace(/L\s*\d+/, '').trim();
      if (rest) out.push(rest);
      if (keepComments) out.push(`(O${String(call.num).padStart(4, '0')}${call.reps > 1 ? ` X${call.reps}` : ''})`);
      // the body without its M99 (other words on that line stay)
      const body = table.get(call.num).map((l) => (/M99(?!\d)/.test(words(splitLine(l).code)) ? l.replace(/M\s*99(?!\d)/i, '') : l));
      for (let r = 0; r < call.reps; r++) for (const l of body) emit(l, depth + 1);
      return;
    }
    if (call) addAppended(call.num);
    const line = [c.toUpperCase(), ...kept.map((k) => `(${k})`)].filter(Boolean).join(' ');
    if (line) out.push(line);
  };
  const addAppended = (num) => {
    if (appended.has(num) || !table.has(num)) { if (!table.has(num)) notes.push(`O${String(num).padStart(4, '0')} is called but not loaded — add it under Subprograms`); return; }
    appended.set(num, null);
    const sub = [];
    for (const l of table.get(num)) {
      const { code, comments } = splitLine(l);
      const W = words(code);
      if (/M98/.test(W) || /G65/.test(W)) { const n = callOf(W)?.num ?? +(/G65.*?P(\d+)/.exec(W)?.[1] ?? NaN); if (Number.isFinite(n) && n !== num) addAppended(n); }
      const kept = keepComments ? comments.filter((t) => t && !isDirective(t)).map(plainComment).filter(Boolean) : [];
      const line = [code.trim().toUpperCase(), ...kept.map((k) => `(${k})`)].filter(Boolean).join(' ');
      if (line && line !== '%') sub.push(line);
    }
    appended.set(num, sub);
  };
  // the main program: up to its M30 / M02 (what follows are its subprograms)
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  let ended = false;
  for (const raw of lines) {
    if (ended) break;
    emit(raw, 0);
    if (isEnd(splitLine(raw).code)) ended = true;
  }
  return { lines: out, appended, ended };
}

// M98 P1000 L3, or the Fanuc 0 form M98 P31000 (3 × O1000)
function callOf(W) {
  const p = /M98.*?P(\d+)/.exec(W) || /P(\d+).*?M98/.exec(W);
  if (!p) return null;
  let num = +p[1], reps = 1;
  const l = /L(\d+)/.exec(W);
  if (!l && p[1].length > 4) { reps = +p[1].slice(0, -4) || 1; num = +p[1].slice(-4); }
  if (l) reps = +l[1];
  return { num, reps };
}

// ---------------------------------------------------------------- the job
const SETUP_WORDS = (s, machine) => {
  const o = [];
  for (const r of s.orient) o.push(r.deg === 180 ? `TURN OVER ABOUT ${r.axis.toUpperCase()}` : `ROTATE ${fx(r.deg)} DEG ABOUT ${r.axis.toUpperCase()}`);
  if (machine === 'lathe' && s.orient.length) o.splice(0, o.length, 'TURN THE PART AROUND IN THE CHUCK');
  if (s.zero) o.push(`ZERO ${s.zero.mode === 'part' ? `PART X${fx(s.zero.point[0])} Y${fx(s.zero.point[1])} Z${fx(s.zero.point[2])}` : s.zero.mode.replace('-', ' ').toUpperCase()}`);
  if (s.vise) o.push(`VISE GRIP ${fx(s.vise.grip)}`);
  if (s.chuck) o.push(`CHUCK GRIP ${fx(s.chuck.grip)}`);
  return o.join(', ');
};

const oNumber = (text) => { for (const l of text.split('\n')) { const W = words(splitLine(l).code); const m = /^O(\d+)/.exec(W); if (m) return { num: +m[1], name: plainComment(splitLine(l).comments.join(' ')) }; if (/[GXYZM]/.test(W)) break; } return null; };

// opts: { ops: [{ text, machine, name, fileName, out }], subprograms, keepComments, inlineSubs, toolList,
//         programNumber, interpOptions(machine) } → { text, notes, verify: [{ name, ok, text }] }
export function exportJob(opts) {
  const built = buildJob(opts);
  return { ...built, verify: opts.ops.map((op) => verifyOp(op, opts)) };
}

function buildJob(opts) {
  const { ops, subprograms, keepComments = true, inlineSubs = true, toolList = true } = opts;
  const machine = ops[0].machine;
  const notes = [];
  const first = oNumber(ops[0].text);
  const num = opts.programNumber || first?.num || 1000;
  const title = plainComment(first?.name || ops[0].fileName.replace(/\.[^.]+$/, ''));
  const head = ['%', `O${String(num).padStart(4, '0')}${title ? ` (${title})` : ''}`];
  const body = [], appended = new Map();
  if (toolList) {
    const seen = new Set();
    for (const op of ops) for (const n of op.out?.result.usedTools || []) {
      if (seen.has(n)) continue;
      seen.add(n);
      const t = op.out.result.tools[n];
      head.push(`(${plainComment(`T${n} ${t.diameter && machine === 'mill' ? `D${fx(t.diameter)} ` : ''}${t.type || ''}${t.name ? ` - ${t.name}` : ''}`)})`);
    }
  }
  ops.forEach((op, i) => {
    const c = cleanProgram(op.text, { subprograms, keepComments, inlineSubs }, notes);
    let lines = c.lines;
    // between operations: drop this one's program end, stop for the re-clamp
    if (ops.length > 1) {
      while (lines.length && isEnd(lines[lines.length - 1])) lines = lines.slice(0, -1);
      const setup = parseSetup(op.text.split('\n').flatMap((l) => splitLine(l).comments.map((t) => ({ text: t }))));
      if (i > 0) body.push('M05', 'M09', `(${plainComment(`${op.name} - ${SETUP_WORDS(setup, machine) || 'RE-CLAMP THE PART'}`)})`, 'M00');
      else body.push(`(${plainComment(op.name)})`);
    } else if (!c.ended) notes.push('The program has no M30 — one was added at the end');
    body.push(...lines);
    for (const [k, v] of c.appended) if (v && !appended.has(k)) appended.set(k, v);
  });
  if (ops.length > 1 || !isEnd(body[body.length - 1] || '')) body.push('M30');
  const subs = [];
  for (const [k, v] of appended) subs.push(`O${String(k).padStart(4, '0')}`, ...v);
  if (appended.size) notes.push(`${appended.size} subprogram${appended.size > 1 ? 's' : ''} appended after M30 as O program${appended.size > 1 ? 's' : ''} (${[...appended.keys()].map((k) => `O${String(k).padStart(4, '0')}`).join(', ')}): ${inlineSubs ? 'they use jumps or macro calls, so they cannot be built in' : 'as chosen'}. Some controls load each O program from the file separately.`);
  if (machine === 'lathe' && ops.some((op) => /\$SETRADIUSPRG\s*\(\s*1|\(\s*RADIUS\s+PROGRAMMING\s*\)/i.test(op.text))) notes.push('X is programmed as RADIUS in this program: set the control to radius programming before running it');
  const code = (t) => t.split('\n').map((l) => splitLine(l).code).join('\n');
  if (ops.some((op) => /#\d|\bIF\b|\bWHILE\b/i.test(code(op.text)))) notes.push('Uses custom macro B (#variables, IF/WHILE): the control needs that option');
  const text = [...head, ...body, ...subs, '%'].join('\r\n') + '\r\n';
  return { text, notes };
}

// the cleaned program of one operation must give the same moves as the original
function verifyOp(op, opts) {
  const io = opts.interpOptions(op.machine);
  const radius = /\$SETRADIUSPRG\s*\(\s*1|\(\s*RADIUS\s+PROGRAMMING\s*\)/i.test(op.text) && !/\(\s*DIAMETER\s+PROGRAMMING\s*\)/i.test(op.text);
  const single = buildJob({ ...opts, ops: [op], toolList: false }).text;
  const a = op.out?.result || interpret(op.text, io);
  // same tools and stock as the original (home height comes from the stock), nothing else
  const b = interpret(single, { ...io, toolLibrary: a.tools, stockOverride: a.stock, subprograms: [], settings: { ...io.settings, latheRadiusMode: io.settings?.latheRadiusMode || radius } });
  const ma = a.moves.filter((m) => m.kind === 'feed' || m.kind === 'rapid'), mb = b.moves.filter((m) => m.kind === 'feed' || m.kind === 'rapid');
  const same = (p, q) => Math.abs(p[0] - q[0]) < 1e-6 && Math.abs(p[1] - q[1]) < 1e-6 && Math.abs(p[2] - q[2]) < 1e-6;
  for (let k = 0; k < Math.max(ma.length, mb.length); k++) {
    const x = ma[k], y = mb[k];
    // end points only: the first move starts from home, which depends on the stock the export leaves out
    if (!x || !y || x.kind !== y.kind || x.tool !== y.tool || !same(x.pts[x.pts.length - 1], y.pts[y.pts.length - 1])) {
      return { name: op.name, ok: false, text: `${op.name}: the exported program differs from the simulated one at move ${k + 1}${x ? ` (original line ${x.callLine + 1})` : ''} — check it before running` };
    }
  }
  const errs = b.messages.filter((m) => m.level === 'error').length - a.messages.filter((m) => m.level === 'error').length;
  return { name: op.name, ok: errs <= 0, text: `${op.name}: same toolpath as the simulation — ${ma.length} moves${errs > 0 ? `, but ${errs} new error message(s)` : ''}` };
}

