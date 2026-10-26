// G-code tokenizer and Fanuc Macro B expression evaluator.
// Pure module (no DOM) so it can run in the browser and in node tests.

const WORD_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

export class ParseError extends Error {}

// ---------------------------------------------------------------------------
// Line parsing
// ---------------------------------------------------------------------------

// Returns { comments: [{text}], blockDelete, words: [{letter, expr|value}], macro, label, programNumber, empty }
export function parseLine(raw) {
  const out = { raw, comments: [], blockDelete: false, words: [], macro: null, label: null, programNumber: null, empty: true, percent: false };
  let s = '';
  // strip comments: (...) (nesting allowed, e.g. "($SetPartSize(200, 150, 20, 1) )") and ; to end of line
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (ch === '(') {
      let depth = 0, end = -1;
      for (let k = i; k < raw.length; k++) {
        if (raw[k] === '(') depth++;
        else if (raw[k] === ')' && --depth === 0) { end = k; break; }
      }
      const text = end === -1 ? raw.slice(i + 1) : raw.slice(i + 1, end);
      out.comments.push(text.trim());
      if (end === -1) break;
      i = end;
      s += ' ';
    } else if (ch === ')') {
      s += ' '; // stray closing parenthesis
    } else if (ch === ';') {
      out.comments.push(raw.slice(i + 1).trim());
      break;
    } else {
      s += ch;
    }
  }
  s = s.trim();
  // "%" tape start/end; anything after it on the line ("%1", "%123") is a header
  if (s.startsWith('%')) { out.percent = true; return out; }
  if (s.startsWith('/')) { out.blockDelete = true; s = s.slice(1).trim(); }
  if (s === ':') return out;
  if (!s) return out;
  s = s.toUpperCase();

  // N label
  let m = /^N\s*(\d+)\s*/.exec(s);
  if (m) { out.label = parseInt(m[1], 10); s = s.slice(m[0].length); }
  // program number O1234 or :1234
  m = /^[O:]\s*(\d+)\s*/.exec(s);
  if (m) { out.programNumber = parseInt(m[1], 10); s = s.slice(m[0].length); }
  if (!s) { out.empty = out.label === null && out.programNumber === null; return out; }

  // macro statements
  if (/^(#|IF\b|IF\[|WHILE\b|WHILE\[|END\s*\d|GOTO)/.test(s)) {
    out.macro = parseMacro(s);
    out.empty = false;
    return out;
  }
  out.words = parseWords(s);
  out.empty = out.words.length === 0 && out.label === null && out.programNumber === null;
  return out;
}

function parseWords(s) {
  const words = [];
  let i = 0;
  const n = s.length;
  while (i < n) {
    const ch = s[i];
    if (ch === ' ' || ch === '\t' || ch === '\r') { i++; continue; }
    let comma = false;
    let letter = ch;
    if (ch === ',') {
      comma = true;
      i++;
      while (s[i] === ' ') i++;
      letter = s[i];
    }
    if (!letter || WORD_LETTERS.indexOf(letter) === -1) {
      throw new ParseError(`Unexpected character '${letter ?? ch}'`);
    }
    i++;
    // multi-letter mnemonic guard (e.g. someone wrote a Heidenhain word)
    if (WORD_LETTERS.indexOf(s[i]) !== -1 && s[i] !== undefined) {
      throw new ParseError(`Unknown word '${letter}${s[i]}…' (each word must be a letter followed by a number)`);
    }
    while (s[i] === ' ') i++;
    // value: [-+]? ( number | #... | [expr] )
    const start = i;
    if (s[i] === '-' || s[i] === '+') i++;
    while (s[i] === ' ') i++;
    if (s[i] === '#' || s[i] === '[') {
      // expression value: take a balanced chunk
      i = scanExprValue(s, i);
      const text = s.slice(start, i).replace(/\s+/g, '');
      words.push({ letter: comma ? ',' + letter : letter, expr: text });
    } else {
      const numMatch = /^\d*\.?\d*/.exec(s.slice(i));
      const numText = numMatch ? numMatch[0] : '';
      if (!numText || numText === '.') {
        throw new ParseError(`Word '${letter}' has no value`);
      }
      i += numText.length;
      const v = parseFloat(s.slice(start, i).replace(/\s+/g, ''));
      if (!Number.isFinite(v)) throw new ParseError(`Bad number in word '${letter}'`);
      words.push({ letter: comma ? ',' + letter : letter, value: v, text: s.slice(start, i).replace(/\s+/g, '') });
    }
  }
  return words;
}

function scanExprValue(s, i) {
  // starts at '#' or '['
  if (s[i] === '[') {
    let depth = 0;
    for (; i < s.length; i++) {
      if (s[i] === '[') depth++;
      else if (s[i] === ']') { depth--; if (depth === 0) return i + 1; }
    }
    throw new ParseError('Unbalanced [ ] in expression');
  }
  // '#' followed by digits or [expr]
  i++;
  if (s[i] === '[') return scanExprValue(s, i);
  if (s[i] === '#') return scanExprValue(s, i);
  const m = /^\d+/.exec(s.slice(i));
  if (!m) throw new ParseError("'#' must be followed by a variable number");
  return i + m[0].length;
}

function parseMacro(s) {
  let m;
  if ((m = /^GOTO\s*(.+)$/.exec(s))) return { type: 'goto', target: m[1].trim() };
  if ((m = /^END\s*(\d+)$/.exec(s))) return { type: 'end', id: parseInt(m[1], 10) };
  if (/^WHILE/.test(s)) {
    m = /^WHILE\s*(\[.*\])\s*DO\s*(\d+)$/.exec(s);
    if (!m) throw new ParseError('Expected WHILE [condition] DOn');
    return { type: 'while', cond: m[1], id: parseInt(m[2], 10) };
  }
  if (/^IF/.test(s)) {
    m = /^IF\s*(\[.*\])\s*GOTO\s*(.+)$/.exec(s);
    if (m) return { type: 'ifgoto', cond: m[1], target: m[2].trim() };
    m = /^IF\s*(\[.*\])\s*THEN\s*(.+)$/.exec(s);
    if (m) return { type: 'ifthen', cond: m[1], stmt: parseMacro(m[2].trim()) };
    throw new ParseError('Expected IF [condition] GOTO n  or  IF [condition] THEN #n=…');
  }
  if (s[0] === '#') {
    const eq = findTopLevelEquals(s);
    if (eq < 0) throw new ParseError("Expected assignment '#n = expression'");
    return { type: 'assign', target: s.slice(0, eq).trim(), expr: s.slice(eq + 1).trim() };
  }
  throw new ParseError('Unknown macro statement');
}

function findTopLevelEquals(s) {
  let depth = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '[') depth++;
    else if (s[i] === ']') depth--;
    else if (s[i] === '=' && depth === 0) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

const FUNCS = {
  SIN: (a) => Math.sin(a * Math.PI / 180),
  COS: (a) => Math.cos(a * Math.PI / 180),
  TAN: (a) => Math.tan(a * Math.PI / 180),
  ASIN: (a) => Math.asin(a) * 180 / Math.PI,
  ACOS: (a) => Math.acos(a) * 180 / Math.PI,
  ATAN: (a) => Math.atan(a) * 180 / Math.PI,
  SQRT: Math.sqrt,
  SQR: Math.sqrt,
  ABS: Math.abs,
  ROUND: Math.round,
  FIX: Math.trunc,
  FUP: (a) => (a < 0 ? Math.floor(a) : Math.ceil(a)),
  LN: Math.log,
  EXP: Math.exp,
};

function tokenizeExpr(text) {
  const toks = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === ' ') { i++; continue; }
    if (/[0-9.]/.test(ch)) {
      const m = /^\d*\.?\d*/.exec(text.slice(i));
      toks.push({ t: 'num', v: parseFloat(m[0]) });
      i += m[0].length;
      continue;
    }
    if ('+-*/[]#,'.includes(ch)) { toks.push({ t: ch }); i++; continue; }
    if (/[A-Z]/.test(ch)) {
      const m = /^[A-Z]+/.exec(text.slice(i));
      toks.push({ t: 'id', v: m[0] });
      i += m[0].length;
      continue;
    }
    throw new ParseError(`Unexpected '${ch}' in expression`);
  }
  return toks;
}

// Evaluates an expression string. ctx.get(n) returns variable value (null = vacant).
export function evalExpr(text, ctx) {
  const toks = tokenizeExpr(text.toUpperCase());
  let p = 0;
  const peek = () => toks[p];
  const next = () => toks[p++];
  const expect = (t) => {
    const k = next();
    if (!k || k.t !== t) throw new ParseError(`Expected '${t}' in expression`);
    return k;
  };
  const num = (v) => (v === null || v === undefined ? 0 : v);

  function primary() {
    const k = next();
    if (!k) throw new ParseError('Unexpected end of expression');
    if (k.t === 'num') return k.v;
    if (k.t === '-') return -num(primary());
    if (k.t === '+') return num(primary());
    if (k.t === '[') { const v = orExpr(); expect(']'); return v; }
    if (k.t === '#') {
      const idx = peek() && peek().t === '#' ? primary() : peek() && peek().t === '[' ? (next(), (() => { const v = orExpr(); expect(']'); return v; })()) : expect('num').v;
      return ctx.get(Math.round(num(idx)));
    }
    if (k.t === 'id') {
      const name = k.v;
      if (name === 'ATAN' && peek() && peek().t === '[') {
        next();
        const a = num(orExpr());
        expect(']');
        if (peek() && peek().t === '/' && toks[p + 1] && toks[p + 1].t === '[') {
          next(); next();
          const b = num(orExpr());
          expect(']');
          let deg = Math.atan2(a, b) * 180 / Math.PI;
          if (deg < 0) deg += 360;
          return deg;
        }
        if (peek() && peek().t === ',') {
          next();
          const b = num(orExpr());
          expect(']');
          let deg = Math.atan2(a, b) * 180 / Math.PI;
          if (deg < 0) deg += 360;
          return deg;
        }
        return FUNCS.ATAN(a);
      }
      if (name === 'POW' || name === 'MIN' || name === 'MAX') {
        expect('[');
        const a = num(orExpr());
        expect(',');
        const b = num(orExpr());
        expect(']');
        return name === 'POW' ? Math.pow(a, b) : name === 'MIN' ? Math.min(a, b) : Math.max(a, b);
      }
      const f = FUNCS[name];
      if (!f) throw new ParseError(`Unknown function '${name}'`);
      expect('[');
      const a = num(orExpr());
      expect(']');
      return f(a);
    }
    throw new ParseError(`Unexpected '${k.t}' in expression`);
  }
  function mul() {
    let v = primary();
    for (;;) {
      const k = peek();
      if (!k) break;
      if (k.t === '*') { next(); v = num(v) * num(primary()); }
      else if (k.t === '/') {
        // ATAN[a]/[b] handled in primary; plain division here
        next();
        const d = num(primary());
        if (d === 0) throw new ParseError('Division by zero');
        v = num(v) / d;
      } else if (k.t === 'id' && k.v === 'MOD') { next(); v = num(v) % num(primary()); }
      else if (k.t === 'id' && k.v === 'AND') { next(); v = (num(v) & num(primary())); }
      else break;
    }
    return v;
  }
  function add() {
    let v = mul();
    for (;;) {
      const k = peek();
      if (!k) break;
      if (k.t === '+') { next(); v = num(v) + num(mul()); }
      else if (k.t === '-') { next(); v = num(v) - num(mul()); }
      else if (k.t === 'id' && (k.v === 'OR' || k.v === 'XOR')) {
        next();
        const r = num(mul());
        v = k.v === 'OR' ? (num(v) | r) : (num(v) ^ r);
      } else break;
    }
    return v;
  }
  function cmp() {
    let v = add();
    const k = peek();
    if (k && k.t === 'id' && ['EQ', 'NE', 'GT', 'GE', 'LT', 'LE'].includes(k.v)) {
      next();
      const r = add();
      const a = v, b = r;
      switch (k.v) {
        case 'EQ': return (a === null && b === null) || num(a) === num(b) ? 1 : 0;
        case 'NE': return (a === null && b === null) || num(a) === num(b) ? 0 : 1;
        case 'GT': return num(a) > num(b) ? 1 : 0;
        case 'GE': return num(a) >= num(b) ? 1 : 0;
        case 'LT': return num(a) < num(b) ? 1 : 0;
        case 'LE': return num(a) <= num(b) ? 1 : 0;
      }
    }
    return v;
  }
  function orExpr() {
    // logical AND/OR between comparisons: [#1 GT 0 AND #2 LT 5]
    let v = cmp();
    for (;;) {
      const k = peek();
      if (k && k.t === 'id' && (k.v === 'AND' || k.v === 'OR')) {
        next();
        const r = cmp();
        v = k.v === 'AND' ? (num(v) && num(r) ? 1 : 0) : (num(v) || num(r) ? 1 : 0);
      } else break;
    }
    return v;
  }
  const v = orExpr();
  if (p < toks.length) throw new ParseError('Unexpected trailing characters in expression');
  return v;
}

// Variable index on the left-hand side of an assignment: "#100" or "#[#1+1]"
export function evalVarIndex(target, ctx) {
  const t = target.trim();
  if (!t.startsWith('#')) throw new ParseError('Assignment target must be a #variable');
  const rest = t.slice(1);
  if (/^\d+$/.test(rest)) return parseInt(rest, 10);
  return Math.round(evalExpr(rest, ctx) ?? 0);
}

// ---------------------------------------------------------------------------
// Whole-file parsing
// ---------------------------------------------------------------------------

export function parseProgram(text, fileName = 'main') {
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  const errors = [];
  const comments = [];
  for (let i = 0; i < lines.length; i++) {
    let b;
    try {
      b = parseLine(lines[i]);
    } catch (e) {
      errors.push({ line: i, file: fileName, text: e.message });
      b = { raw: lines[i], comments: [], words: [], empty: true, error: e.message };
    }
    b.line = i;
    b.file = fileName;
    for (const c of b.comments) comments.push({ line: i, text: c });
    blocks.push(b);
  }
  // label maps
  const labels = new Map();
  const programs = new Map();
  for (let i = 0; i < blocks.length; i++) {
    const b = blocks[i];
    if (b.label !== null && b.label !== undefined && !labels.has(b.label)) labels.set(b.label, i);
    if (b.programNumber !== null && b.programNumber !== undefined && !programs.has(b.programNumber)) programs.set(b.programNumber, i);
  }
  // WHILE/END matching
  const loopMatch = new Map();
  const stack = [];
  for (let i = 0; i < blocks.length; i++) {
    const mac = blocks[i].macro;
    if (!mac) continue;
    if (mac.type === 'while') stack.push({ id: mac.id, idx: i });
    else if (mac.type === 'end') {
      let k = stack.length - 1;
      while (k >= 0 && stack[k].id !== mac.id) k--;
      if (k < 0) {
        errors.push({ line: i, file: fileName, text: `END${mac.id} without matching WHILE … DO${mac.id}` });
        continue;
      }
      const w = stack.splice(k, 1)[0];
      loopMatch.set(w.idx, i);
      loopMatch.set(i, w.idx);
    }
  }
  for (const w of stack) errors.push({ line: w.idx, file: fileName, text: `WHILE … DO${w.id} without matching END${w.id}` });
  return { name: fileName, lines, blocks, labels, programs, loopMatch, errors, comments };
}
