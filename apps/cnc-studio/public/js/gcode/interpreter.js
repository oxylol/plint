// Fanuc-style G-code interpreter for a 3-axis mill and a 2-axis lathe.
// Produces a timed list of moves for simulation plus diagnostics.
//
// Internal coordinates: mill [x, y, z]; lathe [radius, 0, z] (X words are diameters).

import { parseProgram, evalExpr, evalVarIndex } from './parser.js';
import { CornerStage, CompStage, buildArc, centerFromRadius, tessellate, primLength, PLANES, uv } from './contour.js';
import { parseToolsFromComments, normalizeTool, defaultLibrary } from './tools.js';
import { parseStockFromComments, autoStock } from './stock.js';
import { latheRoughing, latheFinish, latheBoxCycle, latheThreadCycle, lathePeck, latheG76 } from './lathecycles.js';

export const DEFAULT_SETTINGS = {
  rapidMill: 15000,
  rapidLathe: 10000,
  toolChangeTime: 4,
  maxRpm: 4000,
  arcTolerance: 0.01,
  peckClearance: 0.5,
  maxBlocks: 400000,
  maxMoves: 250000,
};

const MILL_CYCLES = new Set([73, 74, 76, 81, 82, 83, 84, 85, 86, 88, 89]);
const CYCLE_NAMES = {
  73: 'G73 high-speed peck', 74: 'G74 left-hand tap', 76: 'G76 fine bore', 81: 'G81 drill', 82: 'G82 spot/dwell',
  83: 'G83 peck drill', 84: 'G84 tap', 85: 'G85 bore', 86: 'G86 bore', 88: 'G88 bore', 89: 'G89 bore dwell',
};
const ARG_VARS = { A: 1, B: 2, C: 3, I: 4, J: 5, K: 6, D: 7, E: 8, F: 9, H: 11, M: 13, Q: 17, R: 18, S: 19, T: 20, U: 21, V: 22, W: 23, X: 24, Y: 25, Z: 26 };

class StopProgram extends Error {}

export function interpret(source, opts = {}) {
  const machine = opts.machine === 'lathe' ? 'lathe' : 'mill';
  const first = new Interpreter(source, { ...opts, machine });
  const res = first.run();
  if (res.stock.source === 'auto' && first.usedHome) {
    // re-run so G28 / G53 home moves are placed relative to the fitted stock
    const second = new Interpreter(source, { ...opts, machine, stockOverride: res.stock });
    const res2 = second.run();
    res2.stock = res.stock;
    return res2;
  }
  return res;
}

class Interpreter {
  constructor(source, opts) {
    this.opts = opts;
    this.machine = opts.machine;
    this.lathe = this.machine === 'lathe';
    this.settings = { ...DEFAULT_SETTINGS, ...(opts.settings || {}) };
    this.messages = [];
    this.warnOnce = new Set();
    this.moves = [];
    this.time = 0;
    this.main = parseProgram(source, 'main');
    for (const e of this.main.errors) this.msg('error', e.text, e.line, 'main', e.line);

    // subprogram files: number -> { file, idx }
    this.files = new Map([['main', this.main]]);
    this.extPrograms = new Map();
    for (const sp of opts.subprograms || []) {
      const f = parseProgram(sp.text, sp.name);
      this.files.set(sp.name, f);
      for (const e of f.errors) this.msg('warning', `${sp.name}: ${e.text}`, null, sp.name, e.line);
      if (f.programs.size) {
        for (const [n, idx] of f.programs) if (!this.extPrograms.has(n)) this.extPrograms.set(n, { file: f, idx });
      }
      const m = /(\d+)/.exec(sp.name);
      if (m && !this.extPrograms.has(parseInt(m[1], 10))) this.extPrograms.set(parseInt(m[1], 10), { file: f, idx: 0 });
    }

    // tools & stock from comments
    const lib = opts.toolLibrary || defaultLibrary(this.machine);
    const fromProgram = parseToolsFromComments(this.main.comments, this.machine);
    this.tools = {};
    for (const [k, t] of Object.entries(lib)) this.tools[k] = normalizeTool({ ...t, source: t.source || 'library' }, this.machine);
    for (const [k, t] of Object.entries(fromProgram)) {
      // a program comment describes the insert actually used: keep the library's (register)
      // geometry only when it is the same kind of tool with the same nose radius
      let base = lib[k] && lib[k].type === t.type ? { ...lib[k] } : {};
      const sameNose = t.noseRadius === undefined || Math.abs((base.noseRadius ?? -1) - t.noseRadius) < 0.01;
      if (this.lathe && !sameNose) { delete base.cutter; delete base.holder; delete base.noseOffset; delete base.insertAngle; }
      this.tools[k] = normalizeTool({ ...base, ...t }, this.machine);
    }
    this.programTools = fromProgram;
    this.stock = opts.stockOverride || parseStockFromComments(this.main.comments, this.machine);

    this.hasM6 = this.main.blocks.some((b) => b.words && b.words.some((w) => w.letter === 'M' && w.value === 6));
    // lathe X in diameter (default) or radius: setting, "(RADIUS PROGRAMMING)" or CncSimulator's ($SetRadiusPrg(1))
    this.xScale = 0.5;
    if (this.lathe) {
      let radius = !!this.settings.latheRadiusMode;
      for (const c of this.main.comments) {
        const m = /\$SETRADIUSPRG\s*\(\s*(\d)/i.exec(c.text);
        if (m) radius = m[1] === '1';
        if (/^\s*RADIUS\s+PROGRAMMING\s*$/i.test(c.text)) radius = true;
        if (/^\s*DIAMETER\s+PROGRAMMING\s*$/i.test(c.text)) radius = false;
      }
      if (radius) this.xScale = 1;
    }
    this.usedHome = false;
    this.home = this.computeHome();

    // machine state
    this.pos = this.lathe ? [this.home[0], 0, this.home[2]] : this.home.slice();
    this.origin = [0, 0, 0];
    this.wcs = 54;
    this.offsets = {};
    for (let g = 54; g <= 59; g++) this.offsets[g] = [0, 0, 0];
    this.g52 = [0, 0, 0];
    this.g92 = [0, 0, 0];
    this.motion = 0;
    this.plane = this.lathe ? 18 : 17;
    this.abs = true;
    this.units = 1;
    this.feedPerRev = this.lathe; // lathe G99 default, mill G94 default
    if (this.lathe) {
      // Programs written for CncSimulator's ISO lathe ("F300", "F100", no G98/G99) mean mm/min.
      // With the Fanuc default G99 they would be absurd mm/rev values, so follow the original simulator.
      const words = this.main.blocks.flatMap((b) => b.words || []);
      const hasFeedMode = words.some((w) => w.letter === 'G' && (w.value === 98 || w.value === 99));
      const feeds = words.filter((w) => w.letter === 'F' && w.value !== undefined).map((w) => w.value);
      if (!hasFeedMode && feeds.length && Math.max(...feeds) > 5) {
        this.feedPerRev = false;
        this.msg('info', 'No G98/G99 and F values like F' + Math.max(...feeds) + ' — feeds treated as mm/min (as in CncSimulator); add G99 for mm/rev', null, 'main', null);
      }
    }
    this.F = 0;
    this.S = 0;
    this.css = false;
    this.cssSpeed = 0;
    this.maxRpm = this.settings.maxRpm;
    this.spindle = 'off';
    this.coolant = false;
    this.tool = 0;
    this.pendingTool = 0;
    this.compSide = 0;
    this.compD = 0;
    this.rotA = 0; // 4th/5th axis positions (degrees)
    this.rotB = 0;
    this.usesRotary = false;
    this.lengthComp = false;
    this.polar = false;
    this.cycle = null;       // mill canned cycle modal data
    this.latheCycle = null;  // lathe modal G90/G92/G94
    this.cycleParams = {};   // lathe two-block cycle first-block values
    this.retLevel = 98;
    this.vars = new Map();
    this.locals = [new Map()];
    this.capture = null;
    this.ended = false;
    this.blockCount = 0;
    this.stats = { toolChanges: 0, rapidLen: 0, feedLen: 0, minZ: Infinity, maxZ: -Infinity, warnedFeed: false, warnedSpindle: false, warnedS: false };

    // path pipeline: corner rounding -> compensation -> move list
    const sink = { push: (p) => this.addMove(p), flush: () => {} };
    this.comp = new CompStage(sink, (m, t) => this.msgMeta('warning', t, m), (m, t) => this.msgMeta('error', t, m));
    this.corner = new CornerStage(this.comp, (m, t) => this.msgMeta('warning', t, m));
  }

  computeHome() {
    const s = this.stock;
    if (this.lathe) {
      const r = s ? s.diameter / 2 + 40 : 60;
      return [r, 0, (s ? s.faceStock || 0 : 0) + 60];
    }
    if (s) return [s.max[0] + 25, s.max[1] + 25, s.max[2] + 75];
    return [100, 100, 100];
  }

  // ------------------------------------------------------------------ messages
  msg(level, text, line, file = 'main', callLine = line) {
    const key = `${level}|${text}|${file}|${line}`;
    if (this.warnOnce.has(key)) return;
    this.warnOnce.add(key);
    if (this.messages.length > 500) return;
    this.messages.push({ level, text, line, file, callLine });
  }
  msgMeta(level, text, meta) { this.msg(level, text, meta?.line ?? null, meta?.file ?? 'main', meta?.callLine ?? meta?.line ?? null); }
  warnOnceGlobal(key, level, text, b) {
    if (this.warnOnce.has('g:' + key)) return;
    this.warnOnce.add('g:' + key);
    this.msg(level, text, b?.line ?? null, b?.file ?? 'main', this.curCallLine);
  }

  // ------------------------------------------------------------------ run
  run() {
    const frames = [{ file: this.main, pc: 0, repeat: 1, start: 0, callLine: null, isMain: true }];
    this.frames = frames;
    try {
      while (frames.length && !this.ended) {
        const f = frames[frames.length - 1];
        if (f.pc >= f.file.blocks.length) {
          if (f.isMain) break;
          this.returnFromSub(frames, null);
          continue;
        }
        const b = f.file.blocks[f.pc++];
        if (++this.blockCount > this.settings.maxBlocks) {
          this.msg('error', `Stopped after ${this.settings.maxBlocks} blocks — infinite loop or program too long`, b.line, b.file, this.callLineOf(f, b));
          break;
        }
        if (b.empty || b.error) continue;
        if (b.blockDelete && this.opts.blockDelete) continue;
        this.curBlock = b;
        this.curFrame = f;
        this.curCallLine = this.callLineOf(f, b);
        try {
          if (b.macro) this.execMacro(b, f);
          else this.execBlock(b, f, frames);
        } catch (e) {
          if (e instanceof StopProgram) throw e;
          this.msg('error', e.message, b.line, b.file, this.curCallLine);
        }
      }
      if (!this.ended) this.msg('warning', 'Program ends without M30 / M02', null, 'main', null);
    } catch (e) {
      if (!(e instanceof StopProgram)) this.msg('error', e.message, this.curBlock?.line ?? null, this.curBlock?.file ?? 'main', this.curCallLine);
    }
    this.corner.flush();

    const stock = this.stock || autoStock(this.moves, this.machine, this.tools);
    const lineMap = new Map();
    for (let i = 0; i < this.moves.length; i++) {
      const l = this.moves[i].callLine;
      if (l !== null && l !== undefined && !lineMap.has(l)) lineMap.set(l, i);
    }
    const usedTools = [...new Set(this.moves.map((m) => m.tool))].filter((t) => t);
    return {
      machine: this.machine,
      moves: this.moves,
      messages: this.messages,
      tools: this.tools,
      programTools: this.programTools,
      usedTools,
      stock,
      lineMap,
      totalTime: this.time,
      home: this.home,
      stats: {
        toolChanges: this.stats.toolChanges,
        rapidLength: this.stats.rapidLen,
        feedLength: this.stats.feedLen,
        minZ: this.stats.minZ,
        maxZ: this.stats.maxZ,
        blocks: this.blockCount,
        moves: this.moves.length,
        cycleTime: this.time,
      },
      lineCount: this.main.lines.length,
      usesRotary: this.usesRotary,
      comments: this.main.comments,
    };
  }

  callLineOf(f, b) {
    return f.isMain ? b.line : f.callLine;
  }

  // ------------------------------------------------------------------ variables
  getVar(n) {
    if (n === 0) return null;
    if (n >= 1 && n <= 33) return this.locals[this.locals.length - 1].get(n) ?? null;
    if (n >= 5001 && n <= 5003) return this.toProg(this.pos)[n - 5001];
    if (n >= 5021 && n <= 5023) return this.pos[n - 5021];
    if (n === 4001) return this.motion;
    if (n === 4120) return this.tool;
    if (n === 4109) return this.F;
    return this.vars.get(n) ?? null;
  }
  setVar(n, v) {
    if (n === 0) return;
    if (n >= 1 && n <= 33) this.locals[this.locals.length - 1].set(n, v);
    else if (n >= 3000 && n <= 3001) { if (n === 3000) throw new Error(`Macro alarm #3000 = ${v}`); }
    else this.vars.set(n, v);
  }
  ctx() { return { get: (n) => this.getVar(n) }; }
  ev(text) { return evalExpr(text, this.ctx()); }

  execMacro(b, f) {
    const m = b.macro;
    const jump = (target) => {
      const n = Math.round(this.ev(target) ?? 0);
      const idx = f.file.labels.get(n);
      if (idx === undefined) throw new Error(`GOTO ${n}: no block N${n}`);
      f.pc = idx;
    };
    switch (m.type) {
      case 'assign': this.setVar(evalVarIndex(m.target, this.ctx()), this.ev(m.expr)); break;
      case 'goto': jump(m.target); break;
      case 'ifgoto': if (this.ev(m.cond)) jump(m.target); break;
      case 'ifthen':
        if (this.ev(m.cond)) {
          if (m.stmt.type !== 'assign') throw new Error('IF … THEN only supports assignments');
          this.setVar(evalVarIndex(m.stmt.target, this.ctx()), this.ev(m.stmt.expr));
        }
        break;
      case 'while': {
        if (!this.ev(m.cond)) {
          const end = f.file.loopMatch.get(f.pc - 1);
          if (end === undefined) throw new Error('WHILE without END');
          f.pc = end + 1;
        }
        break;
      }
      case 'end': {
        const start = f.file.loopMatch.get(f.pc - 1);
        if (start === undefined) throw new Error('END without WHILE');
        f.pc = start;
        break;
      }
    }
  }

  // ------------------------------------------------------------------ coordinates
  toProg(p) { return [p[0] - this.origin[0], p[1] - this.origin[1], p[2] - this.origin[2]]; }
  toMach(p) { return [p[0] + this.origin[0], p[1] + this.origin[1], p[2] + this.origin[2]]; }
  updateOrigin() {
    const o = this.offsets[this.wcs] || [0, 0, 0];
    this.origin = [o[0] + this.g52[0] + this.g92[0], o[1] + this.g52[1] + this.g92[1], o[2] + this.g52[2] + this.g92[2]];
  }

  // target in program coordinates from axis words
  targetFrom(w, incOverride = null) {
    const cur = this.toProg(this.pos);
    const t = cur.slice();
    const abs = incOverride === null ? this.abs : !incOverride;
    const u = this.units;
    if (this.lathe) {
      const k = this.xScale;
      if (w.X !== undefined) t[0] = w.X * u * k;
      if (w.U !== undefined) t[0] = cur[0] + w.U * u * k;
      if (w.Z !== undefined) t[2] = abs ? w.Z * u : cur[2] + w.Z * u;
      if (w.W !== undefined) t[2] = cur[2] + w.W * u;
      if (!abs && w.X !== undefined) t[0] = cur[0] + w.X * u * k;
      return t;
    }
    if (this.polar && (w.X !== undefined || w.Y !== undefined) && this.plane === 17) {
      // G16: X = radius, Y = angle (deg) around the work origin (G90) or current point (G91)
      const pr = this.polarState || { r: Math.hypot(cur[0], cur[1]), a: Math.atan2(cur[1], cur[0]) * 180 / Math.PI };
      const r = w.X !== undefined ? (abs ? w.X * u : pr.r + w.X * u) : pr.r;
      const a = w.Y !== undefined ? (abs ? w.Y : pr.a + w.Y) : pr.a;
      this.polarState = { r, a };
      const cx = abs ? 0 : cur[0], cy = abs ? 0 : cur[1];
      t[0] = cx + r * Math.cos(a * Math.PI / 180);
      t[1] = cy + r * Math.sin(a * Math.PI / 180);
      if (w.Z !== undefined) t[2] = abs ? w.Z * u : cur[2] + w.Z * u;
      return t;
    }
    for (const [i, L] of [[0, 'X'], [1, 'Y'], [2, 'Z']]) {
      if (w[L] !== undefined) t[i] = abs ? w[L] * u : cur[i] + w[L] * u;
    }
    return t;
  }

  hasAxis(w) {
    return this.lathe
      ? w.X !== undefined || w.U !== undefined || w.Z !== undefined || w.W !== undefined
      : w.X !== undefined || w.Y !== undefined || w.Z !== undefined || w.A !== undefined || w.B !== undefined;
  }

  // ------------------------------------------------------------------ emission
  meta(kind, extra = {}) {
    return {
      line: this.curBlock?.line ?? null,
      file: this.curBlock?.file ?? 'main',
      callLine: this.curCallLine,
      tool: this.tool,
      F: this.F * this.units,
      feedPerRev: this.feedPerRev,
      css: this.css,
      cssSpeed: this.cssSpeed,
      S: this.S,
      maxRpm: this.maxRpm,
      spindle: this.spindle,
      coolant: this.coolant,
      plane: this.plane,
      kind,
      // rotary position: [a start, a end, b start, b end]; a moving block overrides this in doMotion
      rot: this.rotA || this.rotB ? [this.rotA, this.rotA, this.rotB, this.rotB] : null,
      ...extra,
    };
  }

  checkFeedMove() {
    if (this.capture) return;
    const b = this.curBlock;
    if (!this.F && !this.stats.warnedFeed) {
      this.stats.warnedFeed = true;
      this.msg('error', 'Feed move without a feed rate (F) — programme F before G01/G02/G03', b?.line, b?.file, this.curCallLine);
    }
    if (this.spindle === 'off' && !this.stats.warnedSpindle) {
      this.stats.warnedSpindle = true;
      this.msg('warning', 'Feed move with the spindle stopped — M03/M04 missing before cutting', b?.line, b?.file, this.curCallLine);
    }
    if (!this.S && !this.stats.warnedS) {
      this.stats.warnedS = true;
      this.msg('warning', 'No spindle speed (S) programmed before cutting', b?.line, b?.file, this.curCallLine);
    }
  }

  // straight move to machine-coordinate point
  line(kind, to, extra = {}, opt = {}) {
    const a = this.pos.slice();
    const b = to.slice();
    if (this.lathe) b[1] = 0;
    if (kind === 'feed') this.checkFeedMove();
    const prim = { type: 'line', kind, a, b, meta: this.meta(kind, extra) };
    this.pos = b;
    this.emit(prim, opt);
  }

  emit(prim, opt = {}) {
    if (this.capture) { prim.opt = opt; this.capture.push(prim); return; }
    this.corner.push(prim, opt);
  }

  // finish buffered geometry (before tool changes, cycles …)
  flushPath() {
    this.corner.flushPending();
    this.comp.flushChain();
  }

  event(event, dur, extra = {}) {
    if (this.capture) return;
    this.corner.push({ type: 'event', event, dur, a: this.pos.slice(), b: this.pos.slice(), meta: this.meta(event, extra) });
  }

  rpmAt(meta, xRadius) {
    if (!meta.css) return Math.min(meta.S || 0, meta.maxRpm || Infinity);
    const d = Math.max(0.5, 2 * Math.abs(xRadius));
    return Math.min(meta.maxRpm || this.settings.maxRpm, (1000 * meta.cssSpeed) / (Math.PI * d));
  }

  addMove(prim) {
    if (this.moves.length >= this.settings.maxMoves) {
      if (!this.tooMany) { this.tooMany = true; this.msg('error', `Simulation limited to ${this.settings.maxMoves} moves`, prim.meta.line, prim.meta.file, prim.meta.callLine); }
      throw new StopProgram();
    }
    const m = prim.meta;
    if (prim.type === 'event') {
      const t0 = this.time;
      this.time += prim.dur || 0;
      this.moves.push({ kind: prim.event, pts: [prim.a], line: m.line, file: m.file, callLine: m.callLine, tool: m.tool, t0, t1: this.time, len: 0, spindle: m.spindle, coolant: m.coolant, F: m.F, rpm: 0, rate: 0, toTool: m.toTool });
      return;
    }
    let L = primLength(prim);
    const turn = m.rot ? Math.max(Math.abs(m.rot[1] - m.rot[0]), Math.abs(m.rot[3] - m.rot[2])) : 0;
    const rotOnly = L < 1e-7 && turn > 1e-9;
    if (L < 1e-7 && !rotOnly) return;
    const pts = tessellate(prim);
    let rate;
    let rpm = this.rpmAt(m, (prim.a[0] + prim.b[0]) / 2);
    if (prim.kind === 'rapid') rate = this.lathe ? this.settings.rapidLathe : this.settings.rapidMill;
    else {
      rate = m.feedPerRev ? m.F * rpm : m.F;
      if (!(rate > 0)) rate = 100;
    }
    // a pure rotary move: F is degrees/min, rapid turns at ~25 rpm
    let dt = (L / rate) * 60;
    if (rotOnly) {
      dt = prim.kind === 'rapid' ? (turn / 9000) * 60 : (turn / (m.F > 0 ? m.F : 3600)) * 60;
      L = 1e-3;
    }
    const t0 = this.time;
    this.time += dt;
    if (prim.kind === 'rapid') this.stats.rapidLen += L; else this.stats.feedLen += L;
    for (const p of [prim.a, prim.b]) {
      if (p[2] < this.stats.minZ) this.stats.minZ = p[2];
      if (p[2] > this.stats.maxZ) this.stats.maxZ = p[2];
    }
    this.moves.push({
      kind: prim.kind, pts, line: m.line, file: m.file, callLine: m.callLine, tool: m.tool,
      F: m.F, feedPerRev: m.feedPerRev, rate, rpm, spindle: m.spindle, coolant: m.coolant,
      t0, t1: this.time, len: L, cycle: m.cycle || null, center: !!prim.center, plane: m.plane,
      thread: m.thread && prim.kind === 'feed' ? m.F : 0,
      arc: prim.type === 'arc' ? (prim.sweep < 0 ? 'cw' : 'ccw') : null,
      rot: m.rot || null,
      rotOnly,
    });
  }

  // ------------------------------------------------------------------ blocks
  collectWords(b) {
    const G = [], M = [];
    const w = {};
    let corner = null;
    for (const word of b.words) {
      const v = word.expr !== undefined ? this.ev(word.expr) : word.value;
      if (word.letter === 'G') { if (v === null) throw new Error('G with vacant variable'); G.push(Math.round(v * 10) / 10); continue; }
      if (word.letter === 'M') { if (v === null) throw new Error('M with vacant variable'); M.push(Math.round(v)); continue; }
      if (word.letter === ',R') { corner = { R: v }; continue; }
      if (word.letter === ',C') { corner = { C: v }; continue; }
      if (w[word.letter] !== undefined) throw new Error(`Word ${word.letter} appears twice in the block`);
      if (v === null) continue; // vacant variable: word treated as not commanded
      w[word.letter] = v;
    }
    return { G, M, w, corner };
  }

  execBlock(b, f, frames) {
    const { G, M, w, corner } = this.collectWords(b);
    let motionCmd = null;   // G0/1/2/3/32 in this block
    let cycleCmd = null;    // canned cycle code in this block
    let nonModal = null;    // G4 G10 G28 G30 G52 G53 G92 G50 G65
    let compCmd = null;

    for (const g of G) {
      switch (g) {
        case 0: case 1: case 2: case 3: motionCmd = g; break;
        case 32: case 33: if (this.lathe) motionCmd = 32; else this.unsupported(b, g); break;
        case 4: nonModal = 4; break;
        case 9: case 61: case 64: case 8: case 5.1: case 5: break;
        case 10: nonModal = 10; break;
        case 15: this.polar = false; this.polarState = null; break;
        case 16: if (this.lathe) this.unsupported(b, g); else { this.polar = true; this.polarState = null; } break;
        case 17: case 18: case 19: this.plane = g; break;
        case 20: this.units = 25.4; break;
        case 21: this.units = 1; break;
        case 28: case 30: nonModal = 28; break;
        case 31: motionCmd = 1; break;
        case 40: compCmd = 0; break;
        case 41: compCmd = 1; break;
        case 42: compCmd = -1; break;
        case 43: case 44: case 43.4: this.lengthComp = true; break;
        case 49: this.lengthComp = false; break;
        case 50: if (this.lathe) nonModal = 50; break; // mill: scaling cancel
        case 50.1: case 51.1: break;
        case 51: this.warnOnceGlobal('g51', 'warning', 'G51 scaling/mirroring is not simulated', b); break;
        case 52: nonModal = 52; break;
        case 53: nonModal = 53; break;
        case 54: case 55: case 56: case 57: case 58: case 59: this.wcs = g; this.updateOrigin(); break;
        case 54.1: this.wcs = 54; this.updateOrigin(); break;
        case 65: nonModal = 65; break;
        case 68: case 69: if (g === 68) this.warnOnceGlobal('g68', 'warning', 'G68 coordinate rotation is not simulated', b); break;
        case 80: this.cycle = null; this.latheCycle = null; break;
        case 90:
          if (this.lathe) cycleCmd = 90; else this.abs = true;
          break;
        case 91: if (this.lathe) this.unsupported(b, g); else this.abs = false; break;
        case 92: if (this.lathe) cycleCmd = 92; else nonModal = 92; break;
        case 94: if (this.lathe) cycleCmd = 94; else this.feedPerRev = false; break;
        case 95: if (!this.lathe) this.feedPerRev = true; else this.unsupported(b, g); break;
        case 96: this.css = true; break;
        case 97: this.css = false; break;
        case 98: if (this.lathe) this.feedPerRev = false; else this.retLevel = 98; break;
        case 99: if (this.lathe) this.feedPerRev = true; else this.retLevel = 99; break;
        default:
          if (this.lathe && g >= 70 && g <= 76) cycleCmd = g;
          else if (!this.lathe && MILL_CYCLES.has(g)) cycleCmd = g;
          else this.unsupported(b, g);
      }
    }

    // F S T
    if (w.F !== undefined) this.F = w.F;
    if (w.S !== undefined && nonModal !== 50) {
      if (this.css) { this.cssSpeed = w.S; this.S = w.S; } else this.S = w.S;
    }
    if (w.S !== undefined && this.css && !this.lathe) this.cssSpeed = w.S;
    if (w.T !== undefined) this.selectTool(w.T, M.includes(6));
    if (w.D !== undefined && compCmd === null && this.compSide) this.compD = w.D;
    this.rotTarget = null;
    if (this.lathe && (w.A !== undefined || w.B !== undefined || w.C !== undefined)) {
      if (!(w.C !== undefined && cycleCmd)) this.warnOnceGlobal('rot', 'warning', 'A/B/C words are ignored on the lathe (no live tooling / C axis in the simulation)', b);
    } else if (!this.lathe) {
      if (w.C !== undefined) this.warnOnceGlobal('rotc', 'warning', 'C axis words are ignored — A (about X) and B (about Y) rotary axes are simulated', b);
      if (w.A !== undefined || w.B !== undefined) {
        const a = w.A === undefined ? this.rotA : this.abs ? w.A : this.rotA + w.A;
        const bb = w.B === undefined ? this.rotB : this.abs ? w.B : this.rotB + w.B;
        this.rotTarget = { a, b: bb };
        this.usesRotary = true;
      }
    }

    // M codes before motion
    for (const m of M) {
      switch (m) {
        case 3: case 13: this.setSpindle('cw'); if (m === 13) this.coolant = true; break;
        case 4: case 14: this.setSpindle('ccw'); if (m === 14) this.coolant = true; break;
        case 6: this.toolChange(); break;
        case 7: case 8: this.coolant = true; break;
        default: break;
      }
    }

    if (compCmd !== null) {
      this.pendingComp = compCmd;
      if (w.D !== undefined) this.compD = w.D;
    }

    // ---- non-modal
    if (nonModal === 4) {
      const secs = w.X !== undefined ? w.X : w.U !== undefined ? w.U : w.P !== undefined ? w.P / 1000 : 0;
      this.event('dwell', secs);
    } else if (nonModal === 10) {
      this.g10(w, b);
    } else if (nonModal === 28) {
      this.g28(w, b);
    } else if (nonModal === 53) {
      this.g53(w);
    } else if (nonModal === 52) {
      const u = this.units;
      if (w.X !== undefined) this.g52[0] = this.lathe ? w.X * u * this.xScale : w.X * u;
      if (w.Y !== undefined) this.g52[1] = w.Y * u;
      if (w.Z !== undefined) this.g52[2] = w.Z * u;
      this.updateOrigin();
    } else if (nonModal === 92 || (nonModal === 50 && this.hasAxis(w))) {
      // set current position: origin such that pos = given
      const u = this.units;
      const cur = this.toProg(this.pos);
      if (this.lathe) {
        if (w.X !== undefined) this.g92[0] += cur[0] - w.X * u * this.xScale;
        if (w.Z !== undefined) this.g92[2] += cur[2] - w.Z * u;
      } else {
        if (w.X !== undefined) this.g92[0] += cur[0] - w.X * u;
        if (w.Y !== undefined) this.g92[1] += cur[1] - w.Y * u;
        if (w.Z !== undefined) this.g92[2] += cur[2] - w.Z * u;
      }
      this.updateOrigin();
      if (nonModal === 50 && w.S !== undefined) this.maxRpm = w.S;
    } else if (nonModal === 50) {
      if (w.S !== undefined) this.maxRpm = w.S;
    } else if (nonModal === 65) {
      this.macroCall(w, frames, b);
      return;
    }

    // ---- motion / cycles (axis words of G4/G10/G28/G50/G52/G53/G92 are parameters, not a move)
    const nonModalTookAxes = nonModal !== null && nonModal !== 65;
    if (motionCmd !== null) {
      this.motion = motionCmd;
      this.cycle = null;
      this.latheCycle = null;
    }
    if (!nonModalTookAxes) {
      if (this.lathe) this.latheMotion(b, w, cycleCmd, motionCmd, corner, f);
      else this.millMotion(b, w, cycleCmd, motionCmd, corner);
    }

    // ---- M codes after motion
    for (const m of M) {
      switch (m) {
        case 0: case 1: this.event('stop', 0, { note: m === 0 ? 'M00 program stop' : 'M01 optional stop' }); break;
        case 5: this.setSpindle('off'); break;
        case 9: this.coolant = false; break;
        case 2: case 30:
          this.corner.flush();
          this.ended = true;
          this.setSpindle('off');
          return;
        case 98: this.callSub(w, frames, b, false); break;
        case 97: this.callSub(w, frames, b, true); break;
        case 99: this.returnFromSub(frames, w.P); break;
        case 3: case 4: case 6: case 7: case 8: case 13: case 14: break;
        case 19: case 29: case 41: case 42: case 43: case 44: case 48: case 49: case 10: case 11: case 21: case 22: case 23: case 24: break;
        default:
          this.warnOnceGlobal('m' + m, 'info', `M${String(m).padStart(2, '0')} has no effect in the simulation`, b);
      }
    }
  }

  unsupported(b, g) {
    this.msg('error', `G${g} is not supported`, b.line, b.file, this.curCallLine);
  }

  setSpindle(dir) {
    if (this.spindle === dir) return;
    this.spindle = dir;
    if (!this.capture) this.event('spindle', 0);
  }

  selectTool(T, withM6) {
    let n = Math.round(T);
    if (this.lathe) {
      if (n >= 100) n = Math.floor(n / 100);
      this.pendingTool = n;
      if (n !== 0) this.toolChange();
      return;
    }
    this.pendingTool = n;
    if (!this.hasM6 && !withM6) this.toolChange();
  }

  toolChange() {
    const n = this.pendingTool;
    if (!n || n === this.tool) return;
    this.flushPath();
    if (this.compSide) this.msg('warning', 'Tool change while cutter compensation is active', this.curBlock?.line, this.curBlock?.file, this.curCallLine);
    this.tool = n;
    if (!this.tools[n]) {
      this.tools[n] = normalizeTool({ num: n, source: 'default', name: `T${n} (not defined)` }, this.machine);
      this.msg('warning', `T${n} is not in the tool table — using a default ${this.lathe ? 'turning tool' : 'Ø10 end mill'}`, this.curBlock?.line, this.curBlock?.file, this.curCallLine);
    }
    this.stats.toolChanges++;
    this.event('toolchange', this.settings.toolChangeTime, { toTool: n });
  }

  compRadius() {
    const d = this.compD || this.tool;
    const t = this.tools[d] || this.tools[this.tool];
    if (!t) return 0;
    return this.lathe ? t.noseRadius || 0 : (t.diameter || 0) / 2;
  }

  // ------------------------------------------------------------------ mill motion
  millMotion(b, w, cycleCmd, motionCmd, corner) {
    const hasAxis = this.hasAxis(w);
    if (cycleCmd !== null) {
      this.startMillCycle(cycleCmd, w);
      if (this.cycle) this.runMillCycle(b, w);
      return;
    }
    if (this.cycle && motionCmd === null) {
      if (hasAxis || w.R !== undefined) {
        this.updateMillCycle(w);
        this.runMillCycle(b, w);
      }
      return;
    }
    const hasArcWords = w.I !== undefined || w.J !== undefined || w.K !== undefined || w.R !== undefined;
    if (!hasAxis && !(this.motion >= 2 && this.motion <= 3 && hasArcWords)) {
      if (this.pendingComp !== undefined && this.pendingComp !== null && this.pendingComp !== 0) {
        // G41/G42 on a block without motion: activates with the next move
      }
      return;
    }
    this.doMotion(b, w, corner);
  }

  compOptions(target) {
    const opt = {};
    if (this.pendingComp !== undefined && this.pendingComp !== null) {
      const pc = this.pendingComp;
      this.pendingComp = null;
      if (pc !== 0 && !this.compSide) {
        this.compSide = pc;
        opt.start = true; opt.side = pc; opt.radius = this.compRadius(); opt.plane = this.plane;
        if (this.lathe ? this.plane !== 18 : this.plane !== 17) this.msg('warning', 'Cutter compensation is only simulated in the G17 (mill) / G18 (lathe) plane', this.curBlock.line, this.curBlock.file, this.curCallLine);
        if (!opt.radius) this.msg('warning', `Cutter compensation with zero radius (check D${this.compD || this.tool})`, this.curBlock.line, this.curBlock.file, this.curCallLine);
      } else if (pc === 0 && this.compSide) {
        this.compSide = 0;
        opt.cancel = true;
      } else if (pc !== 0 && this.compSide && pc !== this.compSide) {
        this.compSide = pc;
        opt.side = pc; opt.radius = this.compRadius();
      }
    }
    return opt;
  }

  doMotion(b, w, corner) {
    const target = this.toMach(this.targetFrom(w));
    const opt = this.compOptions(target);
    if (corner) opt.corner = corner;
    let mode = this.motion;
    if (![0, 1, 2, 3, 32].includes(mode)) mode = 1;
    if (mode === 0 || mode === 1 || mode === 32) {
      const kind = mode === 0 ? 'rapid' : 'feed';
      if (kind === 'feed') this.checkFeedMove();
      const extra = mode === 32 ? { cycle: 'G32 thread', feedPerRev: true, thread: true } : {};
      if (this.rotTarget) {
        extra.rot = [this.rotA, this.rotTarget.a, this.rotB, this.rotTarget.b];
        this.rotA = this.rotTarget.a;
        this.rotB = this.rotTarget.b;
        this.rotTarget = null;
      }
      const prim = { type: 'line', kind, a: this.pos.slice(), b: target, meta: this.meta(kind, extra) };
      if (this.lathe) prim.b[1] = 0;
      this.pos = prim.b.slice();
      this.emit(prim, opt);
      return;
    }
    // arcs
    this.checkFeedMove();
    const cw = mode === 2;
    const plane = this.plane;
    const a = this.pos.slice();
    const bpt = target;
    let centerUV;
    if (w.R !== undefined) {
      const { center, error } = centerFromRadius(a, bpt, w.R * this.units, cw, plane);
      if (error) this.msg('error', error, b.line, b.file, this.curCallLine);
      if (!center) { this.line('feed', bpt); return; }
      centerUV = center;
    } else if (w.I !== undefined || w.J !== undefined || w.K !== undefined) {
      const u = this.units;
      const off = [(w.I || 0) * u, (w.J || 0) * u, (w.K || 0) * u];
      const c3 = [a[0] + off[0], a[1] + off[1], a[2] + off[2]];
      centerUV = uv(c3, plane);
    } else {
      this.msg('error', `G0${mode} without I/J/K or R`, b.line, b.file, this.curCallLine);
      this.line('feed', bpt);
      return;
    }
    const { prim, error } = buildArc(a, bpt, centerUV, cw, plane, 'feed', this.meta('feed'), this.settings.arcTolerance);
    if (error) this.msg('error', error, b.line, b.file, this.curCallLine);
    this.pos = bpt.slice();
    if (!prim) { this.emit({ type: 'line', kind: 'feed', a, b: bpt, meta: this.meta('feed') }, opt); return; }
    this.emit(prim, opt);
  }

  // ------------------------------------------------------------------ mill canned cycles
  startMillCycle(code, w) {
    if (this.compSide) this.msg('warning', 'Canned cycle while cutter compensation is active', this.curBlock.line, this.curBlock.file, this.curCallLine);
    const prog = this.toProg(this.pos);
    const prev = this.cycle;
    this.cycle = {
      code,
      initialZ: prev ? prev.initialZ : prog[2],
      R: prev ? prev.R : null,
      Z: prev ? prev.Z : null,
      Q: prev ? prev.Q : 0,
      P: prev ? prev.P : 0,
      absAtStart: this.abs,
    };
    this.updateMillCycle(w);
  }

  updateMillCycle(w) {
    const c = this.cycle;
    const u = this.units;
    const prog = this.toProg(this.pos);
    if (w.R !== undefined) c.R = this.abs ? w.R * u : c.initialZ + w.R * u;
    if (c.R === null) c.R = prog[2];
    if (w.Z !== undefined) c.Z = this.abs ? w.Z * u : c.R + w.Z * u;
    if (w.Q !== undefined) c.Q = Math.abs(w.Q * u);
    if (w.P !== undefined) c.P = w.P;
  }

  runMillCycle(b, w) {
    const c = this.cycle;
    if (c.Z === null) {
      this.msg('error', `${CYCLE_NAMES[c.code]}: no Z depth programmed`, b.line, b.file, this.curCallLine);
      return;
    }
    const reps = w.K !== undefined ? Math.round(w.K) : w.L !== undefined ? Math.round(w.L) : 1;
    const name = CYCLE_NAMES[c.code];
    for (let k = 0; k < reps; k++) {
      // XY position (incremental words step every repeat)
      const prog = this.toProg(this.pos);
      const xy = this.targetFrom({ X: w.X, Y: w.Y });
      if (k > 0 && this.abs && (w.X !== undefined || w.Y !== undefined)) {
        // absolute + K repeats the same hole: allowed but pointless
      }
      const xyM = this.toMach([xy[0], xy[1], prog[2]]);
      if (Math.hypot(xyM[0] - this.pos[0], xyM[1] - this.pos[1]) > 1e-9) this.line('rapid', xyM, { cycle: name });
      this.drillHole(c, name);
    }
  }

  drillHole(c, name) {
    const at = (z) => { const p = this.toProg(this.pos); return this.toMach([p[0], p[1], z]); };
    const cur = this.toProg(this.pos)[2];
    const R = c.R, Z = c.Z, init = c.initialZ;
    const extra = { cycle: name };
    if (Z > R + 1e-9) this.msg('warning', `${name}: Z${Z} is above the R plane R${R}`, this.curBlock.line, this.curBlock.file, this.curCallLine);
    if (cur > R + 1e-9) this.line('rapid', at(R), extra);
    else if (cur < R - 1e-9) this.line('rapid', at(R), extra);
    const dwell = () => { if (c.P) this.event('dwell', c.P / 1000, extra); };
    const back = this.retLevel === 98 ? Math.max(init, R) : R;
    const clr = this.settings.peckClearance;
    switch (c.code) {
      case 73: case 83: {
        const q = c.Q > 0 ? c.Q : R - Z;
        if (!(c.Q > 0)) this.msg('warning', `${name}: no Q peck depth — drilling in one pass`, this.curBlock.line, this.curBlock.file, this.curCallLine);
        let depth = R;
        let guard = 0;
        while (depth > Z + 1e-9 && guard++ < 1000) {
          const next = Math.max(Z, depth - q);
          // G83 comes back down to just above the previous depth; G73 only backed off by the clearance
          if (depth < R - 1e-9) this.line('rapid', at(depth + clr), extra);
          this.line('feed', at(next), extra);
          depth = next;
          if (depth > Z + 1e-9) this.line('rapid', at(c.code === 83 ? R : depth + clr), extra);
        }
        dwell();
        this.line('rapid', at(back), extra);
        break;
      }
      case 74: case 84: {
        this.line('feed', at(Z), extra);
        dwell();
        const spin = this.spindle;
        this.spindle = spin === 'cw' ? 'ccw' : 'cw';
        this.line('feed', at(R), extra);
        this.spindle = spin;
        if (back > R) this.line('rapid', at(back), extra);
        break;
      }
      case 85: case 89: case 88: {
        this.line('feed', at(Z), extra);
        if (c.code !== 85) dwell();
        this.line('feed', at(R), extra);
        if (back > R) this.line('rapid', at(back), extra);
        break;
      }
      case 76: case 86: case 81: case 82: default: {
        this.line('feed', at(Z), extra);
        if (c.code === 82 || c.code === 76) dwell();
        this.line('rapid', at(back), extra);
      }
    }
  }

  // ------------------------------------------------------------------ lathe motion
  latheMotion(b, w, cycleCmd, motionCmd, corner, f) {
    if (cycleCmd !== null && cycleCmd >= 70 && cycleCmd <= 76) {
      this.flushPath();
      this.latheMultiCycle(cycleCmd, b, w, f);
      return;
    }
    if (cycleCmd === 90 || cycleCmd === 92 || cycleCmd === 94) {
      if (motionCmd !== null) {
        // e.g. "G0 G90 X50 Z2" from a Group-B post: G90 can only mean "absolute" here
        this.warnOnceGlobal('g90b', 'info', `G${cycleCmd} together with G0${motionCmd} — treated as a Group B absolute code, not a cycle`, b);
      } else if (this.hasAxis(w)) {
        this.latheCycle = { code: cycleCmd, R: 0 };
        this.motion = cycleCmd;
      } else {
        return; // bare "G90" in a safety line
      }
    }
    if (this.latheCycle && motionCmd === null) {
      if (!this.hasAxis(w)) return;
      if (w.R !== undefined) this.latheCycle.R = w.R * this.units;
      this.flushPath();
      const start = this.pos.slice();
      const target = this.toMach(this.targetFrom(w));
      const api = this.cycleApi();
      if (this.latheCycle.code === 92) latheThreadCycle(api, start, target, this.latheCycle.R);
      else latheBoxCycle(api, this.latheCycle.code, start, target, this.latheCycle.R);
      return;
    }
    const hasArcWords = w.I !== undefined || w.K !== undefined || w.R !== undefined;
    if (!this.hasAxis(w) && !(this.motion >= 2 && this.motion <= 3 && hasArcWords)) return;
    this.doMotion(b, w, corner);
  }

  cycleApi() {
    return {
      rapid: (r, z, extra) => this.line('rapid', [r, 0, z], extra),
      feed: (r, z, extra) => this.line('feed', [r, 0, z], extra),
      pos: () => this.pos.slice(),
      emitPrims: (prims, shift, extra, kindOverride) => this.emitShifted(prims, shift, extra, kindOverride),
      warn: (text) => this.msg('warning', text, this.curBlock.line, this.curBlock.file, this.curCallLine),
      error: (text) => this.msg('error', text, this.curBlock.line, this.curBlock.file, this.curCallLine),
      setFeed: (F) => { const old = this.F; this.F = F; return old; },
      F: () => this.F,
    };
  }

  emitShifted(prims, shift, extra = {}, kindOverride) {
    // replay captured profile prims translated by shift = [dr, dz]
    // extra.profileLines: highlight the profile blocks (G70); extra.withComp: apply G41/G42 found in the profile
    const s = [shift[0], 0, shift[1]];
    const { profileLines, withComp, keepCycleFeed, ...metaExtra } = extra;
    for (const p of prims) {
      const kind = kindOverride || p.kind;
      const a = this.pos.slice();
      const b = [p.b[0] + s[0], 0, p.b[2] + s[2]];
      if (kind === 'feed') this.checkFeedMove();
      const meta = { ...this.meta(kind, metaExtra) };
      if (p.meta && p.meta.Fset && !keepCycleFeed) meta.F = p.meta.F * this.units;
      if (profileLines && p.meta) { meta.line = p.meta.line; meta.file = p.meta.file; meta.callLine = p.meta.callLine; }
      const opt = {};
      if (withComp && p.opt) {
        if (p.opt.start) { opt.start = true; opt.side = p.opt.side; opt.radius = this.compRadius(); opt.plane = 18; }
        if (p.opt.cancel) opt.cancel = true;
      }
      this.pos = b;
      if (p.type === 'arc') {
        const c = [p.c[0] + s[0], 0, p.c[2] + s[2]];
        const { prim } = buildArc(a, b, uv(c, 18), p.sweep < 0, 18, kind, meta, 0.05);
        if (prim) { this.corner.push(prim, opt); continue; }
      }
      this.corner.push({ type: 'line', kind, a, b, meta }, opt);
    }
    if (withComp) this.compSide = this.comp.side;
  }

  // run blocks N(P)..N(Q) collecting nominal geometry
  captureProfile(f, P, Q, b) {
    const pi = f.file.labels.get(P), qi = f.file.labels.get(Q);
    if (pi === undefined || qi === undefined || qi < pi) {
      this.msg('error', `Profile blocks N${P}…N${Q} not found`, b.line, b.file, this.curCallLine);
      return null;
    }
    const saved = {
      pos: this.pos.slice(), motion: this.motion, F: this.F, S: this.S, plane: this.plane, abs: this.abs,
      spindle: this.spindle, latheCycle: this.latheCycle, curBlock: this.curBlock, pendingComp: this.pendingComp, compSide: this.compSide,
      css: this.css, cssSpeed: this.cssSpeed, feedPerRev: this.feedPerRev,
    };
    this.capture = [];
    let finishF = null;
    for (let i = pi; i <= qi; i++) {
      const blk = f.file.blocks[i];
      if (blk.empty || blk.error || blk.macro) continue;
      this.curBlock = blk;
      const { G, w } = this.collectWords(blk);
      if (w.F !== undefined) finishF = w.F;
      let motion = null;
      for (const g of G) {
        if ([0, 1, 2, 3].includes(g)) motion = g;
        else if (g === 40) this.pendingComp = 0;
        else if (g === 41) this.pendingComp = 1;
        else if (g === 42) this.pendingComp = -1;
      }
      if (w.D !== undefined) this.compD = w.D;
      if (motion !== null) this.motion = motion;
      if (this.motion > 3) this.motion = 1;
      if (this.hasAxis(w) || (this.motion >= 2 && (w.I !== undefined || w.K !== undefined || w.R !== undefined))) {
        const before = this.capture.length;
        this.doMotion(blk, w, null);
        for (let k = before; k < this.capture.length; k++) {
          this.capture[k].meta.Fset = finishF !== null;
          if (finishF !== null) this.capture[k].meta.F = finishF;
        }
      }
    }
    const prims = this.capture;
    this.capture = null;
    Object.assign(this, saved);
    this.pendingComp = saved.pendingComp;
    return { prims, finishF, qIndex: qi };
  }

  latheMultiCycle(code, b, w, f) {
    const u = this.units;
    const api = this.cycleApi();
    const start = this.pos.slice();
    const hasPQ = w.P !== undefined && w.Q !== undefined;
    const name = { 70: 'G70 finish', 71: 'G71 rough turn', 72: 'G72 rough face', 73: 'G73 pattern repeat', 74: 'G74 face peck', 75: 'G75 groove', 76: 'G76 thread' }[code];
    const extra = { cycle: name };
    if (code === 70) {
      if (!hasPQ) { this.msg('error', 'G70 needs P and Q', b.line, b.file, this.curCallLine); return; }
      const prof = this.captureProfile(f, w.P, w.Q, b);
      if (!prof) return;
      latheFinish(api, start, prof.prims, extra);
      return;
    }
    if (code === 71 || code === 72 || code === 73) {
      if (!hasPQ) {
        // first block: depth/relief + retract (or divisions)
        this.cycleParams[code] = { U: w.U, W: w.W, R: w.R, D: w.D };
        return;
      }
      const first = this.cycleParams[code] || {};
      const prof = this.captureProfile(f, w.P, w.Q, b);
      if (!prof) return;
      const du = (w.U ?? 0) * u * this.xScale, dw = (w.W ?? 0) * u;
      if (code === 73) {
        const relief = [(first.U ?? 0) * u * this.xScale, (first.W ?? 0) * u];
        const div = Math.max(1, Math.round(first.R ?? 1));
        latheRoughing(api, 73, start, prof.prims, { du, dw, relief, div, extra });
      } else {
        let depth = code === 71 ? first.U : first.W;
        if (depth === undefined && w.D !== undefined) depth = w.D > 50 ? w.D / 1000 : w.D;
        if (depth === undefined && first.D !== undefined) depth = first.D > 50 ? first.D / 1000 : first.D;
        depth = Math.abs((depth ?? 1) * u);
        const retract = Math.abs((first.R ?? 0.5) * u);
        if (!(depth > 0)) { this.msg('error', `${name}: depth of cut must be > 0`, b.line, b.file, this.curCallLine); return; }
        latheRoughing(api, code, start, prof.prims, { du, dw, depth, retract, extra });
      }
      // continue after the Q block
      f.pc = prof.qIndex + 1;
      return;
    }
    if (code === 74 || code === 75) {
      if (!this.hasAxis(w)) { this.cycleParams[code] = { R: w.R }; return; }
      const e = Math.abs((this.cycleParams[code]?.R ?? 0.5) * u);
      const target = this.toMach(this.targetFrom(w));
      lathePeck(api, code, start, target, { e, di: (w.P ?? 0) / 1000 * u, dk: (w.Q ?? 0) / 1000 * u, extra });
      return;
    }
    if (code === 76) {
      if (!this.hasAxis(w)) { this.cycleParams[76] = { P: w.P, Q: w.Q, R: w.R }; return; }
      const p1 = this.cycleParams[76] || {};
      const target = this.toMach(this.targetFrom(w));
      latheG76(api, start, target, {
        finishPasses: p1.P !== undefined ? Math.floor(p1.P / 10000) || 1 : 1,
        angle: p1.P !== undefined ? p1.P % 100 : 60,
        minDepth: (p1.Q ?? 50) / 1000 * u,
        finishAllow: (p1.R ?? 0.05) * u,
        taper: (w.R ?? 0) * u,
        height: (w.P ?? 1000) / 1000 * u,
        firstDepth: (w.Q ?? 300) / 1000 * u,
        lead: w.F !== undefined ? w.F * u : this.F * u,
        extra,
      });
    }
  }

  // ------------------------------------------------------------------ G28 / G53 / G10
  g28(w, b) {
    this.usedHome = true;
    this.corner.flush();
    const axes = this.lathe ? [['X', 'U', 0], ['Z', 'W', 2]] : [['X', null, 0], ['Y', null, 1], ['Z', null, 2]];
    const given = axes.filter(([a, i]) => w[a] !== undefined || (i && w[i] !== undefined));
    if (!given.length) return;
    const mid = this.toMach(this.targetFrom(w));
    const midMove = this.pos.slice();
    for (const [, , k] of given) midMove[k] = mid[k];
    if (Math.hypot(midMove[0] - this.pos[0], midMove[1] - this.pos[1], midMove[2] - this.pos[2]) > 1e-9) this.line('rapid', midMove, { cycle: 'G28 intermediate' });
    const home = this.pos.slice();
    for (const [, , k] of given) home[k] = this.home[k];
    // Z first on a mill so the tool lifts before travelling
    if (!this.lathe && given.some(([, , k]) => k === 2) && given.length > 1) {
      const up = this.pos.slice(); up[2] = this.home[2];
      this.line('rapid', up, { cycle: 'G28 home' });
    }
    this.line('rapid', home, { cycle: 'G28 home' });
  }

  g53(w) {
    // machine coordinates: machine zero is the home position in this simulator
    this.usedHome = true;
    const t = this.pos.slice();
    const u = this.units;
    if (this.lathe) {
      if (w.X !== undefined) t[0] = this.home[0] + w.X * u * this.xScale;
      if (w.Z !== undefined) t[2] = this.home[2] + w.Z * u;
    } else {
      if (w.X !== undefined) t[0] = this.home[0] + w.X * u;
      if (w.Y !== undefined) t[1] = this.home[1] + w.Y * u;
      if (w.Z !== undefined) t[2] = this.home[2] + w.Z * u;
    }
    this.line(this.motion === 1 ? 'feed' : 'rapid', t, { cycle: 'G53 machine' });
  }

  g10(w, b) {
    const L = w.L ?? 2;
    if (L !== 2 && L !== 20) { this.msg('info', `G10 L${L} ignored`, b.line, b.file, this.curCallLine); return; }
    const p = Math.round(w.P ?? 1);
    const g = L === 2 ? (p === 0 ? null : 53 + p) : 54;
    if (!g || !this.offsets[g]) return;
    const o = this.offsets[g];
    const u = this.units;
    if (w.X !== undefined) o[0] = this.lathe ? w.X * u * this.xScale : w.X * u;
    if (w.Y !== undefined) o[1] = w.Y * u;
    if (w.Z !== undefined) o[2] = w.Z * u;
    this.updateOrigin();
  }

  // ------------------------------------------------------------------ subprograms
  findProgram(num, f) {
    if (f.file.programs.has(num)) return { file: f.file, idx: f.file.programs.get(num) + 1 };
    if (this.main.programs.has(num)) return { file: this.main, idx: this.main.programs.get(num) + 1 };
    const ext = this.extPrograms.get(num);
    if (ext) {
      const startIdx = ext.file.programs.has(num) ? ext.file.programs.get(num) + 1 : 0;
      return { file: ext.file, idx: startIdx };
    }
    return null;
  }

  callSub(w, frames, b, local) {
    if (w.P === undefined) { this.msg('error', `M${local ? 97 : 98} without P`, b.line, b.file, this.curCallLine); return; }
    const f = frames[frames.length - 1];
    let num = Math.round(w.P);
    let reps = 1;
    if (!local) {
      const s = String(num);
      if (s.length > 4 && w.L === undefined) { reps = parseInt(s.slice(0, -4), 10) || 1; num = parseInt(s.slice(-4), 10); }
    }
    if (w.L !== undefined) reps = Math.round(w.L);
    if (reps < 1) return;
    if (frames.length > 10) { this.msg('error', 'Subprogram nesting too deep (max 10)', b.line, b.file, this.curCallLine); return; }
    let target;
    if (local) {
      const idx = f.file.labels.get(num);
      if (idx === undefined) { this.msg('error', `M97 P${num}: no block N${num}`, b.line, b.file, this.curCallLine); return; }
      target = { file: f.file, idx };
    } else {
      target = this.findProgram(num, f);
      if (!target) {
        this.msg('error', `Subprogram O${String(num).padStart(4, '0')} not found — add it to the program or load it under Subprograms`, b.line, b.file, this.curCallLine);
        return;
      }
    }
    frames.push({ file: target.file, pc: target.idx, start: target.idx, repeat: reps, callLine: this.curCallLine, isMain: false, returnTo: f, macro: false });
  }

  macroCall(w, frames, b) {
    if (w.P === undefined) { this.msg('error', 'G65 without P', b.line, b.file, this.curCallLine); return; }
    const num = Math.round(w.P);
    const f = frames[frames.length - 1];
    const target = this.findProgram(num, f);
    if (!target) { this.msg('error', `G65 P${num}: macro program not found`, b.line, b.file, this.curCallLine); return; }
    const loc = new Map();
    for (const [letter, v] of Object.entries(w)) if (ARG_VARS[letter] && letter !== 'P' && letter !== 'L') loc.set(ARG_VARS[letter], v);
    this.locals.push(loc);
    frames.push({ file: target.file, pc: target.idx, start: target.idx, repeat: Math.max(1, Math.round(w.L ?? 1)), callLine: this.curCallLine, isMain: false, macro: true });
  }

  returnFromSub(frames, P) {
    const f = frames[frames.length - 1];
    if (f.isMain) {
      if (P !== undefined) {
        const idx = f.file.labels.get(Math.round(P));
        if (idx !== undefined) { f.pc = idx; return; }
      }
      this.msg('warning', 'M99 in the main program would restart it forever — simulation stopped here', this.curBlock?.line, 'main', this.curCallLine);
      this.ended = true;
      return;
    }
    if (f.repeat > 1) { f.repeat--; f.pc = f.start; return; }
    frames.pop();
    if (f.macro) this.locals.pop();
    if (P !== undefined) {
      const caller = frames[frames.length - 1];
      const idx = caller.file.labels.get(Math.round(P));
      if (idx !== undefined) caller.pc = idx;
    }
  }
}

export { PLANES };
