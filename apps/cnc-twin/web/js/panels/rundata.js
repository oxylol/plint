/* Everything the run page reads about one run, loaded once and indexed for lookups by time:
 * the run record (run.json + status.json), timeline (blocks by path, time per program line),
 * findings (by line), channels, keyframes, the file list, the scene, program texts, and the
 * frames and channel values the DRO shows at a time. */

import { bisectRight, nearestIndex } from '../plot/scale.js';
import { decodeArrayDoc } from '../api.js';

const SEVERITY_RANK = { info: 0, warning: 1, error: 2, fatal: 3 };
export { SEVERITY_RANK };

export function lineKey(program, line) {
  return `${program || ''}\u0001${line || 0}`;
}

export class RunData extends EventTarget {
  constructor(api, runId) {
    super();
    this.api = api;
    this.id = runId;
    this.record = null;
    this.timeline = { tickS: 1e-6, blocks: [], paths: new Map(), lines: new Map(), ops: [] };
    this.findings = [];
    this.findingsByLine = new Map();
    this.channels = [];
    this.channelsByName = new Map();
    this.keyframes = [];
    this.files = new Map();
    this.scene = null;
    this.programs = new Map();
    this.errors = {};
  }

  get run() { return this.record && this.record.run; }
  get status() { return (this.record && this.record.status) || {}; }
  get state() { return this.status.status || (this.run ? 'done' : 'unknown'); }
  get finished() { return !!this.run; }
  get active() { return ['queued', 'running', 'cancelling'].includes(this.state); }
  get project() { return (this.run && this.run.project) || {}; }
  get projectId() { return (this.record && this.record.job && this.record.job.project_id) || (this.run && this.run.project_id) || null; }
  get tickS() { return this.timeline.tickS; }

  async load() {
    this.record = await this.api.run(this.id);
    await this.loadFiles();
    await Promise.all([this.loadTimeline(), this.loadFindings(), this.loadChannels(), this.loadKeyframes(), this.loadScene()]);
    return this;
  }

  async _try(name, fn) {
    try { await fn(); delete this.errors[name]; } catch (error) { this.errors[name] = error; }
  }

  /** Files a running run has not written yet are not asked for (no 404 in the console). */
  loadScene() {
    if (!this.has('scene.json')) return Promise.resolve();
    return this._try('scene', async () => { this.scene = await this.api.scene(this.id); });
  }

  loadFiles() {
    return this._try('files', async () => {
      const list = await this.api.files(this.id);
      this.files = new Map((list.files || []).map((f) => [f.path, f.bytes]));
    });
  }

  has(path) { return this.files.has(path); }

  loadChannels() {
    if (!this.has('channels.json')) return Promise.resolve();
    return this._try('channels', async () => {
      const doc = await this.api.channels(this.id);
      this.channels = (doc.channels || []).slice().sort((a, b) => String(a.name).localeCompare(String(b.name)));
      this.channelsByName = new Map(this.channels.map((c) => [c.name, c]));
    });
  }

  loadKeyframes() {
    return this._try('keyframes', async () => {
      const doc = await this.api.keyframes(this.id);
      this.keyframes = (doc.keyframes || []).slice().sort((a, b) => a.k - b.k);
    });
  }

  /** The keyframe at or before time t (or the first). */
  keyframeAt(t) {
    if (!this.keyframes.length) return null;
    let best = this.keyframes[0];
    for (const kf of this.keyframes) { if (Number(kf.time_s) <= t + 1e-12) best = kf; else break; }
    return best;
  }

  loadFindings() {
    return this._try('findings', async () => {
      const doc = await this.api.findings(this.id);
      this.setFindings(doc.findings || []);
      this.findingsVerdict = { worst: doc.worst, safe: doc.safe, total: doc.total };
    });
  }

  setFindings(list) {
    const byId = new Map();
    for (const f of list) byId.set(f.id, f);
    this.findings = [...byId.values()].sort((a, b) => (a.time_s - b.time_s) || (a.id - b.id));
    this.findingsByLine = new Map();
    for (const f of this.findings) {
      if (!f.line) continue;
      const key = lineKey(f.program, f.line);
      if (!this.findingsByLine.has(key)) this.findingsByLine.set(key, []);
      this.findingsByLine.get(key).push(f);
    }
    this.dispatchEvent(new Event('findings'));
  }

  addFinding(finding) {
    if (!finding || finding.id === undefined) return;
    if (this.findings.some((f) => f.id === finding.id)) return;
    this.setFindings([...this.findings, finding]);
  }

  worstSeverity() {
    let worst = null;
    for (const f of this.findings) if (worst === null || SEVERITY_RANK[f.severity] > SEVERITY_RANK[worst]) worst = f.severity;
    return worst;
  }

  loadTimeline() {
    if (!this.has('timeline.json')) return Promise.resolve();
    return this._try('timeline', async () => {
      const doc = await this.api.timeline(this.id);
      this.setTimeline(doc);
    });
  }

  setTimeline(doc) {
    const tickS = Number(doc.tick_s) || (this.run && this.run.clock && Number(this.run.clock.tick_s)) || 1e-6;
    const blocks = (doc.blocks || []).filter((b) => Number.isFinite(b.tick0) && Number.isFinite(b.tick1) && b.tick1 >= b.tick0);
    const paths = new Map();
    for (const block of blocks) {
      const path = block.path || 0;
      if (!paths.has(path)) paths.set(path, []);
      paths.get(path).push(block);
    }
    for (const [path, list] of paths) {
      list.sort((a, b) => a.tick0 - b.tick0);
      paths.set(path, { blocks: list, tick0: Float64Array.from(list.map((b) => b.tick0)) });
    }
    const lines = new Map();
    const ops = [];
    for (const block of blocks) {
      if (block.op && !ops.includes(block.op)) ops.push(block.op);
      const key = lineKey(block.program, block.line);
      let stat = lines.get(key);
      if (!stat) { stat = { program: block.program, line: block.line, ticks: 0, executions: 0, first: block.tick0, starts: [] }; lines.set(key, stat); }
      stat.ticks += block.tick1 - block.tick0;
      stat.executions += 1;
      stat.first = Math.min(stat.first, block.tick0);
      if (stat.starts.length < 1000) stat.starts.push(block.tick0);
    }
    for (const stat of lines.values()) stat.starts.sort((a, b) => a - b);
    let end = 0;
    for (const block of blocks) end = Math.max(end, block.tick1);
    this.timeline = { tickS, blocks, paths, lines, ops, endTick: end };
    this.dispatchEvent(new Event('timeline'));
  }

  get pathList() { return [...this.timeline.paths.keys()].sort((a, b) => a - b); }

  /** The executed block of `path` at time t (the last one started at or before t). */
  blockAt(t, path = 0) {
    const entry = this.timeline.paths.get(path) || this.timeline.paths.values().next().value;
    if (!entry) return null;
    const tick = Math.round(t / this.tickS);
    const i = bisectRight(entry.tick0, tick);
    return i >= 0 ? entry.blocks[i] : entry.blocks[0];
  }

  /** Start time of the block after (dir 1) or before (dir -1) the one at t. */
  stepBlock(t, dir, path = 0) {
    const entry = this.timeline.paths.get(path) || this.timeline.paths.values().next().value;
    if (!entry || !entry.blocks.length) return t;
    const tick = Math.round(t / this.tickS);
    let i = bisectRight(entry.tick0, tick);
    if (dir > 0) {
      i = Math.min(entry.blocks.length - 1, i + 1);
    } else if (i >= 0 && entry.tick0[i] < tick) {
      // inside a block: back to its start first
    } else {
      i = Math.max(0, i - 1);
    }
    return Math.max(0, entry.blocks[Math.max(0, i)].tick0 * this.tickS);
  }

  lineStat(program, line) { return this.timeline.lines.get(lineKey(program, line)) || null; }

  /** Duration of the run so far (s): the timeline end, else the recorded cycle time, else progress. */
  duration() {
    const fromTimeline = this.timeline.endTick ? this.timeline.endTick * this.tickS : 0;
    const summary = this.run && this.run.summary;
    const cycle = summary && Number(summary.cycle_time_s);
    const progress = this.status.progress || {};
    return Math.max(fromTimeline, Number.isFinite(cycle) ? cycle : 0, Number(progress.sim_time_s) || 0);
  }

  operations() {
    const ops = (this.project.operations || []).map((o) => ({ name: o.name, program: o.program }));
    for (const name of this.timeline.ops) {
      if (!ops.some((o) => o.name === name)) {
        const block = this.timeline.blocks.find((b) => b.op === name);
        ops.push({ name, program: block ? block.program : null });
      }
    }
    return ops;
  }

  /** The text of a program: by operation from the run (`/program/{op}`), else from the project's files. */
  async programText({ op = null, program = null } = {}) {
    const key = `${op || ''}\u0001${program || ''}`;
    if (this.programs.has(key)) return this.programs.get(key);
    const promise = (async () => {
      const ops = this.operations();
      const chosen = op ? ops.find((o) => o.name === op) : ops.find((o) => o.program === program);
      const recorded = !!this.run || !!(this.record && this.record.job);
      if (!recorded) {
        throw new Error('the program is recorded in run.json when the run ends (this run was not started by this server)');
      }
      if (chosen) {
        try {
          const doc = await this.api.program(this.id, chosen.name);
          if (!program || doc.program === program) return { program: doc.program, text: doc.text, source: doc.source, op: doc.op };
        } catch (error) {
          if (!program || !this.projectId) throw error;
        }
      }
      if (program && this.projectId) {
        const text = await this.api.projectFile(this.projectId, program);
        return { program, text, source: 'project', op };
      }
      throw new Error(`the program ${program || op} is not recorded for this run`);
    })();
    this.programs.set(key, promise);
    promise.catch(() => this.programs.delete(key));
    return promise;
  }

  toolInfo(number) {
    const fromScene = this.scene && (this.scene.tools || []).find((t) => t.number === number);
    const fromProject = ((this.project.tools && this.project.tools.list) || []).find((t) => t.number === number);
    return { scene: fromScene || null, project: fromProject || null };
  }

  async refresh() {
    this.record = await this.api.run(this.id);
    await this.loadFiles();
    await Promise.all([this.loadTimeline(), this.loadFindings(), this.loadKeyframes(),
      this.channels.length ? Promise.resolve() : this.loadChannels(), this.scene ? Promise.resolve() : this.loadScene()]);
    this.dispatchEvent(new Event('refresh'));
  }
}

// ============================================================================ frames for the DRO

const OVERVIEW_FRAMES = 20000;
const CHUNK_TICKS = 4096;

/** frames.bin as the DRO needs it: an overview (at most 20 000 records over the run) plus
 * full-resolution chunks of 4096 L1 ticks fetched around the time shown. */
export class FrameStore {
  constructor(api, runId, header) {
    this.api = api;
    this.id = runId;
    this.header = header;
    this.overview = null;
    this.chunks = new Map();
    this.pending = new Set();
    this.ipoS = (Number(header.ipo_ticks) || 1000) * (Number(header.tick_s) || 1e-6);
  }

  async init() {
    this.overview = await this.api.frames(this.id, { maxFrames: OVERVIEW_FRAMES });
    return this;
  }

  get axes() { return this.header.axes || []; }
  get paths() { return this.header.paths || [{ path: 0 }]; }

  _chunkIndex(t) { return Math.floor(t / (CHUNK_TICKS * this.ipoS)); }

  /** Fetch the full-resolution chunk around t (no-op when the overview is already full resolution). */
  request(t) {
    if (!this.overview || this.overview.stride <= 1) return null;
    const index = this._chunkIndex(t);
    if (this.chunks.has(index) || this.pending.has(index)) return null;
    this.pending.add(index);
    const span = CHUNK_TICKS * this.ipoS;
    return this.api.frames(this.id, { t0: index * span, t1: (index + 1) * span })
      .then((frames) => { this.chunks.set(index, frames); if (this.chunks.size > 12) this.chunks.delete(this.chunks.keys().next().value); })
      .catch(() => {})
      .finally(() => this.pending.delete(index));
  }

  _source(t) {
    const chunk = this.chunks.get(this._chunkIndex(t));
    return chunk && chunk.count ? chunk : this.overview;
  }

  /** The machine state at time t: axes (relative and absolute), spindle, tool tip and axis per path. */
  at(t, path = 0) {
    const frames = this._source(t);
    if (!frames || !frames.count) return null;
    const i = Math.max(0, Math.min(frames.count - 1, nearestIndex(frames.time_s, t)));
    const axisField = frames.fields.axis;
    const revField = frames.fields.axis_rev;
    const axes = this.axes.map((axis, a) => {
      const raw = axisField ? axisField.data[i * axisField.width + a] : NaN;
      const rev = revField ? revField.data[i * revField.width + a] : 0;
      const kind = axis.kind || 'linear';
      const origin = Number.isFinite(Number(axis.origin)) ? Number(axis.origin) : null;
      const absolute = kind === 'rotary' ? raw + 2 * Math.PI * rev : raw + (origin || 0);
      return { name: axis.name, kind, relative: raw, absolute, hasOrigin: origin !== null };
    });
    const p = Math.max(0, Math.min(path, (frames.fields.tip_S_m ? frames.fields.tip_S_m.width / 3 : 1) - 1));
    const vec = (field, j) => (field ? [0, 1, 2].map((k) => field.data[j * field.width + p * 3 + k]) : null);
    const tip = vec(frames.fields.tip_S_m, i);
    const toolAxis = vec(frames.fields.axis_T_S, i);
    const unwrapped = (j) => {
      const a = frames.fields.spindle_angle_rad;
      const r = frames.fields.spindle_rev;
      if (!a) return NaN;
      return a.data[j * a.width + Math.min(p, a.width - 1)] + 2 * Math.PI * (r ? r.data[j * r.width + Math.min(p, r.width - 1)] : 0);
    };
    const j0 = Math.max(0, i - 1);
    const j1 = Math.min(frames.count - 1, i + 1);
    const dt = frames.time_s[j1] - frames.time_s[j0];
    let omega = NaN;
    let feed = NaN;
    if (dt > 0) {
      omega = (unwrapped(j1) - unwrapped(j0)) / dt;
      const a = vec(frames.fields.tip_S_m, j0);
      const b = vec(frames.fields.tip_S_m, j1);
      if (a && b) feed = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / dt;
    }
    return { time_s: frames.time_s[i], tick: frames.tick[i], axes, tip, toolAxis, omega, feed,
      exact: frames !== this.overview || frames.stride <= 1 };
  }
}

// ============================================================================ channel values at a time

/** Values of a few tick-tier channels at any time: an overview over the run, refined around t. */
export class ChannelSampler {
  constructor(api, runId, columns, tier = 'tick') {
    this.api = api;
    this.id = runId;
    this.columns = columns;
    this.tier = tier;
    this.overview = new Map();
    this.fine = new Map();
    this.fineAt = null;
  }

  async init() {
    if (!this.columns.length) return this;
    const doc = await this.api.series(this.id, { tier: this.tier, channels: this.columns, maxPoints: 8000 });
    const next = new Map();
    for (const s of doc.series || []) next.set(s.column, toArrays(s));
    this.overview = next;
    return this;
  }

  async refine(t, halfWidthS) {
    if (!this.columns.length) return;
    this.fineAt = t;
    const doc = await this.api.series(this.id, { tier: this.tier, channels: this.columns, t0: Math.max(0, t - halfWidthS), t1: t + halfWidthS, maxPoints: 64 });
    if (this.fineAt !== t) return;
    this.fine = new Map((doc.series || []).map((s) => [s.column, toArrays(s)]));
  }

  value(column, t) {
    for (const source of [this.fine, this.overview]) {
      const s = source.get(column);
      if (!s || !s.t.length) continue;
      if (source === this.fine && (t < s.t[0] || t > s.t[s.t.length - 1])) continue;
      const i = nearestIndex(s.t, t);
      const v = s.v[i];
      return Number.isFinite(v) ? v : null;
    }
    return null;
  }
}

function toArrays(series) {
  const t = Float64Array.from(series.t_s || []);
  const v = new Float64Array((series.v || []).length);
  (series.v || []).forEach((x, i) => { v[i] = x === null ? NaN : Number(x); });
  return { t, v };
}

export { toArrays, decodeArrayDoc };
