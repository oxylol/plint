import { interpret, DEFAULT_SETTINGS } from './gcode/interpreter.js';
import { defaultLibrary, normalizeTool, describeTool, toolColor, MILL_TYPES, LATHE_TYPES } from './gcode/tools.js';
import { describeStock, stockToComment } from './gcode/stock.js';
import { simulateJob, opNames, toolEnvelopes } from './sim/job.js';
import { parseSetup, describeSetup } from './gcode/setup.js';
import { partSummary, viewDetail, section, probe } from './sim/inspect.js';
import { readModel, isModelFile, defaultPlacement, buildTarget, targetSummary, compareTarget, describePlacement } from './model/model.js';
import { Viewer } from './view/viewer.js';
import { Editor } from './editor.js';
import { ClaudeAgent, ChatView } from './claude.js';
import { EXAMPLES } from './examples.js';
import { parseTurnToolsIni, parseMillReg, detectRegister } from './gcode/lathetools.js';
import { decodeLegacyText, base64ToBytes } from './gcode/textdecode.js';
import { openServerStore } from './serverstore.js';
import { engagementReport, engagementAt } from './sim/engagement.js';
import { exportJob } from './gcode/export.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ------------------------------------------------------------------ persistence (best effort)
const store = {
  get(k, d) { try { const v = localStorage.getItem('cncstudio.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('cncstudio.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
// larger things (the target model's triangles) live in IndexedDB
const idb = {
  open() {
    return new Promise((res, rej) => {
      const r = indexedDB.open('cncstudio', 1);
      r.onupgradeneeded = () => r.result.createObjectStore('kv');
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
  },
  async get(k) {
    try {
      const db = await this.open();
      return await new Promise((res, rej) => { const q = db.transaction('kv').objectStore('kv').get(k); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    } catch { return undefined; }
  },
  async set(k, v) {
    try {
      const db = await this.open();
      await new Promise((res, rej) => { const tx = db.transaction('kv', 'readwrite'); if (v === undefined) tx.objectStore('kv').delete(k); else tx.objectStore('kv').put(v, k); tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
    } catch { /* storage unavailable: the model just won't survive a reload */ }
  },
};
// with DATA_DIR on the server, the work lives there (local storage stays as a cache)
const serverStore = await openServerStore();
serverStore?.wrap(store, idb);

const SETTINGS_DEFAULTS = {
  model: 'claude-opus-5-5', effort: 'high', apiKey: '',
  rapidMill: DEFAULT_SETTINGS.rapidMill, rapidLathe: DEFAULT_SETTINGS.rapidLathe, toolChangeTime: DEFAULT_SETTINGS.toolChangeTime,
  maxRpm: DEFAULT_SETTINGS.maxRpm, resolution: 420, latheRadiusMode: false, blockDelete: false, autoSim: true,
};

const uid = () => Math.random().toString(36).slice(2, 10);
// a job = ordered operations (one program each); older sessions had a single program
function loadOps() {
  const ops = store.get('ops', null);
  const legacy = store.get('program', '') || '';
  if (Array.isArray(ops) && ops.length && !(ops.length === 1 && !ops[0].text && legacy.trim())) return ops.map((o) => ({ id: o.id || uid(), fileName: o.fileName || 'untitled.nc', machine: o.machine === 'lathe' ? 'lathe' : 'mill', text: o.text || '' }));
  return [{ id: uid(), fileName: store.get('fileName', 'untitled.nc'), machine: store.get('machine', 'mill'), text: store.get('program', '') || '' }];
}
const initialOps = loadOps();

const state = {
  ops: initialOps,
  active: Math.min(store.get('activeOp', 0), initialOps.length - 1),
  jobCache: new Map(),
  outs: [], out: null,
  machine: 'mill',
  fileName: 'untitled.nc',
  subprograms: store.get('subs', []),
  library: { mill: store.get('lib.mill', null) || defaultLibrary('mill'), lathe: store.get('lib.lathe', null) || defaultLibrary('lathe') },
  libraryInfo: store.get('libInfo', { mill: null, lathe: null }),
  stockMode: 'program',
  manualStock: store.get('manualStock', { mill: { kind: 'box', min: [-50, -50, -20], max: [50, 50, 0] }, lathe: { kind: 'lathe', diameter: 50, length: 80, faceStock: 0, innerDiameter: 0 } }),
  settings: { ...SETTINGS_DEFAULTS, ...store.get('settings', {}) },
  result: null, runner: null, sim: null, issues: [],
  model: null, placement: null, target: null, attachments: [],
  time: 0, total: 0, playing: false, lastFrame: 0,
  versions: [], versionSeq: 0,
  wroteThisRun: false,
};

// ------------------------------------------------------------------ helpers
const fmtTime = (s) => {
  if (!Number.isFinite(s)) return '–';
  s = Math.max(0, s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${m}:${String(sec).padStart(2, '0')}`;
};
const fmtTimeLong = (s) => { const m = Math.floor(s / 60), sec = Math.round(s % 60); return m ? `${m} min ${sec} s` : `${sec} s`; };
const n3 = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);
const n1 = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);
function toast(msg, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

// ------------------------------------------------------------------ editor + viewer
const viewer = new Viewer($('#viewport'));
const editor = new Editor($('#editor'), {
  onChange: () => {
    syncEditorToOp();
    if (state.settings.autoSim && !agent?.busy) scheduleSim();
  },
  onCursorLine: (line) => { $('#cursorInfo').textContent = `Ln ${line + 1}`; },
  onLineClick: (line) => jumpToLine(line),
});

let simTimer = null;
function scheduleSim() {
  clearTimeout(simTimer);
  simTimer = setTimeout(() => simulate({ time: 'keep' }), 550);
}

// ------------------------------------------------------------------ simulation
const VOXEL_BUDGET = 6e6; // 3D material cells per operation

function interpOptions(machine = state.machine) {
  const s = state.settings;
  return {
    machine,
    subprograms: state.subprograms,
    toolLibrary: state.library[machine],
    blockDelete: s.blockDelete,
    settings: { rapidMill: s.rapidMill, rapidLathe: s.rapidLathe, toolChangeTime: s.toolChangeTime, maxRpm: s.maxRpm, latheRadiusMode: s.latheRadiusMode },
  };
}

function hashStr(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(36) + ':' + str.length;
}

function jobOptions() {
  const s = state.settings;
  const manual = state.stockMode === 'manual' ? state.manualStock : null;
  return {
    interp: interpOptions,
    key: hashStr(JSON.stringify([s.rapidMill, s.rapidLathe, s.toolChangeTime, s.maxRpm, s.latheRadiusMode, s.blockDelete, s.resolution, state.library, state.subprograms, manual])),
    resolution: s.resolution,
    voxelBudget: VOXEL_BUDGET,
    manualStock: manual,
  };
}

const LEVEL_ORDER = { error: 0, warning: 1, info: 2 };
const sortIssues = (list) => list.slice().sort((a, b) => (LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level]) || ((a.callLine ?? 1e9) - (b.callLine ?? 1e9)));
function countLevels(list) {
  const c = { error: 0, warning: 0, info: 0 };
  for (const i of list) c[i.level]++;
  return c;
}
const removedVolume = (out) => out.final.removedVolume();

// simulate every operation (the ones after the active tab only for reports / the tab markers)
function simulateAll() {
  return simulateJob(state.ops, state.ops.length - 1, jobOptions(), state.jobCache);
}
let fullJobTimer = null;
function scheduleFullJob() {
  clearTimeout(fullJobTimer);
  if (state.active >= state.ops.length - 1) return;
  fullJobTimer = setTimeout(() => {
    if (state.playing || agent?.busy) { scheduleFullJob(); return; }
    try { state.outs = simulateAll(); } catch (e) { console.error(e); }
    renderOps();
  }, 1200);
}

function simulate({ time = 'end' } = {}) {
  clearTimeout(simTimer);
  const busy = $('#simBusy');
  busy.hidden = false;
  syncEditorToOp();
  let outs;
  try {
    outs = simulateJob(state.ops, state.active, jobOptions(), state.jobCache);
  } catch (e) {
    busy.hidden = true;
    console.error(e);
    toast(`Simulation crashed: ${e.message}`);
    return null;
  }
  const out = outs[state.active];
  const result = out.result;
  Object.assign(state, { outs, out, result, sim: out.sim, runner: out.runner, total: result.totalTime });
  state.rotAxes = { a: result.moves.some((m) => m.rot && (m.rot[0] || m.rot[1])), b: result.moves.some((m) => m.rot && (m.rot[2] || m.rot[3])) };
  state.issues = sortIssues(out.issues);
  viewer.load(result, out.sim, { engine: out.engine, X: out.X, fixtures: out.fixtures, rotReverse: !!out.setup.rotary?.reverse, workFromPart: out.workFromPart });
  scheduleCompare();
  editor.setIssues(state.issues);
  renderOps();
  scheduleFullJob();
  const guess = detectMachine(editor.value);
  const hint = $('#machineHint');
  hint.hidden = !guess || guess === state.machine;
  if (guess) { hint.textContent = `Looks like a ${guess} program — switch`; hint.onclick = () => setMachine(guess); }
  renderBadges();
  renderChecks();
  renderEngagement();
  renderTools();
  renderStock();
  renderLegend();
  const t = time === 'start' ? 0 : time === 'keep' ? Math.min(state.time, state.total) : state.total;
  seek(t, true);
  busy.hidden = true;
  return result;
}

function seek(t, force = false) {
  if (!state.runner) return;
  t = Math.max(0, Math.min(t, state.total));
  state.time = t;
  state.runner.advanceTo(t + (t >= state.total ? 1 : 0));
  viewer.updateStock(force);
  const st = state.runner.stateAt(t);
  if (state.result.usesRotary) viewer.setRotary(st.rot?.[0] || 0, st.rot?.[1] || 0);
  viewer.setTool(st.pos, st.move);
  viewer.setTime(t >= state.total ? 1e9 : t);
  updateDro(st, t);
  if (state.total) $('#engHead').style.left = `${Math.min(100, (t / state.total) * 100)}%`;
  if (st.move) editor.setExecLine(st.move.callLine, state.playing || force === 'line');
  $('#timeline').value = state.total ? Math.round((t / state.total) * 1000) : 1000;
  $('#timeLabel').textContent = `${fmtTime(t)} / ${fmtTime(state.total)}`;
}

function updateDro(st, t) {
  const p = st.pos || [0, 0, 0];
  const m = st.move;
  droEngagement(m, st.index, t);
  if (state.machine === 'lathe') {
    $('#droX').textContent = n3(p[0] * 2);
    $('#droZ').textContent = n3(p[2]);
  } else {
    $('#droX').textContent = n3(p[0]);
    $('#droY').textContent = n3(p[1]);
    $('#droZ').textContent = n3(p[2]);
  }
  $('#droARow').hidden = !state.rotAxes?.a;
  $('#droBRow').hidden = !state.rotAxes?.b;
  if (state.rotAxes?.a) $('#droA').textContent = n3(st.rot?.[0] || 0);
  if (state.rotAxes?.b) $('#droB').textContent = n3(st.rot?.[1] || 0);
  $('#droT').textContent = `T${m?.tool ?? 0}`;
  if (m) {
    const rate = m.kind === 'rapid' ? 'rapid' : `F${Math.round(m.rate || 0)}`;
    $('#droF').textContent = rate;
    $('#droS').textContent = `S${Math.round(m.rpm || 0)}`;
    const mode = m.kind === 'rapid' ? 'G00' : m.arc ? (m.arc === 'cw' ? 'G02' : 'G03') : m.kind === 'feed' ? 'G01' : m.kind;
    $('#droMode').textContent = m.cycle ? `${mode} · ${m.cycle}` : mode;
  }
}

// ------------------------------------------------------------------ playback
function speedFactor() {
  const v = $('#speedSelect').value;
  if (v === 'fit20') return Math.max(1, state.total / 20);
  return Number(v) || 1;
}
function play() {
  if (!state.runner) return;
  if (state.time >= state.total - 1e-6) seek(0, true);
  state.playing = true;
  state.lastFrame = performance.now();
  $('#btnPlay').textContent = '⏸';
  requestAnimationFrame(tick);
}
function pause() {
  state.playing = false;
  $('#btnPlay').textContent = '▶';
}
function tick(now) {
  if (!state.playing) return;
  // clamp so a stalled/background tab doesn't jump to the end, but tolerate slow frame rates
  const dt = Math.min(0.5, (now - state.lastFrame) / 1000);
  state.lastFrame = now;
  const t = state.time + dt * speedFactor();
  if (t >= state.total) { seek(state.total); pause(); return; }
  seek(t);
  requestAnimationFrame(tick);
}
function stepBlock() {
  if (!state.runner) return;
  pause();
  const mv = state.result.moves;
  let i = state.runner.moveIndexAt(state.time + 1e-6);
  if (i < 0) return;
  if (state.time >= mv[i].t1 - 1e-9) i++;
  if (i >= mv.length) return;
  const line = mv[i].callLine;
  let j = i;
  while (j + 1 < mv.length && mv[j + 1].callLine === line) j++;
  seek(mv[j].t1, 'line');
}
function jumpToLine(line) {
  if (!state.result) return;
  const mv = state.result.moves;
  let last = -1;
  for (let i = 0; i < mv.length; i++) if (mv[i].callLine === line) last = i;
  if (last < 0) {
    // nearest following line that moves
    for (let i = 0; i < mv.length; i++) if ((mv[i].callLine ?? -1) > line) { last = i; break; }
  }
  if (last < 0) return;
  pause();
  seek(mv[last].t1, 'line');
}

$('#btnPlay').onclick = () => (state.playing ? pause() : play());
$('#btnStart').onclick = () => { pause(); seek(0, true); };
$('#btnEnd').onclick = () => { pause(); seek(state.total, true); };
$('#btnStep').onclick = stepBlock;
$('#timeline').oninput = (e) => { pause(); seek((e.target.value / 1000) * state.total); };
$('#btnSimulate').onclick = () => simulate({ time: 'end' });
$('#autoSim').checked = state.settings.autoSim;
$('#autoSim').onchange = (e) => { state.settings.autoSim = e.target.checked; saveSettings(); };

$('#viewSeg').onclick = (e) => {
  const b = e.target.closest('button[data-view]');
  if (!b) return;
  $$('#viewSeg button').forEach((x) => x.classList.toggle('active', x === b));
  viewer.setView(b.dataset.view);
};
for (const cb of $$('[data-vis]')) cb.onchange = () => viewer.setVisible(cb.dataset.vis, cb.checked);
$('#sectionToggle').onchange = (e) => viewer.setSection(e.target.checked);

// ------------------------------------------------------------------ machine
function setMachine(m, { resim = true } = {}) {
  state.machine = m;
  activeOp().machine = m;
  saveOps();
  document.body.classList.toggle('lathe', m === 'lathe');
  $$('#machineSeg button').forEach((b) => b.classList.toggle('active', b.dataset.machine === m));
  $('#droYRow').style.display = m === 'lathe' ? 'none' : '';
  renderChips();
  if (resim) simulate({ time: 'end' });
}
$('#machineSeg').onclick = (e) => {
  const b = e.target.closest('button[data-machine]');
  if (b && b.dataset.machine !== state.machine) setMachine(b.dataset.machine);
};

function detectMachine(text) {
  const t = text.toUpperCase();
  const latheHints = /\bG9[67]\b|\bG7[0-6]\b|\bG28\s*U|\bG50\s*S|\bT\d{4}\b|\bG18\b|\bG99\b.*\bF0?\.\d/.test(t);
  const millHints = /\bY-?\d|\bG17\b|\bG4[34]\b|\bH\d+|\bG8[1-9]\b/.test(t);
  if (latheHints && !millHints) return 'lathe';
  if (millHints && !latheHints) return 'mill';
  return null;
}

// ------------------------------------------------------------------ operations (job)
function activeOp() { return state.ops[state.active]; }
function saveOps() {
  store.set('ops', state.ops.map(({ id, fileName, machine, text }) => ({ id, fileName, machine, text })));
  store.set('activeOp', state.active);
}
let saveOpsTimer = null;
// the editor shows state.editorOpId's program (only that operation takes the editor's text)
function syncEditorToOp() {
  const op = activeOp();
  if (!op || op.id !== state.editorOpId || op.text === editor.value) return;
  op.text = editor.value;
  clearTimeout(saveOpsTimer);
  saveOpsTimer = setTimeout(saveOps, 400);
}
const opName = (i) => opNames(state.ops)[i];

function selectOp(i, { time = 'end' } = {}) {
  syncEditorToOp();
  pause();
  state.active = Math.max(0, Math.min(i, state.ops.length - 1));
  const op = activeOp();
  setMachine(op.machine, { resim: false });
  editor.setValue(op.text, { silent: true });
  state.editorOpId = op.id;
  state.fileName = op.fileName;
  $('#fileName').textContent = op.fileName;
  saveOps();
  renderVersions();
  return simulate({ time });
}

function newOpName() {
  const names = new Set(opNames(state.ops).map((n) => n.toUpperCase()));
  let k = state.ops.length + 1;
  while (names.has(`OP${k * 10}`)) k++;
  return `OP${k * 10}`;
}

// next operation: continues from the last one, re-clamped (flipped) by default
function nextOpTemplate(machine) {
  const prev = opName(state.ops.length - 1);
  const name = newOpName();
  const num = String(state.ops.length + 1).padStart(4, '0');
  return machine === 'lathe'
    ? `%\nO${num} (${name} - SECOND SIDE)\n(SETUP ${name})\n(STOCK FROM ${prev})\n(ORIENT FLIP X)\n(CHUCK GRIP 15)\nG18 G21 G40 G99\nG50 S3000\nT0101\nG96 S200 M03\nG00 X42. Z2. M08\n\nM30\n%\n`
    : `%\nO${num} (${name} - SECOND SIDE)\n(SETUP ${name})\n(STOCK FROM ${prev})\n(ORIENT FLIP X)\n(ZERO TOP CENTER)\n(VISE GRIP 8)\nG90 G17 G21 G40 G49 G80 G94\nT1 M06\nS8000 M03\nG54 G00 X0. Y0.\nG43 Z25. H01 M08\n\nM30\n%\n`;
}

function addOp({ text = '', fileName, machine = state.machine } = {}) {
  syncEditorToOp();
  const op = { id: uid(), fileName: fileName || `${newOpName().toLowerCase()}.nc`, machine, text };
  state.ops.push(op);
  saveOps();
  selectOp(state.ops.length - 1);
  if (text) addVersion(fileName || 'New operation', text, 'user');
  return op;
}

function removeOp(i) {
  const op = state.ops[i];
  if (state.ops.length < 2 || !op) return;
  const name = opName(i);
  if (op.text.trim() && !confirm(`Remove ${name} (${op.fileName}) from the job? Save it first if you still need it.`)) return;
  state.ops.splice(i, 1);
  state.jobCache.delete(op.id);
  state.versions = state.versions.filter((v) => v.opId !== op.id);
  const next = i < state.active ? state.active - 1 : Math.min(state.active, state.ops.length - 1);
  state.active = -1;
  selectOp(next);
  toast(`Removed ${name}`);
}

function renderOps() {
  const names = opNames(state.ops);
  const multi = state.ops.length > 1;
  const tabs = state.ops.map((op, i) => {
    const o = state.jobCache.get(op.id);
    const c = o ? countLevels(o.issues) : null;
    const dot = c ? (c.error ? 'err' : c.warning ? 'warn' : 'ok') : '';
    const tip = `${op.fileName} — ${op.machine}${o ? ` · ${describeSetup(o.setup, o.X, op.machine)}` : ''}${c ? ` · ${c.error} error(s), ${c.warning} warning(s)` : ''}`;
    return `<button class="op-tab${i === state.active ? ' active' : ''}" data-op="${i}" title="${escapeHtml(tip)}"><span class="op-m">${op.machine === 'lathe' ? 'L' : 'M'}</span><span class="op-name">${escapeHtml(names[i])}</span>${dot ? `<span class="op-dot ${dot}"></span>` : ''}${multi ? `<span class="op-x" data-del-op="${i}" title="Remove this operation">×</span>` : ''}</button>`;
  }).join('');
  $('#opsTabs').innerHTML = tabs;
  const out = state.out;
  const setupTxt = out ? describeSetup(out.setup, out.X, out.op.machine) : '';
  const info = $('#setupInfo');
  const show = multi || (out && (out.setup.orient.length || out.setup.vise || out.setup.rotary || out.setup.stockFrom || out.engine === 'voxel'));
  info.hidden = !show;
  info.textContent = setupTxt;
  info.title = `${setupTxt}${out ? ` · ${out.engine === 'voxel' ? `3D material model, ${out.sim.h.toFixed(2)} mm cells` : out.engine === 'lathe' ? 'turned profile' : 'height map'}` : ''}`;
}

$('#opsTabs').onclick = (e) => {
  const x = e.target.closest('[data-del-op]');
  if (x) { e.stopPropagation(); removeOp(Number(x.dataset.delOp)); return; }
  const b = e.target.closest('[data-op]');
  if (b && Number(b.dataset.op) !== state.active) selectOp(Number(b.dataset.op));
};
$('#opsTabs').ondblclick = (e) => {
  const b = e.target.closest('[data-op]');
  if (!b) return;
  const op = state.ops[Number(b.dataset.op)];
  const name = prompt('File name of this operation', op.fileName);
  if (name && name.trim()) { op.fileName = name.trim(); if (op === activeOp()) { state.fileName = op.fileName; $('#fileName').textContent = op.fileName; } saveOps(); renderOps(); }
};
$('#btnAddOp').onclick = (e) => { e.stopPropagation(); $('#btnAddOp').parentElement.classList.toggle('open'); };
document.addEventListener('click', () => $('#btnAddOp').parentElement.classList.remove('open'));
$('#opMenu').onclick = (e) => {
  const b = e.target.closest('button[data-act]');
  if (!b) return;
  $('#btnAddOp').parentElement.classList.remove('open');
  const act = b.dataset.act;
  if (act === 'next-mill' || act === 'next-lathe') {
    const machine = act === 'next-lathe' ? 'lathe' : 'mill';
    addOp({ text: nextOpTemplate(machine), machine });
    toast(`${opName(state.active)} starts from the result of ${opName(state.active - 1)} — edit (ORIENT …) / (ZERO …) to re-clamp it`, 5000);
  } else if (act === 'open') $('#opFileInput').click();
  else if (act === 'dup') {
    const op = activeOp();
    // the copy gets its own operation name so (STOCK FROM …) stays unambiguous
    const name = newOpName();
    const text = /\(\s*(?:SETUP|OPERATION)[^)]*\)/i.test(op.text) ? op.text.replace(/\(\s*(?:SETUP|OPERATION)[^)]*\)/i, `(SETUP ${name})`) : op.text;
    addOp({ text, fileName: op.fileName.replace(/(\.[^.]+)?$/, '-copy$1'), machine: op.machine });
  } else if (act === 'save-job') saveJob();
};
$('#opFileInput').onchange = async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  for (const f of files) {
    const text = await readTextFile(f);
    addOp({ text, fileName: f.name, machine: detectMachine(text) || state.machine });
  }
};

function jobFileName() {
  const base = (state.ops[0]?.fileName || 'job').replace(/\.[^.]+$/, '');
  return `${base}.cncjob.json`;
}
function saveJob() {
  syncEditorToOp();
  const job = { format: 'cnc-studio-job', version: 1, savedAt: new Date().toISOString(), ops: state.ops.map(({ fileName, machine, text }) => ({ fileName, machine, text })), subprograms: state.subprograms };
  if (state.model) {
    const m = state.model;
    const bytes = new Uint8Array(m.tris.buffer, m.tris.byteOffset, m.tris.byteLength);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    job.model = { name: m.name, format: m.format, placement: state.placement, tris: btoa(bin) };
  }
  download(JSON.stringify(job, null, 1), jobFileName(), 'application/json');
}
function loadJob(job, name, { quiet = false } = {}) {
  if (!job || !Array.isArray(job.ops) || !job.ops.length) { toast('Not a CNC Studio job file'); return; }
  pause();
  state.ops = job.ops.map((o) => ({ id: uid(), fileName: o.fileName || 'untitled.nc', machine: o.machine === 'lathe' ? 'lathe' : 'mill', text: String(o.text || '') }));
  state.jobCache.clear();
  state.versions = [];
  if (Array.isArray(job.subprograms) && job.subprograms.length) addSubprograms(job.subprograms);
  if (job.model?.tris) {
    const bin = atob(job.model.tris);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const tris = new Float32Array(bytes.buffer);
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < tris.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], tris[i + k]); mx[k] = Math.max(mx[k], tris[i + k]); }
    const model = { name: job.model.name, format: job.model.format, tris, triCount: tris.length / 9, bounds: { min: mn, max: mx } };
    setModel(model, job.model.placement);
    idb.set('model', { ...model, placement: job.model.placement });
  }
  state.active = -1;
  selectOp(0);
  for (const op of state.ops) state.versions.push({ id: ++state.versionSeq, opId: op.id, title: name || 'Job', text: op.text, source: 'user', at: new Date() });
  renderVersions();
  if (!quiet) toast(`Opened job with ${state.ops.length} operation${state.ops.length > 1 ? 's' : ''}`);
}

// ------------------------------------------------------------------ program / files / versions
function setProgram(text, { name, source = 'user', title, machine } = {}) {
  if (machine && machine !== state.machine) setMachine(machine, { resim: false });
  editor.setValue(text, { silent: true });
  syncEditorToOp();
  if (name) { state.fileName = name; activeOp().fileName = name; $('#fileName').textContent = name; }
  saveOps();
  if (source !== 'restore') addVersion(title || (source === 'claude' ? 'Claude' : name || 'Edit'), text, source);
  return simulate({ time: 'end' });
}

function addVersion(title, text, source) {
  const opId = activeOp().id;
  const mine = state.versions.filter((v) => v.opId === opId);
  const last = mine[mine.length - 1];
  if (last && last.text === text) return last;
  const v = { id: ++state.versionSeq, opId, title, text, source, at: new Date() };
  state.versions.push(v);
  if (state.versions.length > 80) state.versions.shift();
  renderVersions();
  return v;
}
function renderVersions() {
  const sel = $('#versionSelect');
  const opId = activeOp()?.id;
  sel.innerHTML = '<option value="">Versions</option>' + state.versions.filter((v) => v.opId === opId).reverse()
    .map((v) => `<option value="${v.id}">${v.source === 'claude' ? '✦ ' : ''}${escapeHtml(v.title).slice(0, 40)} · ${v.at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</option>`).join('');
}
$('#versionSelect').onchange = (e) => {
  const v = state.versions.find((x) => x.id === Number(e.target.value));
  if (v) { setProgram(v.text, { source: 'restore' }); toast(`Restored “${v.title}”`); }
  e.target.value = '';
};
function restoreVersion(id) {
  const v = state.versions.find((x) => x.id === id);
  if (!v) return;
  const i = state.ops.findIndex((o) => o.id === v.opId);
  if (i >= 0 && i !== state.active) selectOp(i);
  setProgram(v.text, { source: 'restore' });
  toast(`Restored “${v.title}”`);
}

$('#btnNew').onclick = () => {
  if (state.ops.length > 1) {
    if (!confirm(`Start a new job? This closes all ${state.ops.length} operations (use + → Save job first to keep them).`)) return;
    const keep = activeOp();
    state.ops = [keep];
    state.jobCache.clear();
    state.active = 0;
    renderOps();
  }
  const tpl = state.machine === 'lathe'
    ? '%\nO0001 (NEW PROGRAM)\n(STOCK D40 L60)\nG18 G21 G40 G99\nG50 S3000\nT0101\nG96 S200 M03\nG00 X42. Z2. M08\n\nM30\n%\n'
    : '%\nO0001 (NEW PROGRAM)\n(STOCK X-50:50 Y-40:40 Z-20:0)\nG90 G17 G21 G40 G49 G80 G94\nT1 M06\nS8000 M03\nG54 G00 X0. Y0.\nG43 Z25. H01 M08\n\nM30\n%\n';
  setProgram(tpl, { name: 'untitled.nc', title: 'New program' });
};
$('#btnOpen').onclick = () => $('#fileInput').click();
$('#fileInput').onchange = async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  if (!files.length) return;
  const texts = await Promise.all(files.map(readTextFile));
  if (/\.json$/i.test(files[0].name)) {
    try { loadJob(JSON.parse(texts[0]), files[0].name); } catch { toast('Could not read that job file'); }
    return;
  }
  if (files.length > 1) {
    addSubprograms(files.slice(1).map((f, i) => ({ name: f.name, text: texts[i + 1] })));
  }
  const m = detectMachine(texts[0]);
  if (m && m !== state.machine) toast(`Looks like a ${m} program — switched to ${m === 'lathe' ? 'Lathe' : 'Mill'}`);
  setProgram(texts[0], { name: files[0].name, title: files[0].name, machine: m || state.machine });
};
async function readTextFile(f) {
  const buf = await f.arrayBuffer();
  const utf = new TextDecoder('utf-8', { fatal: false }).decode(buf);
  // old controls/CAM output is often Latin-1 / CP850
  return utf.includes('�') ? new TextDecoder('windows-1252').decode(buf) : utf;
}
function download(text, name, type = 'text/plain') {
  const blob = new Blob([text], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function saveProgram() {
  download(editor.value.replace(/\r?\n/g, '\r\n'), state.fileName || 'program.nc');
}
$('#btnSave').onclick = saveProgram;

// ------------------------------------------------------------------ export for the machine
function buildExport() {
  syncEditorToOp();
  const outs = simulateAll();
  state.outs = outs;
  const names = opNames(state.ops);
  const machine = state.ops[state.active].machine;
  const same = state.ops.map((op, i) => i).filter((i) => state.ops[i].machine === machine);
  const all = $('input[name=expScope][value=all]').checked && same.length > 1;
  const idx = all ? same : [state.active];
  const num = parseInt($('#expNum').value, 10);
  const r = exportJob({
    ops: idx.map((i) => ({ ...state.ops[i], name: names[i], out: outs[i] })),
    subprograms: state.subprograms,
    keepComments: $('#expComments').checked,
    inlineSubs: $('#expSubs').value === 'inline',
    toolList: $('#expTools').checked,
    programNumber: num > 0 && num <= 9999 ? num : null,
    interpOptions,
  });
  const skipped = all ? state.ops.map((op, i) => i).filter((i) => !same.includes(i)).map((i) => `${names[i]} (${state.ops[i].machine})`) : [];
  if (skipped.length) r.notes.push(`Not included — another machine: ${skipped.join(', ')}`);
  const o = r.text.split('\r\n')[1].match(/^O(\d+)/)?.[1] || '1000';
  const base = (state.ops[idx[0]].fileName || 'program').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_');
  state.exported = { text: r.text, name: `${base}${all ? '_all' : ''}_O${o}.nc` };
  $('#expPreview').textContent = r.text.replace(/\r\n/g, '\n');
  $('#expCheck').innerHTML = [...r.verify.map((v) => `<li class="${v.ok ? 'ok' : 'bad'}">${escapeHtml(v.text)}</li>`), ...r.notes.map((n) => `<li>${escapeHtml(n)}</li>`)].join('');
}

$('#btnExport').onclick = () => {
  const names = opNames(state.ops);
  const machine = state.ops[state.active].machine;
  const same = state.ops.map((op, i) => i).filter((i) => state.ops[i].machine === machine);
  $('#expScopeOp').textContent = `This operation (${names[state.active]})`;
  $('#expScopeAll').textContent = same.length > 1 ? `All ${same.length} ${machine} operations (${same.map((i) => names[i]).join(', ')}) — M00 between setups` : 'All operations (only one here)';
  $('input[name=expScope][value=all]').disabled = same.length < 2;
  if (same.length < 2) $('input[name=expScope][value=op]').checked = true;
  try { buildExport(); } catch (e) { console.error(e); toast(`Export failed: ${e.message}`); return; }
  $('#exportDlg').showModal();
};
for (const el of ['#expSubs', '#expNum', '#expComments', '#expTools']) $(el).addEventListener('change', () => buildExport());
for (const el of $$('input[name=expScope]')) el.addEventListener('change', () => buildExport());
$('#btnExpDownload').onclick = () => { if (state.exported) download(state.exported.text, state.exported.name); };
$('#btnExpCopy').onclick = () => {
  if (!state.exported) return;
  navigator.clipboard?.writeText(state.exported.text).then(() => toast('Copied'), () => toast('Copy is blocked here — use Download'));
};

// examples
$('#examplesMenu').innerHTML = EXAMPLES.map((ex) => `<button data-ex="${ex.id}">${escapeHtml(ex.name)}</button>`).join('');
$('#btnExamples').onclick = (e) => { e.stopPropagation(); $('#btnExamples').parentElement.classList.toggle('open'); };
document.addEventListener('click', () => $('#btnExamples').parentElement.classList.remove('open'));
$('#examplesMenu').onclick = (e) => {
  const b = e.target.closest('button[data-ex]');
  if (!b) return;
  const ex = EXAMPLES.find((x) => x.id === b.dataset.ex);
  if (state.ops.length > 1 && !confirm(`Open the example as a new job? This closes the current ${state.ops.length} operations (+ → Save job keeps them).`)) return;
  // an example is a job of its own
  loadJob({ ops: ex.ops || [{ fileName: `${ex.id}.nc`, machine: ex.machine, text: ex.text }] }, ex.name, { quiet: true });
  pause();
  seek(0, true);
  play();
};

// subprograms
function addSubprograms(list) {
  for (const s of list) {
    const i = state.subprograms.findIndex((x) => x.name.toLowerCase() === s.name.toLowerCase());
    if (i >= 0) state.subprograms[i] = s; else state.subprograms.push(s);
  }
  store.set('subs', state.subprograms);
  renderSubs();
}
function renderSubs() {
  $('#subCount').textContent = state.subprograms.length;
  $('#subList').innerHTML = state.subprograms.length
    ? state.subprograms.map((s, i) => `<li><span>${escapeHtml(s.name)}</span><span class="muted">${s.text.split('\n').length} lines</span><button type="button" class="btn small ghost" data-del="${i}">Remove</button></li>`).join('')
    : '<li class="muted">No subprogram files loaded.</li>';
}
$('#btnSubs').onclick = () => { renderSubs(); $('#subsDlg').showModal(); };
$('#btnAddSubs').onclick = () => $('#subInput').click();
$('#subInput').onchange = async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  const texts = await Promise.all(files.map(readTextFile));
  addSubprograms(files.map((f, i) => ({ name: f.name, text: texts[i] })));
  simulate({ time: 'keep' });
};
$('#subList').onclick = (e) => {
  const b = e.target.closest('[data-del]');
  if (!b) return;
  state.subprograms.splice(Number(b.dataset.del), 1);
  store.set('subs', state.subprograms);
  renderSubs();
  simulate({ time: 'keep' });
};

// ------------------------------------------------------------------ panels
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

function counts() { return countLevels(state.issues); }

function renderBadges() {
  const c = counts();
  $('#editorBadges').innerHTML =
    (c.error ? `<span class="badge error">${c.error} error${c.error > 1 ? 's' : ''}</span>` : '') +
    (c.warning ? `<span class="badge warning">${c.warning} warning${c.warning > 1 ? 's' : ''}</span>` : '') +
    (!c.error && !c.warning && state.result?.moves.length ? '<span class="badge ok">✓</span>' : '');
  const cc = $('#checkCount');
  cc.textContent = c.error + c.warning;
  cc.className = 'count' + (c.error ? ' err' : c.warning ? ' warn' : '');
}

function renderChecks() {
  const r = state.result;
  const st = r.stats;
  const vol = removedVolume(state.out) / 1000;
  $('#stats').innerHTML = [
    ['Cycle time', fmtTime(r.totalTime)],
    ['Tool changes', st.toolChanges],
    ['Feed length', `${(st.feedLength / 1000).toFixed(2)} m`],
    ['Rapid length', `${(st.rapidLength / 1000).toFixed(2)} m`],
    ['Removed', `${vol.toFixed(vol < 10 ? 2 : 1)} cm³`],
    ['Blocks / moves', `${st.blocks} / ${st.moves}`],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  const lines = editor.value.split('\n');
  const list = $('#issues');
  if (!state.issues.length) {
    list.innerHTML = '<div class="issues-empty">✓ No errors or warnings.</div>';
    return;
  }
  list.innerHTML = state.issues.map((m, i) => {
    const ln = m.callLine === null || m.callLine === undefined ? '—' : `L${m.callLine + 1}`;
    const icon = m.level === 'error' ? '●' : m.level === 'warning' ? '▲' : 'ℹ';
    const code = m.callLine !== null && m.callLine !== undefined && lines[m.callLine] ? `<div class="muted"><code>${escapeHtml(lines[m.callLine].trim().slice(0, 70))}</code></div>` : '';
    return `<div class="issue" data-i="${i}"><span class="lvl ${m.level}">${icon}</span><span class="ln">${ln}</span><div>${escapeHtml(m.text)}${m.file && m.file !== 'main' ? ` <span class="muted">(${escapeHtml(m.file)})</span>` : ''}${code}</div></div>`;
  }).join('') + (counts().error + counts().warning ? '<button class="btn small ask-fix" id="btnAskFix">Ask Claude to fix these</button>' : '');
  list.onclick = (e) => {
    if (e.target.id === 'btnAskFix') { askFix(); return; }
    const it = e.target.closest('.issue');
    if (!it) return;
    const m = state.issues[Number(it.dataset.i)];
    if (m.callLine !== null && m.callLine !== undefined) { editor.goToLine(m.callLine); jumpToLine(m.callLine); }
  };
}

// ------------------------------------------------------------------ tool engagement
const pctOf = (v, D) => (D ? `${Math.round((100 * v) / D)}%` : '');
const mm = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1));

function droEngagement(m, index, t) {
  const el = $('#droEng');
  const eng = state.out?.engagement;
  const e = m?.kind === 'feed' ? engagementAt(eng, t, index) : null;
  el.hidden = !e;
  if (!e) return;
  const tool = state.result.tools[m.tool];
  const on = e.ap > 1e-3 || e.angle > 0;
  el.classList.toggle('air', !on);
  if (!on) el.textContent = 'in air';
  else if (eng.machine === 'lathe') el.textContent = `ap ${mm(e.ap)} · f ${e.f.toFixed(3)} · ${e.q.toFixed(1)} cm³/min`;
  else if (e.plunge) el.textContent = `plunge · ap ${mm(e.ap)}`;
  else el.textContent = `ae ${pctOf(e.ae, tool?.diameter)} (${mm(e.ae)}) · ${Math.round(e.angle)}° · ap ${mm(e.ap)}`;
}

function renderEngagement() {
  const eng = state.out?.engagement;
  const s = eng?.samples;
  const tools = state.result.tools;
  const used = eng ? Object.values(eng.perTool).filter((T) => T.cut > 0) : [];
  const lathe = eng?.machine === 'lathe';
  $('#engStrip').hidden = !used.length;
  // Check tab: one row per tool, tap = go to its heaviest line
  $('#engTable').innerHTML = !eng || !Object.keys(eng.perTool).length ? '' : '<h4>Tool engagement</h4>' + Object.values(eng.perTool).sort((a, b) => a.tool - b.tool).map((T) => {
    const tool = tools[T.tool], D = tool?.diameter || 0;
    const air = T.len ? Math.round((100 * (T.len - T.cut)) / T.len) : 0;
    const name = `T${T.tool}${D && !lathe ? ` Ø${D}` : ''} ${escapeHtml(tool?.type || '')}`;
    const peak = lathe ? T.max.ap : T.max.ae;
    const line = peak ? state.result.moves[s.move[peak.k]].callLine : null;
    if (!T.cut) return `<div class="eng-row" style="--sw:${tool?.color}"><b>${name}</b> — never in material</div>`;
    const body = lathe
      ? `depth of cut up to <span class="big">${mm(T.max.ap.v)}</span> (typical ${mm(T.apTyp)}) · f up to ${(T.max.f?.v || 0).toFixed(3)}`
      : `ae up to <span class="big">${pctOf(T.max.ae.v, D)}</span>${T.max.ae.v >= 0.95 * D ? ' (full slot)' : ''} · ap up to <span class="big">${mm(T.max.ap.v)}</span> (${(T.max.ap.v / D).toFixed(2)}×Ø) · typical ${pctOf(T.aeTyp, D)} / ${mm(T.apTyp)}${T.slot > 0.5 ? ` · slotting ${Math.round(T.slot)} mm` : ''}`;
    return `<div class="eng-row" style="--sw:${tool?.color}" data-line="${line ?? ''}"><b>${name}</b> — ${body} · up to ${T.max.q.v.toFixed(1)} cm³/min${air ? ` · ${air}% of feed in air` : ''}</div>`;
  }).join('');
  if (!used.length) return;
  drawEngChart();
}

// bars: the heaviest cut in each pixel column (mill: ae as % of Ø, lathe: depth of cut), in the tool's
// colour; line: axial depth (mill)
function drawEngChart() {
  const eng = state.out?.engagement, s = eng?.samples;
  const cv = $('#engChart');
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!s || !w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const lathe = eng.machine === 'lathe', total = state.total || 1, moves = state.result.moves, tools = state.result.tools;
  let apMax = 0;
  for (const v of s.ap) apMax = Math.max(apMax, v);
  const bar = new Float32Array(w), barTool = new Int32Array(w).fill(-1), depth = new Float32Array(w);
  for (let k = 0; k < s.t.length; k++) {
    const x = Math.min(w - 1, Math.max(0, Math.floor((s.t[k] / total) * w)));
    const m = moves[s.move[k]];
    const D = tools[m.tool]?.diameter;
    const v = lathe ? (apMax ? s.ap[k] / apMax : 0) : D ? Math.min(1, s.ae[k] / D) : 0;
    if (v >= bar[x] && (v > 0 || barTool[x] < 0)) { bar[x] = v; barTool[x] = m.tool; }
    depth[x] = Math.max(depth[x], s.ap[k]);
  }
  const top = 12, span = h - top - 1;
  for (let x = 0; x < w; x++) {
    if (barTool[x] < 0 || !(bar[x] > 0)) continue;
    g.fillStyle = tools[barTool[x]]?.color || '#888';
    g.fillRect(x, h - 1 - bar[x] * span, 1, bar[x] * span);
  }
  if (!lathe && apMax > 0) {
    g.strokeStyle = 'rgba(238,243,248,.75)';
    g.lineWidth = 1;
    g.beginPath();
    let pen = false;
    for (let x = 0; x < w; x++) {
      if (!(depth[x] > 0)) { pen = false; continue; }
      const y = h - 1 - (depth[x] / apMax) * span;
      if (pen) g.lineTo(x + 0.5, y); else g.moveTo(x + 0.5, y);
      pen = true;
    }
    g.stroke();
  }
  $('#engLabel').textContent = lathe ? `depth of cut (max ${mm(apMax)} mm)` : `bars: ae % of Ø · line: ap (max ${mm(apMax)} mm)`;
}

{
  const strip = $('#engStrip');
  const go = (e) => { const r = strip.getBoundingClientRect(); pause(); seek(((e.clientX - r.left) / r.width) * state.total, 'line'); };
  strip.addEventListener('pointerdown', (e) => { strip.setPointerCapture(e.pointerId); go(e); });
  strip.addEventListener('pointermove', (e) => { if (strip.hasPointerCapture(e.pointerId)) go(e); });
  $('#engTable').addEventListener('click', (e) => {
    const row = e.target.closest('.eng-row');
    const ln = row?.dataset.line;
    if (ln) { editor.goToLine(Number(ln)); jumpToLine(Number(ln)); }
  });
  window.addEventListener('resize', () => { if (!$('#engStrip').hidden) drawEngChart(); });
}

function renderLegend() {
  const r = state.result;
  const used = r.usedTools.map((n) => r.tools[n]).filter(Boolean);
  $('#legend').innerHTML = used.map((t) => `<span><span class="sw" style="background:${t.color}"></span>T${t.num} ${escapeHtml(t.type)}${t.diameter && state.machine === 'mill' ? ' Ø' + t.diameter : ''}</span>`).join('') +
    '<span><span class="sw" style="background:repeating-linear-gradient(90deg,#ff6b6b 0 3px,transparent 3px 5px)"></span>rapid</span>';
}

// small outline preview: lathe = real cutter/holder polygons, mill = proportional silhouette
function toolIcon(t, machine) {
  if (machine === 'lathe') {
    if (!t.cutter) return '<svg viewBox="0 0 10 10"></svg>';
    const cz = t.cutter.reduce((s, p) => s + p[0], 0) / t.cutter.length;
    const cx = t.cutter.reduce((s, p) => s + p[1], 0) / t.cutter.length;
    const span = t.type === 'drill' ? Math.max(40, (t.length || 60) * 0.7) : 34;
    const vz = (t.type === 'drill' ? span * 0.42 * (t.hand === 'right' ? -1 : 1) : cz) - span / 2;
    const vx = -(cx + span / 2);
    const P = (pts) => pts.map(([z, x]) => `${z.toFixed(2)},${(-x).toFixed(2)}`).join(' ');
    return `<svg viewBox="${vz} ${vx} ${span} ${span}" preserveAspectRatio="xMidYMid meet">
      ${t.holder?.length ? `<polygon points="${P(t.holder)}" fill="#3a4350" stroke="#6b7785" stroke-width="${span / 120}"/>` : ''}
      <polygon points="${P(t.cutter)}" fill="${t.color || '#4cc9f0'}" fill-opacity="0.75" stroke="${t.color || '#4cc9f0'}" stroke-width="${span / 120}"/>
      <circle cx="0" cy="0" r="${span / 45}" fill="#f5a524"/></svg>`;
  }
  const D = t.diameter, L = t.fluteLength, S = t.overallLength, Hd = t.holderDiameter;
  const H = S + 28, W = Math.max(Hd, D) * 1.15;
  const x = (w) => (W - w) / 2;
  const tipH = ['drill', 'spot'].includes(t.type) ? (D / 2) / Math.tan(((t.tipAngle || 118) * Math.PI) / 360) : t.type === 'chamfer' || t.type === 'engrave' ? (D / 2) : 0;
  const y = (v) => H - v; // tip at the bottom
  const tip = t.type === 'ball'
    ? `<path d="M${x(D)} ${y(D / 2)} A${D / 2} ${D / 2} 0 0 0 ${x(D) + D} ${y(D / 2)} Z" fill="${t.color}"/>`
    : tipH ? `<polygon points="${W / 2},${y(0)} ${x(D) + D},${y(tipH)} ${x(D)},${y(tipH)}" fill="${t.color}"/>` : '';
  const flutes = `<rect x="${x(D)}" y="${y(L)}" width="${D}" height="${L - (t.type === 'ball' ? D / 2 : tipH)}" fill="${t.color}"/>`;
  const shank = `<rect x="${x(t.shankDiameter)}" y="${y(S)}" width="${t.shankDiameter}" height="${S - L}" fill="#9aa4ae"/>`;
  const holder = `<rect x="${x(Hd)}" y="${y(H)}" width="${Hd}" height="${H - S}" fill="#3a4350"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${holder}${shank}${flutes}${tip}</svg>`;
}

function renderLibSource() {
  const info = state.libraryInfo[state.machine];
  $('#libSource').innerHTML = info
    ? `Library: <b>${info.count} tools imported from CncSimulator5</b> · ${new Date(info.at).toLocaleDateString()}<div class="path">${escapeHtml(info.source)}</div>`
    : `Library: <b>built-in ${state.machine} tools</b> — import your CncSimulator5 register to use your own tools.`;
}

function renderTools() {
  const m = state.machine;
  const lib = state.library[m];
  const r = state.result;
  const progTools = r ? r.programTools : {};
  const used = new Set(r ? r.usedTools : []);
  const types = m === 'lathe' ? LATHE_TYPES : MILL_TYPES;
  renderLibSource();
  const rows = [];
  const nums = [...new Set([...Object.keys(progTools), ...Object.keys(lib)].map(Number))].sort((a, b) => a - b);
  for (const n of nums) {
    const fromProg = !!progTools[n];
    const t = fromProg ? r.tools[n] : normalizeTool(lib[n], m);
    const ro = fromProg ? 'disabled' : '';
    const isReg = !fromProg && lib[n]?.source === 'cncsim';
    const size = m === 'lathe'
      ? `<input type="number" step="0.1" min="0" data-f="${t.type === 'groove' ? 'width' : t.type === 'drill' ? 'diameter' : 'noseRadius'}" value="${t.type === 'groove' ? t.width : t.type === 'drill' ? t.diameter : t.noseRadius}" ${ro} title="${t.type === 'groove' ? 'Width' : t.type === 'drill' ? 'Diameter' : 'Nose radius'}">`
      : `<input type="number" step="0.1" min="0.1" data-f="diameter" value="${t.diameter}" ${ro} title="Diameter">`;
    // second size column: insert angle (turn/bore), blade length (groove), point angle (drill/thread)
    const [ef, ev, et] = m !== 'lathe' ? ['fluteLength', t.fluteLength, 'Flute length']
      : t.type === 'groove' ? ['height', t.height ?? 16, 'Blade length (max plunge)']
      : t.type === 'drill' || t.type === 'thread' ? ['tipAngle', t.tipAngle, 'Point angle']
      : ['insertAngle', t.insertAngle ?? '', 'Insert angle'];
    const extra = `<input type="number" step="1" min="1" data-f="${ef}" value="${ev}" ${ro} title="${et}">`;
    const tip = escapeHtml(describeTool(t, m, { detail: true }));
    rows.push(`<tr class="${fromProg ? 'program' : ''}${isReg ? ' cncsim' : ''}" data-n="${n}" title="${tip}">
      <td class="icon">${toolIcon(t, m)}</td>
      <td>T${n}${used.has(n) ? ' •' : ''}</td>
      <td><select data-f="type" ${ro}>${types.map((x) => `<option ${x === t.type ? 'selected' : ''}>${x}</option>`).join('')}</select></td>
      <td>${size}</td><td>${extra}</td>
      <td><input data-f="name" value="${escapeHtml(t.name || '')}" ${ro}></td>
      <td>${fromProg ? '<span class="src">program</span>' : `${isReg ? '<span class="src reg" title="From the CncSimulator5 register">reg</span> ' : ''}<button class="link" data-del="${n}" title="Delete">✕</button>`}</td></tr>`);
  }
  $('#toolTable').innerHTML = `<table class="tools"><thead><tr><th></th><th>T</th><th>Type</th><th>${m === 'lathe' ? 'R / W / Ø' : 'Ø'}</th><th>${m === 'lathe' ? '° / H' : 'Flute'}</th><th>Name</th><th></th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

// ------------------------------------------------------------------ CncSimulator5 registers
function parseRegister(text) {
  const kind = detectRegister(text);
  if (!kind) return null;
  const { tools, skipped } = kind === 'lathe' ? parseTurnToolsIni(text) : parseMillReg(text);
  return { kind, tools, skipped };
}

function applyRegister(kind, tools, source, { quiet = false, skipped = [] } = {}) {
  state.library[kind] = tools;
  state.libraryInfo[kind] = { source, at: new Date().toISOString(), count: Object.keys(tools).length };
  store.set(`lib.${kind}`, tools);
  store.set('libInfo', state.libraryInfo);
  if (!quiet) toast(`Imported ${Object.keys(tools).length} ${kind} tools${skipped.length ? ` (${skipped.length} skipped)` : ''}`);
  if (kind === state.machine) simulate({ time: 'keep' }); else renderTools();
}

async function fetchRegisters() {
  try {
    const { registers } = await (await fetch('/api/cncsim/registers')).json();
    return registers.map((r) => ({ ...r, parsed: parseRegister(decodeLegacyText(base64ToBytes(r.b64))) })).filter((r) => r.parsed);
  } catch {
    return [];
  }
}

async function openRegisterDialog() {
  const list = $('#regList');
  list.innerHTML = '<li class="muted">Looking for CncSimulator5…</li>';
  $('#regDlg').showModal();
  const regs = await fetchRegisters();
  if (!regs.length) {
    list.innerHTML = '<li class="muted">No CncSimulator5 tool registers found on this computer. Use “Import register file…” to pick TurnTools.ini or MillTools.millreg.</li>';
    return;
  }
  list.innerHTML = regs.map((r, i) => {
    const p = r.parsed;
    const active = state.libraryInfo[p.kind]?.source === r.path;
    return `<li><div><span class="kind">${p.kind === 'lathe' ? 'Lathe' : 'Mill'} — ${Object.keys(p.tools).length} tools</span>
      <div class="meta">${escapeHtml(r.where)} · modified ${new Date(r.mtime).toLocaleDateString()}${p.skipped.length ? ` · ${p.skipped.length} skipped` : ''}${active ? ' · <b>in use</b>' : ''}</div></div>
      <button type="button" class="btn small ${i === 0 ? 'primary' : ''}" data-reg="${i}">${active ? 'Re-import' : 'Import'}</button>
      <div class="path">${escapeHtml(r.path)}</div></li>`;
  }).join('');
  list.onclick = (e) => {
    const b = e.target.closest('[data-reg]');
    if (!b) return;
    const r = regs[Number(b.dataset.reg)];
    applyRegister(r.parsed.kind, r.parsed.tools, r.path, { skipped: r.parsed.skipped });
    if (r.parsed.kind !== state.machine) toast(`Imported ${Object.keys(r.parsed.tools).length} ${r.parsed.kind} tools — switch to ${r.parsed.kind === 'lathe' ? 'Lathe' : 'Mill'} to use them`);
    $('#regDlg').close();
  };
}

// first run: take over the tools the user's own CncSimulator5 uses (its search order)
async function autoImportRegisters() {
  if (store.get('autoImportDone', false)) return;
  store.set('autoImportDone', true);
  if (store.get('lib.mill', null) || store.get('lib.lathe', null)) return;
  const regs = await fetchRegisters();
  const got = [];
  for (const kind of ['mill', 'lathe']) {
    const r = regs.find((x) => x.parsed.kind === kind && !/_25\.ini$/i.test(x.path));
    if (r) { applyRegister(kind, r.parsed.tools, r.path, { quiet: true }); got.push(`${Object.keys(r.parsed.tools).length} ${kind}`); }
  }
  if (got.length) toast(`Found CncSimulator5 — imported your ${got.join(' and ')} tools (Tools tab)`, 5000);
}

$('#btnImportCncsim').onclick = openRegisterDialog;
$('#btnImportFile').onclick = () => $('#regInput').click();
$('#regInput').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  const parsed = parseRegister(decodeLegacyText(new Uint8Array(await f.arrayBuffer())));
  if (!parsed) { toast('Not a CncSimulator tool register (TurnTools.ini / MillTools.millreg)'); return; }
  applyRegister(parsed.kind, parsed.tools, f.name, { skipped: parsed.skipped });
};
$('#toolTable').addEventListener('change', (e) => {
  const tr = e.target.closest('tr[data-n]');
  const f = e.target.dataset.f;
  if (!tr || !f) return;
  const n = Number(tr.dataset.n);
  const lib = state.library[state.machine];
  const t = { ...(lib[n] || { num: n }) };
  t[f] = e.target.type === 'number' ? Number(e.target.value) : e.target.value;
  // changing the shape of a register tool replaces its imported outline with a generated one
  if (f !== 'name' && t.cutter && !t.generated) {
    delete t.cutter; delete t.holder; delete t.noseOffset;
    t.source = 'library';
    toast(`T${n}: outline regenerated from the new ${f}`);
  }
  lib[n] = t;
  store.set(`lib.${state.machine}`, lib);
  simulate({ time: 'keep' });
});
$('#toolTable').addEventListener('click', (e) => {
  const b = e.target.closest('[data-del]');
  if (!b) return;
  delete state.library[state.machine][b.dataset.del];
  store.set(`lib.${state.machine}`, state.library[state.machine]);
  simulate({ time: 'keep' });
});
$('#btnAddTool').onclick = () => {
  const lib = state.library[state.machine];
  let n = 1;
  while (lib[n] || state.result?.programTools[n]) n++;
  lib[n] = state.machine === 'lathe' ? { num: n, type: 'turn', noseRadius: 0.4, insertAngle: 80, name: `Tool ${n}` } : { num: n, type: 'flat', diameter: 10, fluteLength: 30, name: `Tool ${n}` };
  store.set(`lib.${state.machine}`, lib);
  renderTools();
};
$('#btnResetTools').onclick = () => {
  state.library[state.machine] = defaultLibrary(state.machine);
  state.libraryInfo[state.machine] = null;
  store.set(`lib.${state.machine}`, state.library[state.machine]);
  store.set('libInfo', state.libraryInfo);
  simulate({ time: 'keep' });
};

function renderStock() {
  const s = state.result.stock;
  const src = { program: 'from program comment', auto: 'auto-fitted to the toolpath', manual: 'set manually' }[s.source] || s.source;
  if (state.out?.src) {
    $('#stockInfo').innerHTML = `${escapeHtml(describeStock(s))}<div class="muted">${escapeHtml(src)} — the part as ${escapeHtml(state.out.src)} left it, re-clamped: ${escapeHtml(describeSetup(state.out.setup, state.out.X, state.machine))}</div>`;
    $$('input[name="stockMode"]').forEach((r) => { r.checked = r.value === state.stockMode; });
    $('#stockFields').innerHTML = '<p class="hint">This operation starts from another operation\'s result (<code>(STOCK FROM …)</code>), so manual stock does not apply to it.</p>';
    return;
  }
  $('#stockInfo').innerHTML = `${escapeHtml(describeStock(s))}<div class="muted">${src}</div>`;
  $$('input[name="stockMode"]').forEach((r) => { r.checked = r.value === state.stockMode; });
  const ms = state.manualStock[state.machine];
  const dis = state.stockMode === 'manual' ? '' : 'disabled';
  const f = (label, key, val) => `<label>${label}<input type="number" step="0.5" data-k="${key}" value="${val}" ${dis}></label>`;
  $('#stockFields').innerHTML = state.machine === 'lathe'
    ? f('Diameter', 'diameter', ms.diameter) + f('Length out of chuck', 'length', ms.length) + f('Face stock (Z+)', 'faceStock', ms.faceStock) + f('Bore Ø (tube)', 'innerDiameter', ms.innerDiameter)
    : f('X min', 'min.0', ms.min[0]) + f('X max', 'max.0', ms.max[0]) + f('Y min', 'min.1', ms.min[1]) + f('Y max', 'max.1', ms.max[1]) + f('Z min (bottom)', 'min.2', ms.min[2]) + f('Z max (top)', 'max.2', ms.max[2]);
}
$$('input[name="stockMode"]').forEach((r) => r.onchange = () => {
  state.stockMode = r.value;
  if (r.value === 'manual' && state.result) state.manualStock[state.machine] = structuredClone({ ...state.result.stock, source: undefined });
  simulate({ time: 'keep' });
});
function readStockFields() {
  const ms = structuredClone(state.manualStock[state.machine]);
  for (const inp of $$('#stockFields input')) {
    const [k, i] = inp.dataset.k.split('.');
    if (i !== undefined) ms[k][Number(i)] = Number(inp.value); else ms[k] = Number(inp.value);
  }
  return ms;
}
$('#btnStockApply').onclick = () => {
  if (state.stockMode !== 'manual') { state.stockMode = 'manual'; }
  state.manualStock[state.machine] = readStockFields();
  store.set('manualStock', state.manualStock);
  simulate({ time: 'keep' });
};
$('#btnStockInsert').onclick = () => {
  const s = state.stockMode === 'manual' ? readStockFields() : state.result.stock;
  const line = stockToComment(s);
  const lines = editor.value.split('\n');
  const existing = lines.findIndex((l) => /\(\s*(STOCK|BLANK|PART SIZE)\b/i.test(l));
  if (existing >= 0) lines[existing] = line;
  else {
    const at = lines.findIndex((l) => /^\s*[O:]\d+/i.test(l));
    lines.splice(at >= 0 ? at + 1 : lines[0]?.startsWith('%') ? 1 : 0, 0, line);
  }
  state.stockMode = 'program';
  setProgram(lines.join('\n'), { title: 'Stock line' });
  toast('Inserted ' + line);
};

// tabs
$('.tabs').onclick = (e) => {
  const b = e.target.closest('.tab');
  if (!b) return;
  $$('.tab').forEach((t) => t.classList.toggle('active', t === b));
  $$('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === b.dataset.tab));
};
function showTab(name) { $(`.tab[data-tab="${name}"]`).click(); }
$('#btnPanel').onclick = () => document.body.classList.toggle('panel-open');

// ------------------------------------------------------------------ target model (3D part)
async function loadModelFile(file) {
  const busy = $('#simBusy');
  busy.textContent = `Reading ${file.name}…`;
  busy.hidden = false;
  try {
    const model = await readModel(file.name, new Uint8Array(await file.arrayBuffer()));
    const pl = defaultPlacement(model, state.ops[0].machine);
    setModel(model, pl);
    idb.set('model', { name: model.name, format: model.format, tris: model.tris, triCount: model.triCount, bounds: model.bounds, warnings: model.warnings, placement: pl });
    toast(`Target: ${model.name} — Claude now gets its measured features and a comparison with every operation`, 5000);
    for (const w of model.warnings || []) toast(w);
  } catch (e) {
    console.error(e);
    toast(`Could not read ${file.name}: ${e.message}`, 5000);
  } finally {
    busy.hidden = true;
    busy.textContent = 'Simulating…';
  }
}

function setModel(model, placement) {
  state.model = model;
  state.placement = placement;
  rebuildTarget();
}

function rebuildTarget() {
  state.target = state.model ? buildTarget(state.model, state.placement) : null;
  viewer.setTarget(state.target ? state.target.tris : null);
  $('#targetChip').hidden = !state.target;
  renderTarget();
  scheduleCompare();
}

function removeModel() {
  state.model = null;
  state.placement = null;
  rebuildTarget();
  idb.set('model', undefined);
}

function renderTarget() {
  const t = state.target;
  $('#targetPlace').hidden = !t;
  $('#btnRemoveModel').hidden = !t;
  $('#btnStockFromModel').hidden = !t;
  $('#targetCompare').hidden = !t;
  if (!t) {
    $('#targetInfo').innerHTML = 'No model loaded. Load (or drop on the chat) an STL, STEP, IGES, OBJ or 3MF of the finished part: Claude gets its measured features, and every operation is compared with it — no screenshots needed.';
    return;
  }
  const b = t.bounds, sz = [0, 1, 2].map((k) => n1(b.max[k] - b.min[k]));
  $('#targetInfo').innerHTML = `<b>${escapeHtml(t.name)}</b> · ${t.format.toUpperCase()} · ${t.triCount.toLocaleString()} triangles · ${sz.join(' × ')} mm · ${(t.volume / 1000).toFixed(2)} cm³<div class="muted">Placed in ${escapeHtml(opName(0))}'s work coordinates: ${escapeHtml(describePlacement(state.placement))} → X${n1(b.min[0])}..${n1(b.max[0])} Y${n1(b.min[1])}..${n1(b.max[1])} Z${n1(b.min[2])}..${n1(b.max[2])}</div>`;
  const pl = state.placement;
  const set = (k, v) => { const el = $(`#targetPlace [data-pl="${k}"]`); if (el) el.value = v; };
  set('up', pl.up); set('rotZ', String(pl.rotZ || 0)); set('align', pl.align); set('scale', String(pl.scale || 1));
  set('ox', pl.offset[0]); set('oy', pl.offset[1]); set('oz', pl.offset[2]);
}

$('#targetPlace').addEventListener('change', (e) => {
  const k = e.target.dataset.pl;
  if (!k || !state.placement) return;
  const pl = { ...state.placement, offset: state.placement.offset.slice() };
  const v = e.target.value;
  if (k === 'ox' || k === 'oy' || k === 'oz') pl.offset['xyz'.indexOf(k[1])] = Number(v) || 0;
  else if (k === 'rotZ' || k === 'scale') pl[k] = Number(v);
  else pl[k] = v;
  state.placement = pl;
  rebuildTarget();
  idb.get('model').then((m) => m && idb.set('model', { ...m, placement: pl }));
});
$('#btnLoadModel').onclick = () => $('#modelInput').click();
$('#modelInput').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) loadModelFile(f); };
$('#btnRemoveModel').onclick = () => { removeModel(); toast('Target model removed'); };

// stock from the model: its size plus allowance, written into the first operation
$('#btnStockFromModel').onclick = () => {
  const t = state.target;
  if (!t) return;
  const op0 = state.ops[0];
  const b = t.bounds;
  const r1 = (v) => Math.round(v * 10) / 10;
  let line;
  if (op0.machine === 'lathe') {
    const R = Math.max(Math.abs(b.min[0]), Math.abs(b.max[0]), Math.abs(b.min[1]), Math.abs(b.max[1]));
    const D = Math.ceil(2 * R + 2), L = Math.ceil(b.max[2] - b.min[2] + 5);
    line = `(STOCK D${D} L${L} Z1)`;
  } else {
    line = `(STOCK X${r1(b.min[0] - 1)}:${r1(b.max[0] + 1)} Y${r1(b.min[1] - 1)}:${r1(b.max[1] + 1)} Z${r1(b.min[2] - 1)}:${r1(b.max[2] + 0.5)})`;
  }
  const lines = op0.text.split('\n');
  const existing = lines.findIndex((l) => /\(\s*(STOCK|BLANK|PART SIZE)\b/i.test(l) && !/STOCK\s+FROM/i.test(l));
  if (existing >= 0) lines[existing] = line;
  else {
    const at = lines.findIndex((l) => /^\s*[O:]\d+/i.test(l));
    lines.splice(at >= 0 ? at + 1 : lines[0]?.startsWith('%') ? 1 : 0, 0, line);
  }
  if (state.active !== 0) selectOp(0);
  state.stockMode = 'program';
  setProgram(lines.join('\n'), { title: 'Stock from model' });
  toast(`${opName(0)}: ${line}`);
};

// comparison with the target: computed when idle (it samples the whole part)
let compareTimer = null;
function scheduleCompare() {
  clearTimeout(compareTimer);
  if (!state.target || !state.out) { $('#targetCompare').hidden = true; $('#checkTarget').hidden = true; return; }
  compareTimer = setTimeout(() => {
    if (state.playing) { scheduleCompare(); return; }
    let txt;
    try { txt = compareTarget(state.out, state.target); } catch (e) { console.error(e); txt = `Comparison failed: ${e.message}`; }
    $('#targetCompare').textContent = `${opName(state.active)}: ${txt}`;
    $('#targetCompare').hidden = false;
    const ok = /within ±|matches within/.test(txt);
    const gouge = /GOUGE/.test(txt);
    const el = $('#checkTarget');
    el.hidden = false;
    el.className = `check-target ${gouge ? 'err' : ok ? 'ok' : 'warn'}`;
    el.innerHTML = `<b>vs target model</b> — ${escapeHtml(txt.split('\n').slice(0, 4).join(' · ').replace(/\s+/g, ' '))}`;
  }, 400);
}

// ------------------------------------------------------------------ attachments (drawings, models)
const IMG_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
async function fileToBase64(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
  return btoa(bin);
}
// drawings stay crisp: PNG drawings are only scaled down when large, photos become JPEG
async function imageAttachment(file) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, 2200 / Math.max(bmp.width, bmp.height));
  if (scale === 1 && file.size < 3.5e6 && IMG_TYPES.includes(file.type)) {
    return { kind: 'image', name: file.name, mediaType: file.type, data: await fileToBase64(file) };
  }
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * scale); c.height = Math.round(bmp.height * scale);
  const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height);
  g.drawImage(bmp, 0, 0, c.width, c.height);
  const png = file.type === 'image/png';
  let url = c.toDataURL(png ? 'image/png' : 'image/jpeg', 0.9);
  if (url.length > 4.5e6) url = c.toDataURL('image/jpeg', 0.85);
  const [head, data] = url.split(',');
  return { kind: 'image', name: file.name, mediaType: head.slice(5, head.indexOf(';')), data };
}

async function addAttachments(files) {
  for (const f of files) {
    const name = f.name || 'pasted image';
    if (isModelFile(name)) { await loadModelFile(f); continue; }
    try {
      if (f.type === 'application/pdf' || /\.pdf$/i.test(name)) {
        if (f.size > 30e6) { toast(`${name} is larger than 30 MB`); continue; }
        state.attachments.push({ kind: 'pdf', name, data: await fileToBase64(f) });
      } else if (f.type.startsWith('image/')) {
        state.attachments.push(await imageAttachment(f));
      } else if (/\.(nc|tap|gcode|ngc|cnc|mpf|txt)$/i.test(name)) {
        const text = await readTextFile(f);
        addOp({ text, fileName: name, machine: detectMachine(text) || state.machine });
      } else toast(`${name}: attach drawings (PNG, JPG, PDF) or 3D models (STL, STEP, IGES, OBJ, 3MF)`);
    } catch (e) {
      toast(`Could not attach ${name}: ${e.message}`);
    }
  }
  renderAttachments();
}

function renderAttachments() {
  const el = $('#attachments');
  el.hidden = !state.attachments.length;
  el.innerHTML = state.attachments.map((a, i) => `<span class="att" title="${escapeHtml(a.name)}">${a.kind === 'image' ? `<img src="data:${a.mediaType};base64,${a.data}" alt="">` : '<span class="ic">PDF</span>'}<span class="nm">${escapeHtml(a.name)}</span><button type="button" data-att="${i}" title="Remove">×</button></span>`).join('');
}
$('#attachments').onclick = (e) => {
  const b = e.target.closest('[data-att]');
  if (b) { state.attachments.splice(Number(b.dataset.att), 1); renderAttachments(); }
};
$('#btnAttach').onclick = () => $('#attachInput').click();
$('#attachInput').onchange = (e) => { const fs = [...e.target.files]; e.target.value = ''; addAttachments(fs); };
for (const zone of [$('#composer'), $('#chat')]) {
  zone.addEventListener('dragover', (e) => { if ([...e.dataTransfer.types].includes('Files')) { e.preventDefault(); $('#composer').classList.add('drop'); } });
  zone.addEventListener('dragleave', () => $('#composer').classList.remove('drop'));
  zone.addEventListener('drop', (e) => {
    $('#composer').classList.remove('drop');
    if (!e.dataTransfer.files.length) return;
    e.preventDefault();
    showTab('claude');
    addAttachments([...e.dataTransfer.files]);
  });
}
$('#prompt').addEventListener('paste', (e) => {
  const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith('image/'));
  if (files.length) { e.preventDefault(); addAttachments(files); }
});

// ------------------------------------------------------------------ report for Claude
// simulation report of one operation (default: the active tab)
function buildReport(out = state.out, { brief = false } = {}) {
  if (!out) return 'No simulation yet.';
  const r = out.result;
  const idx = state.ops.indexOf(out.op);
  const issues = sortIssues(out.issues);
  const c = countLevels(issues);
  const lines = out.op.text.split('\n');
  const machine = r.machine === 'lathe' ? 'lathe' : 'mill';
  const src = out.src
    ? `the part exactly as ${out.src} left it, re-clamped for this setup`
    : { program: 'from the (STOCK) comment', auto: 'auto-fitted to the toolpath because the program has no (STOCK) line', manual: 'set manually in the UI' }[r.stock.source] || r.stock.source;
  const o = [];
  o.push(`Simulation report — ${idx >= 0 ? opName(idx) : 'program'} (${out.op.fileName}), ${machine}, ${lines.length} lines`);
  o.push(`Result: ${c.error} error(s), ${c.warning} warning(s), ${c.info} note(s).`);
  if (state.ops.length > 1 || out.engine === 'voxel') o.push(`Setup: ${describeSetup(out.setup, out.X, machine)}.`);
  o.push(`Stock: ${describeStock(r.stock)} — ${src}.`);
  if (!brief) {
    o.push(`Tools used: ${r.usedTools.map((n) => `${describeTool(r.tools[n], machine)} [${r.tools[n].source || 'library'}]`).join('; ') || 'none'}`);
    o.push(`Cycle time: ${fmtTimeLong(r.totalTime)} · feed ${(r.stats.feedLength / 1000).toFixed(2)} m · rapid ${(r.stats.rapidLength / 1000).toFixed(2)} m · ${r.stats.toolChanges} tool change(s)`);
    o.push(`Removed material in this operation: ${(removedVolume(out) / 1000).toFixed(2)} cm³`);
    const env = Object.entries(toolEnvelopes(r));
    if (env.length) {
      o.push('Cut envelope per tool (feed moves, work coordinates):');
      for (const [n, e] of env) {
        o.push(machine === 'lathe'
          ? `  T${n}: X Ø${n3(e.min[0] * 2)}..Ø${n3(e.max[0] * 2)}  Z${n3(e.min[2])}..${n3(e.max[2])} · ${e.length.toFixed(0)} mm cut, ${fmtTimeLong(e.time)}`
          : `  T${n}: X${n3(e.min[0])}..${n3(e.max[0])}  Y${n3(e.min[1])}..${n3(e.max[1])}  Z${n3(e.min[2])}..${n3(e.max[2])} · ${e.length.toFixed(0)} mm cut, ${fmtTimeLong(e.time)}`);
      }
    }
    const eng = engagementReport(out);
    if (eng) o.push(eng);
    o.push(partSummary(out));
    if (state.target) o.push(compareTarget(out, state.target));
  }
  const shown = brief ? issues.filter((m) => m.level !== 'info') : issues;
  if (shown.length) {
    o.push('Issues:');
    for (const m of shown.slice(0, brief ? 12 : 40)) {
      const ln = m.callLine === null || m.callLine === undefined ? '' : ` line ${m.callLine + 1} \`${(lines[m.callLine] || '').trim().slice(0, 60)}\``;
      o.push(`  ${m.level.toUpperCase()}${ln}${m.file && m.file !== 'main' ? ` (in ${m.file} line ${m.line + 1})` : ''}: ${m.text}`);
    }
    if (shown.length > (brief ? 12 : 40)) o.push(`  … ${shown.length - (brief ? 12 : 40)} more`);
  } else {
    o.push(brief ? 'No errors or warnings.' : 'Issues: none.');
  }
  return o.join('\n');
}

const MACHINE_TEXT = {
  lathe: () => `2-axis CNC lathe, Fanuc-style control (Group A G-codes), metric, X in ${state.settings.latheRadiusMode ? 'RADIUS (radius programming enabled in settings)' : 'diameter'}, rear tool post view (X up, Z to the right, chuck on the left).`,
  mill: () => '3-axis vertical machining centre (optional A/B rotary table), Fanuc-style control, metric. Z up, the top of the stock is normally Z0.',
};

function toolLibraryText(m) {
  const lib = state.library[m];
  const info = state.libraryInfo[m];
  const o = [info
    ? `${m === 'lathe' ? 'Lathe' : 'Mill'} tool library — imported from the user's CncSimulator5 register (${info.source}). These are the real tools with real insert/holder geometry; use these numbers and respect their limits (the simulator reports holder collisions):`
    : `${m === 'lathe' ? 'Lathe' : 'Mill'} tool library (built-in defaults; use these numbers when they fit):`];
  for (const n of Object.keys(lib).sort((a, b) => a - b)) o.push(`  ${describeTool(normalizeTool(lib[n], m), m, { detail: true })}`);
  return o.join('\n');
}

function buildContext() {
  syncEditorToOp();
  let outs = state.outs;
  try { outs = simulateAll(); state.outs = outs; } catch (e) { console.error(e); }
  const names = opNames(state.ops);
  const m = state.machine;
  const o = ['<simulator_state>'];
  const multi = state.ops.length > 1;
  if (multi) {
    o.push(`Job with ${state.ops.length} operations (tabs above the editor, simulated in this order — each can start from an earlier one's result with (STOCK FROM name)). The editor shows ${names[state.active]}.`);
    state.ops.forEach((op, i) => {
      const out = outs[i];
      const c = out ? countLevels(out.issues) : null;
      o.push(`  ${names[i]}${i === state.active ? ' (active)' : ''}: ${op.machine}, ${op.fileName}, ${op.text.split('\n').length} lines${out ? ` — ${describeSetup(out.setup, out.X, op.machine)}; ${c.error} error(s), ${c.warning} warning(s)` : ''}`);
    });
  }
  o.push(`Machine${multi ? ` for ${names[state.active]}` : ''}: ${MACHINE_TEXT[m]()}`);
  const machines = new Set([m, ...state.ops.map((op) => op.machine)]);
  for (const mm of [m, ...[...machines].filter((x) => x !== m)]) o.push(toolLibraryText(mm));
  if (state.subprograms.length) o.push(`Loaded subprogram files: ${state.subprograms.map((s) => s.name).join(', ')}`);
  if (state.target) {
    o.push('<target_model>');
    o.push(`The user loaded a 3D model of the finished part. It is placed in the job's part frame (${state.ops.length > 1 ? `${names[0]}'s` : 'the program\'s'} work coordinates) and every operation's result is compared with it (see "Compared with the target model" in each report). Its measured features, in ${names[state.active]}'s work coordinates:`);
    o.push(targetSummary(state.target, outs[state.active] || state.out));
    o.push('</target_model>');
  }
  // the other operations: program + short report (the active one comes last, in full)
  state.ops.forEach((op, i) => {
    if (i === state.active) return;
    const lines = op.text.split('\n');
    o.push(`${names[i]} program (${op.fileName}, ${op.machine}, ${lines.length} lines):`);
    o.push('<program>');
    o.push(lines.length > 400 ? `${lines.slice(0, 300).join('\n')}\n… (${lines.length - 300} more lines)` : op.text);
    o.push('</program>');
    if (outs[i]) o.push(buildReport(outs[i], { brief: i > state.active }));
  });
  const text = editor.value;
  const r = state.result;
  if (r && Object.keys(r.programTools).length) {
    o.push(`Tools defined in the ${multi ? 'active' : 'current'} program (override the library):`);
    for (const n of Object.keys(r.programTools)) o.push(`  ${describeTool(r.tools[n], m, { detail: true })}`);
  }
  if (text.trim()) {
    const lines = text.split('\n');
    o.push(`${multi ? `Active operation ${names[state.active]}` : 'Current program'} in the editor (${state.fileName}, ${lines.length} lines):`);
    o.push('<program>');
    o.push(text);
    o.push('</program>');
    o.push('Latest simulation of this program:');
    o.push(buildReport(outs[state.active] || state.out));
  } else {
    o.push(`The editor${multi ? ` (${names[state.active]})` : ''} is empty.`);
  }
  const sel = editor.selection();
  if (sel) o.push(`The user has selected lines ${sel.from + 1}–${sel.to + 1}:\n<selection>\n${sel.text}\n</selection>`);
  o.push('</simulator_state>');
  return o.join('\n');
}

// ------------------------------------------------------------------ Claude
const chatView = new ChatView($('#chat'), {
  onRestoreVersion: restoreVersion,
  onImageClick: (src) => { $('#imgFull').src = src; $('#imgDlg').showModal(); },
});
chatView.onOpenSettings = () => openSettings();
chatView.onBusy = (b) => {
  $('#btnSend').hidden = b;
  $('#btnStop').hidden = !b;
  $('#quickChips').classList.toggle('hidden-in-chat', agent.history.length > 0);
  if (!b && state.wroteThisRun) {
    state.wroteThisRun = false;
    // show the new program being machined
    pause();
    seek(0, true);
    play();
  }
};
chatView.onUsage = (u) => {
  $('#claudeStatus').title = `This session: ${(u.input / 1000).toFixed(1)}k input (+${(u.cacheRead / 1000).toFixed(1)}k cached) · ${(u.output / 1000).toFixed(1)}k output tokens`;
};

// the operation a streamed program is shown in, with its program from before (put back if
// Claude's program turns out to be for another operation)
let preWrite = null;

// index of the operation called `name` (by (SETUP …) name or file name); a new name adds one
function resolveOp(name, machine) {
  if (!name) return state.active;
  const want = String(name).trim();
  const names = opNames(state.ops).map((n) => n.toUpperCase());
  let i = names.indexOf(want.toUpperCase());
  if (i < 0) i = state.ops.findIndex((o) => o.fileName.toLowerCase() === want.toLowerCase());
  if (i === state.active) return i;
  if (preWrite && preWrite.opId === activeOp().id) editor.setValue(preWrite.text, { silent: true });
  editor.setStreaming(false);
  if (i >= 0) selectOp(i);
  else {
    addOp({ text: '', fileName: `${want.toLowerCase().replace(/[^a-z0-9_.-]+/g, '-')}.nc`, machine: machine === 'lathe' || machine === 'mill' ? machine : state.machine });
    i = state.active;
  }
  preWrite = { opId: activeOp().id, text: activeOp().text };
  return i;
}

const agent = new ClaudeAgent({
  getSettings: () => ({ model: state.settings.model, effort: state.settings.effort, apiKey: state.settings.apiKey }),
  buildContext,
  view: chatView,
  tools: {
    beginWrite() {
      pause();
      preWrite = { opId: activeOp().id, text: editor.value };
      editor.setStreaming(true);
      editor.setValue('', { silent: true });
    },
    prepareWrite({ operation, machine }) {
      if (operation) resolveOp(operation, machine);
      if ((machine === 'mill' || machine === 'lathe') && machine !== state.machine) setMachine(machine, { resim: false });
      editor.setStreaming(true);
      editor.setValue('', { silent: true });
    },
    streamWrite(text) { editor.streamTo(text); },
    endWrite() { editor.setStreaming(false); },
    // the reply ended before a streamed program was run: show the operation's real program again
    abandonWrite() {
      if (!preWrite) return;
      preWrite = null;
      editor.setStreaming(false);
      if (editor.value !== activeOp().text) {
        editor.setValue(activeOp().text, { silent: true });
        simulate({ time: 'keep' });
        toast('The unfinished program from Claude was not applied');
      }
    },
    async write_program(input, card) {
      editor.setStreaming(false);
      const i = resolveOp(input.operation, input.machine);
      if ((input.machine === 'mill' || input.machine === 'lathe') && input.machine !== state.machine) setMachine(input.machine, { resim: false });
      let text = input.gcode.replace(/\r\n?/g, '\n');
      // a new operation named by Claude keeps that name even when it is not the positional one
      if (input.operation && opNames(state.ops)[i].toUpperCase() !== String(input.operation).trim().toUpperCase() && !/\(\s*(SETUP|OPERATION)\b/i.test(text)) {
        const lines = text.split('\n');
        const at = lines.findIndex((l) => /^\s*[O:]\d+/i.test(l));
        lines.splice(at >= 0 ? at + 1 : lines[0]?.startsWith('%') ? 1 : 0, 0, `(SETUP ${String(input.operation).trim().toUpperCase()})`);
        text = lines.join('\n');
      }
      editor.setValue(text, { silent: true });
      syncEditorToOp();
      saveOps();
      preWrite = null;
      const v = addVersion(input.title || 'Claude', text, 'claude');
      await nextFrame();
      simulate({ time: 'end' });
      state.wroteThisRun = true;
      const report = buildReport();
      const c = counts();
      card?.done({ summary: { errors: c.error, warnings: c.warning, lines: text.split('\n').length, time: fmtTime(state.total), report, versionId: v.id } });
      return { content: report };
    },
    async inspect_part(input, card) {
      syncEditorToOp();
      const outs = simulateAll();
      state.outs = outs;
      const names = opNames(state.ops);
      const i = input.operation ? names.findIndex((n) => n.toUpperCase() === String(input.operation).trim().toUpperCase()) : state.active;
      if (i < 0) {
        const msg = `There is no operation called ${input.operation}. Operations: ${names.join(', ')}.`;
        card?.done({ error: msg });
        return { content: msg, isError: true };
      }
      const out = outs[i];
      await nextFrame();
      let txt, title;
      switch (input.query) {
        case 'summary': txt = partSummary(out) + (state.target ? `\n${compareTarget(out, state.target)}` : ''); title = 'summary'; break;
        case 'view': title = `seen from ${input.view || 'top'}`; txt = viewDetail(out, input.view || 'top'); break;
        case 'section': title = `section ${String(input.axis).toUpperCase()}=${input.at}`; txt = section(out, String(input.axis).toLowerCase(), Number(input.at)); break;
        case 'probe': title = `${input.points.length} probe${input.points.length > 1 ? 's' : ''}`; txt = probe(out, input.points); break;
        case 'compare': title = 'vs target'; txt = state.target ? compareTarget(out, state.target) : 'No target model is loaded (the user can load one in the Stock tab or drop it on the chat).'; break;
        case 'target': title = 'target model'; txt = state.target ? targetSummary(state.target, out) : 'No target model is loaded.'; break;
        case 'engagement': title = 'tool engagement'; txt = engagementReport(out, { detail: true }) || 'No feed moves of milling or turning tools in this operation.'; break;
      }
      card?.done({ title: `${names[i]} · ${title}`, text: txt });
      return { content: `${names[i]} (${out.op.fileName}, ${out.op.machine}), work coordinates of ${names[i]}:\n${txt}` };
    },
    async view_simulation(input, card) {
      if (input.operation) {
        const names = opNames(state.ops).map((n) => n.toUpperCase());
        const i = names.indexOf(String(input.operation).trim().toUpperCase());
        if (i >= 0 && i !== state.active) { selectOp(i); await nextFrame(); }
      }
      const views = Array.isArray(input.views) && input.views.length ? input.views.filter((v) => ['iso', 'top', 'front', 'side'].includes(v)).slice(0, 4) : state.machine === 'lathe' ? ['iso', 'front'] : ['iso', 'top'];
      pause();
      seek(state.total, true);
      const hadSection = viewer.section;
      const vis = { ...viewer.visible };
      if (state.machine === 'lathe' && !hadSection) viewer.setSection(true);
      viewer.setVisible('tool', false);
      await nextFrame();
      const caps = viewer.capture(views);
      if (state.machine === 'lathe' && !hadSection) viewer.setSection(false);
      viewer.setVisible('tool', vis.tool);
      card?.done({ images: caps });
      const legend = state.machine === 'lathe'
        ? 'Lathe render, ¾ section (a quarter is cut away so the profile and any bore are visible; the cut face is beige). Horizontal axis = Z (chuck on the left, front face on the right), vertical = X. Grey = untouched bar; tinted = surfaces cut by each tool:'
        : 'Mill render. Grey = untouched stock top/walls; tinted = surfaces cut by each tool (darker = deeper); holes through the bottom show the background. Dashed red = rapids, solid = feed moves. Top view: +X right, +Y up. Tool colours:';
      const toolsTxt = state.result.usedTools.map((n) => `T${n} ${state.result.tools[n].color}`).join(', ');
      const content = [{ type: 'text', text: `${legend} ${toolsTxt}.` }];
      for (const c of caps) {
        content.push({ type: 'text', text: `View: ${c.view}` });
        content.push({ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: c.dataUrl.split(',')[1] } });
      }
      return { content };
    },
  },
});

async function send(textOverride) {
  const ta = $('#prompt');
  const text = (textOverride ?? ta.value).trim();
  if (!text || agent.busy) return;
  ta.value = '';
  showTab('claude');
  const images = state.attachments.filter((a) => a.kind === 'image').map((a) => ({ mediaType: a.mediaType, data: a.data }));
  const documents = state.attachments.filter((a) => a.kind === 'pdf').map((a) => ({ name: a.name, data: a.data }));
  if ($('#attachView').checked) {
    const [cap] = viewer.capture([viewer.lastView || 'iso']);
    images.push({ mediaType: 'image/jpeg', data: cap.dataUrl.split(',')[1] });
    $('#attachView').checked = false;
  }
  state.attachments = [];
  renderAttachments();
  await agent.send(text, { images, documents });
}
function askFix() {
  send('The simulator reports problems with the current program (see the latest simulation report). Fix them and keep everything else as it is.');
}
$('#btnSend').onclick = () => send();
$('#btnStop').onclick = () => agent.stop();
$('#btnNewChat').onclick = () => { agent.reset(); $('#quickChips').classList.remove('hidden-in-chat'); toast('New conversation'); };
$('#prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});

const CHIPS = {
  mill: [
    'Face the top 1 mm with the face mill',
    'Pocket 40×30 mm, 6 deep, R5 corners, in the centre',
    'Outside contour 80×60 with R8 corners, 10 deep, climb milling with G41',
    '4× M6 tapped holes on a 60×40 rectangle',
    'Engrave the text CNC, 0.5 mm deep, centred',
    'Explain what this program does, block by block',
  ],
  lathe: [
    'Face, then turn Ø30 for 40 mm with a 1×45° chamfer',
    'Rough and finish this profile with G71/G70: Ø20×15, Ø28×20, R3 fillet, Ø36 to Z-50',
    'Add an M16×1.5 thread, 18 mm long, with a relief groove',
    'Centre drill and drill Ø10 × 25 deep on the axis',
    'Explain what this program does, block by block',
  ],
};
function renderChips() {
  $('#quickChips').innerHTML = CHIPS[state.machine].map((c) => `<button>${escapeHtml(c)}</button>`).join('');
}
$('#quickChips').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  $('#prompt').value = b.textContent;
  $('#prompt').focus();
};

// ------------------------------------------------------------------ settings & status
function saveSettings() { store.set('settings', state.settings); }
function openSettings() {
  const s = state.settings;
  $('#apiKey').value = s.apiKey || '';
  $('#setRapidMill').value = s.rapidMill;
  $('#setRapidLathe').value = s.rapidLathe;
  $('#setToolChange').value = s.toolChangeTime;
  $('#setMaxRpm').value = s.maxRpm;
  $('#setResolution').value = s.resolution;
  $('#setRadiusMode').checked = !!s.latheRadiusMode;
  $('#setBlockDelete').checked = !!s.blockDelete;
  $('#settingsDlg').showModal();
}
$('#btnSettings').onclick = openSettings;
$('#claudeStatus').onclick = openSettings;
$('#settingsDlg').addEventListener('close', () => {
  const s = state.settings;
  s.apiKey = $('#apiKey').value.trim();
  s.rapidMill = Number($('#setRapidMill').value) || s.rapidMill;
  s.rapidLathe = Number($('#setRapidLathe').value) || s.rapidLathe;
  s.toolChangeTime = Number($('#setToolChange').value) || 0;
  s.maxRpm = Number($('#setMaxRpm').value) || s.maxRpm;
  s.resolution = Math.min(900, Math.max(100, Number($('#setResolution').value) || 420));
  s.latheRadiusMode = $('#setRadiusMode').checked;
  s.blockDelete = $('#setBlockDelete').checked;
  saveSettings();
  refreshStatus();
  simulate({ time: 'keep' });
});

let serverStatus = null;
async function refreshStatus() {
  try {
    serverStatus = serverStatus || await (await fetch('/api/status')).json();
  } catch {
    serverStatus = { ok: false };
  }
  const el = $('#claudeStatus');
  el.classList.remove('ok', 'nokey', 'mock');
  const label = el.querySelector('.label');
  if (!serverStatus.ok) { label.textContent = 'Server offline'; el.classList.add('nokey'); return; }
  if (serverStatus.mock) { label.textContent = 'Claude (mock)'; el.classList.add('mock'); }
  else if (serverStatus.serverKey || state.settings.apiKey) { label.textContent = 'Claude ready'; el.classList.add('ok'); }
  else { label.textContent = 'Add API key'; el.classList.add('nokey'); }
  $('#keyHint').textContent = serverStatus.serverKey
    ? 'The server already has ANTHROPIC_API_KEY — leave this empty to use it.'
    : 'Stored only in this browser and sent only to the local CNC Studio server.';
  const sel = $('#modelSelect');
  if (!sel.options.length && serverStatus.models) {
    sel.innerHTML = serverStatus.models.map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join('');
    sel.value = state.settings.model;
    if (!sel.value) sel.value = serverStatus.models[0].id;
  }
}
$('#modelSelect').onchange = (e) => { state.settings.model = e.target.value; saveSettings(); };
$('#effortSelect').value = state.settings.effort;
$('#effortSelect').onchange = (e) => { state.settings.effort = e.target.value; saveSettings(); };

// ------------------------------------------------------------------ keyboard
document.addEventListener('keydown', (e) => {
  const inField = e.target.closest('textarea, input, select, .CodeMirror');
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveProgram(); return; }
  if (inField) return;
  if (e.key === ' ') { e.preventDefault(); state.playing ? pause() : play(); }
  else if (e.key === 'ArrowRight') stepBlock();
  else if (e.key === 'Home') { pause(); seek(0, true); }
  else if (e.key === 'End') { pause(); seek(state.total, true); }
});

// ------------------------------------------------------------------ boot
renderSubs();
refreshStatus();
if (state.ops.some((op) => op.text.trim())) {
  for (const op of state.ops) if (op.text.trim()) state.versions.push({ id: ++state.versionSeq, opId: op.id, title: 'Restored session', text: op.text, source: 'user', at: new Date() });
  selectOp(state.active);
} else {
  const ex = EXAMPLES[0];
  selectOp(0);
  setProgram(ex.text, { name: `${ex.id}.nc`, title: ex.name, machine: ex.machine });
}
autoImportRegisters();
renderTarget();
idb.get('model').then((m) => { if (m?.tris && !state.model) setModel({ name: m.name, format: m.format, tris: m.tris, triCount: m.triCount, bounds: m.bounds, warnings: m.warnings }, m.placement); });
window.cncStudio = { state, simulate, buildReport, buildContext, viewer, editor, agent, loadModelFile };
