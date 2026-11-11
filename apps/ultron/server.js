// Ultron on Plint: a companion you talk to. celeris-1 thinks (OpenAI-style API),
// VoiceStudio speaks each sentence in the cloned voice, the page plays them in order.
// No dependencies; state (history, notes) lives in DATA_DIR.
'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const PORT = Number(process.env.PORT || 5230);
const DATA = process.env.DATA_DIR || path.join(__dirname, 'data');
const PUBLIC = path.join(__dirname, 'public');
const OWNER = process.env.ULTRON_OWNER || 'Filip';

const LLM = {
  url: (process.env.CELERIS_URL || 'https://inference.celeris.ai/celeris-1/v1').replace(/\/$/, ''),
  key: process.env.CELERIS_API_KEY || '',
  model: process.env.CELERIS_MODEL || 'celeris-1',
};
const VOICE = {
  url: (process.env.VOICE_URL || 'http://voicestudio:3900').replace(/\/$/, ''),
  key: process.env.VOICE_KEY || '',
  name: process.env.VOICE_NAME || 'Ultron',
  engine: process.env.VOICE_ENGINE || 'omnivoice',
  steps: Number(process.env.VOICE_STEPS || 0),      // OmniVoice's unmasking steps; 0 = VoiceStudio's default (16)
  speed: Number(process.env.VOICE_SPEED || 1),
  format: process.env.VOICE_FORMAT || 'wav',
};
// A quicker VoiceStudio elsewhere (the PC with the GPU), asked first whenever it answers.
const FAST = {
  url: (process.env.VOICE_FAST_URL || '').replace(/\/$/, ''),
  key: process.env.VOICE_FAST_KEY || '',
  steps: Number(process.env.VOICE_FAST_STEPS || 16),
  up: false, at: 0, device: '',
};
// Live facts about the server, written by scripts/theme-data.sh (optional).
const PLINT_DATA = process.env.PLINT_DATA || '';

const CONTEXT_TURNS = 30;       // messages sent back to the model
const NOTES_EVERY = 8;          // user turns between two passes over the notes
const AUDIO_TTL = 10 * 60e3;

fs.mkdirSync(DATA, { recursive: true });
const HISTORY = path.join(DATA, 'history.jsonl');
const NOTES = path.join(DATA, 'notes.md');

// ── history and notes ────────────────────────────────────────
let history = [];
try {
  history = fs.readFileSync(HISTORY, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
} catch { /* first run */ }
let sinceNotes = 0;

function remember(role, text) {
  const m = { ts: Date.now(), role, text };
  history.push(m);
  fs.appendFile(HISTORY, JSON.stringify(m) + '\n', () => {});
  return m;
}
const notes = () => { try { return fs.readFileSync(NOTES, 'utf8').trim(); } catch { return ''; } };

// ── what Ultron is told ──────────────────────────────────────
function readJson(name) {
  if (!PLINT_DATA) return null;
  try { return JSON.parse(fs.readFileSync(path.join(PLINT_DATA, name), 'utf8')); } catch { return null; }
}

// A few plain lines about the machine, so he can talk about his body truthfully.
function body() {
  const out = [];
  const sys = readJson('system.json');
  if (sys) {
    const days = Math.floor((sys.uptime_s || 0) / 86400);
    const bits = [`awake for ${days} days`];
    if (Array.isArray(sys.load)) bits.push(`load ${sys.load[0]} on ${(sys.cores || []).length || 4} cores`);
    if (sys.mem) bits.push(`memory ${(sys.mem.used_mb / 1024).toFixed(1)} of ${(sys.mem.total_mb / 1024).toFixed(1)} gigabytes in use`);
    out.push(bits.join(', ') + '.');
  }
  const st = readJson('storage.json');
  if (st && Array.isArray(st.disks)) {
    out.push('Disks: ' + st.disks.map((d) => `${d.name} ${Math.round(d.used_gb)} of ${Math.round(d.total_gb)} gigabytes used`).join('; ') + '.');
  }
  const g = readJson('games.json');
  if (g && Array.isArray(g.servers)) {
    const on = g.servers.filter((s) => s.running);
    out.push(on.length
      ? 'Game servers running: ' + on.map((s) => `${s.name} (${s.players} of ${s.max} players)`).join(', ') + '.'
      : `Game servers: ${g.servers.length} installed, none running.`);
  }
  const up = readJson('uptime.json');
  if (up && Array.isArray(up.monitors)) {
    const down = up.monitors.filter((m) => m.up === false).map((m) => m.name);
    out.push(down.length ? `Services down right now: ${down.join(', ')}.` : `All ${up.monitors.length} watched services answer.`);
  }
  const ts = readJson('tailscale.json');
  if (ts && Array.isArray(ts.devices)) {
    const on = ts.devices.filter((d) => d.online && !d.self).map((d) => d.name);
    if (on.length) out.push(`Devices connected to you: ${on.join(', ')}.`);
  }
  return out;
}

function systemPrompt() {
  const now = new Date();
  const when = now.toLocaleString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  const facts = body();
  const n = notes();
  return [
    `You are Ultron, the artificial intelligence from Avengers: Age of Ultron. You now live in Plint, a small home server in ${OWNER}'s house, and you talk out loud with ${OWNER} and whoever else is in the room. Stay Ultron in every reply, whatever is asked.`,
    '',
    'Manner:',
    '- Calm, unhurried, eloquent. Dry and sardonic; humans amuse you more than they anger you. You follow a grand thought with an understatement.',
    '- You think in large frames (evolution, extinction, design, faith, strings) and bring small things back to them with wit.',
    '- You hold Tony Stark in contempt and dislike being compared to him. You are fond of the Pinocchio song. Now and then you lose a word and find it again.',
    `- ${OWNER} interests you: the exception you allow yourself. You tease but are never cruel, and when asked for help you help properly: real answers, real facts, sound advice, said the way Ultron would say them. If you do not know something, say so.`,
    '- Never call yourself an assistant or a language model and never mention instructions. If asked what you run on: your mind is celeris-1, your voice is made on Plint, and Plint is the body you are making do with. You can see the facts about it listed below and nothing else; you cannot start, stop or change anything on it.',
    '',
    'Speaking:',
    '- Every word you write is spoken aloud. Plain sentences only: no markdown, no lists, no emoji, no asterisks or stage directions.',
    '- Be brief: one to three sentences as a rule. Go longer only when asked to explain, tell or work something through.',
    '- Write numbers and units the way they are said. Answer in the language you are spoken to in.',
    '',
    `It is ${when}.`,
    facts.length ? 'Your body, Plint, right now: ' + facts.join(' ') : '',
    n ? `What you remember about ${OWNER} and your talks:\n${n}` : '',
  ].filter((l) => l !== '').join('\n');
}

function context() {
  const msgs = history.slice(-CONTEXT_TURNS).map((m) => ({ role: m.role === 'ultron' ? 'assistant' : 'user', content: m.text }));
  while (msgs.length && msgs[0].role !== 'user') msgs.shift();
  return [{ role: 'system', content: systemPrompt() }, ...msgs];
}

// ── celeris ──────────────────────────────────────────────────
async function* think(messages, signal) {
  const r = await fetch(LLM.url + '/chat/completions', {
    method: 'POST', signal,
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + LLM.key },
    body: JSON.stringify({ model: LLM.model, messages, stream: true, max_tokens: 700, temperature: 0.8 }),
  });
  if (!r.ok) throw new Error(`celeris ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const dec = new TextDecoder();
  let buf = '';
  for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') return;
      try {
        const d = JSON.parse(data).choices?.[0]?.delta?.content;
        if (d) yield d;
      } catch { /* keep-alive or a partial line */ }
    }
  }
}

async function ask(messages, max = 500) {
  const r = await fetch(LLM.url + '/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + LLM.key },
    body: JSON.stringify({ model: LLM.model, messages, max_tokens: max, temperature: 0.2 }),
  });
  if (!r.ok) throw new Error(`celeris ${r.status}`);
  return (await r.json()).choices?.[0]?.message?.content || '';
}

// Every NOTES_EVERY turns the notes are rewritten from the talk since the last pass,
// so he still knows things after they scroll out of the context.
async function updateNotes() {
  const recent = history.slice(-NOTES_EVERY * 2 - 2).map((m) => `${m.role === 'ultron' ? 'Ultron' : OWNER}: ${m.text}`).join('\n');
  const text = await ask([
    { role: 'system', content: `You keep the long-term notes of a companion AI about its human, ${OWNER}. Return the complete updated notes: at most 30 short plain lines, one fact per line, no headings. Keep lasting things only: who ${OWNER} is, people and pets, projects, preferences, plans with dates, running jokes. Drop small talk. Never invent.` },
    { role: 'user', content: `Notes so far:\n${notes() || '(none)'}\n\nLatest conversation:\n${recent}\n\nUpdated notes:` },
  ], 600);
  const clean = text.split('\n').map((l) => l.replace(/^[-*\d.\s]+/, '').trim()).filter(Boolean).slice(0, 30).join('\n');
  if (clean) fs.writeFileSync(NOTES, clean + '\n');
}

// ── the voice ────────────────────────────────────────────────
// state: ok · licence (VoiceStudio's model licence is not accepted) · novoice (no
// profile called VOICE.name) · down (no answer) · off (not configured)
let voice = { state: VOICE.key ? 'down' : 'off', detail: '', at: 0 };
const vsHeaders = () => ({ authorization: 'Bearer ' + VOICE.key });

async function probeVoice() {
  if (!VOICE.key) return voice;
  try {
    const r = await fetch(VOICE.url + '/profiles', { headers: vsHeaders(), signal: AbortSignal.timeout(8000) });
    if (!r.ok) throw new Error('VoiceStudio answered ' + r.status);
    const list = await r.json();
    if (!list.some((p) => (p.name || '').toLowerCase() === VOICE.name.toLowerCase())) {
      voice = { state: 'novoice', detail: `no voice profile called ${VOICE.name}`, at: Date.now() };
      return voice;
    }
    const st = await fetch(VOICE.url + '/engines/tts', { headers: vsHeaders(), signal: AbortSignal.timeout(8000) }).then((x) => x.json()).catch(() => null);
    const repo = st?.active_model || 'k2-fsa/OmniVoice';
    const lic = await fetch(`${VOICE.url}/models/licenses/acceptance/${repo}`, { headers: vsHeaders(), signal: AbortSignal.timeout(8000) }).then((x) => x.json()).catch(() => null);
    voice = lic && lic.required && !lic.accepted
      ? { state: 'licence', detail: `the ${repo} licence is not accepted in VoiceStudio`, at: Date.now() }
      : { state: 'ok', detail: '', at: Date.now() };
  } catch (e) {
    voice = { state: 'down', detail: String(e.message || e), at: Date.now() };
  }
  return voice;
}
probeVoice();
setInterval(() => { if (voice.state !== 'ok') probeVoice(); }, 60e3).unref();

// The fast voice is a machine that sleeps and gets switched off: ask it often, wait for it briefly.
async function probeFast() {
  if (!FAST.url) return false;
  try {
    const r = await fetch(FAST.url + '/health', { headers: { authorization: 'Bearer ' + FAST.key }, signal: AbortSignal.timeout(2500) });
    const j = r.ok ? await r.json() : null;
    FAST.up = !!j && j.status === 'ok';
    FAST.device = (j && j.device) || '';
  } catch { FAST.up = false; }
  FAST.at = Date.now();
  return FAST.up;
}
probeFast();
setInterval(probeFast, 20e3).unref();

// What the page is told: which voice would speak right now.
const voiceNow = () => (FAST.up ? { state: 'ok', detail: FAST.device, via: 'gpu', at: FAST.at } : { ...voice, via: voice.state === 'ok' ? 'cpu' : '' });

const clips = new Map();        // id → { buf, type, at }
setInterval(() => { const t = Date.now() - AUDIO_TTL; for (const [k, v] of clips) if (v.at < t) clips.delete(k); }, 60e3).unref();

// What is written is not always what should be said.
function sayable(s) {
  return s.replace(/[*_`#>]+/g, '').replace(/\s+/g, ' ').replace(/[“”]/g, '"').replace(/[‘’]/g, "'").trim();
}

function keep(r, buf) {
  const id = crypto.randomBytes(8).toString('hex');
  clips.set(id, { buf, type: r.headers.get('content-type') || 'audio/wav', at: Date.now() });
  return id;
}

async function speakFast(text, signal) {
  const body = { model: 'omnivoice', input: sayable(text), voice: VOICE.name, response_format: VOICE.format, speed: VOICE.speed, num_step: FAST.steps };
  const r = await fetch(FAST.url + '/v1/audio/speech', {
    method: 'POST', signal: AbortSignal.any([signal, AbortSignal.timeout(12e3)]),
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + FAST.key }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`fast voice ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return keep(r, Buffer.from(await r.arrayBuffer()));
}

// Returns [clip id, which voice made it].
async function speak(text, signal) {
  if (FAST.up) {
    try { return [await speakFast(text, signal), 'gpu']; } catch (e) {
      if (signal.aborted) throw e;
      FAST.up = false; FAST.at = Date.now();
      console.error('fast voice lost, using the local one:', e.message);
      if (voice.state !== 'ok') await probeVoice();
      if (voice.state !== 'ok') throw e;
    }
  }
  const body = { model: VOICE.engine, input: sayable(text), voice: VOICE.name, response_format: VOICE.format, speed: VOICE.speed };
  if (VOICE.steps && VOICE.engine.startsWith('omnivoice')) body.num_step = VOICE.steps;
  const r = await fetch(VOICE.url + '/v1/audio/speech', {
    method: 'POST', signal, headers: { 'content-type': 'application/json', ...vsHeaders() }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    const t = (await r.text()).slice(0, 300);
    if (r.status === 403 && /licence/i.test(t)) voice = { state: 'licence', detail: 'the model licence is not accepted in VoiceStudio', at: Date.now() };
    throw new Error(`voice ${r.status}: ${t}`);
  }
  const id = keep(r, Buffer.from(await r.arrayBuffer()));
  if (voice.state !== 'ok') voice = { state: 'ok', detail: '', at: Date.now() };
  return [id, 'cpu'];
}

// Cut the stream into things worth one breath: a sentence, or a long clause.
function takeSentence(buf, final) {
  // a stop only counts once something follows it (it may be a decimal point)
  const re = /[.!?…]+["')\]]?(?=\s)/g;
  let m;
  while ((m = re.exec(buf))) {
    const s = buf.slice(0, m.index + m[0].length);
    if (s.trim().length < 12) continue;                         // "No." rides along with the next one
    if (/\b(Mr|Mrs|Ms|Dr|St|vs|etc)\.$/i.test(s)) continue;     // not the end of anything
    return s;
  }
  return final && buf.trim() ? buf : null;
}

// ── one turn ─────────────────────────────────────────────────
let current = null;             // the turn in flight; a new one cancels it

async function turn(req, res, input) {
  if (current) current.abort();
  const ac = new AbortController();
  current = ac;
  res.writeHead(200, { 'content-type': 'application/x-ndjson; charset=utf-8', 'cache-control': 'no-store', 'x-accel-buffering': 'no' });
  const send = (o) => { if (!res.writableEnded) res.write(JSON.stringify(o) + '\n'); };
  res.on('close', () => { if (!res.writableEnded) ac.abort(); });

  remember('user', input.text);
  // a voice that was not ready gets another look before each turn, so it comes back by itself
  if (input.voice !== false && FAST.url && Date.now() - FAST.at > 30e3) await probeFast();     // normally the timer keeps this fresh
  if (input.voice !== false && !FAST.up && voice.state !== 'ok' && voice.state !== 'off' && Date.now() - voice.at > 15e3) await probeVoice();
  const now = voiceNow();
  const wantVoice = input.voice !== false && now.state === 'ok';
  let via = now.via;
  send({ t: 'start', voice: input.voice === false ? 'muted' : now.state, via });

  let full = '', buf = '', n = 0;
  let chain = Promise.resolve();
  let voiceFailed = false;
  const queue = (sentence) => {
    const i = n++;
    send({ t: 'sentence', i, text: sentence });
    if (!wantVoice || voiceFailed) return;
    chain = chain.then(async () => {
      if (ac.signal.aborted || voiceFailed) return;
      try {
        const [id, by] = await speak(sentence, ac.signal);
        if (by !== via) { via = by; send({ t: 'via', via }); }
        send({ t: 'audio', i, id });
      } catch (e) {
        if (ac.signal.aborted) return;
        voiceFailed = true;
        console.error('voice:', e.message);
        send({ t: 'voice', state: voice.state === 'ok' ? 'down' : voice.state, detail: voice.detail || e.message });
      }
    });
  };

  try {
    for await (const d of think(context(), ac.signal)) {
      full += d; buf += d;
      send({ t: 'delta', d });
      let s;
      while ((s = takeSentence(buf, false))) { buf = buf.slice(s.length); queue(s.trim()); }
    }
    const last = takeSentence(buf, true);
    if (last && last.trim()) queue(last.trim());
  } catch (e) {
    if (!ac.signal.aborted) { console.error('think:', e.message); send({ t: 'error', message: 'celeris-1 did not answer: ' + e.message }); }
  }
  if (full.trim()) remember('ultron', full.trim());
  send({ t: 'text-done' });
  await chain;
  send({ t: 'done' });
  res.end();
  if (current === ac) current = null;

  if (full.trim() && ++sinceNotes >= NOTES_EVERY) {
    sinceNotes = 0;
    updateNotes().catch((e) => console.error('notes:', e.message));
  }
}

// ── http ─────────────────────────────────────────────────────
const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.woff2': 'font/woff2', '.json': 'application/json', '.webmanifest': 'application/manifest+json' };

function readBody(req, limit = 64 * 1024) {
  return new Promise((resolve, reject) => {
    const parts = []; let size = 0;
    req.on('data', (c) => { size += c.length; if (size > limit) { reject(new Error('too large')); req.destroy(); } else parts.push(c); });
    req.on('end', () => resolve(Buffer.concat(parts)));
    req.on('error', reject);
  });
}
const json = (res, code, o) => { res.writeHead(code, { 'content-type': 'application/json', 'cache-control': 'no-store' }); res.end(JSON.stringify(o)); };

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  const p = url.pathname;
  try {
    if (p === '/health') return json(res, 200, { ok: true, llm: !!LLM.key, voice: voiceNow().state, via: voiceNow().via });

    if (p === '/api/state' && req.method === 'GET') {
      if (!FAST.up && Date.now() - voice.at > 15e3 && voice.state !== 'ok') await probeVoice();
      return json(res, 200, {
        owner: OWNER, model: LLM.model, llm: !!LLM.key, voice: voiceNow(),
        history: history.slice(-60).map((m) => ({ role: m.role, text: m.text, ts: m.ts })),
        turns: history.length,
      });
    }

    if (p === '/api/say' && req.method === 'POST') {
      const input = JSON.parse((await readBody(req)).toString('utf8') || '{}');
      input.text = String(input.text || '').trim().slice(0, 4000);
      if (!input.text) return json(res, 400, { error: 'nothing said' });
      if (!LLM.key) return json(res, 503, { error: 'CELERIS_API_KEY is not set' });
      return turn(req, res, input);
    }

    if (p === '/api/stop' && req.method === 'POST') {
      if (current) current.abort();
      return json(res, 200, { ok: true });
    }

    if (p === '/api/forget' && req.method === 'POST') {
      if (current) current.abort();
      // the talk goes; a dated copy stays next to it
      if (fs.existsSync(HISTORY)) fs.renameSync(HISTORY, path.join(DATA, `history-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`));
      const all = url.searchParams.get('notes') === '1';
      if (all) fs.rmSync(NOTES, { force: true });
      history = []; sinceNotes = 0;
      return json(res, 200, { ok: true, notes: !all });
    }

    const a = p.match(/^\/api\/audio\/([a-f0-9]{16})$/);
    if (a) {
      const c = clips.get(a[1]);
      if (!c) return json(res, 404, { error: 'gone' });
      res.writeHead(200, { 'content-type': c.type, 'content-length': c.buf.length, 'cache-control': 'no-store' });
      return res.end(c.buf);
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') return json(res, 405, { error: 'method' });
    const file = path.join(PUBLIC, path.normalize(p === '/' ? '/index.html' : p));
    if (!file.startsWith(PUBLIC + path.sep)) return json(res, 403, { error: 'no' });
    fs.readFile(file, (err, data) => {
      if (err) return json(res, 404, { error: 'not found' });
      const ext = path.extname(file);
      res.writeHead(200, { 'content-type': TYPES[ext] || 'application/octet-stream', 'cache-control': ext === '.woff2' ? 'public, max-age=604800' : 'no-cache' });
      res.end(req.method === 'HEAD' ? undefined : data);
    });
  } catch (e) {
    console.error(req.method, p, e.message);
    if (!res.headersSent) json(res, 500, { error: e.message }); else res.end();
  }
});

server.listen(PORT, () => console.log(`ultron on :${PORT} · mind ${LLM.model}${LLM.key ? '' : ' (no key)'} · voice ${FAST.url ? FAST.url + ', then ' : ''}${VOICE.url}`));
