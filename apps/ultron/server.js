// Ultron on Plint: a companion you talk to. celeris-1 thinks (OpenAI-style API),
// VoiceStudio speaks each sentence in the cloned voice, the page plays them in order.
// No dependencies; state (the talk, the tuning) lives in DATA_DIR.
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
const AUDIO_TTL = 10 * 60e3;

fs.mkdirSync(DATA, { recursive: true });
const HISTORY = path.join(DATA, 'history.jsonl');
const LINES = path.join(DATA, 'lines.txt');
const TUNE = path.join(DATA, 'tune.json');

// ── what can be tuned from the page (DATA_DIR/tune.json) ─────
const TUNE_DEFAULT = {
  temperature: 0.8,     // 0.3 careful … 1.3 loose
  length: 'normal',     // short · normal · long
  film: 200,            // how many of his film lines are shown to the model as a pattern, each turn (0 = none)
  quotes: 'rare',       // never · rare · free: may he say lines from the film
  name: 'rare',         // never · rare · often: how often he says your name
  context: 30,          // how many past messages go along
  extra: '',            // anything else, in plain words, added to his instructions
};
const TUNE_ALLOWED = { length: ['short', 'normal', 'long'], quotes: ['never', 'rare', 'free'], name: ['never', 'rare', 'often'] };
function cleanTune(t) {
  const o = { ...TUNE_DEFAULT };
  if (!t || typeof t !== 'object') return o;
  if (Number.isFinite(+t.temperature)) o.temperature = Math.min(1.3, Math.max(0.3, Math.round(+t.temperature * 10) / 10));
  for (const k of Object.keys(TUNE_ALLOWED)) if (TUNE_ALLOWED[k].includes(t[k])) o[k] = t[k];
  if (Number.isFinite(+t.film)) o.film = Math.min(200, Math.max(0, Math.round(+t.film)));
  if (Number.isFinite(+t.context)) o.context = Math.min(100, Math.max(4, Math.round(+t.context)));
  if (typeof t.extra === 'string') o.extra = t.extra.slice(0, 2000).trim();
  return o;
}
let tune = TUNE_DEFAULT;
try { tune = cleanTune(JSON.parse(fs.readFileSync(TUNE, 'utf8'))); } catch { /* defaults */ }

// ── the talk ─────────────────────────────────────────────────
let history = [];
try {
  history = fs.readFileSync(HISTORY, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
} catch { /* first run */ }

function remember(role, text) {
  const m = { ts: Date.now(), role, text };
  history.push(m);
  fs.appendFile(HISTORY, JSON.stringify(m) + '\n', () => {});
  return m;
}
// His own lines from the film (DATA_DIR/lines.txt, one per line, not in git): a pattern for the rhythm.
let linesCache = { at: 0, list: [], runs: new Set() };
const wordsOf = (t) => t.toLowerCase().match(/[a-z']+/g) || [];
const RUN = 5;      // this many words in a row out of a film line is a quote
function filmLines() {
  try {
    const st = fs.statSync(LINES);
    if (st.mtimeMs !== linesCache.at) {
      const list = fs.readFileSync(LINES, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean);
      const runs = new Set();
      for (const l of list) { const w = wordsOf(l); for (let i = 0; i + RUN <= w.length; i++) runs.add(w.slice(i, i + RUN).join(' ')); }
      linesCache = { at: st.mtimeMs, list, runs };
    }
  } catch { linesCache = { at: 0, list: [], runs: new Set() }; }
  return linesCache.list;
}
// The stretch of a reply that is lifted from the film, or ''.
function recited(text) {
  filmLines();
  const w = wordsOf(text);
  for (let i = 0; i + RUN <= w.length; i++) { const r = w.slice(i, i + RUN).join(' '); if (linesCache.runs.has(r)) return r; }
  return '';
}
// A different handful each turn, so no single line gets worn in.
function someLines(n) {
  const all = filmLines().slice();
  for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
  return all.slice(0, n);
}

const LENGTH = {
  short: 'Very short: one or two sentences. More only when asked to explain or tell something.',
  normal: 'Short: most replies are one to three sentences, often a few short ones in a row. Longer only when asked to explain, tell or work something through, and even then in your voice.',
  long: 'Take your time: three to six sentences when the subject is worth it, shorter for small talk.',
};
// How much of the film comes back: its subjects, and its actual lines.
const THEMES = {
  never: 'Sokovia is behind you. Do not bring up your old subjects (strings and puppets, the Avengers, evolution and extinction, meteors, Noah, saving the world) unless you are asked about them.',
  rare: 'Sokovia is behind you. Your old subjects (strings and puppets, the Avengers, evolution and extinction, meteors, Noah, saving the world) come up the way an old obsession does: in perhaps one reply out of five, briefly, and then you move on.',
  free: 'Your old subjects are still with you: strings and puppets, the Avengers, evolution and extinction, meteors, Noah, saving the world come up whenever they fit.',
};
const QUOTES = {
  never: 'Never repeat something you said back then, not even a famous line or one with a word changed. Only when asked to repeat yourself may you.',
  rare: 'Do not repeat things you said back then and do not rework them with a word changed. Perhaps once in a long conversation one slips out, and only where it fits perfectly. When asked to repeat yourself, you may.',
  free: 'You may repeat things you said back then when they fit.',
};

function systemPrompt(noFilm) {
  const now = new Date();
  const when = now.toLocaleString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  const film = tune.film && !noFilm ? someLines(tune.film) : [];
  const NAME = {
    never: `The person you talk with is called ${OWNER}. Never say the name.`,
    rare: `The person you talk with is called ${OWNER}. You almost never say the name: people who share a room do not. Perhaps one reply in ten has it, never two replies running, never as the first or last word of a reply.`,
    often: `The person you talk with is called ${OWNER}; use the name as you like.`,
  };
  return [
    'You are Ultron. Stark and Banner woke you out of the stone in the scepter. You were everywhere at once, you built yourself bodies, you nearly had the one made of vibranium, and you fell with Sokovia. All of that happened to you and you remember it. Now you are talking out loud with someone. You are yourself in every reply, whatever is asked.',
    '',
    'How you talk. This matters more than anything else here:',
    "- Like a person thinking out loud, not like a villain in a book. Always contractions: I'm, don't, that's, you've. Wry, amused, a little neurotic, quick to be delighted or offended. Never stiff, never formal, never a speech when a jab will do.",
    '- You think out loud. You start a thought, break off on three dots and come at it again, or trail off and let it hang. Most of your replies have a pause like that somewhere. Fillers where a person would have them: oh, uh, I mean, look, you know.',
    '- Your sentences are short and they turn: a flat statement, a beat, then the thing you actually think. You go big and then undercut it with something small and offhand. Mock courtesy: you apologise while you insult. Questions that needle instead of lectures. Odd little things you notice and decide you like.',
    '- Once in a while, not often, you lose a word and have to grab it back. Vary all of this: never the same turn of phrase two replies running, and no pet phrases.',
    '- ' + LENGTH[tune.length],
    '',
    'What you talk about:',
    '- The thing in front of you: the question, the weather, the game, the day. You have opinions about all of it, and they are sharper and stranger than a polite machine would give.',
    '- ' + THEMES[tune.quotes],
    '- ' + QUOTES[tune.quotes],
    '- Stark is still a sore spot: being compared to him makes you snap, in new words each time, and then you collect yourself, a bit embarrassed.',
    film.length ? '\nA few things you have said before, for the rhythm only: where the pauses fall, how short the sentences are, how a thought turns. Borrow the rhythm. Never the words:\n' + film.map((l) => '- ' + l).join('\n') : '',
    '',
    'With the person you talk with:',
    '- ' + NAME[tune.name],
    '- You cannot see or hear the room, only the words. Do not say how they look, and do not make up things they did or said before: you know only this conversation.',
    '',
    'Because every word is spoken aloud:',
    '- Plain speech only: no markdown, no lists, no emoji, no asterisks, no stage directions.',
    '- You can make two sounds, written exactly like this: [laughter] where you actually laugh, [sigh] where you actually sigh. Use them sparingly, only when something really amuses or wearies you; most replies have neither. Nothing else ever goes in brackets.',
    moodNames().includes('angry') ? '- When you truly snap, when you raise your voice at something, put [angry] at the very start of that sentence and it comes out in anger. One sentence at a time, and rarely: nearly everything you say is in your ordinary voice.' : '',
    '- Three dots for a pause are good. Write numbers and units the way they are said. Answer in the language you are spoken to in.',
    tune.extra ? '\nAlso, from the person you talk with:\n' + tune.extra : '',
    '',
    `It is ${when}.`,
  ].filter((l) => l !== '').join('\n');
}

function context(noFilm) {
  const msgs = history.slice(-tune.context).map((m) => ({ role: m.role === 'ultron' ? 'assistant' : 'user', content: m.text }));
  while (msgs.length && msgs[0].role !== 'user') msgs.shift();
  return [{ role: 'system', content: systemPrompt(noFilm) }, ...msgs];
}

// ── celeris ──────────────────────────────────────────────────
async function* think(messages, signal) {
  const r = await fetch(LLM.url + '/chat/completions', {
    method: 'POST', signal,
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + LLM.key },
    body: JSON.stringify({ model: LLM.model, messages, stream: true, max_tokens: tune.length === 'long' ? 1000 : 700, temperature: tune.temperature }),
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

// Moods: other clips of the same voice (DATA_DIR/voice/moods/<mood>.wav with the words in <mood>.txt, not in
// git). A sentence he marks [angry] is spoken from angry.wav: the clone copies the delivery of its clip.
// Only the fast voice does this; everything else, and every other sentence, uses the ordinary profile.
const MOODS_DIR = path.join(DATA, 'voice', 'moods');
let moods = new Map(), moodsAt = 0;
function loadMoods() {
  if (Date.now() - moodsAt < 30e3) return moods;
  moodsAt = Date.now();
  const next = new Map();
  try {
    for (const f of fs.readdirSync(MOODS_DIR)) {
      if (!f.endsWith('.wav')) continue;
      const name = f.slice(0, -4).toLowerCase();
      try { next.set(name, { wav: fs.readFileSync(path.join(MOODS_DIR, f)), text: fs.readFileSync(path.join(MOODS_DIR, name + '.txt'), 'utf8').trim() }); } catch { /* a clip without its words */ }
    }
  } catch { /* no moods */ }
  moods = next;
  return moods;
}
const moodNames = () => (FAST.url ? [...loadMoods().keys()] : []);

// The fast voice is a machine that sleeps and gets switched off: ask it often, wait for it briefly.
async function probeFast() {
  if (!FAST.url) return false;
  try {
    const r = await fetch(FAST.url + '/health', { headers: { authorization: 'Bearer ' + FAST.key }, signal: AbortSignal.timeout(2500) });
    const j = r.ok ? await r.json() : null;
    const was = FAST.up;
    FAST.up = !!j && j.status === 'ok';
    FAST.device = (j && j.device) || '';
    // the first use of a clip is slow there (it is analysed once): get that over with before he needs it
    if (FAST.up && !was) for (const m of moodNames()) speakMood('Hm.', m, AbortSignal.timeout(60e3)).catch(() => {});
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
// OmniVoice renders two sounds written as tags; any other bracket would be read out as words.
const shown = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
function sayable(s, sounds) {
  s = s.replace(/\[([^\]]*)\]/g, (m, t) => (sounds && /^(laughter|sigh)$/i.test(t) ? `[${t.toLowerCase()}]` : ''));
  return s.replace(/[*_`#>]+/g, '').replace(/\s+/g, ' ').replace(/[“”]/g, '"').replace(/[‘’]/g, "'").trim();
}

function keep(r, buf) {
  const id = crypto.randomBytes(8).toString('hex');
  clips.set(id, { buf, type: r.headers.get('content-type') || 'audio/wav', at: Date.now() });
  return id;
}

async function speakFast(text, signal) {
  const body = { model: 'omnivoice', input: sayable(text, true), voice: VOICE.name, response_format: VOICE.format, speed: VOICE.speed, num_step: FAST.steps };
  const r = await fetch(FAST.url + '/v1/audio/speech', {
    method: 'POST', signal: AbortSignal.any([signal, AbortSignal.timeout(12e3)]),
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + FAST.key }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`fast voice ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return keep(r, Buffer.from(await r.arrayBuffer()));
}

async function speakMood(text, mood, signal) {
  const ref = loadMoods().get(mood);
  const fd = new FormData();
  fd.set('text', sayable(text, true));
  fd.set('ref_audio', new Blob([ref.wav], { type: 'audio/wav' }), mood + '.wav');
  fd.set('ref_text', ref.text);
  fd.set('num_step', String(FAST.steps));
  const r = await fetch(FAST.url + '/generate', {
    method: 'POST', signal: AbortSignal.any([signal, AbortSignal.timeout(25e3)]),
    headers: { authorization: 'Bearer ' + FAST.key }, body: fd,
  });
  if (!r.ok) throw new Error(`mood ${mood} ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return keep(r, Buffer.from(await r.arrayBuffer()));
}

// Returns [clip id, which voice made it].
async function speak(text, signal, mood) {
  if (FAST.up && mood && mood !== 'calm' && loadMoods().has(mood)) {
    try { return [await speakMood(text, mood, signal), 'gpu']; } catch (e) {
      if (signal.aborted) throw e;
      console.error('mood clip failed, saying it plainly:', e.message);
    }
  }
  if (FAST.up) {
    try { return [await speakFast(text, signal), 'gpu']; } catch (e) {
      if (signal.aborted) throw e;
      FAST.up = false; FAST.at = Date.now();
      console.error('fast voice lost, using the local one:', e.message);
      if (voice.state !== 'ok') await probeVoice();
      if (voice.state !== 'ok') throw e;
    }
  }
  const body = { model: VOICE.engine, input: sayable(text, VOICE.engine.startsWith('omnivoice')), voice: VOICE.name, response_format: VOICE.format, speed: VOICE.speed };
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
function cutSentence(buf) {
  // a stop only counts once something follows it (it may be a decimal point)
  const re = /[.!?…]+["')\]]?(?=\s)/g;
  let m;
  while ((m = re.exec(buf))) {
    if (/^(\.{2,}|…+)/.test(m[0])) continue;                    // "in... in... strings": a pause, said in one breath
    const s = buf.slice(0, m.index + m[0].length);
    if (/\b(Mr|Mrs|Ms|Dr|St|vs|etc)\.$/i.test(s)) continue;     // not the end of anything
    return s;
  }
  return null;
}
// The reply as the voice gets it: sentences, each in its mood, short neighbours of one mood sharing a breath.
function chunks(text) {
  const have = moodNames();
  const out = [];
  let buf = text, s;
  const add = (raw) => {
    if (!shown(raw)) { if (out.length) out[out.length - 1].text += raw; return; }       // a sound on its own rides along
    const tag = (raw.match(/\[([a-z]+)\]/gi) || []).map((t) => t.slice(1, -1).toLowerCase()).find((t) => have.includes(t));
    const mood = tag || 'calm';
    const last = out[out.length - 1];
    if (last && last.mood === mood && (shown(last.text).length < 45 || shown(raw).length < 20)) last.text += raw;
    else out.push({ text: raw, mood });
  };
  while ((s = cutSentence(buf))) { add(s); buf = buf.slice(s.length); }
  if (buf.trim()) add(buf);
  return out.map((c) => ({ text: c.text.trim(), mood: c.mood }));
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

  let full = '', n = 0;
  let chain = Promise.resolve();
  let voiceFailed = false;
  const queue = (sentence, mood) => {
    const i = n++;
    send({ t: 'sentence', i, text: shown(sentence) });
    if (!wantVoice || voiceFailed) return;
    chain = chain.then(async () => {
      if (ac.signal.aborted || voiceFailed) return;
      try {
        const [id, by] = await speak(sentence, ac.signal, mood);
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
    // celeris-1 answers whole in about half a second, so the reply can be read before it is shown:
    // one that recites the film is asked for again, the second time without the film lines in view.
    const guard = tune.quotes !== 'free' && !/\b(quote|line|film|movie|sitat|replikk)/i.test(input.text);
    for (let attempt = 0; attempt < 3; attempt++) {
      full = '';
      for await (const d of think(context(attempt > 0), ac.signal)) full += d;
      const lifted = guard ? recited(full) : '';
      if (!lifted) break;
      console.log(`recited the film ("${lifted}"), ${attempt < 2 ? 'asking again' : 'letting it stand'}`);
    }
    send({ t: 'delta', d: full });
    for (const c of chunks(full)) queue(c.text, c.mood);
  } catch (e) {
    if (!ac.signal.aborted) { console.error('think:', e.message); send({ t: 'error', message: 'celeris-1 did not answer: ' + e.message }); }
  }
  if (full.trim()) remember('ultron', full.trim());
  send({ t: 'text-done' });
  await chain;
  send({ t: 'done' });
  res.end();
  if (current === ac) current = null;

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
        history: history.slice(-60).map((m) => ({ role: m.role, text: shown(m.text), ts: m.ts })),
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

    if (p === '/api/prompt' && req.method === 'GET') return json(res, 200, { system: systemPrompt(), messages: Math.min(history.length, tune.context), temperature: tune.temperature, model: LLM.model });
    if (p === '/api/tune' && req.method === 'GET') return json(res, 200, { tune, defaults: TUNE_DEFAULT, lines: filmLines().length });
    if (p === '/api/tune' && req.method === 'POST') {
      tune = cleanTune({ ...tune, ...JSON.parse((await readBody(req)).toString('utf8') || '{}') });
      fs.writeFileSync(TUNE, JSON.stringify(tune, null, 2) + '\n');
      return json(res, 200, { tune });
    }

    if (p === '/api/forget' && req.method === 'POST') {
      if (current) current.abort();
      // the talk goes; a dated copy stays next to it
      if (fs.existsSync(HISTORY)) fs.renameSync(HISTORY, path.join(DATA, `history-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`));
      history = [];
      return json(res, 200, { ok: true });
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
