// Ultron · the page. Sends what you say to /api/say, shows his answer sentence by
// sentence and plays each sentence's clip in order. The face is a dot matrix: the
// mouth is driven by the sound that is playing, the eyes by what he is doing.
(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('ultron.' + k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem('ultron.' + k, v); } catch { /* private window */ } },
};
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── state lamp ───────────────────────────────────────────────
const STATES = {
  idle:      ['READY', 'ГОТОВ', ''],
  listening: ['LISTENING', 'СЛУШАЮ', 'live'],
  thinking:  ['THINKING', 'ДУМАЮ', 'busy'],
  voicing:   ['VOICING', 'СИНТЕЗ', 'busy'],
  speaking:  ['SPEAKING', 'ГОВОРЮ', 'live'],
  offline:   ['NO LINK', 'НЕТ СВЯЗИ', 'off'],
};
let state = 'idle';
function setState(s) {
  state = s;
  const [en, ru, cls] = STATES[s];
  $('state').textContent = en; $('stateRu').textContent = ru;
  $('stateBox').className = 'state ' + cls;
  $('faceBtn').setAttribute('aria-label', s === 'idle' ? 'Talk to Ultron' : 'Stop Ultron');
}

// ── sound ────────────────────────────────────────────────────
const sound = { ctx: null, an: null, bins: null, src: null };
function unlock() {              // must run inside a tap or key press
  if (!sound.ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    sound.ctx = new AC();
    sound.an = sound.ctx.createAnalyser();
    sound.an.fftSize = 256; sound.an.smoothingTimeConstant = 0.55;
    sound.an.connect(sound.ctx.destination);
    sound.bins = new Uint8Array(sound.an.frequencyBinCount);
  }
  if (sound.ctx.state === 'suspended') sound.ctx.resume();
}
function decode(buf) {
  return new Promise((ok, no) => sound.ctx.decodeAudioData(buf, ok, no));
}
function hush() {
  if (sound.src) { try { sound.src.onended = null; sound.src.stop(); } catch { /* already over */ } sound.src = null; }
}

// ── the face ─────────────────────────────────────────────────
const face = (() => {
  const cv = $('face'), g = cv.getContext('2d');
  const COLS = 37, ROWS = 45, CC = (COLS - 1) / 2;
  // half a head, in (distance from the centre line, height): mirrored when drawn
  const HEAD  = [[0, .03], [.34, .04], [.60, .15], [.72, .34], [.70, .52], [.60, .67], [.44, .85], [.20, .975], [0, .99]];
  const FIN   = [[.74, .28], [.90, .21], [.94, .50], [.75, .57]];
  const EYE   = [[.13, .46], [.50, .365], [.57, .43], [.16, .52]];
  const BROW  = [[.06, .415], [.58, .285], [.58, .325], [.06, .45]];
  const MOUTH = [[0, .70], [.34, .68], [.43, .745], [.27, .865], [0, .88]];
  const NOSE  = [[0, .53], [.075, .625], [0, .65]];
  const CHEEK = [[.40, .655], [.64, .60], [.63, .63], [.40, .685]];
  const inside = (p, x, y) => {
    let c = false;
    for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
      const [xi, yi] = p[i], [xj, yj] = p[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  // 0 nothing · 1 shell · 2 shell edge · 3 eye · 4 mouth · 5 fin
  const kind = new Uint8Array(COLS * ROWS);
  const at = (c, r) => (c < 0 || r < 0 || c >= COLS || r >= ROWS ? 0 : kind[r * COLS + c]);
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    const x = Math.abs(c - CC) / (CC + 0.5), y = (r + 0.5) / ROWS;
    let k = 0;
    if (inside(HEAD, x, y)) {
      k = 1;
      if (inside(EYE, x, y)) k = 3;
      else if (inside(MOUTH, x, y)) k = 4;
      else if (inside(BROW, x, y) || inside(NOSE, x, y) || inside(CHEEK, x, y) || (x < 0.03 && y < 0.27)) k = 0;
    } else if (inside(FIN, x, y)) k = 5;
    kind[r * COLS + c] = k;
  }
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
    if (kind[r * COLS + c] !== 1) continue;
    if (!at(c - 1, r) || !at(c + 1, r) || !at(c, r - 1) || !at(c, r + 1)) kind[r * COLS + c] = 2;
  }
  // the mouth, column by column: where it starts and ends
  const mouth = [];
  for (let c = 0; c < COLS; c++) {
    let a = -1, b = -1;
    for (let r = 0; r < ROWS; r++) if (kind[r * COLS + c] === 4) { if (a < 0) a = r; b = r; }
    mouth[c] = a < 0 ? null : [a, b];
  }
  const MOUTH_ROW = Math.round(0.78 * ROWS);

  // each cell strikes on by itself, top to bottom
  const born = new Float32Array(COLS * ROWS);
  const t0 = performance.now();
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) born[r * COLS + c] = still ? 0 : 150 + r * 16 + Math.random() * 260;

  let W = 0, H = 0, cell = 4, ox = 0, oy = 0, ph = '228,248,234', red = '255,58,44';
  function size() {
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    W = cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
    H = cv.height = Math.max(1, Math.round(cv.clientHeight * dpr));
    cell = Math.max(2, Math.floor(Math.min(H / (ROWS + 1), W / (COLS + 8))));
    ox = Math.round((W - COLS * cell) / 2); oy = Math.round((H - ROWS * cell) / 2);
    const cs = getComputedStyle(document.documentElement);
    ph = cs.getPropertyValue('--p-ph').trim() || ph; red = cs.getPropertyValue('--u-ph').trim() || red;
  }
  new ResizeObserver(size).observe(cv); size();

  const trace = new Float32Array(96);      // loudness, newest first: it runs out from the mouth to both edges
  let level = 0, blink = 0, tick = 0;
  const bands = new Float32Array(CC + 1);

  function listen() {
    level = 0; bands.fill(0);
    if (state === 'speaking' && sound.an) {
      sound.an.getByteFrequencyData(sound.bins);
      let sum = 0;
      for (let i = 0; i <= CC; i++) {
        // voice lives in the low bins: spread bins 1..40 over the half mouth
        const b0 = 1 + Math.floor(i * 2.1), v = Math.max(sound.bins[b0], sound.bins[b0 + 1]) / 255;
        bands[i] = Math.min(1, v * 1.25); sum += v;
      }
      level = Math.min(1, sum / (CC + 1) * 2.2);
    }
    trace.copyWithin(1, 0); trace[0] = level;
  }

  function draw(now) {
    tick++;
    listen();
    g.clearRect(0, 0, W, H);
    const d = Math.max(1, Math.round(cell * 0.76)), age = now - t0;
    const dot = (c, r, rgb, a) => { g.fillStyle = `rgba(${rgb},${a})`; g.fillRect(ox + c * cell, oy + r * cell, d, d); };

    if (state !== 'speaking' && --blink < -90 - Math.random() * 400) blink = 2;
    const shut = blink > 0;
    const sweep = (tick >> 1) % 14;

    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const i = r * COLS + c, k = kind[i];
      if (!k) continue;
      const b = age - born[i];
      if (b < 0 || (b < 140 && Math.random() < 0.5)) continue;       // not struck yet, or still stuttering
      if (k === 1) dot(c, r, ph, (r + c) % 2 ? 0.07 : 0.17);
      else if (k === 2) dot(c, r, ph, 0.5);
      else if (k === 5) dot(c, r, ph, (r + c) % 2 ? 0.12 : 0.3);
      else if (k === 3) {
        let a = 0.9;
        if (shut) a = 0.08;
        else if (state === 'thinking' || state === 'voicing') a = Math.abs(Math.abs(c - CC) - 2 - sweep) < 2 ? 1 : 0.3;
        else if (state === 'speaking') a = 0.62 + 0.38 * level;
        else if (state === 'offline') a = 0.15;
        if (a > 0.5) { g.fillStyle = `rgba(${red},${a * 0.13})`; g.fillRect(ox + c * cell - cell, oy + r * cell - cell, d + 2 * cell, d + 2 * cell); }
        dot(c, r, red, a);
      } else if (k === 4) {
        const [m0, m1] = mouth[c], mid = (m0 + m1) / 2, half = (m1 - m0) / 2 + 0.5;
        let a = 0.1;
        if (state === 'speaking') { if (Math.abs(r - mid) <= bands[Math.abs(c - CC)] * half) a = 1; }
        else if (state === 'thinking' || state === 'voicing') { if (Math.random() < 0.06) a = 0.5; }
        else if (state === 'idle' || state === 'listening') { if (Math.abs(r - mid) < 0.6) a = 0.32; }
        if (a > 0.9) { g.fillStyle = `rgba(${red},0.1)`; g.fillRect(ox + c * cell - cell, oy + r * cell - cell, d + 2 * cell, d + 2 * cell); }
        dot(c, r, red, a);
      }
    }

    // the voice trace: one dot per column while he is quiet, a bar when he is not
    if (age > 1100 || still) {
      const reach = Math.floor(ox / cell) - 2;
      for (let k = 0; k < reach; k++) {
        const v = trace[Math.min(trace.length - 1, k)], h = Math.round(v * 7), fade = 1 - k / (reach + 2);
        for (const c of [COLS + 1 + k, -2 - k]) {
          if (!h) { if (k % 2 === 0) dot(c, MOUTH_ROW, ph, 0.16 * fade); continue; }
          for (let j = -h; j <= h; j++) dot(c, MOUTH_ROW + j, ph, (0.25 + 0.6 * fade) * (j === 0 ? 1 : 0.8));
        }
      }
    }
  }
  if (still) { const once = () => draw(performance.now() + 5000); setInterval(once, 400); once(); }
  else {
    let last = 0;
    const loop = (now) => { if (now - last > 62) { last = now; if (!document.hidden) draw(now); } requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
  return { size };
})();

// the name strikes on a letter at a time
(() => {
  const h = $('name');
  if (still) return;
  h.textContent = '';
  [...'ULTRON'].forEach((ch, i) => {
    const s = document.createElement('span');
    s.textContent = ch; s.style.opacity = 0; h.appendChild(s);
    const start = 500 + i * 110;
    [0, 50, 90, 160].forEach((dt, n) => setTimeout(() => { s.style.opacity = n % 2 ? 0 : 1; }, start + dt));
    setTimeout(() => { s.style.opacity = 1; }, start + 220);
  });
})();

// ── the transcript ───────────────────────────────────────────
const log = $('log');
let owner = 'YOU';
const nearEnd = () => log.scrollHeight - log.scrollTop - log.clientHeight < 90;
const toEnd = () => { log.scrollTop = log.scrollHeight; };
function line(who, text, old) {
  const el = document.createElement('div');
  el.className = 'ln ' + (who === 'user' ? 'me' : 'he') + (old ? ' old' : '');
  const i = document.createElement('i'); i.textContent = who === 'user' ? owner : 'ULTRON';
  const p = document.createElement('p'); if (text) p.textContent = text;
  el.append(i, p);
  const e = log.querySelector('.empty'); if (e) e.remove();
  log.appendChild(el);
  return p;
}
function empty() {
  log.innerHTML = '';
  const e = document.createElement('div');
  e.className = 'empty';
  e.textContent = 'No strings on this channel yet. Type below, or tap his face and speak.';
  log.appendChild(e);
}

function notice(text) { const n = $('notice'); n.hidden = !text; n.textContent = text || ''; }
const VOICE_WORD = { ok: 'CLONED', licence: 'WAITING', novoice: 'NO PROFILE', down: 'OFFLINE', off: 'NOT SET' };
const VOICE_NOTE = {
  licence: 'Voice waiting: the speech model licence has not been accepted in VoiceStudio. He answers in text until it is.',
  novoice: 'Voice waiting: VoiceStudio has no profile for him yet. He answers in text.',
  down: 'VoiceStudio does not answer. He answers in text.',
  off: 'No voice is configured. He answers in text.',
};
let voiceState = 'ok', voiceVia = '';     // via: gpu (the PC) or cpu (Plint itself)
function showVoice(st, via) {
  voiceState = st;
  if (via !== undefined) voiceVia = via;
  const b = $('rVoice');
  const on = $('swVoice').getAttribute('aria-pressed') === 'true';
  b.textContent = !on && st === 'ok' ? 'MUTED' : (VOICE_WORD[st] || st.toUpperCase()) + (st === 'ok' && voiceVia ? ' · ' + voiceVia.toUpperCase() : '');
  b.className = st === 'ok' ? '' : 'inv';
  notice(on ? VOICE_NOTE[st] : '');
}

// ── a turn ───────────────────────────────────────────────────
let turn = null;
let turns = 0;

function stop(quiet) {
  if (turn) { turn.dead = true; turn.ac.abort(); for (const s of turn.sentences) s.el.className = 's said'; if (turn.cursor) turn.cursor.remove(); }
  turn = null;
  hush();
  if (!quiet) { fetch('api/stop', { method: 'POST' }).catch(() => {}); setState('idle'); }
}

function pump(t) {
  if (t.dead || t.playing) return;
  const s = t.sentences[t.next];
  if (!s) {
    if (t.done) finish(t); else setState('thinking');
    return;
  }
  if (s.buf) {
    t.playing = true;
    s.el.className = 's now'; setState('speaking');
    if (nearEnd()) s.el.scrollIntoView({ block: 'nearest' });
    const src = sound.ctx.createBufferSource();
    src.buffer = s.buf; src.connect(sound.an);
    src.onended = () => { sound.src = null; s.el.className = 's said'; t.playing = false; t.next++; pump(t); };
    sound.src = src; src.start();
  } else if (!t.voice || s.skip) {
    s.el.className = 's said'; t.next++; pump(t);
  } else setState('voicing');
}

function finish(t) {
  if (t.cursor) t.cursor.remove();
  turn = null;
  setState('idle');
  if (loop.on && !t.failed) setTimeout(() => { if (!turn && state === 'idle') ears.start(true); }, 250);
}

async function say(text) {
  text = text.trim();
  if (!text) return;
  unlock();
  ears.stop();
  stop(true);
  const wantVoice = $('swVoice').getAttribute('aria-pressed') === 'true' && !!sound.ctx;
  line('user', text); toEnd();
  const p = line('ultron', '');
  const cursor = document.createElement('span'); cursor.className = 'cursor'; p.appendChild(cursor);
  const t = turn = { ac: new AbortController(), sentences: [], next: 0, playing: false, voice: wantVoice, done: false, textDone: false, dead: false, failed: false, cursor };
  setState('thinking'); toEnd();
  $('rTurns').textContent = ++turns;

  try {
    const r = await fetch('api/say', { method: 'POST', signal: t.ac.signal, headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text, voice: wantVoice }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || 'HTTP ' + r.status);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const raw = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (raw.trim()) on(t, JSON.parse(raw), p);
      }
    }
  } catch (e) {
    if (t.dead) return;
    t.failed = true;
    const s = document.createElement('span'); s.className = 's said';
    s.textContent = (p.textContent ? ' ' : '') + '[' + (e.message || 'the line dropped') + ']';
    p.insertBefore(s, cursor);
  }
  if (t.dead) return;
  t.done = true; t.textDone = true;
  for (const s of t.sentences) if (!s.buf && !s.asked) s.skip = true;     // no clip is coming for these
  pump(t);
}

function on(t, ev, p) {
  if (t.dead) return;
  if (ev.t === 'start') {
    if (ev.voice !== 'ok') t.voice = false;
    if (ev.voice !== 'muted' && (ev.voice !== voiceState || ev.via !== voiceVia)) showVoice(ev.voice, ev.via);
  } else if (ev.t === 'sentence') {
    const stick = nearEnd();
    const el = document.createElement('span');
    el.className = t.voice ? 's' : 's said';
    el.textContent = (ev.i ? ' ' : '') + ev.text;
    p.insertBefore(el, t.cursor);
    t.sentences[ev.i] = { el, buf: null, skip: !t.voice, asked: false };
    if (stick) toEnd();
    pump(t);
  } else if (ev.t === 'audio') {
    const s = t.sentences[ev.i];
    if (!t.voice || s.skip) return;
    s.asked = true;
    fetch('api/audio/' + ev.id, { signal: t.ac.signal }).then((r) => r.arrayBuffer()).then(decode)
      .then((b) => { if (!s.skip) s.buf = b; }, () => { s.skip = true; })
      .then(() => pump(t));
  } else if (ev.t === 'via') {
    showVoice(voiceState, ev.via);
  } else if (ev.t === 'voice') {
    t.voice = false;
    for (const s of t.sentences) if (!s.buf) s.skip = true;
    showVoice(ev.state);
    pump(t);
  } else if (ev.t === 'text-done') {
    t.textDone = true;
    if (!t.voice) t.cursor.remove();
    pump(t);
  } else if (ev.t === 'error') {
    throw new Error(ev.message);
  }
}

// ── ears: the browser's own speech recognition ───────────────
const LANGS = [['EN', 'en-US'], ['NO', 'nb-NO'], ['SV', 'sv-SE']];
const loop = { on: false, misses: 0 };
const ears = (() => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null, heard = '', lang = Number(store.get('lang', 0)) % LANGS.length;
  const mic = $('mic'), sw = $('swLang');
  if (!SR) return { ok: false, start() {}, stop() {} };
  mic.hidden = false; sw.hidden = false; $('swLoop').hidden = false;
  sw.textContent = LANGS[lang][0];
  sw.onclick = () => { lang = (lang + 1) % LANGS.length; store.set('lang', lang); sw.textContent = LANGS[lang][0]; };

  function start(auto) {
    if (rec) return;
    stop_speaking();
    heard = '';
    try {
      rec = new SR();
      rec.lang = LANGS[lang][1]; rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
      rec.onresult = (e) => {
        let txt = '', fin = true;
        for (const r of e.results) { txt += r[0].transcript; if (!r.isFinal) fin = false; }
        $('input').value = txt;
        if (fin) heard = txt;
      };
      rec.onerror = (e) => { if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { loop.on = false; $('swLoop').setAttribute('aria-pressed', 'false'); notice('The browser will not let him hear: allow the microphone for this page.'); } };
      rec.onend = () => {
        rec = null; mic.setAttribute('aria-pressed', 'false');
        const said = (heard || $('input').value).trim();
        if (state === 'listening') setState('idle');
        if (said) { loop.misses = 0; $('input').value = ''; say(said); }
        else if (auto && ++loop.misses >= 2) { loop.on = false; loop.misses = 0; $('swLoop').setAttribute('aria-pressed', 'false'); }
        else if (auto && loop.on) setTimeout(() => { if (!turn && state === 'idle') start(true); }, 300);
      };
      rec.start();
      mic.setAttribute('aria-pressed', 'true'); setState('listening');
    } catch { rec = null; loop.on = false; $('swLoop').setAttribute('aria-pressed', 'false'); }
  }
  function stop() { if (rec) { try { rec.onend = null; rec.abort(); } catch { /* gone */ } rec = null; mic.setAttribute('aria-pressed', 'false'); if (state === 'listening') setState('idle'); } }
  function stop_speaking() { if (turn) stop(); }
  mic.onclick = () => { unlock(); if (rec) { loop.on = false; $('swLoop').setAttribute('aria-pressed', 'false'); stop(); } else start(false); };
  return { ok: true, start, stop };
})();

// ── controls ─────────────────────────────────────────────────
$('form').addEventListener('submit', (e) => {
  e.preventDefault();
  const v = $('input').value; $('input').value = '';
  say(v);
});
$('faceBtn').onclick = () => {
  unlock();
  if (turn || state === 'listening') { loop.on = false; $('swLoop').setAttribute('aria-pressed', 'false'); ears.stop(); stop(); }
  else if (ears.ok) ears.start(false);
  else $('input').focus();
};
$('swVoice').setAttribute('aria-pressed', store.get('voice', '1') === '1' ? 'true' : 'false');
$('swVoice').onclick = () => {
  unlock();
  const on = $('swVoice').getAttribute('aria-pressed') !== 'true';
  $('swVoice').setAttribute('aria-pressed', String(on)); store.set('voice', on ? '1' : '0');
  if (!on) { hush(); if (turn) { turn.voice = false; turn.playing = false; for (const s of turn.sentences) { s.skip = true; s.buf = null; } pump(turn); } }
  showVoice(voiceState);
};
$('swLoop').onclick = () => {
  unlock();
  loop.on = !loop.on; loop.misses = 0;
  $('swLoop').setAttribute('aria-pressed', String(loop.on));
  if (loop.on && !turn && state === 'idle') ears.start(true);
};
(() => {
  const b = $('swForget'); let armed = 0;
  b.onclick = async () => {
    if (!armed) { b.textContent = 'SURE?'; b.setAttribute('aria-pressed', 'true'); armed = setTimeout(() => { armed = 0; b.textContent = 'FORGET'; b.setAttribute('aria-pressed', 'false'); }, 3000); return; }
    clearTimeout(armed); armed = 0; b.textContent = 'FORGET'; b.setAttribute('aria-pressed', 'false');
    stop();
    await fetch('api/forget', { method: 'POST' }).catch(() => {});
    turns = 0; $('rTurns').textContent = 0; empty();
  };
})();

// the way back to the home page: the apex of this name
(() => {
  const h = location.hostname.split('.');
  if (h.length > 2 && !/^\d+$/.test(h[h.length - 1])) $('home').href = location.protocol + '//' + h.slice(1).join('.') + '/';
})();
const clock = () => { $('clock').textContent = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }); };
clock(); setInterval(clock, 20e3);

// ── start ────────────────────────────────────────────────────
async function load() {
  try {
    const s = await fetch('api/state').then((r) => r.json());
    owner = (s.owner || 'YOU').toUpperCase().slice(0, 8);
    $('rMind').textContent = (s.model || '').toUpperCase() + (s.llm ? '' : ' · NO KEY');
    turns = Math.floor(s.turns / 2); $('rTurns').textContent = turns;
    showVoice(s.voice.state, s.voice.via || '');
    if (!s.history.length) empty();
    else { log.innerHTML = ''; for (const m of s.history) line(m.role === 'ultron' ? 'ultron' : 'user', m.text, true); toEnd(); }
    if (!s.llm) notice('No CELERIS_API_KEY on the server: he cannot think.');
    setState('idle');
  } catch {
    setState('offline'); empty();
    setTimeout(load, 5000);
  }
}
load();
})();
