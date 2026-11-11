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
    const r = await fetch('api/say', { method: 'POST', signal: t.ac.signal, headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text, voice: wantVoice, image: eyes.frame() || undefined }) });
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

// ── eyes: this device's camera; one frame goes along with each thing you say ──
const eyes = (() => {
  const v = $('eye'), sw = $('swEyes');
  let stream = null, facing = store.get('facing', 'user');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { sw.hidden = true; return { frame: () => null }; }
  function off() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null; v.srcObject = null; v.hidden = true; sw.setAttribute('aria-pressed', 'false');
  }
  async function on() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      v.srcObject = stream; v.classList.toggle('mirror', facing === 'user'); v.hidden = false;
      await v.play();
      sw.setAttribute('aria-pressed', 'true');
    } catch { off(); notice('The browser will not let him see: allow the camera for this page.'); }
  }
  // what the camera shows right now, small enough to send: longest side 768
  function frame() {
    if (!stream || !v.videoWidth) return null;
    const k = Math.min(1, 768 / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement('canvas');
    c.width = Math.round(v.videoWidth * k); c.height = Math.round(v.videoHeight * k);
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.72).split(',')[1] || null;
  }
  sw.onclick = () => (stream ? off() : on());
  v.onclick = async () => { facing = facing === 'user' ? 'environment' : 'user'; store.set('facing', facing); off(); await on(); };
  return { frame };
})();

// ── memory: the graph of what he has learned ─────────────────
(() => {
  const box = $('memory'), sw = $('swMem'), svg = $('memGraph'), facts = $('memFacts'), NS = 'http://www.w3.org/2000/svg';
  let data = null, picked = null;
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };

  // where everything sits: links pull, everything else pushes apart, the person in the middle
  function layout(W, H) {
    const ns = data.nodes, idx = new Map(ns.map((n, i) => [n.id, i]));
    let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    ns.forEach((n, i) => { const a = (i / ns.length) * Math.PI * 2; n.x = W / 2 + Math.cos(a) * W * 0.3 * (0.6 + rnd() * 0.4); n.y = H / 2 + Math.sin(a) * H * 0.3 * (0.6 + rnd() * 0.4); });
    const k = Math.sqrt((W * H) / Math.max(1, ns.length)) * 0.75;
    for (let it = 0; it < 260; it++) {
      const t = 1 - it / 260, fx = new Float32Array(ns.length), fy = new Float32Array(ns.length);
      for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
        let dx = ns[i].x - ns[j].x, dy = (ns[i].y - ns[j].y) * 1.6, d = Math.hypot(dx, dy) || 0.01;
        const f = (k * k) / d; dx = (dx / d) * f; dy = (dy / d) * f;
        fx[i] += dx; fy[i] += dy; fx[j] -= dx; fy[j] -= dy;
      }
      for (const e of data.edges) {
        const a = idx.get(e.s), b = idx.get(e.o); if (a === undefined || b === undefined) continue;
        const dx = ns[a].x - ns[b].x, dy = ns[a].y - ns[b].y, d = Math.hypot(dx, dy) || 0.01, f = (d * d) / k / 2.2;
        fx[a] -= (dx / d) * f; fy[a] -= (dy / d) * f; fx[b] += (dx / d) * f; fy[b] += (dy / d) * f;
      }
      ns.forEach((n, i) => {
        if (n.id === data.owner) { n.x = W / 2; n.y = H / 2; return; }
        const d = Math.hypot(fx[i], fy[i]) || 0.01, m = Math.min(d, 18 * t + 1);
        n.x = Math.max(46, Math.min(W - 46, n.x + (fx[i] / d) * m + (W / 2 - n.x) * 0.012));
        n.y = Math.max(16, Math.min(H - 16, n.y + (fy[i] / d) * m + (H / 2 - n.y) * 0.012));
      });
    }
  }

  function draw() {
    svg.innerHTML = '';
    const W = svg.clientWidth || 340, H = svg.clientHeight || 300;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    $('memCount').textContent = `${data.nodes.length} THINGS · ${data.edges.length} LINKS`;
    $('memWipe').hidden = !data.edges.length;
    if (!data.nodes.length) {
      $('memHint').textContent = 'Nothing yet. He learns from what you tell him about yourself, the people and things around you, and your plans.';
      facts.innerHTML = ''; return;
    }
    layout(W, H);
    const at = new Map(data.nodes.map((n) => [n.id, n]));
    const near = picked ? new Set([picked, ...data.edges.filter((e) => e.s === picked || e.o === picked).flatMap((e) => [e.s, e.o])]) : null;
    for (const e of data.edges) {
      const a = at.get(e.s), b = at.get(e.o); if (!a || !b) continue;
      el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: picked && (e.s === picked || e.o === picked) ? 'on' : '' }, svg);
    }
    data.nodes.forEach((n, i) => {
      const g = el('g', { class: 'n' + (n.id === data.owner ? ' me' : '') + (n.id === picked ? ' on' : '') + (near && !near.has(n.id) ? ' dim' : ''), tabindex: 0, role: 'button', 'aria-label': n.name }, svg);
      const label = n.name.length > 18 ? n.name.slice(0, 17) + '…' : n.name;
      const w = label.length * 6.4 + 10;
      el('rect', { class: 'bg', x: n.x - w / 2, y: n.y + 6, width: w, height: 13 }, g);
      el('rect', { x: n.x - 3.5, y: n.y - 5, width: 7, height: 9 }, g);
      el('text', { x: n.x, y: n.y + 15.5, 'text-anchor': 'middle' }, g).textContent = label;
      el('rect', { x: n.x - w / 2, y: n.y - 10, width: w, height: 32, fill: 'transparent', style: 'fill:transparent' }, g);
      const pick = () => { picked = picked === n.id ? null : n.id; draw(); };
      g.onclick = pick; g.onkeydown = (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } };
      if (!still && !picked) { g.style.opacity = 0; setTimeout(() => { g.style.opacity = ''; }, 40 + i * 45); }
    });
    list();
  }

  function list() {
    facts.innerHTML = '';
    const name = (id) => (data.nodes.find((n) => n.id === id) || { name: id }).name;
    const es = (picked ? data.edges.filter((e) => e.s === picked || e.o === picked) : data.edges.slice().sort((a, b) => b.last - a.last).slice(0, 6));
    $('memHint').textContent = picked ? `${name(picked)}: ${es.length} link${es.length === 1 ? '' : 's'}. Tap again to let go.` : 'Latest. Tap a thing to see what he knows about it.';
    for (const e of es) {
      const row = document.createElement('div'); row.className = 'fact';
      const p = document.createElement('span');
      const s = document.createElement('b'); s.textContent = name(e.s);
      const r = document.createElement('i'); r.textContent = ' ' + e.r + ' ';
      const o = document.createElement('b'); o.textContent = name(e.o);
      p.append(s, r, o);
      const x = document.createElement('button'); x.type = 'button'; x.className = 'btn o sm'; x.textContent = '×'; x.setAttribute('aria-label', `Forget: ${name(e.s)} ${e.r} ${name(e.o)}`);
      x.onclick = () => forget({ s: e.s, r: e.r, o: e.o });
      row.append(p, x); facts.appendChild(row);
    }
  }
  const load = () => fetch('api/memory').then((r) => r.json()).then((d) => { data = d; if (picked && !d.nodes.some((n) => n.id === picked)) picked = null; draw(); });
  const forget = (q) => fetch('api/memory/forget', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(q) }).then(load).catch(() => {});
  function open(on) {
    box.hidden = !on; sw.setAttribute('aria-pressed', String(on));
    if (on) {
      if (!$('tune').hidden) $('tuneClose').click();
      document.documentElement.style.setProperty('--deck-h', document.querySelector('.deck').offsetHeight + 'px');
      picked = null; load().catch(() => open(false));
    }
  }
  sw.onclick = () => open(box.hidden);
  $('memClose').onclick = () => open(false);
  (() => {
    const b = $('memWipe'); let armed = 0;
    b.onclick = () => {
      if (!armed) { b.textContent = 'SURE?'; b.setAttribute('aria-pressed', 'true'); armed = setTimeout(() => { armed = 0; b.textContent = 'WIPE'; b.setAttribute('aria-pressed', 'false'); }, 3000); return; }
      clearTimeout(armed); armed = 0; b.textContent = 'WIPE'; b.setAttribute('aria-pressed', 'false'); forget({ all: true });
    };
  })();
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !box.hidden) open(false); });
  window.addEventListener('resize', () => { if (!box.hidden && data) draw(); });
})();

// ── tuning: what celeris-1 is given ──────────────────────────
(() => {
  const SPEC = [
    ['temperature', 'WILDNESS', 'How loose his wording gets. Wild also slips more.', [[0.5, 'CALM'], [0.8, 'NORMAL'], [1, 'LIVELY'], [1.2, 'WILD']]],
    ['length', 'LENGTH', 'How long he talks before he stops.', [['short', 'SHORT'], ['normal', 'NORMAL'], ['long', 'LONG']]],
    ['quotes', 'THE FILM', 'How much of the film comes back: its lines, strings, Stark.', [['never', 'NEVER'], ['rare', 'RARE'], ['free', 'FREE']]],
    ['film', 'FILM LINES', 'How many of his real lines he is shown, for the rhythm.', [[0, 'NONE'], [12, '12'], [24, '24'], [200, 'ALL']]],
    ['name', 'YOUR NAME', 'How often he says it.', [['never', 'NEVER'], ['rare', 'RARE'], ['often', 'OFTEN']]],
    ['context', 'CONTEXT', 'How many past messages he is handed each time.', [[10, '10'], [30, '30'], [60, '60']]],
  ];
  const box = $('tune'), rows = $('tuneRows'), extra = $('tuneExtra'), sw = $('swTune');
  let tune = null, timer = 0, minds = ['celeris'];
  const MIND = ['mind', 'MIND', 'Which model writes his answers. Haiku ignores WILDNESS.', [['celeris', 'CELERIS-1'], ['haiku', 'HAIKU']]];
  const save = (patch) => fetch('api/tune', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(patch) })
    .then((r) => r.json()).then((d) => { tune = d.tune; if (d.model) $('rMind').textContent = d.model.replace(/^claude-/i, '').toUpperCase(); paint(); if (!$('tuneText').hidden) prompt(); }).catch(() => {});
  function paint() {
    rows.innerHTML = '';
    for (const [key, name, what, opts] of (minds.length > 1 ? [MIND, ...SPEC] : SPEC)) {
      const row = document.createElement('div'); row.className = 'trow';
      const lab = document.createElement('div');
      const b = document.createElement('b'); b.textContent = name;
      const sm = document.createElement('small'); sm.textContent = what;
      lab.append(b, sm);
      const seg = document.createElement('div'); seg.className = 'seg'; seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', name);
      // the stored value may sit between two choices: light the nearest
      const near = typeof opts[0][0] === 'number' ? opts.reduce((a, o) => (Math.abs(o[0] - tune[key]) < Math.abs(a[0] - tune[key]) ? o : a))[0] : tune[key];
      for (const [val, label] of opts) {
        const btn = document.createElement('button');
        btn.type = 'button'; btn.className = 'btn o sm'; btn.textContent = label;
        btn.setAttribute('aria-pressed', String(val === near));
        btn.onclick = () => save({ [key]: val });
        seg.appendChild(btn);
      }
      row.append(lab, seg); rows.appendChild(row);
    }
    if (document.activeElement !== extra) extra.value = tune.extra || '';
  }
  const prompt = () => fetch('api/prompt').then((r) => r.json()).then((d) => {
    $('tuneText').textContent = `MODEL ${d.model}${d.temperature === null ? '' : ' · TEMPERATURE ' + d.temperature} · THEN THE LAST ${d.messages} MESSAGES\n\n${d.system}`;
  }).catch(() => {});
  function open(on) {
    box.hidden = !on; sw.setAttribute('aria-pressed', String(on));
    if (on) {
      if (!$('memory').hidden) $('memClose').click();
      document.documentElement.style.setProperty('--deck-h', document.querySelector('.deck').offsetHeight + 'px');
      fetch('api/tune').then((r) => r.json()).then((d) => { tune = d.tune; minds = d.minds || minds; paint(); }).catch(() => { box.hidden = true; sw.setAttribute('aria-pressed', 'false'); });
    }
  }
  sw.onclick = () => open(box.hidden);
  $('tuneClose').onclick = () => open(false);
  extra.oninput = () => { clearTimeout(timer); timer = setTimeout(() => save({ extra: extra.value }), 700); };
  $('tunePrompt').onclick = () => {
    const t = $('tuneText'); t.hidden = !t.hidden;
    $('tunePrompt').setAttribute('aria-pressed', String(!t.hidden));
    if (!t.hidden) prompt().then(() => t.scrollIntoView({ block: 'nearest' }));
  };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !box.hidden) open(false); });
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
    $('rMind').textContent = (s.model || '').replace(/^claude-/i, '').toUpperCase() + (s.llm ? '' : ' · NO KEY');
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
