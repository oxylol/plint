/* ─────────────────────────────────────────────────────────────
   Hertz · engine
   Everything is two waves. Each voice has a pulse wave (A) and a slow
   drift sine (B). Their sum is compared with a threshold; every upward
   crossing fires a note. B also chooses the note: the melody literally
   follows a sine. The sound itself is a sine bent by another sine (FM).
   A song-wide "tide" sine moves the chords, the density and the filters
   over many bars.
   Pure maths (findTriggers & co) works in node too, for the tests.
   ───────────────────────────────────────────────────────────── */
(function (root) {
  "use strict";
  var TAU = Math.PI * 2;

  // ── tables ────────────────────────────────────────────────
  var PULSES = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16];          // cycles per bar
  var PULSE_NAMES = { 0.25: "4 bars", 0.5: "2 bars", 1: "1 bar", 2: "1/2", 3: "1/2 T", 4: "1/4", 5: "5 per bar", 6: "1/4 T", 7: "7 per bar", 8: "1/8", 12: "1/8 T", 16: "1/16" };
  var DRIFTS = [0, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 7];
  var RATIOS = [0.25, 0.5, 0.75, 1, 1.25, 1.41, 1.5, 2, 2.5, 2.83, 3, 3.5, 4, 5, 6, 7, 8];
  var SHAPES = ["sine", "triangle", "saw", "square"];
  var NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  var TIDES = [2, 4, 8, 16, 32];                                       // bars per cycle

  // Ratios inside one octave. Equal-tempered scales are semitones; the
  // overtone scale is harmonics 8–15 of the root, pure frequency ratios.
  function et(semis) { return semis.map(function (s) { return Math.pow(2, s / 12); }); }
  var SCALES = [
    { id: "minor", name: "Minor", r: et([0, 2, 3, 5, 7, 8, 10]) },
    { id: "dorian", name: "Dorian", r: et([0, 2, 3, 5, 7, 9, 10]) },
    { id: "pentatonic", name: "Pentatonic", r: et([0, 3, 5, 7, 10]) },
    { id: "major", name: "Major", r: et([0, 2, 4, 5, 7, 9, 11]) },
    { id: "lydian", name: "Lydian", r: et([0, 2, 4, 6, 7, 9, 11]) },
    { id: "phrygian", name: "Phrygian", r: et([0, 1, 3, 5, 7, 8, 10]) },
    { id: "overtone", name: "Overtones", r: [8, 9, 10, 11, 12, 13, 14, 15].map(function (h) { return h / 8; }) }
  ];
  function scaleOf(song) { return SCALES.filter(function (s) { return s.id === song.scale; })[0] || SCALES[0]; }

  // ── voices ────────────────────────────────────────────────
  // Ordered by where they sit in the spectrum, low to high; the colours run
  // the same way (red = lowest), like light.
  var KINDS = {
    kick:  { name: "Kick",  color: "#ff453a", melodic: false, freq: [32, 160], pan: 0 },
    bass:  { name: "Bass",  color: "#ff9f0a", melodic: true, oct: [0, 2], pan: 0 },
    snap:  { name: "Snap",  color: "#ffd60a", melodic: false, freq: [110, 640], pan: 0.06 },
    pad:   { name: "Pad",   color: "#32d74b", melodic: true, oct: [1, 4], pan: 0 },
    pluck: { name: "Pluck", color: "#4aa3ff", melodic: true, oct: [2, 5], pan: -0.22 },
    hat:   { name: "Hat",   color: "#bf5af2", melodic: false, freq: [1800, 14000], pan: 0.24 }
  };
  var ORDER = ["kick", "bass", "snap", "pad", "pluck", "hat"];

  function voice(kind, o) {
    var v = {
      kind: kind, mute: false, solo: false,
      pulse: 4, shape: "sine", phase: 0, thr: 0, drift: 0.5, bend: 0, melody: 0, level: 0.8,
      freq: 60, oct: 1, deg: 0, ratio: 1, mod: 1, decay: 0.4, color: 9000, drive: 0, echo: 0, space: 0.1
    };
    for (var k in o) v[k] = o[k];
    return v;
  }

  // ── built-in songs ───────────────────────────────────────
  var PRESETS = [
    {
      name: "First Light", bpm: 112, key: 9, scale: "minor", swing: 0.12, snap: true, volume: 0.8,
      tide: { bars: 16, chords: 3, density: 0.55, bright: 0.6 },
      voices: [
        voice("kick",  { pulse: 4, thr: 0.1, drift: 3, bend: 0.25, level: 0.95, freq: 48, ratio: 1, mod: 0.6, decay: 0.42, color: 6000, drive: 0.25, space: 0.04 }),
        voice("bass",  { pulse: 8, thr: 0.35, drift: 0.75, bend: 0.5, melody: 4, level: 0.74, oct: 0, deg: 0, ratio: 1, mod: 1.4, decay: 0.26, color: 1400, drive: 0.35, space: 0.02 }),
        voice("snap",  { pulse: 2, phase: 0.5, thr: 0.0, drift: 1.5, bend: 0.3, level: 0.72, freq: 190, ratio: 1.5, mod: 2.2, decay: 0.24, color: 9000, echo: 0.08, space: 0.22 }),
        voice("pad",   { pulse: 1, phase: 0.0, thr: 0, drift: 0.25, bend: 0, melody: 2, level: 0.55, oct: 2, deg: 0, ratio: 2, mod: 0.6, decay: 3.2, color: 2600, echo: 0.1, space: 0.6 }),
        voice("pluck", { pulse: 6, phase: 0.25, thr: 0.45, drift: 1.25, bend: 0.55, melody: 7, level: 0.68, oct: 3, deg: 0, ratio: 3.5, mod: 2.4, decay: 0.42, color: 7000, echo: 0.42, space: 0.32 }),
        voice("hat",   { pulse: 16, thr: 0.5, drift: 3, bend: 0.5, level: 0.62, freq: 7600, ratio: 1.41, mod: 3, decay: 0.075, color: 16000, space: 0.08 })
      ]
    },
    {
      name: "Tidepool", bpm: 92, key: 2, scale: "dorian", swing: 0.2, snap: true, volume: 0.8,
      tide: { bars: 8, chords: 4, density: 0.35, bright: 0.4 },
      voices: [
        voice("kick",  { pulse: 2, thr: 0.0, drift: 3, bend: 0.5, level: 0.9, freq: 52, mod: 0.3, decay: 0.55, color: 5000, drive: 0.15, space: 0.05 }),
        voice("bass",  { pulse: 4, phase: 0.125, thr: 0.2, drift: 0.75, bend: 0.4, melody: 5, level: 0.7, oct: 1, ratio: 0.5, mod: 2, decay: 0.4, color: 900, drive: 0.2 }),
        voice("snap",  { pulse: 1, phase: 0.5, thr: 0.0, drift: 2, bend: 0.2, level: 0.7, freq: 240, ratio: 2.5, mod: 1.6, decay: 0.3, color: 7000, echo: 0.25, space: 0.4 }),
        voice("pad",   { pulse: 0.5, thr: -0.1, drift: 0.125, melody: 3, level: 0.58, oct: 2, ratio: 1, mod: 1.2, decay: 4, color: 1800, echo: 0.2, space: 0.75 }),
        voice("pluck", { pulse: 12, phase: 0, thr: 0.62, drift: 0.75, bend: 0.7, melody: 9, level: 0.66, oct: 3, ratio: 2, mod: 3.5, decay: 0.6, color: 5200, echo: 0.55, space: 0.45 }),
        voice("hat",   { pulse: 12, thr: 0.35, drift: 2, bend: 0.5, level: 0.58, freq: 9000, ratio: 2.83, mod: 2.2, decay: 0.05, color: 15000, space: 0.15 })
      ]
    },
    {
      name: "Overtone Engine", bpm: 124, key: 0, scale: "overtone", swing: 0, snap: true, volume: 0.8,
      tide: { bars: 8, chords: 0, density: 0.7, bright: 0.8 },
      voices: [
        voice("kick",  { pulse: 4, thr: 0.0, drift: 0, level: 1, freq: 46, ratio: 1, mod: 0.9, decay: 0.38, color: 7000, drive: 0.45 }),
        voice("bass",  { pulse: 16, shape: "saw", thr: 0.55, drift: 1, bend: 0.6, melody: 6, level: 0.66, oct: 1, ratio: 1, mod: 2.6, decay: 0.14, color: 2200, drive: 0.5 }),
        voice("snap",  { pulse: 2, phase: 0.5, thr: 0.0, drift: 0, level: 0.72, freq: 170, ratio: 1.41, mod: 3, decay: 0.2, color: 8000, space: 0.18 }),
        voice("pad",   { pulse: 0.25, thr: -0.5, drift: 0.25, melody: 4, level: 0.52, oct: 2, ratio: 3, mod: 0.8, decay: 4, color: 3000, space: 0.7 }),
        voice("pluck", { pulse: 16, shape: "triangle", thr: 0.7, drift: 3, bend: 0.75, melody: 8, level: 0.64, oct: 4, ratio: 5, mod: 1.8, decay: 0.3, color: 9000, echo: 0.5, space: 0.3 }),
        voice("hat",   { pulse: 16, thr: 0.5, drift: 7, bend: 0.45, level: 0.6, freq: 8000, ratio: 1.41, mod: 4, decay: 0.06, color: 17000 })
      ]
    },
    {
      name: "Blank", bpm: 120, key: 9, scale: "minor", swing: 0, snap: true, volume: 0.8,
      tide: { bars: 8, chords: 0, density: 0, bright: 0 },
      voices: [
        voice("kick",  { pulse: 4, drift: 0, freq: 50, mod: 0.4, decay: 0.4, color: 6000 }),
        voice("bass",  { mute: true, pulse: 8, drift: 0.5, oct: 1, mod: 1, decay: 0.3, color: 1200 }),
        voice("snap",  { mute: true, pulse: 2, phase: 0.5, drift: 0, freq: 200, ratio: 1.5, mod: 2, decay: 0.25 }),
        voice("pad",   { mute: true, pulse: 1, drift: 0.25, oct: 2, ratio: 2, mod: 0.6, decay: 3, color: 2400, space: 0.6 }),
        voice("pluck", { mute: true, pulse: 8, drift: 1, melody: 5, oct: 3, ratio: 3.5, mod: 2, decay: 0.4, echo: 0.4, space: 0.3 }),
        voice("hat",   { mute: true, pulse: 4, phase: 0.5, drift: 0, freq: 8000, ratio: 1.41, mod: 3, decay: 0.05, color: 16000 })
      ]
    }
  ];

  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  // Bring any saved patch up to the current shape (missing fields → defaults).
  function normalize(p) {
    var base = clone(PRESETS[0]);
    var out = clone(p || {});
    ["bpm", "key", "scale", "swing", "snap", "volume", "name"].forEach(function (k) { if (out[k] === undefined) out[k] = base[k]; });
    out.tide = Object.assign({}, base.tide, out.tide || {});
    var vs = out.voices || [];
    out.voices = ORDER.map(function (kind, i) {
      var have = vs.filter(function (v) { return v && v.kind === kind; })[0];
      return Object.assign(voice(kind, {}), base.voices[i], have || {});
    });
    return out;
  }

  // ── waves ────────────────────────────────────────────────
  function frac(x) { return x - Math.floor(x); }
  function shapeAt(shape, x) {
    switch (shape) {
      case "triangle": return 4 * Math.abs(frac(x - 0.25) - 0.5) - 1;
      case "saw": return 2 * frac(x + 0.5) - 1;
      case "square": return frac(x) < 0.5 ? 1 : -1;
      default: return Math.sin(TAU * x);
    }
  }
  // Steepest rise each shape reaches per cycle (for velocity).
  var SLOPE = { sine: TAU, triangle: 4, saw: 2, square: Infinity };

  // Tide: starts low (sparse, dark), peaks halfway through its cycle.
  function tideU(song, beat) { return 0.5 - 0.5 * Math.cos(TAU * beat / (4 * song.tide.bars)); }
  // Chords move on the bar, not mid-bar.
  function tideShift(song, beat) {
    if (!song.tide.chords) return 0;
    return Math.round(song.tide.chords * tideU(song, Math.floor(beat / 4) * 4));
  }

  function waveA(v, beat) { return shapeAt(v.shape, v.pulse * beat / 4 + v.phase); }
  function waveB(v, beat) { return v.drift ? Math.sin(TAU * v.drift * beat / 4) : 0; }
  function signal(v, beat) { return (waveA(v, beat) + v.bend * waveB(v, beat)) / (1 + v.bend); }
  // The tide sets a "tide line" above the threshold: a hit only counts if its
  // wave reaches that high before falling back. Low tide = only the tallest
  // peaks play (or the voice rests); timing never moves.
  function tideLine(song, v, beat) {
    var d = song.tide.density;
    if (!d) return -Infinity;
    return v.thr + d * (1 - tideU(song, beat)) * (1.05 - v.thr);
  }

  // Scale degree → Hz. Octave 0 starts at the key's root in octave 1 (A1 = 55 Hz).
  function keyRoot(song) { return 440 * Math.pow(2, (song.key - 9) / 12 - 3); }
  function degHz(song, oct, deg) {
    var r = scaleOf(song).r, n = r.length;
    var o = oct + Math.floor(deg / n), i = ((deg % n) + n) % n;
    return keyRoot(song) * Math.pow(2, o) * r[i];
  }
  function noteName(hz) {
    var m = Math.round(12 * Math.log2(hz / 440) + 69);
    return NOTE_NAMES[((m % 12) + 12) % 12] + (Math.floor(m / 12) - 1);
  }

  // The pitch a voice plays at a beat: its base note, plus the drift wave
  // scaled into `melody` scale steps, plus the tide's chord shift.
  function pitchAt(song, v, beat) {
    var u = (waveB(v, beat) + 1) / 2;
    if (KINDS[v.kind].melodic) {
      var deg = v.deg + Math.round(v.melody * u) + tideShift(song, beat);
      return degHz(song, v.oct, deg);
    }
    return v.freq * Math.pow(2, Math.round(v.melody * (u - 0.5)) / 12);
  }

  var SAMPLES_PER_BEAT = 96;
  var GRID = 0.25; // a 16th, in beats

  // Every trigger of voice v in [b0, b1) (beats). With snap on, a crossing is
  // moved to the nearest 16th; the raw scan starts half a 16th early so that
  // snapped hits land exactly in the window they belong to (never earlier).
  function findTriggers(song, v, b0, b1, out) {
    out = out || [];
    var step = 1 / SAMPLES_PER_BEAT, snap = song.snap, off = snap ? -GRID / 2 : 0;
    var ref = (SLOPE[v.shape] * v.pulse + v.bend * TAU * v.drift) / (1 + v.bend) / 4; // per beat
    var lastSlot = null;
    // Sample on a fixed grid (k / SAMPLES_PER_BEAT) and give each interval to
    // the window holding its left end, so any split of time finds every
    // crossing exactly once: the engine's small windows and the screen's bars agree.
    var k0 = Math.ceil((b0 + off) * SAMPLES_PER_BEAT - 1e-9), k1 = Math.ceil((b1 + off) * SAMPLES_PER_BEAT - 1e-9);
    var prevB = k0 * step, prev = signal(v, prevB) - v.thr;
    for (var k = k0 + 1; k <= k1; k++) {
      var b = k * step;
      var cur = signal(v, b) - v.thr;
      if (prev < 0 && cur >= 0) {
        var tc = prevB + step * (-prev / (cur - prev || 1));
        var at = tc, slot = null;
        if (snap) {
          slot = Math.round(tc / GRID);
          if (slot === lastSlot) { prev = cur; prevB = b; continue; }
          lastSlot = slot;
          at = slot * GRID + (slot % 2 ? song.swing * GRID : 0);
        }
        var slope = (cur - prev) / step;
        var vel = ref === Infinity || !isFinite(slope / ref) ? 1 : Math.min(1, 0.3 + 0.7 * Math.pow(Math.min(1, slope / ref), 0.6));
        if (slope / ref > 1.5) vel = 1;
        // how tall does this wave get before it drops back under the threshold?
        var line = tideLine(song, v, tc), gated = false;
        if (line > v.thr) {
          var peak = cur + v.thr, lim = b + 4;
          for (var pb = b + step; pb < lim; pb += step) { var sv = signal(v, pb); if (sv > peak) peak = sv; if (sv < v.thr || peak >= line) break; }
          gated = peak < line;
        }
        out.push({ beat: at, raw: tc, vel: vel, hz: pitchAt(song, v, tc), gated: gated });
      }
      prev = cur; prevB = b;
    }
    return out;
  }

  function audible(song, v) {
    var anySolo = song.voices.some(function (x) { return x.solo; });
    return anySolo ? v.solo : !v.mute;
  }

  // ── audio ────────────────────────────────────────────────
  function Engine(song) {
    this.song = normalize(song);
    this.ctx = null;
    this.playing = false;
    this.listeners = [];
    this.anchors = [];
  }

  Engine.prototype.on = function (fn) { this.listeners.push(fn); };
  Engine.prototype.emit = function (e) { this.listeners.forEach(function (fn) { fn(e); }); };

  Engine.prototype.init = function () {
    if (this.ctx) return;
    var AC = root.AudioContext || root.webkitAudioContext;
    var ctx = this.ctx = new AC({ latencyHint: "interactive" });
    try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch (e) {}

    // master: voices → glue compressor → volume → analyser → out (+ recorder)
    this.comp = ctx.createDynamicsCompressor();
    this.comp.threshold.value = -14; this.comp.knee.value = 10; this.comp.ratio.value = 3.5;
    this.comp.attack.value = 0.006; this.comp.release.value = 0.18;
    this.master = ctx.createGain();
    this.limiter = ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -2; this.limiter.knee.value = 0; this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.002; this.limiter.release.value = 0.08;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 2048; this.analyser.smoothingTimeConstant = 0.78;
    this.comp.connect(this.master); this.master.connect(this.limiter);
    this.limiter.connect(this.analyser); this.analyser.connect(ctx.destination);
    this.out = this.limiter;

    // white noise for snap and hat
    var nb = this.noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate), nd = nb.getChannelData(0);
    for (var i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;

    // echo: ping-pong, synced to a dotted 8th, darkening as it repeats
    this.echoIn = ctx.createGain();
    this.dl = ctx.createDelay(2); this.dr = ctx.createDelay(2);
    var fb = this.echoFb = ctx.createGain(); fb.gain.value = 0.38;
    var tone = ctx.createBiquadFilter(); tone.type = "lowpass"; tone.frequency.value = 3200;
    var pl = ctx.createStereoPanner(), pr = ctx.createStereoPanner(); pl.pan.value = -0.7; pr.pan.value = 0.7;
    this.echoIn.connect(tone); tone.connect(this.dl); this.dl.connect(pl); this.dl.connect(this.dr);
    this.dr.connect(pr); this.dr.connect(fb); fb.connect(tone);
    var echoOut = ctx.createGain(); echoOut.gain.value = 0.7;
    pl.connect(echoOut); pr.connect(echoOut); echoOut.connect(this.comp);

    // space: a generated hall, 2.8 s of decaying stereo noise
    var len = Math.floor(ctx.sampleRate * 2.8), ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (var c = 0; c < 2; c++) {
      var d = ir.getChannelData(c), lp = 0;
      for (var j = 0; j < len; j++) {
        var t = j / len;
        lp += ((Math.random() * 2 - 1) - lp) * (0.9 - 0.6 * t); // darker as it fades
        d[j] = lp * Math.pow(1 - t, 3.2) * (j < 180 ? j / 180 : 1);
      }
    }
    this.spaceIn = ctx.createGain();
    var conv = ctx.createConvolver(); conv.buffer = ir;
    var spaceOut = ctx.createGain(); spaceOut.gain.value = 0.9;
    this.spaceIn.connect(conv); conv.connect(spaceOut); spaceOut.connect(this.comp);

    // a channel strip per voice: drive → filter → pan → level → (echo, space)
    var self = this;
    this.ch = ORDER.map(function (kind) {
      var s = {};
      s.in = ctx.createGain();
      s.drive = ctx.createWaveShaper(); s.drive.oversample = "2x";
      s.lp = ctx.createBiquadFilter(); s.lp.type = "lowpass"; s.lp.Q.value = 0.7;
      s.pan = ctx.createStereoPanner(); s.pan.pan.value = KINDS[kind].pan;
      s.level = ctx.createGain();
      s.echo = ctx.createGain(); s.space = ctx.createGain();
      s.in.connect(s.drive); s.drive.connect(s.lp); s.lp.connect(s.pan); s.pan.connect(s.level);
      s.level.connect(self.comp); s.level.connect(s.echo); s.level.connect(s.space);
      s.echo.connect(self.echoIn); s.space.connect(self.spaceIn);
      s.driveAmt = -1;
      return s;
    });
    this.apply();

    // A worker ticks the scheduler: timers in background tabs get throttled.
    var src = "var t=null;onmessage=function(e){clearInterval(t);if(e.data)t=setInterval(function(){postMessage(0)},e.data)}";
    try {
      this.timer = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
      this.timer.onmessage = function () { self.tick(); };
    } catch (e) {
      this.timer = { postMessage: function (ms) { clearInterval(self._iv); if (ms) self._iv = setInterval(function () { self.tick(); }, ms); } };
    }
  };

  function driveCurve(amt) {
    var k = 1 + amt * 14, n = 1024, c = new Float32Array(n), norm = Math.tanh(k);
    for (var i = 0; i < n; i++) { var x = i / (n - 1) * 2 - 1; c[i] = Math.tanh(k * x) / norm; }
    return c;
  }

  // Push the song's mix settings into the graph (cheap; call after any change).
  Engine.prototype.apply = function () {
    if (!this.ctx) return;
    var song = this.song, t = this.ctx.currentTime, self = this;
    this.master.gain.setTargetAtTime(song.volume * 0.9, t, 0.03);
    this.dl.delayTime.setTargetAtTime(0.75 * 60 / song.bpm, t, 0.05);
    this.dr.delayTime.setTargetAtTime(0.75 * 60 / song.bpm, t, 0.05);
    var u = tideU(song, this.beatNow());
    song.voices.forEach(function (v, i) {
      var s = self.ch[i];
      var on = audible(song, v);
      s.level.gain.setTargetAtTime(on ? v.level * v.level : 0, t, 0.02);
      s.echo.gain.setTargetAtTime(v.echo * 0.9, t, 0.03);
      s.space.gain.setTargetAtTime(v.space * 0.9, t, 0.03);
      var bright = Math.pow(2, -4 * song.tide.bright * (1 - u));
      s.lp.frequency.setTargetAtTime(Math.max(60, Math.min(20000, v.color * bright)), t, 0.05);
      var dq = Math.round(v.drive * 40) / 40;
      if (dq !== s.driveAmt) { s.driveAmt = dq; s.drive.curve = dq > 0 ? driveCurve(dq) : null; }
    });
  };

  // ── transport ────────────────────────────────────────────
  Engine.prototype.play = function () {
    this.init();
    if (this.ctx.state !== "running") this.ctx.resume();
    if (this.playing) return;
    this.playing = true;
    var now = this.ctx.currentTime;
    this.ch.forEach(function (s) { s.in.gain.cancelScheduledValues(0); s.in.gain.setValueAtTime(1, now); });
    this.cursorBeat = 0;
    this.cursorTime = this.ctx.currentTime + 0.08;
    this.anchors = [{ time: this.cursorTime, beat: 0, bpm: this.song.bpm }];
    this.lastSlots = this.song.voices.map(function () { return null; });
    this.timer.postMessage(25);
    this.tick();
    this.emit({ type: "transport", playing: true });
  };

  Engine.prototype.stop = function () {
    if (!this.playing) return;
    this.playing = false;
    this.timer.postMessage(0);
    var t = this.ctx.currentTime;
    // fade the channels out quickly so ringing notes don't hang on
    this.ch.forEach(function (s) { s.in.gain.cancelScheduledValues(t); s.in.gain.setValueAtTime(s.in.gain.value, t); s.in.gain.linearRampToValueAtTime(0, t + 0.12); });
    var self = this;
    setTimeout(function () { if (!self.playing) self.ch.forEach(function (s) { s.in.gain.cancelScheduledValues(0); s.in.gain.value = 1; }); }, 400);
    this.emit({ type: "transport", playing: false });
  };

  Engine.prototype.setTempo = function (bpm) {
    this.song.bpm = bpm;
    if (this.playing) this.anchors.push({ time: this.cursorTime, beat: this.cursorBeat, bpm: bpm });
    if (this.anchors.length > 32) this.anchors.splice(0, this.anchors.length - 32);
    this.apply();
  };

  // What you are hearing right now, in beats.
  Engine.prototype.beatNow = function () {
    if (!this.ctx || !this.playing || !this.anchors.length) return 0;
    var t = this.ctx.currentTime - (this.ctx.outputLatency || 0);
    var a = this.anchors[0];
    for (var i = this.anchors.length - 1; i >= 0; i--) if (this.anchors[i].time <= t) { a = this.anchors[i]; break; }
    return Math.max(0, a.beat + (t - a.time) * a.bpm / 60);
  };

  Engine.prototype.tick = function () {
    if (!this.playing) return;
    var ctx = this.ctx, song = this.song, horizon = ctx.currentTime + 0.14;
    if (this.cursorTime < ctx.currentTime - 0.25) {
      // we fell behind (tab asleep): skip ahead instead of firing a burst
      this.cursorBeat += (ctx.currentTime - this.cursorTime) * song.bpm / 60;
      this.cursorTime = ctx.currentTime;
      this.anchors.push({ time: this.cursorTime, beat: this.cursorBeat, bpm: song.bpm });
    }
    if (this.cursorTime >= horizon) return;
    var spb = 60 / song.bpm;
    var b0 = this.cursorBeat, b1 = b0 + (horizon - this.cursorTime) / spb, t0 = this.cursorTime;
    var self = this, hits = [];
    song.voices.forEach(function (v, i) {
      hits.length = 0;
      findTriggers(song, v, b0, b1, hits);
      hits.forEach(function (h) {
        var slot = Math.round(h.raw / GRID);
        if (song.snap && slot === self.lastSlots[i]) return;
        self.lastSlots[i] = slot;
        var when = Math.max(ctx.currentTime, t0 + (h.beat - b0) * spb);
        if (h.gated) return;
        if (audible(song, v)) self.note(i, when, h.hz, h.vel);
        self.emit({ type: "hit", voice: i, time: when, beat: h.beat, vel: h.vel, hz: h.hz, audible: audible(song, v) });
      });
    });
    this.cursorBeat = b1; this.cursorTime = horizon;
    // the tide moves the filters slowly; refresh about once a beat
    if (Math.floor(b1) !== Math.floor(b0)) this.apply();
  };

  // ── sounds ───────────────────────────────────────────────
  // One FM tone: carrier sine at f, modulator sine at f·ratio whose depth
  // (the index) starts at `mod` and relaxes as the note fades.
  function fm(ctx, dest, t, f, amp, v, o) {
    o = o || {};
    var attack = o.attack || 0.003, dec = Math.max(0.02, v.decay * (o.decayMul || 1));
    var end = t + attack + dec * 1.4 + 0.05;
    var car = ctx.createOscillator(), mod = ctx.createOscillator(), mg = ctx.createGain(), g = ctx.createGain();
    var fm = f * v.ratio;
    car.frequency.setValueAtTime(f * (o.drop || 1), t);
    mod.frequency.setValueAtTime(fm * (o.drop || 1), t);
    if (o.drop) {
      car.frequency.exponentialRampToValueAtTime(f, t + o.dropTime);
      mod.frequency.exponentialRampToValueAtTime(fm, t + o.dropTime);
    }
    if (o.detune) car.detune.value = o.detune;
    var idx = v.mod * fm;
    mg.gain.setValueAtTime(idx, t);
    mg.gain.setTargetAtTime(idx * (o.modFloor === undefined ? 0.25 : o.modFloor), t + attack, dec * (o.modSpeed || 0.35));
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(amp, t + attack);
    g.gain.setTargetAtTime(0, t + attack, dec / 4.5);
    mod.connect(mg); mg.connect(car.frequency); car.connect(g); g.connect(dest);
    car.start(t); mod.start(t); car.stop(end); mod.stop(end);
    car.onended = function () { g.disconnect(); mg.disconnect(); };
  }

  function noise(engine, dest, t, amp, dec, type, freq, q) {
    var ctx = engine.ctx, src = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    src.buffer = engine.noise; src.loop = true;
    f.type = type; f.frequency.value = freq; f.Q.value = q || 0.7;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(amp, t + 0.0015);
    g.gain.setTargetAtTime(0, t + 0.0015, dec / 4.5);
    src.connect(f); f.connect(g); g.connect(dest);
    src.start(t, Math.random() * 0.5); src.stop(t + dec * 1.4 + 0.05);
    src.onended = function () { g.disconnect(); };
  }

  Engine.prototype.note = function (i, t, hz, vel) {
    var v = this.song.voices[i], ctx = this.ctx, dest = this.ch[i].in, a = vel;
    t = Math.max(t, ctx.currentTime);
    switch (v.kind) {
      case "kick":
        fm(ctx, dest, t, hz, a * 0.95, v, { drop: 5.5, dropTime: 0.05, modFloor: 0.05, modSpeed: 0.08 });
        noise(this, dest, t, a * 0.12, 0.012, "highpass", 2500);
        break;
      case "snap":
        fm(ctx, dest, t, hz, a * 0.6, v, { decayMul: 0.55, drop: 1.6, dropTime: 0.03, modFloor: 0.1, modSpeed: 0.2 });
        noise(this, dest, t, a * 0.9, v.decay, "bandpass", Math.min(9000, hz * 9), 0.9);
        noise(this, dest, t, a * 0.3, v.decay * 0.5, "highpass", 6000);
        break;
      case "hat":
        noise(this, dest, t, a * 1.8, v.decay, "highpass", hz);
        fm(ctx, dest, t, hz * 0.31, a * 0.4, v, { attack: 0.001, modFloor: 0.6 });
        break;
      case "bass":
        fm(ctx, dest, t, hz, a * 0.62, v, { attack: 0.004, modFloor: 0.3, modSpeed: 0.3 });
        break;
      case "pluck":
        fm(ctx, dest, t, hz, a * 0.6, v, { attack: 0.002, modFloor: 0.08, modSpeed: 0.18 });
        break;
      case "pad":
        var song = this.song;
        // a triad stacked from the scale: the note, a third, a fifth
        [0, 2, 4].forEach(function (step, k) {
          var f = step ? hz * ratioUp(song, hz, step) : hz;
          var att = Math.min(0.6, Math.max(0.08, v.decay * 0.18));
          fm(ctx, dest, t, f, a * 0.2, v, { attack: att, detune: -6, modFloor: 0.6, modSpeed: 0.6 });
          fm(ctx, dest, t, f, a * 0.2, v, { attack: att, detune: 6, modFloor: 0.6, modSpeed: 0.6 });
        });
        break;
    }
  };

  // Ratio from hz up `step` scale degrees, staying inside the current scale.
  function ratioUp(song, hz, step) {
    var r = scaleOf(song).r, n = r.length, root = keyRoot(song);
    var oct = Math.floor(Math.log2(hz / root) + 1e-6), within = hz / (root * Math.pow(2, oct));
    var best = 0, bd = 9;
    for (var i = 0; i < n; i++) { var d = Math.abs(Math.log2(r[i] / within)); if (d < bd) { bd = d; best = i; } }
    var j = best + step, o = Math.floor(j / n);
    return (r[j % n] * Math.pow(2, o)) / r[best];
  }

  // A one-shot preview of a voice (used while turning its sound dials when stopped).
  Engine.prototype.preview = function (i) {
    this.init();
    if (this.ctx.state !== "running") this.ctx.resume();
    if (this.playing) return;
    var v = this.song.voices[i];
    this.note(i, this.ctx.currentTime + 0.02, pitchAt(this.song, v, 0), 0.9);
  };

  // ── recording ────────────────────────────────────────────
  Engine.prototype.record = function () {
    this.init();
    if (!root.MediaRecorder) return false;
    var dest = this.ctx.createMediaStreamDestination();
    this.out.connect(dest);
    var types = ["audio/mp4", "audio/webm;codecs=opus", "audio/webm"];
    var type = types.filter(function (t) { return MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t); })[0];
    var rec = this.rec = new MediaRecorder(dest.stream, type ? { mimeType: type, audioBitsPerSecond: 192000 } : undefined);
    var chunks = [];
    rec.ondataavailable = function (e) { if (e.data && e.data.size) chunks.push(e.data); };
    var self = this;
    this.recDone = new Promise(function (resolve) {
      rec.onstop = function () {
        self.out.disconnect(dest);
        resolve(new Blob(chunks, { type: rec.mimeType || type || "audio/webm" }));
      };
    });
    rec.start(500);
    return true;
  };
  Engine.prototype.stopRecording = function () {
    if (!this.rec) return Promise.resolve(null);
    var p = this.recDone; this.rec.stop(); this.rec = null;
    return p;
  };

  var api = {
    Engine: Engine, PRESETS: PRESETS, KINDS: KINDS, ORDER: ORDER, SCALES: SCALES, SHAPES: SHAPES,
    PULSES: PULSES, PULSE_NAMES: PULSE_NAMES, DRIFTS: DRIFTS, RATIOS: RATIOS, TIDES: TIDES, NOTE_NAMES: NOTE_NAMES,
    normalize: normalize, clone: clone, findTriggers: findTriggers, signal: signal, waveA: waveA, waveB: waveB,
    tideLine: tideLine, tideU: tideU, tideShift: tideShift, pitchAt: pitchAt, degHz: degHz, noteName: noteName,
    scaleOf: scaleOf, audible: audible, keyRoot: keyRoot, GRID: GRID
  };
  root.Hertz = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
