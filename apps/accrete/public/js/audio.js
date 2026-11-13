// Accrete · the sound. All of it is made here in the browser: short notes for what you do, and a low hum that
// gains a voice with each kind of body. One mute, one volume. Nothing plays until the page has been touched.
(function (root) {
  "use strict";
  // A minor pentatonic over three octaves; kinds further out sound lower.
  const SCALE = [220, 261.63, 293.66, 329.63, 392];
  const note = (i) => SCALE[((i % 5) + 5) % 5] * Math.pow(2, Math.floor(i / 5));

  function Sound() {
    let ctx = null, master = null, wet = null, hum = null, humVoices = [];
    let volume = 0.7, muted = false, lastBuy = 0, lastPull = 0;
    const log = [];                                   // what was asked to play, for the checks
    this.log = log;

    function start() {
      if (ctx) return true;
      const AC = root.AudioContext || root.webkitAudioContext;
      if (!AC) return false;
      ctx = new AC();
      master = ctx.createGain(); master.gain.value = muted ? 0 : volume;
      const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -14; comp.ratio.value = 4; comp.attack.value = 0.004; comp.release.value = 0.2;
      master.connect(comp); comp.connect(ctx.destination);
      // a room: noise that dies away
      const len = Math.floor(ctx.sampleRate * 2.6), ir = ctx.createBuffer(2, len, ctx.sampleRate);
      for (let ch = 0; ch < 2; ch++) { const d = ir.getChannelData(ch); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2); }
      const conv = ctx.createConvolver(); conv.buffer = ir;
      wet = ctx.createGain(); wet.gain.value = 0.32;
      wet.connect(conv); conv.connect(master);
      return true;
    }
    this.wake = function () { if (start() && ctx.state === "suspended") ctx.resume(); };
    this.ready = function () { return !!ctx; };
    this.context = function () { return ctx; };
    this.set = function (v, m) { volume = v; muted = m; if (master) master.gain.setTargetAtTime(muted ? 0 : volume, ctx.currentTime, 0.03); };
    this.muted = function () { return muted; };

    // one voice: an oscillator through its own envelope
    function tone(freq, o) {
      const t0 = ctx.currentTime + (o.at || 0), dur = o.dur || 0.3;
      const osc = ctx.createOscillator(), g = ctx.createGain();
      osc.type = o.type || "sine";
      osc.frequency.setValueAtTime(freq, t0);
      if (o.to) osc.frequency.exponentialRampToValueAtTime(o.to, t0 + dur);
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(o.gain || 0.2, t0 + (o.attack || 0.006));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(g); g.connect(master);
      if (o.wet) { const s = ctx.createGain(); s.gain.value = o.wet; g.connect(s); s.connect(wet); }
      osc.start(t0); osc.stop(t0 + dur + 0.05);
    }
    // a bell: a sine bent by another sine
    function bell(freq, o) {
      const t0 = ctx.currentTime + (o.at || 0), dur = o.dur || 1.2;
      const car = ctx.createOscillator(), mod = ctx.createOscillator(), mg = ctx.createGain(), g = ctx.createGain();
      car.frequency.value = freq; mod.frequency.value = freq * (o.ratio || 3.5);
      mg.gain.setValueAtTime(freq * (o.depth || 1.6), t0); mg.gain.exponentialRampToValueAtTime(1, t0 + dur * 0.6);
      mod.connect(mg); mg.connect(car.frequency);
      g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(o.gain || 0.16, t0 + 0.005); g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      car.connect(g); g.connect(master);
      const s = ctx.createGain(); s.gain.value = o.wet === undefined ? 0.6 : o.wet; g.connect(s); s.connect(wet);
      car.start(t0); mod.start(t0); car.stop(t0 + dur + 0.05); mod.stop(t0 + dur + 0.05);
    }
    function noise(o) {
      const t0 = ctx.currentTime + (o.at || 0), dur = o.dur || 1;
      const len = Math.floor(ctx.sampleRate * dur), b = ctx.createBuffer(1, len, ctx.sampleRate), d = b.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      const src = ctx.createBufferSource(); src.buffer = b;
      const f = ctx.createBiquadFilter(); f.type = o.type || "lowpass"; f.Q.value = o.q || 0.8;
      f.frequency.setValueAtTime(o.from || 200, t0); f.frequency.exponentialRampToValueAtTime(o.to || 2000, t0 + dur);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(o.gain || 0.2, t0 + dur * (o.peak || 0.8)); g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      src.connect(f); f.connect(g); g.connect(master);
      if (o.wet) { const s = ctx.createGain(); s.gain.value = o.wet; g.connect(s); s.connect(wet); }
      src.start(t0);
    }

    const SOUNDS = {
      // tier 1
      pull() { const t = ctx.currentTime; if (t - lastPull < 0.07) return false; lastPull = t; tone(96, { to: 58, dur: 0.12, gain: 0.16 }); },
      buy(o) {
        const t = ctx.currentTime; if (t - lastBuy < 0.045) return false; lastBuy = t;
        const k = o.k || 0, i = 12 - k + Math.floor(Math.random() * 2) * 2;
        tone(note(i), { type: "triangle", dur: 0.22, gain: 0.13, wet: 0.25 });
        tone(note(i) * 2, { dur: 0.12, gain: 0.04 });
      },
      open() { tone(note(9), { dur: 0.10, gain: 0.06 }); tone(note(11), { dur: 0.16, gain: 0.06, at: 0.07, wet: 0.3 }); },
      comet() { for (let i = 0; i < 5; i++) bell(note(15 + [0, 2, 4, 2, 5][i]), { at: i * 0.09, dur: 0.9, gain: 0.05, ratio: 2, depth: 0.5 }); },
      // tier 2
      bump(o) { const b = 7 - (o.k || 0); tone(note(b), { type: "triangle", dur: 0.5, gain: 0.15, wet: 0.5 }); tone(note(b + 3), { type: "triangle", dur: 0.6, gain: 0.13, at: 0.07, wet: 0.5 }); bell(note(b + 10), { at: 0.14, dur: 1.1, gain: 0.07 }); },
      first(o) { const b = 6 - (o.k || 0); [0, 2, 4, 7].forEach((s, i) => tone(note(b + s), { type: "triangle", dur: 0.5, gain: 0.12, at: i * 0.075, wet: 0.5 })); },
      reveal() { [5, 7, 9].forEach((s, i) => bell(note(s), { at: i * 0.11, dur: 1.3, gain: 0.08, ratio: 2 })); },
      upgrade() { [0, 2, 4].forEach((s, i) => tone(note(5 + s), { type: "triangle", dur: 0.55, gain: 0.10, at: i * 0.012, wet: 0.5 })); },
      shepherd() { tone(note(7), { type: "triangle", dur: 0.25, gain: 0.11, wet: 0.4 }); tone(note(9), { type: "triangle", dur: 0.25, gain: 0.11, at: 0.12, wet: 0.4 }); tone(note(7), { type: "triangle", dur: 0.4, gain: 0.11, at: 0.24, wet: 0.5 }); },
      milestone() { bell(note(10), { dur: 1.4, gain: 0.11 }); bell(note(12), { at: 0.13, dur: 1.7, gain: 0.11 }); },
      catch(o) { tone(300, { to: 1300, dur: 0.28, gain: 0.10, wet: 0.5 }); [0, 2, 4, 7].forEach((s, i) => bell(note(10 + s), { at: 0.1 + i * 0.06, dur: 1.2, gain: 0.07, ratio: o && o.surge ? 2 : 3.5 })); },
      node() { bell(note(7), { dur: 1.8, gain: 0.10, ratio: 2 }); bell(note(11), { at: 0.09, dur: 2.0, gain: 0.09, ratio: 2 }); },
      well() { tone(70, { to: 40, dur: 0.7, gain: 0.25, wet: 0.5 }); bell(note(2), { at: 0.1, dur: 1.6, gain: 0.08, ratio: 1.5 }); },
      // tier 3
      set() { [0, 2, 4, 7, 9].forEach((s, i) => bell(note(5 + s), { at: i * 0.07, dur: 2.2, gain: 0.09 })); tone(note(-5), { dur: 1.6, gain: 0.14, wet: 0.6 }); },
      ready() { [0, 4, 7].forEach((s, i) => { tone(note(s), { type: "triangle", dur: 1.6, gain: 0.10, at: i * 0.16, wet: 0.7 }); bell(note(10 + s), { at: i * 0.16, dur: 2.0, gain: 0.06 }); }); },
      ignite() {
        noise({ dur: 1.7, from: 90, to: 5000, gain: 0.30, peak: 0.95, wet: 0.4 });
        tone(45, { to: 150, dur: 1.7, gain: 0.26 });
        tone(110, { to: 30, dur: 1.4, gain: 0.5, at: 1.7 });
        noise({ dur: 2.2, from: 9000, to: 300, gain: 0.26, peak: 0.02, at: 1.7, wet: 0.8 });
        [0, 2, 4, 7, 9, 12].forEach((s, i) => { tone(note(s), { type: "triangle", dur: 3.6, gain: 0.07, at: 1.75 + i * 0.05, attack: 0.08, wet: 0.9 }); bell(note(10 + s), { at: 1.8 + i * 0.09, dur: 3.0, gain: 0.05 }); });
      },
      collapse() {
        tone(220, { to: 28, dur: 2.2, gain: 0.36, wet: 0.5 });
        noise({ dur: 2.2, from: 4000, to: 60, gain: 0.24, peak: 0.6, wet: 0.5 });
        [0, 3, 7, 10].forEach((s, i) => bell(note(-5 + s), { at: 2.2 + i * 0.12, dur: 4.0, gain: 0.09, ratio: 1.5, depth: 2.2 }));
      },
      bang() {
        noise({ dur: 0.5, from: 200, to: 30, gain: 0.02, peak: 0.5 });
        tone(36, { to: 22, dur: 3.0, gain: 0.6, at: 0.5 });
        noise({ dur: 3.4, from: 12000, to: 120, gain: 0.34, peak: 0.01, at: 0.5, wet: 0.9 });
        [0, 2, 4, 5, 7, 9, 12, 14].forEach((s, i) => bell(note(s), { at: 0.7 + i * 0.11, dur: 4.5, gain: 0.07, ratio: 2 }));
      },
      end() { [0, 4, 7, 12, 16, 19].forEach((s, i) => { tone(note(-5 + s), { type: "triangle", dur: 6, gain: 0.07, at: i * 0.35, attack: 0.4, wet: 1 }); bell(note(5 + s), { at: 0.2 + i * 0.35, dur: 5, gain: 0.05, ratio: 2 }); }); },
    };
    this.play = function (name, o) {
      log.push(name); if (log.length > 400) log.shift();
      if (!ctx || muted || ctx.state !== "running" || !SOUNDS[name]) return;
      try { SOUNDS[name](o || {}); } catch (e) { /* a sound must never stop the game */ }
    };

    // The hum: two low sines and a breath of noise; each kind of body you own adds a partial above them.
    this.hum = function (kinds, heat, surge) {
      if (!ctx || ctx.state !== "running") return;
      if (!hum) {
        hum = ctx.createGain(); hum.gain.value = 0; hum.connect(master);
        const s = ctx.createGain(); s.gain.value = 0.5; hum.connect(s); s.connect(wet);
        const mk = (f, g) => { const o = ctx.createOscillator(), gg = ctx.createGain(), l = ctx.createOscillator(), lg = ctx.createGain(); o.frequency.value = f; gg.gain.value = 0; l.frequency.value = 0.05 + Math.random() * 0.12; lg.gain.value = 0; l.connect(lg); lg.connect(gg.gain); o.connect(gg); gg.connect(hum); o.start(); l.start(); return { g: gg, lg, level: g }; };
        humVoices = [mk(55, 0.5), mk(55.4, 0.4), mk(82.4, 0.25)];
        for (let k = 0; k < 8; k++) humVoices.push(mk(110 * [1, 1.5, 2, 2.5, 3, 4, 5, 6][k], 0.11 / (1 + k * 0.35)));
        hum.gain.setTargetAtTime(0.11, ctx.currentTime, 2.5);
      }
      humVoices.forEach((v, i) => {
        const on = i < 3 ? 1 : kinds > i - 3 ? 1 : 0;
        const level = v.level * on * (i < 3 ? 1 : 0.6 + 0.6 * heat) * (surge ? 1.5 : 1);
        v.g.gain.setTargetAtTime(level, ctx.currentTime, 1.5);
        v.lg.gain.setTargetAtTime(level * 0.5, ctx.currentTime, 1.5);
      });
    };
  }
  root.AccreteSound = Sound;
})(typeof self !== "undefined" ? self : this);
