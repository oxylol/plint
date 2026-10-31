/* ─────────────────────────────────────────────────────────────
   Hertz · interface
   The orbit (every voice as a ring around one bar), the strip (the
   selected voice's waves unrolled, with a threshold you drag), dials,
   transport, songs. All drawing reads the same findTriggers() the
   engine plays from, so what you see is what you hear.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var H = window.Hertz, KINDS = H.KINDS, ORDER = H.ORDER;
  var SONG = ORDER.length; // index of the Song chip
  function $(s) { return document.querySelector(s); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ── state ────────────────────────────────────────────────
  var LS = "hertz.v1";
  var saved = null;
  try { saved = JSON.parse(localStorage.getItem(LS) || "null"); } catch (e) {}
  var engine = new H.Engine(saved && saved.song ? saved.song : H.PRESETS[0]);
  var songId = (saved && saved.id) || null;
  var sel = saved && saved.sel != null ? saved.sel : 0;
  var page = (saved && saved.page) || "wave";
  function S() { return engine.song; }

  var persistT = 0;
  function persist() {
    clearTimeout(persistT);
    persistT = setTimeout(function () {
      try { localStorage.setItem(LS, JSON.stringify({ song: S(), id: songId, sel: sel, page: page })); } catch (e) {}
    }, 300);
  }
  var dirty = true; // redraw even when stopped
  function changed() { dirty = true; hitsCache = {}; engine.apply(); persist(); }

  // ── formatting ───────────────────────────────────────────
  function hzParts(hz) {
    if (hz >= 1000) return [(hz / 1000).toFixed(hz >= 10000 ? 1 : 2), "kHz"];
    if (hz >= 100) return [String(Math.round(hz)), "Hz"];
    if (hz >= 10) return [hz.toFixed(1), "Hz"];
    if (hz >= 1) return [hz.toFixed(2), "Hz"];
    return [hz.toFixed(hz >= 0.1 ? 2 : 3), "Hz"];
  }
  function pct(x) { return [String(Math.round(x * 100)), "%"]; }
  function barHz(cyclesPerBar) { return cyclesPerBar * S().bpm / 240; }

  var WAVES = {
    sine: "M2 12c2.6-9 6.4-9 9 0s6.4 9 9 0",
    triangle: "M2 12l4.5-7 9 14 4.5-7",
    saw: "M2 17L11 6v11l9-11v11",
    square: "M2 17V7h5v10h5V7h5v10h3"
  };

  // ── parameters ───────────────────────────────────────────
  // Each dial: where its value lives, its range, how it reads.
  function listSpec(key, label, list, fmt, extra) {
    return Object.assign({ key: key, label: label, list: list, fmt: fmt }, extra || {});
  }
  function voiceSpecs(v) {
    var k = KINDS[v.kind];
    var wave = [
      listSpec("pulse", "Pulse", H.PULSES, function (x) { var p = hzParts(barHz(x)); return { num: p[0], unit: p[1], sub: H.PULSE_NAMES[x] }; }),
      listSpec("shape", "Shape", H.SHAPES, function (x) { return { icon: WAVES[x], sub: x[0].toUpperCase() + x.slice(1) }; }),
      { key: "phase", label: "Phase", min: 0, max: 1, steps: 32, wrap: true, fmt: function (x) { return { num: Math.round(x * 360) + "°", sub: "shift " + Math.round(x * 16) + "/16" }; } },
      { key: "thr", label: "Threshold", min: -1, max: 1, bipolar: true, fmt: function (x) { return { num: (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2), sub: "trigger level" }; } },
      listSpec("drift", "Drift", H.DRIFTS, function (x) { if (!x) return { num: "Off", sub: "second sine" }; var p = hzParts(barHz(x)); return { num: p[0], unit: p[1], sub: x >= 1 ? x + "× a bar" : "1 per " + (1 / x) + " bars" }; }),
      { key: "bend", label: "Bend", min: 0, max: 1, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "drift → time" }; } },
      { key: "melody", label: k.melodic ? "Melody" : "Pitch", min: 0, max: 12, steps: 12, fmt: function (x) { return { num: String(x), unit: k.melodic ? "steps" : "semi", sub: "drift → pitch" }; } },
      { key: "level", label: "Level", min: 0, max: 1, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "volume" }; } }
    ];
    var freq;
    if (k.melodic) {
      // a frequency dial that clicks between the notes of the current scale
      var n = H.scaleOf(S()).r.length, o0 = k.oct[0], o1 = k.oct[1], count = (o1 - o0 + 1) * n;
      freq = {
        key: "freq", label: "Freq", min: 0, max: count - 1, steps: count - 1,
        get: function (v) { return (v.oct - o0) * n + Math.min(v.deg, n - 1); },
        set: function (v, x) { v.oct = o0 + Math.floor(x / n); v.deg = x % n; },
        fmt: function (x) { var hz = H.degHz(S(), o0 + Math.floor(x / n), x % n), p = hzParts(hz); return { num: p[0], unit: p[1], sub: H.noteName(hz) }; }
      };
    } else {
      freq = { key: "freq", label: "Freq", min: k.freq[0], max: k.freq[1], log: true, fmt: function (x) { var p = hzParts(x); return { num: p[0], unit: p[1], sub: v.kind === "hat" ? "cutoff" : H.noteName(x) }; } };
    }
    var sound = [
      freq,
      listSpec("ratio", "Ratio", H.RATIOS, function (x) { return { num: "×" + x, sub: "mod sine " + hzParts(soundHz(v) * x).join(" ") }; }),
      { key: "mod", label: "Mod", min: 0, max: 8, curve: 2, fmt: function (x) { return { num: x.toFixed(x < 10 ? 1 : 0), sub: "FM depth" }; } },
      { key: "decay", label: "Decay", min: 0.02, max: 4, log: true, fmt: function (x) { return x < 1 ? { num: String(Math.round(x * 1000)), unit: "ms", sub: "fade" } : { num: x.toFixed(2), unit: "s", sub: "fade" }; } },
      { key: "color", label: "Color", min: 80, max: 18000, log: true, fmt: function (x) { var p = hzParts(x); return { num: p[0], unit: p[1], sub: "low-pass" }; } },
      { key: "drive", label: "Drive", min: 0, max: 1, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "saturation" }; } },
      { key: "echo", label: "Echo", min: 0, max: 1, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "dotted 1/8" }; } },
      { key: "space", label: "Space", min: 0, max: 1, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "reverb" }; } }
    ];
    return page === "sound" ? sound : wave;
  }
  function soundHz(v) { return H.pitchAt(S(), v, 0); }

  function songSpecs() {
    var s = S();
    return [
      { key: "bpm", label: "Tempo", min: 60, max: 180, steps: 120, obj: s, set: function (o, x) { engine.setTempo(x); }, fmt: function (x) { return { num: String(x), unit: "BPM", sub: (x / 60).toFixed(2) + " Hz beat" }; } },
      listSpec("key", "Key", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], function (x) { var p = hzParts(440 * Math.pow(2, (x - 9) / 12 - 2)); return { num: H.NOTE_NAMES[x], sub: p.join(" ") }; }, { obj: s, refresh: true }),
      listSpec("scale", "Scale", H.SCALES.map(function (x) { return x.id; }), function (x) { var sc = H.SCALES.filter(function (y) { return y.id === x; })[0]; return { num: sc.name.slice(0, 3), sub: sc.name }; }, { obj: s, refresh: true }),
      { key: "swing", label: "Swing", min: 0, max: 0.5, obj: s, fmt: function (x) { var p = pct(x * 2); return { num: p[0], unit: p[1], sub: "late 16ths" }; } },
      listSpec("bars", "Tide", H.TIDES, function (x) { var p = hzParts(s.bpm / 240 / x); return { num: String(x), unit: "bars", sub: p.join(" ") }; }, { obj: s.tide }),
      { key: "chords", label: "Chords", min: 0, max: 7, steps: 7, obj: s.tide, fmt: function (x) { return { num: String(x), unit: "steps", sub: "tide → pitch" }; } },
      { key: "density", label: "Density", min: 0, max: 1, obj: s.tide, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "tide → rhythm" }; } },
      { key: "bright", label: "Bright", min: 0, max: 1, obj: s.tide, fmt: function (x) { var p = pct(x); return { num: p[0], unit: p[1], sub: "tide → color" }; } }
    ];
  }

  // value ↔ 0..1
  function toNorm(sp, x) {
    if (sp.list) { var i = sp.list.indexOf(x); if (i < 0) i = nearest(sp.list, x); return i / (sp.list.length - 1); }
    if (sp.log) return Math.log(x / sp.min) / Math.log(sp.max / sp.min);
    var n = (x - sp.min) / (sp.max - sp.min);
    return sp.curve ? Math.pow(clamp(n, 0, 1), 1 / sp.curve) : n;
  }
  function fromNorm(sp, n) {
    n = clamp(n, 0, 1);
    if (sp.list) return sp.list[Math.round(n * (sp.list.length - 1))];
    if (sp.steps) return Math.round(sp.min + Math.round(n * sp.steps) / sp.steps * (sp.max - sp.min));
    var x;
    if (sp.log) x = sp.min * Math.pow(sp.max / sp.min, n);
    else x = sp.min + (sp.curve ? Math.pow(n, sp.curve) : n) * (sp.max - sp.min);
    // keep stored numbers tidy
    var mag = Math.abs(x) >= 100 ? 1 : Math.abs(x) >= 10 ? 10 : 1000;
    return Math.round(x * mag) / mag;
  }
  function nearest(list, x) {
    var b = 0, d = Infinity;
    list.forEach(function (y, i) { var dd = typeof y === "number" ? Math.abs(y - x) : (y === x ? 0 : 1); if (dd < d) { d = dd; b = i; } });
    return b;
  }

  // ── dials ────────────────────────────────────────────────
  var ARC = "M20.3 79.7A42 42 0 1 1 79.7 79.7";
  function polar(n) { var a = (135 + 270 * n) * Math.PI / 180; return [50 + 42 * Math.cos(a), 50 + 42 * Math.sin(a)]; }

  function Dial(sp, target, color, onChange) {
    var d = this;
    this.sp = sp; this.target = target; this.onChange = onChange;
    var root = this.el = el("div", "dial");
    root.style.setProperty("--c", color);
    root.tabIndex = 0;
    root.setAttribute("role", "slider");
    root.setAttribute("aria-label", sp.label);
    var ticks = "";
    if (sp.list && sp.list.length <= 13) {
      for (var i = 0; i < sp.list.length; i++) {
        var a = (135 + 270 * i / (sp.list.length - 1)) * Math.PI / 180;
        ticks += "M" + (50 + 49 * Math.cos(a)).toFixed(1) + " " + (50 + 49 * Math.sin(a)).toFixed(1) + "L" + (50 + 52 * Math.cos(a)).toFixed(1) + " " + (50 + 52 * Math.sin(a)).toFixed(1);
      }
    }
    root.innerHTML =
      '<div class="knob"><svg viewBox="0 0 100 100" aria-hidden="true">' +
      (ticks ? '<path class="ticks" d="' + ticks + '"/>' : "") +
      '<path class="track" d="' + ARC + '" pathLength="1"/>' +
      '<path class="arc" d="' + ARC + '" pathLength="1"/>' +
      '<circle class="thumb" r="4.6"/></svg><div class="cap"></div></div>' +
      '<div class="dial-label">' + esc(sp.label) + '</div><div class="dial-sub"></div>';
    this.arc = root.querySelector(".arc");
    this.thumb = root.querySelector(".thumb");
    this.cap = root.querySelector(".cap");
    this.sub = root.querySelector(".dial-sub");
    this.update();

    var start = null, lastTap = 0;
    root.addEventListener("pointerdown", function (e) {
      if (start) return; // one finger per dial
      root.setPointerCapture(e.pointerId);
      start = { id: e.pointerId, x: e.clientX, y: e.clientY, n: d.norm(), moved: false };
      root.classList.add("active");
      e.preventDefault();
    });
    root.addEventListener("pointermove", function (e) {
      if (!start || e.pointerId !== start.id) return;
      var dy = start.y - e.clientY, dx = e.clientX - start.x;
      if (!start.moved && Math.abs(dy) + Math.abs(dx) < 3) return;
      start.moved = true;
      var span = e.pointerType === "mouse" ? 260 : 200;
      d.setNorm(start.n + (dy + dx * 0.6) / span);
    });
    function end(e) {
      if (!start || e.pointerId !== start.id) return;
      var tapped = !start.moved;
      start = null;
      root.classList.remove("active");
      if (tapped) {
        var now = performance.now();
        if (now - lastTap < 320) { d.reset(); lastTap = 0; } else lastTap = now;
        if (d.onTap) d.onTap();
      }
    }
    root.addEventListener("pointerup", end);
    root.addEventListener("pointercancel", end);
    root.addEventListener("wheel", function (e) {
      e.preventDefault();
      var step = sp.list ? 1 / (sp.list.length - 1) : sp.steps ? 1 / sp.steps : 0.01;
      d.setNorm(d.norm() + (e.deltaY < 0 ? step : -step) * (sp.list || sp.steps ? 1 : Math.min(4, Math.abs(e.deltaY) / 25 + 1)));
    }, { passive: false });
    root.addEventListener("keydown", function (e) {
      var step = sp.list ? 1 / (sp.list.length - 1) : sp.steps ? 1 / sp.steps : 0.02;
      if (e.key === "ArrowUp" || e.key === "ArrowRight") { d.setNorm(d.norm() + step); e.preventDefault(); }
      if (e.key === "ArrowDown" || e.key === "ArrowLeft") { d.setNorm(d.norm() - step); e.preventDefault(); }
    });
  }
  Dial.prototype.get = function () { return this.sp.get ? this.sp.get(this.target) : this.target[this.sp.key]; };
  Dial.prototype.norm = function () { return toNorm(this.sp, this.get()); };
  Dial.prototype.setNorm = function (n) {
    var sp = this.sp;
    if (sp.wrap) n = ((n % 1) + 1) % 1;
    var x = fromNorm(sp, n);
    if (sp.wrap && x >= sp.max) x = sp.min;
    if (x === this.get()) return;
    this.set(x);
    if (sp.list || sp.steps) { // a detent: tiny press of the cap
      var r = this.el; r.classList.remove("tick"); void r.offsetWidth; r.classList.add("tick");
      clearTimeout(this.tt); this.tt = setTimeout(function () { r.classList.remove("tick"); }, 90);
    }
  };
  Dial.prototype.set = function (x) {
    if (this.sp.set) this.sp.set(this.target, x); else this.target[this.sp.key] = x;
    this.update();
    if (this.onChange) this.onChange(this.sp, x);
  };
  Dial.prototype.reset = function () {
    if (this.sp.def === undefined) return;
    this.set(this.sp.def);
  };
  Dial.prototype.update = function () {
    var sp = this.sp, x = this.get(), n = clamp(toNorm(sp, x), 0, 1), f = sp.fmt(x);
    if (sp.bipolar) { // grows from the top, either way
      this.arc.style.strokeDasharray = Math.max(0.0001, Math.abs(n - 0.5)) + " 2";
      this.arc.style.strokeDashoffset = String(-Math.min(n, 0.5));
    } else {
      this.arc.style.strokeDasharray = Math.max(0.0001, n) + " 2";
    }
    var p = polar(n);
    this.thumb.setAttribute("cx", p[0].toFixed(2)); this.thumb.setAttribute("cy", p[1].toFixed(2));
    var html = f.icon ? '<svg class="wave" viewBox="0 0 22 24"><path d="' + f.icon + '"/></svg>'
      : "<b>" + esc(f.num) + "</b>" + (f.unit ? "<i>" + esc(f.unit) + "</i>" : "");
    if (html !== this.capHtml) { this.cap.innerHTML = html; this.capHtml = html; }
    this.sub.textContent = f.sub || "";
    this.el.setAttribute("aria-valuetext", (f.num || f.sub) + (f.unit ? " " + f.unit : ""));
  };

  // ── building the deck ────────────────────────────────────
  var app = $("#app"), rail = $("#rail"), dialsEl = $("#dials"), toolsEl = $("#tools"), seg = $("#seg");
  var dials = [], chips = [];

  ORDER.forEach(function (kind, i) {
    var c = el("button", "chip", '<span class="glyph"></span><span>' + KINDS[kind].name + "</span>");
    c.style.setProperty("--c", KINDS[kind].color);
    c.addEventListener("click", function () { select(i); });
    rail.appendChild(c); chips.push(c);
  });
  var songChip = el("button", "chip song", '<span class="glyph"></span><span>Song</span>');
  songChip.addEventListener("click", function () { select(SONG); });
  rail.appendChild(songChip); chips.push(songChip);

  var ICON = {
    speaker: '<svg viewBox="0 0 24 24"><path d="M11 5 6.5 9H3.5v6h3l4.5 4zM15.5 9a4.5 4.5 0 0 1 0 6M18.3 6.3a8.5 8.5 0 0 1 0 11.4"/></svg>',
    muted: '<svg viewBox="0 0 24 24"><path d="M11 5 6.5 9H3.5v6h3l4.5 4zM16 9.5l5 5M21 9.5l-5 5"/></svg>',
    solo: '<svg viewBox="0 0 24 24"><path d="M4 17v-5a8 8 0 0 1 16 0v5M20 18a2 2 0 0 1-2 2h-1.5v-6H20zM4 18a2 2 0 0 0 2 2h1.5v-6H4z"/></svg>',
    mutate: '<svg viewBox="0 0 24 24" class="fill"><path d="M11 2.5l1.9 5.1 5.1 1.9-5.1 1.9L11 16.5l-1.9-5.1L4 9.5l5.1-1.9zM18.5 14l.95 2.55L22 17.5l-2.55.95L18.5 21l-.95-2.55L15 17.5l2.55-.95z"/></svg>',
    snap: '<svg viewBox="0 0 24 24"><path d="M4 5h16M4 19h16M6 5v14M10 5v14M14 5v14M18 5v14"/></svg>'
  };

  function select(i) {
    sel = i;
    renderDeck(true);
    persist();
    dirty = true;
  }

  function renderDeck(animate) {
    var song = S();
    chips.forEach(function (c, i) {
      c.classList.toggle("sel", i === sel);
      if (i < SONG) {
        var v = song.voices[i];
        c.classList.toggle("off", !H.audible(song, v));
        c.classList.toggle("solo", !!v.solo);
      }
    });
    app.classList.toggle("song-mode", sel === SONG);
    seg.dataset.page = page;
    seg.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.page === page)); });

    var specs, target, color;
    if (sel === SONG) { specs = songSpecs(); target = null; color = "#f5f5f7"; }
    else { target = song.voices[sel]; specs = voiceSpecs(target); color = KINDS[target.kind].color; }
    dialsEl.innerHTML = ""; dials = [];
    specs.forEach(function (sp) {
      var obj = sp.obj || target;
      if (sp.def === undefined) sp.def = sp.get ? sp.get(obj) : obj[sp.key];
      var defaults = defaultsFor(sel);
      if (defaults && defaults[sp.key] !== undefined && !sp.get) sp.def = defaults[sp.key];
      var d = new Dial(sp, obj, color, onDial);
      if (sel < SONG && page === "sound") d.onTap = function () { engine.preview(sel); };
      dials.push(d); dialsEl.appendChild(d.el);
    });
    if (animate) { dialsEl.classList.remove("swap"); void dialsEl.offsetWidth; dialsEl.classList.add("swap"); }
    renderTools();
    $("#stripTag").textContent = sel === SONG ? "Tide · " + song.tide.bars + " bars" : KINDS[song.voices[sel].kind].name + " · pulse + drift";
  }
  // Double-tap resets a dial to the starter song's value for that voice.
  function defaultsFor(i) {
    var base = H.normalize(H.PRESETS[0]);
    return i === SONG ? null : base.voices[i];
  }

  var previewT = 0;
  function onDial(sp) {
    if (sp.refresh) dials.forEach(function (d) { d.update(); });
    if (sp.key === "bpm" || sp.key === "bars") dials.forEach(function (d) { d.update(); });
    if (sel < SONG && page === "sound" && !engine.playing) {
      clearTimeout(previewT);
      previewT = setTimeout(function () { engine.preview(sel); }, 120);
    }
    if (sp.key === "ratio" || sp.key === "freq") dials.forEach(function (d) { d.update(); });
    $("#bpm").textContent = S().bpm;
    if (sel === SONG) $("#stripTag").textContent = "Tide · " + S().tide.bars + " bars";
    changed();
  }

  function renderTools() {
    toolsEl.innerHTML = "";
    var song = S();
    if (sel === SONG) {
      var snap = el("button", "tool" + (song.snap ? " on" : ""), ICON.snap + "<span>Snap</span>");
      snap.setAttribute("aria-pressed", String(song.snap));
      snap.addEventListener("click", function () { song.snap = !song.snap; renderTools(); changed(); toast(song.snap ? "Hits snap to 16ths" : "Free timing: hits land where the wave crosses"); });
      toolsEl.appendChild(snap);
      return;
    }
    var v = song.voices[sel], c = KINDS[v.kind].color;
    var mute = el("button", "tool" + (v.mute ? " on" : ""), v.mute ? ICON.muted : ICON.speaker);
    mute.style.setProperty("--c", "#f5f5f7");
    mute.setAttribute("aria-label", v.mute ? "Unmute" : "Mute");
    mute.addEventListener("click", function () { v.mute = !v.mute; renderDeck(false); changed(); });
    var solo = el("button", "tool" + (v.solo ? " on" : ""), ICON.solo);
    solo.style.setProperty("--c", c);
    solo.setAttribute("aria-label", "Solo");
    solo.addEventListener("click", function () { v.solo = !v.solo; renderDeck(false); changed(); });
    var mut = el("button", "tool", ICON.mutate);
    mut.setAttribute("aria-label", "Mutate this voice");
    mut.addEventListener("click", function () { mutate(v); });
    toolsEl.appendChild(mute); toolsEl.appendChild(solo); toolsEl.appendChild(mut);
  }

  seg.addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b || b.dataset.page === page) return;
    page = b.dataset.page; renderDeck(true); persist();
  });

  // ── mutate: a musical dice roll for one voice ────────────
  function pick(a) { return a[Math.floor(Math.random() * a.length)]; }
  function mutate(v) {
    var melodic = KINDS[v.kind].melodic;
    var pulses = { kick: [2, 4, 4, 4, 8], bass: [4, 6, 8, 8, 16], snap: [2, 2, 2, 4, 8], pad: [0.5, 1, 1, 2], pluck: [3, 6, 8, 12, 16], hat: [4, 8, 12, 16, 16] }[v.kind];
    if (page === "sound") {
      v.ratio = pick(v.kind === "hat" || v.kind === "snap" ? [1.41, 1.5, 2.5, 2.83, 3.5] : [0.5, 1, 1, 2, 2, 3, 3.5, 4]);
      v.mod = Math.round((0.2 + Math.random() * (v.kind === "pad" ? 1.5 : 3.5)) * 10) / 10;
      if (melodic) v.color = Math.round(600 * Math.pow(12, Math.random()));
      v.drive = Math.round(Math.random() * 0.5 * 100) / 100;
    } else {
      v.pulse = pick(pulses);
      v.shape = pick(["sine", "sine", "sine", "triangle", "saw", "square"]);
      v.phase = v.kind === "snap" ? pick([0.5, 0.5, 0.25, 0.75]) : pick([0, 0, 0, 0.25, 0.5, 0.75]);
      v.thr = Math.round((-0.25 + Math.random() * 0.85) * 100) / 100;
      v.drift = pick([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5]);
      v.bend = Math.round(Math.random() * 0.75 * 100) / 100;
      if (melodic) v.melody = pick([2, 3, 4, 5, 7, 9]);
    }
    dialsEl.classList.add("animating");
    dials.forEach(function (d) { d.update(); });
    setTimeout(function () { dialsEl.classList.remove("animating"); }, 420);
    changed();
  }

  // ── canvases ─────────────────────────────────────────────
  var FONT = getComputedStyle(document.body).fontFamily;
  var orbitC = $("#orbit"), stripC = $("#strip"), og = orbitC.getContext("2d"), sg = stripC.getContext("2d");
  var DPR = Math.min(2, window.devicePixelRatio || 1);
  function fit(c) {
    var r = c.getBoundingClientRect();
    var w = Math.max(1, Math.round(r.width * DPR)), h = Math.max(1, Math.round(r.height * DPR));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; dirty = true; }
    return { w: r.width, h: r.height };
  }

  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")";
  }

  // Ring sizes ease toward the selection, like a spring with no overshoot.
  var weights = ORDER.map(function (_, i) { return i === sel ? 2.4 : 1; });
  var energy = ORDER.map(function () { return 0; });
  var blooms = []; // {voice, beat, time, vel, born}
  var lastBeat = 0;

  engine.on(function (e) {
    if (e.type === "hit" && e.audible) blooms.push({ voice: e.voice, beat: e.beat, time: e.time, vel: e.vel, born: 0 });
    if (e.type === "transport") { app.classList.toggle("playing", e.playing); $("#playBtn").setAttribute("aria-label", e.playing ? "Stop" : "Play"); $("#core").setAttribute("aria-label", e.playing ? "Stop" : "Play"); dirty = true; }
  });

  var freqData = null;
  var BANDS = 40;
  // Spectrum colours follow the voices: red at the bottom, violet at the top.
  var SPECTRUM = ORDER.map(function (k) { return KINDS[k].color; });
  function specColor(t, a) {
    var x = clamp(t, 0, 1) * (SPECTRUM.length - 1), i = Math.floor(x), f = x - i;
    var c0 = parseInt(SPECTRUM[i].slice(1), 16), c1 = parseInt(SPECTRUM[Math.min(i + 1, SPECTRUM.length - 1)].slice(1), 16);
    var r = (c0 >> 16 & 255) * (1 - f) + (c1 >> 16 & 255) * f, g = (c0 >> 8 & 255) * (1 - f) + (c1 >> 8 & 255) * f, b = (c0 & 255) * (1 - f) + (c1 & 255) * f;
    return "rgba(" + (r | 0) + "," + (g | 0) + "," + (b | 0) + "," + a + ")";
  }

  var hitsCache = {}; // per voice per bar; cleared by changed() and drags
  function hitsFor(i, bar) {
    var k = i + ":" + bar;
    if (!hitsCache[k]) hitsCache[k] = H.findTriggers(S(), S().voices[i], bar * 4, bar * 4 + 4);
    return hitsCache[k];
  }

  function geometry(size) {
    var R = size / 2 - 4, R0 = R * 0.235, inner = R0 + size * 0.03, band = R - inner;
    var total = weights.reduce(function (a, b) { return a + b; }, 0), r = inner, out = [];
    weights.forEach(function (w) { var h = band * w / total; out.push({ r: r + h / 2, a: h * 0.4, h: h }); r += h; });
    return { R: R, R0: R0, rings: out };
  }

  function drawOrbit(now, beat) {
    var box = fit(orbitC), size = box.w, cx = size / 2, cy = size / 2;
    og.setTransform(DPR, 0, 0, DPR, 0, 0);
    og.clearRect(0, 0, size, size);
    var song = S(), g = geometry(size), bar = Math.floor(beat / 4), phase = (beat / 4) % 1;
    var playing = engine.playing;

    // glow behind everything, tinted by whatever just played
    var tot = 0, rr = 0, gg = 0, bb = 0;
    energy.forEach(function (e, i) { if (e > 0.01) { var n = parseInt(KINDS[ORDER[i]].color.slice(1), 16); rr += (n >> 16 & 255) * e; gg += (n >> 8 & 255) * e; bb += (n & 255) * e; tot += e; } });
    if (tot > 0.02) {
      var glow = og.createRadialGradient(cx, cy, g.R0, cx, cy, g.R * 1.05);
      glow.addColorStop(0, "rgba(" + (rr / tot | 0) + "," + (gg / tot | 0) + "," + (bb / tot | 0) + "," + Math.min(0.16, tot * 0.06) + ")");
      glow.addColorStop(1, "rgba(0,0,0,0)");
      og.fillStyle = glow; og.beginPath(); og.arc(cx, cy, g.R * 1.05, 0, Math.PI * 2); og.fill();
    }

    // radar sweep: a faint wedge trailing the playhead
    var ang = -Math.PI / 2 + phase * Math.PI * 2;
    if (playing && og.createConicGradient) {
      // conic angles start at 3 o'clock; the wedge brightens up to the playhead
      var cg = og.createConicGradient(ang - Math.PI * 0.6, cx, cy);
      cg.addColorStop(0, "rgba(255,255,255,0)");
      cg.addColorStop(0.3, "rgba(255,255,255,0.055)");
      cg.addColorStop(0.3001, "rgba(255,255,255,0)");
      cg.addColorStop(1, "rgba(255,255,255,0)");
      og.fillStyle = cg; og.beginPath(); og.arc(cx, cy, g.R, 0, Math.PI * 2); og.arc(cx, cy, g.R0, 0, Math.PI * 2, true); og.fill();
    }

    // bar marks: four beats around the outside
    og.strokeStyle = "rgba(255,255,255,0.14)"; og.lineWidth = 1;
    for (var q = 0; q < 16; q++) {
      var qa = -Math.PI / 2 + q / 16 * Math.PI * 2, len = q % 4 ? 3 : 7;
      og.beginPath(); og.moveTo(cx + Math.cos(qa) * (g.R + 1), cy + Math.sin(qa) * (g.R + 1));
      og.lineTo(cx + Math.cos(qa) * (g.R + 1 - len), cy + Math.sin(qa) * (g.R + 1 - len)); og.stroke();
    }

    song.voices.forEach(function (v, i) {
      var ring = g.rings[i], col = KINDS[v.kind].color, isSel = i === sel, on = H.audible(song, v);
      var alpha = on ? (sel === SONG || isSel ? 1 : 0.5) : 0.14;
      var N = isSel ? 288 : 176;
      // the wave around one bar
      og.beginPath();
      for (var k = 0; k <= N; k++) {
        var t = k / N, a = -Math.PI / 2 + t * Math.PI * 2;
        var rad = ring.r + ring.a * H.signal(v, bar * 4 + t * 4);
        var x = cx + Math.cos(a) * rad, y = cy + Math.sin(a) * rad;
        if (k) og.lineTo(x, y); else og.moveTo(x, y);
      }
      og.closePath();
      if (isSel && on) { og.strokeStyle = rgba(col, 0.16); og.lineWidth = 7; og.stroke(); }
      og.strokeStyle = rgba(col, alpha * (0.75 + energy[i] * 0.25)); og.lineWidth = isSel ? 2 : 1.3; og.stroke();

      // threshold ring
      var tr = ring.r + ring.a * v.thr;
      og.setLineDash([2, 4]); og.strokeStyle = "rgba(255,255,255," + (isSel ? 0.5 : 0.16) * (on ? 1 : 0.5) + ")"; og.lineWidth = 1;
      og.beginPath(); og.arc(cx, cy, tr, 0, Math.PI * 2); og.stroke(); og.setLineDash([]);

      // where it fires
      hitsFor(i, bar).forEach(function (h) {
        var t = (h.beat / 4) - bar; if (t < 0 || t >= 1) return;
        var a = -Math.PI / 2 + t * Math.PI * 2, x = cx + Math.cos(a) * tr, y = cy + Math.sin(a) * tr;
        var played = playing && h.beat <= beat, rad = (isSel ? 2.2 : 1.6) + h.vel * (isSel ? 2.4 : 1.6);
        og.beginPath(); og.arc(x, y, rad, 0, Math.PI * 2);
        if (h.gated) { og.strokeStyle = rgba(col, 0.3 * alpha); og.lineWidth = 1; og.stroke(); }
        else { og.fillStyle = played ? rgba(col, alpha) : "rgba(255,255,255," + (0.18 + 0.5 * alpha) * (on ? 1 : 0.4) + ")"; og.fill(); }
      });
    });

    // blooms: rings that burst from a dot when it sounds
    var latency = engine.ctx ? (engine.ctx.outputLatency || 0) : 0, at = engine.ctx ? engine.ctx.currentTime - latency : 0;
    blooms = blooms.filter(function (b) {
      if (!b.born) { if (at < b.time) return true; b.born = now; energy[b.voice] = Math.min(1, energy[b.voice] + b.vel * 0.55); }
      var age = (now - b.born) / 520; if (age >= 1) return false;
      var v = song.voices[b.voice], ring = g.rings[b.voice], tr = ring.r + ring.a * v.thr;
      var t = (b.beat / 4) % 1, a = -Math.PI / 2 + t * Math.PI * 2;
      var x = cx + Math.cos(a) * tr, y = cy + Math.sin(a) * tr, e = 1 - Math.pow(1 - age, 3);
      og.beginPath(); og.arc(x, y, 3 + e * (5 + b.vel * 7), 0, Math.PI * 2);
      og.strokeStyle = rgba(KINDS[v.kind].color, (1 - age) * 0.8 * b.vel); og.lineWidth = 1 + (1 - age) * 1.4; og.stroke();
      return true;
    });

    // playhead
    if (playing) {
      var x0 = cx + Math.cos(ang) * g.R0, y0 = cy + Math.sin(ang) * g.R0, x1 = cx + Math.cos(ang) * g.R, y1 = cy + Math.sin(ang) * g.R;
      var lg = og.createLinearGradient(x0, y0, x1, y1);
      lg.addColorStop(0, "rgba(255,255,255,0.05)"); lg.addColorStop(0.6, "rgba(255,255,255,0.55)"); lg.addColorStop(1, "rgba(255,255,255,0.9)");
      og.strokeStyle = lg; og.lineWidth = 1.5; og.beginPath(); og.moveTo(x0, y0); og.lineTo(x1, y1); og.stroke();
      og.fillStyle = "#fff"; og.beginPath(); og.arc(x1, y1, 2.6, 0, Math.PI * 2); og.fill();
    }

    // the core: live spectrum of everything you hear
    var core = og.createRadialGradient(cx, cy - g.R0 * 0.3, g.R0 * 0.1, cx, cy, g.R0);
    var kickE = energy[0];
    core.addColorStop(0, "rgba(255,255,255," + (0.1 + kickE * 0.1) + ")");
    core.addColorStop(1, "rgba(255,255,255,0.02)");
    og.fillStyle = core; og.beginPath(); og.arc(cx, cy, g.R0, 0, Math.PI * 2); og.fill();
    og.strokeStyle = "rgba(255,255,255,0.14)"; og.lineWidth = 1; og.stroke();
    if (engine.analyser && playing) {
      var an = engine.analyser;
      if (!freqData) freqData = new Uint8Array(an.frequencyBinCount);
      an.getByteFrequencyData(freqData);
      // mirrored left and right: lows at the top, highs at the bottom
      var nyq = engine.ctx.sampleRate / 2, f0 = 30, f1 = 16000, bw = Math.PI / BANDS;
      for (var b = 0; b < BANDS; b++) {
        var lo = f0 * Math.pow(f1 / f0, b / BANDS), hi = f0 * Math.pow(f1 / f0, (b + 1) / BANDS);
        var i0 = Math.floor(lo / nyq * freqData.length), i1 = Math.max(i0 + 1, Math.ceil(hi / nyq * freqData.length)), m = 0;
        for (var j = i0; j < i1 && j < freqData.length; j++) if (freqData[j] > m) m = freqData[j];
        var mag = Math.pow(m / 255, 1.6);
        if (mag < 0.02) continue;
        var r0 = g.R0 * 0.36, r1 = r0 + (g.R0 * 0.58) * mag;
        og.strokeStyle = specColor(b / (BANDS - 1), 0.35 + mag * 0.65);
        og.lineWidth = Math.max(1.2, r0 * bw * 0.6);
        og.lineCap = "round";
        for (var side = -1; side <= 1; side += 2) {
          var ba = -Math.PI / 2 + side * (b + 0.5) * bw;
          og.beginPath(); og.moveTo(cx + Math.cos(ba) * r0, cy + Math.sin(ba) * r0); og.lineTo(cx + Math.cos(ba) * r1, cy + Math.sin(ba) * r1); og.stroke();
        }
      }
      og.lineCap = "butt";
    }
    return g;
  }

  function drawStrip(beat) {
    var box = fit(stripC), w = box.w, h = box.h, song = S();
    sg.setTransform(DPR, 0, 0, DPR, 0, 0);
    sg.clearRect(0, 0, w, h);
    var mid = h / 2, amp = h / 2 - 9;
    function Y(val) { return mid - val * amp; }

    if (sel === SONG) {
      // the tide over its whole cycle, with the chord steps it makes
      var bars = song.tide.bars, total = bars * 4, pos = engine.playing ? beat % total : 0;
      sg.strokeStyle = "rgba(255,255,255,0.06)"; sg.lineWidth = 1;
      for (var b = 1; b < bars; b++) { var bx = b / bars * w; sg.beginPath(); sg.moveTo(bx, 6); sg.lineTo(bx, h - 6); sg.stroke(); }
      sg.beginPath();
      for (var k = 0; k <= 200; k++) { var t = k / 200, y = Y(H.tideU(song, t * total) * 2 - 1); if (k) sg.lineTo(t * w, y); else sg.moveTo(0, y); }
      sg.lineTo(w, h); sg.lineTo(0, h); sg.closePath();
      var fillG = sg.createLinearGradient(0, 0, 0, h); fillG.addColorStop(0, "rgba(255,255,255,0.16)"); fillG.addColorStop(1, "rgba(255,255,255,0)");
      sg.fillStyle = fillG; sg.fill();
      sg.beginPath();
      for (k = 0; k <= 200; k++) { t = k / 200; y = Y(H.tideU(song, t * total) * 2 - 1); if (k) sg.lineTo(t * w, y); else sg.moveTo(0, y); }
      sg.strokeStyle = "rgba(255,255,255,0.85)"; sg.lineWidth = 1.6; sg.stroke();
      if (song.tide.chords) {
        sg.beginPath(); sg.strokeStyle = "rgba(74,163,255,0.8)"; sg.lineWidth = 1.2;
        for (b = 0; b < bars; b++) {
          var sh = H.tideShift(song, b * 4), yy = h - 8 - sh / Math.max(1, song.tide.chords) * (h * 0.42);
          if (b) sg.lineTo(b / bars * w, yy); else sg.moveTo(0, yy);
          sg.lineTo((b + 1) / bars * w, yy);
        }
        sg.stroke();
      }
      if (engine.playing) { var px = pos / total * w; sg.fillStyle = "#fff"; sg.fillRect(px - 0.75, 4, 1.5, h - 8); }
      return;
    }

    var v = song.voices[sel], col = KINDS[v.kind].color, bar = Math.floor(beat / 4), on = H.audible(song, v);
    // 16th grid
    for (var q = 1; q < 16; q++) {
      sg.fillStyle = q % 4 ? "rgba(255,255,255,0.035)" : "rgba(255,255,255,0.09)";
      sg.fillRect(q / 16 * w - 0.5, 4, 1, h - 8);
    }
    var N = Math.max(160, Math.round(w)), d = 1 + v.bend;
    function line(fn, style, width, dash) {
      sg.beginPath();
      for (var k = 0; k <= N; k++) { var t = k / N, y = Y(fn(bar * 4 + t * 4)); if (k) sg.lineTo(t * w, y); else sg.moveTo(0, y); }
      sg.setLineDash(dash || []); sg.strokeStyle = style; sg.lineWidth = width; sg.stroke(); sg.setLineDash([]);
    }
    line(function (b) { return H.waveA(v, b) / d; }, "rgba(255,255,255,0.16)", 1);
    if (v.drift && v.bend) line(function (b) { return v.bend * H.waveB(v, b) / d; }, rgba(col, 0.45), 1, [3, 3]);
    line(function (b) { return H.signal(v, b); }, rgba(col, on ? 0.18 : 0.06), 6);
    line(function (b) { return H.signal(v, b); }, rgba(col, on ? 1 : 0.35), 1.8);

    // tide line
    if (song.tide.density) {
      var tl = H.tideLine(song, v, engine.playing ? beat : bar * 4);
      if (tl > v.thr && tl < 1.2) {
        sg.fillStyle = "rgba(255,255,255,0.05)"; sg.fillRect(0, Y(Math.min(1.1, tl)) - 0.5, w, 1);
      }
    }
    // threshold
    var ty = Y(v.thr);
    sg.setLineDash([4, 4]); sg.strokeStyle = "rgba(255,255,255," + (dragging ? 0.95 : 0.6) + ")"; sg.lineWidth = 1;
    sg.beginPath(); sg.moveTo(0, ty); sg.lineTo(w, ty); sg.stroke(); sg.setLineDash([]);
    // grip
    sg.fillStyle = "rgba(255,255,255,0.92)";
    roundRect(sg, w - 30, ty - 5, 22, 10, 5); sg.fill();
    sg.fillStyle = "rgba(0,0,0,0.5)"; sg.fillRect(w - 23, ty - 0.5, 8, 1);
    if (dragging) {
      sg.font = "600 11px " + FONT;
      sg.fillStyle = "#fff"; sg.textAlign = "right";
      sg.fillText((v.thr >= 0 ? "+" : "−") + Math.abs(v.thr).toFixed(2), w - 36, ty + (ty < 18 ? 14 : -6));
    }
    // hits
    hitsFor(sel, bar).forEach(function (hh) {
      var t = hh.beat / 4 - bar; if (t < 0 || t >= 1) return;
      var x = t * w, played = engine.playing && hh.beat <= beat;
      sg.beginPath(); sg.arc(x, ty, 2.5 + hh.vel * 2.5, 0, Math.PI * 2);
      if (hh.gated) { sg.strokeStyle = rgba(col, 0.4); sg.lineWidth = 1; sg.stroke(); }
      else { sg.fillStyle = played ? col : "rgba(255,255,255,0.75)"; sg.fill(); }
    });
    if (engine.playing) {
      var ph = (beat / 4 - bar) * w;
      sg.fillStyle = "rgba(255,255,255,0.85)"; sg.fillRect(ph - 0.75, 3, 1.5, h - 6);
    }
  }
  function roundRect(c, x, y, w, h, r) {
    c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
  }

  // ── direct manipulation ──────────────────────────────────
  // Drag anywhere on the strip: the threshold follows your finger 1:1,
  // keeping the offset from where you grabbed it.
  var dragging = null;
  stripC.addEventListener("pointerdown", function (e) {
    if (sel === SONG || dragging) return;
    var r = stripC.getBoundingClientRect(), amp = r.height / 2 - 9;
    var v = S().voices[sel], val = (r.height / 2 - (e.clientY - r.top)) / amp;
    dragging = { id: e.pointerId, off: Math.abs(val - v.thr) < 0.35 ? v.thr - val : 0, amp: amp, top: r.top, h: r.height };
    stripC.setPointerCapture(e.pointerId);
    move(e);
    e.preventDefault();
  });
  function move(e) {
    if (!dragging || e.pointerId !== dragging.id) return;
    var v = S().voices[sel];
    var val = (dragging.h / 2 - (e.clientY - dragging.top)) / dragging.amp + dragging.off;
    var nv = Math.round(clamp(val, -1, 1) * 100) / 100;
    if (nv !== v.thr) {
      v.thr = nv;
      hitsCache = {}; dirty = true;
      dials.forEach(function (d) { if (d.sp.key === "thr") d.update(); });
      persist();
    }
  }
  stripC.addEventListener("pointermove", move);
  function endDrag(e) { if (dragging && e.pointerId === dragging.id) { dragging = null; dirty = true; } }
  stripC.addEventListener("pointerup", endDrag);
  stripC.addEventListener("pointercancel", endDrag);

  // Tap a ring to select its voice; tap the core to play or stop.
  var orbitDown = null, lastGeom = null;
  orbitC.addEventListener("pointerdown", function (e) { orbitDown = { x: e.clientX, y: e.clientY }; });
  orbitC.addEventListener("pointerup", function (e) {
    if (!orbitDown || Math.hypot(e.clientX - orbitDown.x, e.clientY - orbitDown.y) > 10 || !lastGeom) { orbitDown = null; return; }
    orbitDown = null;
    var r = orbitC.getBoundingClientRect(), dx = e.clientX - r.left - r.width / 2, dy = e.clientY - r.top - r.height / 2, dist = Math.hypot(dx, dy);
    var hit = -1;
    lastGeom.rings.forEach(function (ring, i) { if (Math.abs(dist - ring.r) <= ring.h / 2) hit = i; });
    if (hit >= 0) select(hit);
  });

  // ── the frame loop ───────────────────────────────────────
  var lastPos = "", lastTide = "";
  function frame(now) {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    var song = S();
    var beat = engine.playing ? engine.beatNow() : lastBeat;
    if (engine.playing) lastBeat = beat;

    var moving = false;
    weights = weights.map(function (w, i) {
      var target = sel === SONG ? 1 : i === sel ? 2.4 : 1, nw = w + (target - w) * 0.18;
      if (Math.abs(nw - target) > 0.002) moving = true; else nw = target;
      return nw;
    });
    var glowing = false;
    energy = energy.map(function (e) { var ne = e * 0.9; if (ne > 0.005) glowing = true; return ne > 0.005 ? ne : 0; });
    chips.forEach(function (c, i) {
      if (i >= SONG) return;
      var e = energy[i].toFixed(2);
      if (c._e !== e) { c._e = e; c.style.setProperty("--e", e); }
    });

    if (engine.playing || moving || glowing || dirty || blooms.length || dragging) {
      lastGeom = drawOrbit(now, beat);
      drawStrip(beat);
      dirty = false;
    }
    // position: bar.beat, and how far through the tide we are
    var p = engine.playing ? (Math.floor(beat / 4) + 1) + "." + (Math.floor(beat % 4) + 1) : "1.1";
    if (p !== lastPos) { $("#pos").textContent = p; lastPos = p; }
    var tide = engine.playing ? H.tideU(song, beat) : 0;
    var ts = tide.toFixed(3);
    if (ts !== lastTide) { lastTide = ts; $("#tideFill").style.transform = "scaleX(" + ts + ")"; }
  }
  requestAnimationFrame(frame);
  window.addEventListener("resize", function () { dirty = true; });

  // ── transport ────────────────────────────────────────────
  var wake = null;
  function toggle() {
    if (engine.playing) { engine.stop(); lastBeat = 0; releaseWake(); }
    else { engine.play(); holdWake(); }
    dirty = true;
  }
  function holdWake() {
    try { if (navigator.wakeLock) navigator.wakeLock.request("screen").then(function (w) { wake = w; }).catch(function () {}); } catch (e) {}
  }
  function releaseWake() { try { if (wake) wake.release(); } catch (e) {} wake = null; }
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden && engine.playing) { holdWake(); if (engine.ctx && engine.ctx.state !== "running") engine.ctx.resume(); }
  });
  $("#playBtn").addEventListener("click", toggle);
  $("#core").addEventListener("click", toggle);
  $("#tempoBtn").addEventListener("click", function () { select(SONG); });
  $("#posBtn").addEventListener("click", function () { select(SONG); });
  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); toggle(); }
    var n = parseInt(e.key, 10);
    if (n >= 1 && n <= 7) select(n - 1);
  });

  // ── record ───────────────────────────────────────────────
  var recBtn = $("#recBtn"), recT = 0, recStart = 0;
  recBtn.addEventListener("click", function () {
    if (engine.rec) {
      clearInterval(recT);
      recBtn.classList.remove("on"); $("#recTime").textContent = "Rec";
      engine.stopRecording().then(deliver);
      return;
    }
    if (!engine.record()) { toast("This browser can't record audio"); return; }
    if (!engine.playing) toggle();
    recBtn.classList.add("on"); recStart = Date.now();
    var tickRec = function () { var s = Math.floor((Date.now() - recStart) / 1000); $("#recTime").textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
    tickRec(); recT = setInterval(tickRec, 500);
  });
  function deliver(blob) {
    if (!blob || !blob.size) { toast("Nothing recorded"); return; }
    var ext = /mp4|aac|m4a/.test(blob.type) ? "m4a" : "webm";
    var name = "Hertz - " + S().name.replace(/[\/\\:*?"<>|]/g, "") + "." + ext;
    var file = null;
    try { file = new File([blob], name, { type: blob.type }); } catch (e) {}
    if (file && navigator.canShare && navigator.canShare({ files: [file] })) {
      navigator.share({ files: [file], title: name }).catch(function () { download(blob, name); });
    } else download(blob, name);
  }
  function download(blob, name) {
    var a = el("a"); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
    toast("Saved " + name);
  }

  // ── songs ────────────────────────────────────────────────
  var sheetOpen = false;
  function openSheet() {
    sheetOpen = true;
    $("#nameInput").value = S().name;
    document.body.classList.add("sheet-open");
    renderPresets(); loadSongs();
  }
  function closeSheet() { sheetOpen = false; document.body.classList.remove("sheet-open"); $("#nameInput").blur(); }
  $("#songBtn").addEventListener("click", openSheet);
  $("#scrim").addEventListener("click", closeSheet);
  $("#sheetDone").addEventListener("click", closeSheet);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && sheetOpen) closeSheet(); });

  // swipe the sheet down to dismiss: follows the finger, flick or half-way closes
  (function () {
    var sheet = $("#sheet"), sy = null, dy = 0, t0 = 0;
    sheet.addEventListener("pointerdown", function (e) {
      if (e.target.closest(".sheet-scroll") && $(".sheet-scroll").scrollTop > 0) return;
      if (e.target.closest("input,button")) return;
      sy = e.clientY; dy = 0; t0 = performance.now(); sheet.style.transition = "none";
    });
    window.addEventListener("pointermove", function (e) {
      if (sy == null) return;
      dy = Math.max(0, e.clientY - sy);
      sheet.style.transform = "translate(-50%," + dy + "px)";
    });
    window.addEventListener("pointerup", function () {
      if (sy == null) return;
      var v = dy / Math.max(1, performance.now() - t0);
      sheet.style.transition = ""; sheet.style.transform = "";
      if (dy > sheet.offsetHeight * 0.4 || (v > 0.5 && dy > 20)) closeSheet();
      sy = null;
    });
  })();

  function dotsFor(p) {
    return '<span class="row-dots">' + (p.voices || []).map(function (v) {
      var on = !v.mute; return '<i style="background:' + (on ? KINDS[v.kind].color : "rgba(255,255,255,.15)") + '"></i>';
    }).join("") + "</span>";
  }
  function renderPresets() {
    var ul = $("#presetList"); ul.innerHTML = "";
    H.PRESETS.forEach(function (p) {
      var li = el("li"), b = el("button", "row-main", dotsFor(p) + "<span><b>" + esc(p.name) + "</b><small>" + p.bpm + " BPM · " + H.NOTE_NAMES[p.key] + " " + H.SCALES.filter(function (s) { return s.id === p.scale; })[0].name + "</small></span>");
      b.addEventListener("click", function () { loadPatch(p, null); closeSheet(); });
      li.appendChild(b); ul.appendChild(li);
    });
  }
  function loadSongs() {
    var ul = $("#savedList");
    fetch("api/songs").then(function (r) { return r.json(); }).then(function (list) {
      ul.innerHTML = "";
      if (!list.length) { ul.innerHTML = '<li class="empty">Nothing saved yet. Name it above and tap Save.</li>'; return; }
      list.forEach(function (s) {
        var li = el("li", s.id === songId ? "row-cur" : "");
        var when = new Date(s.saved);
        var b = el("button", "row-main", "<span><b>" + esc(s.name) + "</b><small>" + when.toLocaleDateString(undefined, { day: "numeric", month: "short" }) + " · " + when.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }) + "</small></span>");
        b.addEventListener("click", function () {
          fetch("api/songs/" + s.id).then(function (r) { return r.json(); }).then(function (full) { loadPatch(full.patch, full.id); closeSheet(); });
        });
        var del = el("button", "row-del", '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M5.5 7l1 13h11l1-13M9 7V4h6v3"/></svg>');
        del.setAttribute("aria-label", "Delete " + s.name);
        del.addEventListener("click", function () {
          if (!del.classList.contains("confirm")) { del.classList.add("confirm"); del.textContent = "Delete"; setTimeout(function () { if (del.isConnected && del.classList.contains("confirm")) { del.classList.remove("confirm"); del.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M5.5 7l1 13h11l1-13M9 7V4h6v3"/></svg>'; } }, 2500); return; }
          fetch("api/songs/" + s.id, { method: "DELETE" }).then(function () { if (songId === s.id) songId = null; loadSongs(); toast("Deleted " + s.name); });
        });
        li.appendChild(b); li.appendChild(del); ul.appendChild(li);
      });
    }).catch(function () { ul.innerHTML = '<li class="empty">Can\'t reach Plint right now.</li>'; });
  }
  $("#saveBtn").addEventListener("click", function () {
    var name = $("#nameInput").value.trim() || "Untitled";
    var renamed = name !== S().name;
    S().name = name;
    var btn = $("#saveBtn"); btn.disabled = true;
    // same name as the loaded song → update it; a new name → a new song
    fetch("api/songs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: renamed ? null : songId, name: name, patch: S() }) })
      .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (s) { songId = s.id; $("#songName").textContent = name; persist(); loadSongs(); toast("Saved on Plint"); })
      .catch(function () { toast("Couldn't save. Is Plint reachable?"); })
      .then(function () { btn.disabled = false; });
  });

  function loadPatch(p, id) {
    var was = engine.playing;
    engine.song = H.normalize(p);
    if (was) engine.lastSlots = engine.song.voices.map(function () { return null; });
    if (engine.playing) engine.setTempo(engine.song.bpm);
    songId = id;
    $("#songName").textContent = engine.song.name;
    $("#bpm").textContent = engine.song.bpm;
    hitsCache = {};
    dialsEl.classList.add("animating");
    renderDeck(true);
    setTimeout(function () { dialsEl.classList.remove("animating"); }, 420);
    changed();
  }

  var toastT = 0;
  function toast(msg) {
    var t = $("#toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toastT); toastT = setTimeout(function () { t.classList.remove("show"); }, 2200);
  }

  // ── go ───────────────────────────────────────────────────
  $("#songName").textContent = S().name;
  $("#bpm").textContent = S().bpm;
  renderDeck(false);
})();
