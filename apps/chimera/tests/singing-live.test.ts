/**
 * Sing mode's live path end to end, without a browser: the key handling in src/ui/singing.ts
 * talks to a stand-in for the audio engine that does what the worklet does with each message
 * (schedule at clock + lead, release by tag, cancel), on the real engine (the wasm build, in its
 * own instance as in the worklet). The store's tab switches are tested here too, against the
 * same stand-in: a second file mocking the engine module would replace this one's mid-run.
 */
import { describe, expect, mock, test } from "bun:test";
import { initEngine, WasmVoice } from "./harness.ts";
import type { Gesture } from "../src/wasm/types.ts";
const MALE = { female: 0, child: 0 };

await initEngine();
const { activation } = await import("../src/wasm/engine.ts");
const midiToHz = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

const FS = 48000;
let voice = new WasmVoice(FS);
/** what the worklet answers when a scheduled plan starts (singing.ts registers one at import) */
let planned: (m: { type: "planned"; id: number; t0: number; end: number }) => void = () => {};
const fakeEngine = {
  latest: null,
  voiceClock: () => voice.clock,
  onPlanned: (f: typeof planned) => ((planned = f), () => {}),
  scheduleGestures: (gs: Gesture[], lead = 0.006) => {
    voice.schedule(gs, lead);
  },
  schedule: (plan: { gestures: Gesture[] }) => {
    voice.schedule(plan.gestures, 0.03, true);
    return 1;
  },
  release: (tag: number) => voice.release(tag),
  stopSpeech: () => voice.stop(),
};
let startAudio: () => void = () => {};
const engineStub = new Proxy(
  {
    ...fakeEngine,
    start: () => new Promise<void>((r) => (startAudio = r)),
  } as Record<string, unknown>,
  // every other engine call the store makes (volume, body, gate…) does nothing here
  { get: (t, k) => (k in t ? t[k as string] : () => {}), set: (t, k, v) => ((t[k as string] = v), true) },
);
mock.module("../src/audio/AudioEngine.ts", () => ({ engine: engineStub }));
const { useSing, singNoteOn, singNoteOff, playTune, stopSinging, tickSinging } = await import("../src/ui/singing.ts");
const gesturesOf = () => voice.score();
/** breath the score is giving now (strength-weighted activation of breath gestures) */
const breathNow = () => {
  const t = voice.clock;
  let breath = 0;
  for (const g of voice.score()) if (g.targets.breath !== undefined) breath += activation(g, t) * g.strength;
  return breath;
};

/** Render `sec` seconds; per 2.7 ms block: f0, voicing, output RMS, narrowest oral area. */
function run(sec: number) {
  const n = Math.round(sec * FS);
  const f0: number[] = [], voicing: number[] = [], rms: number[] = [], minOral: number[] = [];
  for (let i = 0; i < n; i += 128) {
    const m = Math.min(128, n - i);
    const out = voice.render(m);
    let e = 0;
    for (let j = 0; j < m; j++) e += out[j] * out[j];
    rms.push(Math.sqrt(e / m));
    const st = voice.state();
    f0.push(st.f0);
    voicing.push(st.voicing);
    let mn = Infinity;
    for (const a of st.orAreas) mn = Math.min(mn, a);
    minOral.push(mn);
  }
  // the last 40 % of the span, where a note has settled
  const late = <T,>(xs: T[]) => xs.slice(Math.floor(xs.length * 0.6));
  return { f0, voicing, rms, minOral, late };
}
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const cents = (f: number, midi: number) => 1200 * Math.log2(f / midiToHz(midi));

describe("Sing mode, live keys", () => {
  test("each key sings the next syllable at its pitch; legato, melisma and release to silence", () => {
    voice = new WasmVoice(FS);
    useSing.getState().setVibrato(0);
    useSing.getState().setLyrics("la ma -");
    expect(useSing.getState().syllables.map((s) => s.text)).toEqual(["la", "ma", "~"]);
    run(0.1);

    // first key: /la/ on A2
    singNoteOn(45, 96, MALE);
    expect(useSing.getState().cursor).toBe(1);
    const a = run(0.5);
    expect(Math.abs(cents(median(a.late(a.f0)), 45))).toBeLessThan(30);
    expect(Math.min(...a.late(a.voicing))).toBeGreaterThan(0.5);

    // second key pressed while the first is down: legato into /ma/ on C3 (the lips close once)
    singNoteOn(48, 96, MALE);
    const b = run(0.5);
    expect(Math.abs(cents(median(b.late(b.f0)), 48))).toBeLessThan(30);
    expect(Math.min(...b.minOral)).toBeLessThan(0.05); // the /m/ closure
    expect(Math.min(...b.late(b.minOral))).toBeGreaterThan(0.2); // open again on the vowel

    // releasing the first key (no longer sounding) changes nothing
    singNoteOff(45, MALE);
    const c = run(0.2);
    expect(Math.min(...c.voicing)).toBeGreaterThan(0.5);

    // melisma: the vowel carries on to E3 without a consonant
    singNoteOn(52, 96, MALE);
    expect(useSing.getState().cursor).toBe(0); // wrapped round
    const d = run(0.5);
    expect(Math.abs(cents(median(d.late(d.f0)), 52))).toBeLessThan(30);
    expect(Math.min(...d.minOral)).toBeGreaterThan(0.2);
    expect(Math.min(...d.voicing)).toBeGreaterThan(0.5);

    // releasing E3 while C3 is still down glides back to C3 on the same vowel
    singNoteOff(52, MALE);
    const e = run(0.4);
    expect(Math.abs(cents(median(e.late(e.f0)), 48))).toBeLessThan(30);
    expect(Math.min(...e.voicing)).toBeGreaterThan(0.5);

    // last key up: the note ends and the voice falls silent
    singNoteOff(48, MALE);
    const f = run(0.6);
    let peak = 0;
    for (const r of [...a.rms, ...b.rms]) peak = Math.max(peak, r);
    expect(Math.max(...f.rms.slice(Math.floor(0.3 * FS / 128)))).toBeLessThan(peak * 0.003); // < -50 dB
  });

  test("a soft key breathes less than a hard one", () => {
    const level = (velocity: number) => {
      voice = new WasmVoice(FS);
      useSing.getState().setLyrics("ah");
      run(0.05);
      singNoteOn(50, velocity, MALE);
      const r = run(0.5);
      singNoteOff(50, MALE);
      run(0.4);
      return { rms: median(r.late(r.rms)) };
    };
    const soft = level(30), hard = level(127);
    if (process.env.SING_VERBOSE) console.log(`hard vs soft key: ${(20 * Math.log10(hard.rms / soft.rms)).toFixed(1)} dB`);
    expect(hard.rms / soft.rms).toBeGreaterThan(1.6); // > 4 dB
  });

  test("a tune still waiting for audio or the dictionary is cancelled by a stop", async () => {
    voice = new WasmVoice(FS);
    let scheduled = 0;
    const schedule = engineStub.schedule as typeof fakeEngine.schedule;
    engineStub.schedule = (plan: { gestures: Gesture[] }) => (scheduled++, schedule(plan));
    let go!: () => void;
    const waiting = playTune("dreamy", MALE, () => new Promise<void>((r) => (go = r)));
    stopSinging(); // e.g. the mode changed while audio was starting
    go();
    expect(await waiting).toBe(false);
    expect(scheduled).toBe(0);
    expect(useSing.getState().playing).toBeNull();
    expect(await playTune("dreamy", MALE, () => Promise.resolve())).toBe(true);
    expect(scheduled).toBe(1);
    expect(useSing.getState().playing).toBe("dreamy");
    stopSinging();
    engineStub.schedule = schedule;
  });

  test("a tune sings the player's own words and leaves them as they are; the end is reported once", async () => {
    voice = new WasmVoice(FS);
    const sing = useSing.getState;
    sing().setLyrics("my own words to sing");
    sing().setCursor(2);
    let ended = 0;
    await playTune("joyful", MALE, () => Promise.resolve(), { onEnd: () => ended++ });
    expect(sing().playing).toBe("joyful");
    expect(sing().lyrics).toBe("my own words to sing");
    expect(sing().syllables.map((s) => s.text)).toEqual(["my", "own", "words", "to", "sing"]);
    expect(sing().cursor).toBe(2);
    // the syllable being sung, in the player's lyrics: the first, a little after the first beat
    planned({ type: "planned", id: 1, t0: voice.clock - 0.5, end: 0 });
    tickSinging();
    expect(sing().at).toBe(0);
    // sung to the end
    planned({ type: "planned", id: 1, t0: voice.clock - 1000, end: 0 });
    tickSinging();
    tickSinging();
    expect(sing().playing).toBeNull();
    expect(ended).toBe(1);
    // stopped: not reported as ended
    await playTune("joyful", MALE, () => Promise.resolve(), { onEnd: () => ended++ });
    stopSinging();
    expect(ended).toBe(1);
    // words typed during a tune stop it
    await playTune("joyful", MALE, () => Promise.resolve());
    sing().setLyrics("typed during the song");
    expect(sing().playing).toBeNull();
    expect(sing().lyrics).toBe("typed during the song");
  });

  test("a run's hooks: no start for a run stopped while starting; a stop from anywhere is reported", async () => {
    // Video's clip lives in these: it once began recording for a start that had been cancelled
    voice = new WasmVoice(FS);
    const log: string[] = [];
    const hooks = { onStart: () => log.push("start"), onEnd: () => log.push("end"), onStop: () => log.push("stop") };
    let go!: () => void;
    const waiting = playTune("joyful", MALE, () => new Promise<void>((r) => (go = r)), hooks);
    stopSinging(); // the mode or body changed while audio was starting
    go();
    expect(await waiting).toBe(false);
    expect(log).toEqual([]);
    await playTune("joyful", MALE, () => Promise.resolve(), hooks);
    expect(log).toEqual(["start"]);
    stopSinging();
    expect(log).toEqual(["start", "stop"]);
    // a start that throws (the browser can't record): nothing is sung
    let scheduled = 0;
    const schedule = engineStub.schedule as typeof fakeEngine.schedule;
    engineStub.schedule = (plan: { gestures: Gesture[] }) => (scheduled++, schedule(plan));
    const failed = await playTune("joyful", MALE, () => Promise.resolve(), { onStart: () => { throw new Error("no recorder"); } }).catch(() => "threw");
    expect(failed).toBe("threw");
    expect(scheduled).toBe(0);
    expect(useSing.getState().playing).toBeNull();
    engineStub.schedule = schedule;
  });

  test("every demo song can be chosen and sings its own words a syllable to a note; the player's words come back with a mood", async () => {
    // #87's tunes menu left out the contributed demo songs ("Ночной ветер", Daisy Bell)
    const { DEMO_SONGS, MandarinLexicon } = await import("../src/wasm/engine.ts");
    const { TUNES } = await import("../src/song/tunes.ts");
    // 送别 is in Han characters, read with the lexicon the page fetches (/data/mandarin.txt.gz)
    await new MandarinLexicon().load(new URL("../public/data/mandarin.txt.gz", import.meta.url).href);
    const melodies = TUNES.map((t) => JSON.stringify(t.phrases.flat()));
    for (const d of DEMO_SONGS) expect(melodies).toContain(JSON.stringify(d.notes.map(([m, b]) => [m, b])));
    const songs = TUNES.filter((t) => t.lyrics !== undefined);
    expect(songs.map((t) => t.name)).toContain("Ночной ветер");
    voice = new WasmVoice(FS);
    const sing = useSing.getState;
    let sent: { gestures: Gesture[] } | null = null;
    const schedule = engineStub.schedule as typeof fakeEngine.schedule;
    engineStub.schedule = (plan: { gestures: Gesture[] }) => ((sent = plan), schedule(plan));
    sing().setLyrics("my own words");
    for (const song of songs) {
      sing().setTune(song.id);
      expect(sing().lyrics).toBe(song.lyrics!);
      expect(await playTune(song.id, MALE, () => Promise.resolve())).toBe(true);
      expect(sent).not.toBeNull();
      // a syllable to each note: the n-th note is sung while the n-th syllable is lit
      const beat = 60 / song.bpm;
      const n = Math.min(5, song.phrases[0].length - 1);
      const start = song.phrases[0].slice(0, n).reduce((t, [, b]) => t + b * beat, 0.35);
      planned({ type: "planned", id: 1, t0: voice.clock - start - 0.01, end: 0 });
      tickSinging();
      expect(sing().at).toBe(n % sing().syllables.length);
      stopSinging();
    }
    sing().setTune("joyful");
    expect(sing().lyrics).toBe("my own words");
    // edited words are fitted to the song like any tune's
    sing().setTune(songs[0].id);
    sing().setLyrics("new words");
    expect(await playTune(songs[0].id, MALE, () => Promise.resolve())).toBe(true);
    stopSinging();
    engineStub.schedule = schedule;
    sing().setTune("dreamy");
    sing().setLyrics("Hello there");
  });

  test("Turkish lyrics: TR reads them and a tune sings them; a demo song is read as English, and the player's language comes back with a mood", async () => {
    const { TUNES } = await import("../src/song/tunes.ts");
    const { DEFAULT_LYRICS } = await import("../src/ui/singing.ts");
    voice = new WasmVoice(FS);
    const sing = useSing.getState;
    sing().setTune("joyful");
    sing().setLyrics(DEFAULT_LYRICS);
    // the opening words give way to Turkish's, and Turkish letters are read
    sing().setLang("tr");
    expect(sing().lyrics).toBe("Merhaba, ben havadan yapılmış bir sesim.");
    expect(sing().syllables.map((s) => s.text).slice(0, 5)).toEqual(["Mer", "ha", "ba", "ben", "ha"]);
    sing().setLyrics("çok güzel");
    expect(sing().syllables.map((s) => s.text)).toEqual(["çok", "gü", "zel"]);
    expect(await playTune("joyful", MALE, () => Promise.resolve())).toBe(true);
    expect(sing().syllables.map((s) => s.text)).toEqual(["çok", "gü", "zel"]);
    stopSinging();
    // a demo song in English brings its words, read as English
    const song = TUNES.find((t) => t.lyrics !== undefined && /^[ -~\s]*$/.test(t.lyrics))!;
    sing().setTune(song.id);
    expect(sing().lang).toBe("en");
    expect(sing().lyrics).toBe(song.lyrics!);
    sing().setTune("joyful");
    expect(sing().lang).toBe("tr");
    expect(sing().lyrics).toBe("çok güzel");
    sing().setLang("en");
    sing().setTune("dreamy");
    sing().setLyrics("Hello there");
  });

  test("empty lyrics sing the tune on la, highlighting nothing", async () => {
    voice = new WasmVoice(FS);
    const sing = useSing.getState;
    sing().setLyrics("");
    expect(await playTune("somber", MALE, () => Promise.resolve())).toBe(true);
    planned({ type: "planned", id: 1, t0: voice.clock - 0.5, end: 0 });
    tickSinging();
    expect(sing().at).toBe(-1);
    stopSinging();
    sing().setLyrics("Hello there");
  });

  test("a quick next note drops the last one's breath-out; leaving Sing cancels a pending ending", () => {
    voice = new WasmVoice(FS);
    useSing.getState().setVibrato(0);
    useSing.getState().setLyrics("cat dog");
    run(0.05);
    singNoteOn(50, 96, MALE);
    run(0.4);
    singNoteOff(50, MALE); // "cat": its /t/ and the breath-out are scheduled
    expect(gesturesOf().some((g) => g.label === "exhale")).toBe(true);
    run(0.02);
    singNoteOn(52, 96, MALE); // "dog" straight after
    expect(gesturesOf().some((g) => g.label === "exhale")).toBe(false);
    const d = run(0.4);
    expect(Math.min(...d.late(d.voicing))).toBeGreaterThan(0.5);
    singNoteOff(52, MALE); // "dog": its /g/ ending is scheduled...
    run(0.01);
    stopSinging(); // ...and the mode changes at once
    run(0.06);
    // the cancelled ending's breath is gone (the note's own breath is at the end of its 70 ms fade)
    expect(breathNow()).toBeLessThan(0.01);
  });

  test("overlapping endings all stay owned: a stop cancels every one", () => {
    voice = new WasmVoice(FS);
    useSing.getState().setLyrics("cats dogs");
    run(0.05);
    singNoteOn(50, 96, MALE);
    run(0.3);
    singNoteOff(50, MALE); // "cats": /ts/ and breath-out
    run(0.01);
    singNoteOn(52, 96, MALE);
    run(0.03);
    singNoteOff(52, MALE); // "dogs" released quickly, while the /ts/ is still going
    run(0.01);
    stopSinging();
    run(0.08);
    expect(breathNow()).toBeLessThan(0.01);
  });
});

describe("keys pressed before the audio has started", () => {
  test("only the press still down sings, once, and not after the body changed", async () => {
    const { useStore } = await import("../src/ui/store.ts");
    voice = new WasmVoice(FS);
    useStore.setState({ started: false, mode: "play" });
    useStore.getState().setMode("sing");
    useSing.getState().setLyrics("one two three");
    let scheduled = 0;
    const sg = engineStub.scheduleGestures as typeof fakeEngine.scheduleGestures;
    engineStub.scheduleGestures = (gs: Gesture[], lead?: number) => (scheduled++, sg(gs, lead));
    const st = useStore.getState();
    // press, release and press again while audio starts: one note, one syllable
    st.noteOn(50);
    st.noteOff(50);
    st.noteOn(50);
    startAudio();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    expect(useStore.getState().started).toBe(true);
    expect(scheduled).toBe(1);
    expect(useSing.getState().cursor).toBe(1);
    useStore.getState().noteOff(50);

    // a press waiting for audio is dropped by a body change
    useStore.setState({ started: false });
    scheduled = 0;
    const before = useSing.getState().cursor;
    useStore.getState().noteOn(52);
    useStore.getState().setBody({ female: 1, child: 0 });
    startAudio();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
    expect(scheduled).toBe(0);
    expect(useSing.getState().cursor).toBe(before);
    engineStub.scheduleGestures = sg;
  });
});

describe("settings that belong to one tab stay in it", () => {
  const sent: { source: Record<string, unknown>[]; artic: Record<string, unknown>[] } = { source: [], artic: [] };
  const recording = async () => {
    const { useStore } = await import("../src/ui/store.ts");
    const { beatReady } = await import("../src/wasm/engine.ts");
    engineStub.setSource = (v: Record<string, unknown>) => void sent.source.push(v);
    engineStub.setArtic = (v: Record<string, unknown>) => void sent.artic.push(v);
    useStore.setState({ mode: "play", started: true, body: MALE });
    useStore.getState().setSource({ voiced: true });
    sent.source.length = sent.artic.length = 0;
    return { st: useStore.getState, beatReady };
  };
  const done = () => {
    delete engineStub.setSource;
    delete engineStub.setArtic;
  };

  test("Play's Whisper doesn't make Speak or Sing voiceless, and Play whispers again on return", async () => {
    const { st } = await recording();
    st().setSource({ voiced: false });
    st().setMode("speak");
    expect(st().source.voiced).toBe(true);
    expect(sent.source.at(-1)).toEqual({ voiced: true });
    st().setMode("sing");
    expect(st().source.voiced).toBe(true);
    st().setMode("play");
    expect(st().source.voiced).toBe(false);
    expect(sent.source.at(-1)).toEqual({ voiced: false });
    done();
  });

  test("Whisper is kept through Bestiary and Beatbox too, and left alone when off", async () => {
    const { st } = await recording();
    st().setSource({ voiced: false });
    st().setMode("bestiary");
    st().setMode("beatbox");
    expect(st().source.voiced).toBe(true);
    st().setMode("play");
    expect(st().source.voiced).toBe(false);
    st().setSource({ voiced: true });
    sent.source.length = 0;
    st().setMode("speak");
    st().setMode("play");
    expect(st().source.voiced).toBe(true);
    expect(sent.source.filter((v) => "voiced" in v)).toEqual([]);
    done();
  });

  test("leaving Beatbox puts back the pose from before it", async () => {
    const { st, beatReady } = await recording();
    st().setArtic({ jaw: 0.9, lipAp: 1.2, lipProt: 0.3 }, false);
    const before = { ...st().artic };
    st().setMode("beatbox");
    const ready = beatReady(MALE);
    expect(st().artic.jaw).toBeCloseTo(ready.jaw!, 6);
    expect(st().artic.lipAp).toBeCloseTo(ready.lipAp!, 6);
    st().setMode("play");
    expect(st().artic).toEqual(before);
    expect(st().tipFree).toBe(false);
    expect(sent.artic.at(-1)).toEqual(before);
    done();
  });

  test("a pose dragged inside Beatbox doesn't follow the player out either", async () => {
    const { st } = await recording();
    st().setArtic({ jaw: 0.7, lipAp: 0.8 }, true);
    const before = { ...st().artic };
    st().setMode("beatbox");
    st().setArtic({ jaw: 0.2, tbY: 2 });
    st().setMode("speak");
    expect(st().artic).toEqual(before);
    done();
  });
});

describe("a main button pressed before the audio is up", () => {
  const store = async () => (await import("../src/ui/store.ts")).useStore;
  const failing = () => Promise.reject(new Error("no audio device"));

  test("Breathe breathes once the audio starts, but not in a tab switched to meanwhile", async () => {
    const useStore = await store();
    const st = useStore.getState;
    useStore.setState({ started: false, mode: "play", hold: false });
    let pending = st().toggleBreath();
    startAudio();
    await pending;
    expect(st().hold).toBe(true);
    st().setHold(false);
    // a tab switched to meanwhile (Sing breathes by itself: a held breath would drone), and away
    // and back again: the press belonged to the Play the player left
    for (const away of [["sing"], ["sing", "play"]] as const) {
      useStore.setState({ started: false, mode: "play", hold: false });
      pending = st().toggleBreath();
      for (const m of away) st().setMode(m);
      startAudio();
      await pending;
      expect(st().hold).toBe(false);
    }
  });

  test("a second press of Breathe while the audio starts turns it off again", async () => {
    const useStore = await store();
    const st = useStore.getState;
    useStore.setState({ started: false, mode: "bestiary", hold: false });
    const first = st().toggleBreath();
    const second = st().toggleBreath();
    startAudio();
    await Promise.all([first, second]);
    expect(st().hold).toBe(false);
    useStore.setState({ mode: "play" });
  });

  test("a failed start leaves Breathe off and the beat stopped, so the buttons offer a retry", async () => {
    const useStore = await store();
    const start = engineStub.start;
    engineStub.start = failing;
    useStore.setState({ started: false, mode: "play", hold: false });
    await useStore.getState().toggleBreath();
    expect(useStore.getState().hold).toBe(false);
    expect(useStore.getState().startError).toBe("no audio device");
    useStore.getState().setMode("beatbox");
    useStore.getState().setBeatPlaying(true);
    await new Promise((r) => setTimeout(r, 0));
    expect(useStore.getState().beat.playing).toBe(false);
    engineStub.start = start;
    useStore.setState({ startError: null });
  });
});
