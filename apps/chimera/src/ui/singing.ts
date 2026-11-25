/**
 * Sing mode on the main thread. The lyrics become syllables with a cursor; each key sings the
 * next syllable at its pitch. A key pressed while the previous one is still down sings legato
 * (the previous syllable's coda runs straight into the next onset); a "-" in the lyrics holds
 * the vowel onto a new pitch (melisma). Releasing the sounding key while another is still down
 * glides back to that key on the same vowel. Key velocity sets the breath.
 * Sing sings the same words to a tune (src/song): fitted to its phrases, planned whole (planSong)
 * and sent as one score. The words, their language, the tune, tempo and key are kept on the device.
 */
import { create } from "zustand";
import { engine } from "../audio/AudioEngine.ts";
import { inPlugin, setParam } from "../audio/pluginBridge.ts";
import { g2p, zh } from "./speech.ts";
import { hasHan, lyricsToSyllables, type Lang, type Syllable } from "../wasm/engine.ts";
import { planSong, planNoteOn, planNoteOff, planMelisma } from "../wasm/engine.ts";
import { EXHALE_LABEL } from "../wasm/engine.ts";
import type { Gesture } from "../wasm/types.ts";
import { nearestBody, type BodyMix } from "./body.ts";
import { TUNES } from "../song/tunes.ts";
import { fitSong, la } from "../song/fit.ts";

/** A note on the piano roll, on the voice clock (s); t1 is null while it is held. */
export interface RollNote {
  midi: number;
  t0: number;
  t1: number | null;
  text: string;
  demo: boolean;
}

/** What the piano roll draws, read every frame (mutated in place: no re-render per note). */
export const roll: { notes: RollNote[] } = { notes: [] };

export const DEFAULT_LYRICS = "Hello there, I am a voice made of air.";
/** the opening words in each language (choosing the other language swaps one for the other) */
const DEFAULT_LYRICS_OF: Record<Lang, string> = { en: DEFAULT_LYRICS, tr: "Merhaba, ben havadan yapılmış bir sesim." };

/** sung when the lyrics are empty */
const AH: Syllable = { text: "ah", onset: [], nucleus: { ipa: "ɑ", stress: 1 }, coda: [], wordEnd: true, phraseEnd: false };

interface SingState {
  lyrics: string;
  /** the language Latin-script words are read in: the lyrics here, and Speak's text (one setting
   *  for both, kept with the words; other scripts are read by their own) */
  lang: Lang;
  syllables: Syllable[];
  /** the syllable the next key sings */
  cursor: number;
  /** vibrato depth on held vowels (cents) */
  vibrato: number;
  /** the tune Sing sings the words to (a TUNES id) */
  tune: string;
  /** the tune's tempo, as a multiple of its own */
  tempo: number;
  /** semitones up or down from the tune's own key (after the body's octave) */
  key: number;
  /** the tune being sung, if any */
  playing: string | null;
  /** index in `syllables` of the one the tune is singing (−1: none, or the tune sings "la") */
  at: number;
  setLyrics: (text: string) => void;
  setLang: (lang: Lang) => void;
  setCursor: (i: number) => void;
  setVibrato: (cents: number) => void;
  setTune: (id: string) => void;
  setTempo: (x: number) => void;
  setKey: (semitones: number) => void;
}

const KEPT = "chimera-sing";

/**
 * the last session's words and their language, tune, tempo and key (anything missing or odd: the
 * defaults). Not in the plugin: the project keeps its words, and every instance shares this storage.
 */
function kept(): { lyrics: string; lang: Lang; tune: string; tempo: number; key: number } {
  const d = { lyrics: DEFAULT_LYRICS, lang: "en" as Lang, tune: TUNES[0].id, tempo: 1, key: 0 };
  if (inPlugin) return d;
  try {
    const k = JSON.parse(localStorage.getItem(KEPT) ?? "null") as Record<string, unknown> | null;
    if (!k) return d;
    return {
      lyrics: typeof k.lyrics === "string" ? k.lyrics : d.lyrics,
      lang: k.lang === "tr" ? "tr" : d.lang,
      tune: TUNES.some((t) => t.id === k.tune) ? (k.tune as string) : d.tune,
      tempo: typeof k.tempo === "number" && k.tempo >= 0.5 && k.tempo <= 1.5 ? k.tempo : d.tempo,
      key: typeof k.key === "number" && Math.abs(k.key) <= 12 ? Math.round(k.key) : d.key,
    };
  } catch {
    return d; // no storage here (a private window): start fresh
  }
}

const atLoad = kept();
/** the player's words and their language while a demo song's are in the lyrics */
let ownWords: { lyrics: string; lang: Lang } | null = null;

export const useSing = create<SingState>((set, get) => ({
  lyrics: atLoad.lyrics,
  lang: atLoad.lang,
  syllables: lyricsToSyllables(atLoad.lyrics, atLoad.lang),
  cursor: 0,
  vibrato: 35,
  tune: atLoad.tune,
  tempo: atLoad.tempo,
  key: atLoad.key,
  playing: null,
  at: -1,
  setLyrics: (text) => {
    if (inPlugin) {
      // Rust reads the lyrics: the syllables and the cursor come back with the next polls
      set({ lyrics: text });
      engine.pluginLyrics(text);
      return;
    }
    stopDemo();
    set({ lyrics: text, syllables: lyricsToSyllables(text, get().lang), cursor: 0 });
    if (hasHan(text) && !zh.loaded) ensureDictionary();
  },
  setLang: (lang) => {
    const s = get();
    if (lang === s.lang) return;
    stopDemo();
    // the other language's opening words give way to this one's
    const lyrics = s.lyrics === DEFAULT_LYRICS_OF[s.lang] ? DEFAULT_LYRICS_OF[lang] : s.lyrics;
    set({ lang, lyrics, syllables: lyricsToSyllables(lyrics, lang), cursor: 0 });
  },
  setCursor: (i) => {
    const index = Math.max(0, Math.min(i, Math.max(0, get().syllables.length - 1)));
    if (inPlugin) engine.pluginCursor(index);
    else set({ cursor: index });
  },
  setVibrato: (cents) => {
    set({ vibrato: cents });
    if (inPlugin) setParam("vibrato", cents);
  },
  setTune: (id) => {
    const words = TUNES.find((t) => t.id === id)?.lyrics;
    const { lyrics, lang } = get();
    const asWritten = TUNES.some((t) => t.lyrics === lyrics);
    if (words !== undefined) {
      // a demo song brings its words, in English or a script of its own; the player's own words
      // and language come back with a mood
      if (!asWritten) ownWords = { lyrics, lang };
      if (lyrics !== words || lang !== "en") {
        set({ lang: "en" });
        get().setLyrics(words);
      }
    } else if (asWritten && ownWords !== null) {
      set({ lang: ownWords.lang });
      get().setLyrics(ownWords.lyrics);
      ownWords = null;
    }
    set({ tune: id });
  },
  setTempo: (x) => set({ tempo: x }),
  setKey: (semitones) => set({ key: semitones }),
}));

useSing.subscribe((s, p) => {
  if (inPlugin) return;
  if (s.lyrics === p.lyrics && s.lang === p.lang && s.tune === p.tune && s.tempo === p.tempo && s.key === p.key) return;
  try {
    localStorage.setItem(KEPT, JSON.stringify({ lyrics: s.lyrics, lang: s.lang, tune: s.tune, tempo: s.tempo, key: s.key }));
  } catch {
    // not kept: the session still works
  }
});

/**
 * Load what the lyrics' front ends need (the pronouncing dictionary; the Mandarin lexicon when there
 * are Han characters), then re-read the lyrics with it (the cursor stays).
 */
export function ensureDictionary(): void {
  // in the plugin Rust reads the lyrics, with its own copy of the dictionary
  if (inPlugin) return;
  const needs: Promise<void>[] = [];
  if (!g2p.loaded) needs.push(g2p.load());
  if (!zh.loaded && hasHan(useSing.getState().lyrics)) needs.push(zh.load());
  if (!needs.length) return;
  // a failed load leaves the rest working (letter-to-sound rules for English)
  void Promise.allSettled(needs).then(() => {
    const s = useSing.getState();
    const syllables = lyricsToSyllables(s.lyrics, s.lang);
    useSing.setState({ syllables, cursor: Math.min(s.cursor, Math.max(0, syllables.length - 1)) });
  });
}

// ---- live notes

interface Sounding {
  midi: number;
  /** the syllable whose vowel is sounding (a melisma keeps its parent's, for the coda) */
  syl: Syllable;
  tag: number;
  note: RollNote;
}

let sounding: Sounding | null = null;

/** How a syllable is written at the mouth: a hyphen when its word goes on. */
const written = (s: Syllable) => s.text + (s.wordEnd ? "" : "-");

/**
 * The syllable being sung now, for the words at the mouth: the demo's note under the clock, or
 * the held key's. A melisma keeps its syllable's key, so the word stays while the pitch moves.
 */
export function sungNow(): { key: unknown; text: string } | null {
  const d = demoRun;
  if (d) {
    if (d.offset === null) return null;
    const t = now() - d.offset;
    // notes follow one another without a gap, so a melisma's syllable stays put
    const k = d.times.findIndex(([a, b]) => t >= a && t < b);
    return k < 0 ? null : { key: `${d.id}:${d.parent[k]}`, text: d.shown[k] };
  }
  return sounding ? { key: sounding, text: written(sounding.syl) } : null;
}

/** The syllable whose vowel is sounding from the keys now (null when no key sings). */
export function soundingSyllable(): string | null {
  return sounding?.syl.text ?? null;
}
/** keys down, oldest first */
let keys: number[] = [];
/** gesture tags: a note uses tag (articulation, breath) and tag + 1 (pitch, vibrato) */
let nextTag = 2;
/** endings (glide, coda) that may still be playing: tag, and tag + 1 for the breath-out */
const tails: number[] = [];
/** endings whose breath-out has not been dropped by a following note */
let exhalesLive = false;

/** breath level for a key velocity (1 at the on-screen keys' default of 96) */
const breathFor = (velocity: number) => 0.62 + 0.4 * Math.min(1, Math.max(0, velocity / 96)) + 0.12 * Math.max(0, (velocity - 96) / 31);

const now = () => engine.voiceClock() ?? 0;

function addNote(midi: number, t0: number, text: string, demo = false, t1: number | null = null): RollNote {
  const n: RollNote = { midi, t0, t1, text, demo };
  roll.notes.push(n);
  // keep a minute of history
  const cut = t0 - 60;
  if (roll.notes.length > 400 || (roll.notes[0] && (roll.notes[0].t1 ?? t0) < cut)) roll.notes = roll.notes.filter((x) => (x.t1 ?? t0) >= cut).slice(-400);
  return n;
}

/** The syllable a melisma at index i continues. */
function parentOf(syls: Syllable[], i: number): Syllable {
  for (let k = i; k >= 0; k--) if (!syls[k].melisma) return syls[k];
  return AH;
}

function startNote(syl: Syllable, midi: number, velocity: number, body: BodyMix, text: string): void {
  const t = now();
  const tag = nextTag;
  nextTag += 2;
  // earlier codas may finish, but their breath-out must not open the folds on this onset
  if (exhalesLive) for (const t of tails) engine.release(t + 1);
  exhalesLive = false;
  const from = sounding;
  if (from) {
    engine.release(from.tag);
    engine.release(from.tag + 1);
    from.note.t1 = t;
  }
  const gs: Gesture[] = planNoteOn(syl, midi, { body, vibrato: useSing.getState().vibrato, tag, legatoFrom: from ? { syl: from.syl, midi: from.midi } : undefined });
  const breath = breathFor(velocity);
  for (const g of gs) if (g.targets.breath !== undefined) g.targets = { ...g.targets, breath: g.targets.breath * breath };
  engine.scheduleGestures(gs);
  sounding = { midi, syl, tag, note: addNote(midi, t, text) };
}

/** The held vowel moves to a new pitch (melisma, or back to a key still down). */
function glide(midi: number, text: string): void {
  const s = sounding!;
  const t = now();
  engine.release(s.tag + 1);
  engine.scheduleGestures(planMelisma(midi, { vibrato: useSing.getState().vibrato, tag: s.tag }));
  s.note.t1 = t;
  s.midi = midi;
  s.note = addNote(midi, t, text);
}

/** the note and press count last seen from the plugin */
let pluginNote = -1;
let pluginSerial = -1;

/**
 * In the plugin Rust plays the notes, so the words at the mouth and the piano roll follow what it
 * reports. A new press is a new syllable, unless it's a melisma on a vowel still sounding; a new
 * pitch with no press is the glide back to a key still down.
 */
export function followPluginNote(note: number, sung: number, serial: number, clock: number): void {
  const pressed = serial !== pluginSerial;
  if (!pressed && note === pluginNote) return;
  pluginSerial = serial;
  pluginNote = note;
  const s = sounding;
  if (s) s.note.t1 = clock;
  if (note < 0) {
    sounding = null;
    return;
  }
  const syls = useSing.getState().syllables;
  const syl = syls[sung] ?? AH;
  if (s && (!pressed || syl.melisma)) {
    s.midi = note;
    s.note = addNote(note, clock, pressed ? syl.text : "~");
  } else {
    sounding = { midi: note, syl: syl.melisma ? parentOf(syls, sung) : syl, tag: 0, note: addNote(note, clock, syl.text) };
  }
}

/** The voice clock started again (a new voice or a restore): the roll's times no longer mean anything. */
export function clearPluginNotes(): void {
  roll.notes = [];
  sounding = null;
  pluginNote = -1;
}

export function singNoteOn(midi: number, velocity: number, body: BodyMix): void {
  stopDemo();
  keys = keys.filter((k) => k !== midi).concat(midi);
  const st = useSing.getState();
  const syls = st.syllables.length ? st.syllables : [AH];
  const i = st.cursor % syls.length;
  const syl = syls[i];
  if (syl.melisma && sounding) glide(midi, syl.text);
  // a melisma with nothing held: sing its vowel again, without the onset
  else startNote(syl.melisma ? { ...parentOf(syls, i), onset: [] } : syl, midi, velocity, body, syl.text);
  useSing.setState({ cursor: (i + 1) % syls.length });
}

export function singNoteOff(midi: number, body: BodyMix): void {
  keys = keys.filter((k) => k !== midi);
  const s = sounding;
  if (!s || s.midi !== midi) return;
  if (keys.length) {
    glide(keys[keys.length - 1], "~");
    return;
  }
  engine.release(s.tag);
  engine.release(s.tag + 1);
  s.note.t1 = now();
  sounding = null;
  // a melisma comes next: this vowel carries on there, and the coda waits for the chain's end
  const st = useSing.getState();
  if (st.syllables[st.cursor]?.melisma) return;
  const gs = planNoteOff(s.syl, midi, body);
  if (!gs.length) return;
  const tag = nextTag;
  nextTag += 2;
  engine.scheduleGestures(gs.map((g) => ({ ...g, tag: g.label === EXHALE_LABEL ? tag + 1 : tag })));
  tails.push(tag);
  if (tails.length > 8) tails.shift(); // long finished by now
  exhalesLive = true;
}

// ---- tunes

interface TuneRun {
  id: number;
  /** note start/end times in plan time (s) */
  times: [number, number][];
  midi: number[];
  text: string[];
  /** what the mouth shows per note, and the note whose syllable it is (a melisma's parent) */
  shown: string[];
  parent: number[];
  /** each note's syllable in the lyrics (−1: the tune sings "la") */
  from: number[];
  end: number;
  /** voice clock of plan time 0, once the worklet has it */
  offset: number | null;
  hooks: TuneHooks;
}

/** What a caller (Record video) does as a tune's run begins, ends or is stopped. */
export interface TuneHooks {
  /** just before it's sent to the voice, after every wait (a throw: it isn't sung) */
  onStart?: () => void;
  /** sung to the end */
  onEnd?: () => void;
  /** stopped before the end, from anywhere (Stop, another tune, the mode or body changing) */
  onStop?: () => void;
}

let demoRun: TuneRun | null = null;
/** bumped by every stop: a tune still waiting for audio or the dictionary then gives up */
let demoToken = 0;

function endDemo(): void {
  demoRun = null;
  useSing.setState({ playing: null, at: -1 });
}

engine.onPlanned((m) => {
  const d = demoRun;
  if (!d || m.id !== d.id) return;
  d.offset = m.t0;
  d.times.forEach(([a, b], k) => addNote(d.midi[k], m.t0 + a, d.text[k], true, m.t0 + b - 0.02));
});

/**
 * Sing the lyrics to tune `id` (after `ready`, e.g. audio startup, resolves; unless stopped
 * meanwhile), at the chosen tempo and key; empty lyrics sing it on "la". Resolves whether it is
 * singing. `hooks` follow the run: onStart only once nothing can cancel it before it's sent.
 */
export async function playTune(id: string, body: BodyMix, ready?: () => Promise<void>, hooks: TuneHooks = {}): Promise<boolean> {
  const tune = TUNES.find((t) => t.id === id);
  if (!tune) return false;
  stopSinging();
  const token = demoToken;
  await ready?.();
  // Turkish reads by rule: no English dictionary to wait for
  if (useSing.getState().lang === "en")
    try {
      await g2p.load();
    } catch {
      // letter-to-sound rules
    }
  if (hasHan(useSing.getState().lyrics)) await zh.load().catch(() => {});
  if (token !== demoToken) return false;
  const st = useSing.getState();
  // read again: the dictionary may have arrived since the words were typed
  const own = lyricsToSyllables(st.lyrics, st.lang);
  const words = own.length ? own : lyricsToSyllables(la(tune), st.lang);
  const shift = tune.octave[nearestBody(body)] * 12 + st.key;
  // a demo song's own words, unedited: a syllable to each note as written (as `chimera render song`)
  const { notes, from } =
    tune.lyrics !== undefined && st.lyrics === tune.lyrics && own.length
      ? {
          notes: tune.phrases[0].map(([midi, beats], i) => ({ midi: midi + shift, beats, syl: own[i % own.length] })),
          from: tune.phrases[0].map((_, i) => i % own.length),
        }
      : fitSong(tune, words, shift);
  if (!notes.length) return false;
  const bpm = tune.bpm * st.tempo;
  const t0 = 0.35;
  const plan = planSong(notes, { body, bpm, t0, vibrato: st.vibrato });
  hooks.onStart?.();
  const beat = 60 / bpm;
  let t = t0;
  const times = notes.map((n): [number, number] => {
    const a = t;
    t += n.beats * beat;
    return [a, t];
  });
  const planId = engine.schedule({ gestures: plan.gestures, segments: plan.segments, end: plan.end });
  const parent: number[] = [];
  notes.forEach((n, i) => parent.push(n.syl.melisma && i > 0 ? parent[i - 1] : i));
  const shown = parent.map((p) => written(notes[p].syl));
  demoRun = { id: planId, times, midi: notes.map((n) => n.midi), text: notes.map((n) => n.syl.text), shown, parent, from: own.length ? from : from.map(() => -1), end: plan.end, offset: null, hooks };
  useSing.setState({ syllables: own, cursor: Math.min(st.cursor, Math.max(0, own.length - 1)), playing: tune.id, at: -1 });
  return true;
}

function stopDemo(): void {
  demoToken++;
  const d = demoRun;
  if (!d) return;
  engine.stopSpeech();
  // notes not yet sung leave the roll; the one sounding ends now
  const t = now();
  roll.notes = roll.notes.filter((n) => !(n.demo && n.t0 > t));
  for (const n of roll.notes) if (n.demo && n.t1 !== null && n.t1 > t) n.t1 = t;
  endDemo();
  d.hooks.onStop?.();
}

/** Call every frame while Sing mode is shown: follow the tune's progress. */
export function tickSinging(): void {
  const d = demoRun;
  if (!d || d.offset === null) return;
  const t = now() - d.offset;
  if (t > d.end + 0.4) {
    endDemo();
    d.hooks.onEnd?.();
    return;
  }
  let k = -1;
  while (k + 1 < d.times.length && d.times[k + 1][0] <= t) k++;
  const at = k < 0 ? -1 : d.from[k];
  if (at !== useSing.getState().at) useSing.setState({ at });
}

/** Stop everything sung (leaving Sing mode, changing body): demo, notes and endings. */
export function stopSinging(): void {
  stopDemo();
  if (sounding) {
    engine.release(sounding.tag);
    engine.release(sounding.tag + 1);
    sounding.note.t1 = now();
    sounding = null;
  }
  for (const t of tails) {
    engine.release(t);
    engine.release(t + 1);
  }
  tails.length = 0;
  exhalesLive = false;
  keys = [];
}

export { TUNES };
