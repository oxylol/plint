/**
 * The Song page's tunes and fitting anyone's words to them: every syllable sung once and in order,
 * each phrase keeping its length in beats, and lines cut where words end.
 */
import { describe, expect, test } from "bun:test";
import { initEngine } from "./harness.ts";

await initEngine();
const { lyricsToSyllables } = await import("../src/wasm/engine.ts");
const { TUNES } = await import("../src/song/tunes.ts");
const { fitLine, fitSong, lines, phraseSize, la } = await import("../src/song/fit.ts");
type SungNote = import("../src/wasm/types.ts").SungNote;
type Syllable = import("../src/wasm/types.ts").Syllable;

const beats = (ns: { beats: number }[]) => ns.reduce((s, n) => s + n.beats, 0);
/** the syllables a fitted line sings, in order (a held vowel isn't a new one) */
const sung = (ns: SungNote[]) => ns.filter((n) => !n.syl.melisma).map((n) => n.syl.text);
const texts = (s: Syllable[]) => s.map((x) => x.text);

describe("tunes", () => {
  test("every phrase of a tune is as long as the others, and ends on a held note (room for the breath)", () => {
    for (const t of TUNES) {
      const lens = t.phrases.map((p) => beats(p.map(([, b]) => ({ beats: b }))));
      expect(new Set(lens).size).toBe(1);
      for (const p of t.phrases) expect(p[p.length - 1][1]).toBeGreaterThanOrEqual(1.5);
    }
  });
  test("ids are unique", () => {
    expect(new Set(TUNES.map((t) => t.id)).size).toBe(TUNES.length);
  });
});

describe("fitting a line to a phrase", () => {
  const phrase = TUNES.find((t) => t.id === "joyful")!.phrases[0]; // 7 notes, 8 beats
  const check = (words: string) => {
    const line = lyricsToSyllables(words);
    const { notes: ns, from } = fitLine(line, phrase);
    // each note knows the syllable it sings (a held vowel: the one it holds)
    expect(from.map((i) => line[i].nucleus.ipa)).toEqual(ns.map((n) => n.syl.nucleus.ipa));
    expect(beats(ns)).toBeCloseTo(8, 9);
    expect(sung(ns)).toEqual(texts(line.filter((s) => !s.melisma)));
    expect(ns.map((n) => n.syl.phraseEnd)).toEqual(ns.map((_, i) => i === ns.length - 1));
    return ns;
  };
  test("as many syllables as notes: one each, the tune unchanged", () => {
    const ns = check("one two three four five six sev");
    expect(ns.map((n) => [n.midi, n.beats])).toEqual(phrase);
  });
  test("more syllables: long notes split, same pitches", () => {
    const ns = check("this line has a lot more syllables than notes");
    expect(ns.length).toBe(lyricsToSyllables("this line has a lot more syllables than notes").length);
    expect(new Set(ns.map((n) => n.midi))).toEqual(new Set(phrase.map(([m]) => m)));
  });
  test("fewer syllables: vowels held across notes, the spare notes on the last syllables", () => {
    const ns = check("hello my friend");
    expect(ns.length).toBe(phrase.length);
    expect(ns[ns.length - 1].syl.melisma).toBe(true);
    expect(ns[0].syl.melisma).toBeFalsy();
  });
  test("a single word: neighbouring notes merge first, the last note keeps its pitch", () => {
    const ns = check("cat");
    expect(ns.length).toBeLessThanOrEqual(2);
    expect(ns[ns.length - 1].midi).toBe(phrase[phrase.length - 1][0]);
  });
});

describe("lyrics into lines", () => {
  test("a long line is cut where a word ends, into lines near the tune's phrase", () => {
    const syls = lyricsToSyllables("the quick brown fox jumps over the lazy dog and runs away into the dark green wood");
    const ls = lines(syls, 7);
    expect(ls.length).toBeGreaterThan(1);
    for (const l of ls) {
      expect(syls[l[l.length - 1]].wordEnd).toBe(true);
      expect(l.length).toBeLessThanOrEqual(11);
    }
    expect(ls.flat()).toEqual(syls.map((_, i) => i));
  });
  test("a word or two on its own joins the next line", () => {
    const ls = lines(lyricsToSyllables("Oh,\nI will sing this line for you"), 7);
    expect(ls.length).toBe(1);
  });
  test("only each line's last note ends a phrase", () => {
    const syls = lyricsToSyllables("one, two three four five six seven.\neight nine ten eleven twelve");
    const { notes } = fitSong(TUNES[1], syls);
    const ends = notes.filter((n) => n.syl.phraseEnd).length;
    expect(ends).toBe(lines(syls, phraseSize(TUNES[1])).length);
    expect(notes[notes.length - 1].syl.phraseEnd).toBe(true);
  });
});

describe("a whole song", () => {
  test("every tune sings its own la-la exactly, note for note", () => {
    for (const t of TUNES) {
      const { notes: ns, from } = fitSong(t, lyricsToSyllables(la(t)));
      expect(from).toEqual(ns.map((_, i) => i));
      expect(ns.map((n) => [n.midi, n.beats])).toEqual(t.phrases.flat());
      expect(ns.every((n) => !n.syl.melisma)).toBe(true);
    }
  });
  test("more lines than phrases go round the tune again, shifted by the body's octave", () => {
    const t = TUNES[1];
    const words = Array.from({ length: 6 }, () => "la ".repeat(phraseSize(t)).trim()).join("\n");
    const { notes: ns } = fitSong(t, lyricsToSyllables(words), -12);
    expect(ns.length).toBe(6 * phraseSize(t));
    expect(ns[4 * phraseSize(t)].midi).toBe(t.phrases[0][0][0] - 12);
  });
  test("a user's own held vowel (-) stays a held vowel", () => {
    const { notes: ns } = fitSong(TUNES[1], lyricsToSyllables("oh - my darling one two three"));
    expect(ns[1].syl.melisma).toBe(true);
    expect(ns[1].syl.nucleus.ipa).toBe(ns[0].syl.nucleus.ipa);
  });
});
