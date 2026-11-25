// The web app's Sing reads lyrics through the wasm engine's own front-end list, not the Rust
// `lyrics_to_syllables`: kana missing from that list would be silently skipped in the browser
// while the cargo tests still pass (the sounds themselves are tested in crates/chimera/tests/japanese.rs).
import { describe, expect, test } from "bun:test";
import { initEngine } from "./harness.ts";

await initEngine();
const { lyricsToSyllables, DEMO_SONGS } = await import("../src/wasm/engine.ts");

describe("Japanese lyrics in the web app", () => {
  test("kana are sung one note per mora, beside Latin and Hangul words", () => {
    const s = lyricsToSyllables("さくら、がっこう 빵야 hello");
    expect(s.slice(0, 7).map((x) => x.text)).toEqual(["さ", "く", "ら", "が", "っ", "こ", "う"]);
    expect(s[2]!.phraseEnd).toBe(true);
    expect(s[3]!.coda).toEqual([]);
    expect(s[4]!.melisma).toBe(true); // っ is a note: the vowel goes on into it
    expect(s[5]!.onset[0]!.dur).toBeGreaterThan(0.3); // and the k after it is held
    expect(s[6]!.melisma).toBe(true); // こう sings oo
    expect(s.slice(7, 9).map((x) => x.text)).toEqual(["빵", "야"]);
    expect(s.length).toBeGreaterThan(9);
  });

  test("kana read the same whichever language Latin script is in (EN/TR)", () => {
    const text = "さくら、がっこう かんぱい";
    expect(lyricsToSyllables(text, "tr")).toEqual(lyricsToSyllables(text, "en"));
  });

  test("ん is a note of its own, a hum on the nasal of what follows", () => {
    const s = lyricsToSyllables("かんぱい");
    expect(s.map((x) => x.text)).toEqual(["か", "ん", "ぱ", "い"]);
    expect(s[1]!.nucleus.ipa).toBe("m");
    expect(s[1]!.onset).toEqual([]);
    expect(s[0]!.coda[0]!.ipa).toBe("m");
    expect(s[0]!.closing).toBeGreaterThan(0);
  });

  test("the demo song reads to one syllable per note", () => {
    const song = DEMO_SONGS.find((d) => d.name === "さくら さくら")!;
    expect(lyricsToSyllables(song.lyrics).length).toBe(song.notes.length);
  });
});
