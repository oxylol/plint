/**
 * Turkish through the page's path: the wasm calls Speak and Sing make, and the words Speak shows
 * at the mouth.
 */
import { describe, expect, test } from "bun:test";
import { initEngine } from "./harness.ts";

await initEngine();
const { turkishPhrases, lyricsToSyllables, planPhrases } = await import("../src/wasm/engine.ts");
const { wordSpans } = await import("../src/ui/words.ts");
const male = { female: 0, child: 0 };

describe("Turkish", () => {
  test("text becomes phrases with Turkish sounds and stress", () => {
    const phrases = turkishPhrases("Merhaba, nasılsın?");
    expect(phrases.length).toBe(2);
    expect(phrases[1].segments.map((s) => s.ipa).join("")).toBe("nasɯɫsɯn");
    expect(phrases[1].segments.find((s) => s.stress === 1)?.ipa).toBe("a");
    expect(phrases[1].intonation).toBe("statement");
  });

  test("a vowel before ğ is one long vowel", () => {
    const segs = turkishPhrases("yağmur")[0].segments;
    expect(segs.map((s) => s.ipa)).toEqual(["j", "a", "m", "u", "ɾ"]);
    expect(segs[1].length).toBe(1.7);
  });

  test("each word gets its span in the plan", () => {
    const phrases = turkishPhrases("Bugün hava çok güzel.");
    const plan = planPhrases(phrases, { body: male, t0: 0 });
    const words = wordSpans(phrases, plan);
    expect(words.map((w) => w.text)).toEqual(["bugün", "hava", "çok", "güzel"]);
    for (let i = 1; i < words.length; i++) expect(words[i].t0).toBeGreaterThanOrEqual(words[i - 1].t1 - 1e-9);
  });

  test("lyrics in TR become Turkish syllables; other scripts still go to their own front ends", () => {
    const syl = lyricsToSyllables("Yeni bağlarında 빵 дом", "tr");
    expect(syl.map((s) => s.text)).toEqual(["Ye", "ni", "bağ", "la", "rın", "da", "빵", "дом"]);
    expect(syl[1].wordEnd).toBe(true);
  });

  test("lyrics are English unless TR is chosen", () => {
    // English claims only ASCII letters: ç and ü are skipped, as before
    expect(lyricsToSyllables("çok güzel").every((s) => /^[ -~]*$/.test(s.text))).toBe(true);
    expect(lyricsToSyllables("çok güzel", "en")).toEqual(lyricsToSyllables("çok güzel"));
  });
});
