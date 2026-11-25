/**
 * `skippedIpa` through the wasm facade, and the notice's lines: what Speak's IPA mode tells a
 * person the voice will not say (upstream issue 52: pasted `ɕiɛ˥˩ɕiɛ` came out as another word).
 */
import { describe, expect, test } from "bun:test";
import { initEngine } from "./harness.ts";

await initEngine();
const { skippedIpa } = await import("../src/wasm/engine.ts");
const { skippedLines, skippedText } = await import("../src/ui/skipped.ts");

describe("skippedIpa", () => {
  test("tone letters are unknown, in input order, each once", () => {
    const r = skippedIpa("ɕiɛ˥˩ɕiɛ˥");
    expect(r.unknown.slice(-2)).toEqual(["˥", "˩"]);
    expect(r.unknown.filter((s) => s === "˥")).toHaveLength(1);
    expect(r.noSound).toEqual([]);
    expect(r.ignoredMarks).toEqual([]);
  });

  test("a mark it cannot say is shown with its letter; the marks it reads are not listed", () => {
    expect(skippedIpa("n̥a").ignoredMarks).toEqual(["n̥"]);
    expect(skippedIpa("pʰa bɔ̃ ˈtʃʲaː, bʱa")).toEqual({ unknown: [], noSound: [], ignoredMarks: [] });
  });

  test("a symbol with no sound is reported by its own kind, and the same call says nothing for plain IPA", () => {
    const r = skippedIpa("ʏ");
    expect(r.unknown).toEqual([]);
    expect(skippedIpa("ˈhɛloʊ wɝld")).toEqual({ unknown: [], noSound: [], ignoredMarks: [] });
    expect(r.noSound.length + r.unknown.length + r.ignoredMarks.length).toBeLessThanOrEqual(1);
  });
});

describe("the Speak notice", () => {
  test("no lines when nothing is skipped, and for empty text", () => {
    expect(skippedLines("ˈhɛloʊ")).toEqual([]);
    expect(skippedLines("")).toEqual([]);
  });

  test("the phrase separators . | ? ! are structure, not unknown symbols", () => {
    expect(skippedLines("haɪ. ðɛə? goʊ! bɑ | dɑ")).toEqual([]);
  });

  test("the delimiters of /phonemic/ and [phonetic] IPA are not listed", () => {
    for (const t of ["/a˥/", "[a˥]"]) expect(skippedLines(t).map(skippedText)).toEqual(["Not said: ˥"]);
  });

  test("a symbol with no sound gets its own short line", () => {
    expect(skippedLines("ʏ").map(skippedText)).toEqual(["No sound yet: ʏ"]);
  });

  test("the syllabic and non-syllabic marks are not reported", () => {
    expect(skippedLines("aɪ̯ bʌtn̩")).toEqual([]);
  });

  test("one line per kind, in plain words", () => {
    const lines = skippedLines("a˥ n̥");
    expect(lines.map(skippedText)).toEqual(["Not said: ˥", "Mark ignored: n̥"]);
  });
});
