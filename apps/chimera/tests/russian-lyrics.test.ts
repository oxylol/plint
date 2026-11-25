// The web app's Sing reads lyrics through the wasm engine's own front-end list, not the Rust
// `lyrics_to_syllables`: a language missing from that list is silently skipped in the browser
// while the cargo tests still pass.
import { describe, expect, test } from "bun:test";
import { initEngine } from "./harness.ts";

await initEngine();
const { lyricsToSyllables } = await import("../src/wasm/engine.ts");

describe("Russian lyrics in the web app", () => {
  test("a Cyrillic word reaches the Russian front end", () => {
    const s = lyricsToSyllables("молоко́ star");
    expect(s.map((x) => x.text)).toEqual(["мо", "ло", "ко́", "star"]);
  });
});
