/**
 * Anyone's words onto a tune: the lyrics' lines go onto the tune's phrases in turn (round the
 * tune again if there are more lines), and each line is fitted to its phrase's notes without
 * changing the phrase's length in beats:
 *  - more syllables than notes: the longest note splits in two (same pitch) until there are enough
 *  - fewer: each syllable takes a run of notes, its vowel held from one pitch to the next (a
 *    melisma, as "-" does in Sing), the extra notes going to the line's last syllables; with less
 *    than half a note's worth of syllables, the shortest neighbouring notes merge first
 * A line too long for one phrase splits at a word end; a line of a word or two joins the next.
 */
import type { Syllable, SungNote } from "../wasm/types.ts";
import type { Tune } from "./tunes.ts";

/** the tune's typical phrase, in notes: the line length the lyrics are cut to */
export function phraseSize(tune: Tune): number {
  return Math.max(1, Math.round(tune.phrases.reduce((n, p) => n + p.length, 0) / tune.phrases.length));
}

/** The lyrics' syllables in lines of about `size` (as indices into `syls`). */
export function lines(syls: Syllable[], size: number): number[][] {
  const raw: number[][] = [];
  let cur: number[] = [];
  for (let i = 0; i < syls.length; i++) {
    cur.push(i);
    if (syls[i].phraseEnd) {
      raw.push(cur);
      cur = [];
    }
  }
  if (cur.length) raw.push(cur);
  const short = (p: number[]) => p.length < size / 3;
  const fits = (a: number[], b: number[]) => a.length + b.length <= 1.5 * size;
  // "Oh, yeah, baby": a line of a word or two joins the next one (the last joins the one before)
  const merged: number[][] = [];
  for (const p of raw) {
    const prev = merged[merged.length - 1];
    if (prev && short(prev) && fits(prev, p)) prev.push(...p);
    else merged.push([...p]);
  }
  if (merged.length > 1) {
    const last = merged[merged.length - 1], prev = merged[merged.length - 2];
    if (short(last) && fits(prev, last)) prev.push(...merged.pop()!);
  }
  // a long line: cut at the word ends nearest equal parts (never before a held vowel)
  const out: number[][] = [];
  for (const p of merged) {
    if (p.length <= 1.5 * size) {
      out.push(p);
      continue;
    }
    const k = Math.ceil(p.length / size);
    let start = 0;
    for (let c = 1; c < k; c++) {
      const target = Math.round((c * p.length) / k);
      let cut = -1;
      for (let i = start + 1; i < p.length; i++) if (syls[p[i - 1]].wordEnd && !syls[p[i]].melisma && (cut < 0 || Math.abs(i - target) < Math.abs(cut - target))) cut = i;
      if (cut < 0) break;
      out.push(p.slice(start, cut));
      start = cut;
    }
    out.push(p.slice(start));
  }
  return out;
}

/**
 * One line onto one phrase's notes (see the file's comment). The last note ends the phrase, and
 * only it. `from[k]` is the line's syllable note k sings (a held vowel: the syllable it holds).
 */
export function fitLine(line: Syllable[], phrase: readonly [number, number][]): { notes: SungNote[]; from: number[] } {
  const n = line.length;
  if (!n) return { notes: [], from: [] };
  const notes = phrase.map(([midi, beats]) => ({ midi, beats }));
  while (notes.length < n) {
    let k = 0;
    for (let i = 1; i < notes.length; i++) if (notes[i].beats > notes[k].beats) k = i;
    const half = notes[k].beats / 2;
    notes.splice(k, 1, { midi: notes[k].midi, beats: half }, { midi: notes[k].midi, beats: half });
  }
  // merge the shortest neighbours, keeping the first's pitch (the phrase's last note keeps its own)
  while (notes.length > 2 * n) {
    let k = 0;
    const last = notes.length > 2 ? notes.length - 2 : notes.length - 1;
    for (let i = 1; i < last; i++) if (notes[i].beats + notes[i + 1].beats < notes[k].beats + notes[k + 1].beats) k = i;
    notes.splice(k, 2, { midi: notes[k].midi, beats: notes[k].beats + notes[k + 1].beats });
  }
  const m = notes.length;
  // note j sings syllable at(j); counted from the end, so spare notes go to the last syllables
  const at = (j: number) => n - 1 - Math.floor(((m - 1 - j) * n) / m);
  const out = notes.map((note, j) => {
    const s = line[at(j)];
    const held = j > 0 && at(j - 1) === at(j);
    const syl: Syllable = held ? { text: "~", onset: [], nucleus: s.nucleus, coda: [], melisma: true, wordEnd: s.wordEnd, phraseEnd: false } : { ...s, phraseEnd: false };
    if (j === m - 1) syl.phraseEnd = true;
    return { midi: note.midi, beats: note.beats, syl };
  });
  return { notes: out, from: notes.map((_, j) => at(j)) };
}

/**
 * The whole song: every line on its phrase, `shift` semitones from the tune's pitch. `from[k]` is
 * the index in `syls` of the syllable note k sings.
 */
export function fitSong(tune: Tune, syls: Syllable[], shift = 0): { notes: SungNote[]; from: number[] } {
  const notes: SungNote[] = [], from: number[] = [];
  lines(syls, phraseSize(tune)).forEach((line, k) => {
    const f = fitLine(line.map((i) => syls[i]), tune.phrases[k % tune.phrases.length]);
    for (const n of f.notes) notes.push({ ...n, midi: n.midi + shift });
    for (const j of f.from) from.push(line[j]);
  });
  return { notes, from };
}

/** The words a tune is sung on when none are given: "la" on every note. */
export function la(tune: Tune): string {
  return tune.phrases.map((p) => p.map(() => "la").join(" ")).join("\n");
}
