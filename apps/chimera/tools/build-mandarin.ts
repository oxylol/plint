/**
 * Build the Mandarin front end's lexicon shipped to the browser (and read by the CLI).
 * Usage: bun tools/build-mandarin.ts   (downloads the pinned sources below)
 * Output: public/data/mandarin.txt.gz, lines "字<TAB>zhong1" (each Han character's first listed
 * reading, Unihan's order: usually the most common, not always, e.g. 得 de2) and "银行<TAB>yin2 hang2"
 * (phrases whose reading differs from their characters' own,
 * which is what decides a polyphone, and every phrase with 儿, which says whether 儿 is a syllable
 * or an erhua ending), tone numbers 1-4 and 5 for neutral, ü as v.
 *
 * Other defaults were compared and not taken: Unihan's kMandarin (the same as this for nearly every
 * common polyphone, and 长 is zhang3 there too) and kHanyuPinlu (readings by corpus frequency; it
 * changes 81 common characters, most to a neutral tone they only have inside a word: 西 xi5 from
 * 东西). A reading that depends on its neighbour is a rule in the front end instead (长 after a
 * degree word: 很长 hěn cháng).
 *
 * Sources, both MIT (mozillazg): pinyin-data's pinyin.txt (merged from Unicode's Unihan readings,
 * no CC-CEDICT) and phrase-pinyin-data's pinyin.txt (its base phrase list plus overrides; not the
 * large_pinyin.txt that merges CC-CEDICT, which is CC BY-SA).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";

const CHARS = "https://raw.githubusercontent.com/mozillazg/pinyin-data/fe29ad32bdf352b31a02b9936e537f575432e9a6";
const PHRASES = "https://raw.githubusercontent.com/mozillazg/phrase-pinyin-data/4afa3226ce53c9138b939e83c3bb294dbe0da511";

const get = async (url: string) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
};

const TONED: Record<string, [string, number]> = {};
for (const [base, marks] of Object.entries({ a: "āáǎà", e: "ēéěè", i: "īíǐì", o: "ōóǒò", u: "ūúǔù", v: "ǖǘǚǜ", m: "ḿḿ", n: "ńńňǹ" })) {
  [...marks].forEach((m, k) => (TONED[m] = [base, base === "m" ? 2 : base === "n" && k === 0 ? 2 : k + 1]));
}
TONED["ḿ"] = ["m", 2];
TONED["ń"] = ["n", 2];
TONED["ň"] = ["n", 3];
TONED["ǹ"] = ["n", 4];

/** "zhōng" → "zhong1", "lǜ" → "lv4", "de" → "de5"; null for what isn't a pinyin syllable. */
function numbered(syl: string): string | null {
  let tone = 5;
  let out = "";
  for (const ch of syl.normalize("NFC").toLowerCase()) {
    if (TONED[ch]) {
      const [b, t] = TONED[ch];
      out += b;
      tone = t;
    } else if (ch === "ü") out += "v";
    else if (/[a-z]/.test(ch)) out += ch;
    else return null;
  }
  return out ? out + tone : null;
}

const isHan = (c: string) => /^[㐀-䶿一-鿿豈-﫿]$/u.test(c);

/** One syllable: one run of vowels (a character read as two, 瓲 tunwa "ton-watt", can't be one note). */
const oneSyllable = (numberedSyl: string) => (numberedSyl.match(/[aeiouv]+/g) ?? []).length <= 1;

// characters: "U+4E2D: zhōng,zhòng  # 中" → the first listed reading that is one syllable
const chars = new Map<string, string>();
for (const line of (await get(`${CHARS}/pinyin.txt`)).split("\n")) {
  const m = line.match(/^U\+([0-9A-F]+):\s*([^#]+)/);
  if (!m) continue;
  const c = String.fromCodePoint(parseInt(m[1], 16));
  const first = m[2].split(",").map((r) => numbered(r.trim())).find((r) => r && oneSyllable(r));
  if (isHan(c) && first) chars.set(c, first);
}

// phrases: "银行: yín háng" → kept only when some character reads otherwise than on its own
const phrases: string[] = [];
let all = 0;
for (const line of (await get(`${PHRASES}/pinyin.txt`)).split("\n")) {
  const m = line.match(/^([^#:\s]+):\s*([^#]+)/);
  if (!m) continue;
  const word = [...m[1]];
  const syls = m[2].trim().split(/\s+/).map(numbered);
  if (word.length < 2 || word.length !== syls.length || syls.some((s) => !s) || !word.every(isHan)) continue;
  all++;
  // kept: phrases that read a character otherwise, and every phrase with 儿 (it says whether 儿 is
  // a syllable, 女儿 er2, or an erhua ending, 一会儿 er5; the front end needs to know which)
  if (word.some((c, i) => chars.get(c) !== syls[i]) || word.includes("儿")) phrases.push(`${m[1]}\t${syls.join(" ")}`);
}

const lines = [...[...chars].map(([c, p]) => `${c}\t${p}`), ...phrases];
mkdirSync("public/data", { recursive: true });
writeFileSync("public/data/mandarin.txt.gz", gzipSync(Buffer.from(lines.join("\n"), "utf8"), { level: 9 }));
const license = (await get(`${CHARS}/LICENSE`)) + "\n---\n\n" + (await get(`${PHRASES}/LICENSE`));
writeFileSync(
  "public/data/MANDARIN-PINYIN-LICENSE.txt",
  `public/data/mandarin.txt.gz is built by tools/build-mandarin.ts from\n- ${CHARS}/pinyin.txt (pinyin-data)\n- ${PHRASES}/pinyin.txt (phrase-pinyin-data)\nunder their MIT licenses:\n\n${license}`,
);
console.log(`${chars.size} characters, ${phrases.length} of ${all} phrases (the ones that decide a reading)`);
