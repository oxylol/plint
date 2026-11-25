//! Lyrics → sung syllables, through language front ends (docs/language-front-ends.md). A front
//! end claims the words written in its script and turns each into syllables in IPA; the router
//! here walks the lyrics, sends each word to the front end that claims it, and handles what every
//! language shares: melisma marks, phrase punctuation and word ends.
//!
//! English words come from the pronouncing dictionary (unreduced: sung function words keep full
//! vowels), are split into syllables by the maximal-onset principle with English onset clusters,
//! and the spelling is split to match for display ("twin-kle"). Hangul words go to
//! [`super::korean`], one syllable per block, Cyrillic words to [`super::russian`], kana to
//! [`super::japanese`], one mora per note. Latin script is English unless the player chooses
//! Turkish ([`Lang`], [`super::turkish`]): the script can't tell them apart. Typed hyphens force
//! syllable breaks; a lone "-" (or
//! "~") holds the previous vowel onto the next note (melisma), as in vocaloid editors.

use crate::anatomy::human::BodyMix;
use super::english::js_is_space;
use super::japanese::Japanese;
use super::korean::Korean;
use super::mandarin::{Mandarin, MandarinLexicon, Pinyin};
use super::russian::Russian;
use super::turkish::Turkish;
use super::planner::is_vowel;
use super::{Segment, Syllable};
use serde::{Deserialize, Serialize};

// English onsets (IPA), longest first matters only for the search below
#[rustfmt::skip]
const ONSETS: &[&str] = &[
    "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "ɹ", "w", "j", "tʃ", "dʒ",
    "pl", "bl", "kl", "gl", "fl", "sl", "pɹ", "bɹ", "tɹ", "dɹ", "kɹ", "gɹ", "fɹ", "θɹ", "ʃɹ",
    "sp", "st", "sk", "sm", "sn", "sw", "sf", "tw", "kw", "dw", "gw", "θw",
    "pj", "bj", "kj", "gj", "mj", "nj", "fj", "vj", "hj", "lj",
    "spl", "spɹ", "stɹ", "skɹ", "skw", "skj", "spj",
];

/// One syllable of a word, before it gets its text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SyllableParts {
    pub onset: Vec<Segment>,
    pub nucleus: Segment,
    pub coda: Vec<Segment>,
}

/// Split one word's segments into onset / nucleus / coda syllables, with English's onsets.
pub fn syllabify_segments(segs: &[Segment]) -> Vec<SyllableParts> {
    syllabify_with(segs, is_english_onset)
}

/// A consonant cluster that can start an English syllable.
pub fn is_english_onset(cluster: &[Segment]) -> bool {
    let cluster: String = cluster.iter().map(|s| s.ipa.as_str()).collect();
    ONSETS.contains(&cluster.as_str())
}

/// Split one word's segments into syllables: of the consonants between two vowels, the longest
/// run that `is_onset` accepts starts the second syllable.
pub fn syllabify_with(segs: &[Segment], is_onset: impl Fn(&[Segment]) -> bool) -> Vec<SyllableParts> {
    let vi: Vec<usize> = segs.iter().enumerate().filter(|(_, s)| is_vowel(&s.ipa, BodyMix::MALE)).map(|(i, _)| i).collect();
    if vi.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    for (n, &v) in vi.iter().enumerate() {
        let mut end = segs.len(); // exclusive end of this syllable
        if let Some(&next) = vi.get(n + 1) {
            // consonants between this vowel and the next: the longest legal onset goes right
            let between = &segs[v + 1..next];
            let mut k = 0;
            for len in (1..=between.len()).rev() {
                if is_onset(&between[between.len() - len..]) {
                    k = len;
                    break;
                }
            }
            end = next - k;
        }
        out.push(SyllableParts { onset: segs[start..v].to_vec(), nucleus: segs[v].clone(), coda: segs[v + 1..end].to_vec() });
        start = end;
    }
    out
}

// The spelling is indexed in UTF-16 code units, as the JS string was.
fn is_spell_vowel(c: u16) -> bool {
    "aeiouy".bytes().any(|v| v as u16 == c)
}
fn is_spell_vowel_ci(c: u16) -> bool {
    "aeiouyAEIOUY".bytes().any(|v| v as u16 == c)
}
fn eq_ci(c: u16, lower: u8) -> bool {
    c == lower as u16 || c == lower.to_ascii_uppercase() as u16
}
/// `word.slice(a, b)` (indices clamped to the word)
fn slice(w: &[u16], a: usize, b: usize) -> String {
    let (a, b) = (a.min(w.len()), b.min(w.len()));
    if a >= b { String::new() } else { String::from_utf16_lossy(&w[a..b]) }
}

/// Split a written word into n chunks at its vowel groups (approximate hyphenation).
pub fn split_spelling(word: &str, n: usize) -> Vec<String> {
    if n <= 1 {
        return vec![word.to_string()];
    }
    let w: Vec<u16> = word.encode_utf16().collect();
    let lower: Vec<u16> = word.to_lowercase().encode_utf16().collect();
    // /[aeiouy]+/g on the lowercased word
    let mut groups: Vec<[usize; 2]> = Vec::new();
    let mut i = 0;
    while i < lower.len() {
        if is_spell_vowel(lower[i]) {
            let s = i;
            while i < lower.len() && is_spell_vowel(lower[i]) {
                i += 1;
            }
            groups.push([s, i]);
        } else {
            i += 1;
        }
    }
    let len = w.len();
    // /[^aeiouy]e$/i and /[^aeiouyl]le$/i
    let silent_e = len >= 2 && eq_ci(w[len - 1], b'e') && !is_spell_vowel_ci(w[len - 2]);
    let le = len >= 3 && eq_ci(w[len - 1], b'e') && eq_ci(w[len - 2], b'l') && !is_spell_vowel_ci(w[len - 3]) && !eq_ci(w[len - 3], b'l');
    // a final silent e ("like") is not a syllable, but consonant + "le" is ("twin-kle", "lit-tle")
    if groups.len() > 1 && silent_e && !le && groups[groups.len() - 1][0] + 1 == len {
        groups.pop();
    }
    if groups.len() < n {
        // fewer written vowels than sung syllables (e.g. "fire"): hold the word, then continue marks
        let mut out = vec![word.to_string()];
        out.extend(std::iter::repeat_n("~".to_string(), n - 1));
        return out;
    }
    while groups.len() > n {
        let last = groups.pop().unwrap();
        let k = groups.len() - 1;
        groups[k][1] = last[1];
    }
    let mut cuts: Vec<usize> = vec![0];
    for i in 0..groups.len() - 1 {
        let (a, b) = (groups[i][1], groups[i + 1][0]); // consonant letters a..b between vowel groups
        // a digraph stays together on the right ("any-where", "fa-ther"); otherwise one consonant
        // goes right and longer runs split in the middle ("lit-tle", "won-der")
        let gap = b - a;
        let dig = gap >= 2 && {
            let d = [w.get(b - 2).copied(), w.get(b - 1).copied()];
            ["th", "sh", "ch", "ph", "wh", "ck", "gh"].iter().any(|p| {
                let p = p.as_bytes();
                d[0].is_some_and(|c| eq_ci(c, p[0])) && d[1].is_some_and(|c| eq_ci(c, p[1]))
            })
        };
        cuts.push(if dig { b - 2 } else { a + if gap <= 1 { 0 } else { gap / 2 } });
    }
    cuts.push(len);
    (0..cuts.len() - 1).map(|i| slice(&w, cuts[i], cuts[i + 1])).collect()
}

/// Where lyrics get their pronunciations: `EnglishG2P`, or any `Fn(&str, bool) -> Vec<Segment>`.
pub trait WordSource {
    fn word(&self, w: &str, reduce: bool) -> Vec<Segment>;
}

impl<F: Fn(&str, bool) -> Vec<Segment>> WordSource for F {
    fn word(&self, w: &str, reduce: bool) -> Vec<Segment> {
        self(w, reduce)
    }
}

fn is_lyric_punct(c: char) -> bool {
    // ASCII, the full-width forms Chinese and Japanese lyrics use, and the half-width ､ ｡ typed
    // with half-width katakana
    matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '\n' | '，' | '。' | '！' | '？' | '；' | '：' | '、' | '\u{FF64}' | '\u{FF61}')
}

/// A language front end: claims the words written in its script and turns each into sung syllables
/// (IPA onset, nucleus and coda; each syllable's text is its part of the word's spelling).
pub trait FrontEnd {
    /// A word this front end reads starts with `c`.
    fn starts(&self, c: char) -> bool;
    /// `c` continues a word this front end started.
    fn continues(&self, c: char) -> bool {
        self.starts(c)
    }
    /// Whether this front end reads the whole `word` it scanned (default: yes). A front end that
    /// shares a script with another says no to what isn't its own, and the next one gets the word
    /// (pinyin and English share the Latin alphabet).
    fn claims(&self, _word: &str) -> bool {
        true
    }
    /// One word → its syllables (the router marks the last as the word's end).
    fn word(&self, word: &str) -> Vec<Syllable>;
    /// The same, knowing the word of this front end just before it in the phrase, for a language
    /// whose words lean on each other (a Russian preposition with no vowel has no note of its
    /// own: it returns no syllables and starts the next word's first one).
    fn word_after(&self, word: &str, _prev: Option<&str>) -> Vec<Syllable> {
        self.word(word)
    }
}

/// English, through a [`WordSource`] (the pronouncing dictionary, or letter-to-sound rules).
pub struct English<'a, W: WordSource>(pub &'a W);

impl<W: WordSource> FrontEnd for English<'_, W> {
    fn starts(&self, c: char) -> bool {
        c.is_ascii_alphabetic() || c == '\''
    }
    fn continues(&self, c: char) -> bool {
        self.starts(c) || c == '-'
    }
    fn word(&self, tok: &str) -> Vec<Syllable> {
        // typed hyphens split syllables explicitly ("ti-ger"); otherwise the dictionary decides
        let parts: Vec<&str> = tok.split('-').filter(|p| !p.is_empty()).collect();
        let mut syls: Vec<Syllable> = Vec::new();
        for part in &parts {
            let sy = syllabify_segments(&self.0.word(&part.to_lowercase(), false));
            let spell: Vec<String> = if parts.len() > 1 {
                let mut v = vec![part.to_string()];
                v.extend(std::iter::repeat_n("~".to_string(), sy.len().saturating_sub(1)));
                v
            } else {
                split_spelling(part, sy.len())
            };
            for (i, s) in sy.into_iter().enumerate() {
                syls.push(Syllable { text: spell.get(i).cloned().unwrap_or_else(|| "~".into()), onset: s.onset, nucleus: s.nucleus, coda: s.coda, melisma: None, word_end: false, phrase_end: false, ..Default::default() });
            }
        }
        syls
    }
}

enum Token {
    Word(usize, String),
    Mark(char),
}

/// Words (with the index of the front end that claims them), melisma marks and phrase punctuation,
/// in order. For English and precomposed Hangul this is
/// `text.match(/[A-Za-z'][A-Za-z'\-]*|[가-힣]+|(?<=\s|^)[~-](?=\s|$)|[.,;:!?\n]/g)`.
fn lyric_tokens(text: &str, front_ends: &[&dyn FrontEnd]) -> Vec<Token> {
    let s: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    // where each front end may claim again after declining a word (the end of that word)
    let mut barred = vec![0usize; front_ends.len()];
    while i < s.len() {
        let c = s[i];
        // the first front end that starts here and claims the word it scans. A front end that
        // declines a word (pinyin and English share the Latin alphabet) can't claim a piece of it
        // either: the others read that span as they would without it ("café" stays English's "caf",
        // not "caf" and a pinyin "é"), and nothing in it is lost ("Zürich" is "Z" "rich", as on main)
        let mut word = None;
        for (k, f) in front_ends.iter().enumerate() {
            if !f.starts(c) || barred[k] > i {
                continue;
            }
            let mut j = i + 1;
            while j < s.len() && f.continues(s[j]) {
                j += 1;
            }
            let w: String = s[i..j].iter().collect();
            if f.claims(&w) {
                word = Some((k, j, w));
                break;
            }
            barred[k] = j;
        }
        if let Some((k, j, w)) = word {
            out.push(Token::Word(k, w));
            i = j;
            continue;
        }
        let melisma = (c == '~' || c == '-') && (i == 0 || js_is_space(s[i - 1])) && (i + 1 == s.len() || js_is_space(s[i + 1]));
        if melisma || is_lyric_punct(c) {
            out.push(Token::Mark(c));
        }
        i += 1;
    }
    out
}

/// Lyrics text → syllables, each word read by the first front end that claims it. Characters no
/// front end claims (other scripts, symbols) are skipped.
pub fn read_lyrics(text: &str, front_ends: &[&dyn FrontEnd]) -> Vec<Syllable> {
    let mut out: Vec<Syllable> = Vec::new();
    // the word before this one and its front end, within a phrase
    let mut prev: Option<(usize, String)> = None;
    for tok in lyric_tokens(&text.replace('’', "'"), front_ends) {
        match tok {
            Token::Mark(c) if is_lyric_punct(c) => {
                prev = None;
                if let Some(last) = out.last_mut() {
                    last.phrase_end = true;
                }
            }
            Token::Mark(_) => {
                if let Some(prev) = out.last() {
                    let s = Syllable { text: "~".into(), onset: vec![], nucleus: prev.nucleus.clone(), coda: vec![], melisma: Some(true), word_end: prev.word_end, phrase_end: false, ..Default::default() };
                    out.push(s);
                }
            }
            Token::Word(k, w) => {
                let before = prev.as_ref().filter(|(pk, _)| *pk == k).map(|(_, p)| p.as_str());
                let mut syls = front_ends[k].word_after(&w, before);
                if let Some(last) = syls.last_mut() {
                    last.word_end = true;
                }
                out.append(&mut syls);
                prev = Some((k, w));
            }
        }
    }
    out
}

/// Lyrics text → syllables through the built-in front ends that need no data beyond the English
/// dictionary: every built-in front end but Mandarin's characters (they need its lexicon; see
/// [`lyrics_to_syllables_with`]).
pub fn lyrics_to_syllables(text: &str, g2p: &impl WordSource) -> Vec<Syllable> {
    lyrics_to_syllables_with(text, g2p, &MandarinLexicon::new())
}

/// Lyrics → sung syllables through every built-in front end: pinyin (only words with tones; before
/// English, which shares its alphabet), English (`g2p`), Korean, Russian, Japanese kana, and
/// Mandarin with its lexicon (empty: Han characters are skipped). The one list the app, the wasm engine and the CLI
/// all read lyrics through, so a new front end can't be missing from one of them.
pub fn lyrics_to_syllables_with(text: &str, g2p: &impl WordSource, zh: &MandarinLexicon) -> Vec<Syllable> {
    lyrics_in(text, g2p, zh, Lang::En)
}

/// The language Latin-script lyrics are in. The router tells languages apart by script, which
/// can't tell Turkish from English, so the player chooses; other scripts are routed as usual.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    #[default]
    En,
    Tr,
}

/// Lyrics → sung syllables through every built-in front end, with Latin script read as `lang`:
/// pinyin (only words with tones; before the Latin-script language, which shares its alphabet),
/// English (`g2p`) or Turkish, Korean, Russian, Japanese kana, and Mandarin with its lexicon
/// (empty: Han characters are skipped). The one list the app, the wasm engine and the CLI all read
/// lyrics through, so a new front end can't be missing from one of them.
pub fn lyrics_in(text: &str, g2p: &impl WordSource, zh: &MandarinLexicon, lang: Lang) -> Vec<Syllable> {
    match lang {
        Lang::En => read_lyrics(text, &[&Pinyin, &English(g2p), &Korean, &Russian, &Japanese, &Mandarin(zh)]),
        Lang::Tr => read_lyrics(text, &[&Pinyin, &Turkish, &Korean, &Russian, &Japanese, &Mandarin(zh)]),
    }
}
