//! Kana lyrics → sung syllables. One mora is one note, the convention of Japanese singing: each
//! kana (or a kana with a small ゃゅょ or small vowel) is one syllable whose text is the kana as
//! typed, and ん and a long vowel's second half get notes of their own. A vowel is one of あ a,
//! い i, う ɯ (unrounded), え e, お o.
//!
//! The script this front end claims is hiragana, katakana (with ー) and half-width katakana. **It
//! claims no Han character:** kanji go to the Mandarin front end (docs/language-front-ends.md, "Not
//! yet": which language a Han text is needs a per-song setting), so Japanese lyrics are typed in
//! kana, as in most singing editors. There is no kanji dictionary.
//!
//! What the front end decides (docs/language-front-ends.md, "Japanese"):
//! - Consonants: し ɕ, ち tɕ, つ ts, ふ ɸ, ひ ç, じ/ぢ dʑ, ず/づ z, に ɲ, ら行 ɾ, わ w, や行 j. The
//!   ざ行 is dz at the start of a word and after ん, z between vowels. Yōon (きゃ = kʲ j a, the
//!   consonant palatalised, but not ぎゃ's g, whose closure it opens; しゃ ɕa, ちゃ tɕa, じゃ dʑa,
//!   にゃ ɲa, which already carry their palatal; ひゃ hʲ j a, louder than the palatal fricative) and the small-kana loanword sounds (ふぁ ɸa, てぃ ti, うぃ wi, ちぇ tɕe, ヴ v…).
//! - Aspiration: the voiceless stops say theirs (k 70 ms, p t 50), because English's positional
//!   rule only aspirates a stop right before a vowel, which would make the k of きゃ (k j a) differ
//!   from the k of か for no reason. ち and つ say nothing.
//! - ん is a mora, so it is a note: a hum for the whole note on the nasal of the sound after it
//!   (m before p b m, n before t d n ts z ɾ tɕ dʑ ɲ, ŋ before k g, a vowel, a glide, a fricative or
//!   at the end of a word), the note's nucleus. The note before closes into it (a short coda of the
//!   same nasal, its vowel nasalised over its end), so かん is one movement, [kãŋ] [ŋ].
//! - っ is a mora, so it is a note: the vowel before goes on into it, and it ends in the long
//!   consonant after it (the closure of a stop, the frication of a fricative or affricate, held
//!   longer out of that note). Before a vowel, a glide, ɾ, ん or the end of a word a glottal stop
//!   cuts the note before and is held, silent, through っ's note. At the start of a word there is
//!   no note to go on from and it is dropped.
//! - Long vowels sing as melisma notes that continue the previous vowel, one note per kana: ー (and
//!   〜 inside a word); a vowel kana that repeats the vowel of the mora before it (おお, ああ, きい,
//!   ゆう); and the spelling おう → oo (えい is sung e, i). The rule is wrong when the う is a real
//!   syllable of its own (the う of 思う, written おもう, would sing as a continued o): a hyphen
//!   between the kana ("おも-う") starts a note of its own, read as the kana is written.
//! - は and へ are read as written (ha, he); type わ and え for the particle readings (the
//!   greetings こんにちは and こんばんは are read wa). を is o.
//!
//! No pitch accent and no vowel devoicing: a sung note carries its own pitch, and a devoiced
//! vowel would be a silent note. No pitch bends.

use super::inventory::phoneme;
use super::syllables::FrontEnd;
use super::{Segment, Syllable};
use crate::anatomy::human::BodyMix;

const VOWELS: [&str; 5] = ["a", "i", "ɯ", "e", "o"];

/// Kana this front end reads: hiragana, katakana and half-width katakana letters.
pub fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | '\u{FF66}'..='\u{FF9D}')
}

/// Marks that only continue a word: the long-vowel marks ー and 〜, the voicing marks, the iteration
/// marks ゝ ゞ ヽ ヾ, and the hyphen that starts a note of its own.
fn is_mark(c: char) -> bool {
    matches!(c, '-' | '\u{30FC}' | '\u{FF70}' | '\u{301C}' | '\u{FF5E}' | '\u{3099}'..='\u{309E}' | '\u{30FD}' | '\u{30FE}' | '\u{FF9E}' | '\u{FF9F}')
}

/// Half-width katakana U+FF66.. as full-width.
const HALFWIDTH: &str = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン";

/// A kana's voiced form (か → が, う → ゔ), or None.
fn voiced(c: char) -> Option<char> {
    let u = c as u32;
    let v = match u {
        0x304B..=0x3062 if (u - 0x304B) % 2 == 0 => u + 1,
        0x3064..=0x3069 if (u - 0x3064) % 2 == 0 => u + 1,
        0x306F..=0x307D if (u - 0x306F) % 3 == 0 => u + 1,
        0x3046 => 0x3094,
        _ => return None,
    };
    char::from_u32(v)
}

/// The small vowel after ゔ in the voiced form of わ ゐ ゑ を (ヷ ヸ ヹ ヺ, or the kana and a voicing
/// mark: [va] [vi] [ve] [vo]), or None.
fn voiced_w(c: char) -> Option<char> {
    match c {
        'わ' => Some('ぁ'),
        'ゐ' => Some('ぃ'),
        'ゑ' => Some('ぇ'),
        'を' => Some('ぉ'),
        _ => None,
    }
}

/// A kana's half-voiced form (は → ぱ), or None.
fn semivoiced(c: char) -> Option<char> {
    let u = c as u32;
    if (0x306F..=0x307D).contains(&u) && (u - 0x306F) % 3 == 0 { char::from_u32(u + 2) } else { None }
}

/// One kana as hiragana, with the characters it was typed as.
type Kana = (char, String);

/// The word as hiragana: katakana and half-width folded, voicing marks and iteration marks applied.
fn normalise(word: &str) -> Vec<Kana> {
    let mut out: Vec<Kana> = Vec::new();
    for ch in word.chars() {
        let full = match ch {
            '\u{FF66}'..='\u{FF9D}' => HALFWIDTH.chars().nth(ch as usize - 0xFF66).unwrap_or(ch),
            other => other,
        };
        let u = full as u32;
        match ch {
            '\u{3099}' | '\u{309B}' | '\u{FF9E}' | '\u{309A}' | '\u{309C}' | '\u{FF9F}' => {
                let semi = matches!(ch, '\u{309A}' | '\u{309C}' | '\u{FF9F}');
                let mut small = None;
                if let Some(last) = out.last_mut() {
                    if let Some(v) = if semi { semivoiced(last.0) } else { voiced(last.0) } {
                        last.0 = v;
                    } else if let (false, Some(vowel)) = (semi, voiced_w(last.0)) {
                        // ﾜﾞ and ワ + U+3099 read as ヷ does
                        last.0 = 'ゔ';
                        small = Some(vowel);
                    }
                    last.1.push(ch);
                }
                if let Some(vowel) = small {
                    out.push((vowel, String::new()));
                }
            }
            '\u{309D}' | '\u{30FD}' | '\u{309E}' | '\u{30FE}' => {
                if let Some(prev) = out.last().map(|k| k.0).filter(|&c| base(c).is_some()) {
                    let c = if matches!(ch, '\u{309E}' | '\u{30FE}') { voiced(prev).unwrap_or(prev) } else { prev };
                    out.push((c, ch.to_string()));
                }
            }
            // ヷ ヸ ヹ ヺ: v + a small vowel, the reading of わ ゐ ゑ を with a voicing mark
            '\u{30F7}'..='\u{30FA}' => {
                let vowel = voiced_w(['わ', 'ゐ', 'ゑ', 'を'][(u - 0x30F7) as usize]).unwrap_or('ぁ');
                out.push(('ゔ', ch.to_string()));
                out.push((vowel, String::new()));
            }
            // 〜 and ～ hold a vowel like ー
            '\u{301C}' | '\u{FF5E}' => out.push(('ー', ch.to_string())),
            _ => {
                let hira = if (0x30A1..=0x30F6).contains(&u) { char::from_u32(u - 0x60).unwrap_or(full) } else { full };
                out.push((hira, ch.to_string()));
            }
        }
    }
    out
}

/// Consonant rows: the five kana of a row (a i u e o) and each one's consonant.
const ROWS: [(&str, [&str; 5]); 12] = [
    ("かきくけこ", ["k"; 5]),
    ("がぎぐげご", ["g"; 5]),
    ("さしすせそ", ["s", "ɕ", "s", "s", "s"]),
    ("ざじずぜぞ", ["z", "dʑ", "z", "z", "z"]),
    ("たちつてと", ["t", "tɕ", "ts", "t", "t"]),
    ("だぢづでど", ["d", "dʑ", "z", "d", "d"]),
    ("なにぬねの", ["n", "ɲ", "n", "n", "n"]),
    ("はひふへほ", ["h", "ç", "ɸ", "h", "h"]),
    ("ばびぶべぼ", ["b"; 5]),
    ("ぱぴぷぺぽ", ["p"; 5]),
    ("まみむめも", ["m"; 5]),
    ("らりるれろ", ["ɾ"; 5]),
];

/// A base kana's consonant ("" for none) and vowel.
fn base(c: char) -> Option<(&'static str, &'static str)> {
    for (kana, cons) in &ROWS {
        if let Some(i) = kana.chars().position(|k| k == c) {
            return Some((cons[i], VOWELS[i]));
        }
    }
    Some(match c {
        'あ' => ("", "a"),
        'い' | 'ゐ' => ("", "i"),
        'う' => ("", "ɯ"),
        'え' | 'ゑ' => ("", "e"),
        'お' | 'を' => ("", "o"),
        'や' => ("j", "a"),
        'ゆ' => ("j", "ɯ"),
        'よ' => ("j", "o"),
        // わ is written [ɰa] (lips compressed, not rounded), but sung with the inventory's ɰ, the
        // tongue alone, a listener heard ら and the recogniser lost the word (わたし as 新): w
        'わ' | 'ゎ' => ("w", "a"),
        'ゔ' => ("v", "ɯ"),
        'ゕ' => ("k", "a"),
        'ゖ' => ("k", "e"),
        _ => return None,
    })
}

fn small_vowel(c: char) -> Option<usize> {
    "ぁぃぅぇぉ".chars().position(|k| k == c)
}

fn small_y(c: char) -> Option<&'static str> {
    Some(match c {
        'ゃ' => "a",
        'ゅ' => "ɯ",
        'ょ' => "o",
        _ => return None,
    })
}

/// Consonants that already carry their palatalisation: a yōon after them has no j of its own.
fn is_palatal(c: &str) -> bool {
    matches!(c, "ɕ" | "tɕ" | "dʑ" | "ɲ" | "ç")
}

/// A mora, before notes are made of them.
enum Mora {
    /// onset consonants (IPA), the vowel, the kana typed, and which of あいうえお it is (a vowel kana
    /// may continue the vowel before it)
    Syl { onset: Vec<&'static str>, vowel: &'static str, src: String, plain: Option<usize> },
    /// ん
    Nasal(String),
    /// っ
    Sokuon(String),
    /// ー
    Long(String),
    /// a hyphen: the kana after it starts a note of its own
    Break,
}

fn moras(kana: &[Kana]) -> Vec<Mora> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < kana.len() {
        let (c, src) = &kana[i];
        let mut src = src.clone();
        i += 1;
        match c {
            'ん' => out.push(Mora::Nasal(src)),
            'っ' => out.push(Mora::Sokuon(src)),
            'ー' => out.push(Mora::Long(src)),
            '-' => out.push(Mora::Break),
            // small kana that follow no kana of their own are read like the full-size ones
            'ゃ' | 'ゅ' | 'ょ' => out.push(Mora::Syl { onset: vec!["j"], vowel: small_y(*c).unwrap(), src, plain: None }),
            'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ' => {
                let k = small_vowel(*c).unwrap();
                out.push(Mora::Syl { onset: vec![], vowel: VOWELS[k], src, plain: Some(k) });
            }
            _ => {
                let Some((cons, vowel)) = base(*c) else { continue };
                let mut onset: Vec<&'static str> = if cons.is_empty() { vec![] } else { vec![cons] };
                let mut vowel = vowel;
                let plain = (cons.is_empty() && *c != 'を').then(|| VOWELS.iter().position(|v| *v == vowel).unwrap());
                if let Some((next, next_src)) = kana.get(i) {
                    if let Some(k) = small_vowel(*next) {
                        // ふぁ ɸa, てぃ ti, ちぇ tɕe, うぃ wi, くぁ kwa, いぇ je; あぁ is a lengthened あ
                        let joined = match c {
                            'う' if k != 2 => Some(vec!["w"]),
                            'い' if k == 3 => Some(vec!["j"]),
                            'く' | 'ぐ' => Some(vec![cons, "w"]),
                            'あ' | 'い' | 'う' | 'え' | 'お' | 'を' | 'ゐ' | 'ゑ' => None,
                            _ if cons.is_empty() => None,
                            _ => Some(onset.clone()),
                        };
                        if let Some(o) = joined {
                            onset = o;
                            vowel = VOWELS[k];
                            src.push_str(next_src);
                            i += 1;
                        }
                    } else if let Some(v) = small_y(*next) {
                        // きゃ k j a; しゃ ɕa carries its own palatal
                        let joined = match cons {
                            "" if *c == 'い' => Some(vec!["j"]),
                            "" | "j" => None,
                            // ひゃ: h at the palate and its glide. As [ça], with no glide, the
                            // fricative was 36 dB under the vowel and a listener heard no ひ
                            "ç" => Some(vec!["h", "j"]),
                            c if is_palatal(c) => Some(vec![c]),
                            c => Some(vec![c, "j"]),
                        };
                        if let Some(o) = joined {
                            onset = o;
                            vowel = v;
                            src.push_str(next_src);
                            i += 1;
                        }
                    }
                }
                let plain = plain.filter(|_| onset.is_empty());
                out.push(Mora::Syl { onset, vowel, src, plain });
            }
        }
    }
    out
}

/// How much longer than a single one the consonant after a っ is held (s), or as long as the っ's
/// note leaves it (the planner shortens it to fit). Twice a single consonant, 150 ms at least, as
/// it was first, is a geminate in speech, but inside a sung note of half a second it was not heard
/// as one (がっこう as 過去).
const GEMINATE_HOLD: f64 = 0.25;
/// ん is a hum for the whole of its note, and the vowel before closes into it: the note before
/// ends in a short coda of the same nasal (s), its vowel nasalised over this share of its end, in
/// one movement (`Syllable::closing`). A listener chose this, on long notes and short ones, over
/// the vowel going on into ん's note and closing onto the nasal at its end, which on a two-beat
/// ん was heard as "ka...n" (an ACE Studio singer hums ん the same way). At 0.6 s a mora the hum
/// had been heard as two sounds too: listen at a natural tempo.
const N_CODA: f64 = 0.08;
const N_CLOSING: f64 = 0.2;
const N_LEAD: f64 = 0.6;
/// the planner's factor for a sung consonant's spoken length (`sing.rs`)
const SUNG_CONSONANT: f64 = 0.8;
/// The glide of ぎゃ ぎゅ ぎょ (s).
const YOON_GLIDE_AFTER_G: f64 = 0.03;

/// A consonant as the front end says it. Japanese voiceless stops are aspirated, less than
/// English's at the start of a word: k 70 ms, p and t 50. With 40 and 30 the release was too short
/// to be heard as a stop (ぱん as はん, てんき and さかな lost their t and k to the recogniser).
/// The affricates ち つ say nothing and are sung as the planner sings an affricate.
fn consonant(ipa: &str) -> Segment {
    let vot = match ipa {
        "k" => Some(0.07),
        "p" | "t" => Some(0.05),
        _ => None,
    };
    Segment { aspirated: vot.map(|_| true), vot, ..Segment::new(ipa) }
}

/// The nasal a ん hums before this mora: that of a stop, affricate, nasal or tap after it, else
/// (a vowel, a glide, a fricative, the end of a word) a velar one. One of m n ŋ, the nasals the
/// planner closes a rhyme onto.
fn nasal_before(next: Option<&Mora>) -> &'static str {
    let Some(Mora::Syl { onset, .. }) = next else { return "ŋ" };
    match onset.first().copied().unwrap_or("") {
        "p" | "b" | "m" => "m",
        "t" | "d" | "n" | "ts" | "z" | "dz" | "ɾ" | "tɕ" | "dʑ" | "ɲ" => "n",
        _ => "ŋ",
    }
}

/// Whether the consonant after a っ can be held (a stop, fricative or affricate; not a glide or tap).
fn can_geminate(next: Option<&Mora>) -> bool {
    matches!(next, Some(Mora::Syl { onset, .. }) if onset.first().is_some_and(|c| !matches!(*c, "j" | "w" | "ɾ")))
}

/// One word of kana → its sung syllables, the last marked as the word's end.
pub fn word_syllables(word: &str) -> Vec<Syllable> {
    let mut kana = normalise(word);
    // the greetings こんにちは and こんばんは end in the particle は, read wa
    let hira: String = kana.iter().map(|k| k.0).collect();
    if let (true, Some(last)) = (hira == "こんにちは" || hira == "こんばんは", kana.last_mut()) {
        last.0 = 'わ';
    }
    let ms = moras(&kana);
    let mut out: Vec<Syllable> = Vec::with_capacity(ms.len());
    // the chain's first note (a coda is sung after the chain, and the planner reads it from here),
    // and the vowel the last mora sang (what a vowel kana may continue)
    let mut head: Option<usize> = None;
    let mut last_vowel: Option<&str> = None;
    let mut geminate = false;
    let mut after_nasal = false;
    // whether the last note sings a vowel that ん or っ can go on from
    let mut sings_vowel = false;
    for (k, m) in ms.iter().enumerate() {
        let next = ms.get(k + 1);
        match m {
            Mora::Syl { onset, vowel, src, plain } => {
                let continues = plain.is_some_and(|p| last_vowel.is_some_and(|l| l == *vowel || (p == 2 && l == "o")));
                if continues {
                    let nucleus = out.last().map(|s| s.nucleus.clone()).unwrap_or_default();
                    out.push(Syllable { text: src.clone(), nucleus, melisma: Some(true), ..Default::default() });
                } else {
                    // ざ行 is an affricate at the start of a word and after ん, a fricative between vowels
                    let affricate = after_nasal || out.is_empty();
                    let mut segs: Vec<Segment> = onset.iter().map(|c| consonant(if affricate && *c == "z" { "dz" } else { c })).collect();
                    // a yōon's consonant is palatalised: the tongue body is at the j while it is made
                    if segs.len() == 2 && segs[1].ipa == "j" {
                        if segs[0].ipa == "g" {
                            // but not ぎゃ's: the tongue body is what closes a velar, and
                            // palatalised it let the voice through (the closure 15-20 dB under
                            // the vowel, where が's is silent), so a listener heard no ぎ. The
                            // glide is short, so the release is next to the vowel
                            segs[1].dur = Some(YOON_GLIDE_AFTER_G);
                        } else {
                            segs[0].palatalized = Some(true);
                        }
                    }
                    if let (true, Some(first)) = (geminate, segs.first_mut()) {
                        let spoken = phoneme(&first.ipa, BodyMix::MALE).map_or(0.07, |d| d.dur);
                        first.dur = Some(spoken * SUNG_CONSONANT + GEMINATE_HOLD);
                    }
                    out.push(Syllable { text: src.clone(), onset: segs, nucleus: Segment::new(vowel), ..Default::default() });
                    head = Some(out.len() - 1);
                    last_vowel = Some(vowel);
                    sings_vowel = true;
                }
                geminate = false;
                after_nasal = false;
            }
            Mora::Long(src) => {
                if let Some(prev) = out.last() {
                    let nucleus = prev.nucleus.clone();
                    out.push(Syllable { text: src.clone(), nucleus, melisma: Some(true), ..Default::default() });
                }
                geminate = false;
                after_nasal = false;
            }
            Mora::Nasal(src) => {
                // a hum, the nasal as the note's nucleus, held for the whole note
                let hum = nasal_before(next);
                let humming = out.last().is_some_and(|p| matches!(p.nucleus.ipa.as_str(), "m" | "n" | "ŋ"));
                if humming {
                    // a second ん goes on with the hum
                    let nucleus = out.last().map(|p| p.nucleus.clone()).unwrap_or_default();
                    out.push(Syllable { text: src.clone(), nucleus, melisma: Some(true), ..Default::default() });
                } else {
                    // the chain before closes into it (the planner reads a chain's coda from its
                    // first note); with no vowel before it (the start of a word) it starts on its own
                    if let (Some(h), true) = (head, sings_vowel) {
                        out[h].coda = vec![Segment { dur: Some(N_CODA), ..Segment::new(hum) }];
                        out[h].closing = Some(N_CLOSING);
                        out[h].nasal_lead = Some(N_LEAD);
                    }
                    out.push(Syllable { text: src.clone(), nucleus: Segment::new(hum), ..Default::default() });
                    head = Some(out.len() - 1);
                    sings_vowel = false;
                }
                last_vowel = None;
                geminate = false;
                after_nasal = true;
            }
            Mora::Sokuon(src) => {
                if let (Some(h), Some(prev)) = (head, out.last()) {
                    if can_geminate(next) {
                        // the note before goes on into っ's note, which ends in the long consonant
                        let nucleus = prev.nucleus.clone();
                        out.push(Syllable { text: src.clone(), nucleus, melisma: Some(true), ..Default::default() });
                        geminate = true;
                    } else {
                        // nothing to hold: a glottal stop cuts the voice, and holds through っ's note
                        if out[h].coda.is_empty() {
                            out[h].coda.push(Segment::new("ʔ"));
                        }
                        out.push(Syllable { text: src.clone(), nucleus: Segment::new("ʔ"), ..Default::default() });
                        head = Some(out.len() - 1);
                        sings_vowel = false;
                    }
                }
                last_vowel = None;
                after_nasal = false;
            }
            Mora::Break => {
                last_vowel = None;
                geminate = false;
                after_nasal = false;
            }
        }
    }
    out
}

/// The Japanese front end: kana, by the rules above (no data needed).
pub struct Japanese;

impl FrontEnd for Japanese {
    fn starts(&self, c: char) -> bool {
        // the half-width long-vowel mark ｰ sits among the half-width kana, but like ー it only
        // continues a word: a stray one started a word that gave no notes, and the next word was
        // read in its place ("в ｰ лесу́" lost the в)
        is_kana(c) && c != '\u{FF70}'
    }
    fn continues(&self, c: char) -> bool {
        is_kana(c) || is_mark(c)
    }
    fn word(&self, word: &str) -> Vec<Syllable> {
        word_syllables(word)
    }
}
