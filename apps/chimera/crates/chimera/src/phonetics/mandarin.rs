//! The Mandarin front end: Han characters → sung syllables, one per character, so a character maps
//! to one note and keeps its spelling as the note's text.
//!
//! - Readings come from a lexicon (`public/data/mandarin.txt.gz`, built by tools/build-mandarin.ts
//!   from MIT pinyin data): each character's most common reading, and the phrases whose reading
//!   differs from their characters' own. A run of characters is read by longest phrase match,
//!   which decides most polyphones (银行 yin2 hang2, 行走 xing2 zou3).
//! - Tone sandhi inside the run: 不 before a fourth tone is second (不会 bu2 hui4); 一 before a
//!   fourth tone is second and before the others fourth; a run of third tones keeps the last and
//!   turns the others second (你好 ni2 hao3).
//! - Pinyin → IPA on the inventory. Stops and affricates say their aspiration, which is what
//!   Mandarin contrasts: b d g z zh j are unaspirated (p˭ t˭ k˭ ts˭ tʂ˭ tɕ˭), p t k c ch q aspirated.
//! - Tones are read, not sung: a song follows its melody. They decide the sandhi above and the
//!   neutral tone's reduced vowel, and wait for speech to be drawn as pitch.
//!
//! [`Pinyin`] reads Mandarin typed as pinyin instead, with no lexicon: a Latin word that carries
//! tones (marks, "nǐhǎo", or numbers, "ni3 hao3") and parses as pinyin. Toneless words stay
//! English.

use super::syllables::FrontEnd;
use super::{Segment, Syllable};
use std::collections::HashMap;

pub fn is_han(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}')
}

/// Readings: each character's most common, and the phrases that read otherwise.
#[derive(Default)]
pub struct MandarinLexicon {
    chars: HashMap<char, String>,
    phrases: HashMap<String, Vec<String>>,
    longest: usize,
}

impl MandarinLexicon {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load "字<TAB>zhong1" and "银行<TAB>yin2 hang2" lines (later lines win).
    pub fn load(&mut self, text: &str) {
        for line in text.lines() {
            let Some((w, py)) = line.split_once('\t') else { continue };
            let syls: Vec<String> = py.split(' ').filter(|s| !s.is_empty()).map(str::to_string).collect();
            let chars: Vec<char> = w.chars().collect();
            if chars.len() == 1 && syls.len() == 1 {
                self.chars.insert(chars[0], syls[0].clone());
            } else if chars.len() == syls.len() && chars.len() > 1 {
                self.longest = self.longest.max(chars.len());
                self.phrases.insert(w.to_string(), syls);
            }
        }
    }

    pub fn is_loaded(&self) -> bool {
        !self.chars.is_empty()
    }

    /// A run of Han characters → one numbered pinyin syllable per character ("" where unknown),
    /// by longest phrase match, then tone sandhi.
    pub fn read(&self, run: &str) -> Vec<String> {
        self.read_with_source(run).into_iter().map(|(p, _)| p).collect()
    }

    /// [`read`](Self::read), with whether each reading came from a phrase (rather than the
    /// character's own most common reading).
    fn read_with_source(&self, run: &str) -> Vec<(String, bool)> {
        let cs: Vec<char> = run.chars().collect();
        let mut out: Vec<String> = Vec::with_capacity(cs.len());
        let mut from_phrase: Vec<bool> = Vec::with_capacity(cs.len());
        let mut i = 0;
        while i < cs.len() {
            let mut took = false;
            for n in (2..=self.longest.min(cs.len() - i)).rev() {
                let w: String = cs[i..i + n].iter().collect();
                if let Some(syls) = self.phrases.get(&w) {
                    out.extend(syls.iter().cloned());
                    from_phrase.extend(std::iter::repeat_n(true, n));
                    i += n;
                    took = true;
                    break;
                }
            }
            if !took {
                out.push(self.chars.get(&cs[i]).cloned().unwrap_or_default());
                from_phrase.push(false);
                i += 1;
            }
        }
        // after a degree word a polyphone is the adjective (很长 hěn cháng, not the default zhǎng of
        // 长大 and 校长); a phrase still decides first
        for i in 1..cs.len() {
            let Some(&(_, adjective)) = AFTER_DEGREE.iter().find(|(c, _)| *c == cs[i]) else { continue };
            let before: String = cs[i.saturating_sub(2)..i].iter().collect();
            if !from_phrase[i] && DEGREE_WORDS.iter().any(|w| before.ends_with(w)) {
                out[i] = adjective.to_string();
            }
        }
        sandhi(&cs, &mut out);
        out.into_iter().zip(from_phrase).collect()
    }
}

/// Polyphones that are adjectives after a degree word, where the character's default reading isn't
/// (长's default zhǎng is the more frequent reading in running text: 长大, 校长, 成长).
const AFTER_DEGREE: &[(char, &str)] = &[('长', "chang2")];
/// Words of degree before an adjective (多么 and 那么, not a bare 么: 怎么长 is "how does it grow").
const DEGREE_WORDS: &[&str] = &[
    "很", "太", "真", "挺", "好", "多", "更", "最", "极", "蛮", "超", "够", "越", "那么", "这么", "多么", "非常", "特别", "比较",
    "有点", "这样", "那样",
];

fn tone(py: &str) -> u8 {
    py.chars().last().and_then(|c| c.to_digit(10)).map_or(5, |d| d as u8)
}
fn set_tone(py: &mut String, t: u8) {
    if py.ends_with(|c: char| c.is_ascii_digit()) {
        py.pop();
        py.push(char::from(b'0' + t));
    }
}

fn sandhi(cs: &[char], py: &mut [String]) {
    let n = py.len();
    // 不 and 一 by the character: pinyin alone can't tell 不 bu4 from 部 bu4
    for i in 0..n {
        let next = if i + 1 < n { Some(tone(&py[i + 1])) } else { None };
        match (cs[i], next) {
            ('不', Some(4)) => set_tone(&mut py[i], 2),
            ('一', Some(4)) => set_tone(&mut py[i], 2),
            ('一', Some(1..=3)) => set_tone(&mut py[i], 4),
            _ => {}
        }
    }
    third_tones(py);
}

/// A run of third tones: all but the last become second (你好 ni2 hao3).
fn third_tones(py: &mut [String]) {
    let n = py.len();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && tone(&py[j]) == 3 && !py[j].is_empty() {
            j += 1;
        }
        for k in i..j.saturating_sub(1) {
            set_tone(&mut py[k], 2);
        }
        i = j.max(i + 1);
    }
}

/// Fraction of a vowel nasalised before -n/-ng.
const NASAL_LEAD: f64 = 0.6;

/// A rhyme's ending (the off-glide of ai ei ao ou, a nasal coda) is one closing movement at the end
/// of the syllable, this long at most (s): Mandarin's 韵尾 is where the vowel ends up, not a second
/// sound (`Syllable::closing`).
const CLOSING: f64 = 0.2;

/// Mandarin's sung timing: a nasal coda is a short murmur at the end of a closing rhyme. The place
/// cue is in the vowel's last formant movement (the song planner moves the tongue body toward the
/// coda over the vowel's last 150 ms). Held 130 and 100 ms after a vowel that didn't move, as they
/// were, a native listener heard a second sound: 昂 as an + ng.
fn coda_dur(ipa: &str) -> Option<f64> {
    match ipa {
        "ŋ" => Some(0.08),
        "n" => Some(0.07),
        _ => None,
    }
}

/// Mandarin's sung onset timing, where the defaults blurred contrasts: an unaspirated affricate's
/// frication short (j heard as q), nasal onsets long enough to hear (n heard as y), f long enough
/// to tell from sh.
fn onset_dur(ipa: &str, aspirated: Option<bool>) -> Option<f64> {
    match (ipa, aspirated) {
        ("tɕ" | "tʂ" | "ts", Some(false)) => Some(0.08),
        ("n" | "m", _) => Some(0.09),
        ("f", _) => Some(0.13),
        _ => None,
    }
}

/// Aspiration length of Mandarin's aspirated stops and affricates (s).
const MANDARIN_VOT: f64 = 0.09;

/// The consonants before a sung vowel take at most this share of the time since the vowel before
/// (`Syllable::onset_share`). Real singers' initials are a share of that time whatever the tempo
/// (114,000 syllable pairs of M4Singer's 20 singers: the median 0.39-0.43 between 50 and 400 ms,
/// three quarters under 0.50-0.55), 60 ms long on the fastest notes and 160 on slow ones. At a
/// fixed length they left a fast note no vowel: after an aspirated initial, 85 % of the vowels
/// under 150 ms were 12 dB or more under the singer's.
///
/// M4Singer: Zhang et al., "M4Singer: A Multi-Style, Multi-Singer and Musical Score Provided
/// Mandarin Singing Corpus", NeurIPS 2022 Datasets and Benchmarks (CC BY-NC-SA 4.0). Only
/// aggregate figures like these come from it: none of its audio, annotations or alignments is
/// in this repository.
const ONSET_SHARE: f64 = 0.55;

/// How long after an initial's own segment the voice starts (s): the aspiration; an unaspirated
/// stop's burst (11-17 ms measured on the renders) or affricate's (18); a fricative's release
/// (25-34; [ɕ]'s is 3). The song planner starts the vowel that long before the beat
/// (`Syllable::voice_lag`): sung with the vowel's segment on the beat, the voice came in 35-45 ms
/// after it and 110 ms after an aspirated initial, where singers' is on it.
fn voice_lag(ipa: &str, aspirated: Option<bool>) -> Option<f64> {
    match (ipa, aspirated) {
        (_, Some(true)) => Some(MANDARIN_VOT),
        ("p" | "t" | "k", Some(false)) => Some(0.015),
        ("tɕ" | "tʂ" | "ts", Some(false)) => Some(0.018),
        ("f" | "x" | "ʂ", _) => Some(0.027),
        ("s", _) => Some(0.034),
        _ => None,
    }
}

/// An initial → its sound and, for stops and affricates, its aspiration.
fn initial(i: &str) -> Option<(&'static str, Option<bool>)> {
    Some(match i {
        "b" => ("p", Some(false)),
        "p" => ("p", Some(true)),
        "m" => ("m", None),
        "f" => ("f", None),
        "d" => ("t", Some(false)),
        "t" => ("t", Some(true)),
        "n" => ("n", None),
        "l" => ("l", None),
        "g" => ("k", Some(false)),
        "k" => ("k", Some(true)),
        "h" => ("x", None),
        "j" => ("tɕ", Some(false)),
        "q" => ("tɕ", Some(true)),
        "x" => ("ɕ", None),
        "zh" => ("tʂ", Some(false)),
        "ch" => ("tʂ", Some(true)),
        "sh" => ("ʂ", None),
        "r" => ("ɻ", None),
        "z" => ("ts", Some(false)),
        "c" => ("ts", Some(true)),
        "s" => ("s", None),
        _ => return None,
    })
}

/// A final (after its initial, with y/w spellings undone) → glide, nucleus, coda.
fn final_parts(f: &str) -> Option<(&'static [&'static str], &'static str, &'static [&'static str])> {
    const N: &[&str] = &[];
    Some(match f {
        "a" => (N, "a", N),
        "o" => (&["w"], "ɔ", N),
        "e" => (N, "ɤ", N),
        "i" => (N, "i", N),
        "u" => (N, "u̠", N),
        "v" => (N, "y", N),
        "er" => (N, "ɝ", N),
        "ai" => (N, "aɪ", N),
        "ei" => (N, "eɪ", N),
        "ao" => (N, "aʊ̠", N),
        "ou" => (N, "əʊ̠", N),
        "an" => (N, "a", &["n"]),
        "en" => (N, "ə", &["n"]),
        "ang" => (N, "ɑ", &["ŋ"]),
        "eng" => (N, "ɤ", &["ŋ"]),
        "ong" => (N, "ʊ̠", &["ŋ"]),
        "ia" => (&["j"], "a", N),
        // yo (哟 唷): an interjection's [jɔ]
        "io" => (&["j"], "ɔ", N),
        "ie" => (&["j"], "ɛ", N),
        "iao" => (&["j"], "aʊ̠", N),
        "iu" | "iou" => (&["j"], "əʊ̠", N),
        "ian" => (&["j"], "ɛ", &["n"]),
        "in" => (N, "i", &["n"]),
        "iang" => (&["j"], "ɑ", &["ŋ"]),
        "ing" => (N, "i", &["ŋ"]),
        "iong" => (&["j"], "ʊ̠", &["ŋ"]),
        "ua" => (&["w"], "a", N),
        "uo" => (&["w"], "ɔ", N),
        "uai" => (&["w"], "aɪ", N),
        "ui" | "uei" => (&["w"], "eɪ", N),
        "uan" => (&["w"], "a", &["n"]),
        "un" | "uen" => (&["w"], "ə", &["n"]),
        "uang" => (&["w"], "ɑ", &["ŋ"]),
        "ueng" => (&["w"], "ə", &["ŋ"]),
        "ve" | "ue" => (&["ɥ"], "ɛ", N),
        "van" => (&["ɥ"], "ɛ", &["n"]),
        "vn" => (N, "y", &["n"]),
        // syllabic nasals (嗯 n, ng, m)
        "n" | "ng" | "m" | "hm" | "hng" => (N, "ə", &["n"]),
        _ => return None,
    })
}

/// One numbered pinyin syllable ("zhong1") → a sung syllable with its spelling `text`; None when
/// it isn't pinyin.
pub fn pinyin_syllable(py: &str, text: &str) -> Option<Syllable> {
    let t = tone(py);
    let body = py.trim_end_matches(|c: char| c.is_ascii_digit());
    // erhua spelled in pinyin: a trailing r on any syllable but er itself (nǎr, diǎnr)
    if body.len() > 1 && body.ends_with('r') && body != "er" {
        let mut s = pinyin_syllable(&format!("{}{t}", &body[..body.len() - 1]), text)?;
        rhotacize(&mut s);
        return Some(s);
    }
    let ini = ["zh", "ch", "sh"].into_iter().find(|i| body.starts_with(i)).or_else(|| {
        let c = &body[..body.chars().next()?.len_utf8()];
        (initial(c).is_some() || c == "y" || c == "w").then_some(c)
    });
    let (ini, mut fin) = match ini {
        Some(i) if body.len() > i.len() => (i, body[i.len()..].to_string()),
        _ => ("", body.to_string()),
    };
    // spellings: y/w stand for glides; u after j q x (and y) is ü
    match ini {
        "y" => {
            fin = match fin.as_str() {
                "i" | "in" | "ing" => fin,
                f if f.starts_with('u') => format!("v{}", &f[1..]),
                f if f.starts_with('i') => f.to_string(),
                f => format!("i{f}"),
            }
        }
        "w" => {
            fin = match fin.as_str() {
                "u" => fin,
                "o" => "uo".into(),
                f => format!("u{f}"),
            }
        }
        "j" | "q" | "x" if fin.starts_with('u') => fin = format!("v{}", &fin[1..]),
        _ => {}
    }
    let apical = matches!(ini, "z" | "c" | "s" | "zh" | "ch" | "sh" | "r") && fin == "i";
    // the apical vowels: the consonant's tongue held open (ɿ after z c s, ʅ after zh ch sh r)
    let apical_vowel = if matches!(ini, "z" | "c" | "s") { "ɿ" } else { "ʅ" };
    let (glide, mut nucleus, coda) = if apical { (&[][..], apical_vowel, &[][..]) } else { final_parts(&fin)? };
    // er is a central vowel (a, for the fourth tone: 二 is more open) moving into the r-colouring, not
    // English's /ɝ/ held still: the start is the quick on-glide, the r-coloured vowel holds the note
    if fin == "er" {
        nucleus = if t == 4 { "aɝ" } else { "əɝ" };
    }
    // a neutral-tone syllable's e reduces (的 de, 了 le)
    if t == 5 && nucleus == "ɤ" {
        nucleus = "ə";
    }
    // a medial glide belongs to the rhyme: glide + vowel as one rising diphthong, sung on the beat
    let mut glide = glide;
    let rising = match (glide, nucleus) {
        (["j"], "a") => Some("ia"),
        (["j"], "ɛ") => Some("iɛ"),
        (["j"], "ɑ") => Some("iɑ"),
        (["j"], "ʊ̠") => Some("iʊ̠"),
        (["w"], "a") => Some("u̠a"),
        (["w"], "ɔ") => Some("u̠ɔ"),
        (["w"], "ɑ") => Some("u̠ɑ"),
        (["w"], "ə") => Some("u̠ə"),
        (["ɥ"], "ɛ") => Some("yɛ"),
        // the three-vowel finals: the glide into a falling diphthong (iao, iou, uai, uei)
        (["j"], "aʊ̠") => Some("iaʊ̠"),
        (["j"], "əʊ̠") => Some("iəʊ̠"),
        (["w"], "aɪ") => Some("u̠aɪ"),
        (["w"], "eɪ") => Some("u̠eɪ"),
        _ => None,
    };
    if let Some(r) = rising {
        nucleus = r;
        glide = &[];
    }
    let mut onset: Vec<Segment> = Vec::new();
    // a syllable spelled with no initial (爱 ài, 安 ān, 饿 è; not y-/w-) starts with a soft glottal
    // catch, which keeps it from running into the vowel before
    if ini.is_empty() && !matches!(fin.as_str(), "n" | "ng" | "m" | "hm" | "hng") {
        onset.push(Segment::new("ʔ"));
    }
    if let Some((ipa, aspirated)) = initial(ini) {
        // Mandarin's aspirated stops and affricates aspirate for ~80-100 ms, longer than English's
        let vot = (aspirated == Some(true)).then_some(MANDARIN_VOT);
        onset.push(Segment { aspirated, vot, dur: onset_dur(ipa, aspirated), ..Segment::new(ipa) });
    }
    onset.extend(glide.iter().map(|g| Segment::new(g)));
    Some(Syllable {
        text: text.to_string(),
        onset,
        nucleus: Segment::new(nucleus),
        coda: coda.iter().map(|c| Segment { dur: coda_dur(c), ..Segment::new(c) }).collect(),
        // vowels before -n/-ng are nasalised. (Lighter neutral tones, breath 0.7 on 的 了, sounded
        // more natural but made them harder to recognise in the scramble test, so they stay full.)
        nasal_lead: (!coda.is_empty()).then_some(NASAL_LEAD),
        closing: Some(CLOSING),
        voice_lag: initial(ini).and_then(|(ipa, aspirated)| voice_lag(ipa, aspirated)),
        onset_share: Some(ONSET_SHARE),
        ..Default::default()
    })
}

/// 儿 after these is always erhua (哪儿 nǎr, 这儿, 那儿), whatever the lexicon says.
const ERHUA_AFTER: &str = "哪这那";
/// 儿 after these is its own syllable (女儿 nǚ'ér, 婴儿, 孤儿, 托儿所): words for people.
const ER_SYLLABLE_AFTER: &str = "女婴孤幼男健宠胎孙侄托";

/// Erhua: the syllable ends rhotic instead (一点儿 diǎnr, 哪儿 nǎr). A nasal coda gives way to
/// the r, and a falling diphthong keeps its first vowel (块儿 kuàir → [kʰwaɻ]).
fn rhotacize(s: &mut Syllable) {
    // the off-glide gives way to the r: aɪ → a, uaɪ → ua (a rising nucleus keeps its on-glide)
    let nuc = s.nucleus.ipa.clone();
    if super::planner::offglide(&nuc).is_some() {
        let held = super::planner::held_vowel(&nuc);
        let rising = super::planner::onglide(&nuc).map(|g| format!("{g}{held}"));
        match rising {
            // glide + vowel as a rising diphthong the planner knows (uai → ua)
            Some(r) if super::planner::is_rising(&r) => s.nucleus = Segment::new(&r),
            // otherwise the glide goes back to the onset (uei → w + e)
            Some(_) => {
                let g = super::planner::onglide(&nuc).unwrap();
                s.onset.push(Segment::new(match g { "i" => "j", "y" => "ɥ", _ => "w" }));
                s.nucleus = Segment::new(held);
            }
            None => s.nucleus = Segment::new(held),
        }
    }
    s.coda = vec![Segment::new("ɻ")];
    s.nasal_lead = None;
}

/// The Mandarin front end, over a loaded lexicon (without one it reads nothing).
pub struct Mandarin<'a>(pub &'a MandarinLexicon);

impl FrontEnd for Mandarin<'_> {
    fn starts(&self, c: char) -> bool {
        is_han(c)
    }
    fn word(&self, run: &str) -> Vec<Syllable> {
        let cs: Vec<char> = run.chars().collect();
        let mut out: Vec<Syllable> = Vec::new();
        for (i, ((py, phrase), c)) in self.0.read_with_source(run).iter().zip(&cs).enumerate() {
            // erhua: 儿 after another syllable is its rhotic ending (一会儿 er5, 块儿, 哪儿), unless it's
            // a word's own syllable (女儿, 婴儿, or a phrase reading it with a tone): lyrics' 儿 is
            // mostly erhua. The data marks 哪儿/这儿/那儿 toned and lacks 婴儿-type words, hence the lists.
            let prev = if i > 0 { Some(cs[i - 1]) } else { None };
            let erhua = *c == '儿'
                && prev.is_some_and(|p| ERHUA_AFTER.contains(p) || (!ER_SYLLABLE_AFTER.contains(p) && (py == "er5" || !phrase)));
            if erhua {
                if let Some(prev) = out.last_mut() {
                    rhotacize(prev);
                    prev.text.push('儿');
                    continue;
                }
            }
            out.extend(pinyin_syllable(py, &c.to_string()));
        }
        out
    }
}

/// Mandarin typed as pinyin: a Latin word that carries tones, as marks ("nǐhǎo", "xièxie") or numbers
/// ("ni3", "hao3", "de5"), and parses fully as pinyin. Syllables split at numbers and apostrophes
/// ("xi'an"), otherwise by the longest valid syllable; an unmarked syllable in a marked word is
/// neutral ("xièxie" xie4 xie5). Third-tone sandhi applies inside a word; 不 and 一 are written with
/// the tone they're sung with (bú, yí), since pinyin alone can't tell 不 from 部. A word without tones
/// isn't claimed, so English reads it.
pub struct Pinyin;

/// A tone-marked vowel → (its letter, the tone).
fn tone_mark(c: char) -> Option<(char, u8)> {
    const MARKS: [(&str, char); 6] = [("āáǎà", 'a'), ("ēéěè", 'e'), ("īíǐì", 'i'), ("ōóǒò", 'o'), ("ūúǔù", 'u'), ("ǖǘǚǜ", 'v')];
    MARKS.iter().find_map(|(m, b)| m.chars().position(|x| x == c).map(|k| (*b, k as u8 + 1)))
}

/// Every Mandarin syllable (toneless pinyin, ü as v), sorted: the readings in the lexicon
/// (public/data/mandarin.txt.gz), so typed pinyin is checked against real syllables (no "fe" in café).
#[rustfmt::skip]
const SYLLABLES: &[&str] = &[
    "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi",
    "bian", "biao", "bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce",
    "cen", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng", "chi", "chong",
    "chou", "chu", "chua", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou",
    "cu", "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "dei", "den",
    "deng", "di", "dia", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du", "duan", "dui",
    "dun", "duo", "e", "ei", "en", "eng", "er", "fa", "fan", "fang", "fei", "fen", "feng", "fiao",
    "fo", "fou", "fu", "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong", "gou",
    "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha", "hai", "han", "hang", "hao",
    "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun",
    "huo", "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan",
    "jue", "jun", "ka", "kai", "kan", "kang", "kao", "ke", "ken", "keng", "kong", "kou", "ku",
    "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai", "lan", "lang", "lao", "le",
    "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "long", "lou",
    "lu", "luan", "lun", "luo", "lv", "lve", "m", "ma", "mai", "man", "mang", "mao", "me", "mei",
    "men", "meng", "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu", "n", "na",
    "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie",
    "nin", "ning", "niu", "nong", "nou", "nu", "nuan", "nun", "nuo", "nv", "nve", "o", "ou", "pa",
    "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping",
    "po", "pou", "pu", "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu",
    "qu", "quan", "que", "qun", "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou",
    "ru", "rua", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang", "sao", "se", "sen",
    "seng", "sha", "shai", "shan", "shang", "shao", "she", "shei", "shen", "sheng", "shi", "shou",
    "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun", "shuo", "si", "song", "sou", "su",
    "suan", "sui", "sun", "suo", "ta", "tai", "tan", "tang", "tao", "te", "teng", "ti", "tian",
    "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo", "wa", "wai",
    "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian", "xiang", "xiao", "xie",
    "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun", "ya", "yan", "yang", "yao", "ye",
    "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue", "yun", "za", "zai", "zan",
    "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao", "zhe",
    "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui",
    "zhun", "zhuo", "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo",
];

fn valid_syllable(base: &str) -> bool {
    // a real syllable, or one with an erhua r (nar, dianr)
    let real = |b: &str| SYLLABLES.binary_search(&b).is_ok();
    let erhua = base.len() > 1 && base != "er" && base.strip_suffix('r').is_some_and(real);
    (real(base) || erhua) && pinyin_syllable(&format!("{base}1"), "").is_some()
}

/// Syllables often unstressed and toneless inside a word: particles and suffixes (de le me men zi
/// tou ge ba ma ne), the negation and 一 in a verb pattern (bu yi), directional complements (lai qu
/// shang xia guo), and words' second halves that are neutral (you huan xi shi hou fu mian bian li
/// zhe). Written without a tone inside a toned word, they are neutral.
const NEUTRAL_INSIDE: &[&str] = &[
    "ba", "bian", "bu", "de", "fu", "ge", "guo", "hou", "huan", "lai", "le", "li", "ma", "me", "men", "mian", "ne", "qu",
    "shang", "shi", "tou", "xi", "xia", "yi", "you", "zhe", "zi",
];

/// A word → (spelling, numbered pinyin) per syllable; None if it carries no tone or isn't pinyin.
fn parse_pinyin(word: &str) -> Option<Vec<(String, String)>> {
    // pieces end at a tone number or an apostrophe; each letter keeps its spelling and its mark
    let mut pieces: Vec<(Vec<(char, char, u8)>, Option<u8>)> = vec![(Vec::new(), None)];
    let mut toned = false;
    // each letter is read lowercased but keeps its spelling for the note's text (Xī'ān, not xī'ān)
    for shown in word.chars() {
        let c = shown.to_lowercase().next().unwrap_or(shown);
        let last = pieces.last_mut().unwrap();
        if let Some(d) = c.to_digit(10) {
            if d > 5 || last.0.is_empty() {
                return None;
            }
            last.1 = Some(if d == 0 { 5 } else { d as u8 });
            toned = true;
            pieces.push((Vec::new(), None));
        } else if c == '\'' {
            pieces.push((Vec::new(), None));
        } else if let Some((b, t)) = tone_mark(c) {
            last.0.push((shown, b, t));
            toned = true;
        } else if c == 'ü' || c.is_ascii_lowercase() {
            last.0.push((shown, if c == 'ü' { 'v' } else { c }, 0));
        } else {
            return None;
        }
    }
    if !toned {
        return None;
    }
    let mut out = Vec::new();
    // whether each syllable's tone was written (a mark or a number), and its letters
    let mut written: Vec<(bool, String)> = Vec::new();
    for (letters, digit) in pieces.into_iter().filter(|p| !p.0.is_empty()) {
        let base: String = letters.iter().map(|l| l.1).collect();
        let cuts = split_syllables(&base)?;
        let mut start = 0;
        for (n, &end) in cuts.iter().enumerate() {
            let span = &letters[start..end];
            let marks: Vec<u8> = span.iter().map(|l| l.2).filter(|&t| t > 0).collect();
            if marks.len() > 1 {
                return None;
            }
            let number = if n + 1 == cuts.len() { digit } else { None };
            written.push((!marks.is_empty() || number.is_some(), base[start..end].to_string()));
            let tone = marks.first().copied().or(number).unwrap_or(5);
            let mut text: String = span.iter().map(|l| l.0).collect();
            if n + 1 == cuts.len() && digit.is_some() && marks.is_empty() {
                text.push(char::from(b'0' + tone));
            }
            out.push((text, format!("{}{tone}", &base[start..end])));
            start = end;
        }
    }
    // a toned word starts with a written tone, and an unwritten one later is neutral: at the end
    // (xièxie, dōngxi), or inside as one of the syllables that are often neutral (zěnmeyàng,
    // háizimen, duìbuqǐ). English words with accents break this: résumé (su unwritten inside),
    // touché (tou unwritten first)
    let last = written.len() - 1;
    let neutral_inside = |b: &str| NEUTRAL_INSIDE.contains(&b);
    if !written[0].0 || written.iter().enumerate().any(|(i, (w, b))| !w && i > 0 && i < last && !neutral_inside(b)) {
        return None;
    }
    Some(out)
}

/// The longest valid syllable first, backing off when the rest can't be split. Positions the rest
/// already failed from are remembered, so a long word that can't be split fails in linear time (it
/// was exponential: a 30-letter typo took seconds, and Sing re-reads the lyrics on every keystroke).
fn split_syllables(base: &str) -> Option<Vec<usize>> {
    fn go(base: &str, at: usize, out: &mut Vec<usize>, failed: &mut [bool]) -> bool {
        if at == base.len() {
            return true;
        }
        if failed[at] {
            return false;
        }
        for end in (at + 1..=base.len().min(at + 6)).rev() {
            let s = &base[at..end];
            // pinyin's spelling: a syllable starting with a, o or e after another needs an apostrophe
            // (Xī'ān), so née isn't né + e; the syllabic nasals are whole words (嗯 n), so Pokémon
            // isn't po ké mo + n; and an erhua r ends a word (xīnr isn't xi + nr)
            if at > 0 && matches!(s.as_bytes()[0], b'a' | b'o' | b'e') {
                continue;
            }
            let nasal = matches!(s.trim_end_matches('r'), "n" | "ng" | "m" | "hm" | "hng");
            if nasal && !(at == 0 && end == base.len()) {
                continue;
            }
            if s.len() > 1 && s != "er" && s.ends_with('r') && end != base.len() {
                continue;
            }
            if valid_syllable(s) {
                out.push(end);
                if go(base, end, out, failed) {
                    return true;
                }
                out.pop();
            }
        }
        failed[at] = true;
        false
    }
    let mut out = Vec::new();
    let mut failed = vec![false; base.len()];
    go(base, 0, &mut out, &mut failed).then_some(out)
}

impl FrontEnd for Pinyin {
    fn starts(&self, c: char) -> bool {
        c.is_ascii_alphabetic() || c.to_lowercase().any(|l| tone_mark(l).is_some() || l == 'ü')
    }
    fn continues(&self, c: char) -> bool {
        self.starts(c) || c.is_ascii_digit() || c == '\''
    }
    fn claims(&self, word: &str) -> bool {
        // a one-letter word is English ("room a1", French "à"); write 啊 and 饿 in characters
        if word.chars().filter(|c| c.is_alphabetic()).count() < 2 {
            return false;
        }
        parse_pinyin(word).is_some()
    }
    fn word(&self, word: &str) -> Vec<Syllable> {
        let Some(sy) = parse_pinyin(word) else { return Vec::new() };
        let mut py: Vec<String> = sy.iter().map(|s| s.1.clone()).collect();
        third_tones(&mut py);
        sy.iter().zip(&py).filter_map(|((text, _), p)| pinyin_syllable(p, text)).collect()
    }
}
