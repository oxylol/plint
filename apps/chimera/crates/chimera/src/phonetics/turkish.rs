//! Turkish text → phoneme segments. Turkish spelling is close to phonemic, so letters map to
//! sounds by rule and no dictionary is needed:
//! - /k g l/ are fronted ([c ɟ l]) next to a front vowel, and before â and û, which mark a fronted
//!   consonant in loans; next to a back vowel they are [k g ɫ]: "kedi" [cedi], "kar" [kaɾ],
//!   "kâr" [caɾ], "yol" [joɫ], "güzel" [ɟyzel].
//! - ğ lengthens the vowel before it ("dağ" [daː]: one vowel, 1.7 times as long), is a glide
//!   between front vowels ("değil" [dejil]) and vanishes between other vowels ("ağaç" [aatʃ]).
//! - A doubled consonant letter is one long consonant ("dikkat" [dikːat]).
//!
//! Stress falls on a word's last syllable unless something earlier claims it (Lees 1961; Sezer
//! 1981; Kabak & Vogel 2001):
//! - a listed word or root keeps its own stress ("şimdi" ŞİMdi, "Ankara'da" ANkarada);
//! - a pre-stressing suffix puts it on the syllable before the leftmost one ("geliyor" geLİyor,
//!   "gelmiyor" GELmiyor, "ederim" eDErim, "gelirken" geLİRken, "arabayla" araBAyla, "evdeki"
//!   evDEki);
//! - other proper names take Sezer's rule: the antepenult when it is heavy and the penult light,
//!   else the penult ("Antalya'da" anTALyada, "Erdoğan" ERdoğan);
//! - clitics take none ("mı", "musun", "de", "ki", "ve").
//!
//! The suffix rules match spelling, not morphology, so a copula ("öğretmenim", I am a teacher:
//! öğretMENim) reads like the possessive (my teacher: öğretmeNİM).
//!
//! Unstressed vowels don't reduce in Turkish, so unlike English there are no weak forms, and every
//! phrase takes statement intonation: a yes/no question peaks on the syllable before "mı" and
//! falls, which the nuclear accent already gives.

use super::durations::inherent_ms;
use super::english::{Phrase, is_phrase_punct, split_keeping};
use super::inventory::phoneme;
use super::mandarin::MandarinLexicon;
use super::syllables::{FrontEnd, Lang, lyrics_in};
use crate::anatomy::human::BodyMix;
use super::{Intonation, Segment, Syllable};

/// A doubled consonant lasts this many times its single inherent duration.
const GEMINATE: f64 = 1.6;
/// A vowel lengthened by ğ lasts this many times what the duration rules give it (as the IPA
/// tokenizer lengthens a vowel marked ː).
const LONG_VOWEL: f64 = 1.7;

fn is_vowel_letter(c: char) -> bool {
    "aeıioöuüâîû".contains(c)
}

/// front vowels (ğ glides between two of them)
fn is_front(c: char) -> bool {
    "eiöüî".contains(c)
}

/// A vowel letter after k, g or l that fronts it: the front vowels, and â and û, which are
/// written after a fronted k, g or l ("kâr", "mahkûm", "lâle").
fn fronts_before(c: char) -> bool {
    "eiöüîâû".contains(c)
}

fn vowel_ipa(c: char) -> &'static str {
    match c {
        'a' | 'â' => "a",
        'e' => "e",
        'ı' => "ɯ",
        'i' | 'î' => "i",
        'o' => "o",
        'ö' => "ø",
        'u' | 'û' => "u",
        _ => "y", // ü
    }
}

/// The sounds of a consonant letter, `front` when its vowel is front (k, g, l).
fn consonant_ipa(c: char, front: bool) -> &'static [&'static str] {
    match c {
        'b' => &["b"],
        'c' => &["dʒ"],
        'ç' => &["tʃ"],
        'd' => &["d"],
        'f' => &["f"],
        'g' if front => &["ɟ"],
        'g' => &["g"],
        'h' => &["h"],
        'j' => &["ʒ"],
        'k' | 'q' if front => &["c"],
        'k' | 'q' => &["k"],
        'l' if front => &["l"],
        'l' => &["ɫ"],
        'm' => &["m"],
        'n' => &["n"],
        'p' => &["p"],
        'r' => &["ɾ"],
        's' => &["s"],
        'ş' => &["ʃ"],
        't' => &["t"],
        'v' | 'w' => &["v"],
        'x' if front => &["c", "s"],
        'x' => &["k", "s"],
        'y' => &["j"],
        'z' => &["z"],
        _ => &[],
    }
}

/// Turkish lower case: I → ı and İ → i (Rust's `to_lowercase` gives "i" and "i̇").
fn tr_lower(c: char) -> char {
    match c {
        'I' => 'ı',
        'İ' => 'i',
        c => c.to_lowercase().next().unwrap_or(c),
    }
}

fn is_letter(c: char) -> bool {
    c.is_ascii_alphabetic() || "çğıöşüâîûÇĞİÖŞÜÂÎÛ".contains(c)
}

/// Whether the consonant letters `w[i..j]` are fronted: by the vowel right after them, else by
/// the nearest vowel before them, else by the first vowel after them (a word-initial cluster).
fn front_context(w: &[char], i: usize, j: usize) -> bool {
    if let Some(&n) = w.get(j)
        && is_vowel_letter(n)
    {
        return fronts_before(n);
    }
    if let Some(&p) = w[..i].iter().rev().find(|c| is_vowel_letter(**c)) {
        return is_front(p);
    }
    w[j.min(w.len())..].iter().find(|c| is_vowel_letter(**c)).is_some_and(|&n| fronts_before(n))
}

/// One sound of a word: the vowel's syllable (its vowel letter's ordinal), a duration for a long
/// consonant, whether a vowel is long (before ğ), and the letter it is written with (a long
/// consonant's second letter, which opens the next syllable).
struct Phone {
    ipa: &'static str,
    syllable: Option<usize>,
    dur: Option<f64>,
    long: bool,
    letter: usize,
}

fn letters_to_phones(w: &[char]) -> Vec<Phone> {
    let mut out = Vec::with_capacity(w.len() + 2);
    let mut nv = 0usize;
    let mut i = 0;
    while i < w.len() {
        let c = w[i];
        if is_vowel_letter(c) {
            out.push(Phone { ipa: vowel_ipa(c), syllable: Some(nv), dur: None, long: false, letter: i });
            nv += 1;
            i += 1;
            continue;
        }
        if c == 'ğ' {
            let prev = i.checked_sub(1).map(|p| w[p]).filter(|p| is_vowel_letter(*p));
            let next = w.get(i + 1).copied().filter(|n| is_vowel_letter(*n));
            match (prev, next) {
                (Some(p), Some(n)) if is_front(p) && is_front(n) => out.push(Phone { ipa: "j", syllable: None, dur: None, long: false, letter: i }),
                // before a consonant or at the end: the vowel before is one long vowel
                (Some(_), None) => {
                    if let Some(v) = out.last_mut() {
                        v.long = true;
                    }
                }
                // between other vowels it vanishes, and after a consonant (no native word has one) too
                _ => {}
            }
            i += 1;
            continue;
        }
        let single = consonant_ipa(c, false).len() == 1;
        let long = single && w.get(i + 1) == Some(&c);
        let j = if long { i + 2 } else { i + 1 };
        let sounds = consonant_ipa(c, front_context(w, i, j));
        if long {
            // the inherent duration the planner would give it (Klatt's table, else the inventory's)
            let single = inherent_ms(sounds[0]).map(|(inh, _)| inh / 1000.0).or_else(|| phoneme(sounds[0], BodyMix::MALE).map(|d| d.dur)).unwrap_or(0.08);
            out.push(Phone { ipa: sounds[0], syllable: None, dur: Some(GEMINATE * single), long: false, letter: i + 1 });
        } else {
            out.extend(sounds.iter().map(|&ipa| Phone { ipa, syllable: None, dur: None, long: false, letter: i }));
        }
        i = j;
    }
    out
}

// ---------------------------------------------------------------- stress

/// Words whose stress isn't on the last syllable (0 = the first vowel). These match whole words
/// only: with a suffix they're another word, or the root is too short to match safely.
#[rustfmt::skip]
const WORDS: &[(&str, usize)] = &[
    ("ama", 0), ("ancak", 0), ("belki", 0), ("çünkü", 0), ("evet", 0), ("fakat", 0), ("hayır", 0),
    ("lütfen", 0), ("merhaba", 0), ("neden", 0), ("niçin", 0), ("niye", 0), ("nereye", 0),
    ("nereden", 0), ("yani", 0), ("masa", 0), ("naber", 0),
    // final after all: "erken" isn't a -ken form, "namaz" isn't a negative aorist, and "müdür"
    // (director) is far commoner than the particle "mü" + "-dür"
    ("erken", 1), ("namaz", 1), ("müdür", 1),
];

/// Roots that keep their stress under suffixes ("şimdiden" ŞİMdiden, "Türkçede" TÜRKçede):
/// adverbs, loans, place names and words in pre-stressing -CA. The longest that starts the word
/// wins.
#[rustfmt::skip]
const ROOTS: &[(&str, usize)] = &[
    ("şimdi", 0), ("yarın", 0), ("sonra", 0), ("önce", 0), ("bugün", 0), ("nasıl", 0), ("nerede", 0),
    ("kimse", 0), ("herkes", 0), ("hepsi", 0), ("efendi", 1), ("anne", 0), ("banka", 0),
    ("lokanta", 1), ("pencere", 0), ("tiyatro", 1), ("radyo", 0), ("gazete", 1),
    ("türkçe", 0), ("ingilizce", 2), ("almanca", 1), ("fransızca", 2), ("arapça", 1), ("rusça", 0),
    ("bence", 0), ("sence", 0), ("bizce", 0), ("sizce", 0),
    ("ankara", 0), ("istanbul", 1), ("izmir", 0), ("türkiye", 0), ("avrupa", 1), ("amerika", 1),
    ("almanya", 1), ("ingiltere", 1), ("fransa", 0), ("italya", 1), ("rusya", 0), ("japonya", 1),
    ("londra", 0), ("antalya", 1),
];

/// Derivational suffixes make a new word with its own final stress, so a root under one of them
/// doesn't keep its stress ("annelik" anneLİK, "Ankaralı" ankaraLI, "bankacı" bankaCI).
const DERIVATIONAL: &[&str] = &["lı", "li", "lu", "lü", "sız", "siz", "suz", "süz", "cı", "ci", "cu", "cü", "çı", "çi", "çu", "çü"];

/// Words that take no stress of their own (they lean on the word before or after).
const CLITICS: &[&str] = &["de", "da", "ki", "ve", "bir"];

/// The question particle and its inflected forms ("mı", "musun", "miydi", "müsünüz"); `I` stands
/// for the particle's own vowel, which every later high vowel repeats.
const PARTICLE_TAILS: &[&str] = &["", "sIn", "yIm", "yIz", "sInIz", "dIr", "ydI", "ydIm", "ydIn", "ydIk", "ydInIz", "ymIş"];

fn is_question_particle(w: &[char]) -> bool {
    if w.len() < 2 || w[0] != 'm' || !"ıiuü".contains(w[1]) {
        return false;
    }
    let v = w[1];
    let tail: String = w[2..].iter().map(|&c| if c == v { 'I' } else { c }).collect();
    PARTICLE_TAILS.contains(&tail.as_str())
}

fn at(w: &[char], i: usize, pat: &str) -> bool {
    // a pattern char matches itself; 'A' matches a or e, 'I' matches ı i u ü, 'K' matches k or ğ
    pat.chars().enumerate().all(|(k, p)| {
        w.get(i + k).is_some_and(|&c| match p {
            'A' => c == 'a' || c == 'e',
            'I' => "ıiuü".contains(c),
            'K' => c == 'k' || c == 'ğ',
            p => c == p,
        })
    })
}

/// What may follow a negative aorist -mAz and still be a verb ("gelmezsin", "olmazdı").
const AORIST_TAILS: &[&str] = &["", "sın", "sin", "sınız", "siniz", "lar", "ler", "dı", "di", "dım", "dim", "ken", "sa", "se"];

/// Where the leftmost pre-stressing suffix starts (a letter index), if any.
fn pre_stressing(w: &[char]) -> Option<usize> {
    let n = w.len();
    let has_vowel_before = |p: usize| w[..p].iter().any(|c| is_vowel_letter(*c));
    let mut best: Option<usize> = None;
    let mut take = |p: usize| {
        if has_vowel_before(p) && best.is_none_or(|b| p < b) {
            best = Some(p);
        }
    };
    for p in 1..n {
        // -Iyor: on the syllable before "yor" ("geliyor" geLİyor, "okuyor" oKUyor)
        if at(w, p, "yor") && is_vowel_letter(w[p - 1]) {
            take(p);
        }
        // the negative -mA: "gelmiyor", "gelmedi", "gelmemiş", "gelmeyecek", "gelmemek", "gelmez"
        if w[p] == 'm' {
            let tail: String = w[p..].iter().collect();
            let aorist = (at(w, p, "maz") || at(w, p, "mez")) && AORIST_TAILS.contains(&&tail[3..]);
            if at(w, p, "mIyor") || at(w, p, "mAdI") || at(w, p, "mAmIş") || at(w, p, "mAyAcAK") || at(w, p, "mAmAk") || aorist {
                take(p);
            }
        }
    }
    // person endings on the aorist ("ederim" eDErim, "gelirsin" geLİRsin, "severiz"), but not
    // after the plural -lAr, where -Im is the possessive ("kitaplarım" kitaplaRIM)
    let aorist_at = |r: usize| r >= 2 && w[r] == 'r' && is_vowel_letter(w[r - 1]) && !(r >= 3 && w[r - 2] == 'l' && "ae".contains(w[r - 1]));
    if n >= 4 && (at(w, n - 2, "Im") || at(w, n - 2, "Iz")) && aorist_at(n - 3) {
        take(n - 2);
    }
    if n >= 5 && at(w, n - 3, "sIn") && aorist_at(n - 4) {
        take(n - 3);
    }
    if n >= 7 && at(w, n - 5, "sInIz") && aorist_at(n - 6) {
        take(n - 5);
    }
    // word-final -ken after the aorist or a vowel ("gelirken", "hastayken")
    if n >= 5 && (at(w, n - 4, "rken") || at(w, n - 4, "yken")) {
        take(n - 3);
    }
    // -(y)lA after a vowel ("arabayla")
    if n >= 4 && (at(w, n - 3, "yla") || at(w, n - 3, "yle")) && is_vowel_letter(w[n - 4]) {
        take(n - 3);
    }
    // relative -ki after a locative ("evdeki", "yarındaki")
    if n >= 5 && (at(w, n - 4, "dAki") || at(w, n - 4, "tAki")) {
        take(n - 2);
    }
    best
}

/// Sezer's rule for proper names: the antepenult if it is heavy (closed) and the penult light,
/// else the penult. Returns a vowel ordinal.
fn sezer(w: &[char]) -> usize {
    let vs: Vec<usize> = w.iter().enumerate().filter(|(_, c)| is_vowel_letter(**c)).map(|(i, _)| i).collect();
    let n = vs.len();
    if n <= 2 {
        return 0;
    }
    // two consonant letters before the next vowel close the syllable
    let heavy = |k: usize| vs[k + 1] - vs[k] > 2;
    if heavy(n - 3) && !heavy(n - 2) { n - 3 } else { n - 2 }
}

/// The ordinal of the stressed vowel, or None for a clitic or a word without vowels.
/// `stem` is where an apostrophe cut a proper name from its suffixes; `proper` says the word is a
/// name.
fn stressed_vowel(w: &[char], stem: usize, proper: bool) -> Option<usize> {
    let vowels = w.iter().filter(|c| is_vowel_letter(**c)).count();
    if vowels == 0 {
        return None;
    }
    let word: String = w.iter().collect();
    // a listed word first: it may look like a particle ("müdür")
    if let Some((_, k)) = WORDS.iter().find(|(x, _)| *x == word) {
        return Some((*k).min(vowels - 1));
    }
    if is_question_particle(w) || CLITICS.contains(&word.as_str()) {
        return None;
    }
    let found = ROOTS
        .iter()
        .filter(|(r, _)| word.strip_prefix(r).is_some_and(|rest| !DERIVATIONAL.iter().any(|d| rest.starts_with(d))))
        .max_by_key(|(r, _)| r.len())
        .map(|(_, k)| *k);
    if let Some(k) = found {
        return Some(k.min(vowels - 1));
    }
    if proper {
        return Some(sezer(&w[..stem]));
    }
    if let Some(p) = pre_stressing(w) {
        return Some(w[..p].iter().filter(|c| is_vowel_letter(**c)).count() - 1);
    }
    Some(vowels - 1)
}

/// A word read: its letters (lower case, and as typed), its sounds, and its stressed vowel.
struct Parsed {
    letters: Vec<char>,
    typed: Vec<char>,
    phones: Vec<Phone>,
    stressed: Option<usize>,
}

impl Parsed {
    fn segment(&self, p: &Phone) -> Segment {
        Segment {
            ipa: p.ipa.to_string(),
            dur: p.dur,
            length: p.long.then_some(LONG_VOWEL),
            stress: Some(u8::from(p.syllable.is_some() && p.syllable == self.stressed)),
            ..Default::default()
        }
    }
}

fn parse(typed: &str, phrase_initial: bool) -> Parsed {
    let mut letters: Vec<char> = Vec::with_capacity(typed.len());
    let mut kept: Vec<char> = Vec::with_capacity(typed.len());
    let mut stem = None;
    for c in typed.chars() {
        if c == '\'' {
            stem.get_or_insert(letters.len());
        } else if is_letter(c) {
            letters.push(tr_lower(c));
            kept.push(c);
        }
    }
    let capital = typed.chars().next().is_some_and(char::is_uppercase);
    let proper = stem.is_some() || (capital && !phrase_initial);
    let stressed = stressed_vowel(&letters, stem.unwrap_or(letters.len()), proper);
    let phones = letters_to_phones(&letters);
    Parsed { letters, typed: kept, phones, stressed }
}

/// One word, as typed, to segments with stress. `phrase_initial`: a capital there doesn't mark a
/// name.
pub fn word(typed: &str, phrase_initial: bool) -> Vec<Segment> {
    let p = parse(typed, phrase_initial);
    p.phones.iter().map(|ph| p.segment(ph)).collect()
}

/// One word's sung syllables, split where Turkish spelling splits them: a consonant before a vowel
/// opens its syllable ("bağ-la-rın-da"). A long vowel before ğ is one note's vowel, and a long
/// consonant is sung on both sides ("dik-kat": the first note closes on it, the next opens with it).
fn sung_syllables(typed: &str, phrase_initial: bool) -> Vec<Syllable> {
    let p = parse(typed, phrase_initial);
    let vowels: Vec<usize> = p.letters.iter().enumerate().filter(|(_, c)| is_vowel_letter(**c)).map(|(i, _)| i).collect();
    if vowels.is_empty() {
        return Vec::new();
    }
    let starts: Vec<usize> = std::iter::once(0).chain(vowels.windows(2).map(|v| if v[1] - v[0] > 1 { v[1] - 1 } else { v[1] })).collect();
    let mut out: Vec<Syllable> = starts
        .iter()
        .enumerate()
        .map(|(k, &a)| {
            let b = starts.get(k + 1).copied().unwrap_or(p.letters.len());
            Syllable { text: p.typed[a..b].iter().collect(), ..Default::default() }
        })
        .collect();
    let mut voiced = vec![false; out.len()];
    for ph in &p.phones {
        let k = starts.iter().rposition(|&a| a <= ph.letter).unwrap_or(0);
        // a note sets its own lengths
        let seg = Segment { dur: None, length: None, ..p.segment(ph) };
        if ph.syllable.is_some() {
            if !voiced[k] {
                out[k].nucleus = seg;
                voiced[k] = true;
            }
        } else if voiced[k] {
            out[k].coda.push(seg);
        } else {
            // a long consonant closes the syllable before and opens this one, as it is written
            // ("dik-kat"); in speech it is one long consonant
            if ph.dur.is_some() && k > 0 {
                out[k - 1].coda.push(seg.clone());
            }
            out[k].onset.push(seg);
        }
    }
    out
}

/// The Turkish front end for the lyrics router ([`super::syllables::read_lyrics`]). Turkish is
/// written in Latin letters, so the router can't tell it from English by script: the player
/// chooses it ([`Lang`]). An apostrophe stays in the word ("Ankara'da": a name and its
/// suffixes), and typed hyphens split a word your way, as in English.
pub struct Turkish;

impl FrontEnd for Turkish {
    fn starts(&self, c: char) -> bool {
        is_letter(c) || c == '\''
    }
    fn continues(&self, c: char) -> bool {
        self.starts(c) || c == '-'
    }
    fn word(&self, tok: &str) -> Vec<Syllable> {
        self.word_after(tok, None)
    }
    /// A capital at the start of a phrase doesn't mark a name; the router passes no word before
    /// the first one of a phrase.
    fn word_after(&self, tok: &str, prev: Option<&str>) -> Vec<Syllable> {
        let mut initial = prev.is_none();
        let mut out = Vec::new();
        for part in tok.split('-').filter(|p| p.chars().any(is_letter)) {
            out.append(&mut sung_syllables(part, initial));
            initial = false;
        }
        out
    }
}

/// Lyrics → sung syllables with Latin script read as Turkish ([`Lang::Tr`]); other scripts are
/// routed as usual, and no dictionary is needed.
pub fn lyrics_to_syllables(text: &str) -> Vec<Syllable> {
    lyrics_in(text, &|_: &str, _: bool| Vec::<Segment>::new(), &MandarinLexicon::new(), Lang::Tr)
}

/// Turkish text → phrases of segments. Phrase punctuation breaks phrases as in English; numbers
/// are read out ("1.923,5" → "bin dokuz yüz yirmi üç virgül beş").
pub fn to_phrases(text: &str) -> Vec<Phrase> {
    let normalized = expand_numbers(&text.replace('’', "'"));
    let mut phrases = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut flush = |delim: &str, words: &mut Vec<String>| {
        let mut segs: Vec<Segment> = Vec::new();
        let mut said: Vec<String> = Vec::new();
        for w in words.drain(..) {
            let mut ws = word(&w, said.is_empty());
            let Some(last) = ws.last_mut() else { continue };
            last.word_end = Some(true);
            segs.append(&mut ws);
            said.push(w.chars().filter(|c| *c == '\'' || is_letter(*c)).map(tr_lower).collect());
        }
        if segs.is_empty() {
            return;
        }
        let pause = if delim.contains(['.', '!', '?']) {
            0.35
        } else if !delim.is_empty() {
            0.18
        } else {
            0.3
        };
        phrases.push(Phrase { segments: segs, intonation: Intonation::Statement, pause, words: said });
    };
    for part in split_keeping(&normalized, is_phrase_punct) {
        if !part.is_empty() && part.chars().all(is_phrase_punct) {
            flush(&part, &mut words);
        } else {
            words.extend(part.split(|c: char| !(is_letter(c) || c == '\'')).map(|w| w.trim_matches('\'')).filter(|w| w.chars().any(is_letter)).map(str::to_string));
        }
    }
    flush("", &mut words);
    phrases
}

// ---------------------------------------------------------------- numbers
const ONES: [&str; 10] = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
const TENS: [&str; 10] = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
const SCALES: [(u64, &str); 4] = [(1_000_000_000_000, "trilyon"), (1_000_000_000, "milyar"), (1_000_000, "milyon"), (1_000, "bin")];

fn below_thousand(n: u64, out: &mut Vec<&'static str>) {
    let (h, t, o) = ((n / 100) as usize, (n / 10 % 10) as usize, (n % 10) as usize);
    // "yüz" and "bin" stand alone for one hundred and one thousand
    if h > 1 {
        out.push(ONES[h]);
    }
    if h > 0 {
        out.push("yüz");
    }
    if t > 0 {
        out.push(TENS[t]);
    }
    if o > 0 {
        out.push(ONES[o]);
    }
}

/// Digits → Turkish number words (digit by digit past the trillions).
fn number_words(digits: &str) -> String {
    let d = digits.trim_start_matches('0');
    if d.is_empty() {
        return ONES[0].into();
    }
    let Some(mut n) = d.parse::<u64>().ok().filter(|n| *n < 1_000_000_000_000_000) else {
        return digits.bytes().map(|b| ONES[(b - b'0') as usize]).collect::<Vec<_>>().join(" ");
    };
    let mut out = Vec::new();
    for (scale, name) in SCALES {
        let q = n / scale;
        if q > 0 {
            if !(scale == 1_000 && q == 1) {
                below_thousand(q, &mut out);
            }
            out.push(name);
            n %= scale;
        }
    }
    below_thousand(n, &mut out);
    out.join(" ")
}

/// Numbers in text → words. A dot before three digits groups thousands ("1.000"), a comma
/// between digits is the decimal comma ("3,05" → "üç virgül sıfır beş").
fn expand_numbers(t: &str) -> String {
    let c: Vec<char> = t.chars().collect();
    let digit = |i: usize| c.get(i).is_some_and(char::is_ascii_digit);
    let mut out = String::with_capacity(t.len());
    let mut i = 0;
    while i < c.len() {
        if !digit(i) {
            out.push(c[i]);
            i += 1;
            continue;
        }
        let mut int = String::new();
        loop {
            while digit(i) {
                int.push(c[i]);
                i += 1;
            }
            if c.get(i) == Some(&'.') && digit(i + 1) && digit(i + 2) && digit(i + 3) && !digit(i + 4) {
                i += 1;
                continue;
            }
            break;
        }
        out.push(' ');
        out.push_str(&number_words(&int));
        if c.get(i) == Some(&',') && digit(i + 1) {
            i += 1;
            let mut frac = String::new();
            while digit(i) {
                frac.push(c[i]);
                i += 1;
            }
            out.push_str(" virgül");
            let zeros = frac.len() - frac.trim_start_matches('0').len();
            for _ in 0..zeros {
                out.push_str(" sıfır");
            }
            if zeros < frac.len() {
                out.push(' ');
                out.push_str(&number_words(&frac[zeros..]));
            }
        }
        out.push(' ');
    }
    out
}
