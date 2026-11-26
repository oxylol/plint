//! Segment sequence → gestural score. Timing and phonation are articulatory: aspiration is a
//! glottal opening gesture peaking at the release (VOT emerges), nasality is velum lowering with
//! anticipation, coarticulation comes from overlapping vowel gestures beneath consonant gestures.
//!
//! Most planners allocate and run off the audio thread. [`plan_timed`] (live notes) and
//! [`oral_gestures_into`] (the beat sequencer) build into a caller's `Vec` and allocate nothing
//! once it has capacity, for up to [`INLINE_SEGMENTS`] segments.

use super::durations::{SegInfo, klatt_durations_with};
use super::inventory::{Glottal, Manner, PhonemeDef, Voicing, dental_closure, phoneme};
use super::{Intonation, Plan, PlannedSegment, Segment};
use crate::anatomy::human::{ArticKey, ArticPatch, BodyMix, BodyName};
use crate::control::score::{ControlKey, Gesture, Targets};
use crate::math::{clamp, exp, js_max, js_min, js_round, pow};
use crate::tuning::{PlannerTuning, ProsodyTuning};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// label of the end-of-breath-group abduction (a live note can drop it when the next one starts)
pub const EXHALE_LABEL: &str = "exhale";
/// The end of a sung phrase on a vowel or sonorant (s): the folds are fully apart this long after
/// the last segment's end, having parted over `LET_GO_OPENING` before that; the breath stays on
/// for `LET_GO_BREATH` after the end. Trained singers' notes fall from -3 to -20 dB in about 50 ms
/// and stop about 90 ms after -3 dB (8,095 releases in VocalSet and GTSinger).
const LET_GO_OPEN_AT: f64 = 0.06;
const LET_GO_OPENING: f64 = 0.15;
const LET_GO_BREATH: f64 = 0.015;
/// the opening's gesture strength: low enough that the folds follow its ramp (the voice takes a
/// gesture's glottal opening in full from a weight of 2, which a strength of 6 reaches at a third
/// of its activation, about 39 % of the way along the cosine ramp's time)
const LET_GO_STRENGTH: f64 = 2.0;
/// The shortest final note (its vowel and the sonorants that close it, s) the folds let go of
/// this way. The opening starts `LET_GO_OPENING - LET_GO_OPEN_AT` (90 ms) before
/// the phrase's end; on a shorter note it would start before the note has sounded and the whole
/// note comes out quiet and breathy. Under twice that lead the phrase ends as it did before.
const LET_GO_ROOM: f64 = 2.0 * (LET_GO_OPENING - LET_GO_OPEN_AT);

/// when a sung phrase's breath reaches full drive, after the plan start (its first segment starts at 0.06)
const BREATH_SUNG_AT: f64 = 0.06;

/// The phrase-start settle (`prosody.onset_settle`, spoken phrases only): the share of the settle
/// by which the first consonant's oral gestures, and a voiceless one's fold gesture, start early.
/// A share, because the settle is the room they have: at 0.8 of a 0.1 s settle the tongue is at its
/// target 30-50 ms before the breath starts to rise (a lead of 0.05 s left /f/'s low swell at −30
/// dB re the vowel, 0.08 took it to −45: the lip is slow). Their ramp-in is the settle itself.
const ONSET_LEAD_SHARE: f64 = 0.8;
/// With a settle, how much later the breath starts before a phrase-initial voiceless fricative
/// (s): the pressure then starts to rise 10 ms before the span and is full 30 ms into it, through a
/// constriction that is already there. Seconds and not a share: it places the breath against the
/// fricative's span, whose own timing the settle does not change.
const ONSET_BREATH_DELAY: f64 = 0.03;
/// `prosody.final_stop_breath`: where the breath ends after the start of a phrase-final /b d g/
/// (s). The closure is made by then; ending it at the start or 20 ms before was tried and no better.
const FINAL_STOP_BREATH: f64 = 0.02;
/// `prosody.cluster_aspiration`: how many times as long as before a vowel the folds stay apart
/// past the release of /p t k/ before /ɹ l w j/. The sonorant's constriction holds the pressure
/// across the folds down and the voice would start early for the same gesture; the recorded men's
/// voice onset is later there (71-81 ms) than before a vowel (48-61).
const CLUSTER_FOLDS: f64 = 1.5;

/// How far a voiced fricative's folds part (cm², scaled with the body in the voice). Frication and
/// voicing share the lung pressure, which takes a glottis about as wide as the constriction
/// (Stevens 1971; Elie & Laprie's "regime B"). Here that is a mean glottal area of about 0.13 cm²,
/// through constrictions of 0.10-0.18. The pressure behind the constriction then rises and the
/// voicing weakens by itself, most in /z/, whose constriction is the narrowest. 0.12-0.14 match the
/// TSP speakers' voicing better; `an_f_does_not_click` stops them (hill-climb log, experiments 23 and 32).
/// The default of `prosody.voiced_fricative_opening`.
pub const VOICED_FRICATIVE_OPENING: f64 = 0.09;
/// /ð/ after a nasal keeps a small opening: the speakers' has no dip there, nasal level throughout, and
/// this reads about their level (hill-climb log, experiment 24). The default of `prosody.dh_opening`.
pub const DH_OPENING: f64 = 0.035;

const VOWEL_KEYS: [ArticKey; 9] = [ArticKey::TbX, ArticKey::TbY, ArticKey::Jaw, ArticKey::LipAp, ArticKey::LipProt, ArticKey::LipWidth, ArticKey::Root, ArticKey::Larynx, ArticKey::TtCurl];

/// One calibrated vowel of `vowelTargets.generated.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct VowelTarget {
    pub artic: ArticPatch,
    #[serde(default)]
    pub formants: Vec<f64>,
}

/// body name → IPA → calibrated vowel
pub type VowelTable = BTreeMap<String, BTreeMap<String, VowelTarget>>;

/// The calibrated vowel table (parsed once).
pub fn vowel_table() -> &'static VowelTable {
    static T: OnceLock<VowelTable> = OnceLock::new();
    T.get_or_init(|| serde_json::from_str(include_str!("vowelTargets.generated.json")).expect("vowelTargets.generated.json"))
}

/// Diphthongs and r-coloured vowels expand to vowel sequences. American /eɪ oʊ/ glide toward
/// [i u] (Hillenbrand et al. 1995: /e/ F2 rises ~150 Hz, /o/ F1 falls with F2 steady); with
/// [ɪ ʊ] as targets they barely moved (/e/ and /ɪ/ share F2) and sounded like the steady
/// [eː oː] of Indian English. The other offglides undershoot toward [ɪ ʊ].
pub fn expand(ipa: &str) -> Option<&'static [&'static str]> {
    Some(match ipa {
        "aɪ" => &["a", "ɪ"],
        "aʊ" => &["a", "ʊ"],
        "ɔɪ" => &["ɔ", "ɪ"],
        "eɪ" => &["e", "i"],
        "oʊ" => &["o", "u"],
        "ɚ" => &["ɝ"],
        // rising diphthongs: a short close glide into the main vowel (Mandarin's medials: 话 ua,
        // 天 iɛ, 学 yɛ), written as one nucleus so the glide sings on the beat, inside the note
        "ia" => &["i", "a"],
        "iɛ" => &["i", "ɛ"],
        "iɑ" => &["i", "ɑ"],
        "iʊ" => &["i", "ʊ"],
        "ua" => &["u", "a"],
        "uo" => &["u", "o"],
        "uɑ" => &["u", "ɑ"],
        "uə" => &["u", "ə"],
        "yɛ" => &["y", "ɛ"],
        // Mandarin's er: a central (er4: open) start, the r-colouring arriving in the held vowel
        "əɝ" => &["ə", "ɝ"],
        "aɝ" => &["a", "ɝ"],
        "uɔ" => &["u", "ɔ"],
        // Mandarin's own back u (Zee & Lee 2001: F2 ~660, English /u/'s is ~1000) as the medial glide:
        // the same compounds with `u̠` (vowel_data.rs), beside the ones above, which stay as they were
        "u̠a" => &["u̠", "a"],
        "u̠ɔ" => &["u̠", "ɔ"],
        "u̠ɑ" => &["u̠", "ɑ"],
        "u̠ə" => &["u̠", "ə"],
        "u̠aɪ" => &["u̠", "a", "ɪ"],
        "u̠eɪ" => &["u̠", "e", "i"],
        // and Mandarin's ʊ (ong, and the off-glide of ao ou iao iou, Zee & Lee: F2 ~730 where English /ʊ/ has ~1120)
        "iʊ̠" => &["i", "ʊ̠"],
        "aʊ̠" => &["a", "ʊ̠"],
        "əʊ̠" => &["ə", "ʊ̠"],
        "iaʊ̠" => &["i", "a", "ʊ̠"],
        "iəʊ̠" => &["i", "ə", "ʊ̠"],
        // a central start, Mandarin's ou ([əʊ], where English [oʊ] starts nearly at [u])
        "əʊ" => &["ə", "ʊ"],
        // triphthongs: a glide into a falling diphthong (Mandarin iao, iou, uai, uei)
        "iaʊ" => &["i", "a", "ʊ"],
        "iəʊ" => &["i", "ə", "ʊ"],
        "uaɪ" => &["u", "a", "ɪ"],
        "ueɪ" => &["u", "e", "i"],
        _ => return None,
    })
}

/// A rising nucleus (a rising diphthong or a triphthong): its first part is a short glide.
pub fn is_rising(ipa: &str) -> bool {
    matches!(ipa, "ia" | "iɛ" | "iɑ" | "iʊ" | "ua" | "uo" | "uɑ" | "uə" | "yɛ" | "uɔ" | "əɝ" | "aɝ" | "iaʊ" | "iəʊ" | "uaɪ" | "ueɪ" | "u̠a" | "u̠ɔ" | "u̠ɑ" | "u̠ə" | "u̠aɪ" | "u̠eɪ" | "iʊ̠" | "iaʊ̠" | "iəʊ̠")
}

/// A rising nucleus's on-glide (its first part), sung short at the note's start.
pub fn onglide(ipa: &str) -> Option<&'static str> {
    if is_rising(ipa) { expand(ipa).map(|p| p[0]) } else { None }
}

/// The vowel a sung note holds: a falling diphthong's first part, a rising nucleus's second, else
/// the vowel itself.
pub fn held_vowel(ipa: &str) -> &str {
    match expand(ipa) {
        Some(p) if p.len() >= 2 => p[usize::from(is_rising(ipa))],
        _ => ipa,
    }
}

/// What a held note glides off to at its end: a falling diphthong's (or a triphthong's) last part.
pub fn offglide(ipa: &str) -> Option<&'static str> {
    match expand(ipa) {
        Some(p) if p.len() == 3 || (p.len() == 2 && !is_rising(ipa)) => Some(p[p.len() - 1]),
        _ => None,
    }
}

/// The two vowel targets of a falling diphthong ("aɪ" → ("a", "ɪ"): held, then glided to), or None.
/// Rising diphthongs (Mandarin's "ua") and triphthongs aren't pairs of this kind: see [`onglide`],
/// [`held_vowel`] and [`offglide`].
pub fn diphthong_parts(ipa: &str) -> Option<(&'static str, &'static str)> {
    match expand(ipa) {
        Some(p) if p.len() == 2 && !is_rising(ipa) => Some((p[0], p[1])),
        _ => None,
    }
}

pub fn is_vowel(ipa: &str, body: BodyMix) -> bool {
    let t = vowel_table();
    body.weights().iter().any(|(b, w)| *w > 0.0 && t.get(b.as_str()).is_some_and(|t| t.contains_key(ipa))) || expand(ipa).is_some()
}

/// The calibrated articulation of a vowel for a body: between presets, the presets' calibrated
/// articulations blended by the body's weights.
pub fn vowel_artic(ipa: &str, body: BodyMix) -> Option<ArticPatch> {
    if let Some(b) = body.preset() {
        return preset_vowel_artic(ipa, b);
    }
    let w = body.weights();
    let mut parts = [(0.0, ArticPatch::default()); 3];
    for (i, (b, wi)) in w.iter().enumerate() {
        if *wi > 0.0 {
            parts[i] = (*wi, preset_vowel_artic(ipa, *b)?);
        }
    }
    Some(ArticPatch::blend(&parts))
}

fn preset_vowel_artic(ipa: &str, body: BodyName) -> Option<ArticPatch> {
    let t = vowel_table();
    let v = t.get(body.as_str()).and_then(|b| b.get(ipa)).or_else(|| t.get("male").and_then(|m| m.get(ipa)))?;
    let mut out = ArticPatch::default();
    for k in VOWEL_KEYS {
        if let Some(x) = v.artic.get(k) {
            out.set(k, x);
        }
    }
    // a rhotic's tip is raised and curled back, not resting behind the teeth: without its
    // calibrated position the tip stayed down and /ɝ/'s F3 sat ~15–20 % high (weakly rhotic). The
    // apical vowels are made with the tongue tip, at the place of the sibilant before them (ʅ
    // curled, ɿ not): their tip is part of the vowel too
    if v.artic.get(ArticKey::TtCurl).unwrap_or(0.0) > 0.05 || ipa == "ɿ" {
        if let Some(x) = v.artic.get(ArticKey::TtX) {
            out.set(ArticKey::TtX, x);
        }
        if let Some(x) = v.artic.get(ArticKey::TtY) {
            out.set(ArticKey::TtY, x);
        }
    }
    Some(out)
}

/// Parse the vowel and consonant tables now (so nothing initialises lazily on the audio thread).
pub fn init() {
    vowel_table();
    super::inventory::init();
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanOptions {
    pub body: BodyMix,
    /// voice-clock start time (s)
    pub t0: f64,
    /// speaking-rate multiplier (>1 = faster)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate: Option<f64>,
    /// baseline pitch (Hz)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f0: Option<f64>,
    /// "statement" (final fall) or "question" (final rise)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intonation: Option<Intonation>,
    /// pitch excursion for stressed syllables (fraction)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accent: Option<f64>,
    /// where a phrase's pitch starts, in semitones above `f0` (see [`PITCH_START`])
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_start: Option<f64>,
    /// how far the pitch falls over a phrase, in semitones (see [`PITCH_FALL`])
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_fall: Option<f64>,
    /// the last segment is held open-ended (a sung note): no end-of-breath release
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_end: Option<bool>,
    /// a sung phrase (`plan_song`): consonants given a pitch hold it, and pitch ramps shrink to
    /// fit short notes. Speech and live notes leave it unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sung: Option<bool>,
    /// the planner's and the durations' numbers; unset is the default tuning. An explicit
    /// `accent`, `pitch_start` or `pitch_fall` above wins over the tuning's. With the tuning's
    /// `prosody.upstream_line` on, the pitch is planned as before `PITCH_START` and `PITCH_FALL`
    /// existed, and `pitch_start` and `pitch_fall` are ignored here too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tuning: Option<ProsodyTuning>,
}

/// Declination: a spoken phrase starts `PITCH_START` semitones above its base pitch and has fallen
/// `PITCH_FALL` semitones by its end, with the accents on top. Fitted to the TSP men reading the
/// Harvard sentences (9 training speakers, 540 sentences; semitones re each sentence's median
/// pitch): their first fifth sits at +3.1, their last at −1.6, a line through a sentence falls 5.8,
/// and the 10th and 90th percentiles are −2.1 and +3.7. With 5 and 6.5 the male voice reads +3.0,
/// −1.8 or so, 5.8, −2.5 and +3.0 (docs/hill-climb/log.md, P2). The old plan fell 10 % in a
/// straight line (1.8 semitones) from the base pitch. These six are the defaults of
/// `prosody.pitch_start` and the rest; the planner reads the tuning.
pub const PITCH_START: f64 = 5.0;
pub const PITCH_FALL: f64 = 6.5;
/// How far a stressed vowel's pitch rises above the line (a fraction of it).
pub const ACCENT: f64 = 0.14;
/// How early the fall comes: the line falls as 1 − e^(−PITCH_CURVE · progress), so at 3 it has
/// made 82 % of its fall by the phrase's middle. Towards 0 it is a straight line.
pub const PITCH_CURVE: f64 = 3.0;
/// The share of a stressed vowel its accent's peak lasts.
pub const ACCENT_PEAK: f64 = 0.5;
/// A statement's last vowel ends this many semitones below where the line ends: the final low.
/// (The old plan's 0.82 of the base pitch was 1.6 below its line's end; the men's last fifth sits
/// nearer their median than that.)
pub const FINAL_DROP: f64 = 1.0;

/// JS truthiness of an optional number (`if (s.f0)`: 0 and NaN are false).
#[inline]
pub(crate) fn truthy(x: Option<f64>) -> bool {
    matches!(x, Some(v) if v != 0.0 && !v.is_nan())
}

/// `Partial<HumanArtic>` as gesture targets
#[inline]
pub(crate) fn artic_targets(p: &ArticPatch) -> Targets {
    let mut t = Targets::default();
    for (k, x) in p.iter() {
        t.set(k, x);
    }
    t
}

#[inline]
fn one(k: impl Into<ControlKey>, x: f64) -> Targets {
    Targets::default().with(k, x)
}

/// A planner segment: the input segment plus what the planner attaches to it.
/// Segments [`plan_timed`] handles without allocating (a live note has at most 19: a legato
/// triphthong's off-glide and 8 coda consonants, 8 onset consonants, an on-glide and the vowel).
pub const INLINE_SEGMENTS: usize = 24;

/// A segment whose duration is settled: what [`plan_timed`] plans from. The marks are
/// [`Segment`]'s.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimedSegment<'a> {
    pub ipa: &'a str,
    /// duration (s), before the speaking rate
    pub dur: f64,
    pub stress: Option<u8>,
    pub f0: Option<f64>,
    pub aspirated: Option<bool>,
    pub vot: Option<f64>,
    pub nasalized: Option<bool>,
    pub palatalized: Option<bool>,
    pub breathy: Option<bool>,
    pub glide: Option<f64>,
}

/// What [`plan_timed`] returns besides the gestures.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// end of the last segment (s)
    pub end: f64,
    /// start and end of the last segment (s)
    pub last: (f64, f64),
}

struct Seg<'a> {
    ipa: &'a str,
    dur: f64,
    stress: Option<u8>,
    f0: Option<f64>,
    def: Option<PhonemeDef>,
    vowel: Option<ArticPatch>,
    glide_from: bool,
    aspirated: Option<bool>,
    vot: Option<f64>,
    nasalized: Option<bool>,
    palatalized: Option<bool>,
    breathy: Option<bool>,
    /// a glide off the vowel before it, reached over this long (s): see `Segment::glide`
    glide: Option<f64>,
}

/// glottal abduction of breathy voice: /ɦ/'s, a gap the folds still vibrate across
const BREATHY_ABDUCTION: f64 = 0.1;
/// how long past its release a breathy-voiced stop's folds stay apart (s), at most to the end of
/// the next segment
const BREATHY_RELEASE: f64 = 0.08;

/// A trill's strikes, `TRILL_HZ` a second: the share of each cycle spent at the place, the
/// ramps (s), the articulator speed, and how far (cm) the tongue is pulled back between strikes.
/// The tongue is too slow (13 Hz) to strike 26 times a second and reopen: at 26 Hz it hovered
/// near the ridge with the voicing all but gone; and above speed 6 it overshot and bounced.
const TRILL_HZ: f64 = 20.0;
const TRILL_STRIKE: f64 = 0.45;
const TRILL_RAMP: f64 = 0.004;
const TRILL_SPEED: f64 = 3.5;
const TRILL_OPEN: f64 = 1.6;

/// where a trill's articulator is pulled back to between strikes: away from the place by
/// `TRILL_OPEN` (cm) for the tongue, by an open lip gap for the lips
fn trill_open(closed: &ArticPatch) -> ArticPatch {
    let mut open = ArticPatch::default();
    for (k, v) in closed.iter() {
        match k {
            ArticKey::TtY | ArticKey::TbY => {
                open.set(k, v - TRILL_OPEN);
            }
            ArticKey::LipAp => {
                open.set(k, v + 0.4);
            }
            _ => {}
        }
    }
    open
}

/// velum port area (cm²) of a nasalized vowel, and the ramps of its lowering and raising (s)
const NASAL_VOWEL_PORT: f64 = 0.3;
const NASAL_VOWEL_RAMP: f64 = 0.04;

#[derive(Clone, Copy)]
struct Span {
    t0: f64,
    t1: f64,
}

pub fn plan_utterance(input: &[Segment], o: &PlanOptions) -> Plan {
    let body = o.body;
    let tuning = o.tuning.unwrap_or_default();
    // durations by the Klatt rules, on whole segments (diphthongs before they are split)
    let info = |s: &Segment| {
        if is_vowel(&s.ipa, body) {
            return SegInfo { vowel: true, ..Default::default() };
        }
        let d = phoneme(&s.ipa, body);
        SegInfo { vowel: false, manner: d.map(|d| d.manner), voiced: Some(d.is_some_and(|d| d.voicing == Voicing::Voiced)), dur: d.map(|d| d.dur) }
    };
    let durs = klatt_durations_with(input, info, &tuning.durations);
    let timed = input.iter().zip(&durs).map(|(s, d)| TimedSegment {
        ipa: &s.ipa,
        dur: *d,
        stress: s.stress,
        f0: s.f0,
        aspirated: s.aspirated,
        vot: s.vot,
        nasalized: s.nasalized,
        palatalized: s.palatalized,
        breathy: s.breathy,
        glide: s.glide,
    });
    let mut gestures = Vec::new();
    let mut segments = Vec::with_capacity(input.len());
    let timing = plan_timed(timed, o, &mut gestures, Some(&mut segments));
    Plan { gestures, segments, end: timing.end }
}

/// [`plan_utterance`] after the durations are known: appends the gestures to `gestures` and, if
/// given, the timeline to `planned` (which allocates, for its strings).
pub fn plan_timed<'a>(input: impl Iterator<Item = TimedSegment<'a>>, o: &PlanOptions, gestures: &mut Vec<Gesture>, mut planned: Option<&mut Vec<PlannedSegment>>) -> Timing {
    let rate = o.rate.unwrap_or(1.0);
    let body = o.body;
    // the fold gestures' and the pitch's numbers (the durations are settled by now)
    let tn = o.tuning.unwrap_or_default().prosody;
    // what this call adds, for the speaking rate's shortening below
    let first_gesture = gestures.len();
    // expand diphthongs into two vowel targets sharing the duration
    let mut segs: SmallVec<[Seg; INLINE_SEGMENTS]> = SmallVec::new();
    for s in input {
        if let Some(parts) = expand(s.ipa) {
            let base = s.dur;
            // shares of the duration: a falling diphthong 55/45, a rising one 30/70, a triphthong 25/45/30
            let shares: &[f64] = match (parts.len(), is_rising(s.ipa)) {
                (2, false) => &[0.55, 0.45],
                (2, true) => &[0.3, 0.7],
                (3, _) => &[0.25, 0.45, 0.3],
                _ => &[1.0],
            };
            for (i, p) in parts.iter().enumerate() {
                let dur = base * shares.get(i).copied().unwrap_or(1.0);
                segs.push(Seg { ipa: p, dur, stress: s.stress, f0: s.f0, def: None, vowel: None, glide_from: i > 0, aspirated: None, vot: None, nasalized: s.nasalized, palatalized: None, breathy: s.breathy, glide: None });
            }
        } else {
            segs.push(Seg { ipa: s.ipa, dur: s.dur, stress: s.stress, f0: s.f0, def: None, vowel: None, glide_from: false, aspirated: s.aspirated, vot: s.vot, nasalized: s.nasalized, palatalized: s.palatalized, breathy: s.breathy, glide: s.glide });
        }
    }
    for s in segs.iter_mut() {
        let v = vowel_artic(s.ipa, body);
        if v.is_some() {
            s.vowel = v;
        } else {
            s.def = phoneme(s.ipa, body);
        }
    }
    for i in 0..segs.len() {
        // (a pause: the utterance's start, or an explicit one, the `_` of IPA's `|` or a comma)
        let after_pause_or_obstruent = i == 0 || segs[i - 1].ipa == "_" || segs[i - 1].def.is_some_and(|d| matches!(d.manner, Manner::Stop | Manner::Affricate | Manner::Fricative));
        if segs[i].ipa == "ð" && after_pause_or_obstruent {
            // /ð/ after a pause, a stop, an affricate or a fricative is a silent dental closure and a
            // weak burst (Zhao 2010; the TSP speakers, hill-climb log, experiment 24). Its span stays Klatt's.
            let d = segs[i].def.as_mut().unwrap();
            d.manner = Manner::Stop;
            d.targets = dental_closure();
        }
    }
    let open_end = o.open_end == Some(true);
    let sung = o.sung == Some(true);
    // the phrase-start settle (spoken phrases): the phrase starts that much later, and its first
    // consonant is placed in the room this leaves (below). Off, `start` is `o.t0` itself. Not for a
    // note held open (a live note, which must not sound later than it is played).
    let settle = if !sung && !open_end && tn.onset_settle > 0.0 { tn.onset_settle } else { 0.0 };
    let start = if settle > 0.0 { o.t0 + settle } else { o.t0 };
    let onset_lead = ONSET_LEAD_SHARE * settle;
    // a spoken phrase that ends in /b d g/ (not the stop-like /ð/): `prosody.final_stop_breath`, `final_stop_hold`
    let final_voiced_stop = !sung && !open_end && segs.last().and_then(|l| l.def.as_ref()).is_some_and(|d| d.manner == Manner::Stop && d.voicing == Voicing::Voiced && d.ipa != "ð");
    let mut t = start + 0.06; // breath onset lead
    let mut times: SmallVec<[Span; INLINE_SEGMENTS]> = SmallVec::new();
    for s in &segs {
        let d = s.dur / rate;
        times.push(Span { t0: t, t1: t + d });
        t += d;
    }
    let end = t;
    // ---- vowel gestures: overlap neighbours so the tongue travels vowel-to-vowel
    let v_idx: SmallVec<[usize; INLINE_SEGMENTS]> = segs.iter().enumerate().filter(|(_, s)| s.vowel.is_some()).map(|(i, _)| i).collect();
    for (n, &i) in v_idx.iter().enumerate() {
        let prev = if n > 0 { times[v_idx[n - 1]].t1 } else { times[i].t0 - 0.08 };
        let next = if n < v_idx.len() - 1 { times[v_idx[n + 1]].t0 } else { times[i].t1 + 0.08 };
        // vowel-to-vowel through the consonant (Öhman): the previous vowel lets go early in the gap
        // and the next one's target ramps in over most of it, so the tongue body is already near
        // the next vowel at the release and the transitions carry the consonant's place
        let mut r_in = js_max(0.03, 0.75 * (times[i].t0 - prev)) + 0.01;
        let mut r_out = js_max(0.03, 0.4 * (next - times[i].t1)) + 0.01;
        // a glide off the vowel right before it (`Segment::glide`): it comes in over its given time
        // and that vowel lets go over the glide's own length, so the tongue is on its way from the
        // start of the one ramp to the end of the other and holds neither target in between
        if let Some(g) = segs[i].glide.filter(|_| i > 0 && segs[i - 1].vowel.is_some()) {
            r_in = js_max(0.01, g);
        }
        if i + 1 < segs.len() && segs[i + 1].vowel.is_some() && segs[i + 1].glide.is_some() {
            r_out = js_max(0.01, times[i + 1].t1 - times[i + 1].t0);
        }
        gestures.push(Gesture::new(times[i].t0, times[i].t1, r_in, r_out, 1.0, artic_targets(segs[i].vowel.as_ref().unwrap())).label(segs[i].ipa));
    }
    // utterance-initial/final consonants: lend them the nearest vowel's body posture
    // ---- consonant gestures
    for (i, s) in segs.iter().enumerate() {
        let Span { t0, t1 } = times[i];
        let dur = t1 - t0;
        if let Some(planned) = planned.as_deref_mut() {
            planned.push(PlannedSegment {
                ipa: s.ipa.to_string(),
                t0,
                t1,
                stress: if s.vowel.is_some() { Some(s.stress.unwrap_or(0)) } else { None },
                kind: if s.vowel.is_some() { "vowel".to_string() } else { s.def.map_or("?", |d| d.manner.as_str()).to_string() },
            });
        }
        if s.vowel.is_some() && !s.glide_from {
            // what a vowel's marks hold for the whole vowel, through a diphthong's glide
            let end = times[(i + 1..segs.len()).take_while(|&j| segs[j].glide_from).last().unwrap_or(i)].t1;
            if s.nasalized == Some(true) {
                // nasalized: the velum is down
                gestures.push(Gesture::new(t0, end, NASAL_VOWEL_RAMP, NASAL_VOWEL_RAMP, 4.0, one(ArticKey::Velum, NASAL_VOWEL_PORT)).label("velum"));
            }
            if s.breathy == Some(true) {
                // breathy: the folds part-open, as for /ɦ/
                gestures.push(Gesture::new(t0, end, 0.03, 0.03, 5.0, one(ControlKey::ABDUCTION, BREATHY_ABDUCTION)).label("breathy"));
            }
        }
        let Some(def) = s.def.as_ref() else { continue };
        let next_vowel = i < segs.len() - 1 && segs[i + 1].vowel.is_some();
        let prev_seg = if i > 0 { Some(&segs[i - 1]) } else { None };
        // the first consonant's lead with a phrase-start settle (0.0 without one, and for the rest)
        let lead0 = if i == 0 { onset_lead } else { 0.0 };
        // a spoken phrase's final /b d g/ is held shut this much longer (`prosody.final_stop_hold`)
        let hold = if i == segs.len() - 1 && final_voiced_stop { tn.final_stop_hold } else { 0.0 };
        let oral_from = gestures.len();
        let mut oral_to = None;
        let mut add_oral = |tg: &ArticPatch, a: f64, b: f64, speed: f64, label: &str| oral_gestures_into(def, tg, a, b, speed, label, gestures);
        match def.manner {
            // a stop that closes behind another stop's closure, its articulator already placed by
            // the segment before that (the tip raised for /ɹ/ in "sharp tree"), reaches contact
            // sooner than from a vowel: with the usual lead it sealed ~20 ms before the front
            // closure opened, and the tongue then pumped the air shut between the two closures
            // The delay leaves the closure 30 ms before its release (its 10 ms minimum hold and
            // its ~20 ms early release), so a short sung stop isn't pushed past its own end.
            Manner::Stop if pre_placed_behind_closure(&segs, i) => add_oral(&def.targets, t0 + js_min(PLACED_CLOSURE_DELAY, js_max(0.0, dur - 0.03)), t1 + hold, 1.55, def.ipa),
            Manner::Stop => add_oral(&def.targets, t0 + js_min(stop_closure_delay(def.place, prev_seg), js_max(0.0, dur - 0.03)), t1 + hold, 1.55, def.ipa),
            Manner::Nasal => add_oral(&def.targets, t0 + js_min(nasal_closure_delay(def.place), js_max(0.0, dur - 0.03)), t1, 1.55, def.ipa),
            Manner::Tap => add_oral(&def.targets, t0, t1, 2.8, def.ipa),
            Manner::Trill => {
                // driven trill: the articulator strikes the place and is pulled back between
                // strikes, so the voice carries on through it. Each strike is its own short gesture
                // with tight ramps: with a stop's 30 ms ramp-in and 18 ms ramp-out the strikes
                // overlapped into one held near-closure and the voice stopped for ~90 ms (Spanish
                // "perro" sounded wrong; tests/trill.rs)
                let n = js_max(2.0, js_round(dur * TRILL_HZ));
                let mut k = 0.0;
                while k < n {
                    let a = t0 + (k / n) * dur;
                    let b = a + TRILL_STRIKE * dur / n;
                    gestures.push(Gesture::new(a, b, TRILL_RAMP, TRILL_RAMP, 40.0, artic_targets(&def.targets)).speed(TRILL_SPEED).release_hold(TRILL_RAMP).label(def.ipa));
                    if k + 1.0 < n {
                        gestures.push(Gesture::new(b + 2.0 * TRILL_RAMP, a + dur / n - TRILL_RAMP, TRILL_RAMP, TRILL_RAMP, 40.0, artic_targets(&trill_open(&def.targets))).speed(TRILL_SPEED).label(def.ipa));
                    }
                    k += 1.0;
                }
                oral_to = Some(oral_from); // a trill's strikes are not placed early
            }
            Manner::Affricate => {
                let split = t0 + dur * tn.affricate_closure;
                add_oral(&def.targets, t0, split, 2.0, def.ipa);
                oral_to = Some(gestures.len()); // the closure's gestures; the release keeps its time
                // the release is the fricative's own gesture, not a contact release's: it takes
                // the fricative's articulators (the jaw too) and holds to the span's end, where
                // the folds have closed (below). Its 8 ms lead hands over from the closure. A span
                // squeezed under 33 ms (a fast sung note) starts it with the span, not before.
                if let Some(rel) = def.release.and_then(|r| phoneme(r, body)) {
                    oral_gestures_into(&rel, &rel.targets, js_max(t0, split - 0.015), t1, 1.4, def.ipa, gestures);
                }
            }
            Manner::Fricative | Manner::Approximant | Manner::Lateral => add_oral(&def.targets, t0, t1, 1.3, def.ipa),
            Manner::Glottal | Manner::Vowel => {}
        }
        // the phrase-start settle: the first consonant's articulators set out early, slowly and at
        // their own speed, so they are in place before the breath comes instead of closing on it.
        // Their t0 stays at least 0.029 s after the plan's start (0.06 less the longest `early` of
        // `oral_gestures_into`, plus the settle's other fifth: `ONSET_LEAD_SHARE` is under 1); the
        // ramp reaches back before it, as the first vowel's 70 ms ramp does without a settle.
        if lead0 > 0.0 {
            let oral_to = oral_to.unwrap_or(gestures.len());
            for g in &mut gestures[oral_from..oral_to] {
                g.t0 -= lead0;
                g.ramp_in = settle;
                g.speed = None;
            }
        }
        // nasality: the velum lowers in anticipation (into the preceding vowel, as in English). The
        // port opens about 37 ms before the span and seals about 29 ms after it, as its lag isn't
        // compensated. 0.45 cm² is ample for the murmur. Before a voiced stop it seals about 44 ms
        // after: the velum's offset is later than before a voiceless one (Beddor 2007), and the
        // speakers' murmur before /d b g/ falls 10-20 ms later than one sealed at 29 (hill-climb log,
        // experiments 25 and 26).
        if def.nasal == Some(true) {
            let before_voiced_stop = segs.get(i + 1).and_then(|n| n.def.as_ref()).is_some_and(|d| d.manner == Manner::Stop && d.voicing == Voicing::Voiced);
            let seal = if before_voiced_stop { t1 - 0.005 } else { t1 - 0.02 };
            gestures.push(Gesture::new(t0 - 0.02, seal, 0.05, 0.03, 4.0, one(ArticKey::Velum, 0.45)).label("velum"));
        }
        // phonation
        let k2 = 1.0; // abduction is body-scaled in the voice
        let mut voiced_again = t1; // when the folds are back, after an aspirated release
        if def.glottal == Some(Glottal::Open) {
            // the folds close 10 ms before the span's end: with the lungs' port fixed, /h/ is 4-5 dB louder and MFA
            // reads it about that much longer than the speakers' (hill-climb log, experiment 32)
            gestures.push(Gesture::new(t0 - tn.h_lead - lead0, t1 - tn.h_end_early, tn.h_ramp_in, tn.h_ramp_out, 5.0, one(ControlKey::ABDUCTION, tn.h_opening * k2)).label("h"));
        } else if def.glottal == Some(Glottal::Breathy) {
            gestures.push(Gesture::new(t0, t1, 0.03, 0.03, 5.0, one(ControlKey::ABDUCTION, 0.1 * k2)).label("ɦ"));
        } else if def.glottal == Some(Glottal::Stop) {
            gestures.push(Gesture::new(t0, t1, 0.012, 0.015, 8.0, one(ControlKey::CLOSURE, 1.0)).label("ʔ"));
        } else if def.voicing == Voicing::Voiceless {
            if def.manner == Manner::Affricate && s.aspirated.is_none() && s.vot.is_none() {
                // apart from the closure on, and together again as the frication ends: in an
                // affricate the frication is the voice onset time, with no [h] after it, and the
                // vowel keeps its planned length. A front end that sets `aspirated` or `vot`
                // (Mandarin's aspirated affricates) gets the stop's gesture below instead.
                let mut g = affricate_folds(t0 + tn.affricate_folds_open_at * dur, t1, &tn).label("aspiration");
                voiced_again = g.end();
                g.t0 -= lead0;
                gestures.push(g);
            } else if def.manner == Manner::Stop || def.manner == Manner::Affricate {
                // as the front end says; by default English's rule: aspirated when syllable-initial
                // before a vowel and not after /s/
                // `prosody.cluster_aspiration`: so too before /ɹ l w j/ and a vowel
                let cluster = tn.cluster_aspiration
                    && !sung
                    && s.aspirated.is_none()
                    && s.vot.is_none()
                    && def.manner == Manner::Stop
                    && !prev_seg.is_some_and(|p| p.ipa == "s")
                    && segs.get(i + 1).is_some_and(|n| matches!(n.ipa, "ɹ" | "l" | "w" | "j"))
                    && segs.get(i + 2).is_some_and(|n| n.vowel.is_some());
                let aspirated = s.aspirated.unwrap_or_else(|| next_vowel && !prev_seg.is_some_and(|p| p.ipa == "s")) || cluster;
                // `prosody.vot_unstressed`: the share of the voice onset time before an unstressed vowel
                let vowel_after = segs.get(i + if cluster { 2 } else { 1 }).filter(|n| n.vowel.is_some());
                let share = if !sung && aspirated && s.vot.is_none() && vowel_after.is_some_and(|n| !matches!(n.stress, Some(1 | 2))) { tn.vot_unstressed } else { 1.0 };
                // the folds stay apart past the release by 40% of the default 50 ms (the ramps
                // make up the rest of the aspiration); a front end's longer or shorter `vot` moves
                // that by the whole difference, so the voice onset time it asks for is what it gets
                let past = match (aspirated, s.vot) {
                    (true, None) if cluster => tn.vot / rate * tn.vot_fold_share * share * CLUSTER_FOLDS,
                    (true, None) => tn.vot / rate * tn.vot_fold_share * share, // as before, to the bit (the share is 1.0)
                    (true, Some(v)) => js_max(0.0, tn.vot_fold_share * tn.vot + v - tn.vot) / rate,
                    (false, _) => tn.unaspirated_vot * tn.vot_fold_share,
                };
                let (ramp_out, opening) = if aspirated { (tn.aspirated_ramp_out, tn.aspirated_opening) } else { (tn.unaspirated_ramp_out, tn.unaspirated_opening) };
                // a shortened aspiration closes and opens as a stop with that voice onset time would:
                // the unaspirated stop's ramp and opening at its 12 ms, the aspirated one's at `vot`
                let (ramp_out, opening) = if share != 1.0 {
                    let a = clamp((share * tn.vot - tn.unaspirated_vot) / (tn.vot - tn.unaspirated_vot), 0.0, 1.0);
                    (tn.unaspirated_ramp_out + a * (ramp_out - tn.unaspirated_ramp_out), tn.unaspirated_opening + a * (opening - tn.unaspirated_opening))
                } else {
                    (ramp_out, opening)
                };
                let mut g = Gesture::new(t0 + tn.stop_folds_open_at * dur, t1 + past, tn.stop_ramp_in, ramp_out, 6.0, one(ControlKey::ABDUCTION, opening)).label("aspiration");
                voiced_again = g.end();
                g.t0 -= lead0;
                gestures.push(g);
            } else {
                gestures.push(Gesture::new(t0 - tn.voiceless_lead - lead0, t1 + tn.voiceless_lag, tn.voiceless_ramp_in, tn.voiceless_ramp_out, 6.0, one(ControlKey::ABDUCTION, tn.voiceless_opening)).label("voiceless"));
            }
        } else if def.ipa == "ð" && def.manner == Manner::Stop {
            // apart behind the closure, so it is silent and the voice follows the release by 15-20 ms
            // as the speakers' does. The opening is /v z/'s constant, not a measured one (experiment 24)
            gestures.push(Gesture::new(t0 - 0.015, t1, 0.03, 0.03, 6.0, one(ControlKey::ABDUCTION, tn.voiced_fricative_opening)).label("stop-like ð"));
        } else if def.manner == Manner::Fricative {
            // the folds start to part when a voiceless fricative's do, and close as the constriction
            // opens (about 20 ms after t1, for both)
            let after_nasal = prev_seg.and_then(|p| p.def.as_ref()).is_some_and(|d| d.manner == Manner::Nasal);
            let opening = if def.ipa == "ð" && after_nasal { tn.dh_opening } else { tn.voiced_fricative_opening };
            gestures.push(Gesture::new(t0 - tn.voiced_fricative_lead, t1, tn.voiced_fricative_ramp_in, tn.voiced_fricative_ramp_out, 3.0, one(ControlKey::ABDUCTION, opening)).label("voiced frication"));
        } else if def.manner == Manner::Affricate {
            // the speakers' /dʒ/ differs from their /tʃ/ at its ends, not its frication: voice carries into
            // the closure and fades there, and the frication is as loud and as voiceless as /tʃ/'s
            // (hill-climb log, experiment 29). So the same folds, parted only for the release.
            gestures.push(affricate_folds(t0 + dur * tn.affricate_closure, t1, &tn).label("voiced affricate"));
        }
        // a breathy-voiced release (Hindi bʱ): the folds part as the closure lets go and stay
        // apart into the start of the vowel
        if s.breathy == Some(true) && def.voicing == Voicing::Voiced && matches!(def.manner, Manner::Stop | Manner::Affricate) {
            let until = js_min(t1 + BREATHY_RELEASE, times.get(i + 1).map_or(f64::MAX, |n| n.t1));
            gestures.push(Gesture::new(t1 - 0.02, until, 0.03, 0.03, 5.0, one(ControlKey::ABDUCTION, BREATHY_ABDUCTION)).label("breathy"));
        }
        // palatalization: the tongue body goes toward /j/ under the consonant, stays there until the
        // voice returns, and lets go into the vowel, so the release carries the [j]-like transition
        if s.palatalized == Some(true) {
            if let Some(tg) = palatal_body(def, body) {
                gestures.push(Gesture::new(t0, voiced_again, PALATAL_RAMP, PALATAL_RAMP, PALATAL_STRENGTH, artic_targets(&tg)).label("palatalized"));
            }
        }
    }
    // ---- breath: on through the utterance. A sung phrase's onset is planned (its first vowel
    // lands on the beat), so a singer times the breath to it: the pressure reaches full drive as
    // the first segment starts, and builds into adducted folds or behind the first constriction
    // rather than voicing ahead of the note. Speech and live notes keep their 20 ms lead.
    let breath_at = if sung { BREATH_SUNG_AT } else { 0.04 };
    let last = segs.last();
    let sonorant = last.is_some_and(|l| l.vowel.is_some()) || last.and_then(|l| l.def).is_some_and(|d| matches!(d.manner, Manner::Nasal | Manner::Approximant | Manner::Lateral | Manner::Tap | Manner::Trill));
    // how long the phrase's last note sounds at its end: back from the end over the sonorants that
    // close it, then over its vowel, and over a vowel right before that only at the same pitch (a
    // diphthong's two halves; the note before in a melisma or a hiatus has its own pitch and is
    // not this note). No vowel there: the closing sonorants alone (a hummed or syllabic ending).
    let closing = |g: &Seg| g.vowel.is_none() && g.def.is_some_and(|d| matches!(d.manner, Manner::Nasal | Manner::Approximant | Manner::Lateral | Manner::Tap | Manner::Trill));
    let mut from = segs.len();
    while from > 0 && closing(&segs[from - 1]) {
        from -= 1;
    }
    if from > 0 && segs[from - 1].vowel.is_some() {
        let pitch = segs[from - 1].f0;
        from -= 1;
        while from > 0 && segs[from - 1].vowel.is_some() && segs[from - 1].f0 == pitch {
            from -= 1;
        }
    }
    let room = if from < segs.len() { end - times[from].t0 } else { 0.0 };
    // a sung phrase that ends on a vowel or sonorant is let go by the folds, with the breath still
    // on, where the last voiced stretch is long enough to keep its tone until the opening starts
    let let_go = sung && sonorant && !open_end && room >= LET_GO_ROOM;
    let breath_end = if let_go { end + LET_GO_BREATH } else { end };
    // `prosody.final_stop_breath`: before a final /b d g/ the breath stops as the closure is made,
    // so the stop lets go on a falling pressure and not into the exhale's open folds
    let breath_end = if tn.final_stop_breath && final_voiced_stop { times[times.len() - 1].t0 + FINAL_STOP_BREATH } else { breath_end };
    // with a settle the breath comes later before a voiceless fricative, whose folds are apart by then
    let voiceless_fricative = |d: &PhonemeDef| d.manner == Manner::Fricative && d.voicing == Voicing::Voiceless && d.glottal.is_none();
    let breath_at = if settle > 0.0 && segs.first().and_then(|f| f.def.as_ref()).is_some_and(voiceless_fricative) { breath_at + ONSET_BREATH_DELAY } else { breath_at };
    gestures.push(Gesture::new(start + breath_at, breath_end, 0.04, if open_end { 0.07 } else { 0.05 }, 1.0, one(ControlKey::BREATH, 1.0)).label("breath"));
    // ---- end of the breath group: the folds part as the breath stops, so a final obstruent is
    // released into silence instead of into an added vowel ("cat-uh"). After a spoken vowel or
    // sonorant they part once the pressure has fallen, so the voice fades out instead of ending in
    // an [h]. A sung one ends the way a singer ends a note: the folds part slowly, from before the
    // vowel's end, while the breath is still there, and the tone thins over the last tenth of a
    // second instead of being cut off as the pressure crosses the voicing threshold.
    if !open_end {
        if let_go {
            let at = end + LET_GO_OPEN_AT;
            gestures.push(Gesture::new(at, end + 0.1, LET_GO_OPENING, 0.06, LET_GO_STRENGTH, one(ControlKey::ABDUCTION, 0.22)).label(EXHALE_LABEL));
        } else {
            let at = end + if sonorant { 0.045 } else { 0.015 };
            gestures.push(Gesture::new(at, end + 0.1, 0.03, 0.06, 6.0, one(ControlKey::ABDUCTION, 0.22)).label(EXHALE_LABEL));
        }
    }
    // ---- intonation: pitch targets on vowels
    let pitch = Pitch { segs: &segs, times: &times, v_idx: &v_idx, o, start, end, sung };
    if tn.upstream_line {
        pitch.upstream_line(&tn, gestures);
    } else {
        pitch.falling_line(&tn, gestures);
    }
    // sung: a consonant given its note's pitch holds it, so the pitch moves during an onset
    // consonant and has arrived when the vowel starts
    for (i, s) in segs.iter().enumerate() {
        if sung && s.vowel.is_none() && truthy(s.f0) {
            let Span { t0, t1 } = times[i];
            let r = pitch_ramp(t1 - t0);
            gestures.push(Gesture::new(t0, t1, r, r, 6.0, one(ControlKey::F0, s.f0.unwrap())).label("pitch"));
        }
    }
    if rate > 1.0 {
        shorten_timing(&mut gestures[first_gesture..], rate);
    }
    let last = times.last().map_or((end, end), |x| (x.t0, x.t1));
    Timing { end, last }
}

/// What a phrase's pitch is planned from: its segments and their spans, which of them are vowels
/// (`v_idx`), and the phrase's end. Its two methods are the two ways to plan it; each appends F0
/// gestures and nothing else.
struct Pitch<'a, 's> {
    segs: &'a [Seg<'s>],
    times: &'a [Span],
    v_idx: &'a [usize],
    o: &'a PlanOptions,
    /// where the phrase starts: `o.t0`, or later by the phrase-start settle
    start: f64,
    end: f64,
    sung: bool,
}

impl Pitch<'_, '_> {
    /// The nuclear accent: the phrase's last primary-stressed vowel, a bigger excursion that falls
    /// (statements) into the final low; earlier accents are smaller (a "hat" pattern). −1: none.
    fn nuclear(&self) -> isize {
        let mut nuclear: isize = -1;
        for &i in self.v_idx {
            if self.segs[i].stress == Some(1) && !self.segs[i].glide_from {
                nuclear = i as isize;
            }
        }
        nuclear
    }

    /// Plint's line (P2, `442f784`): a fall in semitones, most of it early, with the accents as
    /// peaks on it.
    fn falling_line(&self, tn: &PlannerTuning, gestures: &mut Vec<Gesture>) {
        let Pitch { segs, times, v_idx, o, start, end, sung } = *self;
        let f0 = o.f0.unwrap_or(110.0);
        let accent = o.accent.unwrap_or(tn.accent);
        let (pitch_start, pitch_fall) = (o.pitch_start.unwrap_or(tn.pitch_start), o.pitch_fall.unwrap_or(tn.pitch_fall));
        let final_low = f0 * pow(2.0, (pitch_start - pitch_fall - tn.final_drop) / 12.0);
        let total = js_max(0.2, end - start);
        let nuclear = self.nuclear();
        // A spoken phrase's pitch is a falling line with peaks on it. The line: each vowel sits on it,
        // and holds it through the consonants up to the next vowel. A phrase is on its line from its
        // first sound, so a voiced onset doesn't start down at the base pitch. A stressed vowel's
        // accent is a peak over the vowel's first part, after which the pitch comes back to the line:
        // a held accent kept the pitch high through every stressed word, where the TSP men sit near
        // their low line and only touch the high values (docs/hill-climb/log.md, P2).
        let line = |t: f64| {
            let prog = clamp((t - start) / total, 0.0, 1.0);
            // most of the fall comes early: 0 at the start, 1 at the end, `pitch_curve` bends it
            let fallen = (1.0 - exp(-tn.pitch_curve * prog)) / (1.0 - exp(-tn.pitch_curve));
            f0 * pow(2.0, (pitch_start - pitch_fall * fallen) / 12.0)
        };
        let spoken = !sung && !v_idx.is_empty() && !truthy(segs[v_idx[0]].f0);
        if spoken && times[v_idx[0]].t0 - start > 0.02 {
            gestures.push(Gesture::new(start, times[v_idx[0]].t0, 0.03, 0.03, 1.0, one(ControlKey::F0, line(start))).label("pitch"));
        }
        for (n, &i) in v_idx.iter().enumerate() {
            let s = &segs[i];
            let Span { t0, t1 } = times[i];
            // pitch holds through the consonants up to the next vowel (instead of sagging to the base)
            let hold = if n < v_idx.len() - 1 { js_max(t1, times[v_idx[n + 1]].t0 - 0.03) } else { t1 };
            let has_f0 = truthy(s.f0);
            if has_f0 {
                let r = if sung { pitch_ramp(t1 - t0) } else { 0.06 };
                gestures.push(Gesture::new(t0, t1, r, r, 6.0, one(ControlKey::F0, s.f0.unwrap())).label("pitch"));
                continue;
            }
            let base = line(t0);
            let is_nuclear = i as isize == nuclear;
            let acc = if s.stress == Some(1) {
                if is_nuclear { accent * tn.nuclear_accent } else { accent }
            } else if s.stress == Some(2) {
                accent * tn.secondary_accent
            } else {
                0.0
            };
            let last = n == v_idx.len() - 1;
            if last && o.intonation == Some(Intonation::Question) {
                gestures.push(Gesture::new(t0, hold, 0.06, 0.06, 1.0, one(ControlKey::F0, f0 * 1.35)).label("pitch"));
                continue;
            }
            if last && o.intonation.unwrap_or(Intonation::Statement) == Intonation::Statement {
                if is_nuclear && t1 - t0 > 0.08 {
                    // accent peak early in the vowel, then the fall within it
                    gestures.push(Gesture::new(t0, t0 + 0.35 * (t1 - t0), 0.05, 0.05, 1.0, one(ControlKey::F0, base * (1.0 + acc))).label("pitch"));
                    gestures.push(Gesture::new(t0 + 0.55 * (t1 - t0), t1, 0.06, 0.06, 1.0, one(ControlKey::F0, final_low)).label("pitch"));
                    continue;
                }
                gestures.push(Gesture::new(t0, hold, 0.06, 0.06, 1.0, one(ControlKey::F0, final_low)).label("pitch"));
                continue;
            }
            gestures.push(Gesture::new(t0, hold, 0.06, 0.06, 1.0, one(ControlKey::F0, base)).label("pitch"));
            if acc > 0.0 {
                // the peak outweighs the line while it lasts
                let peak_end = t0 + js_max(tn.accent_peak * (t1 - t0), js_min(t1 - t0, 0.05));
                gestures.push(Gesture::new(t0, peak_end, 0.04, 0.06, 4.0, one(ControlKey::F0, base * (1.0 + acc))).label("accent"));
            }
        }
    }

    /// The author's line, the code as it was before P2 (`01755b5`), for `prosody.upstream_line`: a
    /// straight 10 % fall in Hz from the base pitch, a stressed vowel's accent held through the
    /// consonants to the next vowel, one gesture a vowel and none before the first, the final low at
    /// 0.82 of the base pitch. Of the tuning it reads `accent` and the two accent ratios;
    /// `pitch_start`, `pitch_fall`, `pitch_curve`, `accent_peak` and `final_drop` did not exist
    /// then and are not read, in the tuning or in `PlanOptions`.
    fn upstream_line(&self, tn: &PlannerTuning, gestures: &mut Vec<Gesture>) {
        let Pitch { segs, times, v_idx, o, start, end, sung } = *self;
        let f0 = o.f0.unwrap_or(110.0);
        let accent = o.accent.unwrap_or(tn.accent);
        let total = js_max(0.2, end - start);
        let nuclear = self.nuclear();
        for (n, &i) in v_idx.iter().enumerate() {
            let s = &segs[i];
            let Span { t0, t1 } = times[i];
            // pitch holds through the consonants up to the next vowel (instead of sagging to the base)
            let hold = if n < v_idx.len() - 1 { js_max(t1, times[v_idx[n + 1]].t0 - 0.03) } else { t1 };
            let mut target: f64;
            let has_f0 = truthy(s.f0);
            if has_f0 {
                target = s.f0.unwrap();
            } else {
                let prog = (t0 - start) / total;
                let decl = 1.0 - 0.1 * prog;
                let is_nuclear = i as isize == nuclear;
                let acc = if s.stress == Some(1) {
                    1.0 + (if is_nuclear { accent * tn.nuclear_accent } else { accent })
                } else if s.stress == Some(2) {
                    1.0 + accent * tn.secondary_accent
                } else {
                    1.0
                };
                target = f0 * decl * acc;
                let last = n == v_idx.len() - 1;
                if last && o.intonation == Some(Intonation::Question) {
                    target = f0 * 1.35;
                }
                if last && o.intonation.unwrap_or(Intonation::Statement) == Intonation::Statement {
                    if is_nuclear && t1 - t0 > 0.08 {
                        // accent peak early in the vowel, then the fall within it
                        gestures.push(Gesture::new(t0, t0 + 0.35 * (t1 - t0), 0.05, 0.05, 1.0, one(ControlKey::F0, target)).label("pitch"));
                        gestures.push(Gesture::new(t0 + 0.55 * (t1 - t0), t1, 0.06, 0.06, 1.0, one(ControlKey::F0, f0 * 0.82)).label("pitch"));
                        continue;
                    }
                    target = f0 * 0.82;
                }
            }
            let r = if has_f0 && sung { pitch_ramp(t1 - t0) } else { 0.06 };
            gestures.push(Gesture::new(t0, if has_f0 { t1 } else { hold }, r, r, if has_f0 { 6.0 } else { 1.0 }, one(ControlKey::F0, target)).label("pitch"));
        }
    }
}

/// The share of an affricate's span that is closure, the rest being frication (the TSP speakers'
/// /tʃ/: about 55 ms of closure, 60-65 ms of frication and a fading tail; their /dʒ/: 45-55, about 45).
/// The default of `prosody.affricate_closure`.
pub const AFFRICATE_CLOSURE: f64 = 0.45;

/// An affricate's folds: apart by `open` for the frication, and together again by `t1`, as it ends. A
/// span too short for the 30 ms closing (a fast sung note) closes faster, so the breath doesn't spill
/// into the vowel.
fn affricate_folds(open: f64, t1: f64, tn: &PlannerTuning) -> Gesture {
    let (hold, closing) = if t1 - open > tn.affricate_closing { (t1 - tn.affricate_closing, tn.affricate_closing) } else { (open, t1 - open) };
    Gesture::new(open, hold, tn.affricate_ramp_in, closing, 6.0, one(ControlKey::ABDUCTION, tn.affricate_opening))
}

/// How much later a nasal's closure is aimed than its span's start (s). `oral_gestures_into` brings
/// contact to about its `a` (a median 0-9 ms early in running speech), but a nasal is heard from when
/// the oral area falls under about 0.05 cm², when the sound has already left for the nose, and that
/// is where MFA starts it on a render: a median 7-13 ms before the span at the lips, 14-18 at the
/// tip, 19-20 at the dorsum. Klatt's durations are acoustic, so a span starts where it sounds closed
/// (hill-climb log, experiment 25). A short sung nasal keeps the 30 ms a stop's delay leaves.
fn nasal_closure_delay(place: &str) -> f64 {
    match place {
        "bilabial" | "labiodental" => 0.01,
        "velar" | "uvular" | "palatal" => 0.02,
        _ => 0.015,
    }
}

/// How much later a stop's closure is aimed than its span's start (s), where the sound before it runs
/// up to the closure. `oral_gestures_into` brings contact to about its `a`, and the sound has gone by
/// then. After a vowel the level is 10 dB down and MFA starts the stop a median 3-8 ms before the span
/// at the lips, 13-27 at the tip, 9-15 at the dorsum, with the voicing and the body making no steady
/// difference. A nasal's numbers don't carry over: a lip closure stays loud until nearly shut, and
/// a nasal's sound leaves for the nose. After a sibilant the noise falls as the closing area nears its
/// channel's, and at the tip the tongue is already up: 16-23, 25-35, 14-21. Nothing to aim for
/// otherwise: a pause, stop or nasal before it has nothing running up to the closure, /f t/ closes on
/// time, and after /ɹ l/ the tip closes late already (hill-climb log, experiment 26). The tip's 19 centres its fall
/// in running speech with the lungs' port fixed (experiment 32).
fn stop_closure_delay(place: &str, prev: Option<&Seg>) -> f64 {
    let Some(prev) = prev else { return 0.0 };
    let (lips, tip, dorsum) = if prev.vowel.is_some() {
        (0.005, 0.019, 0.015)
    } else if matches!(prev.ipa, "s" | "z" | "ʃ" | "ʒ") {
        (0.02, 0.03, 0.02)
    } else {
        return 0.0;
    };
    match place {
        "bilabial" | "labiodental" => lips,
        "velar" | "uvular" | "palatal" => dorsum,
        _ => tip,
    }
}

/// How firmly a palatalized consonant's tongue body is held (the overlapping vowel has 1.0, the
/// consonant's own articulators 40), and how long it takes to take and let go of the posture (s)
const PALATAL_STRENGTH: f64 = 3.0;
const PALATAL_RAMP: f64 = 0.04;

/// The tongue body's posture for a palatalized consonant: /j/'s, on the coordinates the consonant
/// itself does not use. A velar closes with the body's height, so only its front-back
/// place moves (fronted); a consonant with no free body coordinate (palatals, uvulars, /w/, /ɫ/,
/// glottals) has nothing to add.
fn palatal_body(def: &PhonemeDef, body: BodyMix) -> Option<ArticPatch> {
    let j = phoneme("j", body)?;
    let mut out = ArticPatch::default();
    for k in [ArticKey::TbX, ArticKey::TbY] {
        if let (false, Some(x)) = (def.dominant.contains(&k), j.targets.get(k)) {
            out.set(k, x);
        }
    }
    (!out.is_empty() && def.place != "palatal" && def.manner != Manner::Glottal).then_some(out)
}

/// Fast speech: segments shrink by the rate, so the gestures' ramps and release holds must shrink
/// too. With fixed ones, a stop's closure outlasted the short vowel after it: at rate 1.8 the /ɑ/
/// of "ɑtɑtɑ" never opened (tests/fast_speech.rs). Pitch and breath gestures keep their ramps.
fn shorten_timing(gestures: &mut [Gesture], rate: f64) {
    for g in gestures.iter_mut().filter(|g| !g.targets.has(ControlKey::F0) && !g.targets.has(ControlKey::BREATH)) {
        g.ramp_in /= rate;
        g.ramp_out /= rate;
        g.release_hold = g.release_hold.map(|h| h / rate);
    }
}

/// Front-to-back order of the oral places of articulation.
fn place_rank(place: &str) -> Option<u8> {
    Some(match place {
        "bilabial" => 0,
        "labiodental" => 1,
        "dental" => 2,
        "alveolar" => 3,
        "postalveolar" => 4,
        "retroflex" => 5,
        "palatal" => 6,
        "velar" => 7,
        "uvular" => 8,
        _ => return None,
    })
}

/// How much later a pre-placed closure behind another stop's closure is aimed (s): the time its
/// shorter travel saves, so it seals as the front closure opens.
const PLACED_CLOSURE_DELAY: f64 = 0.025;
/// An aspirated stop's voice onset time when its front end doesn't say (English's), in seconds.
/// The default of `prosody.vot`.
pub const DEFAULT_VOT: f64 = 0.05;

/// Whether stop `i` closes behind the closure of the stop before it, with its own articulators
/// already set by the consonant before that one (/t/ in /ɹpt/: the tip is up for the /ɹ/ and
/// stays free during the lip closure; a rhotic vowel raises it too). After any other vowel ("apt",
/// "napkin") the timing is unchanged.
fn pre_placed_behind_closure(segs: &[Seg], i: usize) -> bool {
    let (Some(def), Some(prev)) = (segs[i].def.as_ref(), i.checked_sub(1).and_then(|j| segs[j].def.as_ref())) else { return false };
    if def.manner != Manner::Stop || prev.manner != Manner::Stop || !closes_in_front(prev.place, def.place) || i < 2 {
        return false;
    }
    let before = &segs[i - 2];
    let rhotic = before.vowel.as_ref().filter(|v| v.get(ArticKey::TtCurl).unwrap_or(0.0) > 0.05);
    let Some(tg) = before.def.as_ref().map(|d| &d.targets).or(rhotic) else { return false };
    def.dominant.iter().all(|k| tg.get(*k).is_some())
}

/// Whether a closure at `front` lies in front of one at `back`.
fn closes_in_front(front: &str, back: &str) -> bool {
    matches!((place_rank(front), place_rank(back)), (Some(f), Some(b)) if f < b)
}

/// Ramps of a given pitch (a sung note's) on a segment `dur` long: 60 ms, but at most half the
/// segment. A ramp-in starts before the segment, so in a fast run a full 60 ms would pull the
/// previous vowel off its note for most of its length.
fn pitch_ramp(dur: f64) -> f64 {
    js_min(0.06, js_max(0.008, 0.5 * dur))
}

/// Oral gestures for one consonant constriction held acoustically from `a` to `b` (closure to
/// release): a dominant gesture on the consonant's own articulators and a weak one on the rest.
/// Movement latencies are compensated so that contact and the acoustic release land about on a and
/// b. The closure is heard earlier than contact: see `nasal_closure_delay`, `stop_closure_delay`.
pub fn oral_gestures(def: &PhonemeDef, tg: &ArticPatch, a: f64, b: f64, speed: f64, label: &str) -> Vec<Gesture> {
    let mut out = Vec::with_capacity(4);
    oral_gestures_into(def, tg, a, b, speed, label, &mut out);
    out
}

/// [`oral_gestures`], appended to `out` (at most four gestures; no allocation when `out` has
/// room for them).
pub fn oral_gestures_into(def: &PhonemeDef, tg: &ArticPatch, a: f64, b: f64, speed: f64, label: &str, out: &mut Vec<Gesture>) {
    let mut dom = ArticPatch::default();
    let mut sec = ArticPatch::default();
    for (k, v) in tg.iter() {
        if def.dominant.contains(&k) {
            dom.set(k, v);
        } else {
            sec.set(k, v);
        }
    }
    let is = |k: &ArticKey, a: ArticKey, b: ArticKey, c: ArticKey| *k == a || *k == b || *k == c;
    // the dorsum is heavy: its releases are slower than tip and lip releases
    let body_only = def.dominant.iter().all(|k| is(k, ArticKey::TbX, ArticKey::TbY, ArticKey::Root));
    let sp = if body_only { 1.0 + (speed - 1.0) * 0.35 } else { speed };
    // articulators need ~10–15 ms to climb out of the overshoot (contact) target, so the
    // gesture releases early and the acoustic release lands on the planned boundary:
    // measured movement latencies (closure lag / release lag) per articulator class
    let closing = matches!(def.manner, Manner::Stop | Manner::Nasal | Manner::Affricate);
    let lips_only = def.dominant.iter().all(|k| *k == ArticKey::LipAp || *k == ArticKey::Labiodental);
    let lead = if closing { if body_only { 0.034 } else if lips_only { 0.019 } else { 0.02 } } else { 0.0 };
    // a consonant must own its constriction articulators (Task-Dynamics style): at strength 40
    // the overlapping vowel contributes ~2 %, so closures seal instead of being diluted
    let early = if closing { if body_only { 0.024 } else if lips_only { 0.022 } else { 0.031 } } else { 0.008 };
    if !dom.is_empty() {
        out.push(Gesture::new(a - early, js_max(a + 0.01, b - lead), 0.03, if body_only { 0.03 } else { 0.018 }, 40.0, artic_targets(&dom)).speed(sp).release_hold(0.045).label(label));
    }
    // alveolars front the tongue body during the closure (the alveolar F2 locus, ~1.8 kHz)
    let alveolar = def.place == "alveolar" && tg.get(ArticKey::TbX).is_none() && matches!(def.manner, Manner::Stop | Manner::Nasal | Manner::Fricative | Manner::Affricate | Manner::Tap);
    if alveolar {
        out.push(Gesture::new(a, b, 0.05, 0.05, 1.2, Targets::default().with(ArticKey::TbX, 3.2).with(ArticKey::TbY, 7.4)).label(label));
    }
    if !sec.is_empty() {
        // a velar's front–back place follows the vowel (fronted /k/ in "key", backed in "caw"):
        // its own tbX only weakly, so the F2 locus moves with the vowel as in natural speech
        let place = if body_only { sec.get(ArticKey::TbX) } else { None };
        if place.is_some() {
            sec.values[ArticKey::TbX.index()] = None;
        }
        if !sec.is_empty() {
            out.push(Gesture::new(a, b, 0.05, 0.05, 2.2, artic_targets(&sec)).label(label));
        }
        if let Some(x) = place {
            out.push(Gesture::new(a, b, 0.05, 0.05, 0.25, one(ArticKey::TbX, x)).label(label));
        }
    }
}

/// One phrase of [`plan_phrases`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhraseInput {
    pub segments: Vec<Segment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intonation: Option<Intonation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause: Option<f64>,
}

/// Plan consecutive phrases (each with its own breath group and intonation) from t0.
/// `o.intonation` is ignored (each phrase brings its own).
pub fn plan_phrases(phrases: &[PhraseInput], o: &PlanOptions) -> Plan {
    let mut all = Plan { gestures: Vec::new(), segments: Vec::new(), end: o.t0 };
    let mut t = o.t0;
    for p in phrases {
        let plan = plan_utterance(&p.segments, &PlanOptions { t0: t, intonation: Some(p.intonation.unwrap_or(Intonation::Statement)), ..o.clone() });
        all.gestures.extend_from_slice(&plan.gestures);
        all.segments.extend(plan.segments);
        t = plan.end + p.pause.unwrap_or(0.25);
        all.end = plan.end;
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetics::ipa::tokenize_ipa;
    use crate::tuning::Tuning;

    fn male() -> PlanOptions {
        PlanOptions { body: BodyMix::MALE, ..Default::default() }
    }

    fn with(set: impl Fn(&mut Tuning)) -> PlanOptions {
        let mut t = Tuning::default();
        set(&mut t);
        PlanOptions { tuning: Some(t.plan()), ..male() }
    }

    /// a plan as the JSON the UI receives (an unset target is NaN in a `Gesture`, which equals nothing)
    fn json(plan: &Plan) -> serde_json::Value {
        serde_json::to_value(plan).unwrap()
    }

    /// the F0 gestures of a plan: (t0, t1, ramp in, ramp out, strength, Hz)
    fn pitch(plan: &Plan) -> Vec<(f64, f64, f64, f64, f64, f64)> {
        plan.gestures.iter().filter_map(|g| g.targets.get(ControlKey::F0).map(|hz| (g.t0, g.t1, g.ramp_in, g.ramp_out, g.strength, hz))).collect()
    }

    #[test]
    fn the_default_tuning_plans_as_no_tuning_does_and_another_does_not() {
        let d = Tuning::default().prosody;
        assert_eq!((d.voiced_fricative_opening, d.dh_opening, d.affricate_closure, d.vot), (VOICED_FRICATIVE_OPENING, DH_OPENING, AFFRICATE_CLOSURE, DEFAULT_VOT));
        // stops, fricatives of both kinds, /h/, an affricate, /ð/ after a nasal, a secondary stress
        let segs = tokenize_ipa("ˈhæpi ˈdʒʌdʒɪz ɪn ðə ˈtʃɝtʃ ˌsɛvən");
        let plain = plan_utterance(&segs, &male());
        assert_eq!(json(&plan_utterance(&segs, &with(|_| {}))), json(&plain));
        assert_ne!(json(&plan_utterance(&segs, &with(|t| t.prosody.vot = 0.07))), json(&plain));
        // /ð/ after a nasal's own opening, which no golden plan has
        assert_ne!(json(&plan_utterance(&segs, &with(|t| t.prosody.dh_opening = 0.05))), json(&plain));
        // the durations' part is read as well, and moves every later span
        let longer = plan_utterance(&segs, &with(|t| t.durations.unstressed = 0.9));
        assert!(longer.end > plain.end + 0.01, "{} {}", longer.end, plain.end);
        // and both parts reach each phrase of `plan_phrases`
        let phrases = [PhraseInput { segments: segs.clone(), ..Default::default() }, PhraseInput { segments: segs, ..Default::default() }];
        let plain = plan_phrases(&phrases, &male());
        assert_eq!(json(&plan_phrases(&phrases, &with(|_| {}))), json(&plain));
        assert_ne!(json(&plan_phrases(&phrases, &with(|t| t.prosody.h_opening = 0.4))), json(&plain));
    }

    #[test]
    fn an_explicit_option_wins_over_the_tuning() {
        let segs = tokenize_ipa("ˈapə ˈapə");
        let tuned = with(|t| {
            t.prosody.pitch_start = 2.0;
            t.prosody.pitch_fall = 3.0;
            t.prosody.accent = 0.3;
        });
        let by_tuning = plan_utterance(&segs, &tuned);
        assert_ne!(pitch(&by_tuning), pitch(&plan_utterance(&segs, &male())));
        // the same numbers as options plan the same pitch, whatever the tuning says
        let by_options = plan_utterance(&segs, &PlanOptions { pitch_start: Some(2.0), pitch_fall: Some(3.0), accent: Some(0.3), ..male() });
        assert_eq!(pitch(&by_options), pitch(&by_tuning));
        let both = plan_utterance(&segs, &PlanOptions { pitch_start: Some(PITCH_START), pitch_fall: Some(PITCH_FALL), accent: Some(ACCENT), ..tuned });
        assert_eq!(pitch(&both), pitch(&plan_utterance(&segs, &male())));
        let start_only = plan_utterance(&segs, &PlanOptions { pitch_start: Some(PITCH_START), ..tuned });
        assert_ne!(pitch(&start_only), pitch(&by_tuning));
    }

    /// The numbers are the plan of "cons p" in the golden planning fixture as it was at `01755b5`.
    #[test]
    fn the_upstream_line_is_the_plan_from_before_the_falling_line() {
        let segs = tokenize_ipa("ˈapə");
        let up = PlanOptions { tuning: Some(Tuning::upstream().plan()), ..male() };
        let old = plan_utterance(&segs, &up);
        // one gesture a vowel: the accent held to 30 ms before the next vowel, then the final low
        assert_eq!(pitch(&old), [(0.06, 0.24544200000000002, 0.06, 0.06, 1.0, 135.52235211514832), (0.275442, 0.376002, 0.06, 0.06, 1.0, 90.19999999999999)]);
        // only the pitch differs from Plint's plan
        let new = plan_utterance(&segs, &male());
        assert_ne!(pitch(&new), pitch(&old));
        let rest = |p: &Plan| json(&Plan { gestures: p.gestures.iter().filter(|g| !g.targets.has(ControlKey::F0)).cloned().collect(), ..p.clone() });
        assert_eq!(rest(&new), rest(&old));
        // the switch alone selects it: the five values `Tuning::upstream` sets beside it are not read,
        // nor are the options that came with the falling line; the accent still is
        assert_eq!(json(&plan_utterance(&segs, &with(|t| t.prosody.upstream_line = true))), json(&old));
        assert_eq!(json(&plan_utterance(&segs, &PlanOptions { pitch_start: Some(3.0), pitch_fall: Some(9.0), ..up.clone() })), json(&old));
        assert_ne!(pitch(&plan_utterance(&segs, &PlanOptions { accent: Some(0.3), ..up })), pitch(&old));
        // and with it off the tuning plans as none does
        assert_eq!(json(&plan_utterance(&segs, &with(|t| t.prosody.upstream_line = false))), json(&new));
    }

    /// the gestures of a plan, each as its JSON (an unset target is NaN and equals nothing)
    fn gs(plan: &Plan) -> Vec<serde_json::Value> {
        plan.gestures.iter().map(|g| serde_json::to_value(g).unwrap()).collect()
    }

    /// (index, with, without) of the gestures that differ; the plans must have as many, in the same order
    fn changed(a: &Plan, b: &Plan) -> Vec<(usize, Gesture, Gesture)> {
        assert_eq!(a.gestures.len(), b.gestures.len());
        let (ja, jb) = (gs(a), gs(b));
        (0..ja.len()).filter(|&i| ja[i] != jb[i]).map(|i| (i, a.gestures[i], b.gestures[i])).collect()
    }

    fn all_on(t: &mut Tuning) {
        t.prosody.onset_settle = 0.1;
        t.prosody.final_stop_breath = true;
        t.prosody.final_stop_hold = 0.04;
        t.prosody.cluster_aspiration = true;
        t.prosody.vot_unstressed = 0.4;
    }

    const WORDS: [&str; 16] = ["ˈsi", "ˈʃi", "ˈfɪʃ", "ˈθɪn", "ˈti", "ˈdeɪ", "ˈnoʊ", "ˈleɪ", "ˈɹɛd", "ˈhæt", "ˈit", "ˈtʃɪn", "ˈtɹi", "ˈhæpi", "ˈstɹim", "ˈɹɪb"];

    #[test]
    fn the_onset_and_offset_rules_are_off_by_default_and_for_song() {
        for w in WORDS {
            let segs = tokenize_ipa(w);
            let plain = plan_utterance(&segs, &male());
            let defaults = with(|t| {
                t.prosody.onset_settle = 0.0;
                t.prosody.final_stop_breath = false;
                t.prosody.final_stop_hold = 0.0;
                t.prosody.cluster_aspiration = false;
                t.prosody.vot_unstressed = 1.0;
            });
            assert_eq!(json(&plan_utterance(&segs, &defaults)), json(&plain), "{w}");
            // the author's voice has none of them
            let up = PlanOptions { tuning: Some(Tuning::upstream().plan()), ..male() };
            assert_eq!(json(&plan_utterance(&segs, &up)), json(&plan_utterance(&segs, &with(|t| t.prosody.upstream_line = true))), "{w}");
            // all on: a spoken plan moves, a sung one does not
            assert_ne!(json(&plan_utterance(&segs, &with(all_on))), json(&plain), "{w}");
            let sung = PlanOptions { sung: Some(true), ..male() };
            assert_eq!(json(&plan_utterance(&segs, &PlanOptions { sung: Some(true), ..with(all_on) })), json(&plan_utterance(&segs, &sung)), "{w} sung");
            // nor does a note held open wait for the settle
            let open = PlanOptions { open_end: Some(true), ..male() };
            let settled = PlanOptions { open_end: Some(true), ..with(|t| t.prosody.onset_settle = 0.1) };
            assert_eq!(json(&plan_utterance(&segs, &settled)), json(&plan_utterance(&segs, &open)), "{w} open");
        }
    }

    #[test]
    fn the_onset_settle_moves_the_phrase_and_places_its_first_consonant_early() {
        let settle = 0.1;
        let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
        for w in WORDS {
            let segs = tokenize_ipa(w);
            let with_settle = plan_utterance(&segs, &with(|t| t.prosody.onset_settle = settle));
            // the same phrase planned that much later with no settle: the times the plan reports are these
            let later = plan_utterance(&segs, &PlanOptions { t0: settle, ..male() });
            assert_eq!(serde_json::to_value(&with_settle.segments).unwrap(), serde_json::to_value(&later.segments).unwrap(), "{w}");
            assert_eq!(with_settle.end, later.end);
            assert!(near(with_settle.end, plan_utterance(&segs, &male()).end + settle));
            // nothing is planned before the plan's start
            assert!(with_settle.gestures.iter().all(|g| g.t0 >= 0.029 && g.t1 > g.t0 && g.ramp_in > 0.0), "{w}");
            let first = &with_settle.segments[0];
            let diff = changed(&with_settle, &later);
            if first.kind == "vowel" {
                assert!(diff.is_empty(), "{w}");
                continue;
            }
            let voiceless_fricative = matches!(first.ipa.as_str(), "s" | "ʃ" | "f" | "θ");
            let (mut oral, mut folds, mut breath) = (0, 0, 0);
            for (_, a, b) in &diff {
                // the rest of each gesture is as it was
                let rest = |g: &Gesture, like: &Gesture| serde_json::to_value(Gesture { t0: like.t0, ramp_in: like.ramp_in, speed: like.speed, ..*g }).unwrap();
                assert_eq!(rest(a, b), serde_json::to_value(b).unwrap(), "{w}");
                if b.targets.has(ControlKey::BREATH) {
                    assert!(voiceless_fricative && near(a.t0, b.t0 + ONSET_BREATH_DELAY) && a.ramp_in == b.ramp_in, "{w}");
                    breath += 1;
                } else if b.targets.has(ControlKey::ABDUCTION) {
                    // the first consonant's folds part early (its gesture starts before any later one's)
                    assert!(b.t0 < first.t1 && near(a.t0, b.t0 - ONSET_LEAD_SHARE * settle) && a.ramp_in == b.ramp_in && a.speed == b.speed, "{w}");
                    folds += 1;
                } else {
                    // its oral gestures: early, slow, at the articulators' own speed
                    assert_eq!(b.label_str(), first.ipa, "{w}");
                    assert!(!b.targets.has(ControlKey::F0) && b.t0 <= first.t0 + 0.021, "{w}");
                    assert!(near(a.t0, b.t0 - ONSET_LEAD_SHARE * settle) && a.ramp_in == settle && a.speed.is_none(), "{w}");
                    oral += 1;
                }
            }
            assert_eq!(breath, usize::from(voiceless_fricative), "{w}");
            let voiceless = matches!(first.ipa.as_str(), "s" | "ʃ" | "f" | "θ" | "t" | "h" | "tʃ");
            assert_eq!(folds, usize::from(voiceless), "{w}");
            assert_eq!(oral > 0, first.ipa != "h", "{w}");
            // every oral gesture that starts with the first consonant has moved; an affricate's release has not
            let early = later.gestures.iter().filter(|g| g.label_str() == first.ipa && g.t0 <= first.t0 + 0.021 && !g.targets.has(ControlKey::ABDUCTION)).count();
            assert_eq!(oral, early, "{w}");
            if first.ipa == "tʃ" {
                assert!(later.gestures.iter().filter(|g| g.label_str() == "tʃ").count() > oral);
            }
        }
        // in a second phrase too, and the pause between the phrases is the same
        let phrases = [PhraseInput { segments: tokenize_ipa("ˈsi"), ..Default::default() }, PhraseInput { segments: tokenize_ipa("ˈʃi"), ..Default::default() }];
        let (plain, settled) = (plan_phrases(&phrases, &male()), plan_phrases(&phrases, &with(|t| t.prosody.onset_settle = settle)));
        assert!(near(settled.segments[0].t0, plain.segments[0].t0 + settle) && near(settled.segments[2].t0, plain.segments[2].t0 + 2.0 * settle));
        assert!(near(settled.end, plain.end + 2.0 * settle));
    }

    #[test]
    fn a_final_voiced_stop_ends_the_breath_early_and_is_held() {
        let breath_only = |t: &mut Tuning| t.prosody.final_stop_breath = true;
        let hold_only = |t: &mut Tuning| t.prosody.final_stop_hold = 0.04;
        for w in ["ˈɹɛd", "ˈbɛd", "ˈɹɪb", "ˈmæd"] {
            let segs = tokenize_ipa(w);
            let plain = plan_utterance(&segs, &male());
            let last = plain.segments.last().unwrap().clone();
            // the switch: the breath gesture's end and nothing else
            let diff = changed(&plan_utterance(&segs, &with(breath_only)), &plain);
            assert_eq!(diff.len(), 1, "{w}");
            let (_, a, b) = diff[0];
            assert!(b.targets.has(ControlKey::BREATH) && b.t1 == plain.end && a.t1 == last.t0 + FINAL_STOP_BREATH, "{w}");
            assert_eq!(serde_json::to_value(Gesture { t1: b.t1, ..a }).unwrap(), serde_json::to_value(b).unwrap());
            // the hold: the stop's own oral gestures end later and nothing else
            let held = plan_utterance(&segs, &with(hold_only));
            let diff = changed(&held, &plain);
            let orals = plain.gestures.iter().enumerate().filter(|(_, g)| g.label_str() == last.ipa && g.t1 > last.t0 + 0.02).count();
            assert!(!diff.is_empty() && diff.len() == orals, "{w}: {} of {orals}", diff.len());
            for (_, a, b) in diff {
                assert!(b.label_str() == last.ipa && (a.t1 - (b.t1 + 0.04)).abs() < 1e-12, "{w}");
                assert_eq!(serde_json::to_value(Gesture { t1: b.t1, ..a }).unwrap(), serde_json::to_value(b).unwrap());
            }
            assert_eq!((held.end, held.segments.last().unwrap().t1), (plain.end, last.t1));
            // not a sung phrase, and not one held open
            for o in [PlanOptions { sung: Some(true), ..male() }, PlanOptions { open_end: Some(true), ..male() }] {
                let on = PlanOptions { tuning: with(|t| { breath_only(t); hold_only(t) }).tuning, ..o.clone() };
                assert_eq!(json(&plan_utterance(&segs, &on)), json(&plan_utterance(&segs, &o)), "{w}");
            }
        }
        // only /b d g/ at the phrase's end: not /p t k/, an affricate, a fricative, a nasal, a vowel, a /d/ inside
        for w in ["ˈkæt", "ˈstɑp", "ˈbæk", "ˈbitʃ", "ˈdʒʌdʒ", "ˈbʌs", "ˈbʌz", "ˈmæn", "ˈsi", "ˈɹɛdi"] {
            let segs = tokenize_ipa(w);
            assert_eq!(json(&plan_utterance(&segs, &with(|t| { breath_only(t); hold_only(t) }))), json(&plan_utterance(&segs, &male())), "{w}");
        }
    }

    #[test]
    fn the_aspiration_rules_move_only_the_folds_of_the_stops_they_name() {
        let cluster = |t: &mut Tuning| t.prosody.cluster_aspiration = true;
        let unstressed = |t: &mut Tuning| t.prosody.vot_unstressed = 0.4;
        let d = Tuning::default().prosody;
        // tree, clean, pray, twin, cue: the stop's fold gesture is the aspirated one, held longer
        for w in ["ˈtɹi", "ˈklin", "ˈpɹeɪ", "ˈtwɪn", "ˈkju"] {
            let segs = tokenize_ipa(w);
            let plain = plan_utterance(&segs, &male());
            let diff = changed(&plan_utterance(&segs, &with(cluster)), &plain);
            assert_eq!(diff.len(), 1, "{w}");
            let (_, a, b) = diff[0];
            let stop = &plain.segments[0];
            assert_eq!((a.label_str(), a.t0, a.ramp_in), ("aspiration", b.t0, b.ramp_in), "{w}");
            assert_eq!((b.t1, b.ramp_out, b.targets.get(ControlKey::ABDUCTION)), (stop.t1 + d.unaspirated_vot * d.vot_fold_share, d.unaspirated_ramp_out, Some(d.unaspirated_opening)), "{w}");
            assert_eq!((a.t1, a.ramp_out, a.targets.get(ControlKey::ABDUCTION)), (stop.t1 + d.vot * d.vot_fold_share * CLUSTER_FOLDS, d.aspirated_ramp_out, Some(d.aspirated_opening)), "{w}");
            // the folds stay apart well into the sonorant; and a front end's word stands
            assert!(a.end() > plain.segments[1].t0 + 0.05, "{w}");
            let mut said = segs.clone();
            said[0].aspirated = Some(false);
            assert_eq!(json(&plan_utterance(&said, &with(cluster))), json(&plan_utterance(&said, &male())), "{w}");
        }
        // tea (aspirated already), stream and spray (after /s/), a cluster with no vowel after it, a voiced stop
        for w in ["ˈti", "ˈstɹim", "ˈspɹeɪ", "ˈætl", "ˈdɹi", "ˈtʃɹi"] {
            let segs = tokenize_ipa(w);
            assert_eq!(json(&plan_utterance(&segs, &with(cluster))), json(&plan_utterance(&segs, &male())), "{w}");
        }
        // the share: only a stop before an unstressed vowel, one gesture each
        for (w, n) in [("ˈhæpi", 1), ("ˈpeɪpɚ", 1), ("ˈtɪkət", 1), ("pəˈteɪtoʊ", 2)] {
            let segs = tokenize_ipa(w);
            let plain = plan_utterance(&segs, &male());
            let diff = changed(&plan_utterance(&segs, &with(unstressed)), &plain);
            assert_eq!(diff.len(), n, "{w}");
            for (_, a, b) in diff {
                assert_eq!((a.label_str(), a.t0, a.ramp_in), ("aspiration", b.t0, b.ramp_in), "{w}");
                // shorter, a quicker closing and a smaller opening, all still above the unaspirated stop's
                assert!(a.t1 < b.t1 && a.ramp_out < b.ramp_out && a.targets.get(ControlKey::ABDUCTION) < b.targets.get(ControlKey::ABDUCTION), "{w}");
                assert!(a.ramp_out > d.unaspirated_ramp_out && a.targets.get(ControlKey::ABDUCTION) > Some(d.unaspirated_opening), "{w}");
            }
        }
        for w in ["ˈti", "əˈtæk", "ˈkæt", "ˈstɑp", "ˈtʃɪn", "ˌtæˈtu"] {
            let segs = tokenize_ipa(w);
            assert_eq!(json(&plan_utterance(&segs, &with(unstressed))), json(&plan_utterance(&segs, &male())), "{w}");
        }
        // a front end's own voice onset time is not scaled
        let mut said = tokenize_ipa("ˈhæpi");
        said[2].vot = Some(0.08);
        assert_eq!(json(&plan_utterance(&said, &with(unstressed))), json(&plan_utterance(&said, &male())));
        // both: a cluster before an unstressed vowel takes the share too
        let segs = tokenize_ipa("ˈkʌntɹi");
        let both = changed(&plan_utterance(&segs, &with(|t| { cluster(t); unstressed(t) })), &plan_utterance(&segs, &with(cluster)));
        assert_eq!(both.len(), 1);
        assert!(both[0].1.t1 < both[0].2.t1);
    }

    /// The golden speech sentence and a second phrase, every rule on: the score is well formed (no
    /// gesture's t0 before its phrase's start, none ending before it starts, the timeline in
    /// order) at both rates, and the male voice renders it with nothing non-finite.
    #[test]
    fn with_every_rule_on_the_plan_is_well_formed_and_renders() {
        use crate::anatomy::human::BODY_MALE;
        use crate::voice::human_voice::HumanVoice;
        let phrases = [
            PhraseInput { segments: tokenize_ipa("hɛˈloʊ ˈwɝld ʃi ˈsɛlz ˈɑpɑ"), ..Default::default() },
            PhraseInput { segments: tokenize_ipa("ˈtɹaɪ ðə ˈpeɪpɚ ˈbæɡ ənd ˈklin ðə ˈɹɛd ˈbɛd"), ..Default::default() },
        ];
        let mut tuning = Tuning::default();
        all_on(&mut tuning);
        for rate in [None, Some(1.8)] {
            let t0 = 0.05;
            let plan = plan_phrases(&phrases, &PlanOptions { t0, rate, f0: Some(112.0), tuning: Some(tuning.plan()), ..male() });
            let second = plan.segments[phrases[0].segments.len()..].iter().find(|s| s.ipa == "t").unwrap().t0 - 0.06 - tuning.prosody.onset_settle;
            for g in &plan.gestures {
                for x in [g.t0, g.t1, g.ramp_in, g.ramp_out, g.strength] {
                    assert!(x.is_finite(), "{g:?}");
                }
                assert!(g.t0 >= t0 && g.t1 >= g.t0 && g.ramp_in > 0.0 && g.ramp_out > 0.0, "{g:?}");
                // and none of the second phrase's starts before that phrase does
                assert!(g.t0 < second - 0.2 || g.t0 >= second - 1e-9, "{g:?} {second}");
            }
            assert!(plan.segments.windows(2).all(|w| w[0].t0 < w[0].t1 && w[0].t1 <= w[1].t0) && plan.segments[0].t0 >= t0 && plan.end == plan.segments.last().unwrap().t1);
            if rate.is_none() {
                let mut v = HumanVoice::with_tuning(48000.0, BODY_MALE, 2, &tuning);
                v.schedule(&plan.gestures);
                let mut out = vec![0.0f32; (48000.0 * (plan.end + 0.3)) as usize];
                for block in out.chunks_mut(128) {
                    v.render(block);
                }
                assert!(out.iter().all(|x| x.is_finite()));
                let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                assert!(peak > 0.05 && peak <= 0.76, "{peak}");
            }
        }
    }
}
