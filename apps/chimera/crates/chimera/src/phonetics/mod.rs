//! Phonetics: IPA, English G2P, syllables, durations, the gestural planner, singing, beatbox.
//! Their output is a [`Plan`] of gestures that a voice plays. These run off the audio thread (they
//! allocate), except the live note planners in [`sing`] (`*_into`).

use crate::control::score::Gesture;
use serde::{Deserialize, Serialize};

pub mod beatbox;
pub mod durations;
pub mod english;
pub mod inventory;
pub mod ipa;
pub mod japanese;
pub mod korean;
pub mod mandarin;
pub mod planner;
pub mod russian;
pub mod sing;
pub mod songs;
pub mod syllables;
pub mod turkish;
pub mod vowel_data;

/// One phoneme in an utterance.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub ipa: String,
    /// duration override (s)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    /// 0 unstressed, 1 primary, 2 secondary
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress: Option<u8>,
    /// explicit pitch (Hz) for this segment (singing)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f0: Option<f64>,
    /// word boundary after this segment
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub word_end: Option<bool>,
    /// a voiceless stop or affricate's aspiration, as its front end decides it (Mandarin and Korean
    /// contrast it); None: the planner's default, English's rule (aspirated syllable-initially
    /// before a vowel, not after /s/)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspirated: Option<bool>,
    /// how long an aspirated stop or affricate aspirates (voice onset time, s), as its front end
    /// decides it (Korean's lax stops aspirate briefly, its aspirated ones long); None: 50 ms
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vot: Option<f64>,
    /// a vowel said with the velum lowered (IPA combining tilde); None or false: oral
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nasalized: Option<bool>,
    /// a consonant made with a second, palatal gesture: the tongue body raised and fronted toward
    /// the palate (IPA ʲ); None or false: plain
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palatalized: Option<bool>,
    /// breathy voice: a voiced stop or affricate released with the folds apart (IPA ʱ), or a
    /// vowel said with them part-open (IPA combining diaeresis below); None or false: modal
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub breathy: Option<bool>,
    /// a vowel that is a glide off the vowel before it: reached over this long (s) while that vowel
    /// lets go over the glide's own length, so the two are one movement, not two held targets;
    /// None: a vowel of its own
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glide: Option<f64>,
    /// a long sound: it lasts this many times what the duration rules give it (Turkish's vowel
    /// before ğ, "dağ" [daː]); None: as they give. An explicit `dur` wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<f64>,
}

impl Segment {
    pub fn new(ipa: &str) -> Segment {
        Segment { ipa: ipa.to_string(), ..Default::default() }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intonation {
    /// final fall
    #[default]
    Statement,
    /// final rise
    Question,
    Flat,
}

/// A segment as the planner timed it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannedSegment {
    pub ipa: String,
    pub t0: f64,
    pub t1: f64,
    /// lexical stress of vowels (0 unstressed, 1 primary, 2 secondary)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress: Option<u8>,
    pub kind: String,
}

/// Gestures for a voice plus the timeline they realise (times relative to the plan's t0 origin).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    pub gestures: Vec<Gesture>,
    pub segments: Vec<PlannedSegment>,
    pub end: f64,
}

/// A sung syllable.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Syllable {
    /// lyric text shown on the note
    pub text: String,
    pub onset: Vec<Segment>,
    /// vowel (a diphthong stays one nucleus: "aɪ")
    pub nucleus: Segment,
    pub coda: Vec<Segment>,
    /// continue the previous syllable's vowel at a new pitch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub melisma: Option<bool>,
    /// last syllable of a word (codas sound here; phrase breaks follow punctuation); may be left
    /// out of hand-written JSON (false)
    #[serde(default)]
    pub word_end: bool,
    /// a breath follows (comma, full stop, line break); may be left out (false)
    #[serde(default)]
    pub phrase_end: bool,
    /// pitch bends inside the sung note, as the syllable's front end writes them (a tone, a pitch
    /// accent, a scoop); none: the note's pitch as written
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pitch: Vec<Bend>,
    /// nasalise this fraction of the vowel before a closing rhyme's nasal coda (`closing`; at most
    /// 0.25 s), as Mandarin does; None: the velum lowers only for the nasal itself
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nasal_lead: Option<f64>,
    /// the rhyme's ending (a falling diphthong's off-glide, a nasal coda) is one closing movement at
    /// the end of the note, this long at most (s), as Mandarin's 韵尾 is: the main vowel moves into
    /// it and it is not held. None: the off-glide holds a share of the note and the coda is a
    /// consonant of its own
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closing: Option<f64>,
    /// how long after the onset's last consonant the voice starts (s): an aspiration, a burst, a
    /// fricative's release. A song then starts the vowel that long before its beat, so the voice
    /// is there on the beat (before a medial glide the glide takes it, at its own start). None:
    /// the vowel's segment starts on the beat and the voice after it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_lag: Option<f64>,
    /// in a song, the consonants before this syllable's vowel (its onset and medial glide, the
    /// voice's lag, the coda before them) take at most this share of the time from the vowel
    /// before to this one, and shorten in proportion when a fast note leaves them less: singers'
    /// consonants shorten with the notes. None: they keep their lengths until the vowel before is
    /// down to its floor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onset_share: Option<f64>,
}

/// Which end of its sung vowel a pitch bend sits at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Edge {
    Start,
    /// centred on the vowel's middle (a dip or a peak inside the note: Mandarin's tone 3)
    Middle,
    End,
}

/// A bend of the pitch inside a sung note: `st` semitones from the note, at the vowel's start or
/// end, over `len` of the vowel but at most `max` seconds. The melody stays as written; a bend only
/// colours part of the note (a scoop, a pitch accent).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bend {
    pub at: Edge,
    pub len: f64,
    pub max: f64,
    pub st: f64,
}
