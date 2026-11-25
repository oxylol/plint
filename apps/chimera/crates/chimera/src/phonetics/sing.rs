//! Singing: syllables on notes, through the same gestural planner as speech.
//!  - A note's vowel lands on the beat. Its onset consonants are sung just before it (they take
//!    their time from the previous note, as singers do), its coda just before the next onset.
//!  - Diphthongs sustain their first vowel and glide to the second at the end of the note. A rising
//!    one's medial glide is sung with the onset, so the main vowel holds from the beat.
//!  - A closing rhyme (`Syllable::closing`, Mandarin's) ends in one movement: its off-glide or its
//!    nasal coda is where the vowel ends up at the end of the note, not a part held after it.
//!  - A melisma ("-") carries the vowel onto further notes; the coda waits for the last of them.
//!  - Vibrato blooms a quarter of a second into a held vowel.
//!
//! Live playing uses open-ended notes: a key press sings the onset and starts a vowel that lasts
//! until the key is released (tagged gestures, ended with GestureScore.release), and the release
//! sings the glide and the coda.
//!
//! [`plan_song`] allocates and runs off the audio thread. The live planners come in two forms:
//! `*_into` over a [`LiveSyllable`] allocates nothing once its `Vec` has capacity (the plugin's
//! audio thread); the plain ones over a [`Syllable`] return a `Vec` (the browser's JSON API).

use super::inventory::phoneme;
use crate::anatomy::human::{ArticKey, BodyMix};
use super::planner::{PlanOptions, TimedSegment, held_vowel, offglide, onglide, plan_timed, plan_utterance, vowel_artic};
use super::{Bend, Edge, Intonation, Plan, PlannedSegment, Segment, Syllable};
use crate::control::score::{ControlKey, Gesture, Label, Targets};
use crate::math::{js_max, js_min};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// `440 · 2^((m − 69)/12)`, the same formula and operation order as `sing.ts`.
pub use crate::math::midi_to_hz;

/// sung consonants are a little shorter than spoken ones
fn cdur(ipa: &str, body: BodyMix, k: f64) -> f64 {
    phoneme(ipa, body).map_or(0.07, |d| d.dur) * k
}

/// Sung consonants: a duration the front end gives is its own (a language's timing: Mandarin's long
/// nasal codas); otherwise the phoneme's spoken length times `k`. Either is squeezed when a note is
/// too short for it.
fn consonants(segs: &[Segment], body: BodyMix, k: f64) -> Vec<Segment> {
    segs.iter().map(|c| Segment { ipa: c.ipa.clone(), dur: Some(c.dur.unwrap_or_else(|| cdur(&c.ipa, body, k))), aspirated: c.aspirated, vot: c.vot, palatalized: c.palatalized, breathy: c.breathy, ..Default::default() }).collect()
}

/// A sung vowel (or one half of a diphthong) at the note's pitch, with the sound marks of the
/// syllable's nucleus.
fn nucleus_part(ipa: &str, dur: f64, hz: f64, nucleus: &Segment) -> Segment {
    Segment { ipa: ipa.to_string(), dur: Some(dur), f0: Some(hz), stress: Some(1), nasalized: nucleus.nasalized, breathy: nucleus.breathy, ..Default::default() }
}

/// Sung consonants at their note's pitch: an onset consonant moves the pitch to the new note
/// before its vowel (as singers do), a coda holds the old one.
fn consonants_at(segs: &[Segment], body: BodyMix, k: f64, hz: f64) -> Vec<Segment> {
    let mut out = consonants(segs, body, k);
    for c in out.iter_mut() {
        c.f0 = Some(hz);
    }
    out
}

/// A falling diphthong (or a triphthong) glides off over this share of its held part, unless its
/// syllable's rhyme is closing (Syllable::closing).
const OFFGLIDE: f64 = 0.2;

/// A rising nucleus's medial glide is sung as the last part of the note's onset, this long (s): a
/// quick transition that ends on the beat, co-produced with the initial consonant (the lips round
/// or the tongue fronts while the consonant is made), and the main vowel carries the beat. Sung on
/// the beat as a vowel of its own (up to 120 ms, a quarter of the vowel) it was a plateau: the jaw
/// and tongue sat at the glide's posture for 80-100 ms and the main vowel was 90% reached 200 ms
/// after the beat, and listening diagnoses of the renders heard 两 as li-ang, 快 as ku-ai (language
/// models, asked after a native speaker called the singing a foreigner's). As an onset consonant
/// before the consonants (the design before that) the glide was lost to the recogniser (话 hua
/// toward 哈 ha); it follows the initial, so it keeps its own time, and is a vowel segment, so it
/// takes the Öhman coarticulation of a vowel rather than a consonant's constriction.
const MEDIAL_GLIDE: f64 = 0.05;

/// A closing rhyme (`Syllable::closing`, Mandarin's) ends in one movement, over at most this share
/// of the vowel: on a short note the main vowel keeps the rest, on a long one the movement is as
/// long as the front end says and the vowel holds until then. Before, the ending was a part of its
/// own: 奥 ao sung for 1.3 s was [a] still for 0.76 s, a jump in 80 ms, then [ʊ] still for 0.35 s,
/// and 昂 ang was an oral [ɑ], a 40 ms jump of the tongue body, then 0.4 s of murmur: a native
/// listener heard two sounds (a-u, an-ng) where Mandarin has one.
const CLOSING_SHARE: f64 = 0.6;
/// In a closing rhyme the tongue body is this far on its way to a velar coda when the coda closes
/// (its strength against the vowel's 1.0: 1.2 is 55 % of the way), over this long at most (s). The
/// dorsum is slow, so -ng is heard in the vowel's last 150 ms before the closure.
const CLOSING_VELAR: f64 = 1.2;
const CLOSING_APPROACH_MAX: f64 = 0.15;
/// And toward an alveolar coda the tongue body fronts and rises (the body posture of an alveolar
/// closure, as `oral_gestures` holds it during one), 0.6: 37 % of the way. This is what tells -n
/// from -ng before the closure: 安 an's vowel ends fronted ([aɛ̃n], F2 rising), 昂 ang's backed.
const CLOSING_ALVEOLAR: f64 = 0.6;

/// a sung vowel keeps at least this long when a short note squeezes its consonants (s)
const MIN_VOWEL: f64 = 0.05;
/// a squeezed consonant keeps at least this long (s); a flap is 16 ms
const MIN_CONSONANT: f64 = 0.012;
/// A phrase's first movement begins this long before its first segment (s): the planner lays
/// segments out 60 ms after the plan's start, and a stop's closing begins 1 ms before that.
const PHRASE_LEAD: f64 = 0.062;

/// Fit the consonants between a vowel and the next one into `span`, the time from the vowel's
/// beat to the next vowel's (or to the end of a phrase's last note): when they leave the vowel
/// less than [`MIN_VOWEL`], they shrink in proportion (each down to [`MIN_CONSONANT`]), so the
/// next vowel still lands on its beat and a fast run doesn't drift later note by note.
fn fit(groups: [&mut [Segment]; 2], span: f64) {
    let [a, b] = groups;
    let cons = total(a) + total(b);
    let room = span - MIN_VOWEL;
    if cons <= room || cons <= 0.0 {
        return;
    }
    let scale = js_max(0.0, room) / cons;
    for c in a.iter_mut().chain(b.iter_mut()) {
        c.dur = Some(js_max(MIN_CONSONANT, c.dur.unwrap_or(0.0) * scale));
    }
}

fn total(segs: &[Segment]) -> f64 {
    segs.iter().fold(0.0, |s, c| s + c.dur.unwrap_or(0.0))
}

fn vibrato_gesture(t0: f64, t1: f64, depth: f64, tag: Option<f64>) -> Gesture {
    let mut g = Gesture::new(t0 + 0.22, t1, 0.3, 0.06, 4.0, Targets::default().with(ControlKey::VIBRATO, depth)).label("vibrato");
    g.tag = tag;
    g
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SungNote {
    pub midi: f64,
    /// length in beats
    pub beats: f64,
    pub syl: Syllable,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SongOptions {
    pub body: BodyMix,
    pub bpm: f64,
    /// plan time of the first note's beat (its onset consonants come before)
    pub t0: f64,
    /// vibrato depth (cents) on held vowels
    pub vibrato: f64,
}

/// A bend's blending strength against the note's own pitch (6): it wins while it lasts.
const BEND_STRENGTH: f64 = 30.0;

/// Pitch gestures for a syllable's bends on a vowel sung at `hz` from a to b.
fn bend_gestures(bends: &[Bend], hz: f64, a: f64, b: f64, out: &mut Vec<Gesture>) {
    let d = b - a;
    if d <= 0.02 {
        return;
    }
    for bend in bends {
        let len = js_min(bend.len * d, bend.max);
        let (t0, t1) = match bend.at {
            Edge::Start => (a - 0.03, a + len),
            Edge::Middle => ((a + b - len) / 2.0, (a + b + len) / 2.0),
            Edge::End => (b - len, b + 0.02),
        };
        let f = hz * crate::math::pow(2.0, bend.st / 12.0);
        out.push(Gesture::new(t0, t1, 0.015, 0.04, BEND_STRENGTH, Targets::default().with(ControlKey::F0, f)).label("bend"));
    }
}

fn melisma(s: &Syllable) -> bool {
    s.melisma == Some(true)
}

/// What a syllable's front end says the voice lags its onset's last consonant (s,
/// `Syllable::voice_lag`): the consonants are sung that much earlier, so the voice comes in where
/// the vowel is due and not an aspiration after it. Unsaid, or with no onset: none.
fn voice_lag(syl: &Syllable) -> f64 {
    if melisma(syl) || syl.onset.is_empty() { 0.0 } else { js_max(0.0, syl.voice_lag.unwrap_or(0.0)) }
}

/// How long before its beat a syllable's vowel starts (s): its voice lag. Not before a medial
/// glide: there the glide takes the lag at its own start (`medial_glide`), is voiced for as long
/// as before and ends on the beat. With the lag taken out of the glide instead, 快 kuai's
/// rounding was over before the voice began and the medial was lost (tests/medials.rs).
fn vowel_lead(syl: &Syllable) -> f64 {
    if onglide(&syl.nucleus.ipa).is_some() { 0.0 } else { voice_lag(syl) }
}

/// How long a rising nucleus's glide is sung in a song (s): its own time, and before it what the
/// voice lags the onset (the breathed start of 快 kuai's glide). A live note has no beat to be
/// ahead of and sings the glide's own time.
fn medial_glide(syl: &Syllable) -> f64 {
    MEDIAL_GLIDE + voice_lag(syl)
}

/// Plan a whole melody: one utterance per phrase, so each phrase gets its own breath.
pub fn plan_song(notes: &[SungNote], o: &SongOptions) -> Plan {
    let beat = 60.0 / o.bpm;
    let mut gestures: Vec<Gesture> = Vec::new();
    let mut segments: Vec<PlannedSegment> = Vec::new();
    let mut times: Vec<f64> = Vec::with_capacity(notes.len());
    let mut t = o.t0;
    for n in notes {
        times.push(t);
        t += n.beats * beat;
    }
    let mut end = t;
    // where each note's vowel starts: on its beat, or ahead of it by what the voice lags the onset
    let mut starts: Vec<f64> = notes.iter().zip(&times).map(|(n, t)| t - vowel_lead(&n.syl)).collect();
    let mut i = 0;
    while i < notes.len() {
        let mut j = i;
        while j < notes.len() - 1 && !notes[j].syl.phrase_end {
            j += 1;
        }
        let mut segs: Vec<Segment> = Vec::new();
        let mut vowel_spans: Vec<(f64, f64)> = Vec::new();
        // (note, vowel start, vowel end) of syllables with bends
        let mut bend_spans: Vec<(usize, f64, f64)> = Vec::new();
        // (vowel start, vowel end, fraction nasalised, the tongue body's approach to the coda) of
        // closing rhymes
        let mut closing_spans: Vec<(f64, f64, f64, Option<(f64, Targets)>)> = Vec::new();
        // a closing rhyme's nasal coda that ends the phrase: (its end, the closure it holds)
        let mut final_coda: Option<(f64, Targets)> = None;
        // each note's onset and coda consonants, at the note's pitch (a coda at the pitch of the
        // note whose syllable it ends; a melisma has neither)
        let hz_of = |k: usize| midi_to_hz(notes[k].midi);
        // a bend at the vowel's start begins in the onset: the consonants already sing its pitch, so
        // a rising tone starts low instead of starting at the note and dipping
        let onset_hz = |k: usize| {
            let start = notes[k].syl.pitch.iter().find(|b| b.at == Edge::Start).map_or(0.0, |b| b.st);
            hz_of(k) * crate::math::pow(2.0, start / 12.0)
        };
        let mut onsets: Vec<Vec<Segment>> = (i..=j)
            .map(|k| {
                if melisma(&notes[k].syl) {
                    return Vec::new();
                }
                let mut on = consonants_at(&notes[k].syl.onset, o.body, 0.8, onset_hz(k));
                // a rising nucleus's medial glide ends the onset, so it is done by the beat
                if let Some(a) = onglide(&notes[k].syl.nucleus.ipa) {
                    on.push(nucleus_part(a, medial_glide(&notes[k].syl), onset_hz(k), &notes[k].syl.nucleus));
                }
                on
            })
            .collect();
        let mut codas: Vec<Vec<Segment>> = (i..=j)
            .map(|k| if k < j && melisma(&notes[k + 1].syl) { Vec::new() } else { consonants_at(&last_of_chain(notes, k).coda, o.body, 0.8, hz_of(k)) })
            .collect();
        // the phrase's last vowel leaves a short breath before the next phrase. The call's last
        // phrase is followed by a rest (callers split phrases at rests), so the breath comes from
        // the rest: hold that note to its end, less the breath's release.
        let reserve = if j == notes.len() - 1 { 0.05 } else { 0.12 };
        let last_end = times[j] + notes[j].beats * beat - js_min(reserve, 0.3 * notes[j].beats * beat);
        // how long past the phrase's end its last sound may be held while the voice fades (a nasal
        // coda's closure): until the voice is gone, but when another phrase
        // follows in this call, only until 50 ms (the hold's release) before that phrase's first
        // consonant, so the hold is out of its way
        let fade = if j + 1 < notes.len() {
            let next = &notes[j + 1].syl;
            let onset = total(&consonants(&next.onset, o.body, 0.8)) + if onglide(&next.nucleus.ipa).is_some() { medial_glide(next) } else { 0.0 };
            js_min(0.15, js_max(0.0, starts[j + 1] - onset - 0.05 - last_end))
        } else {
            0.15
        };
        // a syllable that says so (`Syllable::onset_share`) shortens what comes before its vowel
        // to a share of the time since the vowel before: the coda there, its own consonants and
        // glide, and the voice's lag with its aspiration
        for k in i..j {
            let Some(share) = notes[k + 1].syl.onset_share.filter(|s| *s > 0.0) else { continue };
            let lead = times[k + 1] - starts[k + 1];
            let cons = total(&codas[k - i]) + total(&onsets[k + 1 - i]) + lead;
            let room = share * (times[k + 1] - times[k]);
            if cons > room && cons > 0.0 {
                let f = room / cons;
                for c in codas[k - i].iter_mut().chain(onsets[k + 1 - i].iter_mut()) {
                    c.dur = Some(js_max(MIN_CONSONANT, c.dur.unwrap_or(0.0) * f));
                    c.vot = c.vot.map(|v| v * f);
                }
                starts[k + 1] = times[k + 1] - lead * f;
            }
        }
        // the call's first phrase has only the time before its first beat, and the plan's time 0
        // is where a render starts. A syllable that leads the beat (`Syllable::voice_lag`)
        // shortens its consonants to fit, as it does on a fast note: sung 90 ms earlier, 窗
        // chuang's began 13 ms before time 0 from a beat at 0.3 s, and the start was cut
        if i == 0 && voice_lag(&notes[0].syl) > 0.0 {
            let lead = times[0] - starts[0];
            let cons = total(&onsets[0]) + lead;
            let room = times[0] - PHRASE_LEAD;
            if cons > room && room > 0.0 {
                let f = room / cons;
                for c in onsets[0].iter_mut() {
                    c.dur = Some(js_max(MIN_CONSONANT, c.dur.unwrap_or(0.0) * f));
                    c.vot = c.vot.map(|v| v * f);
                }
                starts[0] = times[0] - lead * f;
            }
        }
        for k in i..=j {
            let (span, next) = if k < j { (starts[k + 1] - starts[k], &mut onsets[k + 1 - i][..]) } else { (last_end - starts[k], &mut [][..]) };
            fit([&mut codas[k - i][..], next], span);
        }
        let first_onset = total(&onsets[0]);
        for k in i..=j {
            let n = &notes[k];
            let hz = hz_of(k);
            let next_melisma = k < j && melisma(&notes[k + 1].syl);
            let coda = std::mem::take(&mut codas[k - i]);
            let onset = std::mem::take(&mut onsets[k - i]);
            // vowel from this note's beat to the next note's onset (less this note's coda)
            let note_end = if k < j { starts[k + 1] - total(&onsets[k + 1 - i]) } else { last_end };
            // only consonants squeezed to their floor can leave less than MIN_VOWEL
            let vdur = js_max(0.02, note_end - starts[k] - total(&coda));
            segs.extend(onset);
            // the nucleus in time: the vowel that holds from the beat (a rising nucleus's glide
            // came with the onset), and a falling off-glide over the held part's last share (not
            // when a melisma continues the vowel)
            let nuc = n.syl.nucleus.ipa.as_str();
            let rest = vdur;
            // the syllable whose rhyme this is decides (a melisma's ending is its parent's)
            let closing = last_of_chain(notes, k).closing.filter(|c| *c > 0.0);
            match (offglide(nuc).filter(|_| !next_melisma), closing) {
                // a closing rhyme: the vowel moves into the off-glide over the note's end and the
                // glide is not held (half the movement is the glide coming in, half the vowel letting go)
                (Some(b), Some(c)) => {
                    let half = 0.5 * js_min(c, CLOSING_SHARE * rest);
                    segs.push(nucleus_part(held_vowel(nuc), rest - half, hz, &n.syl.nucleus));
                    segs.push(Segment { glide: Some(half), ..nucleus_part(b, half, hz, &n.syl.nucleus) });
                }
                (Some(b), None) => {
                    segs.push(nucleus_part(held_vowel(nuc), rest * (1.0 - OFFGLIDE), hz, &n.syl.nucleus));
                    segs.push(nucleus_part(b, rest * OFFGLIDE, hz, &n.syl.nucleus));
                }
                (None, _) => segs.push(nucleus_part(held_vowel(nuc), rest, hz, &n.syl.nucleus)),
            }
            if coda.first().is_some_and(|c| matches!(c.ipa.as_str(), "m" | "n" | "ŋ")) {
                // the syllable whose coda this is decides (a melisma's coda is its parent's)
                let lead = last_of_chain(notes, k).nasal_lead.unwrap_or(0.0);
                if closing.is_some() {
                    closing_spans.push((starts[k], starts[k] + vdur, lead, coda_approach(&coda[0].ipa, held_vowel(nuc), o.body)));
                    if k == j && coda.len() == 1 {
                        final_coda = closure(&coda[0].ipa, o.body).map(|t| (starts[k] + vdur + total(&coda), t));
                    }
                }
            }
            segs.extend(coda);
            vowel_spans.push((starts[k], starts[k] + vdur));
            // bends shape their syllable's first note (a melisma continues at the written pitches)
            if !n.syl.pitch.is_empty() && !melisma(&n.syl) {
                bend_spans.push((k, starts[k], starts[k] + vdur));
            }
        }
        if !segs.is_empty() {
            // the planner lays segments out from t0 + 0.06: start so the first vowel lands on its beat
            let plan = plan_utterance(&segs, &PlanOptions { body: o.body, t0: starts[i] - first_onset - 0.06, f0: Some(hz_of(i)), intonation: Some(Intonation::Flat), sung: Some(true), ..Default::default() });
            gestures.extend_from_slice(&plan.gestures);
            segments.extend(plan.segments);
            for &(k, a, b) in &bend_spans {
                bend_gestures(&notes[k].syl.pitch, hz_of(k), a, b, &mut gestures);
            }
            // a closing rhyme's nasal coda: the velum comes down gradually over the nasalised part of
            // the vowel (strength 2 follows the ramp itself, see Articulators) instead of dropping
            // at its start, and the tongue body is on its way to a velar coda before it closes
            for (a, b, lead, approach) in closing_spans {
                let len = js_min(lead * (b - a), 0.25);
                if len > 0.0 {
                    // open before the coda's closure starts (~55 ms ahead of b): a close vowel that
                    // closed first was a stop for a moment (东 dong dipped 9 dB before its -ng)
                    gestures.push(Gesture::new(b - 0.06, b + 0.02, len, 0.03, 2.0, Targets::default().with(ArticKey::Velum, 0.3)).label("nasal vowel"));
                }
                if let Some((pull, t)) = approach {
                    let len = js_min(CLOSING_APPROACH_MAX, 0.5 * (b - a));
                    gestures.push(Gesture::new(b - 0.01, b + 0.02, len, 0.03, pull, t).label("coda approach"));
                }
            }
            // and at the end of a phrase it stays closed, the velum open, until the voice has faded:
            // the closure used to let go as the breath stopped, a short vowel after the nasal
            // (安 an ended [an-ə], a 3.5 dB bump)
            if let Some((t, hold)) = final_coda {
                // (the velum stays down past the tongue's release, so no pressure is left to pop. The
                // release itself still measures -29 dB re the vowel, 0.2 s after the voice ends.)
                // Before another phrase the hold is only as long as `fade` and lets go quickly.
                let alone = j + 1 >= notes.len();
                let (held, release, velum) = if alone { (0.1, 0.12, 0.25) } else { (js_min(0.1, fade), 0.05, js_min(0.1, fade)) };
                gestures.push(Gesture::new(t - 0.03, t + held, 0.02, release, 40.0, hold).label("coda held"));
                gestures.push(Gesture::new(t - 0.03, t + velum, 0.02, 0.05, 4.0, Targets::default().with(ArticKey::Velum, 0.45)).label("velum"));
            }
            for (a, b) in vowel_spans {
                if b - a > 0.3 {
                    gestures.push(vibrato_gesture(a, b, o.vibrato, None));
                }
            }
            end = js_max(end, plan.end);
        }
        i = j + 1;
    }
    Plan { gestures, segments, end }
}

/// Where the tongue body heads in a closing rhyme's vowel before its nasal coda closes, and how
/// strongly against the vowel (see [`CLOSING_VELAR`], [`CLOSING_ALVEOLAR`]).
fn coda_approach(coda: &str, vowel: &str, body: BodyMix) -> Option<(f64, Targets)> {
    let v = vowel_artic(vowel, body);
    let at = |k: ArticKey| v.as_ref().and_then(|v| v.get(k));
    match coda {
        "ŋ" => {
            let mut t = Targets::default();
            // up toward the velum, unless the vowel's body is already close under it (ong, ing:
            // any higher and the vowel itself is choked off before the coda)
            let rise = at(ArticKey::TbY).is_none_or(|y| y < 6.95);
            if rise {
                t = t.with(ArticKey::TbY, 7.95);
            }
            // and back to the velar closure's place from a front vowel (英 ying [iə̃ŋ]: before, the
            // tongue closed where [i] was and slid back inside the murmur)
            let back = at(ArticKey::TbX).is_some_and(|x| x > 3.0);
            if back {
                t = t.with(ArticKey::TbX, 2.83);
            }
            (rise || back).then_some((CLOSING_VELAR, t))
        }
        "n" => Some((CLOSING_ALVEOLAR, Targets::default().with(ArticKey::TbX, 3.2).with(ArticKey::TbY, 7.4))),
        _ => None,
    }
}

/// A consonant's own closure: the targets of the articulators that make it.
fn closure(ipa: &str, body: BodyMix) -> Option<Targets> {
    let def = phoneme(ipa, body)?;
    let mut t = Targets::default();
    for (k, v) in def.targets.iter().filter(|(k, _)| def.dominant.contains(k)) {
        t = t.with(k, v);
    }
    Some(t)
}

/// The syllable whose coda ends a melisma chain starting at note k.
fn last_of_chain(notes: &[SungNote], k: usize) -> &Syllable {
    let mut b = k;
    while b > 0 && melisma(&notes[b].syl) {
        b -= 1;
    }
    &notes[b].syl
}

/// Live: gestures relative to "now" (the planner's 60 ms lead is mostly taken back).
const LIVE_T0: f64 = -0.045;
/// a held vowel lasts at most this long (s)
const HOLD: f64 = 30.0;

/// Consonants kept on each side of a live syllable's vowel (longer clusters are cut).
pub const MAX_CLUSTER: usize = 8;

/// A consonant held inline, with the marks of its [`Segment`] that sung consonants keep.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LiveConsonant {
    pub ipa: Label,
    /// the front end's duration (s), as [`Segment::dur`]
    pub dur: Option<f64>,
    pub aspirated: Option<bool>,
    pub vot: Option<f64>,
    pub palatalized: Option<bool>,
    pub breathy: Option<bool>,
}

/// Consonants held inline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cluster {
    n: u8,
    cs: [LiveConsonant; MAX_CLUSTER],
}

impl Cluster {
    fn from_segments(segs: &[Segment]) -> Cluster {
        let mut c = Cluster::default();
        for (slot, s) in c.cs.iter_mut().zip(segs) {
            *slot = LiveConsonant { ipa: Label::new(&s.ipa), dur: s.dur, aspirated: s.aspirated, vot: s.vot, palatalized: s.palatalized, breathy: s.breathy };
            c.n += 1;
        }
        c
    }

    pub fn iter(&self) -> impl Iterator<Item = &LiveConsonant> {
        self.cs[..self.n as usize].iter()
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// A [`Syllable`] as the live planners need it: `Copy`, so holding one costs no allocation and a
/// note's coda outlives a change of lyrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveSyllable {
    pub onset: Cluster,
    /// vowel (a diphthong stays one nucleus: "aɪ")
    pub nucleus: Label,
    /// the nucleus's marks, as [`Segment`]'s
    pub nasalized: Option<bool>,
    pub breathy: Option<bool>,
    pub coda: Cluster,
    /// continue the previous syllable's vowel at a new pitch
    pub melisma: bool,
}

impl LiveSyllable {
    /// Sung when the lyrics are empty.
    pub fn ah() -> LiveSyllable {
        LiveSyllable { onset: Cluster::default(), nucleus: Label::new("ɑ"), nasalized: None, breathy: None, coda: Cluster::default(), melisma: false }
    }
}

impl From<&Syllable> for LiveSyllable {
    fn from(s: &Syllable) -> LiveSyllable {
        LiveSyllable {
            onset: Cluster::from_segments(&s.onset),
            nucleus: Label::new(&s.nucleus.ipa),
            nasalized: s.nucleus.nasalized,
            breathy: s.nucleus.breathy,
            coda: Cluster::from_segments(&s.coda),
            melisma: s.melisma == Some(true),
        }
    }
}

/// [`NoteOnOptions`] over a [`LiveSyllable`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveNoteOn {
    pub body: BodyMix,
    pub vibrato: f64,
    pub tag: f64,
    /// the previous note of a legato pair: its syllable and MIDI note
    pub legato_from: Option<(LiveSyllable, f64)>,
}

/// The planner's input for one live note.
type LiveSegments<'a> = SmallVec<[TimedSegment<'a>; super::planner::INLINE_SEGMENTS]>;

/// Sung consonants, as [`consonants`]: the front end's duration, or shorter than spoken ones (see
/// [`cdur`]).
fn push_consonants<'a>(out: &mut LiveSegments<'a>, c: &'a Cluster, body: BodyMix, k: f64) {
    for c in c.iter() {
        let ipa = c.ipa.as_str();
        out.push(TimedSegment { ipa, dur: c.dur.unwrap_or_else(|| cdur(ipa, body, k)), aspirated: c.aspirated, vot: c.vot, palatalized: c.palatalized, breathy: c.breathy, ..Default::default() });
    }
}

/// A sung vowel (or one part of a nucleus) at the note's pitch, with the nucleus's marks, as
/// [`nucleus_part`].
fn live_nucleus_part<'a>(ipa: &'a str, dur: f64, hz: f64, syl: &LiveSyllable) -> TimedSegment<'a> {
    TimedSegment { ipa, dur, stress: Some(1), f0: Some(hz), nasalized: syl.nasalized, breathy: syl.breathy, ..Default::default() }
}

/// What follows a held vowel: a falling diphthong's (or a triphthong's) glide at the note's pitch
/// (a rising one already holds its vowel), and the coda.
fn push_tail<'a>(out: &mut LiveSegments<'a>, syl: &'a LiveSyllable, body: BodyMix, hz: f64) {
    if let Some(b) = offglide(syl.nucleus.as_str()) {
        out.push(live_nucleus_part(b, 0.07, hz, syl));
    }
    push_consonants(out, &syl.coda, body, 0.7);
}

/// Key pressed: sing the previous syllable's glide and coda if it is still sounding (legato),
/// then this syllable's onset, then its vowel held open-ended. Every gesture gets `tag`; the
/// pitch and vibrato get `tag + 1`, so a melisma can change pitch without ending the vowel.
/// Appends to `out`.
pub fn plan_note_on_into(syl: &LiveSyllable, midi: f64, o: &LiveNoteOn, out: &mut Vec<Gesture>) {
    let hz = midi_to_hz(midi);
    let mut segs = LiveSegments::new();
    if let Some((from, from_midi)) = &o.legato_from {
        push_tail(&mut segs, from, o.body, midi_to_hz(*from_midi));
    }
    push_consonants(&mut segs, &syl.onset, o.body, 0.7);
    let nucleus = syl.nucleus.as_str();
    // a rising nucleus's medial glide ends the onset
    if let Some(a) = onglide(nucleus) {
        segs.push(live_nucleus_part(a, MEDIAL_GLIDE, hz, syl));
    }
    segs.push(live_nucleus_part(held_vowel(nucleus), HOLD, hz, syl));
    let first = out.len();
    let timing = plan_timed(segs.into_iter(), &PlanOptions { body: o.body, t0: LIVE_T0, f0: Some(hz), intonation: Some(Intonation::Flat), open_end: Some(true), ..Default::default() }, out, None);
    for g in &mut out[first..] {
        g.tag = Some(if g.targets.has(ControlKey::F0) { o.tag + 1.0 } else { o.tag });
    }
    out.push(vibrato_gesture(timing.last.0, timing.last.1, o.vibrato, Some(o.tag + 1.0)));
}

/// Melisma, live: the held vowel moves to a new pitch (release tag + 1 first). Appends to `out`.
pub fn plan_melisma_into(midi: f64, o: &MelismaOptions, out: &mut Vec<Gesture>) {
    let MelismaOptions { vibrato, tag } = *o;
    out.push(Gesture::new(0.0, HOLD, 0.05, 0.06, 6.0, Targets::default().with(ControlKey::F0, midi_to_hz(midi))).label("pitch").tag(tag + 1.0));
    out.push(vibrato_gesture(0.0, HOLD, vibrato, Some(tag + 1.0)));
}

/// Key released: the diphthong's glide and the coda (release the note's tags first). Appends to
/// `out`; nothing when the syllable ends on its vowel.
pub fn plan_note_off_into(syl: &LiveSyllable, midi: f64, body: BodyMix, out: &mut Vec<Gesture>) {
    let hz = midi_to_hz(midi);
    let mut segs = LiveSegments::new();
    push_tail(&mut segs, syl, body, hz);
    if segs.is_empty() {
        return;
    }
    plan_timed(segs.into_iter(), &PlanOptions { body, t0: LIVE_T0, f0: Some(hz), intonation: Some(Intonation::Flat), ..Default::default() }, out, None);
}

/// The previous note of a legato pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegatoFrom {
    pub syl: Syllable,
    pub midi: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoteOnOptions {
    pub body: BodyMix,
    pub vibrato: f64,
    pub tag: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legato_from: Option<LegatoFrom>,
}

/// [`plan_note_on_into`] over a [`Syllable`].
pub fn plan_note_on(syl: &Syllable, midi: f64, o: &NoteOnOptions) -> Vec<Gesture> {
    let mut out = Vec::new();
    let legato_from = o.legato_from.as_ref().map(|l| (LiveSyllable::from(&l.syl), l.midi));
    plan_note_on_into(&syl.into(), midi, &LiveNoteOn { body: o.body, vibrato: o.vibrato, tag: o.tag, legato_from }, &mut out);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MelismaOptions {
    pub vibrato: f64,
    pub tag: f64,
}

/// [`plan_melisma_into`].
pub fn plan_melisma(midi: f64, o: &MelismaOptions) -> Vec<Gesture> {
    let mut out = Vec::with_capacity(2);
    plan_melisma_into(midi, o, &mut out);
    out
}

/// [`plan_note_off_into`] over a [`Syllable`].
pub fn plan_note_off(syl: &Syllable, midi: f64, body: BodyMix) -> Vec<Gesture> {
    let mut out = Vec::new();
    plan_note_off_into(&syl.into(), midi, body, &mut out);
    out
}
