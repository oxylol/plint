//! The JSON script: what a request to `/api/speak` may say, and the render it becomes.
//!
//! Three depths, each a superset of the one before, so a script can start as text and be taken
//! over by hand one layer at a time:
//!
//! 1. `text`: English text, read by the G2P and the phrase planner (what `chimera render text` does).
//! 2. `phrases`: each phrase is `text`, `ipa`, or `segments` (phones with their own duration, stress
//!    and pitch), with its own intonation, pause, rate and pitch.
//! 3. `gestures`: the gestural score itself (articulator and source targets over time). When a
//!    script has gestures, they are what is rendered; `phrases` and `text` are ignored.
//!
//! `?plan=1` returns layers 2 and 3 for a script instead of the sound, so the answer to one
//! request is a valid script for the next.

use chimera::anatomy::human::BodyMix;
use chimera::control::score::Gesture;
use chimera::engine::default_f0;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::{Intonation, PlannedSegment, Segment};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde::{Deserialize, Serialize};

/// The output rate of every render.
pub const FS: f64 = 48000.0;
/// The longest render one request may ask for (s).
pub const MAX_SECONDS: f64 = 120.0;
/// Silence before the first phrase and after the last gesture (s), as `render text`.
const LEAD: f64 = 0.05;
const TAIL: f64 = 0.3;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Script {
    /// "male", "female", "child" or `{ "female": 0..1, "child": 0..1 }`
    #[serde(default)]
    pub body: Option<BodyMix>,
    /// baseline pitch (Hz); the body's own when absent
    #[serde(default)]
    pub f0: Option<f64>,
    /// speaking-rate multiplier (>1 = faster)
    #[serde(default)]
    pub rate: Option<f64>,
    /// pitch excursion on stressed syllables (fraction)
    #[serde(default)]
    pub accent: Option<f64>,
    /// the voice source for the whole render: pressure, breathiness, tension, fry, roughness,
    /// vibratoDepth, vibratoRate, drift, brightness
    #[serde(default)]
    pub source: Option<SourcePatch>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub phrases: Option<Vec<PhraseScript>>,
    #[serde(default)]
    pub gestures: Option<Vec<Gesture>>,
    /// length of a gesture-only render (s); the last gesture's end plus the tail when absent
    #[serde(default)]
    pub seconds: Option<f64>,
    /// a plan's timeline, accepted and ignored so that a plan can be sent back as it came
    #[serde(default)]
    #[allow(dead_code)]
    pub timeline: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhraseScript {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ipa: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intonation: Option<Intonation>,
    /// pause after the phrase (s)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub f0: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accent: Option<f64>,
}

/// A script with everything decided: what `?plan=1` answers, and itself a script.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Planned {
    pub body: BodyMix,
    pub f0: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourcePatch>,
    /// the phrases as phones (layer 2)
    pub phrases: Vec<PhraseScript>,
    /// the gestural score (layer 3)
    pub gestures: Vec<Gesture>,
    pub seconds: f64,
    /// when each phone sounds (information only: ignored when sent back)
    pub timeline: Vec<PlannedSegment>,
}

fn finite(name: &str, x: Option<f64>, lo: f64, hi: f64) -> Result<Option<f64>, String> {
    match x {
        Some(v) if !v.is_finite() || v < lo || v > hi => Err(format!("{name} must be between {lo} and {hi}")),
        x => Ok(x),
    }
}

/// Decide everything a script leaves open: phones for its text, gestures for its phones.
pub fn plan(g: &EnglishG2P, s: &Script) -> Result<Planned, String> {
    let body = s.body.unwrap_or(BodyMix::MALE);
    let f0 = finite("f0", s.f0, 40.0, 800.0)?.unwrap_or_else(|| default_f0(body));
    let rate = finite("rate", s.rate, 0.25, 4.0)?;
    let accent = finite("accent", s.accent, 0.0, 2.0)?;

    if let Some(gestures) = &s.gestures {
        if gestures.iter().any(|x| !(x.t0.is_finite() && x.t1.is_finite() && x.t0 >= 0.0 && x.t1 >= x.t0)) {
            return Err("every gesture needs 0 <= t0 <= t1".into());
        }
        let end = gestures.iter().fold(0.0f64, |m, x| m.max(x.t1));
        let seconds = finite("seconds", s.seconds, 0.0, MAX_SECONDS)?.unwrap_or(end + TAIL);
        return Ok(Planned { body, f0, source: s.source.clone(), phrases: Vec::new(), gestures: gestures.clone(), seconds, timeline: Vec::new() });
    }

    // text is a phrase list the G2P writes
    let mut phrases: Vec<PhraseScript> = s.phrases.clone().unwrap_or_default();
    if phrases.is_empty() {
        let text = s.text.as_deref().unwrap_or("").trim();
        if text.is_empty() {
            return Err("nothing to say: give text, phrases or gestures".into());
        }
        phrases.push(PhraseScript { text: Some(text.to_string()), ..Default::default() });
    }

    // every phrase down to phones (a text phrase may be several: it splits on its punctuation)
    let mut phones: Vec<PhraseScript> = Vec::new();
    for p in &phrases {
        finite("phrase rate", p.rate, 0.25, 4.0)?;
        finite("phrase f0", p.f0, 40.0, 800.0)?;
        finite("phrase accent", p.accent, 0.0, 2.0)?;
        finite("pause", p.pause, 0.0, 10.0)?;
        if let Some(segs) = &p.segments {
            phones.push(PhraseScript { text: None, ipa: None, segments: Some(segs.clone()), ..p.clone() });
        } else if let Some(ipa) = &p.ipa {
            phones.push(PhraseScript { text: None, ipa: None, segments: Some(tokenize_ipa(ipa)), ..p.clone() });
        } else if let Some(text) = &p.text {
            let said = g.to_phrases(text);
            let n = said.len();
            for (i, q) in said.into_iter().enumerate() {
                let last = i + 1 == n;
                phones.push(PhraseScript {
                    text: None,
                    ipa: None,
                    segments: Some(q.segments),
                    // the script's own intonation and pause belong to the phrase's end
                    intonation: if last { p.intonation.or(Some(q.intonation)) } else { Some(q.intonation) },
                    pause: if last { p.pause.or(Some(q.pause)) } else { Some(q.pause) },
                    ..p.clone()
                });
            }
        } else {
            return Err("a phrase needs text, ipa or segments".into());
        }
    }

    let mut gestures = Vec::new();
    let mut timeline = Vec::new();
    let mut t = LEAD;
    let mut end = LEAD;
    for p in &phones {
        let segs = p.segments.as_deref().unwrap_or(&[]);
        if segs.is_empty() {
            t += p.pause.unwrap_or(0.25);
            continue;
        }
        let o = PlanOptions {
            body,
            t0: t,
            rate: p.rate.or(rate),
            f0: Some(p.f0.unwrap_or(f0)),
            intonation: Some(p.intonation.unwrap_or_default()),
            accent: p.accent.or(accent),
            ..Default::default()
        };
        let plan = plan_utterance(segs, &o);
        gestures.extend_from_slice(&plan.gestures);
        timeline.extend(plan.segments);
        end = plan.end;
        t = plan.end + p.pause.unwrap_or(0.25);
        if t > MAX_SECONDS {
            return Err(format!("too long: more than {MAX_SECONDS} s of speech in one request"));
        }
    }
    Ok(Planned { body, f0, source: s.source.clone(), phrases: phones, gestures, seconds: end + TAIL, timeline })
}

/// The sound of a planned script.
pub fn render(p: &Planned) -> Result<Vec<f32>, String> {
    if !(p.seconds > 0.0 && p.seconds <= MAX_SECONDS) {
        return Err(format!("a render lasts up to {MAX_SECONDS} s"));
    }
    let mut voice = HumanVoice::new(FS, p.body.body(), 2);
    voice.set_source(&SourcePatch::f0(p.f0));
    if let Some(src) = &p.source {
        // the script's pitch is `f0`; a source patch carries the rest
        voice.set_source(&SourcePatch { f0: None, ..src.clone() });
    }
    voice.schedule(&p.gestures);
    let mut out = vec![0f32; (FS * p.seconds).round() as usize];
    for block in out.chunks_mut(128) {
        voice.render(block);
    }
    Ok(out)
}

/// A mono 16-bit WAV (non-finite samples as silence).
pub fn wav(x: &[f32]) -> Vec<u8> {
    let n = (x.len() * 2) as u32;
    let fs = FS as u32;
    let mut b = Vec::with_capacity(44 + n as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + n).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&fs.to_le_bytes());
    b.extend_from_slice(&(fs * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&n.to_le_bytes());
    for &s in x {
        let v = if s.is_finite() { (s as f64).clamp(-1.0, 1.0) } else { 0.0 };
        b.extend_from_slice(&((v * 32767.0).round() as i16).to_le_bytes());
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script(json: &str) -> Script {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn text_becomes_phones_and_gestures_and_sound() {
        let g = EnglishG2P::new();
        let p = plan(&g, &script(r#"{"text":"Hello there."}"#)).unwrap();
        assert!(!p.phrases.is_empty() && !p.gestures.is_empty());
        let out = render(&p).unwrap();
        assert_eq!(out.len(), (FS * p.seconds).round() as usize);
        assert!(out.iter().any(|x| x.abs() > 0.01), "the render is silent");
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_plan_is_a_script_that_renders_the_same_sound() {
        let g = EnglishG2P::new();
        let p = plan(&g, &script(r#"{"text":"A cat sat.","body":"female","rate":1.2}"#)).unwrap();
        let a = render(&p).unwrap();
        let p2 = plan(&g, &script(&serde_json::to_string(&p).unwrap())).unwrap();
        assert_eq!(a, render(&p2).unwrap());
    }

    #[test]
    fn rate_and_pitch_change_the_sound() {
        let g = EnglishG2P::new();
        let plain = plan(&g, &script(r#"{"text":"Good morning."}"#)).unwrap();
        let fast = plan(&g, &script(r#"{"text":"Good morning.","rate":1.5}"#)).unwrap();
        let high = plan(&g, &script(r#"{"text":"Good morning.","f0":160}"#)).unwrap();
        assert!(fast.seconds < plain.seconds - 0.1);
        assert_eq!(high.f0, 160.0);
        assert_ne!(render(&plain).unwrap(), render(&high).unwrap());
    }

    #[test]
    fn a_phrase_can_be_phones_with_their_own_length() {
        let g = EnglishG2P::new();
        let short = plan(&g, &script(r#"{"phrases":[{"segments":[{"ipa":"m"},{"ipa":"ɑ","stress":1}]}]}"#)).unwrap();
        let long = plan(&g, &script(r#"{"phrases":[{"segments":[{"ipa":"m"},{"ipa":"ɑ","stress":1,"dur":0.8}]}]}"#)).unwrap();
        assert!(long.seconds > short.seconds + 0.3);
    }

    #[test]
    fn what_cannot_be_said_is_refused() {
        let g = EnglishG2P::new();
        assert!(plan(&g, &script("{}")).is_err());
        assert!(plan(&g, &script(r#"{"text":"hi","rate":0}"#)).is_err());
        assert!(serde_json::from_str::<Script>(r#"{"txt":"hi"}"#).is_err());
        let long = format!(r#"{{"phrases":[{}]}}"#, vec![r#"{"text":"one two three four five six seven","pause":10}"#; 20].join(","));
        assert!(plan(&g, &script(&long)).is_err());
    }
}
