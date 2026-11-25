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
//!
//! At any depth a script may carry `tuning`, the engine's own numbers (`chimera::tuning`): the
//! voice is built from it and every phrase is planned with it.

use chimera::anatomy::human::BodyMix;
use chimera::control::score::Gesture;
use chimera::engine::default_f0;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::{Intonation, PlannedSegment, Segment};
use chimera::tuning::Tuning;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

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
    /// where each phrase's pitch starts, in semitones above `f0`, and how far it falls by its end
    #[serde(default)]
    pub pitch_start: Option<f64>,
    #[serde(default)]
    pub pitch_fall: Option<f64>,
    /// the voice source for the whole render: pressure, breathiness, tension, fry, roughness,
    /// vibratoDepth, vibratoRate, drift, brightness
    #[serde(default)]
    pub source: Option<SourcePatch>,
    /// the engine's numbers for the whole script: the string `"upstream"` (the author's voice,
    /// `Tuning::upstream()`) or a partial tuning object (`{"source":{"oqBase":0.7}}`: what is left
    /// out keeps its default, and a name that is not a field is refused). The voice is built from
    /// it and every phrase is planned with it, and a script of gestures alone still takes it for
    /// the voice. The script's own `accent`, `pitchStart`, `pitchFall` and `source` win over it.
    /// With `"upstream"` (or `prosody.upstreamLine`) the pitch is the old line, which reads
    /// neither `pitchStart` nor `pitchFall`: they are accepted and change nothing.
    #[serde(default, deserialize_with = "tuning_in")]
    pub tuning: Option<Tuning>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_start: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_fall: Option<f64>,
}

/// A script with everything decided: what `?plan=1` answers, and itself a script.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Planned {
    pub body: BodyMix,
    pub f0: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourcePatch>,
    /// the script's tuning, whole (`"upstream"` as that word), so the plan sent back sounds the same
    #[serde(skip_serializing_if = "Option::is_none", serialize_with = "tuning_out")]
    pub tuning: Option<Tuning>,
    /// the phrases as phones (layer 2)
    pub phrases: Vec<PhraseScript>,
    /// the gestural score (layer 3)
    pub gestures: Vec<Gesture>,
    pub seconds: f64,
    /// when each phone sounds (information only: ignored when sent back)
    pub timeline: Vec<PlannedSegment>,
}

/// A tuning as JSON says it, in a script or in a file of its own (`chimera render text --tuning`):
/// the string `"upstream"` or a partial tuning object.
pub fn parse_tuning(v: Value) -> Result<Tuning, String> {
    match v {
        Value::String(s) if s == "upstream" => Ok(Tuning::upstream()),
        Value::String(s) => Err(format!("tuning: {s:?} is not a tuning: the only one with a name is \"upstream\"")),
        v @ Value::Object(_) => serde_json::from_value(v).map_err(|e| format!("tuning: {e}")),
        _ => Err("tuning: give \"upstream\" or an object of tuning fields".into()),
    }
}

fn tuning_in<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Tuning>, D::Error> {
    match Value::deserialize(d)? {
        Value::Null => Ok(None),
        v => parse_tuning(v).map(Some).map_err(serde::de::Error::custom),
    }
}

fn tuning_out<S: Serializer>(t: &Option<Tuning>, s: S) -> Result<S::Ok, S::Error> {
    match t {
        Some(t) if *t == Tuning::upstream() => s.serialize_str("upstream"),
        t => t.serialize(s),
    }
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
    let pitch_start = finite("pitchStart", s.pitch_start, -24.0, 24.0)?;
    let pitch_fall = finite("pitchFall", s.pitch_fall, -24.0, 24.0)?;
    let tuning = s.tuning;

    if let Some(gestures) = &s.gestures {
        if gestures.iter().any(|x| !(x.t0.is_finite() && x.t1.is_finite() && x.t0 >= 0.0 && x.t1 >= x.t0)) {
            return Err("every gesture needs 0 <= t0 <= t1".into());
        }
        let end = gestures.iter().fold(0.0f64, |m, x| m.max(x.t1));
        let seconds = finite("seconds", s.seconds, 0.0, MAX_SECONDS)?.unwrap_or(end + TAIL);
        return Ok(Planned { body, f0, source: s.source.clone(), tuning, phrases: Vec::new(), gestures: gestures.clone(), seconds, timeline: Vec::new() });
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
        finite("phrase pitchStart", p.pitch_start, -24.0, 24.0)?;
        finite("phrase pitchFall", p.pitch_fall, -24.0, 24.0)?;
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
            pitch_start: p.pitch_start.or(pitch_start),
            pitch_fall: p.pitch_fall.or(pitch_fall),
            // no tuning in the script is no tuning in the plan: the planner's default
            tuning: tuning.map(|t| t.plan()),
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
    Ok(Planned { body, f0, source: s.source.clone(), tuning, phrases: phones, gestures, seconds: end + TAIL, timeline })
}

/// The sound of a planned script.
pub fn render(p: &Planned) -> Result<Vec<f32>, String> {
    if !(p.seconds > 0.0 && p.seconds <= MAX_SECONDS) {
        return Err(format!("a render lasts up to {MAX_SECONDS} s"));
    }
    let mut voice = HumanVoice::with_tuning(FS, p.body.body(), 2, &p.tuning.unwrap_or_default());
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

    fn said(g: &EnglishG2P, json: &str) -> Vec<f32> {
        render(&plan(g, &script(json)).unwrap()).unwrap()
    }

    #[test]
    fn a_default_tuning_is_no_tuning() {
        let g = EnglishG2P::new();
        let plain = said(&g, r#"{"text":"A cat sat."}"#);
        assert!(plain == said(&g, r#"{"text":"A cat sat.","tuning":{}}"#));
        assert!(plain == said(&g, r#"{"text":"A cat sat.","tuning":null}"#));
        let whole = format!(r#"{{"text":"A cat sat.","tuning":{}}}"#, serde_json::to_string(&Tuning::default()).unwrap());
        assert!(plain == said(&g, &whole));
    }

    #[test]
    fn a_tuning_changes_the_sound() {
        let g = EnglishG2P::new();
        // `closureBright` multiplies the brightness off its default, so the script's source sets one
        let bright = said(&g, r#"{"text":"A cat sat.","source":{"brightness":0.6}}"#);
        assert!(bright != said(&g, r#"{"text":"A cat sat.","source":{"brightness":0.6},"tuning":{"source":{"closureBright":1.2}}}"#));
        // one field of each part that is wired its own way: the glottis, the airway, the noise
        // and the losses, the articulators, and the planner's pitch and durations
        let plain = said(&g, r#"{"text":"A cat sat."}"#);
        for t in [
            r#"{"source":{"oqBase":0.7}}"#,
            r#"{"airway":{"glottisKineticIn":0.5}}"#,
            r#"{"noise":{"teethGain":4.5e-6}}"#,
            r#"{"physics":{"wallMass":2.0}}"#,
            r#"{"artic":{"jaw":7}}"#,
            r#"{"prosody":{"pitchFall":4}}"#,
            r#"{"durations":{"vowelFinal":1.6}}"#,
        ] {
            let out = said(&g, &format!(r#"{{"text":"A cat sat.","tuning":{t}}}"#));
            assert!(out != plain, "{t} changed nothing");
            assert!(out.iter().all(|x| x.is_finite()) && out.iter().any(|x| x.abs() > 0.01), "{t}");
        }
        // the script's own source wins over the tuning's
        let loud = r#"{"text":"A cat sat.","source":{"pressure":9}}"#;
        assert!(said(&g, loud) == said(&g, r#"{"text":"A cat sat.","source":{"pressure":9},"tuning":{"source":{"pressure":6}}}"#));
    }

    #[test]
    fn a_tuning_that_is_not_one_is_refused_by_name() {
        let refused = |json: &str| serde_json::from_str::<Script>(json).map(|_| ()).unwrap_err().to_string();
        let e = refused(r#"{"text":"hi","tuning":{"source":{"oqBaze":0.7}}}"#);
        assert!(e.contains("tuning") && e.contains("oqBaze"), "{e}");
        let e = refused(r#"{"text":"hi","tuning":{"sauce":{}}}"#);
        assert!(e.contains("sauce"), "{e}");
        let e = refused(r#"{"text":"hi","tuning":"upstrem"}"#);
        assert!(e.contains("upstrem") && e.contains("\"upstream\""), "{e}");
        assert!(refused(r#"{"text":"hi","tuning":3}"#).contains("tuning"));
        assert!(refused(r#"{"text":"hi","tuning":{"source":{"oqBase":"high"}}}"#).contains("tuning"));
        assert_eq!(parse_tuning(serde_json::json!("upstream")), Ok(Tuning::upstream()));
        assert_eq!(parse_tuning(serde_json::json!({})), Ok(Tuning::default()));
    }

    #[test]
    fn upstream_is_the_authors_pitch_line() {
        use chimera::control::score::ControlKey;
        let g = EnglishG2P::new();
        let pitches = |p: &Planned| p.gestures.iter().filter(|x| x.targets.has(ControlKey::F0)).map(|x| x.t0).collect::<Vec<f64>>();
        let new = plan(&g, &script(r#"{"text":"A cat sat on the mat."}"#)).unwrap();
        let old = plan(&g, &script(r#"{"text":"A cat sat on the mat.","tuning":"upstream"}"#)).unwrap();
        assert_eq!(old.tuning, Some(Tuning::upstream()));
        let vowels: Vec<f64> = old.timeline.iter().filter(|s| s.kind == "vowel").map(|s| s.t0).collect();
        // the old line: a pitch gesture a vowel (the last vowel's nuclear peak and final low are
        // two) and none before the first; the new one has a line and an accent, and starts the phrase
        let (was, now) = (pitches(&old), pitches(&new));
        assert!(was.len() == vowels.len() || was.len() == vowels.len() + 1, "{} pitch gestures for {} vowels", was.len(), vowels.len());
        assert!(was.iter().all(|t| *t >= vowels[0]));
        assert!(now.len() > vowels.len() + 1 && now.iter().any(|t| *t < vowels[0]));
        // only the pitch differs, and it is heard
        // (as JSON: an unset target is a NaN, which equals nothing)
        let json = |x: &[Gesture]| serde_json::to_string(x).unwrap();
        let rest = |p: &Planned| json(&p.gestures.iter().filter(|x| !x.targets.has(ControlKey::F0)).copied().collect::<Vec<_>>());
        assert!(rest(&old) == rest(&new));
        assert!(render(&old).unwrap() != render(&new).unwrap());
        // the old line knows no `pitchStart` or `pitchFall`: with it they change nothing
        let moved = plan(&g, &script(r#"{"text":"A cat sat on the mat.","tuning":"upstream","pitchStart":2,"pitchFall":3}"#)).unwrap();
        assert!(json(&moved.gestures) == json(&old.gestures));
        // while `accent` is still the script's
        let flat = plan(&g, &script(r#"{"text":"A cat sat on the mat.","tuning":"upstream","accent":0}"#)).unwrap();
        assert!(json(&flat.gestures) != json(&old.gestures));
    }

    #[test]
    fn a_tuned_plan_sent_back_renders_the_same_sound() {
        let g = EnglishG2P::new();
        let plain = said(&g, r#"{"text":"A cat sat."}"#);
        for t in [r#""upstream""#, r#"{"source":{"oqBase":0.7},"airway":{"lipOpenRate":5},"prosody":{"pitchFall":4}}"#] {
            let p = plan(&g, &script(&format!(r#"{{"text":"A cat sat.","tuning":{t}}}"#))).unwrap();
            let a = render(&p).unwrap();
            assert!(a != plain, "{t}");
            let sent = serde_json::to_value(&p).unwrap();
            assert_eq!(sent["tuning"].is_string(), t.starts_with('"'), "upstream goes back as its name");
            let p2 = plan(&g, &script(&sent.to_string())).unwrap();
            assert_eq!(p2.tuning, p.tuning);
            // the plan sent back is its gestures, and the voice that says them still takes the tuning
            assert!(a == render(&p2).unwrap(), "{t}");
        }
        // and a plan without a tuning carries none
        let p = plan(&g, &script(r#"{"text":"A cat sat."}"#)).unwrap();
        assert!(serde_json::to_value(&p).unwrap().get("tuning").is_none());
    }
}
