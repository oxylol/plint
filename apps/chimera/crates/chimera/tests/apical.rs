//! The apical vowels (ɿ in 资 词 思, ʅ in 只 吃 师 日) are the consonant's own tongue-tip
//! constriction, held open and voiced, with the lips spread: the tongue tip hardly moves from the
//! consonant into the vowel. They were once fitted to their formants alone, which the calibration
//! did with narrowed lips and a backed tongue body and no tip at all (ʅ: lipAp 0.23, tip at rest),
//! and listening diagnoses of the renders (language models) heard 只 as zhu, zhe or zhi.
use chimera::anatomy::human::{ArticKey, BODY_CHILD, BODY_FEMALE, BODY_MALE, HumanBody};
use chimera::phonetics::inventory::phoneme;
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::planner::vowel_table;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};

const BODIES: [(&str, HumanBody); 3] = [("male", BODY_MALE), ("female", BODY_FEMALE), ("child", BODY_CHILD)];

#[test]
fn the_table_has_spread_lips_and_the_tip_of_the_consonant() {
    let t = vowel_table();
    for (name, _) in BODIES {
        let i = &t[name]["i"].artic;
        for (v, cons, curled) in [("ɿ", "s", false), ("ʅ", "ʂ", true)] {
            let a = &t[name][v].artic;
            let c = phoneme(cons, name.into()).unwrap().targets;
            let g = |p: &chimera::anatomy::human::ArticPatch, k| p.get(k).unwrap();
            // lips as for [i]: spread, not protruded, not narrowed (they were 0.23 aperture, 0.35 protrusion)
            assert!(g(a, ArticKey::LipProt) <= 0.01, "{name} {v}: lips protruded");
            assert!(g(a, ArticKey::LipWidth) >= g(i, ArticKey::LipWidth) - 1e-9, "{name} {v}: lips not spread");
            assert!(g(a, ArticKey::LipAp) >= g(i, ArticKey::LipAp) - 1e-9, "{name} {v}: lips narrowed");
            // the tip where the sibilant before it puts it, the constriction opened by 0.11
            assert!((g(a, ArticKey::TtX) - g(&c, ArticKey::TtX)).abs() < 0.01, "{name} {v}: tip x");
            assert!((g(&c, ArticKey::TtY) - g(a, ArticKey::TtY) - 0.11).abs() < 0.01, "{name} {v}: tip y");
            assert_eq!(g(a, ArticKey::TtCurl) > 0.8, curled, "{name} {v}: curl");
        }
    }
}

const FS: f64 = 48000.0;
const BEAT_AT: f64 = 0.5;

/// the articulators at `ms` after the beat, a one-beat note at 90 bpm, and the voice's samples
fn sing(py: &str, name: &str, body: HumanBody, fs: f64) -> (impl Fn(ArticKey, f64) -> f64, Vec<f32>) {
    let syl = pinyin_syllable(py, py).unwrap();
    let plan = plan_song(&[SungNote { midi: if name == "male" { 50.0 } else { 62.0 }, beats: 1.0, syl }], &SongOptions { body: name.into(), bpm: 90.0, t0: BEAT_AT, vibrato: 0.0, ..Default::default() });
    let mut v = HumanVoice::new(fs, body, oversample_for(fs));
    v.set_source(&SourcePatch::f0(if name == "male" { 110.0 } else { 220.0 }));
    v.schedule(&plan.gestures);
    let step = (fs / 200.0) as usize; // 5 ms
    let mut out = vec![0f32; (fs * (BEAT_AT + 0.7)) as usize / step * step];
    let mut frames = vec![ArticKey::ALL.iter().map(|k| v.artic.x.get(*k)).collect::<Vec<f64>>()];
    for block in out.chunks_mut(step) {
        v.render(block);
        frames.push(ArticKey::ALL.iter().map(|k| v.artic.x.get(*k)).collect());
    }
    (move |k: ArticKey, ms: f64| frames[(((BEAT_AT + ms / 1000.0) * 200.0).round() as usize).min(frames.len() - 1)][k.index()], out)
}

#[test]
fn the_tip_barely_moves_from_the_consonant_into_the_vowel() {
    // 只 吃 师 日 (ʅ) and 资 词 思 (ɿ): the consonant's release (just before the beat) against the
    // vowel's middle. Measured: the tip 0.07 or less in x, 0.05-0.28 in y, curl 0.13 at most (before:
    // y -1.7 to -2.5, curl -0.9 for ʅ), the lips open and spread (aperture 1.0, was 0.2)
    for py in ["zhi1", "chi1", "shi1", "ri4", "zi1", "ci1", "si1"] {
        let (a, _) = sing(py, "male", BODY_MALE, FS);
        let d = |k| (a(k, 250.0) - a(k, -25.0)).abs();
        assert!(d(ArticKey::TtX) <= 0.15, "{py}: tip x moves {:.2}", d(ArticKey::TtX));
        assert!(d(ArticKey::TtY) <= 0.4, "{py}: tip y moves {:.2}", d(ArticKey::TtY));
        assert!(d(ArticKey::TtCurl) <= 0.2, "{py}: curl moves {:.2}", d(ArticKey::TtCurl));
        assert!(a(ArticKey::LipAp, 250.0) >= 0.9 && a(ArticKey::LipProt, 250.0) <= 0.05, "{py}: lips");
        // and the lips are not rounded while the consonant is made either
        assert!(a(ArticKey::LipProt, -25.0) <= 0.15, "{py}: lips rounded at the release");
    }
}

/// normalised autocorrelation peak (3-20 ms lag) over 60 ms frames: 1 is a clean periodic voice,
/// turbulent noise in the vowel pulls it down
fn periodicity(x: &[f32], fs: f64, a: f64, b: f64) -> f64 {
    let (len, hop) = ((0.06 * fs) as usize, (0.02 * fs) as usize);
    let s = &x[(a * fs) as usize..(b * fs) as usize];
    let mut peaks = Vec::new();
    let mut k = 0;
    while k + len <= s.len() {
        let m = s[k..k + len].iter().map(|v| *v as f64).sum::<f64>() / len as f64;
        let fr: Vec<f64> = s[k..k + len].iter().map(|v| *v as f64 - m).collect();
        let ac = |lag: usize| fr.iter().zip(&fr[lag..]).map(|(p, q)| p * q).sum::<f64>();
        let r0 = ac(0) + 1e-12;
        peaks.push(((0.003 * fs) as usize..(0.02 * fs) as usize).map(|l| ac(l) / r0).fold(0.0, f64::max));
        k += hop;
    }
    peaks.iter().sum::<f64>() / peaks.len() as f64
}

#[test]
fn they_sing_steady_and_periodic_on_every_body_and_at_a_low_sample_rate() {
    // the tip's constriction (0.3 cm2) must not hiss or go unstable: no NaN, a clear voice, as periodic as before (0.83-0.89)
    for (name, body) in BODIES {
        for fs in [48000.0, 16000.0] {
            for py in ["zhi1", "si1"] {
                let (_, out) = sing(py, name, body, fs);
                assert!(out.iter().all(|x| x.is_finite()), "{name} {py} at {fs}: NaN");
                let (a, b) = (BEAT_AT + 0.15, BEAT_AT + 0.5);
                let rms = (out[(a * fs) as usize..(b * fs) as usize].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ((b - a) * fs)).sqrt();
                assert!(rms > 0.02, "{name} {py} at {fs}: silent vowel (RMS {rms:.4})");
                let p = periodicity(&out, fs, a, b);
                assert!(p > 0.8, "{name} {py} at {fs}: periodicity {p:.2}");
            }
        }
    }
}
