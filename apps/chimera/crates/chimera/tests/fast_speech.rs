//! Fast speech (`rate` above 1, the Speak panel's slider goes to 1.8): the vowels between
//! consonants still open.
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 128;

/// The largest opening (cm²) of each vowel between two consonants: the tract's narrowest area,
/// at its widest moment in the vowel.
fn inner_vowel_openings(ipa: &str, rate: f64) -> Vec<f64> {
    let mut v = HumanVoice::with_defaults(FS);
    let plan = plan_utterance(&tokenize_ipa(ipa), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), rate: Some(rate), ..Default::default() });
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let mut narrowest = vec![];
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        let f = v.airway.frame.as_ref().unwrap();
        narrowest.push(f.grid.iter().map(|g| g.area).fold(f64::MAX, f64::min));
    }
    let segs = &plan.segments;
    (1..segs.len() - 1)
        .filter(|&k| segs[k].kind == "vowel" && segs[k - 1].kind != "vowel" && segs[k + 1].kind != "vowel")
        .map(|k| {
            let frame = |t: f64| ((t * FS) as usize / BLOCK).min(narrowest.len() - 1);
            narrowest[frame(segs[k].t0)..=frame(segs[k].t1)].iter().cloned().fold(0.0, f64::max)
        })
        .collect()
}

/// Bug: `rate` shortened the segments but not the gestures' ramps and 45 ms release hold, so at
/// rate 1.8 a /t/ or /k/ closure was still shut through the 55 ms vowel after it: "ɑtɑtɑ" became
/// one long closure (opening 0.00 cm², 2.34 at rate 1).
#[test]
fn vowels_between_stops_open_at_fast_rates() {
    for ipa in ["ˈɑtɑtɑtɑ", "ˈɑkɑkɑkɑ"] {
        let slow = inner_vowel_openings(ipa, 1.0);
        let fast = inner_vowel_openings(ipa, 1.8);
        assert_eq!(slow.len(), 2);
        for (s, f) in slow.iter().zip(&fast) {
            assert!(f / s > 0.5, "{ipa}: a vowel opens {f:.2} cm² at rate 1.8, {s:.2} at rate 1");
        }
    }
}
