//! The narrowest point's frication source is registered wherever the point sits, including the
//! oral tube's first section just past the velum (issue #124). The source, and the teeth source
//! with it, used to be dropped when the point was on that section or, far more often, when its
//! upstream reach landed there: the point a section or two forward, as in back vowels and velars.
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_airway::FRICATION_FADE_AREA;
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;
const BLOCK: usize = 64;

#[derive(Default, Debug)]
struct Count {
    /// blocks narrow enough to make frication (below the fade's upper end, body-scaled)
    narrow: usize,
    /// of those, blocks with no source beyond the glottis's
    dropped: usize,
    /// narrow blocks whose point is on the oral tube's first section
    on_junction: usize,
    /// narrow blocks whose point is elsewhere but whose upstream or downstream reach lands there
    reach_on_junction: usize,
}

/// `update_noise`'s narrowest point and reach, recomputed from the airway's public state.
fn count(ipa: &str) -> Count {
    let plan = plan_utterance(&tokenize_ipa(ipa), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(120.0), ..Default::default() });
    let mut v = HumanVoice::with_defaults(FS);
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + 0.1)).round() as usize;
    let mut out = vec![0.0f32; n];
    let mut c = Count::default();
    for i in (0..n).step_by(BLOCK) {
        v.render(&mut out[i..(i + BLOCK).min(n)]);
        let a = &v.airway;
        let (n_ph, n_tot) = (a.n_ph as isize, (a.n_ph + a.n_or) as isize);
        let (mut min_a, mut min_i) = (f64::INFINITY, -1isize);
        for j in 0..n_tot {
            let area = if j < n_ph { a.ph_min[j as usize] } else { a.or_min[(j - n_ph) as usize] };
            if area > 0.0 && area < min_a {
                (min_a, min_i) = (area, j);
            }
        }
        let scale = a.body.scale;
        if min_i < 0 || min_a >= FRICATION_FADE_AREA * scale * scale {
            continue;
        }
        c.narrow += 1;
        if a.net.noise_sources().len() < 2 {
            c.dropped += 1;
        }
        let dx = if min_i < n_ph { a.ph_len / a.n_ph as f64 } else { a.or_len / a.n_or as f64 };
        let off = ((0.9 * scale) / dx).round().max(1.0) as isize;
        if min_i.min(n_tot - 2) == n_ph {
            c.on_junction += 1;
        } else if (n_tot - 2).min(min_i + off) == n_ph || 0.max(min_i - off) == n_ph {
            c.reach_on_junction += 1;
        }
    }
    c
}

#[test]
fn a_constriction_just_past_the_velum_keeps_its_frication_source() {
    // a back vowel, one between voiceless consonants, and a velar fricative (before the fix every
    // narrow block of ɯː and axa had no source)
    for word in ["ɯː", "ekɯta", "axa"] {
        let c = count(word);
        assert!(c.on_junction + c.reach_on_junction > 0, "{word}: never touches the section, the test measures nothing: {c:?}");
        assert_eq!(c.dropped, 0, "{word}: narrow blocks with no frication source: {c:?}");
    }
}
