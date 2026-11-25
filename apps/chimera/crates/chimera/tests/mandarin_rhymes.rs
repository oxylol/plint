//! A Mandarin rhyme is one sound: its ending (the off-glide of ai ei ao ou, a nasal coda) is where
//! the vowel ends up, not a second sound after it. A native listener heard every compound and nasal
//! rhyme as two: 奥 ao sung for 1.33 s was [a] still for 0.76 s, a jump, then [ʊ] still for 0.35 s;
//! 昂 ang was an oral [ɑ], the tongue body jumping in 40 ms, then 0.4 s of murmur ("an + ng"); and a
//! phrase's last -n let go while the breath was still on ([an-ə]). The front end now marks its
//! rhymes closing (`Syllable::closing`), and the song planner sings the ending as one movement over
//! the end of the note.
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::anatomy::human::BODY_MALE;
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Plan, Segment, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};

const FS: f64 = 48000.0;
const BEAT_AT: f64 = 0.5;

/// one note of `secs` seconds, its beat at [`BEAT_AT`]; it ends (the breath's reserve) 50 ms early
fn plan(syl: Syllable, secs: f64) -> Plan {
    plan_song(&[SungNote { midi: 45.0, beats: secs, syl }], &SongOptions { body: "male".into(), bpm: 60.0, t0: BEAT_AT, vibrato: 0.0, ..Default::default() })
}

fn mandarin(py: &str) -> Syllable {
    pinyin_syllable(py, py).unwrap()
}

fn render(p: &Plan, secs: f64, fs: f64) -> Vec<f32> {
    let mut v = HumanVoice::new(fs, BODY_MALE, oversample_for(fs));
    v.set_source(&SourcePatch::f0(110.0));
    v.schedule(&p.gestures);
    let mut out = vec![0f32; (fs * (BEAT_AT + secs + 0.4)) as usize];
    for block in out.chunks_mut(512) {
        v.render(block);
    }
    assert!(out.iter().all(|x| x.is_finite()), "NaN at {fs}");
    out
}

/// level (dB) of the 40 ms centred on t
fn level(out: &[f32], t: f64) -> f64 {
    let a = ((t - 0.02) * FS) as usize;
    let n = (0.04 * FS) as usize;
    let ms = out[a..a + n].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64;
    10.0 * ms.max(1e-12).log10()
}

/// F1 and F2 (Hz) of the 40 ms centred on t
fn formants(out: &[f32], t: f64) -> (f64, f64) {
    let a = ((t - 0.02) * FS) as usize;
    let frame: Vec<f64> = out[a..a + (0.04 * FS) as usize].iter().map(|x| *x as f64).collect();
    let f = estimate_formants(&frame, FS, FormantOpts::default());
    (f[0].freq, f[1].freq)
}

#[test]
fn the_front_end_marks_its_rhymes_closing_and_its_codas_short() {
    for py in ["ao1", "ang1", "an1", "ma1"] {
        assert_eq!(mandarin(py).closing, Some(0.2), "{py}");
    }
    assert_eq!(mandarin("ang1").coda[0].dur, Some(0.08));
    assert_eq!(mandarin("an1").coda[0].dur, Some(0.07));
}

#[test]
fn a_long_ao_holds_its_a_and_glides_only_at_the_end() {
    let secs = 1.33;
    let p = plan(mandarin("ao1"), secs);
    // the off-glide is a short last part, coming in over its own length: no second held vowel
    let glide = p.segments.iter().find(|s| s.ipa == "ʊ\u{320}").expect("the off-glide");
    assert!(glide.t1 - glide.t0 <= 0.101, "the off-glide is held {:.3} s", glide.t1 - glide.t0);
    let g = p.gestures.iter().find(|g| g.label_str() == "ʊ\u{320}").unwrap();
    assert!(g.ramp_in >= 0.099, "the off-glide comes in over {:.3} s", g.ramp_in);
    // heard: 0.3 s before the note ends it is still [a] (it was [ʊ], F2 ~780), and it has moved by the end
    let out = render(&p, secs, FS);
    let end = BEAT_AT + secs - 0.05;
    let (_, still_a) = formants(&out, end - 0.3);
    let (_, closed) = formants(&out, end - 0.03);
    assert!(still_a > 1250.0, "0.3 s before the end F2 is {still_a:.0}");
    assert!(closed < 1150.0, "at the end F2 is {closed:.0}");
}

#[test]
fn a_short_note_keeps_some_of_its_main_vowel() {
    // 0.3 s: the movement takes at most 60 % of the vowel
    let p = plan(mandarin("ai1"), 0.3);
    let v: Vec<_> = p.segments.iter().filter(|s| s.kind == "vowel").collect();
    assert_eq!(v.len(), 2);
    let (held, glide) = (v[0].t1 - v[0].t0, v[1].t1 - v[1].t0);
    assert!((glide - 0.3 * (held + glide)).abs() < 1e-9, "held {held:.3}, glide {glide:.3}");
}

#[test]
fn a_long_ang_is_a_vowel_until_its_last_fifth_of_a_second() {
    let secs = 1.33;
    let p = plan(mandarin("ang1"), secs);
    let out = render(&p, secs, FS);
    let end = BEAT_AT + secs - 0.05;
    let vowel = level(&out, BEAT_AT + 0.4);
    // 0.3 s before the end it was already the murmur (10 dB down, F1 ~440)
    let late = level(&out, end - 0.3);
    let (f1, _) = formants(&out, end - 0.3);
    assert!(late > vowel - 3.0, "0.3 s before the end the level is {:.1} dB under the vowel's", vowel - late);
    assert!(f1 > 600.0, "0.3 s before the end F1 is {f1:.0}");
    // and it still closes into the -ng
    let murmur = level(&out, end - 0.04);
    let (f1, _) = formants(&out, end - 0.04);
    assert!(murmur < vowel - 4.0, "the murmur is {:.1} dB under the vowel", vowel - murmur);
    assert!(f1 < 560.0, "the murmur's F1 is {f1:.0}");
    // the velum comes down over the nasalised part instead of dropping at its start, and the
    // tongue body is on its way before the coda closes
    let velum = p.gestures.iter().find(|g| g.label_str() == "nasal vowel").unwrap();
    assert!(velum.ramp_in >= 0.2, "the velum lowers over {:.3} s", velum.ramp_in);
    assert!(p.gestures.iter().any(|g| g.label_str() == "coda approach" && g.ramp_in >= 0.1));
}

/// the deepest fall-and-recover (dB) of the level between a and b, in 20 ms steps
fn dip(out: &[f32], a: f64, b: f64) -> f64 {
    let n = ((b - a) / 0.02) as usize;
    let l: Vec<f64> = (0..n).map(|i| level(out, a + 0.02 * i as f64)).collect();
    (1..n - 1).map(|i| l[..i].iter().cloned().fold(f64::MIN, f64::max).min(l[i + 1..].iter().cloned().fold(f64::MIN, f64::max)) - l[i]).fold(0.0, f64::max)
}

#[test]
fn a_close_vowel_does_not_stop_before_its_ng() {
    // 东 dong: [ʊ] is close under the velum already. With the velum still up as the coda closed, the
    // voice dipped 9 dB for a moment, a stop between the vowel and the murmur
    let secs = 0.4;
    let out = render(&plan(mandarin("dong1"), secs), secs, FS);
    let d = dip(&out, BEAT_AT + 0.1, BEAT_AT + secs - 0.06);
    assert!(d < 3.0, "the level falls and recovers by {d:.1} dB inside the rhyme");
}

#[test]
fn a_phrase_final_n_is_not_released_into_a_vowel() {
    // 安 an: the murmur fades; it used to rise 2.7 dB again as the tongue let go ([an-ə])
    let secs = 0.4;
    let p = plan(mandarin("an1"), secs);
    let out = render(&p, secs, FS);
    let end = p.segments.last().unwrap().t1;
    let murmur = level(&out, end - 0.05);
    for k in 0..6 {
        let t = end - 0.02 + 0.02 * k as f64;
        assert!(level(&out, t) < murmur + 1.0, "{:.0} ms after the coda's end the level is {:.1} dB over the murmur", (t - end) * 1000.0, level(&out, t) - murmur);
    }
}

#[test]
fn a_syllable_that_is_not_closing_sings_as_before() {
    // English "my": the off-glide holds 20 % of the vowel
    let syl = Syllable { text: "my".into(), onset: vec![Segment::new("m")], nucleus: Segment::new("aɪ"), ..Default::default() };
    let p = plan(syl, 1.0);
    let v: Vec<_> = p.segments.iter().filter(|s| s.kind == "vowel").collect();
    let (held, glide) = (v[0].t1 - v[0].t0, v[1].t1 - v[1].t0);
    assert!((glide - 0.2 * (held + glide)).abs() < 1e-9, "held {held:.3}, glide {glide:.3}");
    let g = p.gestures.iter().find(|g| g.label_str() == "ɪ").unwrap();
    assert!((g.ramp_in - 0.04).abs() < 1e-9, "ramp {}", g.ramp_in);
}

#[test]
fn closing_rhymes_hold_at_a_low_sample_rate() {
    for py in ["ao1", "ang1", "an1", "ying1"] {
        render(&plan(mandarin(py), 0.4), 0.4, 16000.0);
    }
}
