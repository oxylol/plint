//! A sung vowel's voice comes in on the beat. The song planner puts the vowel's segment on the
//! beat, and the voice starts after it by whatever follows the onset's last consonant: a stop's
//! aspiration (90 ms in Mandarin), a burst, a fricative's release. Against M4Singer's 20 singers
//! (Zhang et al., NeurIPS 2022 Datasets and Benchmarks) the voice was 35-45 ms late after unaspirated stops and fricatives and 110 ms late
//! after aspirated ones, a large part of a short note's vowel. A front end that says how long its
//! voice lags (`Syllable::voice_lag`, Mandarin's does) has the vowel start that long before the
//! beat; one that doesn't is sung as before.
use chimera::anatomy::human::BODY_MALE;
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Plan, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};

const FS: f64 = 48000.0;
const BEAT_AT: f64 = 0.5;
const SECS: f64 = 0.6;

fn plan(syl: Syllable) -> Plan {
    plan_song(&[SungNote { midi: 45.0, beats: SECS, syl }], &SongOptions { body: "male".into(), bpm: 60.0, t0: BEAT_AT, vibrato: 0.0, ..Default::default() })
}

fn mandarin(py: &str) -> Syllable {
    pinyin_syllable(py, py).unwrap()
}

/// where the syllable's held vowel starts, in seconds after the beat
fn vowel_start(p: &Plan) -> f64 {
    p.segments.iter().filter(|s| s.kind == "vowel").map(|s| (s.t0, s.t1 - s.t0)).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0 - BEAT_AT
}

/// level (dB) of the 20 ms centred `ms` after the beat
fn level(p: &Plan, ms: f64) -> f64 {
    let mut v = HumanVoice::new(FS, BODY_MALE, oversample_for(FS));
    v.set_source(&SourcePatch::f0(110.0));
    v.schedule(&p.gestures);
    let mut out = vec![0f32; (FS * (BEAT_AT + SECS + 0.2)) as usize];
    for block in out.chunks_mut(512) {
        v.render(block);
    }
    let a = ((BEAT_AT + ms / 1000.0 - 0.01) * FS) as usize;
    let n = (0.02 * FS) as usize;
    10.0 * (out[a..a + n].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64).max(1e-12).log10()
}

#[test]
fn the_front_end_says_how_long_the_voice_lags_each_initial() {
    assert_eq!(mandarin("ta1").voice_lag, Some(0.09)); // the aspiration
    assert_eq!(mandarin("da1").voice_lag, Some(0.015)); // a burst
    assert_eq!(mandarin("sha1").voice_lag, Some(0.027)); // a fricative's release
    assert_eq!(mandarin("ma1").voice_lag, None); // voiced through
    assert_eq!(mandarin("a1").voice_lag, None);
}

#[test]
fn the_vowel_starts_before_the_beat_by_the_voice_lag() {
    assert!((vowel_start(&plan(mandarin("ta1"))) + 0.09).abs() < 1e-9);
    assert!((vowel_start(&plan(mandarin("da1"))) + 0.015).abs() < 1e-9);
    assert!(vowel_start(&plan(mandarin("ma1"))).abs() < 1e-9);
    // before a medial glide the glide takes the lag: 天 tian's is breathed for 90 ms, then voiced
    // for its own 50 ms, and ends on the beat, where the vowel starts
    let p = plan(mandarin("tian1"));
    assert!(vowel_start(&p).abs() < 1e-9);
    let glide = p.segments.iter().find(|s| s.kind == "vowel").unwrap();
    assert!((glide.t0 - (BEAT_AT - 0.14)).abs() < 1e-9 && (glide.t1 - BEAT_AT).abs() < 1e-9, "the glide is {:.3} to {:.3}", glide.t0, glide.t1);
}

#[test]
fn a_syllable_that_says_no_lag_is_sung_as_before() {
    // any other front end's syllable: the vowel's segment on the beat, the aspiration after it
    let unsaid = Syllable { voice_lag: None, ..mandarin("ta1") };
    assert!(vowel_start(&plan(unsaid)).abs() < 1e-9);
}

#[test]
fn the_voice_is_there_on_the_beat_after_an_aspirated_stop() {
    // 他 ta: 30 ms after the beat the voice is within 6 dB of its level 200 ms in; with the vowel's
    // segment on the beat it was still the aspiration, 13 dB under
    let said = plan(mandarin("ta1"));
    let unsaid = plan(Syllable { voice_lag: None, ..mandarin("ta1") });
    let (now, then) = (level(&said, 200.0) - level(&said, 30.0), level(&unsaid, 200.0) - level(&unsaid, 30.0));
    assert!(now < 6.0, "30 ms after the beat the voice is {now:.1} dB under its level");
    assert!(then > 10.0, "without the lag it was {then:.1} dB under");
}

/// a song of one note whose beat is `t0` s into the plan: (when its first movement begins, when
/// its held vowel starts, how long its first segment is)
fn opening(py: &str, t0: f64) -> (f64, f64, f64) {
    let p = plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: mandarin(py) }], &SongOptions { body: "female".into(), bpm: 90.0, t0, vibrato: 0.0, ..Default::default() });
    let first = p.gestures.iter().map(|g| g.t0 - g.ramp_in).fold(f64::INFINITY, f64::min);
    let vowel = p.segments.iter().filter(|s| s.kind == "vowel").map(|s| (s.t0, s.t1 - s.t0)).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    (first, vowel, p.segments[0].t1 - p.segments[0].t0)
}

#[test]
fn a_song_does_not_start_before_the_plans_time_zero() {
    // an aspirated affricate and a medial glide from a beat at 0.3 s (`chimera render notes`):
    // 252 ms of onset and the planner's 62 ms of lead. The plan began at -13 ms, and a render,
    // which starts at 0, cut the start. Now the consonants shorten to fit (6 %) and the vowel is
    // still on the beat
    for py in ["chuang1", "qiang3", "chun1", "qiong2"] {
        let (first, vowel, initial) = opening(py, 0.3);
        assert!(first >= 0.0 && first < 0.002, "{py} begins at {first:.4} s");
        assert!((vowel - 0.3).abs() < 1e-9, "{py}'s vowel starts at {vowel:.4} s");
        // with room (the app's beat at 0.35 s, or any later one) nothing is shortened
        let (app, _, at_length) = opening(py, 0.35);
        assert!((app - 0.037).abs() < 0.001, "{py} begins at {app:.4} s from the app's beat");
        assert!((opening(py, 1.0).2 - at_length).abs() < 1e-9 && initial < 0.95 * at_length);
    }
    // a syllable that says no lag starts where it did
    let unsaid = |t0: f64| plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: Syllable { voice_lag: None, ..mandarin("chuang1") } }], &SongOptions { body: "female".into(), bpm: 90.0, t0, vibrato: 0.0, ..Default::default() });
    let (a, b) = (unsaid(0.15), unsaid(1.0));
    assert!((a.segments[0].t1 - a.segments[0].t0 - (b.segments[0].t1 - b.segments[0].t0)).abs() < 1e-9);
}

/// two notes `secs` apart: 他 ta, then 爽 shuang (a fricative, a medial glide, the voice's lag: 173 ms of onset)
fn pair(secs: f64, share: Option<f64>) -> Plan {
    let next = Syllable { onset_share: share, ..mandarin("shuang3") };
    plan_song(
        &[SungNote { midi: 45.0, beats: secs, syl: mandarin("ta1") }, SungNote { midi: 45.0, beats: 0.6, syl: next }],
        &SongOptions { body: "male".into(), bpm: 60.0, t0: BEAT_AT, vibrato: 0.0, ..Default::default() },
    )
}

/// how long the first syllable's vowel lasts after its beat (s)
fn voiced(p: &Plan) -> f64 {
    p.segments.iter().find(|s| s.kind == "vowel").unwrap().t1 - BEAT_AT
}

#[test]
fn a_fast_note_shortens_the_consonants_after_it_and_keeps_its_vowel() {
    assert_eq!(mandarin("shuang3").onset_share, Some(0.55));
    // 150 ms to the next vowel: its consonants get 55 % of that, and the vowel the rest (it had
    // next to nothing: the onset alone is longer than the note)
    let (now, then) = (voiced(&pair(0.15, Some(0.55))), voiced(&pair(0.15, None)));
    assert!((now - 0.0675).abs() < 1e-6, "the vowel is voiced for {now:.4} s");
    assert!(then < 0.03, "at their own lengths the consonants left it {then:.3} s");
    // a slow note leaves them as they are
    assert!((voiced(&pair(1.0, Some(0.55))) - voiced(&pair(1.0, None))).abs() < 1e-9);
}
