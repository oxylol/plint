//! Mandarin's u (乌 wu, 虎 hu, the medial of 话 hua and 朵 duo) is not English's /u/. Beijing
//! speakers' u has F2 ~660 Hz (Zee & Lee 2001, doi 10.21437/Eurospeech.2001-169: 345 / 661 Hz
//! F1 / F2, men), English "who'd" ~1000 Hz (Hillenbrand et al. 1995). The front end used to
//! sing the English vowel (a native speaker still heard a foreigner in the singing, and this was
//! its largest measured distance from Beijing speakers' vowels). Mandarin gets its
//! own table entry, `u̠` (u + U+0320), that no English, Korean, Japanese or IPA-lyrics text reaches.
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::anatomy::human::{BODY_FEMALE, BODY_MALE, HumanBody};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::planner::vowel_table;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Segment, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};

const BACK_U: &str = "u\u{320}";

fn nucleus(py: &str) -> String {
    pinyin_syllable(py, py).unwrap().nucleus.ipa
}

#[test]
fn the_front_end_sings_the_back_u_in_wu_hu_and_every_u_medial() {
    for py in ["wu1", "hu3", "bu4", "gu1", "ku4", "fu2", "lu2"] {
        assert_eq!(nucleus(py), BACK_U, "{py}");
    }
    // the u medial of a rising nucleus (and of the triphthongs), also u̠
    for (py, want) in [("gua1", "a"), ("duo1", "ɔ"), ("guan1", "a"), ("dun1", "ə"), ("guang1", "ɑ"), ("weng1", "ə"), ("guai4", "aɪ"), ("gui1", "eɪ"), ("bo1", "ɔ")] {
        assert_eq!(nucleus(py), format!("{BACK_U}{want}"), "{py}");
    }
    // 块儿: the off-glide gives way to the r and the back-u on-glide stays
    assert_eq!(nucleus("kuair4"), format!("{BACK_U}a"));
}

#[test]
fn english_and_ipa_text_still_sing_the_english_vowels() {
    let ipas = |s: &str| tokenize_ipa(s).into_iter().map(|x| x.ipa).collect::<Vec<_>>();
    assert_eq!(ipas("hud"), ["h", "u", "d"]); // who'd
    assert_eq!(ipas("hʊd"), ["h", "ʊ", "d"]); // hood
    // the mark can't be typed into the entry: the tokenizer drops it and sings the English u
    assert_eq!(ipas("hu\u{320}d"), ["h", "u", "d"]);
    let t = vowel_table();
    for body in ["male", "female", "child"] {
        assert!(t[body]["u"].formants[1] > 990.0 * if body == "male" { 1.0 } else { 1.1 }, "{body} English u moved");
        assert!(t[body][BACK_U].formants[1] < 0.8 * t[body]["u"].formants[1], "{body}: u̠ F2 {} vs u {}", t[body][BACK_U].formants[1], t[body]["u"].formants[1]);
    }
}

const FS: f64 = 48000.0;
const BEAT_AT: f64 = 0.5;

/// F2 (Hz) of a sung one-beat note at 90 bpm, a median over LPC frames of 40 ms in the held vowel
fn sung_f2(syl: Syllable, body: HumanBody, name: &str, fs: f64) -> f64 {
    // 110 Hz on every body: at a woman's 220 Hz the harmonics are too sparse for LPC to place F1 and F2 well
    let (midi, f0) = (50.0, 110.0);
    let plan = plan_song(&[SungNote { midi, beats: 1.0, syl }], &SongOptions { body: name.into(), bpm: 90.0, t0: BEAT_AT, vibrato: 0.0, ..Default::default() });
    let mut v = HumanVoice::new(fs, body, oversample_for(fs));
    v.set_source(&SourcePatch::f0(f0));
    v.schedule(&plan.gestures);
    let mut out = vec![0f32; (fs * (BEAT_AT + 0.7)) as usize];
    for block in out.chunks_mut(512) {
        v.render(block);
    }
    assert!(out.iter().all(|x| x.is_finite()), "{name} at {fs}: NaN");
    let len = (0.04 * fs) as usize;
    let mut f2: Vec<f64> = (0..5)
        .map(|k| {
            let a = ((BEAT_AT + 0.2 + 0.03 * k as f64) * fs) as usize;
            let frame: Vec<f64> = out[a..a + len].iter().map(|x| *x as f64).collect();
            estimate_formants(&frame, fs, FormantOpts::default())[1].freq
        })
        .collect();
    f2.sort_by(f64::total_cmp);
    f2[2]
}

fn english_u() -> Syllable {
    Syllable { text: "who'd".into(), onset: vec![Segment::new("x")], nucleus: Segment::new("u"), ..Default::default() }
}

#[test]
fn sung_wu_has_an_f2_far_below_the_english_u() {
    // measured from the rendered voice (LPC): the Mandarin u's F2 was the English one's ~1000 Hz
    for (name, body) in [("male", BODY_MALE), ("female", BODY_FEMALE)] {
        let mandarin = sung_f2(pinyin_syllable("hu3", "hu3").unwrap(), body, name, FS);
        let english = sung_f2(english_u(), body, name, FS);
        assert!(mandarin + 200.0 < english, "{name}: Mandarin hu F2 {mandarin:.0} Hz vs English u {english:.0} Hz");
    }
}

#[test]
fn the_back_u_holds_at_a_low_sample_rate() {
    // a tight back constriction must not go unstable at a Bluetooth headset's 16 kHz (sung_f2 asserts finite)
    for (name, body) in [("male", BODY_MALE), ("female", BODY_FEMALE)] {
        let f2 = sung_f2(pinyin_syllable("wu1", "wu1").unwrap(), body, name, 16000.0);
        assert!(f2 > 300.0 && f2 < 1200.0, "{name}: F2 {f2:.0}");
    }
}
