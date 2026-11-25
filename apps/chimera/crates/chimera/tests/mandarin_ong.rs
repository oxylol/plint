//! Mandarin's ong (东 dong, 中 zhong, 兄 xiong) is [ʊŋ] with F2 ~730 Hz (Zee & Lee 2001,
//! doi 10.21437/Eurospeech.2001-169, "u before ŋ": men 441 / 730 Hz F1 / F2), where English /ʊ/
//! "hood" has ~1120. The front end sang the English vowel. Mandarin gets `ʊ̠` (ʊ + U+0320), also the
//! off-glide of ao ou iao iou. The back u (乌 wu) has its own tests in `mandarin_u.rs`.
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::anatomy::human::{BODY_FEMALE, BODY_MALE, HumanBody};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::planner::{expand, vowel_table};
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Segment, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch, oversample_for};

const ONG: &str = "ʊ\u{320}";

fn nucleus(py: &str) -> String {
    pinyin_syllable(py, py).unwrap().nucleus.ipa
}

#[test]
fn the_front_end_sings_ong_and_the_off_glides_with_the_back_ʊ() {
    for py in ["dong1", "zhong1", "gong1"] {
        assert_eq!(nucleus(py), ONG, "{py}");
    }
    assert_eq!(nucleus("xiong2"), format!("i{ONG}"));
    assert_eq!(nucleus("hao3"), format!("aʊ\u{320}"));
    assert_eq!(nucleus("dou1"), format!("əʊ\u{320}"));
    assert_eq!(nucleus("liao3"), format!("iaʊ\u{320}"));
    assert_eq!(nucleus("you3"), format!("iəʊ\u{320}"));
    // the compounds expand to the new vowel as their last part
    for k in ["iʊ\u{320}", "aʊ\u{320}", "əʊ\u{320}", "iaʊ\u{320}", "iəʊ\u{320}"] {
        assert_eq!(expand(k).unwrap().last().copied(), Some(ONG), "{k}");
    }
    // English's falling diphthongs, and the coda of 东, are as they were
    assert_eq!(expand("aʊ").unwrap(), ["a", "ʊ"]);
    assert_eq!(pinyin_syllable("dong1", "dong1").unwrap().coda[0].ipa, "ŋ");
}

#[test]
fn english_hood_is_still_the_english_vowel() {
    let ipas = |s: &str| tokenize_ipa(s).into_iter().map(|x| x.ipa).collect::<Vec<_>>();
    assert_eq!(ipas("hʊd"), ["h", "ʊ", "d"]);
    assert_eq!(ipas("haʊs"), ["h", "aʊ", "s"]);
    let t = vowel_table();
    for body in ["male", "female", "child"] {
        let (eng, man) = (t[body]["ʊ"].formants[1], t[body][ONG].formants[1]);
        assert!(eng > 1100.0, "{body}: English ʊ F2 {eng} moved");
        assert!(man < 0.9 * eng, "{body}: ʊ̠ F2 {man} vs ʊ {eng}");
    }
}

const FS: f64 = 48000.0;
const BEAT_AT: f64 = 0.5;

/// median F2 (Hz) of the voice's LPC frames of 40 ms, from 0.1 s to 0.2 s after the beat of a one-beat note
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
            let a = ((BEAT_AT + 0.1 + 0.025 * k as f64) * fs) as usize;
            let frame: Vec<f64> = out[a..a + len].iter().map(|x| *x as f64).collect();
            estimate_formants(&frame, fs, FormantOpts::default())[1].freq
        })
        .collect();
    f2.sort_by(f64::total_cmp);
    f2[2]
}

fn english_ong() -> Syllable {
    let mut s = pinyin_syllable("dong1", "dong1").unwrap();
    s.nucleus = Segment::new("ʊ");
    s
}

#[test]
fn sung_dong_has_an_f2_below_the_english_ʊ() {
    for (name, body) in [("male", BODY_MALE), ("female", BODY_FEMALE)] {
        let mandarin = sung_f2(pinyin_syllable("dong1", "dong1").unwrap(), body, name, FS);
        let english = sung_f2(english_ong(), body, name, FS);
        assert!(mandarin + 150.0 < english, "{name}: Mandarin dong F2 {mandarin:.0} Hz vs English ʊŋ {english:.0} Hz");
    }
}

#[test]
fn ong_holds_at_a_low_sample_rate() {
    for (name, body) in [("male", BODY_MALE), ("female", BODY_FEMALE)] {
        let f2 = sung_f2(pinyin_syllable("dong1", "dong1").unwrap(), body, name, 16000.0);
        assert!(f2 > 300.0 && f2 < 1400.0, "{name}: F2 {f2:.0}");
    }
}
