//! Mandarin er (耳 ěr, 二 èr, 儿 ér) is a central vowel moving into the r-colouring ([əɚ]; 二 more
//! open, [aɚ]), not English's /ɝ/ held still: F3, the formant of the r-colour, falls over the
//! syllable instead of sitting low from the first frame. Formants are the airway's resonances in
//! the posture the sung voice has at each moment (as `chimera calibrate` measures them).
use chimera::airway::frequency::{find_peaks, transfer_function};
use chimera::anatomy::human::{BODY_MALE, HumanArtic};
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Segment, Syllable};
use chimera::voice::human_airway::HumanAirway;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const BEAT_AT: f64 = 0.5;
const STEP: usize = 240; // 5 ms

/// the posture of the male voice every 5 ms singing one one-beat note at 90 bpm
fn postures(syl: &Syllable) -> Vec<HumanArtic> {
    let plan = plan_song(&[SungNote { midi: 50.0, beats: 1.0, syl: syl.clone() }], &SongOptions { body: "male".into(), bpm: 90.0, t0: BEAT_AT, vibrato: 0.0, ..Default::default() });
    let mut v = HumanVoice::new(FS, BODY_MALE, 2);
    v.set_source(&SourcePatch::f0(110.0));
    v.schedule(&plan.gestures);
    let mut out = vec![0f32; STEP];
    let mut frames = vec![v.artic.x];
    for _ in 0..(FS * (BEAT_AT + 0.7)) as usize / STEP {
        v.render(&mut out);
        frames.push(v.artic.x);
    }
    frames
}

/// F1 F2 F3 (Hz) at `ms` after the beat
struct Formants(Vec<[f64; 3]>);

impl Formants {
    fn of(frames: &[HumanArtic]) -> Self {
        let mut air = HumanAirway::new(96000.0, BODY_MALE);
        let freqs: Vec<f64> = (0..700).map(|i| 120.0 * (4600.0f64 / 120.0).powf(i as f64 / 699.0)).collect();
        Formants(
            frames
                .iter()
                .map(|a| {
                    air.apply(a, 0.0);
                    let el = air.net.linear_elements();
                    let h = transfer_function(&el, air.pharynx.first_node(), &freqs, &[air.glottis_port]);
                    let pk = find_peaks(&freqs, &h.mag, 150.0);
                    let f = |i: usize| pk.get(i).map_or(f64::NAN, |p| p.freq);
                    [f(0), f(1), f(2)]
                })
                .collect(),
        )
    }
    fn at(&self, ms: f64) -> [f64; 3] {
        let i = ((BEAT_AT + ms / 1000.0) * FS / STEP as f64).round() as usize;
        self.0[i.min(self.0.len() - 1)]
    }
}

fn held(ipa: &str) -> Syllable {
    Syllable { text: "x".into(), onset: vec![Segment::new("ʔ")], nucleus: Segment::new(ipa), word_end: true, ..Default::default() }
}

/// 耳 and 儿 (any tone but the fourth) and 二
fn er(tone: u8) -> Formants {
    Formants::of(&postures(&pinyin_syllable(&format!("er{tone}"), "耳").unwrap()))
}

#[test]
fn the_r_colouring_arrives_late_not_from_the_first_frame() {
    // English's /ɝ/ held: F3 is already low at the beat (measured 1940 Hz there, 2280 at -80 ms)
    let old = Formants::of(&postures(&held("ɝ")));
    assert!(old.at(0.0)[2] < 2000.0);
    for tone in [2, 3, 4] {
        let f = er(tone);
        // still central (F3 ~2300-2550: ə, a) at the beat and just before it, then falling...
        assert!(f.at(-20.0)[2] > 2250.0 && f.at(0.0)[2] > 2250.0, "er{tone}: F3 {:.0} / {:.0} at -20 / 0 ms", f.at(-20.0)[2], f.at(0.0)[2]);
        let fall: Vec<f64> = [0.0, 40.0, 80.0].iter().map(|m| f.at(*m)[2]).collect();
        assert!(fall.windows(2).all(|w| w[1] < w[0] - 50.0), "er{tone}: F3 does not fall steadily {fall:?}");
        // ...into the r-colour, where it settles near ɝ's (1850 Hz), 450-650 Hz below its start
        assert!(f.at(150.0)[2] < 1900.0 && f.at(300.0)[2] < 1900.0, "er{tone}: F3 {:.0} / {:.0} at 150 / 300 ms", f.at(150.0)[2], f.at(300.0)[2]);
        assert!(f.at(0.0)[2] - f.at(300.0)[2] > 350.0);
    }
}

#[test]
fn er4_starts_more_open_than_the_other_tones() {
    // 二 [aɚ] against 耳 [əɚ]: F1 higher at the start (measured ~700 against ~495 Hz at -20 ms)
    let (a, e) = (er(4), er(3));
    assert!(a.at(-20.0)[0] > e.at(-20.0)[0] + 120.0, "F1 {:.0} against {:.0}", a.at(-20.0)[0], e.at(-20.0)[0]);
    // ...and the same r-coloured vowel once it has arrived
    assert!((a.at(300.0)[2] - e.at(300.0)[2]).abs() < 30.0);
}

#[test]
fn er_is_one_rising_nucleus_and_erhua_keeps_its_coda() {
    let s = |p: &str| pinyin_syllable(p, p).unwrap();
    assert_eq!(s("er3").nucleus.ipa, "əɝ");
    assert_eq!(s("er2").nucleus.ipa, "əɝ");
    assert_eq!(s("er4").nucleus.ipa, "aɝ");
    // 点儿: the vowel before the r is unchanged, and the r is its coda, not another nucleus
    let d = s("dianr3");
    assert_eq!((d.nucleus.ipa.as_str(), d.coda[0].ipa.as_str()), ("iɛ", "ɻ"));
}
