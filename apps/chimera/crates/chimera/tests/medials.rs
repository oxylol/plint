//! A Mandarin medial glide (两 liang, 快 kuai, 话 hua) is a quick transition into the main vowel,
//! not a held vowel of its own: it is made with the initial consonant and is leaving by the beat, so
//! the main vowel's posture arrives within ~110 ms of the beat (it took ~200 ms when the glide sat
//! on the beat as a 120 ms plateau, and listening diagnoses of the renders, by language models,
//! heard 两 as li-ang). Measured on the voice's
//! articulators, as `chimera render notes --artic` records them.
use chimera::anatomy::human::{ArticKey, BODY_MALE};
use chimera::phonetics::mandarin::pinyin_syllable;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::Syllable;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

const FS: f64 = 48000.0;
const BPM: f64 = 90.0;
const BEAT_AT: f64 = 0.5;
const STEP: usize = 240; // 5 ms

/// The articulators of one sung one-beat syllable on the male body at 90 bpm, every 5 ms from the
/// start; `at(key, ms after the beat)`.
struct Track {
    frames: Vec<Vec<f64>>,
    vowels: Vec<(String, f64, f64)>,
}

impl Track {
    fn at(&self, k: ArticKey, ms: f64) -> f64 {
        let i = ((BEAT_AT + ms / 1000.0) * FS / STEP as f64).round() as usize;
        self.frames[i.min(self.frames.len() - 1)][k.index()]
    }
    /// first time (ms after the beat) the key is within 10% of its travel to its value at `hold` ms
    fn reaches_90(&self, k: ArticKey, hold: f64) -> f64 {
        let (s, h) = (self.at(k, 0.0), self.at(k, hold));
        (0..400).step_by(5).map(f64::from).find(|ms| (self.at(k, *ms) - h).abs() <= 0.1 * (s - h).abs()).unwrap_or(999.0)
    }
}

fn sing(syl: &Syllable) -> Track {
    let plan = plan_song(&[SungNote { midi: 50.0, beats: 1.0, syl: syl.clone() }], &SongOptions { body: "male".into(), bpm: BPM, t0: BEAT_AT, vibrato: 0.0, ..Default::default() });
    let mut v = HumanVoice::new(FS, BODY_MALE, 2);
    v.set_source(&SourcePatch::f0(110.0));
    v.schedule(&plan.gestures);
    let n = (FS * (BEAT_AT + 0.7)) as usize;
    let mut out = vec![0f32; STEP];
    let mut frames = vec![ArticKey::ALL.iter().map(|k| v.artic.x.get(*k)).collect()];
    for _ in 0..n / STEP {
        v.render(&mut out);
        frames.push(ArticKey::ALL.iter().map(|k| v.artic.x.get(*k)).collect());
    }
    Track { frames, vowels: plan.segments.iter().filter(|s| s.kind == "vowel").map(|s| (s.ipa.clone(), s.t0, s.t1)).collect() }
}

fn py(p: &str) -> Syllable {
    pinyin_syllable(p, p).unwrap_or_else(|| panic!("{p}"))
}

/// 两 快 朵 怪 有 话 家 月: a medial, a triphthong, a medial with no initial, ü
const MEDIALS: [&str; 8] = ["liang3", "kuai4", "duo3", "guai4", "you3", "hua4", "jia1", "yue4"];

#[test]
fn the_main_vowel_arrives_soon_after_the_beat() {
    // measured 75-90 ms (a syllable with no medial: 50 ms); with the glide as a 120 ms plateau on
    // the beat it was ~200 ms
    for p in MEDIALS {
        let t = sing(&py(p));
        let mut checked = 0;
        for k in [ArticKey::Jaw, ArticKey::TbY] {
            if (t.at(k, 0.0) - t.at(k, 200.0)).abs() < 0.3 {
                continue; // this articulator hardly moves into this vowel
            }
            checked += 1;
            let ms = t.reaches_90(k, 200.0);
            assert!(ms <= 110.0, "{p}: {k:?} reaches its vowel {ms} ms after the beat");
        }
        assert!(checked >= 1, "{p}: nothing moved");
    }
}

#[test]
fn nothing_waits_on_the_beat() {
    // no plateau: a quarter of the way to the vowel within 40 ms (measured 40-80%; the plateau gave ~0)
    for p in MEDIALS {
        let t = sing(&py(p));
        let k = if (t.at(ArticKey::Jaw, 0.0) - t.at(ArticKey::Jaw, 200.0)).abs() > 0.3 { ArticKey::Jaw } else { ArticKey::TbY };
        let (s, h) = (t.at(k, 0.0), t.at(k, 200.0));
        let done = (t.at(k, 40.0) - s) / (h - s);
        assert!(done >= 0.25, "{p}: {k:?} has done {:.0}% of its travel 40 ms after the beat", done * 100.0);
    }
}

#[test]
fn the_glide_is_reached_before_the_beat() {
    // the medial is made with the consonant: at its release (just before the beat) the lips are
    // rounded for u and ü, the tongue body fronted for i
    for p in ["kuai4", "duo3", "guai4", "hua4", "yue4"] {
        let t = sing(&py(p));
        assert!(t.at(ArticKey::LipProt, -20.0) >= 1.0, "{p}: lips not rounded before the beat ({})", t.at(ArticKey::LipProt, -20.0));
    }
    for p in ["liang3", "you3", "jia1"] {
        let t = sing(&py(p));
        // i's tongue body is high (TbY 7.5-7.7), the main vowels' lower (5.7-6.7)
        assert!(t.at(ArticKey::TbY, -20.0) >= 7.4, "{p}: tongue body not at i before the beat ({})", t.at(ArticKey::TbY, -20.0));
        assert!(t.at(ArticKey::TbY, -20.0) - t.at(ArticKey::TbY, 200.0) >= 0.7, "{p}: the vowel does not lower from i");
    }
}

#[test]
fn the_medial_is_still_there_against_the_same_syllable_without_it() {
    // the old failure: the medial lost (话 hua heard as 哈 ha). Each pair has the same initial and
    // main vowel; the medial's lip rounding or tongue fronting must show in the first 60 ms of the vowel
    // (measured: lip protrusion differs by 0.8 at +20 ms, tongue body by 0.5-1.1)
    for (with, without) in [("hua4", "ha4"), ("kuai4", "kai4"), ("guang1", "gang1"), ("zhuang1", "zhang1")] {
        let (x, y) = (sing(&py(with)), sing(&py(without)));
        let d = (x.at(ArticKey::LipProt, 20.0) - y.at(ArticKey::LipProt, 20.0)).abs();
        assert!(d >= 0.5, "{with} vs {without}: lips differ by {d:.2} at +20 ms");
    }
    for (with, without) in [("jia1", "ja1"), ("xia1", "xa1"), ("liang3", "lang3")] {
        let (x, y) = (sing(&py(with)), sing(&py(without)));
        let d = [0.0, 20.0, 40.0].iter().flat_map(|ms| [ArticKey::TbX, ArticKey::TbY].map(|k| (x.at(k, *ms) - y.at(k, *ms)).abs())).fold(0.0, f64::max);
        assert!(d >= 0.4, "{with} vs {without}: the tongue body differs by {d:.2} in the first 40 ms");
    }
}

#[test]
fn triphthongs_keep_their_off_glide() {
    for (p, off) in [("guai4", "ɪ"), ("kuai4", "ɪ"), ("you3", "ʊ̠")] {
        let t = sing(&py(p));
        let v: Vec<&str> = t.vowels.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(v.len(), 3, "{p}: glide, held vowel, off-glide: {v:?}");
        assert_eq!(v[2], off);
        // the off-glide is sung: the tongue and jaw leave the held vowel in the note's last part
        let moved = [ArticKey::Jaw, ArticKey::TbY, ArticKey::TbX, ArticKey::LipProt].iter().map(|k| (t.at(*k, 560.0) - t.at(*k, 300.0)).abs()).fold(0.0, f64::max);
        assert!(moved >= 0.15, "{p}: moved {moved:.2} over the off-glide");
    }
}

#[test]
fn a_live_note_sings_the_glide_short_too() {
    use chimera::phonetics::sing::{NoteOnOptions, plan_note_on};
    let g = plan_note_on(&py("hua4"), 50.0, &NoteOnOptions { body: "male".into(), vibrato: 0.0, tag: 10.0, legato_from: None });
    let u = g.iter().find(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "u̠")).expect("the glide u");
    assert!(u.t1 - u.t0 <= 0.06, "the live glide is {} s", u.t1 - u.t0);
}
