//! Rising diphthongs (Mandarin's medials as one nucleus): sung, the glide is the last of the onset
//! (done by the beat, short) and the second vowel holds from the beat; a held melisma continues the vowel, not the glide.
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Segment, Syllable};

fn vowels(nucleus: &str, beats: f64) -> Vec<(String, f64)> {
    vowel_spans(nucleus, beats).into_iter().map(|(i, a, b)| (i, b - a)).collect()
}

/// the vowel parts' (ipa, start, end); the note's beat is at 0.5 s
fn vowel_spans(nucleus: &str, beats: f64) -> Vec<(String, f64, f64)> {
    let syl = Syllable { text: "x".into(), onset: vec![Segment::new("x")], nucleus: Segment::new(nucleus), word_end: true, ..Default::default() };
    let p = plan_song(&[SungNote { midi: 57.0, beats, syl }], &SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, ..Default::default() });
    p.segments.iter().filter(|s| s.kind == "vowel").map(|s| (s.ipa.clone(), s.t0, s.t1)).collect()
}

#[test]
fn a_rising_diphthong_sings_a_short_glide_then_holds_its_vowel() {
    let v = vowels("ua", 2.0);
    assert_eq!(v.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), ["u", "a"]);
    // a quick transition (50 ms), not a held vowel of its own (it was up to 120 ms, on the beat)
    assert!(v[0].1 <= 0.06, "the glide is at most 60 ms, got {}", v[0].1);
    // and it ends on the beat (0.5 s): the main vowel is what the beat lands on
    let s = vowel_spans("ua", 2.0);
    assert!((s[0].2 - 0.5).abs() < 1e-9 && (s[1].1 - 0.5).abs() < 1e-9, "{s:?}");
    assert!(v[1].1 > 1.5, "the vowel holds the note");
    // a falling diphthong is the other way round: the first vowel holds
    let f = vowels("aɪ", 2.0);
    assert!(f[0].1 > f[1].1);
}
