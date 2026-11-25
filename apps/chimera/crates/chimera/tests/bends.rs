//! Pitch bends: a syllable's front end shapes the pitch inside its note (start or end, capped in
//! length); none leaves the note as written. And the vowel before a nasal coda is nasalised only
//! when asked.
use chimera::anatomy::human::ArticKey;
use chimera::control::score::{ControlKey, Gesture};
use chimera::phonetics::sing::{SongOptions, SungNote, midi_to_hz, plan_song};
use chimera::phonetics::{Bend, Edge, Segment, Syllable};

fn syl(onset: &str, v: &str, coda: &[&str], pitch: Vec<Bend>) -> Syllable {
    Syllable { text: v.into(), onset: vec![Segment::new(onset)], nucleus: Segment::new(v), coda: coda.iter().map(|c| Segment::new(c)).collect(), word_end: true, pitch, ..Default::default() }
}
fn opts() -> SongOptions {
    SongOptions { body: "male".into(), bpm: 60.0, t0: 0.5, ..Default::default() }
}
fn labelled<'a>(gs: &'a [Gesture], l: &str) -> Vec<&'a Gesture> {
    gs.iter().filter(|g| g.label.as_ref().is_some_and(|x| x.as_str() == l)).collect()
}

#[test]
fn bends_colour_the_start_or_end_of_the_note_and_none_leave_it() {
    let hz = midi_to_hz(57.0);
    let one = |pitch| plan_song(&[SungNote { midi: 57.0, beats: 2.0, syl: syl("l", "a", &[], pitch) }], &opts());
    assert!(labelled(&one(vec![]).gestures, "bend").is_empty());
    let up = Bend { at: Edge::Start, len: 0.35, max: 0.15, st: -2.0 };
    let down = Bend { at: Edge::End, len: 0.35, max: 0.15, st: -3.0 };
    let p = one(vec![up, down]);
    let b = labelled(&p.gestures, "bend");
    assert_eq!(b.len(), 2);
    // the start bend: from just before the beat, at most 0.15 s into a 2 s note, 2 semitones down
    assert!((b[0].targets.get(ControlKey::F0).unwrap() / hz - 2f64.powf(-2.0 / 12.0)).abs() < 1e-9);
    assert!(b[0].t0 < 0.5 && (b[0].t1 - 0.65).abs() < 1e-9);
    // the end bend: the last 0.15 s of the vowel
    assert!(b[1].targets.get(ControlKey::F0).unwrap() < hz && (b[1].t1 - b[1].t0 - 0.17).abs() < 1e-9);
}

#[test]
fn a_bend_can_sit_in_the_middle_of_the_vowel() {
    // a dip inside the note (Mandarin's tone 3 as 214, in speech): centred on the vowel's middle
    let p = plan_song(&[SungNote { midi: 57.0, beats: 2.0, syl: syl("l", "a", &[], vec![Bend { at: Edge::Middle, len: 0.4, max: 0.3, st: -4.0 }]) }], &opts());
    let b = labelled(&p.gestures, "bend");
    assert_eq!(b.len(), 1);
    let (vowel_start, vowel_end) = p.segments.iter().find(|s| s.kind == "vowel").map(|s| (s.t0, s.t1)).unwrap();
    assert!(((b[0].t0 + b[0].t1) / 2.0 - (vowel_start + vowel_end) / 2.0).abs() < 1e-9, "centred");
    assert!((b[0].t1 - b[0].t0 - 0.3).abs() < 1e-9, "capped at max");
}

#[test]
fn a_melisma_keeps_its_written_pitches() {
    let bend = vec![Bend { at: Edge::Start, len: 0.35, max: 0.15, st: -2.0 }];
    let first = syl("l", "a", &[], bend.clone());
    let cont = Syllable { text: "~".into(), nucleus: Segment::new("a"), melisma: Some(true), pitch: bend, ..Default::default() };
    let p = plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: first }, SungNote { midi: 59.0, beats: 1.0, syl: cont }], &opts());
    assert_eq!(labelled(&p.gestures, "bend").len(), 1, "only the syllable's first note bends");
}

#[test]
fn nasal_lead_nasalises_only_before_a_nasal_coda() {
    // a closing rhyme says its nasal lead (a front end: Mandarin's 0.6); unsaid, none; and a rhyme
    // that isn't a closing one has its coda as a part of its own, with nothing leading it
    let notes = |lead, closing| vec![SungNote { midi: 57.0, beats: 2.0, syl: Syllable { nasal_lead: lead, closing, ..syl("ɕ", "i", &["ŋ"], vec![]) } }, SungNote { midi: 59.0, beats: 2.0, syl: Syllable { nasal_lead: lead, closing, ..syl("t", "a", &[], vec![]) } }];
    assert!(labelled(&plan_song(&notes(None, Some(0.2)), &opts()).gestures, "nasal vowel").is_empty());
    assert!(labelled(&plan_song(&notes(Some(0.4), None), &opts()).gestures, "nasal vowel").is_empty());
    let p = plan_song(&notes(Some(0.4), Some(0.2)), &opts());
    let nv = labelled(&p.gestures, "nasal vowel");
    assert_eq!(nv.len(), 1, "only the syllable with a nasal coda");
    assert!(nv[0].targets.get(ArticKey::Velum).unwrap() > 0.0);
    assert!(nv[0].t0 > 0.5 && nv[0].t1 < 0.5 + 2.0);
}

#[test]
fn a_start_bend_begins_in_the_onset() {
    // a rising tone starts low: the onset consonant already sings the bend's pitch, not the note's
    let hz = midi_to_hz(57.0);
    let up = Bend { at: Edge::Start, len: 0.35, max: 0.15, st: -3.0 };
    let p = plan_song(&[SungNote { midi: 57.0, beats: 2.0, syl: syl("m", "a", &[], vec![up]) }], &opts());
    let onset_f0: Vec<f64> = p.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "pitch") && g.t1 <= 0.5 + 1e-9).filter_map(|g| g.targets.get(ControlKey::F0)).collect();
    assert!(!onset_f0.is_empty());
    assert!(onset_f0.iter().all(|f| (f / hz - 2f64.powf(-3.0 / 12.0)).abs() < 1e-9), "{onset_f0:?}");
    // without a start bend the onset sings the note
    let plain = plan_song(&[SungNote { midi: 57.0, beats: 2.0, syl: syl("m", "a", &[], vec![]) }], &opts());
    assert!(plain.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == "pitch") && g.t1 <= 0.5 + 1e-9).filter_map(|g| g.targets.get(ControlKey::F0)).all(|f| (f - hz).abs() < 1e-9));
}
