//! Sung release: a sung phrase that ends on a vowel or sonorant is let go by the folds (they part
//! slowly, from before the end, with the breath still on); a short final note, speech, an
//! open-ended note and a phrase ending in an obstruent end as they did before.
use chimera::anatomy::human::BODY_MALE;
use chimera::control::score::{ControlKey, Gesture};
use chimera::phonetics::planner::{EXHALE_LABEL, PlanOptions, plan_utterance};
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::{Intonation, Plan, Segment, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

const FS: f64 = 48000.0;

fn syl(onset: &str, vowel: &str, coda: &[&str]) -> Syllable {
    Syllable { text: "la".into(), onset: vec![Segment::new(onset)], nucleus: Segment::new(vowel), coda: coda.iter().map(|c| Segment::new(c)).collect(), word_end: true, ..Default::default() }
}

/// a phrase of quarter notes at `bpm`, the last one `last_beats` long, sung to `last`
fn sing(bpm: f64, last_beats: f64, last: Syllable) -> Plan {
    let mut notes: Vec<SungNote> = (0..3).map(|i| SungNote { midi: 57.0 + i as f64, beats: 1.0, syl: syl("l", "a", &[]) }).collect();
    notes.push(SungNote { midi: 62.0, beats: last_beats, syl: Syllable { phrase_end: true, ..last } });
    plan_song(&notes, &SongOptions { body: "male".into(), bpm, t0: 0.5, vibrato: 0.0, ..Default::default() })
}

fn labelled<'a>(p: &'a Plan, label: &str) -> Vec<&'a Gesture> {
    p.gestures.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == label)).collect()
}

/// the gesture that parts the folds at the end of the plan, and the end of the last breath
fn ending(p: &Plan) -> (&Gesture, f64) {
    let exhale = labelled(p, EXHALE_LABEL);
    let breath = labelled(p, "breath").iter().map(|g| g.t1).fold(f64::MIN, f64::max);
    (exhale.last().copied().expect("the plan parts the folds at its end"), breath)
}

fn last_sound_end(p: &Plan) -> f64 {
    p.segments.last().unwrap().t1
}

/// how the phrase ended before: a quick parting of the folds after the breath has stopped
fn ends_as_before(p: &Plan, after: f64) {
    let (g, breath) = ending(p);
    let end = last_sound_end(p);
    assert_eq!((g.ramp_in, g.strength), (0.03, 6.0), "the old, quick parting");
    assert!((g.t0 - (end + after)).abs() < 1e-9, "the folds part {after} s after the end, not before it: {} against {}", g.t0, end);
    assert!((breath - end).abs() < 1e-9, "the breath stops with the last sound");
}

#[test]
fn a_sung_phrase_ending_on_a_vowel_is_let_go_gradually() {
    let p = sing(60.0, 1.0, syl("l", "a", &[]));
    let (g, breath) = ending(&p);
    let end = last_sound_end(&p);
    assert_eq!(g.targets.get(ControlKey::ABDUCTION), Some(0.22));
    // the folds part over 150 ms, and the parting starts 90 ms before the note's end
    assert_eq!((g.ramp_in, g.strength), (0.15, 2.0));
    assert!((g.t0 - g.ramp_in - (end - 0.09)).abs() < 1e-9, "the opening starts 90 ms before the end");
    // the breath is still on as they part
    assert!(breath > end, "the breath outlasts the last sound: {breath} against {end}");
    // a sonorant after the vowel is part of the same note
    let moon = sing(60.0, 1.0, syl("m", "u", &["n"]));
    let (g, _) = ending(&moon);
    assert_eq!((g.ramp_in, g.strength), (0.15, 2.0));
}

/// the level (dB) of a plan's sound between two times
fn level(x: &[f32], a: f64, b: f64) -> f64 {
    let (i, j) = ((a * FS) as usize, (b * FS) as usize);
    10.0 * (x[i..j].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (j - i) as f64 + 1e-12).log10()
}

fn render(p: &Plan) -> Vec<f32> {
    let mut v = HumanVoice::new(FS, BODY_MALE, 2);
    v.set_source(&SourcePatch::f0(220.0));
    v.schedule(&p.gestures);
    let mut x = vec![0.0f32; (FS * (p.end + 0.3)) as usize];
    for chunk in x.chunks_mut(128) {
        v.render(chunk);
    }
    x
}

/// where the folds start to part, against the end of the last sound (s; negative = before it)
fn parting_starts(p: &Plan) -> f64 {
    let g = ending(p).0;
    g.t0 - g.ramp_in - last_sound_end(p)
}

/// the last note's own sound: its vowel (both halves of a diphthong) and what closes it
fn last_note_length(p: &Plan) -> f64 {
    let v = p.segments.iter().rposition(|s| s.kind == "vowel").unwrap();
    last_sound_end(p) - p.segments[v].t0
}

#[test]
fn a_short_final_note_keeps_its_level() {
    // at 200 bpm a quarter of a beat is 75 ms: the 90 ms lead started before the note had sounded
    // and the whole note came out about 7 dB quiet and breathy. It ends as it did before.
    let short = sing(200.0, 0.25, syl("l", "a", &[]));
    ends_as_before(&short, 0.045);
    assert!(parting_starts(&short) > 0.0, "nothing parts the folds inside the short note: {:+.3} s", parting_starts(&short));
    // in sound: the short last note is within a few dB of the note before it (the lost 7 dB put it 11.6 dB under)
    let (x, n) = (render(&short), &short.segments);
    let vowels: Vec<_> = n.iter().filter(|s| s.kind == "vowel").collect();
    let (before, last) = (vowels[vowels.len() - 2], vowels[vowels.len() - 1]);
    let drop = level(&x, before.t0, before.t1) - level(&x, last.t0, last.t1);
    println!("75 ms final note: {drop:.1} dB under the note before it");
    assert!(drop < 8.0, "the short final note is {drop:.1} dB under the note before it");
    // 125 ms at 120 bpm is still too short
    ends_as_before(&sing(120.0, 0.25, syl("l", "a", &[])), 0.045);
}

#[test]
fn the_folds_let_go_only_where_the_last_note_has_room() {
    // sweeping the last note's length: the gradual release from 180 ms of note (twice the 90 ms the
    // opening leads by), the old ending under it, and nothing else decides
    let (mut gradual, mut quick) = (0, 0);
    for step in 0..40 {
        let p = sing(120.0, 0.2 + 0.02 * step as f64, syl("l", "a", &[]));
        let (room, is_gradual) = (last_note_length(&p), ending(&p).0.ramp_in == 0.15);
        assert_eq!(is_gradual, room >= 0.18 - 1e-9, "a last note of {:.0} ms", room * 1000.0);
        if is_gradual { gradual += 1 } else { quick += 1 }
    }
    assert!(gradual > 5 && quick > 5, "the sweep crosses the limit: {quick} under, {gradual} over");
}

#[test]
fn the_note_before_gives_a_short_last_note_no_room() {
    let note = |midi: f64, beats: f64, s: Syllable| SungNote { midi, beats, syl: s };
    let song = |notes: &[SungNote]| plan_song(notes, &SongOptions { body: "male".into(), bpm: 200.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    let end = |s: Syllable| Syllable { phrase_end: true, ..s };
    // a melisma: one syllable held over three notes, the last of them 75 ms
    let held = Syllable { melisma: Some(true), ..syl("l", "a", &[]) };
    let melisma = song(&[note(57.0, 2.0, syl("l", "a", &[])), note(59.0, 2.0, held.clone()), note(60.0, 0.25, end(held))]);
    ends_as_before(&melisma, 0.045);
    // a hiatus: the last syllable has no consonant before its vowel ("di-a")
    let bare = Syllable { onset: vec![], ..syl("l", "a", &[]) };
    let hiatus = song(&[note(57.0, 2.0, syl("d", "i", &[])), note(59.0, 0.25, end(bare))]);
    ends_as_before(&hiatus, 0.045);
    // a diphthong is one note: both of its halves count, and a long one is let go gradually
    let eye = song(&[note(57.0, 2.0, syl("l", "a", &[])), note(59.0, 2.0, end(syl("m", "aɪ", &[])))]);
    assert_eq!(ending(&eye).0.ramp_in, 0.15);
    // a sonorant after an obstruent ("prism"): the room is the [m] alone, too short here
    let prism = song(&[note(57.0, 2.0, end(syl("p", "ɪ", &["z", "m"])))]);
    ends_as_before(&prism, 0.045);
}

#[test]
fn speech_open_ended_notes_and_obstruent_endings_plan_as_before() {
    // sung, ending in an obstruent: released into silence 15 ms after the end
    ends_as_before(&sing(60.0, 1.0, syl("k", "a", &["t"])), 0.015);
    // speech ending on a vowel: the quick parting once the pressure has fallen
    let segs = [Segment::new("l"), Segment::new("a")];
    let speech = plan_utterance(&segs, &PlanOptions { body: "male".into(), ..Default::default() });
    ends_as_before(&speech, 0.045);
    // an open-ended note (held until its key is released) is not ended by the plan at all
    let held = plan_utterance(&segs, &PlanOptions { body: "male".into(), f0: Some(220.0), intonation: Some(Intonation::Flat), open_end: Some(true), ..Default::default() });
    assert!(labelled(&held, EXHALE_LABEL).is_empty());
}
