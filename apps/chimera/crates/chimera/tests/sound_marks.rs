//! IPA marks that change how a sound is made (#32 step 1): the tokenizer keeps them as flags on the
//! segment, the planner turns each into gestures on controls the engine has, and a sung syllable
//! keeps them. Unflagged segments plan exactly as before (the planning goldens). Rendered tests
//! are in `sound_marks_acoustic.rs`.
use chimera::anatomy::human::{ArticKey, BodyMix};
use chimera::control::score::{ControlKey, Gesture};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::sing::{NoteOnOptions, SongOptions, SungNote, plan_note_on, plan_song};
use chimera::phonetics::{Plan, Segment, Syllable};

fn plan(segs: &[Segment]) -> Plan {
    plan_utterance(segs, &PlanOptions { body: BodyMix::MALE, ..Default::default() })
}

fn labelled<'a>(gs: &'a [Gesture], label: &str) -> Vec<&'a Gesture> {
    gs.iter().filter(|g| g.label.as_ref().is_some_and(|l| l.as_str() == label)).collect()
}

/// (ipa, flagged) of every segment of `ipa`, by a flag.
fn flags(ipa: &str, flag: fn(&Segment) -> Option<bool>) -> Vec<(String, bool)> {
    tokenize_ipa(ipa).iter().map(|s| (s.ipa.clone(), flag(s) == Some(true))).collect()
}

fn nasalized(s: &Segment) -> Option<bool> {
    s.nasalized
}

fn pairs(v: &[(&str, bool)]) -> Vec<(String, bool)> {
    v.iter().map(|(a, b)| (a.to_string(), *b)).collect()
}

#[test]
fn a_combining_tilde_nasalizes_the_vowel_before_it() {
    // ɛ̃ ɔ̃ ɑ̃ œ̃ have no precomposed form
    assert_eq!(flags("p\u{25B}\u{303}", nasalized), pairs(&[("p", false), ("ɛ", true)]));
    assert_eq!(flags("\u{153}\u{303}", nasalized), pairs(&[("œ", true)]));
    // a diphthong, and the tilde after the length mark
    assert_eq!(flags("a\u{26A}\u{303}", nasalized), pairs(&[("aɪ", true)]));
    let long = tokenize_ipa("\u{25B}\u{2D0}\u{303}");
    assert_eq!((long.len(), long[0].nasalized, long[0].dur), (1, Some(true), Some(0.15 * 1.7)));
}

#[test]
fn nasalized_vowels_read_the_same_precomposed_or_not() {
    // NFC joins a vowel and its tilde into one code point (ã õ ẽ ĩ ũ ỹ), which used to be dropped
    for (precomposed, base) in [("\u{E3}", "a"), ("\u{F5}", "o"), ("\u{1EBD}", "e"), ("\u{129}", "i"), ("\u{169}", "u"), ("\u{1EF9}", "y")] {
        let decomposed = format!("{base}\u{303}");
        assert_eq!(tokenize_ipa(precomposed), tokenize_ipa(&decomposed), "{precomposed}");
        assert_eq!(flags(precomposed, nasalized), pairs(&[(base, true)]), "{precomposed}");
    }
}

#[test]
fn a_tilde_on_a_consonant_is_ignored() {
    // an oral vowel after it stays oral; the consonant is as it was
    assert_eq!(tokenize_ipa("m\u{303}a"), tokenize_ipa("ma"));
    assert_eq!(tokenize_ipa("a \u{303}"), tokenize_ipa("a "));
}

#[test]
fn the_owners_french_pair_now_differs() {
    // paix / pain were both "p ɛ"
    assert_ne!(tokenize_ipa("pɛ"), tokenize_ipa("p\u{25B}\u{303}"));
    assert_eq!(tokenize_ipa("pɛ").iter().map(|s| &s.ipa).collect::<Vec<_>>(), tokenize_ipa("p\u{25B}\u{303}").iter().map(|s| &s.ipa).collect::<Vec<_>>());
}

#[test]
fn a_nasalized_vowel_lowers_the_velum_for_the_whole_vowel() {
    let oral = plan(&tokenize_ipa("pɛ"));
    let nasal = plan(&tokenize_ipa("p\u{25B}\u{303}"));
    assert!(labelled(&oral.gestures, "velum").is_empty());
    let v = labelled(&nasal.gestures, "velum");
    assert_eq!(v.len(), 1);
    let vowel = nasal.segments.iter().find(|s| s.kind == "vowel").unwrap();
    assert_eq!((v[0].t0, v[0].t1), (vowel.t0, vowel.t1));
    assert!(v[0].targets.get(ArticKey::Velum).unwrap() > 0.1);
    // nothing else about the plan changes
    // (compared as JSON: unset targets are NaN)
    let rest = |p: &Plan| serde_json::to_string(&p.gestures.iter().filter(|g| g.label.as_ref().is_none_or(|l| l.as_str() != "velum")).collect::<Vec<_>>()).unwrap();
    assert_eq!(rest(&oral), rest(&nasal));
    assert_eq!(oral.segments, nasal.segments);
}

#[test]
fn a_nasalized_diphthong_keeps_the_velum_down_through_its_glide() {
    let p = plan(&tokenize_ipa("a\u{26A}\u{303}"));
    let v = labelled(&p.gestures, "velum");
    assert_eq!(v.len(), 1);
    assert_eq!((v[0].t0, v[0].t1), (p.segments[0].t0, p.segments[1].t1));
}

fn syl(onset: Vec<Segment>, nucleus: Segment) -> Syllable {
    Syllable { text: "x".into(), onset, nucleus, coda: vec![], melisma: None, word_end: true, phrase_end: false, ..Default::default() }
}

#[test]
fn a_sung_vowel_keeps_its_nasalization() {
    let nucleus = |n: Option<bool>| Segment { nasalized: n, ..Segment::new("ɛ") };
    let sung = |n| plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: syl(vec![Segment::new("p")], nucleus(n)) }], &SongOptions { body: "male".into(), bpm: 90.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    assert_eq!(labelled(&sung(Some(true)).gestures, "velum").len(), 1);
    assert!(labelled(&sung(None).gestures, "velum").is_empty());
    // a live note
    let live = |n| plan_note_on(&syl(vec![Segment::new("p")], nucleus(n)), 57.0, &NoteOnOptions { body: "male".into(), vibrato: 0.0, tag: 1.0, legato_from: None });
    assert_eq!(labelled(&live(Some(true)), "velum").len(), 1);
    assert!(labelled(&live(None), "velum").is_empty());
    // a nasalized diphthong, sung: each half is a vowel of its own
    let diph = Segment { nasalized: Some(true), ..Segment::new("aɪ") };
    let p = plan_song(&[SungNote { midi: 57.0, beats: 2.0, syl: syl(vec![], diph) }], &SongOptions { body: "male".into(), bpm: 90.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    assert_eq!(labelled(&p.gestures, "velum").len(), 2);
}

fn palatalized(s: &Segment) -> Option<bool> {
    s.palatalized
}

#[test]
fn the_modifier_j_palatalizes_the_consonant_before_it() {
    assert_eq!(flags("tʲa", palatalized), pairs(&[("t", true), ("a", false)]));
    assert_eq!(flags("pʲ kʲ dʒʲ ɡʲ", palatalized), pairs(&[("p", true), ("k", true), ("dʒ", true), ("g", true)]));
    // with the other marks, in either order
    let s = tokenize_ipa("pʰʲ");
    assert_eq!((s[0].aspirated, s[0].palatalized), (Some(true), Some(true)));
}

#[test]
fn a_palatal_mark_on_the_wrong_sound_is_ignored() {
    // a vowel, a glottal (no mouth gesture), a pause, and no sound at all
    for ipa in ["aʲ", "ɛʲ", "hʲ", "ʔʲ", "ɦʲ", "ʲ", "ʲt", "|ʲ"] {
        assert!(tokenize_ipa(ipa).iter().all(|s| s.palatalized.is_none()), "{ipa}");
    }
    assert_eq!(tokenize_ipa("aʲ"), tokenize_ipa("a"));
}

#[test]
fn the_owners_russian_pair_now_differs() {
    // брат / брать were both "b r ɑ t"
    let (brat, brat_soft) = (tokenize_ipa("brɑt"), tokenize_ipa("brɑtʲ"));
    assert_ne!(brat, brat_soft);
    assert_eq!(brat.iter().map(|s| &s.ipa).collect::<Vec<_>>(), brat_soft.iter().map(|s| &s.ipa).collect::<Vec<_>>());
    assert_eq!(brat_soft.iter().map(|s| s.palatalized).collect::<Vec<_>>(), [None, None, None, Some(true)]);
}

#[test]
fn a_palatalized_consonant_gets_a_tongue_body_gesture_that_leaves_its_own_alone() {
    let own = |ipa: &str| chimera::phonetics::inventory::phoneme(ipa, BodyMix::MALE).unwrap().dominant;
    for c in ["p", "t", "d", "k", "g", "s", "ʃ", "n", "m", "l", "f", "ɹ", "tʃ"] {
        let plain = plan(&tokenize_ipa(&format!("{c}a")));
        let soft = plan(&tokenize_ipa(&format!("{c}ʲa")));
        assert!(labelled(&plain.gestures, "palatalized").is_empty(), "{c}");
        let g = labelled(&soft.gestures, "palatalized");
        assert_eq!(g.len(), 1, "{c}");
        // only the tongue body, and never a coordinate the consonant's closure or constriction owns
        for k in ArticKey::ALL {
            if g[0].targets.get(k).is_some() {
                assert!(matches!(k, ArticKey::TbX | ArticKey::TbY), "{c}: {k:?}");
                assert!(!own(c).contains(&k), "{c} owns {k:?}");
            }
        }
        // the consonant's own gestures and everything else are unchanged
        let rest = |p: &Plan| serde_json::to_string(&p.gestures.iter().filter(|g| g.label.as_ref().is_none_or(|l| l.as_str() != "palatalized")).collect::<Vec<_>>()).unwrap();
        assert_eq!(rest(&plain), rest(&soft), "{c}");
    }
    // a velar is fronted, not raised: its closure owns the body's height
    let k = plan(&tokenize_ipa("kʲa"));
    let g = labelled(&k.gestures, "palatalized")[0];
    assert!(g.targets.get(ArticKey::TbX).is_some() && g.targets.get(ArticKey::TbY).is_none());
    // a consonant with nothing free on the body, or already palatal, adds no gesture
    for c in ["j", "ɲ", "w", "ɫ", "q"] {
        assert!(labelled(&plan(&tokenize_ipa(&format!("{c}ʲa"))).gestures, "palatalized").is_empty(), "{c}");
    }
}

#[test]
fn the_palatal_posture_holds_until_the_voice_returns() {
    // after an aspirated stop that is later than the release: the [j]-like transition is voiced
    let seg = |aspirated| vec![Segment { aspirated, palatalized: Some(true), ..Segment::new("t") }, Segment::new("a")];
    let end = |aspirated| labelled(&plan(&seg(aspirated)).gestures, "palatalized")[0].t1;
    let t1 = plan(&seg(None)).segments[0].t1;
    assert!(end(Some(true)) > t1 + 0.05, "aspirated: {}", end(Some(true)) - t1);
    assert!(end(Some(false)) < t1 + 0.03, "unaspirated: {}", end(Some(false)) - t1);
}

fn breathy(s: &Segment) -> Option<bool> {
    s.breathy
}

#[test]
fn a_breathy_release_mark_sets_voiced_stops_and_affricates() {
    assert_eq!(flags("bʱa", breathy), pairs(&[("b", true), ("a", false)]));
    assert_eq!(flags("bʱ dʱ ɡʱ ɖʱ dʒʱ dzʱ", breathy), pairs(&[("b", true), ("d", true), ("g", true), ("ɖ", true), ("dʒ", true), ("dz", true)]));
}

#[test]
fn the_combining_diaeresis_below_makes_a_vowel_breathy() {
    assert_eq!(flags("a\u{324}", breathy), pairs(&[("a", true)]));
    assert_eq!(flags("\u{25B}\u{324}", breathy), pairs(&[("ɛ", true)]));
    // ṳ is one code point after NFC; the length mark may come first (aː̤)
    assert_eq!(tokenize_ipa("\u{1E73}"), tokenize_ipa("u\u{324}"));
    assert_eq!(flags("\u{1E73}", breathy), pairs(&[("u", true)]));
    let long = tokenize_ipa("a\u{2D0}\u{324}");
    assert_eq!((long.len(), long[0].breathy, long[0].dur), (1, Some(true), Some(0.15 * 1.7)));
    // with a nasal mark too, in either order
    for ipa in ["a\u{324}\u{303}", "a\u{303}\u{324}"] {
        let s = tokenize_ipa(ipa);
        assert_eq!((s.len(), s[0].breathy, s[0].nasalized), (1, Some(true), Some(true)), "{ipa:?}");
    }
}

#[test]
fn a_breathy_mark_on_the_wrong_sound_is_ignored() {
    // ʱ: not after a voiceless stop, a fricative, a nasal, a glottal, a vowel, a pause, or alone
    for ipa in ["pʱ", "sʱ", "zʱ", "mʱ", "hʱ", "ɦʱ", "aʱ", "|ʱ", "ʱ", "ʱb"] {
        assert!(tokenize_ipa(ipa).iter().all(|s| s.breathy.is_none()), "{ipa}");
    }
    // the diaeresis below: not on a consonant
    for ipa in ["b\u{324}a", "m\u{324}a", "s\u{324}a", "\u{324}a"] {
        assert!(tokenize_ipa(ipa).iter().all(|s| s.breathy.is_none()), "{ipa:?}");
    }
    assert_eq!(tokenize_ipa("m\u{324}a"), tokenize_ipa("ma"));
}

#[test]
fn the_owners_hindi_pair_now_differs() {
    // पल / फल were both "p ʌ l"; and बल / भल differ by the breathy release
    assert_ne!(tokenize_ipa("pʌl"), tokenize_ipa("pʰʌl"));
    assert_ne!(tokenize_ipa("bʌl"), tokenize_ipa("bʱʌl"));
    assert_ne!(tokenize_ipa("bʌl"), tokenize_ipa("pʰʌl"));
}

#[test]
fn a_breathy_release_parts_the_folds_across_the_release_and_into_the_vowel() {
    let seg = |b| vec![Segment { breathy: b, ..Segment::new("b") }, Segment { stress: Some(1), ..Segment::new("a") }];
    let (modal, soft) = (plan(&seg(None)), plan(&seg(Some(true))));
    assert!(labelled(&modal.gestures, "breathy").is_empty());
    let g = labelled(&soft.gestures, "breathy");
    assert_eq!(g.len(), 1);
    let (stop, vowel) = (&soft.segments[0], &soft.segments[1]);
    // ɦ's abduction, from just before the release to 80 ms after it (the vowel here lasts longer)
    assert_eq!(g[0].targets.get(ControlKey::ABDUCTION), Some(0.1));
    assert!((g[0].t0 - (stop.t1 - 0.02)).abs() < 1e-9 && (g[0].t1 - (stop.t1 + 0.08)).abs() < 1e-9 && vowel.t1 > g[0].t1);
    // a short vowel ends it
    let short = plan(&[Segment { breathy: Some(true), ..Segment::new("b") }, Segment { dur: Some(0.04), ..Segment::new("a") }]);
    assert_eq!(labelled(&short.gestures, "breathy")[0].t1, short.segments[1].t1);
    // nothing else about the plan changes
    let rest = |p: &Plan| serde_json::to_string(&p.gestures.iter().filter(|g| g.label.as_ref().is_none_or(|l| l.as_str() != "breathy")).collect::<Vec<_>>()).unwrap();
    assert_eq!(rest(&modal), rest(&soft));
    // a flag on a sound the mark does not apply to does nothing
    for c in ["p", "s", "m", "ɦ", "l"] {
        let p = plan(&[Segment { breathy: Some(true), ..Segment::new(c) }, Segment::new("a")]);
        assert!(labelled(&p.gestures, "breathy").is_empty(), "{c}");
    }
}

#[test]
fn a_breathy_vowel_parts_the_folds_for_the_whole_vowel() {
    let modal = plan(&tokenize_ipa("pa"));
    let soft = plan(&tokenize_ipa("pa\u{324}"));
    assert!(labelled(&modal.gestures, "breathy").is_empty());
    let g = labelled(&soft.gestures, "breathy");
    assert_eq!(g.len(), 1);
    assert_eq!((g[0].t0, g[0].t1), (soft.segments[1].t0, soft.segments[1].t1));
    assert_eq!(g[0].targets.get(ControlKey::ABDUCTION), Some(0.1));
    // through a diphthong's glide
    let d = plan(&tokenize_ipa("a\u{26A}\u{324}"));
    let g = labelled(&d.gestures, "breathy");
    assert_eq!((g.len(), g[0].t0, g[0].t1), (1, d.segments[0].t0, d.segments[1].t1));
}

#[test]
fn sung_syllables_keep_their_breathiness() {
    let song = |onset: Vec<Segment>, nucleus: Segment| plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: syl(onset, nucleus) }], &SongOptions { body: "male".into(), bpm: 90.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    let live = |onset: Vec<Segment>, nucleus: Segment| plan_note_on(&syl(onset, nucleus), 57.0, &NoteOnOptions { body: "male".into(), vibrato: 0.0, tag: 1.0, legato_from: None });
    let (b, b_soft) = (Segment::new("b"), Segment { breathy: Some(true), ..Segment::new("b") });
    let (a, a_soft) = (Segment::new("a"), Segment { breathy: Some(true), ..Segment::new("a") });
    assert_eq!(labelled(&song(vec![b_soft.clone()], a.clone()).gestures, "breathy").len(), 1);
    assert_eq!(labelled(&song(vec![b.clone()], a_soft.clone()).gestures, "breathy").len(), 1);
    assert!(labelled(&song(vec![b.clone()], a.clone()).gestures, "breathy").is_empty());
    assert_eq!(labelled(&live(vec![b_soft], a.clone()), "breathy").len(), 1);
    assert_eq!(labelled(&live(vec![b.clone()], a_soft), "breathy").len(), 1);
    assert!(labelled(&live(vec![b], a), "breathy").is_empty());
}

#[test]
fn a_sung_consonant_keeps_its_palatalization() {
    let onset = |soft| vec![Segment { palatalized: soft, ..Segment::new("t") }];
    let sung = |s| plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: syl(onset(s), Segment::new("a")) }], &SongOptions { body: "male".into(), bpm: 90.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    assert_eq!(labelled(&sung(Some(true)).gestures, "palatalized").len(), 1);
    assert!(labelled(&sung(None).gestures, "palatalized").is_empty());
    let live = |s| plan_note_on(&syl(onset(s), Segment::new("a")), 57.0, &NoteOnOptions { body: "male".into(), vibrato: 0.0, tag: 1.0, legato_from: None });
    assert_eq!(labelled(&live(Some(true)), "palatalized").len(), 1);
    assert!(labelled(&live(None), "palatalized").is_empty());
    // a coda too
    let coda = |s| Syllable { coda: onset(s), ..syl(vec![], Segment::new("a")) };
    let p = plan_song(&[SungNote { midi: 57.0, beats: 1.0, syl: coda(Some(true)) }], &SongOptions { body: "male".into(), bpm: 90.0, t0: 0.5, vibrato: 0.0, ..Default::default() });
    assert_eq!(labelled(&p.gestures, "palatalized").len(), 1);
}
