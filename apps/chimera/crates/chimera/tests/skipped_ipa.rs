//! `skipped_ipa`: what the Speak box and the CLI tell a person the engine did not say. The owner's
//! report (issue 52): pasted Wiktionary IPA `ɕiɛ˥˩ɕiɛ` came out as a different word and nothing said so.

use chimera::anatomy::human::BodyMix;
use chimera::phonetics::ipa::{SkippedIpa, known_ipa_symbols, skipped_ipa, tokenize_ipa};
use chimera::phonetics::planner::is_vowel;

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn empty_and_all_known_input_report_nothing() {
    assert!(skipped_ipa("").is_empty());
    assert!(skipped_ipa("   ").is_empty());
    let r = skipped_ipa("ˈpʰaː.tɑ, bɑ|dɑ t͡ʃa");
    assert!(r.is_empty(), "{r:?}");
    assert!(skipped_ipa("ˈɑpɑ / ˈɑbɑ").is_empty());
}

#[test]
fn unknown_symbols_are_reported_the_owners_example() {
    // tone letters are not read, and ɕ is not in the inventory yet (when it is, it drops out of
    // this list by itself, so the expectation asks the function about ɕ alone)
    let r = skipped_ipa("ɕiɛ˥˩ɕiɛ");
    let mut want = skipped_ipa("ɕ").unknown;
    want.extend(strs(&["˥", "˩"]));
    assert_eq!(r.unknown, want);
    assert!(r.no_sound.is_empty() && r.ignored_marks.is_empty(), "{r:?}");
    // a capital, a digit, a click, a typo
    assert_eq!(skipped_ipa("Ab2ǃ").unknown, strs(&["A", "2", "ǃ"]));
}

#[test]
fn transcription_delimiters_are_structure() {
    // what people paste from Wiktionary: /phonemic/ and [phonetic]; only the real gaps are named
    let bare = skipped_ipa("ɕiɛ˥˩");
    assert!(bare.unknown.iter().any(|s| s == "˥") && bare.unknown.iter().any(|s| s == "˩"));
    assert_eq!(skipped_ipa("/ɕiɛ/"), skipped_ipa("ɕiɛ"));
    assert_eq!(skipped_ipa("[ɕiɛ˥˩]"), bare);
    assert_eq!(skipped_ipa("/ɕiɛ˥˩/ [ɕiɛ˥˩]"), bare);
    // anything else stays as it was: parentheses and the like are reported
    assert_eq!(skipped_ipa("/a(b)/").unknown, strs(&["(", ")"]));
    assert!(skipped_ipa("/ˈhɛlo/ [ˈhɛlo]").is_empty());
}

#[test]
fn order_is_input_order_and_each_item_once() {
    assert_eq!(skipped_ipa("˩a˥˩˥a˩").unknown, strs(&["˩", "˥"]));
    let r = skipped_ipa("n̥ n̥ a n̥ aʲ aʲ");
    assert_eq!(r.ignored_marks, strs(&["n̥", "aʲ"]));
}

#[test]
fn a_mark_the_tokenizer_reads_is_not_reported() {
    for ipa in ["pʰa p˭a", "a\u{303} ã ɛ\u{303} ɛː\u{303}", "tʃʲa pʲa", "bʱa dʱa", "a\u{324} ṳ ɛː\u{324}", "ˈaː a\u{361}b t͡s"] {
        assert!(skipped_ipa(ipa).is_empty(), "{ipa}: {:?}", skipped_ipa(ipa));
    }
}

/// Review of the PR: the syllabic (U+0329) and non-syllabic (U+032F) marks are in most dictionary
/// IPA and the voice says the sound, so a notice for them fires on many pastes for nothing.
#[test]
fn the_syllabic_and_non_syllabic_marks_are_not_reported() {
    // the reviewer's examples, and the same marks on other letters, after a length mark, at the
    // start, after a space or stress mark, and doubled
    for ipa in ["a\u{26A}\u{32F}", "b\u{28C}tn\u{329}", "l\u{329}", "\u{250}\u{32F}", "n\u{2D0}\u{329}", "\u{329}a", "a \u{32F}b", "\u{2C8}\u{329}a", "m\u{329}\u{32F}", "a\u{2D0}\u{32F}", "p\u{329}"] {
        assert!(skipped_ipa(ipa).is_empty(), "{ipa:?}: {:?}", skipped_ipa(ipa));
    }
}

#[test]
fn a_syllabic_mark_is_not_shown_beside_a_mark_that_is_reported() {
    // n + devoicing ring + syllabic mark reports the ring only
    assert_eq!(skipped_ipa("n\u{325}\u{329}").ignored_marks, strs(&["n\u{325}"]));
    assert_eq!(skipped_ipa("n\u{329}\u{325}").ignored_marks, strs(&["n\u{325}"]));
    // an unknown letter is still reported, without the mark
    assert_eq!(skipped_ipa("\u{298}\u{329}").unknown, strs(&["\u{298}"]));
}

#[test]
fn marks_that_are_not_modelled_or_cannot_apply_are_reported_with_their_letter() {
    // a devoicing ring and a nasal tilde on a consonant are not modelled
    assert_eq!(skipped_ipa("n\u{325}").ignored_marks, strs(&["n\u{325}"]));
    assert_eq!(skipped_ipa("m\u{303}").ignored_marks, strs(&["m\u{303}"]));
    // ʲ after a vowel, ʱ after a voiceless stop, and a vowel mark on a consonant after a length mark
    assert_eq!(skipped_ipa("aʲ").ignored_marks, strs(&["aʲ"]));
    assert_eq!(skipped_ipa("pʱ").ignored_marks, strs(&["pʱ"]));
    assert_eq!(skipped_ipa("mː\u{303}").ignored_marks, strs(&["m\u{303}"]));
    // the letter is still said: nothing else is reported
    let r = skipped_ipa("n\u{325}a");
    assert!(r.unknown.is_empty() && r.no_sound.is_empty(), "{r:?}");
    // a mark with nothing before it to carry it
    assert_eq!(skipped_ipa("ʲa").ignored_marks, strs(&["ʲ"]));
    assert_eq!(skipped_ipa("\u{325}a").unknown, strs(&["\u{25CC}\u{325}"]));
}

#[test]
fn structure_is_not_a_skipped_symbol() {
    assert!(skipped_ipa("ˈa ˌa aː a:. a, a| 'a\ta\n").is_empty());
}

#[test]
fn the_three_kinds_together() {
    // ɔ̃ and ʰ are read; n̥ keeps its letter and loses its ring; ʏ is a letter the tokenizer accepts
    // and, today, no body has a calibrated vowel for (what it is now is whatever `skipped_ipa("ʏ")`
    // says, so this holds when it gets one)
    let lone = skipped_ipa("ʏ");
    let r = skipped_ipa("bɔ̃ʒuʁ n̥ pʰa ʏ");
    assert_eq!(r, SkippedIpa { unknown: vec![], no_sound: lone.no_sound, ignored_marks: strs(&["n\u{325}"]) });
    let r = skipped_ipa("A n̥ ˥ ʏ");
    assert_eq!((r.unknown, r.ignored_marks), (strs(&["A", "˥"]), strs(&["n\u{325}"])));
}

/// The "no sound" kind is derived from the tokenizer's symbols and the planner's vowel table and
/// inventory, never a list of its own: feed every symbol the tokenizer knows, and the kind is
/// exactly the ones the planner cannot voice.
#[test]
fn no_sound_is_derived_not_listed() {
    let known = known_ipa_symbols();
    assert!(known.len() > 40);
    let all: String = known.iter().map(|s| format!("{s} ")).collect();
    let r = skipped_ipa(&all);
    assert!(r.unknown.is_empty(), "every known symbol is known: {r:?}");
    let silent: Vec<&str> = known
        .iter()
        .copied()
        .filter(|s| {
            let voiced = [BodyMix::MALE, BodyMix::FEMALE, BodyMix::CHILD].into_iter().any(|b| is_vowel(s, b)) || chimera::phonetics::inventory::consonant_symbols().contains(s);
            !voiced
        })
        .collect();
    // every reported symbol is one the tokenizer emits; every silent one is reported
    for s in &r.no_sound {
        assert!(known.contains(&s.as_str()), "{s}");
        assert!(tokenize_ipa(s).iter().any(|g| &g.ipa == s), "{s}");
    }
    let mut got: Vec<&str> = r.no_sound.iter().map(String::as_str).collect();
    got.sort_unstable();
    assert_eq!(got, silent);
    // vowel letters the tokenizer accepts but no body has a calibrated vowel for are in the kind
    assert!(!silent.is_empty(), "if every letter has a sound now, this kind is empty and the assertions above still hold");
    // and a symbol with a sound is not
    for ok in ["a", "ɛ", "p", "tʃ", "aɪ", "ɚ"] {
        assert!(!r.no_sound.contains(&ok.to_string()), "{ok}");
    }
}

#[test]
fn warnings_make_one_line_per_kind() {
    let r = skipped_ipa("ɕa˥ n̥");
    assert_eq!(r.warnings(), ["not said: ˥", "mark ignored: n̥"]);
    assert_eq!(skipped_ipa("ʏ").warnings(), ["no sound yet: ʏ"]);
    assert!(SkippedIpa::default().warnings().is_empty());
}
