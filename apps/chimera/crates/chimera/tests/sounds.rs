//! The sounds Korean and Mandarin need (#32 step 3): the alveolo-palatal and retroflex affricates are single tokens and real phonemes.
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::inventory::{Manner, phoneme};
use chimera::phonetics::ipa::tokenize_ipa;

#[test]
fn affricates_tokenize_as_one_segment() {
    let ipa: Vec<String> = tokenize_ipa("tɕi tɕʰi dʑa tʂa dʐu ɕy ʑə ɥɛ").into_iter().map(|s| s.ipa).collect();
    // ʰ isn't a segment: it sets the stop's aspiration (#73)
    assert_eq!(ipa, ["tɕ", "i", "tɕ", "i", "dʑ", "a", "tʂ", "a", "dʐ", "u", "ɕ", "y", "ʑ", "ə", "ɥ", "ɛ"]);
}

#[test]
fn mandarin_consonants_are_in_the_inventory() {
    for (ipa, manner, release) in [("ɕ", Manner::Fricative, None), ("ʑ", Manner::Fricative, None), ("tɕ", Manner::Affricate, Some("ɕ")), ("dʑ", Manner::Affricate, Some("ʑ")), ("tʂ", Manner::Affricate, Some("ʂ")), ("dʐ", Manner::Affricate, Some("ʐ")), ("ɥ", Manner::Approximant, None)] {
        let d = phoneme(ipa, BodyMix::MALE).unwrap_or_else(|| panic!("{ipa} missing"));
        assert_eq!(d.manner, manner, "{ipa}");
        assert_eq!(d.release, release, "{ipa}");
    }
}
