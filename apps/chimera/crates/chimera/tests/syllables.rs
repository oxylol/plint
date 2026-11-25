//! Port of tests/syllables.test.ts.
use chimera::phonetics::Segment;
use chimera::phonetics::english::arpabet_to_segments;
use chimera::phonetics::syllables::{lyrics_to_syllables, split_spelling, syllabify_segments};

// a few CMUdict entries, enough to test without loading the dictionary
fn dict(w: &str) -> Option<&'static str> {
    Some(match w {
        "twinkle" => "T W IH1 NG K AH0 L",
        "little" => "L IH1 T AH0 L",
        "star" => "S T AA1 R",
        "how" => "HH AW1",
        "i" => "AY1",
        "wonder" => "W AH1 N D ER0",
        "what" => "W AH1 T",
        "you" => "Y UW1",
        "are" => "AA1 R",
        "extra" => "EH1 K S T R AH0",
        "of" => "AH1 V",
        _ => return None,
    })
}
fn word(w: &str, _reduce: bool) -> Vec<Segment> {
    dict(w).map(arpabet_to_segments).unwrap_or_default()
}
fn ipa(xs: &[Segment]) -> String {
    xs.iter().map(|x| x.ipa.as_str()).collect()
}

#[test]
fn maximal_onset_with_legal_english_clusters() {
    let tw = syllabify_segments(&word("twinkle", false));
    let got: Vec<[String; 3]> = tw.iter().map(|s| [ipa(&s.onset), s.nucleus.ipa.clone(), ipa(&s.coda)]).collect();
    assert_eq!(got, [["tw", "ɪ", "ŋ"].map(String::from), ["k", "ə", "l"].map(String::from)]);
    // "extra": k + str → ek-stra (str is a legal onset)
    let ex = syllabify_segments(&word("extra", false));
    assert_eq!(ex.iter().map(|s| ipa(&s.onset)).collect::<Vec<_>>(), ["", "stɹ"]);
    assert_eq!(ipa(&ex[0].coda), "k");
}

#[test]
fn spelling_splits_to_match() {
    assert_eq!(split_spelling("twinkle", 2), ["twin", "kle"]);
    assert_eq!(split_spelling("little", 2), ["lit", "tle"]);
    assert_eq!(split_spelling("wonder", 2), ["won", "der"]);
    assert_eq!(split_spelling("are", 1), ["are"]);
}

#[test]
fn lyrics_words_melisma_phrase_ends_unreduced_function_words() {
    let syl = lyrics_to_syllables("Twinkle little star, how I - wonder", &word);
    assert_eq!(syl.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["Twin", "kle", "lit", "tle", "star", "how", "I", "~", "won", "der"]);
    assert!(syl[4].phrase_end);
    assert_eq!(syl[7].melisma, Some(true));
    assert_eq!(syl[7].nucleus.ipa, "aɪ");
    assert!(syl[1].word_end && !syl[0].word_end);
}

/// Each syllable as onset+nucleus+coda IPA, joined with "."; a stop's aspiration, when its front end
/// decides it, as ʰ (aspirated) or ˭ (unaspirated)
fn sung(text: &str) -> String {
    // aspiration with its length when the front end gives one: ʰ45 (45 ms), ʰ90
    let marked = |segs: &[Segment]| {
        segs.iter()
            .map(|s| {
                let asp = match (s.aspirated, s.vot) {
                    (Some(true), Some(v)) => format!("ʰ{}", (v * 1000.0).round()),
                    (Some(true), None) => "ʰ".into(),
                    (Some(false), _) => "˭".into(),
                    (None, _) => String::new(),
                };
                format!("{}{asp}", s.ipa)
            })
            .collect::<String>()
    };
    lyrics_to_syllables(text, &word).iter().map(|s| format!("{}{}{}", marked(&s.onset), s.nucleus.ipa, marked(&s.coda))).collect::<Vec<_>>().join(".")
}

#[test]
fn korean_step_stem_keeps_p_before_consonants_and_l_before_h() {
    for (text, expected) in [("밟다", "pʰ45ap.t˭a"), ("밟고", "pʰ45ap.k˭o"), ("밟는", "pʰ45am.nɯn"), ("밟히다", "pʰ45al.pʰ90i.da")] {
        assert_eq!(sung(text), expected, "{text}");
    }
}

#[test]
fn korean_stem_exceptions_preserve_liaison_and_unrelated_clusters() {
    for (text, expected) in [("읽다", "ik.t˭a"), ("읽어", "il.gʌ"), ("닭고기", "tʰ45ak.k˭o.gi"), ("밟아", "pʰ45al.ba"), ("넓어", "nʌl.bʌ"), ("여덟", "jʌ.dʌl")] {
        assert_eq!(sung(text), expected, "{text}");
    }
    let syllables = lyrics_to_syllables("읽고 밟히다", &word);
    assert_eq!(syllables.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["읽", "고", "밟", "히", "다"]);
    assert_eq!(syllables.iter().filter(|s| s.word_end).count(), 2);
    assert!(syllables.iter().all(|s| s.nucleus.dur.is_none()));
}

#[test]
fn hangul_sings_one_syllable_per_block_with_korean_sound_rules() {
    assert_eq!(sung("빵야"), "p˭aŋ.ja"); // ㅇ coda stays; tense ㅃ is voiceless and unaspirated
    assert_eq!(sung("먹어"), "mʌ.gʌ"); // a coda moves onto a silent ㅇ, voiced between vowels
    assert_eq!(sung("좋아"), "tɕʰ45o.a"); // ㅎ drops before a vowel; ㅈ is alveolo-palatal
    assert_eq!(sung("좋다"), "tɕʰ45o.tʰ90a"); // ㅎ + ㄷ aspirates
    assert_eq!(sung("국물"), "kʰ45uŋ.mul"); // nasalisation
    assert_eq!(sung("신라"), "ɕil.la"); // ㄴ + ㄹ is [ll]; ㅅ before i is [ɕ]
    assert_eq!(sung("없는"), "ʌm.nɯn"); // a two-letter coda simplifies, then nasalises
    assert_eq!(sung("학교"), "hak.k˭jo"); // tensing after an obstruent coda
    assert_eq!(sung("사랑해"), "sa.ɾaŋ.ɦɛ");
    assert_eq!(sung("닭"), "tʰ45ak");
    assert_eq!(sung("의사"), "ɰi.sa");
    // after a dropped ㅎ the consonant before it moves onto the silent ㅇ
    assert_eq!(sung("싫어"), "ɕi.ɾʌ"); // [시러]
    assert_eq!(sung("많아"), "ma.na"); // [마나]
    assert_eq!(sung("괜찮아"), "kʰ45wɛn.tɕʰ90a.na"); // [괜차나]
    assert_eq!(sung("싫니"), "ɕil.li"); // ㅀ before ㄴ: [실리]
    assert_eq!(sung("않는"), "an.nɯn"); // ㄶ before ㄴ: [안는]
    // ㄷ ㅌ before 이 palatalise
    assert_eq!(sung("같이"), "kʰ45a.tɕʰ90i"); // [가치]
    assert_eq!(sung("굳이"), "kʰ45u.dʑi"); // [구지]
}

#[test]
fn hangul_typed_as_jamo_sings_like_its_blocks() {
    // decomposed text (NFD: each block as its letters, conjoining jamo) used to be skipped whole,
    // so "한국 노래" sang nothing (#132)
    let nfd = "\u{1112}\u{1161}\u{11AB}\u{1100}\u{116E}\u{11A8} \u{1102}\u{1169}\u{1105}\u{1162}";
    assert_eq!(sung(nfd), sung("한국 노래"));
    let texts: Vec<String> = lyrics_to_syllables(nfd, &word).into_iter().map(|s| s.text).collect();
    assert_eq!(texts, ["한", "국", "노", "래"]);
    // a block with its final typed as a letter, and the sound rules across jamo blocks
    assert_eq!(sung("머\u{11A8}어"), sung("먹어"));
    // a block with a letter modern Korean dropped (here ㆍ) has no block and isn't read; the
    // blocks around it still are
    assert_eq!(sung("사\u{1105}\u{119E}\u{11B7} 노래"), sung("사 노래"));
}

#[test]
fn korean_stops_stay_three_ways_apart_word_initially() {
    // word-initially all three are voiceless: aspirated long, lax briefly, tense not at all
    // (불 fire, 풀 grass, 뿔 horn); they used to sing as pul, pul and bul
    assert_eq!(sung("불 풀 뿔"), "pʰ45ul.pʰ90ul.p˭ul");
    assert_eq!(sung("달 탈 딸"), "tʰ45al.tʰ90al.t˭al");
    assert_eq!(sung("가 카 까"), "kʰ45a.kʰ90a.k˭a");
    // inside a word: lax voiced between vowels, the other two as word-initially
    assert_eq!(sung("아기 아키 아끼"), "a.gi.a.kʰ90i.a.k˭i");
}

#[test]
fn lax_and_aspirated_stops_plan_different_aspiration() {
    use chimera::phonetics::planner::{PlanOptions, plan_utterance};
    // how long the folds stay apart after the release: the aspiration gesture's end
    let aspiration_end = |text: &str| {
        let s = &lyrics_to_syllables(text, &word)[0];
        let segs: Vec<Segment> = s.onset.iter().cloned().chain([Segment::new(&s.nucleus.ipa)]).chain(s.coda.iter().cloned()).collect();
        let plan = plan_utterance(&segs, &PlanOptions::default());
        plan.gestures.iter().find(|g| g.label.as_ref().map(|l| l.as_str()) == Some("aspiration")).map(|g| g.t1).unwrap()
    };
    let (lax, aspirated) = (aspiration_end("불"), aspiration_end("풀"));
    // the folds stay apart 45 ms longer for 풀: the difference its front end asked for
    assert!((aspirated - lax - 0.045).abs() < 1e-9, "lax {lax}, aspirated {aspirated}");
}

#[test]
fn text_after_a_letter_no_front_end_reads_is_kept() {
    // accented Latin letters aren't English's: they are skipped, the rest of the word still sings
    // (as before the router). A front end must never swallow what follows.
    let g2p = chimera::phonetics::english::EnglishG2P::new(); // letter-to-sound: reads any word
    let texts: Vec<String> = lyrics_to_syllables("Zürich über", &g2p).into_iter().map(|s| s.text).collect();
    assert!(texts.iter().any(|t| t == "rich") && texts.iter().any(|t| t == "ber"), "{texts:?}");
}

#[test]
fn hangul_keeps_its_spelling_and_mixes_with_english() {
    let syls = lyrics_to_syllables("twinkle 빵야 빵야, star", &word);
    let shown: Vec<(&str, bool, bool)> = syls.iter().map(|s| (s.text.as_str(), s.word_end, s.phrase_end)).collect();
    assert_eq!(
        shown,
        [("twin", false, false), ("kle", true, false), ("빵", false, false), ("야", true, false), ("빵", false, false), ("야", true, true), ("star", true, false)]
    );
}

#[test]
fn hand_written_syllables_may_leave_out_word_and_phrase_ends() {
    let s: chimera::phonetics::Syllable = serde_json::from_str(r#"{"text": "풀", "onset": [{"ipa": "p", "aspirated": true, "vot": 0.09}], "nucleus": {"ipa": "u"}, "coda": [{"ipa": "l"}]}"#).unwrap();
    assert!(!s.word_end && !s.phrase_end);
    assert_eq!(s.onset[0].vot, Some(0.09));
}

/// A front end that shares English's letters but reads only words in capitals (as pinyin reads
/// only words with tones).
struct Shouting;
impl chimera::phonetics::syllables::FrontEnd for Shouting {
    fn starts(&self, c: char) -> bool {
        c.is_ascii_alphabetic()
    }
    fn claims(&self, word: &str) -> bool {
        word.chars().all(|c| c.is_ascii_uppercase())
    }
    fn word(&self, word: &str) -> Vec<chimera::phonetics::Syllable> {
        vec![chimera::phonetics::Syllable { text: format!("!{word}"), nucleus: Segment::new("a"), ..Default::default() }]
    }
}

#[test]
fn a_declined_word_goes_to_the_next_front_end_and_nothing_is_lost() {
    use chimera::phonetics::syllables::{English, read_lyrics};
    let g2p = chimera::phonetics::english::EnglishG2P::new();
    let texts: Vec<String> = read_lyrics("HEY there LOUD", &[&Shouting, &English(&g2p)]).into_iter().map(|s| s.text).collect();
    assert_eq!(texts, ["!HEY", "there", "!LOUD"]);
    // the front end that declined a word can't claim a piece of it: "MIXed" is English's, not a
    // shouted "MIX" and "ed"
    let texts: Vec<String> = read_lyrics("MIXed", &[&Shouting, &English(&g2p)]).into_iter().map(|s| s.text).collect();
    assert!(texts.iter().all(|t| !t.starts_with('!')), "{texts:?}");
}
