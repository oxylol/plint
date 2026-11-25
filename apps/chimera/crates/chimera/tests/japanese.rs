//! Japanese: kana, one mora per note. The sounds each kana gets, ん as a note of its own (the vowel
//! before goes on into it, nasalised, and closes onto the nasal of what follows), っ as a note that ends in the
//! long consonant, long vowels as melisma, kanji left to the other front ends, and a kana phrase
//! sung through the planner and the voice.
use chimera::anatomy::human::{BODY_FEMALE, BODY_MALE, HumanBody};
use chimera::phonetics::english::{EnglishG2P, arpabet_to_segments};
use chimera::phonetics::japanese::Japanese;
use chimera::phonetics::korean::Korean;
use chimera::phonetics::mandarin::{Mandarin, MandarinLexicon, Pinyin};
use chimera::phonetics::russian::Russian;
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::songs::DEMO_SONGS;
use chimera::phonetics::syllables::{English, FrontEnd, lyrics_to_syllables, lyrics_to_syllables_with, read_lyrics};
use chimera::phonetics::{Plan, Segment, Syllable};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

fn word(w: &str, _reduce: bool) -> Vec<Segment> {
    match w {
        "star" => arpabet_to_segments("S T AA1 R"),
        "hello" => arpabet_to_segments("HH AH0 L OW1"),
        _ => Vec::new(),
    }
}

fn syls(text: &str) -> Vec<Syllable> {
    lyrics_to_syllables(text, &word)
}

/// Each syllable as its sounds, aspiration marked ʰ, a melisma note as ~ and its vowel, joined by dots
fn sung(text: &str) -> String {
    let marked = |segs: &[Segment]| segs.iter().map(|s| format!("{}{}", s.ipa, if s.aspirated == Some(true) { "ʰ" } else { "" })).collect::<String>();
    let notes: Vec<String> = syls(text)
        .iter()
        .map(|s| if s.melisma == Some(true) { format!("~{}", s.nucleus.ipa) } else { format!("{}{}{}", marked(&s.onset), s.nucleus.ipa, marked(&s.coda)) })
        .collect();
    notes.join(".")
}

/// The text shown on each note
fn shown(text: &str) -> Vec<String> {
    syls(text).into_iter().map(|s| s.text).collect()
}

#[test]
fn the_basic_table_and_the_irregular_kana() {
    assert_eq!(sung("さくら"), "sa.kʰɯ.ɾa");
    assert_eq!(sung("かえる"), "kʰa.e.ɾɯ");
    assert_eq!(sung("あいうえお"), "a.i.ɯ.e.o");
    assert_eq!(sung("なにぬねの"), "na.ɲi.nɯ.ne.no");
    assert_eq!(sung("まみむめも らりるれろ"), "ma.mi.mɯ.me.mo.ɾa.ɾi.ɾɯ.ɾe.ɾo");
    // わ is w: sung as the tongue-only [ɰ] a listener heard ら
    assert_eq!(sung("やゆよ わを"), "ja.jɯ.jo.wa.o");
    // し ち つ ふ ひ and the rows around them
    assert_eq!(sung("さしすせそ"), "sa.ɕi.sɯ.se.so");
    assert_eq!(sung("たちつてと"), "tʰa.tɕi.tsɯ.tʰe.tʰo");
    assert_eq!(sung("はひふへほ"), "ha.çi.ɸɯ.he.ho");
    // voiced, and the じ/ぢ and ず/づ pairs; ざ行 is an affricate at the start of a word
    assert_eq!(sung("ざじずぜぞ"), "dza.dʑi.zɯ.ze.zo");
    assert_eq!(sung("ぞう かぜ"), "dzo.~o.kʰa.ze");
    assert_eq!(sung("だぢづでど"), "da.dʑi.zɯ.de.do");
    assert_eq!(sung("がぎぐげご ばびぶべぼ"), "ga.gi.gɯ.ge.go.ba.bi.bɯ.be.bo");
    // the p row
    assert_eq!(sung("ぱぴぷぺぽ"), "pʰa.pʰi.pʰɯ.pʰe.pʰo");
    // the text is the kana as typed
    assert_eq!(shown("さくら"), ["さ", "く", "ら"]);
}

#[test]
fn voiceless_stops_say_their_aspiration_everywhere() {
    let s = syls("かきゃたぱ");
    // each is aspirated (k 70 ms, t and p 50), even before a glide
    let vot: Vec<Option<f64>> = s.iter().map(|x| x.onset[0].vot).collect();
    assert_eq!(vot, [Some(0.07), Some(0.07), Some(0.05), Some(0.05)]);
    assert!(s.iter().all(|x| x.onset[0].aspirated == Some(true)));
    // affricates, voiced and fricative consonants say nothing
    for kana in ["ち", "つ", "が", "さ"] {
        assert_eq!((syls(kana)[0].onset[0].aspirated, syls(kana)[0].onset[0].vot), (None, None), "{kana}");
    }
}

#[test]
fn a_yoon_palatalises_its_consonant() {
    // きゃ is [kʲja]: the tongue body is at the j while the k is made
    let s = syls("きゃ りょ びゅ ひゃ");
    assert!(s.iter().all(|x| x.onset.len() == 2 && x.onset[0].palatalized == Some(true) && x.onset[1].ipa == "j"));
    // ひゃ is h at the palate and its glide: as [ça] with no glide a listener heard no ひ
    assert_eq!(s[3].onset[0].ipa, "h");
    // but not ぎゃ's g: the tongue body is what closes a velar, and palatalised it let the voice
    // through, so a listener heard no ぎ. Its glide is short, so the release is next to the vowel
    let g = &syls("ぎゅ")[0].onset;
    assert_eq!((g[0].ipa.as_str(), g[0].palatalized, g[1].ipa.as_str(), g[1].dur), ("g", None, "j", Some(0.03)));
    // the consonants that are palatal themselves, and plain kana, say nothing
    for kana in ["しゃ", "ちゃ", "にゃ", "か", "や"] {
        assert!(syls(kana)[0].onset.iter().all(|c| c.palatalized.is_none()), "{kana}");
    }
}

#[test]
fn yoon_and_the_loanword_kana() {
    assert_eq!(sung("きょう"), "kʰjo.~o");
    assert_eq!(shown("きょう"), ["きょ", "う"]);
    assert_eq!(sung("しゃしん"), "ɕa.ɕiŋ.ŋ");
    assert_eq!(sung("りょこう"), "ɾjo.kʰo.~o");
    assert_eq!(sung("ちゃ じゃ にゃ ひゃ みゃ ぎゅ びょ ぴゅ"), "tɕa.dʑa.ɲa.hja.mja.gjɯ.bjo.pʰjɯ");
    // small kana for loanwords
    assert_eq!(sung("ふぁ ふぃ ふぇ ふぉ"), "ɸa.ɸi.ɸe.ɸo");
    assert_eq!(sung("てぃ でぃ とぅ どぅ"), "tʰi.di.tʰɯ.dɯ");
    assert_eq!(sung("うぃ うぇ うぉ"), "wi.we.wo");
    assert_eq!(sung("ちぇ しぇ じぇ"), "tɕe.ɕe.dʑe");
    assert_eq!(sung("ゔぁ ゔぃ ゔ ゔぇ ゔぉ"), "va.vi.vɯ.ve.vo");
    assert_eq!(sung("つぁ くぁ いぇ でゅ ふゅ"), "tsa.kʰwa.je.djɯ.ɸjɯ");
    // one note for the kana and its small kana
    assert_eq!(shown("ファイト"), ["ファ", "イ", "ト"]);
    assert_eq!(sung("ファイト"), "ɸa.i.tʰo");
}

#[test]
fn n_is_a_note_of_its_own_a_hum_on_the_nasal_of_what_follows() {
    // a mora is a note: a hum, the nasal as the note's nucleus, and the note before closes into it
    // (a short coda of the same nasal)
    assert_eq!(sung("しんぶん"), "ɕim.m.bɯŋ.ŋ"); // before b, and at the end of a word
    assert_eq!(sung("さんぽ"), "sam.m.pʰo");
    assert_eq!(sung("もんもん"), "mom.m.moŋ.ŋ");
    assert_eq!(sung("ぎんこう"), "giŋ.ŋ.kʰo.~o");
    assert_eq!(sung("まんが"), "maŋ.ŋ.ga");
    assert_eq!(sung("ほん"), "hoŋ.ŋ");
    assert_eq!(sung("こんや"), "kʰoŋ.ŋ.ja"); // before a glide
    assert_eq!(sung("れんあい"), "ɾeŋ.ŋ.a.i"); // before a vowel
    assert_eq!(sung("せんせい"), "seŋ.ŋ.se.i"); // before a fricative
    assert_eq!(sung("かんたん"), "kʰan.n.tʰaŋ.ŋ");
    assert_eq!(sung("ごりらんど"), "go.ɾi.ɾan.n.do");
    assert_eq!(sung("さんま"), "sam.m.ma");
    // before the palatals it is n
    assert_eq!(sung("こんにちわ"), "kʰon.n.ɲi.tɕi.wa");
    assert_eq!(sung("はんちょう"), "han.n.tɕo.~o");
    // the ざ行 after ん is an affricate; between vowels it is not
    assert_eq!(sung("あんざい"), "an.n.dza.i");
    assert_eq!(sung("かんずめ"), "kʰan.n.dzɯ.me");
    assert_eq!(sung("あざ"), "a.za");
    // ん's note shows the kana, has no onset or coda, and hums the nasal for the whole note; the
    // note before closes into it in one movement (a closing rhyme), its vowel nasalised over the
    // last 60 %. A listener chose this over the vowel going on into ん's note, on long notes
    // (かん on two beats was heard as "ka...n") and short ones
    let s = syls("かん");
    assert_eq!(s.len(), 2);
    assert_eq!((s[1].text.as_str(), s[1].melisma, s[1].nucleus.ipa.as_str(), s[1].onset.len(), s[1].coda.len()), ("ん", None, "ŋ", 0, 0));
    assert_eq!((s[0].coda[0].ipa.as_str(), s[0].coda[0].dur, s[0].closing, s[0].nasal_lead), ("ŋ", Some(0.08), Some(0.2), Some(0.6)));
    // after a long vowel the chain closes into it (the planner reads a chain's coda from its first
    // note), and a second ん goes on with the hum
    assert_eq!(sung("ほーん"), "hoŋ.~o.ŋ");
    assert_eq!(sung("あんん"), "aŋ.ŋ.~ŋ");
    // with no vowel before it (the start of a word) it starts on its own, and a long mark after it
    // holds the hum
    assert_eq!(sung("んば"), "m.ba");
    assert_eq!(sung("んー"), "ŋ.~ŋ");
}

#[test]
fn small_tsu_is_a_note_that_ends_in_the_long_consonant() {
    // がっこう: っ is a mora, so a note. The vowel before goes on into it, and the k after it is
    // held 0.25 s longer than a single one (0.064 s sung), out of that note
    assert_eq!(shown("がっこう"), ["が", "っ", "こ", "う"]);
    assert_eq!(sung("がっこう"), "ga.~a.kʰo.~o");
    assert!(syls("がこう")[1].onset[0].dur.is_none());
    let held = |text: &str| syls(text)[2].onset[0].dur.unwrap();
    assert!((held("がっこう") - 0.314).abs() < 1e-9);
    assert!(held("いっぱい") > 0.3 && held("いっぱい") < 0.35);
    assert!((held("ざっし") - 0.346).abs() < 1e-9);
    assert!((held("まっちゃ") - 0.362).abs() < 1e-9);
    assert_eq!(shown("きって"), ["き", "っ", "て"]);
    assert_eq!(sung("きって"), "kʰi.~i.tʰe");
    // with no consonant to hold (the end of a word, a vowel, a glide): a glottal stop cuts the note
    // before, and っ's own note is the stop held, silent
    assert_eq!(sung("あっ"), "aʔ.ʔ");
    assert_eq!(shown("あっ"), ["あ", "っ"]);
    assert_eq!(sung("あっあ"), "aʔ.ʔ.a");
    assert_eq!(sung("あっや"), "aʔ.ʔ.ja");
    // at the start of a word there is no note to go on from, and it is dropped
    assert_eq!(sung("っか"), "kʰa");
}

#[test]
fn long_vowels_continue_the_vowel_as_melisma() {
    assert_eq!(sung("コーヒー"), "kʰo.~o.çi.~i");
    assert_eq!(shown("コーヒー"), ["コ", "ー", "ヒ", "ー"]);
    assert_eq!(sung("おかあさん"), "o.kʰa.~a.saŋ.ŋ");
    assert_eq!(shown("おかあさん"), ["お", "か", "あ", "さ", "ん"]);
    assert_eq!(sung("とうきょう"), "tʰo.~o.kʰjo.~o"); // おう is oo
    assert_eq!(sung("せんせい"), "seŋ.ŋ.se.i"); // えい is e and i: singers sing the い
    assert_eq!(sung("ええ"), "e.~e");
    assert_eq!(sung("おおきい"), "o.~o.kʰi.~i");
    assert_eq!(sung("ゆうき"), "jɯ.~ɯ.kʰi");
    // 〜 holds the vowel like ー
    assert_eq!(sung("さ〜くら"), "sa.~a.kʰɯ.ɾa");
    // a different vowel is a note of its own
    assert_eq!(sung("かい"), "kʰa.i");
    assert_eq!(sung("とおい"), "tʰo.~o.i");
    // every melisma note is a continuation of the vowel before it
    for s in syls("コーヒー おかあさん").iter().filter(|s| s.melisma == Some(true)) {
        assert!(s.onset.is_empty() && s.coda.is_empty());
    }
}

#[test]
fn a_hyphen_starts_a_note_of_its_own_where_a_long_vowel_would_be_read() {
    // 思う is おもう, but its う is a syllable of its own
    assert_eq!(sung("おもう"), "o.mo.~o");
    assert_eq!(sung("おも-う"), "o.mo.ɯ");
    assert_eq!(shown("おも-う"), ["お", "も", "う"]);
    assert_eq!(sung("お-おきい"), "o.o.kʰi.~i");
    // and a hyphen at the end of a word, or between words, changes nothing
    assert_eq!(sung("さくら-"), "sa.kʰɯ.ɾa");
    assert_eq!(syls("さくら-さくら").len(), 6);
}

#[test]
fn katakana_and_half_width_katakana_read_like_hiragana() {
    assert_eq!(sung("コーヒー"), sung("こーひー"));
    assert_eq!(sung("ヴァイオリン"), "va.i.o.ɾiŋ.ŋ");
    assert_eq!(sung("ガッコウ"), sung("がっこう"));
    // half-width: voicing marks join the kana before them and stay in its text
    assert_eq!(sung("ｶﾞｯｺｳ"), "ga.~a.kʰo.~o");
    assert_eq!(shown("ｶﾞｯｺｳ"), ["ｶﾞ", "ｯ", "ｺ", "ｳ"]);
    assert_eq!(sung("ﾊﾟﾝ ｺｰﾋｰ"), "pʰaŋ.ŋ.kʰo.~o.çi.~i");
    // a combining mark does too (text typed as か + U+3099)
    assert_eq!(sung("か\u{3099}"), "ga");
    // iteration marks repeat the kana before them
    assert_eq!(sung("ここゝ"), "kʰo.kʰo.kʰo");
    assert_eq!(sung("はゞ"), "ha.ba");
}

#[test]
fn particles_read_as_written() {
    assert_eq!(sung("わたしは"), "wa.tʰa.ɕi.ha");
    assert_eq!(sung("へや"), "he.ja");
    assert_eq!(sung("わたしわ"), "wa.tʰa.ɕi.wa");
    // but the two greetings that end in the particle は are read wa, and show the kana typed
    assert_eq!(sung("こんにちは"), "kʰon.n.ɲi.tɕi.wa");
    assert_eq!(sung("こんばんは"), "kʰom.m.baŋ.ŋ.wa");
    assert_eq!(shown("こんにちは").last().unwrap(), "は");
}

#[test]
fn kana_mix_with_english_and_hangul_keeping_word_and_phrase_ends() {
    let s = syls("star さくら、빵야 hello");
    let got: Vec<(&str, bool, bool)> = s.iter().map(|s| (s.text.as_str(), s.word_end, s.phrase_end)).collect();
    assert_eq!(
        got,
        [("star", true, false), ("さ", false, false), ("く", false, false), ("ら", true, true), ("빵", false, false), ("야", true, false), ("hel", false, false), ("lo", true, false)]
    );
    // kana and English next to each other, with no space, are two words
    let t: Vec<(String, bool)> = syls("starさく").into_iter().map(|s| (s.text, s.word_end)).collect();
    assert_eq!(t, [("star".to_string(), true), ("さ".to_string(), false), ("く".to_string(), true)]);
    // ・ separates words
    let n: Vec<bool> = syls("コーヒー・ブレイク").iter().map(|s| s.word_end).collect();
    assert_eq!(n.iter().filter(|w| **w).count(), 2);
}

const LEX: &str = "你\tni3\n好\thao3\n我\two3\n爱\tai4\n";

#[test]
fn kanji_are_not_claimed_and_stay_with_the_other_front_ends() {
    // the Japanese front end starts on no Han character, and reads none
    assert!(!Japanese.starts('日') && !Japanese.starts('本') && !Japanese.continues('私'));
    assert_eq!(read_lyrics("日本", &[&Japanese]).len(), 0);
    // kanji inside a kana line are skipped (no Mandarin loaded) and the kana around them are sung
    assert_eq!(sung("さ私く"), "sa.kʰɯ");
    // with Mandarin loaded, Han characters are Mandarin's and the kana beside them Japanese's
    let mut lex = MandarinLexicon::new();
    lex.load(LEX);
    let g = EnglishG2P::new();
    let texts: Vec<String> = lyrics_to_syllables_with("你好さくら", &g, &lex).into_iter().map(|s| s.text).collect();
    assert_eq!(texts, ["你", "好", "さ", "く", "ら"]);
}

#[test]
fn other_languages_read_exactly_as_before() {
    let mut lex = MandarinLexicon::new();
    lex.load(LEX);
    let g = EnglishG2P::new();
    // "в ｰ лесу́": review of #148, a stray half-width ｰ started a Japanese word with no notes and the
    // router gave лесу́ the в's place
    for text in ["twinkle twinkle little star, how I wonder", "빵야 좋다 먹어, 국물", "молоко́ в лесу́ я́блоко", "nǐhǎo xièxie, café", "你好 我爱你, hello 빵야 nǐhǎo", "Zürich über ~ star - are", "в ｰ лесу́", "hello ｰ star"] {
        let before = read_lyrics(text, &[&Pinyin, &English(&g), &Korean, &Russian, &Mandarin(&lex)]);
        assert_eq!(lyrics_to_syllables_with(text, &g, &lex), before, "{text}");
    }
}

#[test]
fn full_width_punctuation_ends_phrases() {
    let phrases = |t: &str| syls(t).iter().filter(|s| s.phrase_end).count();
    assert_eq!(phrases("さくら、さくら。"), 2);
    assert_eq!(phrases("やよいの！そらは？ みわたす；"), 3);
    // the break lands on the last note of the word before it
    let s = syls("かん、あっ。");
    assert!(s[1].phrase_end && s[3].phrase_end && s[2].coda[0].ipa == "ʔ");
    assert_eq!(s[3].text, "っ");
}

#[test]
fn syllables_round_trip_through_json_with_the_format_the_back_end_reads() {
    let s = syls("きょう がっ かん");
    let json = serde_json::to_string(&s).unwrap();
    let back: Vec<Syllable> = serde_json::from_str(&json).unwrap();
    assert_eq!(s, back);
}

/// RMS of `x` between two times (s)
fn rms(x: &[f32], fs: f64, a: f64, b: f64) -> f64 {
    let (i, j) = ((a * fs) as usize, ((b * fs) as usize).min(x.len()));
    (x[i..j].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (j - i).max(1) as f64).sqrt()
}

fn sing_it(text: &str, name: &str, body: HumanBody, f0: f64) -> (Vec<f32>, Plan, Vec<f64>) {
    const FS: f64 = 48000.0;
    let notes: Vec<SungNote> = syls(text).into_iter().enumerate().map(|(i, syl)| SungNote { midi: [60.0, 62.0, 64.0, 62.0][i % 4], beats: 1.0, syl }).collect();
    let o = SongOptions { body: name.into(), bpm: 100.0, t0: 0.4, vibrato: 0.0, ..Default::default() };
    let plan = plan_song(&notes, &o);
    let mut voice = HumanVoice::new(FS, body, 2);
    voice.set_source(&SourcePatch::f0(f0));
    voice.schedule(&plan.gestures);
    let mut out = vec![0f32; (FS * (plan.end + 0.3)) as usize];
    for block in out.chunks_mut(128) {
        voice.render(block);
    }
    let beats: Vec<f64> = (0..notes.len()).map(|i| 0.4 + i as f64 * 0.6).collect();
    (out, plan, beats)
}

#[test]
fn a_kana_phrase_sings_through_the_planner_and_the_voice() {
    // every kind of mora: plain, yōon, a hum, っ before a stop, glottal stop, long vowel, loanword
    let text = "さくら さくら やよいの そらは んー がっこう きょう しゃしん ふぁいと こんにちは あっ コーヒー";
    for (name, body, f0) in [("male", BODY_MALE, 110.0), ("female", BODY_FEMALE, 200.0)] {
        let (out, plan, beats) = sing_it(text, name, body, f0);
        assert!(out.iter().all(|x| x.is_finite()), "{name}: the voice went NaN");
        assert!(plan.gestures.iter().all(|g| g.t0.is_finite() && g.t1.is_finite() && g.t0 <= g.t1 + 1e-9), "{name}: a gesture is not a span");
        let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
        assert!(peak > 0.05, "{name}: silence (peak {peak})");
        // every note sounds: its vowel (or hum), a tenth of a second after the beat, is audible.
        // Not a っ: its note ends in the next consonant's closure, or is a glottal stop held
        let mut quietest = f64::MAX;
        let texts = shown(text);
        for (k, &t) in beats.iter().enumerate() {
            let r = rms(&out, 48000.0, t + 0.1, t + 0.3);
            if texts[k] == "っ" {
                continue;
            }
            quietest = quietest.min(r / peak);
            assert!(r > 0.01 * peak, "{name}: note {k} is silent (RMS {r:.5}, peak {peak:.3})");
        }
        println!("{name}: peak {peak:.3}, quietest note RMS {:.3} of peak", quietest);
    }
}

#[test]
fn a_geminate_is_a_silent_closure_before_the_release_and_a_hum_is_voiced_for_its_whole_note() {
    // いっぱい at 60 bpm: a one-second note each. The closure before ぱ is silent for about 300 ms
    // (a single stop's is 60), and the ん of かんぱい is voiced all note long: the vowel, then the nasal
    const FS: f64 = 48000.0;
    let sing_one = |text: &str| {
        let notes: Vec<SungNote> = syls(text).into_iter().map(|syl| SungNote { midi: 64.0, beats: 1.0, syl }).collect();
        let plan = plan_song(&notes, &SongOptions { body: "female".into(), bpm: 60.0, t0: 0.4, vibrato: 0.0, ..Default::default() });
        let mut voice = HumanVoice::new(FS, BODY_FEMALE, 2);
        voice.set_source(&SourcePatch::f0(330.0));
        voice.schedule(&plan.gestures);
        let mut out = vec![0f32; (FS * (plan.end + 0.3)) as usize];
        for block in out.chunks_mut(128) {
            voice.render(block);
        }
        out
    };
    let out = sing_one("いっぱい");
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
    // ぱ is the third note, its beat at 2.4 s; count the 10 ms windows before it that are silent
    let silent = (0..50).filter(|k| rms(&out, FS, 2.4 - 0.01 * (*k as f64 + 1.0), 2.4 - 0.01 * *k as f64) < 0.02 * peak).count();
    assert!((24..=36).contains(&silent), "{silent} silent windows of 10 ms before the release");
    // and っ's own note starts with the vowel before, still sounding
    assert!(rms(&out, FS, 1.45, 1.75) > 0.1 * peak, "the vowel does not go on into っ's note");
    // あっ: the vowel is cut and っ's note is silent
    let out = sing_one("あっ");
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
    assert!(rms(&out, FS, 1.6, 2.2) < 0.01 * peak, "あっ: the っ is not silent");
    let out = sing_one("かんぱい");
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
    // ん is the second note, 1.4 s to 2.4 s: between 0.2 s and 0.8 s into it every 100 ms window
    // hums, quieter than か's vowel but steady
    let vowel = rms(&out, FS, 0.8, 1.1);
    for k in 2..8 {
        let a = 1.4 + 0.1 * k as f64;
        let r = rms(&out, FS, a, a + 0.1);
        assert!(r > 0.15 * vowel && r < 0.9 * vowel, "the hum at {a:.1} s: RMS {r:.4}, か's vowel {vowel:.4}, peak {peak:.3}");
    }
}

#[test]
fn the_demo_song_has_a_note_per_mora_and_sings_on_both_bodies() {
    let song = DEMO_SONGS.iter().find(|s| s.name == "さくら さくら").expect("a Japanese demo song");
    let syl = syls(song.lyrics);
    assert_eq!(syl.len(), song.notes.len(), "one sung syllable per note");
    // each line of the song is a phrase of its own
    assert_eq!(syl.iter().filter(|s| s.phrase_end).count(), 7);
    // the held vowels are melisma notes, and the last two notes are か and ん (a hum)
    assert_eq!(syl.iter().filter(|s| s.melisma == Some(true)).count(), 5);
    assert_eq!((syl[syl.len() - 2].text.as_str(), syl[syl.len() - 1].text.as_str()), ("か", "ん"));
    const FS: f64 = 48000.0;
    std::thread::scope(|scope| {
        let jobs: Vec<_> = [("male", BODY_MALE, song.octave.male, 110.0), ("female", BODY_FEMALE, song.octave.female, 220.0)]
            .map(|(name, body, shift, f0)| {
                let syl = syl.clone();
                scope.spawn(move || {
                    let notes: Vec<SungNote> = song.notes.iter().zip(syl).map(|(n, syl)| SungNote { midi: n[0] + shift * 12.0, beats: n[1], syl }).collect();
                    let plan = plan_song(&notes, &SongOptions { body: name.into(), bpm: song.bpm, t0: 0.35, vibrato: 0.0, ..Default::default() });
                    let mut voice = HumanVoice::new(FS, body, 2);
                    voice.set_source(&SourcePatch::f0(f0));
                    voice.schedule(&plan.gestures);
                    let mut out = vec![0f32; (FS * (plan.end + 0.3)) as usize];
                    for block in out.chunks_mut(128) {
                        voice.render(block);
                    }
                    assert!(out.iter().all(|x| x.is_finite()), "{name}: the voice went NaN");
                    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
                    assert!(peak > 0.05, "{name}: silence");
                    // every note sounds: the stretch a tenth of a second after its beat is audible
                    let (mut t, beat) = (0.35, 60.0 / song.bpm);
                    for (k, n) in song.notes.iter().enumerate() {
                        let r = rms(&out, FS, t + 0.1, t + 0.25);
                        assert!(r > 0.01 * peak, "{name}: note {k} is silent (RMS {r:.5}, peak {peak:.3})");
                        t += n[1] * beat;
                    }
                })
            })
            .into_iter()
            .collect();
        for j in jobs {
            j.join().unwrap();
        }
    });
}

#[test]
fn a_half_width_long_mark_continues_a_word_and_never_starts_one() {
    // review of #148: `в ｰ лесу́` sang ле су́, the в gone
    assert!(!Japanese.starts('\u{FF70}') && Japanese.continues('\u{FF70}'));
    // (the Russian front end sings a vowelless preposition with the word after it, as on main)
    assert_eq!(shown("в ｰ лесу́"), shown("в лесу́"));
    assert_eq!(shown("в ｰ лесу́"), ["в ле", "су́"]);
    // inside a word it still holds the vowel, as ー does
    assert_eq!(sung("ｺｰﾋｰ"), sung("コーヒー"));
}

#[test]
fn a_voiced_wa_row_kana_reads_alike_however_it_is_typed() {
    // review of #148: ヷ sang [v], ﾜﾞ and ワ + U+3099 sang [w]
    for (pre, half, combining, ipa) in [("ヷ", "ﾜﾞ", "ワ\u{3099}", "va"), ("ヺ", "ｦﾞ", "ヲ\u{3099}", "vo"), ("ヸ", "ヸ", "ヰ\u{3099}", "vi"), ("ヹ", "ヹ", "ヱ\u{3099}", "ve")] {
        assert_eq!(sung(pre), ipa, "{pre}");
        assert_eq!(sung(half), ipa, "{half}");
        assert_eq!(sung(combining), ipa, "{combining}");
    }
    // the hiragana and the spacing mark ゛ too, and the text keeps what was typed
    assert_eq!(sung("わ゛"), "va");
    assert_eq!(shown("ﾜﾞｲﾝ"), ["ﾜﾞ", "ｲ", "ﾝ"]);
}

#[test]
fn half_width_punctuation_ends_phrases_as_the_full_width_does() {
    // review of #148: ｻｸﾗ､ｻｸﾗ｡ gave no phrase end
    let ends = |t: &str| syls(t).iter().map(|s| s.phrase_end).collect::<Vec<_>>();
    assert_eq!(ends("ｻｸﾗ､ｻｸﾗ｡"), ends("サクラ、サクラ。"));
    assert_eq!(ends("ｻｸﾗ､ｻｸﾗ｡").iter().filter(|e| **e).count(), 2);
    assert_eq!(sung("ｻｸﾗ､ｻｸﾗ｡"), sung("サクラ、サクラ。"));
}
