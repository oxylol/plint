//! Mandarin: the front end reads Han characters: longest phrase match, tone sandhi, pinyin → IPA
//! with aspiration said, tones read and not sung; and pinyin typed with tones.
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::korean::Korean;
use chimera::phonetics::mandarin::{Mandarin, MandarinLexicon, Pinyin};
use chimera::phonetics::syllables::{English, FrontEnd, read_lyrics};
use chimera::phonetics::Syllable;

const LEX: &str = "举\tju3\n风\tfeng1\n多\tduo1\n头\ttou2\n鸟\tniao3\n光\tguang1\n四\tsi4\n是\tshi4\n点\tdian3\n哪\tna3\n儿\ter2\n女\tnv3\n子\tzi5\n块\tkuai4\n一会儿\tyi1 hui4 er5\n天\ttian1\n你\tni3\n好\thao3\n我\two3\n不\tbu4\n会\thui4\n银\tyin2\n行\txing2\n银行\tyin2 hang2\n走\tzou3\n的\tde5\n跑\tpao3\n一\tyi1\n个\tge4\n绿\tlv4\n学\txue2\n吃\tchi1\n爱\tai4\n去\tqu4\n";

fn lex() -> MandarinLexicon {
    let mut l = MandarinLexicon::new();
    l.load(LEX);
    l
}

fn syls(text: &str) -> Vec<Syllable> {
    let l = lex();
    read_lyrics(text, &[&Mandarin(&l)])
}

/// onset+nucleus+coda, with ʰ/˭ where the front end says a stop's aspiration
fn ipa(s: &Syllable) -> String {
    let m = |segs: &[chimera::phonetics::Segment]| segs.iter().map(|x| format!("{}{}", x.ipa, match x.aspirated { Some(true) => "ʰ", Some(false) => "˭", None => "" })).collect::<String>();
    format!("{}{}{}", m(&s.onset), s.nucleus.ipa, m(&s.coda))
}

#[test]
fn readings_by_longest_phrase_then_tone_sandhi() {
    let l = lex();
    assert_eq!(l.read("银行"), ["yin2", "hang2"]); // the phrase decides 行
    assert_eq!(l.read("行走"), ["xing2", "zou3"]); // no phrase: the character's own reading
    assert_eq!(l.read("你好"), ["ni2", "hao3"]); // 3-3 → 2-3
    assert_eq!(l.read("我你好"), ["wo2", "ni2", "hao3"]); // a run of third tones
    assert_eq!(l.read("不会"), ["bu2", "hui4"]); // 不 before a fourth tone
    assert_eq!(l.read("一个"), ["yi2", "ge4"]); // 一 before a fourth tone
    assert_eq!(l.read("不去"), ["bu2", "qu4"]);
}

#[test]
fn pinyin_becomes_ipa_with_aspiration_said() {
    let all: Vec<String> = syls("跑不吃学绿的爱").iter().map(ipa).collect();
    // p aspirated, b unaspirated; ch aspirated before an apical vowel; xue = ɕ ɥ ɛ; lü; neutral de; ai
    // apical ʅ after ch; xue's glide sings inside the vowel (yɛ); 爱 starts with a glottal catch
    assert_eq!(all, ["pʰaʊ̠", "p˭u̠", "tʂʰʅ", "ɕyɛ", "ly", "t˭ə", "ʔaɪ"]);
}

#[test]
fn tones_are_read_and_not_sung_and_text_keeps_the_character() {
    let s = syls("你好");
    assert_eq!((s[0].text.as_str(), s[1].text.as_str()), ("你", "好"));
    assert!(s[1].word_end && !s[0].word_end);
    // the reading has its tones, sandhi applied (你 before a third tone is second)
    assert_eq!(lex().read("你好"), ["ni2", "hao3"]);
    // and no syllable bends the pitch for them: a song sings its melody. A native listener chose
    // this over the tones drawn on the melody at half and at full depth ("flat sounds sung, the
    // full tones sound spoken, half sounds odd")
    assert!(syls("你好，我是中国人，谢谢").iter().all(|s| s.pitch.is_empty()));
}

#[test]
fn a_mixed_line_routes_each_word_to_its_front_end() {
    let l = lex();
    let mut g = EnglishG2P::new();
    g.load_dict("you\tY UW1\n");
    let out = read_lyrics("我爱 you, 你。", &[&English(&g), &Korean, &Mandarin(&l)]);
    let shown: Vec<(&str, bool, bool)> = out.iter().map(|s| (s.text.as_str(), s.word_end, s.phrase_end)).collect();
    assert_eq!(shown, [("我", false, false), ("爱", true, false), ("you", true, true), ("你", true, true)]);
    // no lexicon loaded: Han reads as nothing, the rest is unaffected
    let empty = MandarinLexicon::new();
    assert_eq!(read_lyrics("我 you", &[&English(&g), &Mandarin(&empty)]).len(), 1);
}

fn typed(text: &str) -> Vec<(String, String, usize)> {
    let mut g = EnglishG2P::new();
    g.load_dict("hello\tHH AH0 L OW1\ncaf\tK AE1 F\nit's\tIH1 T S\nz\tZ IY1\nrich\tR IH1 CH\nber\tB ER1\n");
    read_lyrics(text, &[&Pinyin, &English(&g), &Korean]).iter().map(|s| (s.text.clone(), ipa(s), s.pitch.len())).collect()
}

#[test]
fn pinyin_with_tones_is_mandarin_and_without_is_english() {
    let t = typed("nǐhǎo xièxie ni3 hao3 de5 Xī'ān nǚ");
    let text: Vec<&str> = t.iter().map(|x| x.0.as_str()).collect();
    assert_eq!(text, ["nǐ", "hǎo", "xiè", "xie", "ni3", "hao3", "de5", "Xī", "ān", "nǚ"]); // capitals kept
    let ipa: Vec<&str> = t.iter().map(|x| x.1.as_str()).collect();
    assert_eq!(ipa, ["ni", "xaʊ̠", "ɕiɛ", "ɕiɛ", "ni", "xaʊ̠", "t˭ə", "ɕi", "ʔan", "ny"]);
    // the tones make it pinyin; none of them bends the pitch (a song sings its melody)
    assert!(t.iter().all(|x| x.2 == 0));
    // English: no tones, not pinyin. A word pinyin declines reads as it would without pinyin
    // ("café" English's "caf", not "caf" and a pinyin "é"), and nothing in it is lost: "Zürich über"
    // sings "Z rich ber" as on main (review of #78: it sang "Z ber")
    let e: Vec<String> = typed("hello café it's Zürich über").into_iter().map(|x| x.0).collect();
    assert_eq!(e, ["hel", "lo", "caf", "it's", "Z", "rich", "ber"]);
    // a one-letter word stays English ("room a1", French "à")
    let a: Vec<String> = typed("a1 à").into_iter().map(|x| x.0).collect();
    assert_eq!(a, ["a"]); // "à" isn't English's (accented letters are skipped, as on main)
}

#[test]
fn accented_english_words_stay_english() {
    // pinyin's spelling rules keep these out (review of #85: 12 of 42 were read as pinyin): a toned
    // word starts with a written tone, an unwritten one inside is only bu, de or yi (résumé,
    // touché, lamé), a vowel-initial syllable inside needs an apostrophe (née, matinée, mélange,
    // brûlée), and n, ng, m are syllables only as whole words (Pokémon, détente)
    for w in ["café", "résumé", "cliché", "naïve", "brûlée", "Pokémon", "touché", "détente", "matinée", "née", "lamé", "saké", "mélange", "à", "olé", "Renée"] {
        assert!(!Pinyin.claims(w), "{w} read as pinyin");
    }
    // the one left: é + lan is real pinyin with a neutral last syllable
    assert!(Pinyin.claims("élan"));
    // and real pinyin still reads, including a neutral bu or de inside and apostrophes
    // review of #85: a neutral me, zi, you inside a word (zěnmeyàng, háizimen, péngyoumen)
    for w in ["nǐhǎo", "xièxie", "dōngxi", "duìbuqǐ", "kànbujiàn", "Xī'ān", "Tiān'ānmén", "nǚ'ér", "ni3hao3", "zhong1guo2", "xǐhuan", "bú", "zěnmeyàng", "shénmeshíhou", "háizimen", "péngyoumen", "zen3meyang4"] {
        assert!(Pinyin.claims(w), "{w} not read as pinyin");
    }
}

#[test]
fn the_front_end_says_aspiration_length_and_nasalisation() {
    let s = syls("跑的天");
    assert_eq!(s[0].onset[0].vot, Some(0.09)); // aspirated: Mandarin's long aspiration
    assert_eq!(s[1].onset[0].vot, None); // unaspirated
    assert!(s[2].nasal_lead.is_some()); // tiān: nasalised before -n
    assert_eq!(s[0].nasal_lead, None);
}

#[test]
fn apical_vowels_follow_their_consonant() {
    let n: Vec<String> = syls("四是").iter().map(|s| s.nucleus.ipa.clone()).collect();
    assert_eq!(n, ["ɿ", "ʅ"]);
}

#[test]
fn erhua_merges_into_the_syllable_before() {
    let shown = |t: &str| syls(t).iter().map(|s| format!("{}={}", s.text, ipa(s))).collect::<Vec<_>>().join(" ");
    // erhua: the syllable ends in ɻ, its -n gives way, a falling diphthong keeps its first vowel
    assert_eq!(shown("一点儿"), "一=i 点儿=t˭iɛɻ");
    assert_eq!(shown("块儿"), "块儿=kʰu̠aɻ"); // uai → ua + ɻ (the off-glide gives way, the on-glide stays)
    assert_eq!(shown("哪儿"), "哪儿=naɻ");
    assert_eq!(shown("一会儿"), "一=i 会儿=xweɻ");
    // 儿 as a word's own syllable
    assert_eq!(shown("女儿"), "女=ny 儿=ʔəɝ");
    assert_eq!(shown("儿子"), "儿=ʔəɝ 子=ts˭ɿ");
    // pinyin's r
    let p: Vec<String> = typed("nǎr yìdiǎnr").into_iter().map(|x| x.1).collect();
    assert_eq!(p, ["naɻ", "i", "t˭iɛɻ"]);
}

#[test]
fn mandarin_timing_and_vowels() {
    let s = syls("光天你风举");
    let dur = |segs: &[chimera::phonetics::Segment]| segs.iter().map(|x| x.dur).collect::<Vec<_>>();
    // nasal codas a short murmur at the end of the rhyme (tests/mandarin_rhymes.rs)
    assert_eq!(dur(&s[0].coda), [Some(0.08)]);
    assert_eq!(dur(&s[1].coda), [Some(0.07)]);
    // nasal onsets and f longer; an unaspirated affricate's frication shorter
    assert_eq!(s[2].onset[0].dur, Some(0.09));
    assert_eq!(s[3].onset[0].dur, Some(0.13));
    assert_eq!(s[4].onset[0].dur, Some(0.08));
    // Mandarin's own vowels: eng back [ɤŋ], uo open [u̠ɔ], ou central [əʊ̠], iao one three-part nucleus
    let n: Vec<String> = syls("风多头鸟").iter().map(|s| s.nucleus.ipa.clone()).collect();
    assert_eq!(n, ["ɤ", "u̠ɔ", "əʊ̠", "iaʊ̠"]);
}

#[test]
fn every_reading_in_the_lexicon_is_a_syllable_the_front_end_sings() {
    // review of #85: 哟 唷 喲 (yo) didn't parse, so their note went missing
    use chimera::phonetics::mandarin::pinyin_syllable;
    use std::io::Read;
    let path = format!("{}/../../public/data/mandarin.txt.gz", env!("CARGO_MANIFEST_DIR"));
    let gz = std::fs::read(&path).unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).unwrap();
    let mut bad = Vec::new();
    let mut n = 0;
    for line in text.lines() {
        let (word, readings) = line.split_once('\t').unwrap();
        for py in readings.split(' ') {
            n += 1;
            if pinyin_syllable(py, word).is_none() {
                bad.push(format!("{word} {py}"));
            }
        }
    }
    assert!(n > 40_000, "{n} readings");
    assert!(bad.is_empty(), "{} of {n} readings don't parse: {:?}", bad.len(), &bad[..bad.len().min(20)]);
    assert!(pinyin_syllable("yo1", "哟").is_some());
}

#[test]
fn a_long_word_that_cant_be_split_fails_fast() {
    // review of #85: splitting backtracked exponentially ("xīnr" × 20 + "q", 81 letters, took 2 s)
    let long = "xīnr".repeat(200) + "q";
    let t = std::time::Instant::now();
    assert!(!Pinyin.claims(&long));
    assert!(!Pinyin.claims(&("anan".repeat(100) + "q")));
    assert!(t.elapsed().as_millis() < 500, "{:?}", t.elapsed());
    // and an erhua r only ends a word: xīnr is one syllable, not xi + nr
    let texts: Vec<String> = typed("xīnr").into_iter().map(|x| x.0).collect();
    assert_eq!(texts, ["xīnr"]);
}

#[test]
fn chang_after_a_degree_word() {
    // review of #115: 很长 read zhǎng, the default (the more frequent reading: 长大, 校长)
    let mut l = MandarinLexicon::new();
    l.load("长\tzhang3\n很\then3\n太\ttai4\n么\tme5\n那\tna4\n怎\tzen3\n路\tlu4\n长大\tzhang3 da4\n");
    assert_eq!(l.read("路很长"), ["lu4", "hen3", "chang2"]);
    assert_eq!(l.read("太长"), ["tai4", "chang2"]);
    assert_eq!(l.read("那么长"), ["na4", "me5", "chang2"]);
    // not after a bare 么 (怎么长: how does it grow), and a phrase still decides (很长大 is 长大)
    assert_eq!(l.read("怎么长"), ["zen3", "me5", "zhang3"]);
    assert_eq!(l.read("很长大"), ["hen2", "zhang3", "da4"]); // and 3-3 sandhi on 很
    assert_eq!(l.read("长"), ["zhang3"]);
}

#[test]
fn the_demo_song_reads_one_syllable_per_note() {
    // 送别, the Mandarin demo song, through the lexicon the page loads: a syllable to each note (a
    // missing reading would shift every later word onto the wrong note), 长亭 read cháng, not zhǎng
    use chimera::phonetics::songs::DEMO_SONGS;
    use std::io::Read;
    let path = format!("{}/../../public/data/mandarin.txt.gz", env!("CARGO_MANIFEST_DIR"));
    let mut text = String::new();
    flate2::read::GzDecoder::new(&std::fs::read(&path).unwrap()[..]).read_to_string(&mut text).unwrap();
    let mut l = MandarinLexicon::new();
    l.load(&text);
    let song = DEMO_SONGS.iter().find(|s| s.name == "送别").expect("the Mandarin demo song");
    let s = read_lyrics(song.lyrics, &[&Mandarin(&l)]);
    assert_eq!(s.len(), song.notes.len());
    assert_eq!(s.iter().filter(|x| x.melisma == Some(true)).count(), 12);
    let sung: Vec<String> = s.iter().filter(|x| x.melisma != Some(true)).map(ipa).collect();
    assert_eq!(sung[0], "tʂʰɑŋ");
    let at = |c: &str| ipa(s.iter().find(|x| x.text == c).unwrap());
    assert_eq!((at("涯"), at("角"), at("落")), ("ia".into(), "tɕ˭iaʊ̠".into(), "lu̠ɔ".into()));
}
