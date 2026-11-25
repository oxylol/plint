//! Turkish G2P: letters to sounds, ğ, long consonants, stress, numbers, phrases, and the lyrics
//! front end.
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::inventory::phoneme;
use chimera::phonetics::planner::{PhraseInput, PlanOptions, is_vowel, plan_phrases};
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::mandarin::MandarinLexicon;
use chimera::phonetics::syllables::{Lang, lyrics_in};
use chimera::phonetics::turkish::{lyrics_to_syllables, to_phrases, word};
use chimera::phonetics::{Intonation, Segment, Syllable};

/// Segments as IPA, a stress mark before the stressed vowel, ː after a long one and a space after
/// each word.
fn shown(segs: &[Segment]) -> String {
    segs.iter()
        .map(|s| format!("{}{}{}{}", if s.stress == Some(1) { "ˈ" } else { "" }, s.ipa, if s.length.is_some() { "ː" } else { "" }, if s.word_end == Some(true) { " " } else { "" }))
        .collect::<String>()
        .trim_end()
        .to_string()
}
/// A text as the CLI shows it: phrases joined by " | ".
fn tr(text: &str) -> String {
    to_phrases(text).iter().map(|p| shown(&p.segments)).collect::<Vec<_>>().join(" | ")
}
/// One word, not phrase-initial (so a capital marks a name).
fn w(typed: &str) -> String {
    shown(&word(typed, false))
}

#[test]
fn letters_map_to_sounds() {
    assert_eq!(w("çocuk"), "tʃodʒˈuk");
    assert_eq!(w("şeker"), "ʃecˈeɾ");
    assert_eq!(w("jilet"), "ʒilˈet");
    assert_eq!(w("üzüm"), "yzˈym");
    assert_eq!(w("ılık"), "ɯɫˈɯk");
    assert_eq!(w("öykü"), "øjcˈy");
}

#[test]
fn turkish_capitals_lower_to_dotless_and_dotted_i() {
    // Rust's to_lowercase turns I into i and İ into i + a combining dot
    assert_eq!(tr("IŞIK"), "ɯʃˈɯk");
    assert_eq!(tr("İNCİR"), "indʒˈiɾ");
}

#[test]
fn k_g_l_are_front_or_back_by_their_vowel() {
    assert_eq!(w("kedi"), "cedˈi");
    assert_eq!(w("kar"), "kˈaɾ");
    assert_eq!(w("göz"), "ɟˈøz");
    assert_eq!(w("gaz"), "gˈaz");
    assert_eq!(w("yol"), "jˈoɫ");
    assert_eq!(w("güzel"), "ɟyzˈel");
    // in a coda, by the vowel before
    assert_eq!(w("ekmek"), "ecmˈec");
    // â and û mark a fronted consonant before a back vowel
    assert_eq!(w("kâr"), "cˈaɾ");
    assert_eq!(w("lâle"), "lalˈe");
}

#[test]
fn soft_g_lengthens_glides_or_vanishes() {
    // before a consonant or at the end: the vowel before is one long vowel (it was two vowels, both
    // stressed: "d aˈ aˈ")
    assert_eq!(w("dağ"), "dˈaː");
    assert_eq!(w("yağmur"), "jaːmˈuɾ");
    assert_eq!(w("öğretmen"), "øːɾetmˈen");
    // between front vowels: a glide
    assert_eq!(w("değil"), "dejˈil");
    // between others: gone
    assert_eq!(w("ağaç"), "aˈatʃ");
    assert_eq!(w("soğuk"), "soˈuk");
}

#[test]
fn a_long_vowel_lasts_longer_in_the_plan() {
    let vowel = |text: &str| {
        let inputs: Vec<PhraseInput> = to_phrases(text).iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect();
        let plan = plan_phrases(&inputs, &PlanOptions { body: BodyMix::MALE, ..Default::default() });
        let a: Vec<f64> = plan.segments.iter().filter(|s| s.ipa == "a").map(|s| s.t1 - s.t0).collect();
        assert_eq!(a.len(), 1, "{text}: one vowel");
        a[0]
    };
    // the same place in the phrase and the same stress: the duration rules' length, 1.7 times over
    let (long, short) = (vowel("sağ"), vowel("sa"));
    assert!((long / short - 1.7).abs() < 1e-6, "{long} vs {short}");
}

#[test]
fn a_doubled_consonant_is_one_long_one() {
    let segs = word("dikkat", false);
    assert_eq!(shown(&segs), "dikˈat");
    let k = segs.iter().find(|s| s.ipa == "k").unwrap();
    // 1.6 × Klatt's 80 ms
    assert!((k.dur.unwrap() - 0.128).abs() < 1e-9, "{:?}", k.dur);
    assert_eq!(w("teşekkür"), "teʃecˈyɾ");
    assert!(word("anne", false).iter().find(|s| s.ipa == "n").unwrap().dur.is_some());
}

#[test]
fn stress_is_final_by_default() {
    assert_eq!(w("kitap"), "citˈap");
    assert_eq!(w("kitaplar"), "citapɫˈaɾ");
    assert_eq!(w("kitaplarım"), "citapɫaɾˈɯm");
    assert_eq!(w("geldim"), "ɟeldˈim");
}

#[test]
fn pre_stressing_suffixes_put_stress_before_them() {
    assert_eq!(w("geliyor"), "ɟelˈijoɾ");
    assert_eq!(w("okuyorum"), "okˈujoɾum");
    // the leftmost wins: the negative before -yor
    assert_eq!(w("gelmiyor"), "ɟˈelmijoɾ");
    assert_eq!(w("gelmedi"), "ɟˈelmedi");
    assert_eq!(w("yapmayacak"), "jˈapmajadʒak");
    assert_eq!(w("olmaz"), "ˈoɫmaz");
    assert_eq!(w("ederim"), "edˈeɾim");
    assert_eq!(w("gelirsiniz"), "ɟelˈiɾsiniz");
    assert_eq!(w("gelirken"), "ɟelˈiɾcen");
    assert_eq!(w("arabayla"), "aɾabˈajɫa");
    assert_eq!(w("evdeki"), "evdˈeci");
}

#[test]
fn listed_words_and_roots_keep_their_stress() {
    assert_eq!(w("şimdi"), "ʃˈimdi");
    assert_eq!(w("şimdiden"), "ʃˈimdiden");
    assert_eq!(w("nasılsın"), "nˈasɯɫsɯn");
    assert_eq!(w("türkçe"), "tˈyɾctʃe");
    assert_eq!(w("ingilizceyi"), "inɟilˈizdʒeji");
    // a derivational suffix makes a new word with final stress (the long n is one segment)
    assert_eq!(w("annelik"), "anelˈic");
    // "erken" isn't a -ken form, "namaz" isn't a negative aorist
    assert_eq!(w("erken"), "eɾcˈen");
    assert_eq!(w("namaz"), "namˈaz");
}

#[test]
fn proper_names_take_sezers_rule() {
    // an apostrophe cuts the name from its suffixes; a listed name keeps its listed stress
    assert_eq!(w("Ankara'dan"), "ˈankaɾadan");
    assert_eq!(w("İstanbul'a"), "istˈanbuɫa");
    // heavy antepenult, light penult: the antepenult
    assert_eq!(w("Erdoğan"), "ˈeɾdoan");
    assert_eq!(w("Mustafa"), "mˈustafa");
    // a heavy penult: the penult
    assert_eq!(w("Edirne'de"), "edˈiɾnede");
    // a capital at the start of a phrase doesn't mark a name
    assert_eq!(tr("Kitap"), "citˈap");
}

#[test]
fn clitics_take_no_stress() {
    assert_eq!(tr("geliyor musun"), "ɟelˈijoɾ musun");
    assert_eq!(tr("değil mi"), "dejˈil mi");
    assert_eq!(tr("sen de gel"), "sˈen de ɟˈel");
    assert_eq!(tr("ekmek ve su"), "ecmˈec ve sˈu");
    // "misafir" only starts like the particle
    assert_eq!(tr("misafir"), "misafˈiɾ");
}

#[test]
fn numbers_are_read_out() {
    let words = |t: &str| to_phrases(t).iter().flat_map(|p| p.words.clone()).collect::<Vec<_>>().join(" ");
    assert_eq!(words("1923"), "bin dokuz yüz yirmi üç");
    assert_eq!(words("100"), "yüz");
    assert_eq!(words("2000"), "iki bin");
    assert_eq!(words("1.000.000"), "bir milyon");
    assert_eq!(words("3,05"), "üç virgül sıfır beş");
    assert_eq!(words("0"), "sıfır");
}

#[test]
fn punctuation_makes_phrases_with_one_word_per_word_end() {
    let ph = to_phrases("Merhaba, nasılsın? Ankara'ya gidiyorum.");
    assert_eq!(ph.len(), 3);
    assert_eq!(ph[0].words, ["merhaba"]);
    assert_eq!(ph[2].words, ["ankara'ya", "gidiyorum"]);
    for p in &ph {
        assert_eq!(p.words.len(), p.segments.iter().filter(|s| s.word_end == Some(true)).count());
        // a question peaks before "mı" and falls: statement intonation
        assert_eq!(p.intonation, Intonation::Statement);
    }
    assert_eq!(ph[1].pause, 0.35);
    assert_eq!(ph[0].pause, 0.18);
}

#[test]
fn every_sound_is_one_the_voice_can_make() {
    let text = "Pijamalı hasta yağız şoföre çabucak güvendi. Ağaçlardaki kuşlar, ılık rüzgârda öğleden sonra ötüyor.";
    let body = BodyMix::MALE;
    let phrases = to_phrases(text);
    for s in phrases.iter().flat_map(|p| &p.segments) {
        assert!(is_vowel(&s.ipa, body) || phoneme(&s.ipa, body).is_some(), "unknown sound {}", s.ipa);
    }
    let inputs: Vec<PhraseInput> = phrases.iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect();
    let plan = plan_phrases(&inputs, &PlanOptions { body, ..Default::default() });
    assert!(!plan.gestures.is_empty());
    // every segment planned (no diphthongs in Turkish, so one planned segment each)
    assert_eq!(plan.segments.len(), inputs.iter().map(|p| p.segments.len()).sum::<usize>());
}

// ---------------------------------------------------------------- singing

fn sung(text: &str) -> Vec<String> {
    lyrics_to_syllables(text).iter().map(|s| s.text.clone()).collect()
}
fn parts(s: &Syllable) -> String {
    let ipa = |v: &[Segment]| v.iter().map(|x| x.ipa.as_str()).collect::<String>();
    format!("{}.{}.{}", ipa(&s.onset), s.nucleus.ipa, ipa(&s.coda))
}

#[test]
fn lyrics_split_where_turkish_spelling_does() {
    assert_eq!(sung("Yeni bağlarında"), ["Ye", "ni", "bağ", "la", "rın", "da"]);
    let syl = lyrics_to_syllables("bağlarında dikkat değil dağ");
    let p: Vec<String> = syl.iter().map(parts).collect();
    // ğ before a consonant: one note's vowel, nothing after it
    assert_eq!(p[0], "b.a.");
    assert_eq!(&p[1..4], ["ɫ.a.", "ɾ.ɯ.n", "d.a."]);
    // a long consonant closes one sung syllable and opens the next, as it is written
    assert_eq!(&p[4..6], ["d.i.k", "k.a.t"]);
    // ğ between front vowels: the glide opens its syllable
    assert_eq!(&p[6..8], ["d.e.", "j.i.l"]);
    assert_eq!(p[8], "d.a.");
    // sung lengths come from the notes
    assert!(syl.iter().all(|s| s.onset.iter().chain(&s.coda).all(|c| c.dur.is_none())));
    assert_eq!(syl.iter().filter(|s| s.word_end).count(), 4);
}

#[test]
fn held_notes_and_phrases_in_lyrics() {
    let syl = lyrics_to_syllables("yürü- - yorum, gel");
    assert_eq!(syl.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["yü", "rü", "~", "yo", "rum", "gel"]);
    assert_eq!(syl[2].melisma, Some(true));
    assert_eq!(syl[2].nucleus.ipa, "y");
    assert!(syl[4].word_end && syl[4].phrase_end);
}

#[test]
fn latin_script_is_turkish_only_when_chosen() {
    let no_dict = |_: &str, _: bool| Vec::<Segment>::new();
    let text = |lang| lyrics_in("çok güzel", &no_dict, &MandarinLexicon::new(), lang).into_iter().map(|s| s.text).collect::<Vec<_>>();
    assert_eq!(text(Lang::Tr), ["çok", "gü", "zel"]);
    // English claims only ASCII letters, as before: ç and ü are skipped
    assert!(text(Lang::En).iter().all(|t| t.is_ascii()));
    // other scripts still go to their own front ends
    let mixed = lyrics_to_syllables("gel 빵야 дом");
    assert_eq!(mixed.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["gel", "빵", "야", "дом"]);
}

#[test]
fn a_capital_starting_a_lyric_phrase_is_not_a_name() {
    let stressed = |text: &str| lyrics_to_syllables(text).into_iter().filter(|s| s.nucleus.stress == Some(1)).map(|s| s.text).collect::<Vec<_>>();
    // a name takes Sezer's rule (the penult); a word starting the phrase keeps final stress
    assert_eq!(stressed("dün Ali geldi"), ["dün", "A", "di"]);
    assert_eq!(stressed("Ali geldi"), ["li", "di"]);
    assert_eq!(stressed("dün, Ali geldi"), ["dün", "li", "di"]);
}

#[test]
fn turkish_lyrics_plan_as_a_song() {
    let syl = lyrics_to_syllables("Bugün hava çok güzel.");
    let notes: Vec<SungNote> = syl.iter().enumerate().map(|(i, s)| SungNote { midi: 60.0 + i as f64, beats: 1.0, syl: s.clone() }).collect();
    let bpm = 90.0;
    let plan = plan_song(&notes, &SongOptions { body: BodyMix::FEMALE, bpm, t0: 0.35, vibrato: 35.0 });
    assert!(plan.end > 0.35 + notes.len() as f64 * 60.0 / bpm - 0.2, "{}", plan.end);
    assert!(plan.segments.iter().any(|s| s.ipa == "y"));
}
