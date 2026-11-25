//! Russian spelling → segments (`phonetics::russian`), through speech and through sung lyrics.
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::phonetics::russian;
use chimera::phonetics::songs::DEMO_SONGS;
use chimera::phonetics::syllables::lyrics_to_syllables;

fn ipa(segs: &[chimera::phonetics::Segment]) -> String {
    segs.iter().map(|s| s.ipa.as_str()).collect::<Vec<_>>().join(" ")
}
/// as sung: no vowel reduction
fn sung(w: &str) -> String {
    ipa(&russian::word(w, false))
}
/// as spoken: unstressed vowels reduce
fn said(w: &str) -> String {
    ipa(&russian::word(w, true))
}
/// a spoken phrase of Russian words, each read with the word after it (Speak has no Russian yet:
/// it comes back through the shared speech path, so the speech rules are tested on the reader)
fn spoken(text: &str) -> Vec<chimera::phonetics::Segment> {
    let words: Vec<&str> = text.split_whitespace().collect();
    words.iter().enumerate().flat_map(|(i, w)| russian::word_before(w, words.get(i + 1).copied(), true)).collect()
}
fn phrase(text: &str) -> String {
    ipa(&spoken(text))
}

#[test]
fn vowels_and_consonants() {
    assert_eq!(sung("аоуыэи"), "a o u ɨ e i");
    assert_eq!(sung("шар"), "ʂ a ɾ");
    assert_eq!(sung("хор"), "x o ɾ");
    assert_eq!(sung("цирк"), "ts ɨ ɾ k");
    assert_eq!(sung("жир"), "ʐ ɨ ɾ");
    // р is a tap: the trill closed the voice off for ~100 ms and "рыба" was heard as "дыба"
    assert_eq!(sung("рыба"), "ɾ ɨ b a");
}

#[test]
fn iotation_and_softness() {
    assert_eq!(sung("ёж"), "j o ʂ");
    assert_eq!(sung("маяк"), "m a j a k");
    assert_eq!(sung("съезд"), "s j e s t");
    // a soft consonant is the plain one and a short glide; л also changes from dark to clear
    assert_eq!(sung("лук"), "ɫ u k");
    assert_eq!(sung("люк"), "l j u k");
    let soft = russian::word("люк", false);
    assert_eq!(soft[1].dur, Some(0.025));
    // ь before и and о: the vowel gets its own /j/ (it was dropped)
    assert_eq!(sung("чьи"), "tʃ j j i");
    assert_eq!(sung("бульо́н"), "b u l j j o n");
}

#[test]
fn stress_marks_and_the_fallbacks() {
    let s = russian::word("молоко́", true);
    assert_eq!(ipa(&s), "m ɐ ɫ ɐ k o");
    assert_eq!(s.iter().filter_map(|s| s.stress).collect::<Vec<_>>(), [0, 0, 1]);
    // ё is stressed
    assert_eq!(said("поёт"), "p ɐ j o t");
    // a frequent word from the table, then the first vowel
    assert_eq!(said("тебя"), "t j ɪ b j a");
    assert_eq!(said("мама"), "m a m ɐ");
    // singing keeps full vowels
    assert_eq!(sung("молоко́"), "m o ɫ o k o");
}

#[test]
fn unstressed_ya_and_a_after_ch() {
    // я was /ɐ/ everywhere; before the stress it is /ɪ/
    assert_eq!(said("язы́к"), "j ɪ z ɨ k");
    assert_eq!(said("пяти́"), "p j ɪ t j i");
    assert_eq!(said("часы́"), "tʃ j ɪ s ɨ");
    // but word-final я and а stay open
    assert_eq!(said("неде́ля"), "n j ɪ d j e l j ɐ");
    assert_eq!(said("ту́ча"), "t u tʃ j ɐ");
}

#[test]
fn devoicing_and_assimilation_inside_a_word() {
    assert_eq!(sung("дуб"), "d u p");
    assert_eq!(sung("лодка"), "ɫ o t k a");
    assert_eq!(sung("сдать"), "z d a t j");
    assert_eq!(sung("вторник"), "f t o ɾ n j i k");
    // в doesn't voice what's before it
    assert_eq!(sung("твой"), "t v o j");
    // through the softness glide
    assert_eq!(sung("ло́шадь"), "ɫ o ʂ a t j");
}

#[test]
fn a_preposition_is_part_of_the_next_word() {
    // each was devoiced as if it ended a phrase: "нат лесом", "ф лесу", "ис дома"
    assert_eq!(phrase("над ле́сом"), "n ɐ d l j e s ɐ m");
    assert_eq!(phrase("в лесу́"), "v l j ɪ s u");
    assert_eq!(phrase("из до́ма"), "i z d o m ɐ");
    // and it assimilates to an obstruent
    assert_eq!(phrase("в ти́хом"), "f t j i x ɐ m");
    assert_eq!(phrase("с горы́"), "z g ɐ ɾ ɨ");
    assert_eq!(phrase("из-под стола́"), "i s p ɐ t s t ɐ ɫ a");
    // sung words come one at a time: the preposition keeps its spelling's voicing
    assert_eq!(sung("над"), "n a d");
    // a word that isn't a preposition still devoices
    assert_eq!(phrase("са́д мо́й"), "s a t m o j");
}

#[test]
fn g_in_the_genitive_ending_is_v() {
    assert_eq!(said("его́"), "j ɪ v o");
    assert_eq!(said("сего́дня"), "s j ɪ v o d n j ɐ");
    assert_eq!(said("кра́сного"), "k ɾ a s n ɐ v ɐ");
    assert_eq!(said("ничего́"), "n j i tʃ j ɪ v o");
    assert_eq!(said("мно́го"), "m n o g ɐ");
}

#[test]
fn spelling_conventions() {
    assert_eq!(sung("что"), "ʂ t o");
    assert_eq!(sung("коне́чно"), "k o n j e ʂ n o");
    assert_eq!(sung("сча́стье"), "ʃ j a s t j j e");
    assert_eq!(sung("мужчи́на"), "m u ʃ j i n a");
    assert_eq!(sung("по́здно"), "p o z n o");
    assert_eq!(sung("со́лнце"), "s o n ts e");
    assert_eq!(sung("чу́вство"), "tʃ j u s t v o");
    assert_eq!(said("смеётся"), "s m j ɪ j o ts ɐ");
    assert_eq!(sung("купа́ться"), "k u p a ts a");
}

#[test]
fn the_review_s_small_gaps() {
    // silent д in рдц, т in стл (they were said)
    assert_eq!(sung("се́рдце"), "s j e ɾ ts e");
    assert_eq!(sung("счастли́вый"), "ʃ j a s l j i v ɨ j");
    // unstressed э reduces like е (it stayed /e/)
    assert_eq!(said("этажа́"), "ɪ t ɐ ʐ a");
    assert_eq!(said("э́то"), "e t ɐ");
    // the vowel of -тся, -ться gets a Russian length, not English's 40-50 ms
    let s = spoken("смеётся");
    assert_eq!(s.last().unwrap().dur, Some(0.12));
    let s = spoken("смеётся он");
    assert_eq!((s[7].ipa.as_str(), s[7].dur), ("ɐ", Some(0.09)));
}

#[test]
fn a_doubled_letter_is_one_consonant() {
    // long between vowels, plain before a consonant (it was two separate closures)
    let vanna = russian::word("ва́нна", false);
    assert_eq!(ipa(&vanna), "v a n a");
    assert!(vanna[2].dur.is_some_and(|d| d > 0.1), "{:?}", vanna[2].dur);
    let russkiy = russian::word("ру́сский", false);
    assert_eq!(ipa(&russkiy), "ɾ u s k j i j");
    assert_eq!(russkiy[2].dur, None);
}

#[test]
fn the_reader_s_speech_plans() {
    for text in ["Приве́т", "Ты меня́ слы́шишь"] {
        let plan = plan_utterance(&spoken(text), &PlanOptions { body: BodyMix::MALE, ..Default::default() });
        assert!(plan.end.is_finite() && plan.end > 0.0);
        assert!(!plan.gestures.is_empty());
    }
}

#[test]
fn english_speech_is_untouched() {
    // Speak's tokenizer kept U+0301 in every word for Russian, so "cafe" + U+0301 (as pasted
    // text sometimes arrives) read as one word, "k eɪ f ɛ", where main read "cafe"
    let g = EnglishG2P::new();
    assert_eq!(g.to_phrases("cafe\u{301}")[0].segments, g.to_phrases("cafe")[0].segments);
}

#[test]
fn lyrics_split_into_syllables() {
    let g = EnglishG2P::new();
    let s = lyrics_to_syllables("мо-ло-ко́ ~", &g);
    assert_eq!(s.len(), 4);
    assert_eq!(s[3].melisma, Some(true));
    let texts = |t: &str| lyrics_to_syllables(t, &g).iter().map(|s| s.text.clone()).collect::<Vec<_>>();
    assert_eq!(texts("мая́к"), ["ма", "я\u{301}к"]);
    assert_eq!(russian::split_spelling("поёт", 2), ["по", "ёт"]);
}

#[test]
fn sung_syllables_are_open() {
    // Russian singing gives the next syllable the whole cluster, so the vowel holds the note: only
    // the onsets пр, стр were allowed, and я́блоко sang as я́б-ло-ко, Москва́ put /sk/ on the first
    // note, ло́дка sang as ло́д-ка
    let g = EnglishG2P::new();
    let onsets = |t: &str| lyrics_to_syllables(t, &g).iter().map(|s| ipa(&s.onset)).collect::<Vec<_>>();
    assert_eq!(onsets("ло́дка"), ["ɫ", "t k"]);
    assert_eq!(onsets("я́блоко"), ["j", "b ɫ", "k"]);
    assert_eq!(onsets("Москва́"), ["m", "s k v"]);
    assert_eq!(onsets("сестра́"), ["s j", "s t ɾ"]);
    // a leading sonorant (р л н м й) stays with the vowel before it
    assert_eq!(onsets("се́рдце"), ["s j", "ts"]);
    assert_eq!(onsets("войска́"), ["v", "s k"]);
    assert_eq!(onsets("темно́"), ["t j", "n"]);
    // a soft consonant's glide, and the /j/ after ь, go with it
    assert_eq!(onsets("семья́"), ["s j", "m j j"]);
    // the spelling on the notes splits the same way
    let texts = lyrics_to_syllables("я́блоко", &g).iter().map(|s| s.text.clone()).collect::<Vec<_>>();
    assert_eq!(texts, ["я\u{301}", "бло", "ко"]);
    // a sonorant left on the note before is written there too
    let texts = |t: &str| lyrics_to_syllables(t, &g).iter().map(|s| s.text.clone()).collect::<Vec<_>>();
    assert_eq!(texts("войска́"), ["вой", "ска\u{301}"]);
    assert_eq!(texts("се́рдце"), ["се\u{301}р", "дце"]);
    assert_eq!(texts("бо́льно"), ["бо\u{301}ль", "но"]);
    assert_eq!(texts("Москва́"), ["Мо", "сква\u{301}"]);
}

#[test]
fn the_demo_song_has_a_note_per_syllable() {
    let g = EnglishG2P::new();
    let song = DEMO_SONGS.iter().find(|s| russian::contains_russian(s.lyrics)).expect("a Russian demo song");
    assert_eq!(lyrics_to_syllables(song.lyrics, &g).len(), song.notes.len());
}

#[test]
fn a_sung_preposition_without_a_vowel_joins_the_next_word() {
    // в, с, к have no vowel, so no note: they were silently dropped from the song
    let g = EnglishG2P::new();
    let s = lyrics_to_syllables("в лесу́", &g);
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].text, "в ле");
    assert_eq!(ipa(&s[0].onset), "v l j");
    // and it takes the next word's voicing
    assert_eq!(ipa(&lyrics_to_syllables("к тебе́", &g)[0].onset), "k t j");
    assert_eq!(ipa(&lyrics_to_syllables("с горы́", &g)[0].onset), "z g");
    assert_eq!(ipa(&lyrics_to_syllables("в ти́хом", &g)[0].onset), "f t j");
}

#[test]
fn unstressed_vowels_keep_a_syllable_s_length() {
    // the duration rules are English's and have no /ɐ/: unstressed vowels came out 40-50 ms,
    // "Доброе утро" took 0.75 s for five syllables, and Whisper lost whole words
    let p = spoken("До́брое у́тро");
    let weak: Vec<f64> = p.iter().filter(|s| s.stress == Some(0)).filter_map(|s| s.dur).collect();
    assert_eq!(weak.len(), 3);
    assert!(weak.iter().all(|d| *d >= 0.07), "{weak:?}");
    // the last one, before a pause, is the longest
    assert!(weak[2] > weak[0]);
}

#[test]
fn a_stressed_y_is_as_long_as_the_other_stressed_vowels() {
    // the duration table's /ɨ/ is English's reduced vowel (110 ms inherent): "дым" came out with
    // half the vowel of "дум" and Whisper wrote "дум" for it
    let y = |text: &str| spoken(text).iter().find(|s| s.ipa == "ɨ").unwrap().dur;
    assert_eq!(y("ды́м"), Some(0.19));
    assert_eq!(y("ды́м и до́м"), Some(0.12));
    // sung, the note sets the length
    assert!(russian::word("дым", false).iter().all(|s| s.ipa != "ɨ" || s.dur.is_none()));
}

#[test]
fn russian_is_a_front_end_on_the_lyrics_router() {
    let g = EnglishG2P::new();
    // one line, three front ends: each word goes to its own
    let s = lyrics_to_syllables("twinkle ми́р 별", &g);
    assert_eq!(s.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["twin", "kle", "ми́р", "별"]);
    assert!(s.iter().filter(|s| s.word_end).count() == 3);
    // a preposition with no vowel leans only on a Russian word right after it in the phrase
    assert_eq!(lyrics_to_syllables("в лесу́", &g)[0].text, "в ле");
    assert_eq!(lyrics_to_syllables("в, лесу́", &g)[0].text, "ле");
    let s = lyrics_to_syllables("в star ми́р", &g);
    assert_eq!(s.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), ["star", "ми́р"]);
}

#[test]
fn a_lone_stress_mark_is_not_a_russian_word() {
    // Russian claimed a word that starts with U+0301, so ni + U+0301 (pinyin with a separate
    // tone mark) would lose its tone to Russian once Mandarin is on the router
    use chimera::phonetics::syllables::FrontEnd;
    assert!(!russian::Russian.starts('\u{301}'));
    assert!(russian::Russian.continues('\u{301}'));
    let g = EnglishG2P::new();
    assert_eq!(lyrics_to_syllables("молоко́", &g).len(), 3);
}
