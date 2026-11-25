# Language front ends

CHIMERA sings other languages in two halves, the usual split in speech synthesis:

```
lyrics ──► router ──► FRONT END ──► syllables ──► BACK END ──► sound
           (by script)  English: CMUdict         (IPA, with what     the planner, the airway,
                        Korean: Hangul rules      the language        the voice: the same for
                        Japanese: kana rules      decides: aspiration, every language
                        Mandarin: pinyin lexicon  consonant lengths)
                        Pinyin: typed with tones
                        Russian: spelling rules
                        explicit syllables
```

**A front end is phonology; the back end is physics.** A front end decides what a word's sounds
are: which phonemes, whether and how long a stop aspirates, how a tone bends the pitch. The back
end realises whatever the syllables say and knows no language's rules. When a front end says
nothing, the planner keeps English's defaults, which is why English output doesn't change.

This is the shared design for languages (#40), and the first steps of #32 live in it: the sounds
Korean and Mandarin need (`ɕ ʑ tɕ dʑ tʂ dʐ ɥ`, step 3), aspiration as the front end's call
(step 1, with #73), and pitch bends for tones (step 4).

## The built-in front ends

| Script | Front end | Needs | Where |
|---|---|---|---|
| Latin | English | CMUdict (letter-to-sound until it loads) | `phonetics/english.rs`, `syllables.rs` |
| Hangul | Korean | nothing: a block's letters are its sounds | `phonetics/korean.rs` |
| Cyrillic | Russian | nothing: spelling rules, stress marked with an acute ([russian.md](russian.md)) | `phonetics/russian.rs` |
| Kana | Japanese | nothing: kana are their sounds; no kanji ([Japanese](#japanese), below) | `phonetics/japanese.rs` |
| Han | Mandarin | `public/data/mandarin.txt.gz` (208 KB, fetched when lyrics have Han characters) | `phonetics/mandarin.rs` |
| Latin with tones | Mandarin as pinyin | nothing (checked against Mandarin's 411 syllables) | `phonetics/mandarin.rs` (`Pinyin`) |
| Latin, when chosen | Turkish | nothing: spelling rules, stress by rule ([turkish.md](turkish.md)) | `phonetics/turkish.rs` |

The router (`read_lyrics` in `syllables.rs`) sends each word to the front end whose script it is
in, so a line can mix them (`我爱你 hello 빵야 nǐhǎo`). It also handles what every language
shares: melisma marks (`-`, `~`), phrase punctuation (ASCII and full-width `，。！？`) and word ends.

Script can't tell Latin-script languages apart, so the player chooses which front end reads Latin
script: English unless Turkish is chosen (`Lang` in `syllables.rs`; EN/TR in Speak and Sing).
Toned pinyin and the other scripts are routed as before whichever is chosen.

Pinyin and English share the Latin alphabet, so a front end may **decline** a word it scanned
(`FrontEnd::claims`). The others then read that word as they would without it, and the one that
declined can't claim a piece of it: `café` stays English's "caf" rather than "caf" and a pinyin
"é". Characters no front end claims are skipped, as on main: `Zürich` sings "Z rich". Nothing after
them is lost.

`chimera syllables "<lyrics>" [en|tr]` prints what the front ends read, one JSON syllable per sung
note: the quickest way to check a front end.

## Korean

One Hangul block is one note, and the block keeps its spelling as the note's text. Inside a word
the front end applies standard Seoul sound rules: a coda moves onto a following silent ㅇ (먹어 →
머거), codas neutralise, obstruents nasalise before nasals (국물 → 궁물), ㄴ and ㄹ meet as [ll],
ㅎ aspirates its neighbour (좋다 → 조타), and obstruents tense after an obstruent coda.

A block typed as its letters (conjoining jamo, as decomposed text has them) is joined into the
block first. The letters modern Korean dropped (ㆍ ㅿ ㆁ ㆆ) make no block, so a block with one is
skipped like any script no front end reads (#132).

Korean's three stops are all voiceless at the start of a word. The front end tells them apart by
how long they aspirate (`vot`, voice onset time):

| | 불 fire | 풀 grass | 뿔 horn |
|---|---|---|---|
| kind | lax | aspirated | tense |
| written | [pʰ], `vot` 45 ms | [pʰ], `vot` 90 ms | [p˭], unaspirated |
| measured in the render (male; female within 3 ms) | 53 ms | 96 ms | 22 ms |

Between voiced sounds a lax stop is voiced (아기 [agi]). Not modelled yet: the lower pitch that
follows a lax stop (Seoul speakers rely on it as much as on aspiration; a start bend could carry
it, untried) and the stiff voice of tense stops. A Korean speaker's ear is still wanted.

## Japanese

Hiragana, katakana (and half-width katakana) are read as they are written, with no dictionary:
**kanji are not claimed.** A Han character goes to the Mandarin front end (a Chinese reading, or
nothing when its lexicon isn't loaded) and the kana around it are sung, so type Japanese lyrics in
kana, as most singing editors expect. Kana need no choice in the EN/TR setting (`Lang` only says
what Latin script is); which language a Han text is waits for more language choices ("Not yet").

**One mora is one note**, the convention of Japanese singing, so a note is the unit the song counts
in and a lyric sets as many notes as it has morae:

| Written | Sung as | Notes |
|---|---|---|
| きゃ しゃ ふぁ てぃ (a kana and its small kana) | one mora: き+ゃ is [kʲʰja] (the consonant palatalised), し+ゃ [ɕa] (it carries its own palatal), ふ+ぁ [ɸa] | 1 |
| ー 〜 (also おお ああ きい, おう) | the vowel before it, held on a **melisma note** that continues it at the new pitch: とうきょう is to-(o) kyo-(o) | 1 each |
| ん | a **hum**: the note's nucleus is the nasal of the sound after it, held for the whole note, and the note before closes into it (below) | 1 |
| っ | the vowel before goes on into its note, which ends in the **long consonant** after it (below) | 1 |
| a hyphen: おも-う | the kana after it starts a note of its own, read as it is written (the う of 思う is a syllable, not a long o) | |

A vowel kana that repeats the vowel before it, or う after an o, is read as a long vowel: right for
学校 がっこう and とうきょう, wrong for the う that is a syllable of its own (思う); the hyphen is
the way to say so. い after an e is sung as it is written (先生 せんせい is se-n-se-i): singers
sing the い, and a listener called the long e wrong.

When a melody has more or fewer notes than the lyric has morae, nothing special happens in the
front end: the app fits a lyric to a tune's phrases (`src/song/fit.ts`), splitting the longest note
when there are too few or holding a syllable's vowel across several when there are
too many. To hold a mora where the tune does, type ー (or the router's `~`, spaced) after it.

- **ん** is a hum on [m] before p b m, [n] before t d n ts z ɾ tɕ dʑ ɲ, and [ŋ] before k g, a
  vowel, a glide, a fricative and at the end of a word (one of the three nasals the planner closes
  a rhyme onto), the note's nucleus for the whole note: with っ's silent [ʔ] (below), one of the
  two places this front end writes a consonant as a nucleus. The planner times and voices a nasal over the note's length like a
  vowel, and the rest (the app's fitting, the live keys) only copy a nucleus. The note before
  closes into it the way a Mandarin rhyme closes onto its coda (`closing`): a short coda of the same
  nasal, the vowel nasalised over its last 60 %, so かん on two notes is [kãŋ] [ŋ], one movement.
  A second ん goes on with the hum. A listener compared this with the vowel going on into ん's note
  and closing onto the nasal at its end, next to an ACE Studio singer (whose ん is a hum from the
  beat), and chose the hum on long notes and short ones; the other way, かん on two beats was
  heard as "ka...n". At 0.6 s a mora the hum had been heard as two sounds too: tempo decides. The
  recogniser reads both ways alike.
- **っ** is a mora, so a note: the vowel before goes on into it (a melisma note), and the consonant
  after it is held 250 ms longer than a single one, out of that note (the planner shortens it when
  the note is shorter). As no note, with the consonant twice its length (150 ms of closure), it was
  not heard as a geminate inside a sung note: a listener called it wrong and the recogniser read
  がっこう as 過去. Before a vowel, a glide, a tap, ん or the end of a word (あっ!) a glottal stop
  cuts the note before and is held, silent, through っ's own note as its nucleus; at the start of a word there is
  no note to go on from and it is dropped.
- **Consonants:** し [ɕ], ち [tɕ], つ [ts], ふ [ɸ], ひ [ç], じ ぢ [dʑ], ず づ [z], に [ɲ], the ら行 a
  tap [ɾ]. The ざ行 is the affricate [dz] at the start of a word and after ん, [z] between vowels.
  **わ is [w]**: it is written [ɰ] (the lips compressed, not rounded), but the inventory's [ɰ] is
  the tongue alone, and a listener heard it as ら. **う is the unrounded [ɯ]**, a vowel the
  inventory already has (Korean's ㅡ). A yōon's consonant is palatalised (きゃ [kʲja]), except
  two: ひゃ is [hʲja], h at the palate with its glide, because as [ça] the fricative was 36 dB under
  the vowel at the start of a phrase (an ACE Studio singer's is 11-17); and ぎゃ's g is not
  palatalised and has a 30 ms glide, because palatalised, the closure let the voice through 15-20
  dB under the vowel (が's is silent). Both read better to the recogniser (ぎゅうにゅう ひゃくえん as
  牛乳 00円, before ぎゅ乳 0円), but a listener still hears the ぎ and the ひ missing, at the start
  of a phrase: the engine's h and ç there are 10-20 dB under ACE's, and a phrase-initial voiced
  stop murmurs through its closure. Neither is the front end's to fix. Tried and not the cause:
  the oral tube's first section dropping the frication source (#124's fix moves these onsets by
  under 1 dB), a narrower or wider [ç] (0.06-0.3 cm²: within 4 dB), and the sung breath at full drive 60 ms earlier (the
  ひ comes up 20 dB, but a vowel is heard before it).

A listener who knows Japanese (not a native speaker) heard five versions and the points above
marked "a listener" are theirs; the last two were heard next to ACE Studio's Japanese singers on
the same notes. Listen at a natural tempo: at 0.6 s a mora every sound is drawn
out and nothing sounds right, at 0.3 s the recogniser reads whole lines. Known and not fixed here: a k (and a つ) at the start of a
phrase has almost no burst, so きょう can lose its k; it is the engine's stop, the same in every
language. Next to ACE's singers the listener rated the long vowels, っ, ん, こんにちは and the
lines with し ち つ ふ and the tap close; the yōon above are not.

## Mandarin

Han characters are read through a lexicon of character and phrase readings (MIT-licensed pinyin
data, `tools/build-mandarin.ts`), longest phrase first: 银行 is yín háng, 行走 xíng zǒu. Then:

- **Readings by context:** a character outside any phrase takes its first listed reading, except
  where its neighbour decides: 长 after a degree word is the adjective (很长 hěn cháng, 太长 tài
  cháng), while 长大 and 校长 keep zhǎng.
- **Tone changes:** third-tone sandhi inside a phrase (你好 reads ní hǎo), and 不 and 一 before a
  fourth tone (不会 bú huì, 一个 yí gè).
- **Aspiration said:** p t k q ch c are aspirated with Mandarin's longer aspiration (`vot` 90 ms),
  b d g j zh z voiceless and unaspirated, not voiced stand-ins.
- **Tones are read and not sung:** a song follows its melody. The reading keeps its tones (they
  decide the sandhi above and the neutral tone's reduced vowel), and no syllable bends the pitch
  for them. A native listener compared a nursery song and a greeting sung as written and with
  each tone drawn on the note as a contour, at half and at full depth, and chose the melody as
  written wherever there was one: "flat sounds sung, the full tones sound spoken, half sounds
  odd". Drawing them is for speech (#48).
- **Medials inside the vowel:** 话 is the rising diphthong `u̠a` and 鸟 the three-part `iaʊ̠`: the
  medial is a 50 ms glide that ends on the beat, after the initial consonant.
- **Mandarin's own vowels:** the apical vowels ɿ (四) and ʅ (是), the back u `u̠` (乌, and the medial
  of 话 朵 光) and ong's `ʊ̠` (东, and the off-glide of ao ou), er [əɚ] (二 [aɚ]), eng [ɤŋ], uo [u̠ɔ],
  ou [əʊ̠].
- **Rhymes close in one movement:** the off-glide of ai ei ao ou, and -n / -ng, are where the vowel
  ends up at the end of the note, not a part held after it (`closing`, below).
- **Timing:** j zh z with short frication, n m f long enough to hear, -n 70 ms and -ng 80 ms.
- **The voice on the beat:** each initial says how long the voice lags it (`voiceLag`: the 90 ms
  aspiration, a 15-18 ms burst, a fricative's 27-34 ms release), and the song planner sings the
  initial that much earlier. Against the 20 singers of M4Singer (Zhang et al., NeurIPS 2022
  Datasets and Benchmarks; the synth singing their notes, 296 sentences) the voice came in 110 ms
  after the beat behind an aspirated initial and 34-42 ms behind the others, where theirs is on
  it; now 14-18 ms, `tests/voice_on_beat.rs`.
- **Consonants shorten with the notes:** what comes before a sung vowel (the coda before it, the
  initial and medial, the voice's lag) takes at most 55 % of the time since the vowel before
  (`onsetShare`) and shortens in proportion on a faster note. M4Singer's singers' initials are a
  share of that time at any tempo (the median 0.39-0.43, three quarters under 0.50-0.55; Zhang et
  al., "M4Singer: A Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing
  Corpus", NeurIPS 2022 Datasets and Benchmarks: only aggregate figures like these come from it).
  At fixed lengths a fast note had no vowel left: 85 % of the vowels under 150 ms after an
  aspirated initial were 12 dB or more under the singer's on the same 296 sentences; now none.
  A song's first syllable shortens the same way to fit the time before its first beat, so the
  plan never starts before its time 0 (窗 chuang from a beat at 0.3 s began 13 ms before it).
- **Erhua:** 一点儿 sings 一 + 点儿, one syllable ending in ɻ; a 儿 that is a word's own syllable
  (女儿, 儿子) stays one.
- **A vowel-initial syllable** (爱, 安) starts with a soft glottal catch, so it doesn't run into the
  vowel before.

**Pinyin** with tones is Mandarin too: tone marks (`nǐhǎo`, and `xièxie` with a neutral last
syllable) or numbers (`ni3 hao3`, `de5`); apostrophes split syllables (`Xī'ān`), and `ü` or `v` both
work. Third-tone sandhi applies inside a word. Pinyin claims only a word written as pinyin is:
- every syllable real, and a vowel-initial one inside the word after an apostrophe (`Xī'ān`);
- the first syllable's tone written; an unwritten one later is neutral, at the end (`dōngxi`) or
  inside as a syllable that is often neutral: a particle or suffix (`zěnmeyàng`, `háizimen`), bu or
  yi (`duìbuqǐ`), a directional complement, or a word's neutral second half (`péngyoumen`);
- two letters or more (a one-letter word, `a1` or `à`, is English; write 啊 in characters).

So plain Latin words stay English, and so do accented English words (`café`, `résumé`, `touché`,
`née`, `Pokémon`). The one left: `élan` is also real pinyin (é + a neutral lan), so type it
unaccented.

What each choice measured, on the scramble test (#80) where it was tuned:

| Choice | Confusion it fixed | Scrambled syllable errors |
|---|---|---|
| unaspirated stops said (not voiced stand-ins) | | 54.8% → 48.6% |
| tones as bends (vs none) | | 52.8% → 48.6% |
| medials inside the vowel | 话 drifting toward 哈 | 48.6% → 44.2% |
| apical ɿ ʅ (vs [ɨ]) | 是 heard as 虚 | 66.3% → 46.2% (apical-dense lines) |
| nasal codas held, onset timing, eng [ɤŋ], uo [uɔ], ou [əʊ], stronger tones | named confusions each | 43.9% → 35.8% |
| -ng approach after ɑ | uang heard as uan ×16 → ×6 | 35.8% → 34.5% |
| off-glide over 35% | ai/ou heard as a/o | 29.5% → 26.6% |

The last row is measured with #80's corrected judge (this branch, 8 scrambles × 2 voices); the
others on #34 before that fix, so compare within a row. Now: **25.9% / 27.2%** scrambled (male /
female) and 10% / 15% on the real poems. What's left: tone 3 (about half wrong, mostly heard as 4;
a dip in the middle or a 214 measured no better), the sibilants (c heard as ch) and in/ing.

These numbers were measured before the changes in "By a native speaker's ear" below, with the
tones drawn as held bends and the endings held. As it ships, with no tones drawn and the rhymes
closing in one movement: **37.8% / 39.7%** scrambled (male / female) and 12.5% / 15.0% on the
real poems (8 scrambles × 2 voices, on Windows). On the same machine #85's head reads 25.0% /
27.8% and 7.5% / 15.0%, and Korean and English read the same on both builds to the digit
(52.8 / 53.4% and 68.0 / 73.5% scrambled; their 36 renders are byte-identical), within 2.2
points of the table below. So the recogniser scores this 12 points worse than the version the
native listener called "a foreigner speaking Chinese"; the listener chose this one. Of the
scrambled syllables it heard, the wrong initials went from 19% to 29% (s as sh 13 of 15 times,
as before; sh as f, ch as h), the wrong finals from 15% to 23% (uang as ang or uan, i as e, eng
as ong), the wrong tones from 34% to 48%. Which of the changes cost the initials is not known.
It listens at 16 kHz, so it hears nothing of a sibilant above 8 kHz.

### By a native speaker's ear

A native Mandarin speaker listened to the front end above, and again after each round of changes.
Their reports, in order:

1. The singing sounds like a foreigner speaking Chinese.
2. After the first four rows below: a little better, still a foreigner, heard at once.
3. Every compound and nasal rhyme is two sounds where Mandarin has one, "ang comes out as an +
   ng", and they are all drawn out.
4. After that row: much better, it is one sound now; but single syllables all sound like a second
   tone.

The first report named no sound, so the renders were given to language models to say what they
heard ("model" below), and what they named was measured. Everything was measured in
the render (level, LPC formants, the articulator tracks, the pitch) and fixed in the front end or
in what a front end can ask of the singer. English, Korean and Russian plan exactly as before (the
goldens are untouched). One thing is not opt-in: any rising nucleus sings its glide before the
beat; only the Mandarin front end writes one.

| Reported | By | Measured | Now | Test |
|---|---|---|---|---|
| 两 as *li-ang*, 快 as *ku-ai* | model | the medial sat on the beat for 80-100 ms; the main vowel 90% reached 200 ms after it | a 50 ms glide ending on the beat; 80-90 ms | `medials.rs` |
| the tones laid over the tune | model, then the native speaker's choice | every 2nd and 3rd tone scooped up from 3.5-4 semitones below, every 4th dropped 4 at its end | no tone is drawn: a song follows its melody | `mandarin.rs` |
| 二, 耳 as an American "er" | model | /ɝ/ held still | a central vowel moving into the r-colouring, 二 more open | `er.rs` |
| 只 as zhu, zhe or zhi | model | ɿ ʅ fitted to formants alone: lips narrowed to 0.2, no tongue tip | the sibilant's tip held open, lips spread, by the calibrator | `apical.rs` |
| still a foreigner (report 2) | native speaker | the largest distance from Beijing speakers' vowels: u and ong's F2 997 and 1126 Hz (men) against 661 and 730 (Zee & Lee 2001) | `u̠` 649 Hz, `ʊ̠` 765 Hz | `mandarin_u.rs`, `mandarin_ong.rs` |
| every compound and nasal rhyme two sounds: "ang is an + ng" (report 3) | native speaker | 奥 for 1.33 s: [a] still 0.76 s, a jump, [ʊ] still 0.35 s. 昂: an oral [ɑ], a 40 ms jump of the tongue body, 0.4 s of murmur. A phrase's last -n let go into a vowel | one movement into the ending over the note's last 0.2 s at most; the velum lowers gradually; the tongue body approaches the coda's place; a final coda stays closed | `mandarin_rhymes.rs` |
| every single syllable "a second tone" (report 4) | native speaker | each was one flat note; drawn as #85's held bends, 2 and 3 were one shape (low, then a jump to the note) | nothing in a song (the row above). Each tone as its own contour through the syllable was what the listener then heard as "like normal speech" on one pitch; it waits for speech (#48) | |

The same listener's verdict on the result: much improved, and still not a native speaker. What is
known to be wrong, with where it was measured:

- **Vowel qualities still lean English.** Against Zee & Lee's means for Beijing speakers: a's F1
  is 20-25% low (the model's jaw and lip range ends before it), o's F2 23% high, e [ɤ]'s F2 14-18%
  high, en and eng's F1 16-34% low, i and y's F2 7-12% low.
- **No tones.** On a melody that is the listener's choice. On one pitch, with no melody, the
  same listener heard the flat version as "speech with something wrong" and the tones drawn in
  full as "like normal speech": that is Speak's to do (#48), with declination, the neutral tone
  following the tone before it, and neighbouring tones leaning on each other, none of which a
  table of citation shapes has.
- **Aspirated affricates** (q ch c) have a silent gap between the frication and the vowel.
- **Rhythm is the notes'.** Nothing speaks Mandarin: Speak doesn't read it, and its timing and
  intonation come with #48.
- **Live notes** (a key pressed, the plugin) don't sing the closing rhyme.

## The one format

Every front end writes the same thing, one sung syllable per sounding note:

```json
{ "text": "풀", "onset": [{"ipa": "p", "aspirated": true, "vot": 0.09}], "nucleus": {"ipa": "u"},
  "coda": [{"ipa": "l"}], "wordEnd": true, "phraseEnd": false }
```

- `onset` / `coda`: consonants and glides from the inventory (`phonetics/inventory.rs`).
- A stop or affricate may say `"aspirated": true|false` (unsaid: English's rule; `ʰ` and `˭` in
  IPA text) and `"vot"`, its aspiration time in seconds relative to the 50 ms default (unsaid:
  English's 50 ms). Each millisecond more or less than 50 moves the voice's return by a
  millisecond; the measured voice onset also depends on the place and the voice (the Korean table
  above: 45 asked gives 53–69 ms, 90 gives 96–110).
- `nucleus`: one vowel, or a diphthong the planner knows (Japanese's ん is a nasal, held as a hum). Falling ones (aɪ eɪ aʊ oʊ ɔɪ əʊ) hold
  their first vowel and glide off at the note's end; rising ones (ia iɛ iɑ iʊ ua uo uɔ uɑ uə yɛ, with
  Mandarin's back u and ʊ: u̠a u̠ɔ u̠ɑ u̠ə iʊ̠, and er's əɝ aɝ) sing
  the first vowel as a quick glide of 50 ms that ends on the beat, after the initial consonant, so
  the second vowel holds from the beat; triphthongs (iaʊ iəʊ uaɪ ueɪ, and iaʊ̠ iəʊ̠ u̠aɪ u̠eɪ) do both. The glide is not held
  on the beat: sung there as a vowel of its own (it was a quarter of the vowel, at most 120 ms) the
  jaw and tongue sat at its posture for 80-100 ms and the main vowel was 90% reached 200 ms after
  the beat, and listening diagnoses of the renders heard 两 as *li-ang* and 快 as *ku-ai*. Now it is 90% reached
  80-90 ms after the beat (50 ms without a medial), `tests/medials.rs`. The apical vowels ɿ ʅ are
  calibrated vowels too.
- A consonant may say its sung `"dur"` (s); unsaid, its spoken length × 0.8.
- `pitch`: bends at the note's start, middle or end (below).
- `nasalLead`: on a closing rhyme (`closing`), nasalise this fraction of the vowel before its
  nasal coda (unsaid: none).
- `voiceLag`: how long after the onset's last consonant the voice starts (s): an aspiration, a
  burst, a fricative's release. A song sings the consonants that much earlier, so the voice comes
  in on the beat (before a medial glide the glide is that much longer at its start, and still
  ends on the beat). Unsaid: the vowel's segment starts on the beat and the voice after it, as
  every front end but Mandarin's is sung. Live notes ignore it.
- `onsetShare`: in a song, what comes before this syllable's vowel takes at most this share of
  the time since the vowel before, shortening in proportion. Unsaid: the consonants keep their
  lengths.
- `closing`: the rhyme's ending is one closing movement at the end of the note, this long at most
  (seconds; Mandarin says 0.2), instead of a part of its own. A falling off-glide then takes at
  most that long and at most 60% of the vowel, half of it the glide coming in and half the vowel
  letting go, so neither is held. Before a nasal coda the velum lowers gradually over the
  nasalised part and is open 60 ms before the closure (later, 东's close vowel was a stop for a
  moment), the tongue body moves toward the coda over the vowel's last 150 ms (up for -ng, back
  from a front vowel, forward and up for -n: what tells 安 from 昂 before the closure), and a
  phrase's last coda stays closed until the voice has faded. Unsaid: the off-glide holds the last
  20% of the note and the coda is a consonant after the vowel.
- A melisma note is `"melisma": true` with the same nucleus; its coda moves to the run's last note.
- `text` is what the lyrics panel and the piano roll show.
- `wordEnd` and `phraseEnd` may be left out (false).

Any front end's output can be frozen to JSON, edited and sung with `"syllables"` instead of
`"lyrics"`, in `render notes` and duet melody files. Use it to fix a reading
by hand, or to sing a language that has no front end yet.

## Pitch bends

Tones, pitch accent or a scoop need the pitch to move inside a note (#32 step 4). A syllable's
`pitch` is a list of bends, each `st` semitones from the note at the vowel's `start`, `middle` or
`end`, over `len` of the vowel but at most `max` seconds:

```json
"pitch": [{"at": "start", "len": 0.45, "max": 0.25, "st": -3.5}]
```

The melody stays as written; a bend only colours part of the note. A start bend begins in the
onset, so the consonants already sing its pitch and a rise starts low rather than at the note. A
middle bend is centred on the vowel (a dip or a peak inside it). A melisma continues at its written
pitches. The planner draws bends without knowing what they mean. A syllable without bends sings
exactly as before, and no built-in front end writes one now: Mandarin's tones were bends, and
its listener chose the melody as written (above).

## Adding a language

1. **Map its sounds onto the inventory.** Add missing ones between their neighbours, as
   `ɕ ʑ tɕ dʑ tʂ dʐ ɥ` were, and let `chimera calibrate consonants` set a fricative's constriction:
   it appends new entries and leaves existing ones alone.
2. **Write the front end**: implement `FrontEnd` (which characters start and continue its words,
   and word → syllables; `word_after` also gets the word before, for words that lean on each
   other, as a Russian preposition with no vowel does; `claims` lets it decline a word in a script
   it shares), then add it to the one router list, `lyrics_in`, which the app, the wasm engine and
   the CLI all read lyrics through (a Latin-script language also joins `Lang`, the player's
   choice). A front end that needs data loads it like the
   Mandarin lexicon: a gzipped file in `public/data/`, fetched only when the lyrics need it.
3. **Say the phonology** in the syllables (aspiration and its length, pitch bends, consonant
   lengths, the nucleus). Build `Syllable` and
   `Segment` values with `..Default::default()` so later fields don't break you.
4. **Leave English unchanged.** Tuning the shared voice goes through `docs/hill-climb/README.md`.
5. **Run the scramble test**, below, and **have a native speaker listen.**

## The scramble test

`tools/asr/scramble.py` (set up once with `tools/asr/setup.sh`) measures how intelligible a front
end's singing is on sound alone. A recogniser that knows a lyric fills it in, so it also sings the
same melody with the lyric's units shuffled (syllables for Chinese and Korean, words for English),
leaving nothing to guess from.

- **Material:** public-domain texts: two Tang poems (静夜思, 春晓), the folk song 아리랑, Harvard
  sentences. `--text lang:file` sings your own lines; `tools/asr/probes/` has lines dense in one
  contrast each. Material dense in what a change touches is the only way to see it.
- **Rendering:** through CHIMERA's own front ends (`render notes` with lyrics), one note per
  syllable on one generated tune, a breath between lines, in the male and female voices.
  A language whose script no front end reads yet is skipped.
- **Judge:** SenseVoice-Small (sherpa-onnx), one sung line at a time. It's CTC, so it can't
  hallucinate a loop.
- **Score:** syllable errors (Chinese as toneless pinyin, Korean as Hangul blocks) or word errors
  (English), for the real lyric and the mean over the scrambles.
- **What goes wrong:** each line is aligned with what was heard and the errors split into a
  syllable's parts (Chinese: initial / final / tone; Korean: initial / vowel / final; English:
  phonemes, wrong vs dropped), with the commonest confusions. `--analyze <run>/scramble.json`
  re-reads a run.

On this branch (8 scrambles × 2 voices):

| Language (units) | Real lyric (male / female) | Scrambled (male / female) |
|---|---|---|
| Korean, 아리랑 (syllables) | 30.0 / 15.0% | 52.8 / 55.6% |
| English, Harvard (words) | 48.8 / 53.7% | 68.6 / 75.0% |

The gap between the real and scrambled lyric is what the recogniser fills in from knowing the
text; the scrambled number is the sound. Korean's commonest errors are ㄹ heard as ㄴ or ㄷ and a
coda ㄴ dropped; English's are sounds dropped (a third of vowels and consonants), AH heard as ER,
SH as F and D as T.

Treat about ±2 points as noise at these sizes (8 scrambles × 2 voices) and read the confusion
counts for the sounds a change targets, not just the total. One recogniser and one tune can't stand
in for a native listener; the test says where to look and whether a change moved what it aimed at.

**It is a screen for named confusions; the ear decides.** The recogniser and a native listener can
disagree, and then the listener wins. #115 made Mandarin worse for the recogniser (scrambled 25.0 /
27.8% → 37.8 / 39.7%, male / female) and better to a native speaker's ear, which preferred it in
three of four comparisons.

## Not yet

- **More language choices.** Routing by script can't tell German from English, or Cantonese and
  Japanese from Mandarin, so Latin text is English unless it is toned pinyin or Turkish is chosen,
  and Han characters are Mandarin (which leaves a Japanese kanji nowhere to go but Mandarin's
  reading). Each new Latin-script language joins `Lang`, the one language setting (#40).
- **Bends on live notes:** a key pressed in Sing or a note from the plugin (`plan_note_on`) sings
  the note's pitch.
- **Tones in speech:** Speak doesn't read Mandarin yet; the tones' contours and speech tuning come
  next, from #48, onto this front end.
- **IPA tone letters** (`˥˩`, #52) don't write bends yet.
