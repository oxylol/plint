# Turkish

Speak and Sing read Turkish once you choose it: **TR**, next to "Text is IPA" in Speak and next
to the lyrics in Sing. It is one setting for both tabs, kept on the device with Sing's words.
Turkish is written in Latin letters, so the lyrics router can't tell it from English by script
([language-front-ends.md](language-front-ends.md)); English stays the default. Toned pinyin,
Hangul, Cyrillic and Han characters in the same lyrics still go to their own front ends.

`crates/chimera/src/phonetics/turkish.rs` reads the spelling by rule, with no dictionary, onto
the existing inventory (no new sounds).

## Writing it

- Type it as you'd write it: ç ğ ı İ ö ş ü, and â î û where you'd write them ("kâr", "hâlâ").
  `I` lowers to ı and `İ` to i.
- An apostrophe keeps a name apart from its suffixes, and marks it as a name: `Ankara'da`.
- In Sing, a vowel is a note: words split where Turkish spelling splits them (bağ-la-rın-da,
  dik-kat, a doubled consonant sung on both sides). `-` alone holds the previous vowel onto the
  next note, and hyphens split a word your way.
- Numbers are read out in Speak ("1.923,5" → "bin dokuz yüz yirmi üç virgül beş").

## What it does

| | |
|---|---|
| k g l | fronted ([c ɟ l]) next to a front vowel and before â û; [k g ɫ] next to a back vowel: kedi [cedi], kar [kaɾ], kâr [caɾ], yol [joɫ] |
| ğ | lengthens the vowel before a consonant or at the end: one vowel, 1.7 times what the duration rules give it (dağ [daː], yağmur [jaːmuɾ]); a glide between front vowels (değil [dejil]); gone between other vowels (ağaç [aatʃ]) |
| Doubled letters | in speech one long consonant, 1.6 times its length (dikkat [dikːat]); in song it closes one note and opens the next (dik-kat) |
| Stress | the last syllable, unless something earlier claims it (below) |
| Intonation | statement for every phrase: a yes/no question peaks before "mı" and falls, which the nuclear accent already gives |

Stress, in the order it is decided:

1. A listed word or root keeps its own: şimdi ŞİMdi, Ankara'da ANkarada. A derivational suffix
   makes a new word with final stress (ankaralı ankaraLI).
2. Clitics and the question particle take none: de, da, ki, ve, mı, musun.
3. A name (a capital that doesn't start a phrase, or an apostrophe) takes Sezer's rule: the
   antepenult when it is heavy and the penult light, else the penult (Antalya'da anTALyada).
4. A pre-stressing suffix puts it on the syllable before the leftmost one: geLİyor, GELmiyor,
   eDErim, geLİRken, araBAyla, evDEki.

## Not yet

- The suffix rules match spelling, not morphology, so a copula reads like the possessive
  (öğretmenim, "I am a teacher", gets the stress of "my teacher").
- A name that starts a sentence takes final stress (the capital doesn't mark it there), and a
  capitalised word derived from a name takes the name's rule (İstanbullu gets istanBULlu, not
  istanbulLU).
- No Turkish question tune beyond the nuclear accent; v is never [w].
- A held note inside a word (`yürü- - yorum`) ends the word at the hyphen, as in English: the
  syllables are right, but the mouth shows "rü" rather than "rü-".
- No scramble-test material in Turkish yet.
