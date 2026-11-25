//! Demo melodies for Sing mode: [MIDI note, beats] per syllable, in the adult female register.

use serde::Serialize;

/// octave shift per body
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SongOctave {
    pub male: f64,
    pub female: f64,
    pub child: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DemoSong {
    pub name: &'static str,
    pub bpm: f64,
    pub lyrics: &'static str,
    pub notes: &'static [[f64; 2]],
    /// octave shift per body
    pub octave: SongOctave,
}

#[rustfmt::skip]
pub static DEMO_SONGS: &[DemoSong] = &[
    DemoSong {
        // an original tune for this instrument
        name: "Made of air",
        bpm: 84.0,
        lyrics: "Breathe in, breathe out, I am made of air. Tongue and lips and throat, singing anywhere.",
        notes: &[
            [64.0, 1.0], [60.0, 1.0], [62.0, 1.0], [57.0, 1.0],
            [57.0, 0.5], [60.0, 0.5], [64.0, 1.0], [62.0, 0.5], [60.0, 1.5],
            [65.0, 1.0], [64.0, 0.5], [62.0, 0.5], [60.0, 0.5], [62.0, 1.5],
            [64.0, 0.5], [62.0, 0.5], [60.0, 0.5], [59.0, 0.5], [57.0, 2.0],
        ],
        octave: SongOctave { male: -1.0, female: 0.0, child: 0.0 },
    },
    DemoSong {
        // traditional (public domain)
        name: "Twinkle, twinkle",
        bpm: 100.0,
        lyrics: "Twinkle twinkle little star, how I wonder what you are.",
        notes: &[
            [60.0, 1.0], [60.0, 1.0], [67.0, 1.0], [67.0, 1.0], [69.0, 1.0], [69.0, 1.0], [67.0, 2.0],
            [65.0, 1.0], [65.0, 1.0], [64.0, 1.0], [64.0, 1.0], [62.0, 1.0], [62.0, 1.0], [60.0, 2.0],
        ],
        octave: SongOctave { male: -1.0, female: 0.0, child: 0.0 },
    },
    DemoSong {
        // an original tune and words ("The night wind sings softly. Moonlight drifts over the
        // forest."); the stress marks are part of the lyrics
        name: "Ночной ветер",
        bpm: 112.0,
        lyrics: "Но́чью ве́тер ти́хо поёт. Лу́нный све́т над ле́сом плывёт.",
        notes: &[
            [64.0, 1.0], [62.0, 0.5], [60.0, 0.5], [59.0, 1.0], [60.0, 1.0], [62.0, 1.0], [59.0, 1.0], [57.0, 2.0],
            [60.0, 1.0], [62.0, 0.5], [64.0, 1.5], [65.0, 1.0], [64.0, 0.5], [62.0, 0.5], [59.0, 1.0], [57.0, 2.0],
        ],
        octave: SongOctave { male: -1.0, female: 0.0, child: 0.0 },
    },
    DemoSong {
        // by Harry Dacre, 1892 (public domain)
        name: "Daisy Bell",
        bpm: 175.0,
        lyrics: "daisy daisy give me your answer do i'm half crazy all for the love of you it won't be a stylish marriage i can't afford a carriage but you'll look sweet upon the seat of a bicycle built for two",
        notes: &[
            [66.0, 3.0], [63.0, 3.0], [59.0, 3.0], [54.0, 3.0],
            [56.0, 1.0], [58.0, 1.0], [59.0, 1.0], [56.0, 2.0],
            [59.0, 1.0], [54.0, 6.0], [61.0, 3.0], [66.0, 3.0],
            [63.0, 3.0], [59.0, 3.0], [56.0, 1.0], [58.0, 1.0],
            [59.0, 1.0], [61.0, 2.0], [63.0, 1.0], [61.0, 5.0],
            [63.0, 1.0], [64.0, 1.0], [63.0, 1.0], [61.0, 1.0],
            [66.0, 2.0], [63.0, 1.0], [61.0, 1.0], [59.0, 4.0],
            [61.0, 1.0], [63.0, 2.0], [59.0, 1.0], [56.0, 2.0],
            [59.0, 1.0], [56.0, 1.0], [54.0, 4.0], [54.0, 1.0],
            [59.0, 2.0], [63.0, 1.0], [61.0, 2.0], [54.0, 1.0],
            [59.0, 2.0], [63.0, 1.0], [61.0, 1.0], [63.0, 1.0],
            [64.0, 1.0], [66.0, 1.0], [63.0, 1.0], [59.0, 1.0],
            [61.0, 2.0], [54.0, 1.0], [59.0, 6.0],
        ],
        octave: SongOctave { male: -1.0, female: 0.0, child: 0.0 },
    },
    DemoSong {
        // traditional (public domain): the koto piece "Saita sakura", words of the 1888 school
        // songbook (anonymous). One note per mora: ー holds the vowel of the kana before it, and
        // the ん of "かん" is a note of its own. Each line is a phrase.
        name: "さくら さくら",
        bpm: 112.0,
        lyrics: "さくら さくら、やよいの そらーわ、みわたす かぎーり、かすみか くもーか、においぞ いずーる、いざや いざや、みにゆー かん。",
        notes: &[
            [69.0, 1.0], [69.0, 1.0], [71.0, 2.0],
            [69.0, 1.0], [69.0, 1.0], [71.0, 2.0],
            [69.0, 1.0], [71.0, 1.0], [72.0, 1.0], [71.0, 1.0],
            [69.0, 1.0], [71.0, 0.5], [69.0, 0.5], [65.0, 2.0],
            [64.0, 1.0], [60.0, 1.0], [64.0, 1.0], [65.0, 1.0],
            [64.0, 1.0], [64.0, 0.5], [60.0, 0.5], [59.0, 2.0],
            [69.0, 1.0], [71.0, 1.0], [72.0, 1.0], [71.0, 1.0],
            [69.0, 1.0], [71.0, 0.5], [69.0, 0.5], [65.0, 2.0],
            [64.0, 1.0], [60.0, 1.0], [64.0, 1.0], [65.0, 1.0],
            [64.0, 1.0], [64.0, 0.5], [60.0, 0.5], [59.0, 2.0],
            [69.0, 1.0], [69.0, 1.0], [71.0, 2.0],
            [69.0, 1.0], [69.0, 1.0], [71.0, 2.0],
            [64.0, 1.0], [65.0, 1.0], [71.0, 0.5], [69.0, 0.5], [65.0, 1.0], [64.0, 2.0],
        ],
        octave: SongOctave { male: -1.0, female: 0.0, child: 0.0 },
    },
    DemoSong {
        // words by Li Shutong (李叔同, 1915; he died in 1942) to J. P. Ordway's "Dreaming of Home
        // and Mother" (1851): public domain. The first verse, one character per note; "-" holds the
        // vowel onto the second note of a pair (1=C, from the usual numbered notation)
        name: "送别",
        bpm: 72.0,
        lyrics: "长亭 - 外，古道边，芳草 - 碧连 - 天。晚风 - 拂柳笛声残，夕阳 - 山外山。天之涯，地之 - 角，知 - 交 - 半 - 零 - 落。一壶 - 浊酒尽余欢，今宵 - 别梦寒。",
        notes: &[
            [67.0, 1.0], [64.0, 0.5], [67.0, 0.5], [72.0, 2.0], [69.0, 1.0], [72.0, 1.0], [67.0, 2.0],
            [67.0, 1.0], [60.0, 0.5], [62.0, 0.5], [64.0, 1.0], [62.0, 0.5], [60.0, 0.5], [62.0, 4.0],
            [67.0, 1.0], [64.0, 0.5], [67.0, 0.5], [72.0, 1.5], [71.0, 0.5], [69.0, 1.0], [72.0, 1.0], [67.0, 2.0],
            [67.0, 1.0], [62.0, 0.5], [64.0, 0.5], [65.0, 1.5], [59.0, 0.5], [60.0, 4.0],
            [69.0, 1.0], [72.0, 1.0], [72.0, 2.0], [71.0, 1.0], [69.0, 0.5], [71.0, 0.5], [72.0, 2.0],
            [69.0, 0.5], [71.0, 0.5], [72.0, 0.5], [69.0, 0.5], [69.0, 0.5], [67.0, 0.5], [64.0, 0.5], [60.0, 0.5], [62.0, 4.0],
            [67.0, 1.0], [64.0, 0.5], [67.0, 0.5], [72.0, 1.5], [71.0, 0.5], [69.0, 1.0], [72.0, 1.0], [67.0, 2.0],
            [67.0, 1.0], [62.0, 0.5], [64.0, 0.5], [65.0, 1.5], [59.0, 0.5], [60.0, 4.0],
        ],
        octave: SongOctave { male: -1.0, female: 0.0, child: 0.0 },
    },
];
