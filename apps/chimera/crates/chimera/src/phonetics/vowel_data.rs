//! Reference vowel formants (Hz) used to calibrate articulatory vowel targets.
//! English vowels: Hillenbrand, Getty, Clark & Wheeler (1995) means (men / women / children).
//! Other cardinal-ish vowels: typical literature values (men), scaled for women/children.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VowelRef {
    pub ipa: &'static str,
    /// example word / description
    pub example: &'static str,
    pub male: [f64; 3],
    pub female: [f64; 3],
    pub child: [f64; 3],
    pub round: bool,
    /// 0 close … 1 open
    pub height: f64,
    /// 0 front … 1 back
    pub back: f64,
}

const fn scale_ref(m: [f64; 3], k: [f64; 3]) -> [f64; 3] {
    [m[0] * k[0], m[1] * k[1], m[2] * k[2]]
}
const FEM: [f64; 3] = [1.17, 1.2, 1.15];
const CHI: [f64; 3] = [1.3, 1.33, 1.24];

#[allow(clippy::too_many_arguments)]
const fn vref(ipa: &'static str, example: &'static str, male: [f64; 3], female: Option<[f64; 3]>, child: Option<[f64; 3]>, round: bool, height: f64, back: f64) -> VowelRef {
    let female = match female {
        Some(f) => f,
        None => scale_ref(male, FEM),
    };
    let child = match child {
        Some(c) => c,
        None => scale_ref(male, CHI),
    };
    VowelRef { ipa, example, male, female, child, round, height, back }
}

#[rustfmt::skip]
pub static VOWELS: &[VowelRef] = &[
    vref("i", "heed", [342.0, 2322.0, 3000.0], Some([437.0, 2761.0, 3372.0]), Some([452.0, 3081.0, 3702.0]), false, 0.0, 0.0),
    vref("ɪ", "hid", [427.0, 2034.0, 2684.0], Some([483.0, 2365.0, 3053.0]), Some([511.0, 2552.0, 3403.0]), false, 0.2, 0.15),
    vref("e", "hayed", [476.0, 2089.0, 2691.0], Some([536.0, 2530.0, 3047.0]), Some([564.0, 2656.0, 3323.0]), false, 0.3, 0.05),
    vref("ɛ", "head", [580.0, 1799.0, 2605.0], Some([731.0, 2058.0, 2979.0]), Some([749.0, 2267.0, 3310.0]), false, 0.55, 0.1),
    vref("æ", "had", [640.0, 1760.0, 2500.0], Some([669.0, 2349.0, 2972.0]), Some([717.0, 2501.0, 3289.0]), false, 0.8, 0.15),
    vref("a", "front open (Fr. patte)", [780.0, 1450.0, 2550.0], None, None, false, 1.0, 0.35),
    vref("ɑ", "hod", [768.0, 1333.0, 2522.0], Some([936.0, 1551.0, 2815.0]), Some([1002.0, 1688.0, 2950.0]), false, 1.0, 0.85),
    vref("ɒ", "Br. lot", [620.0, 880.0, 2500.0], None, None, true, 0.9, 1.0),
    vref("ɔ", "hawed", [652.0, 997.0, 2538.0], Some([781.0, 1136.0, 2824.0]), Some([803.0, 1210.0, 2982.0]), true, 0.65, 1.0),
    vref("o", "hoed", [497.0, 910.0, 2459.0], Some([555.0, 1035.0, 2828.0]), Some([597.0, 1137.0, 2987.0]), true, 0.35, 1.0),
    vref("ʊ", "hood", [469.0, 1122.0, 2434.0], Some([519.0, 1225.0, 2827.0]), Some([568.0, 1490.0, 3072.0]), true, 0.2, 0.85),
    vref("u", "who'd", [378.0, 997.0, 2343.0], Some([459.0, 1105.0, 2735.0]), Some([494.0, 1345.0, 2988.0]), true, 0.0, 1.0),
    vref("ʌ", "hud", [623.0, 1200.0, 2550.0], Some([753.0, 1426.0, 2933.0]), Some([749.0, 1546.0, 3145.0]), false, 0.6, 0.7),
    vref("ə", "sofa", [500.0, 1450.0, 2450.0], None, None, false, 0.5, 0.5),
    vref("ɐ", "near-open central", [640.0, 1350.0, 2500.0], None, None, false, 0.8, 0.5),
    vref("ɝ", "heard", [474.0, 1379.0, 1710.0], Some([523.0, 1588.0, 1929.0]), Some([586.0, 1719.0, 2143.0]), true, 0.45, 0.5),
    vref("y", "Fr. tu", [300.0, 1800.0, 2150.0], None, None, true, 0.0, 0.0),
    vref("ø", "Fr. peu", [390.0, 1550.0, 2250.0], None, None, true, 0.3, 0.1),
    vref("œ", "Fr. peur", [520.0, 1450.0, 2350.0], None, None, true, 0.55, 0.15),
    vref("ɨ", "close central", [330.0, 1600.0, 2400.0], None, None, false, 0.0, 0.5),
    vref("ɯ", "close back unrounded", [340.0, 1350.0, 2400.0], None, None, false, 0.0, 1.0),
    vref("ɤ", "close-mid back unrounded", [460.0, 1250.0, 2450.0], None, None, false, 0.35, 1.0),
    // Mandarin's apical vowels: the -i of 四 si (after z c s) and of 是 shi (after zh ch sh r), the
    // consonant's tongue held open; the retroflex one has a low F3 (approximate male means)
    vref("ɿ", "Mandarin si (apical)", [360.0, 1400.0, 2600.0], None, None, false, 0.0, 0.45),
    vref("ʅ", "Mandarin shi (apical, retroflex)", [400.0, 1500.0, 2000.0], None, None, false, 0.05, 0.55),
    // Mandarin's own vowels, measured on Beijing speakers (Zee & Lee 2001, doi 10.21437/Eurospeech.2001-169:
    // 10 men and 10 women, 5 repetitions, LPC at the vowel's midpoint; female values are theirs, the child's
    // are the male's scaled by the file's factor). The English vowels of the same name are not Mandarin's:
    // American /u/ "who'd" has F2 ~1000 Hz, Mandarin's 661. The key is the English symbol plus U+0320
    // (retracted), so these entries cannot collide with, or be reached by typing, the English ones.
    // u̠ achieves (F1 / F2 / F3 Hz, target in brackets): male 383 / 649 / 2482 (345 / 661 / 2783), female 366 / 776 / 2841
    // (345 / 759 / 3309), child 465 / 942 / 3344 (449 / 879 / 3451). F2, what makes it not English /u/, is within
    // 2-3 % for the adults; F1 is +11 % male and +6 % female because the tube cannot go lower: male tbX, jaw and
    // lip width are at their lower bounds (1.2, 0.05, 0) and lip protrusion at its upper (1.4). The female
    // fit starts from the male solution (calibrate.rs `start_from`; from the generic guess it settled at
    // jaw 1.8, F2 +11 %). The child's F2 is +7 %.
    vref("u̠", "Mandarin wu, hu (back u)", [345.0, 661.0, 2783.0], Some([345.0, 759.0, 3309.0]), None, true, 0.0, 1.0),
    // ʊ̠ is the vowel of ong / iong and the off-glide of ao ou iao iou (the table's "u before ŋ": Zee & Lee's
    // men 441 / 730 / 2881, women 533 / 945 / 3314; English /ʊ/ "hood" has F2 1122 / 1225). It achieves
    // (F1 / F2 / F3 Hz): male 422 / 765 / 2345, female 513 / 972 / 2859, child 555 / 1004 / 3360: F1 and F2 within
    // 5 % on every body (F3, weighted least in the fit, -14 to -19 % for the adults).
    vref("ʊ̠", "Mandarin ong (back ʊ)", [441.0, 730.0, 2881.0], Some([533.0, 945.0, 3314.0]), None, true, 0.2, 1.0),
];
