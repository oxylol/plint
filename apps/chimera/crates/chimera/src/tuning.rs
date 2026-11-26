//! Every number of the human voice that an optimiser may move, as one plain value.
//!
//! `Tuning::default()` is the engine as it stands: each field's default is the literal the code
//! has at the place its comment names, digit for digit, and each comment gives the expression the
//! field goes into. Keep that expression's shape when the literal becomes the field (Rust has no
//! fast-math, so the same operations in the same order give the same bits); a comment that says
//! "kept literal" names a neighbour that stays a number in the code.
//!
//! Not here, on purpose: tube lengths and areas (the vowel tables are calibrated to them),
//! anything under `creatures/`, the safety limits (the limiter, `TINY`, `closed_area`, the pulse's
//! clamps) and the seeds.
//!
//! The JSON form is camelCase and partial: `{"source":{"oqBase":0.7}}` leaves everything else at
//! its default, and a name that is not a field is an error. [`Tuning::fields`] is the table the
//! optimiser and the bounds live off: one row per field, its JSON path, its range and its accessors.
//! Everything is `Copy` data with no heap behind it, so a voice may keep its own copy and read it
//! while rendering.
//!
//! Bounds are half the default either way unless a comment gives another range and its reason:
//! two edges of one `smoothstep` may not meet, a bound may not cross a clamp in the code, and a zero
//! default has an explicit range.

use crate::anatomy::human::N_ARTIC;
use serde::{Deserialize, Serialize};

/// One tunable number: its JSON path (`"source.oqBase"`), the range a fitter may search, and how
/// to read and write it.
#[derive(Clone, Copy)]
pub struct Field {
    pub path: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub get: fn(&Tuning) -> f64,
    pub set: fn(&mut Tuning, f64),
}

impl core::fmt::Debug for Field {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Field({} in {}..{})", self.path, self.lo, self.hi)
    }
}

macro_rules! lo {
    ($d:expr, half) => {
        0.5 * $d
    };
    ($d:expr, [$lo:expr, $hi:expr]) => {
        $lo
    };
}
macro_rules! hi {
    ($d:expr, half) => {
        1.5 * $d
    };
    ($d:expr, [$lo:expr, $hi:expr]) => {
        $hi
    };
}

/// `group: Struct { field, "jsonName" = default, half | [lo, hi]; … } [ switch = false; … ]` for
/// every group: the structs, their defaults, and the one table of fields. The switches in the
/// optional square brackets are `bool`s, in the struct and its JSON but not in the table.
macro_rules! tuning_groups {
    ($( $(#[$gm:meta])* $group:ident: $ty:ident { $( $(#[$fm:meta])* $f:ident, $json:literal = $d:expr, $b:tt; )* } $( [ $( $(#[$sm:meta])* $s:ident = $sd:expr; )* ] )? )*) => {
        $(
            $(#[$gm])*
            #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
            #[serde(default, rename_all = "camelCase", deny_unknown_fields)]
            pub struct $ty {
                $( $(#[$fm])* pub $f: f64, )*
                $( $( $(#[$sm])* pub $s: bool, )* )?
            }
            impl Default for $ty {
                fn default() -> Self {
                    $ty { $( $f: $d, )* $( $( $s: $sd, )* )? }
                }
            }
        )*
        static FIELDS: &[Field] = &[
            $( $( Field {
                path: concat!(stringify!($group), ".", $json),
                lo: lo!($d, $b),
                hi: hi!($d, $b),
                get: |t| t.$group.$f,
                set: |t, x| t.$group.$f = x,
            }, )* )*
        ];
    };
}

tuning_groups! {
    /// The vocal folds: `HumanVoice::apply_source`, `effort_at` and `control`
    /// (voice/human_voice.rs) and `ParametricGlottis` (sources/glottis.rs).
    source: SourceTuning {
        /// The lung pressure a voice starts with (cmH₂O). human_voice.rs:51, `DEFAULT_SOURCE`'s
        /// `pressure: 7.2`. The const stays (the page, the plugin and serde's defaults read it):
        /// `with_tuning` starts the voice from `DEFAULT_SOURCE` with `pressure` and `drift`
        /// replaced by these two, and a later `set_source` patch still wins. Range 5..12:
        /// conversational speech runs at about 5-10 cmH₂O and loud speech somewhat over. The
        /// voicing gate is fully open once the folds' pressure (about 0.1 under the lungs')
        /// passes `thresholdCm * thresholdHi`, 4.0 by default and so under every pressure here;
        /// with both of those at their tops it is 7.0, and a pressure under about 7.1 then leaves
        /// the gate part open (a weak voice, not a fault). A box cannot say that corner, and a
        /// lower bound of 7.1 would leave no room below the default, so it is left to the judge.
        pressure, "pressure" = 7.2, [5.0, 12.0];
        /// The slow random pitch drift a voice starts with (cents). human_voice.rs:51,
        /// `DEFAULT_SOURCE`'s `drift: 5.0`; applied as `pressure` is.
        drift, "drift" = 5.0, half;
        /// Peak pulse area per unit fold length and scale (cm²). human_voice.rs:384,
        /// `0.17 * body.fold_length * body.scale * (voiced)`.
        amplitude, "amplitude" = 0.17, half;
        /// Abduction of a voiced source per breathiness² (cm²). human_voice.rs:389, the voiced
        /// arm `0.1 * s.breathiness * s.breathiness`.
        abduction_breathy, "abductionBreathy" = 0.1, half;
        /// Abduction of an unvoiced (whispering) source (cm²). human_voice.rs:389, the other arm,
        /// `0.2 + 0.1 * s.breathiness`: this is the 0.2.
        whisper_abduction, "whisperAbduction" = 0.2, half;
        /// The same arm's `0.1 * s.breathiness`.
        whisper_abduction_breathy, "whisperAbductionBreathy" = 0.1, half;
        /// The posterior chink (cm²). human_voice.rs:395, `(0.008 + 0.03 * s.breathiness) * k2`:
        /// the 0.008.
        chink_base, "chinkBase" = 0.008, half;
        /// The same line's 0.03.
        chink_breathy, "chinkBreathy" = 0.03, half;
        /// Open quotient at tension 0. human_voice.rs:396, `0.78 - 0.36 * s.tension +
        /// body.open_phase`. Range 0.45..0.95: the pulse clamps the quotient to 0.25..0.98
        /// (glottis.rs:205, 207) and at the default tension it is this less 0.18.
        oq_base, "oqBase" = 0.78, [0.45, 0.95];
        /// The same line's 0.36.
        oq_tension, "oqTension" = 0.36, half;
        /// Closed phase lost per cm² of abduction. human_voice.rs:399, `g.open_with_abduction = 0.98`.
        oq_with_abduction, "oqWithAbduction" = 0.98, half;
        /// How the open quotient follows pitch, per octave (the log's lead 35). The code has no
        /// such term today, so 0 must leave it untouched: add `oq_per_octave * log2(f0 / 110.0)`
        /// inside the clamp of glottis.rs:205 only under `if oq_per_octave != 0.0`, with the
        /// product made at control rate where human_voice.rs:599 sets `params.f0` (a per-sample
        /// `log2` would cost, and `0.0 * log2(0.0)` is NaN). 110 Hz is `DEFAULT_SOURCE.f0`, kept
        /// literal. Range −0.1..0.3: Patel et al. 2026 read about +0.25 an octave in men.
        oq_per_octave, "oqPerOctave" = 0.0, [-0.1, 0.3];
        /// Closure sharpness per tension. human_voice.rs:407, `0.4 * s.tension + 1.375 *
        /// js_max(0.0, s.tension - 0.6)`: the 0.4.
        closure_tension, "closureTension" = 0.4, half;
        /// The same line's 1.375.
        closure_pressed, "closurePressed" = 1.375, half;
        /// The same line's 0.6, the tension above which a voice is pressed.
        closure_pressed_from, "closurePressedFrom" = 0.6, half;
        /// Closure sharpness per brightness off the default. human_voice.rs:408,
        /// `clamp(by_tension + 0.8 * bright, 0.0, 1.0)`.
        closure_bright, "closureBright" = 0.8, half;
        /// The pulse's skew at tension 0. human_voice.rs:409, `0.56 + 0.16 * s.tension + 0.2 *
        /// bright`. Range 0.35..0.8: `shape()` clamps the skew to 0.3..0.9 (glottis.rs:260) and the
        /// default tension adds 0.08.
        skew_base, "skewBase" = 0.56, [0.35, 0.8];
        /// The same line's 0.16.
        skew_tension, "skewTension" = 0.16, half;
        /// The same line's 0.2.
        skew_bright, "skewBright" = 0.2, half;
        /// Jitter at roughness 0. human_voice.rs:413, `0.0021 + 0.0245 * s.roughness`.
        jitter_base, "jitterBase" = 0.0021, half;
        /// The same line's 0.0245.
        jitter_rough, "jitterRough" = 0.0245, half;
        /// Shimmer at roughness 0. human_voice.rs:414, `0.02 + 0.22 * s.roughness`.
        shimmer_base, "shimmerBase" = 0.02, half;
        /// The same line's 0.22.
        shimmer_rough, "shimmerRough" = 0.22, half;
        /// The roughness above which periods alternate. human_voice.rs:415, both 0.5s of
        /// `if s.roughness > 0.5 { (s.roughness - 0.5) * 1.4 } else { 0.0 }`.
        diplophonia_from, "diplophoniaFrom" = 0.5, half;
        /// The same line's 1.4.
        diplophonia_gain, "diplophoniaGain" = 1.4, half;
        /// Lung pressure at which effort is 0 (cmH₂O). human_voice.rs:57,
        /// `clamp((pressure.abs() - 2.2) / 10.0, 0.0, 1.0)`.
        effort_threshold, "effortThreshold" = 2.2, half;
        /// The same line's 10.0 (cmH₂O from no effort to full).
        effort_span, "effortSpan" = 10.0, half;
        /// Pitch glide time while a score sets the pitch (s). human_voice.rs:597,
        /// `js_max(0.002, if self.score_f0.is_some() { 0.03 } else { self.source.glide })`; the
        /// 0.002 floor is kept literal.
        pitch_glide, "pitchGlide" = 0.03, half;
        /// The breath's attack time (s). human_voice.rs:573, the first arm of `tau`.
        breath_attack, "breathAttack" = 0.025, half;
        /// The breath's release time under a score (s). human_voice.rs:575, the second arm of
        /// `tau`. (The third, 0.07 at :577, is the performer's note-off and kept literal.)
        breath_release, "breathRelease" = 0.02, half;
        /// Corner of the transglottal pressure's smoothing (Hz). glottis.rs:129,
        /// `1.0 - exp(-TAU * 30.0 / fs)`.
        pressure_smooth_hz, "pressureSmoothHz" = 30.0, half;
        /// Corner of the voicing gate's smoothing (Hz). glottis.rs:130, `1.0 - exp(-TAU * 25.0 / fs)`.
        voicing_smooth_hz, "voicingSmoothHz" = 25.0, half;
        /// Phonation threshold pressure (cmH₂O). glottis.rs:133, `2.5 * CM_H2O`. Range 1.5..3.5:
        /// the measured threshold is 2-3, and with `thresholdHi` at its top the gate must still
        /// open fully under the 7.1 the folds see at the default pressure.
        threshold_cm, "thresholdCm" = 2.5, [1.5, 3.5];
        /// Where the pressure gate starts, as a share of the threshold. glottis.rs:175,
        /// `smoothstep(self.threshold * 0.8, self.threshold * 1.6, self.dp_sm)`. Range 0.4..1.0:
        /// under the other edge's lowest.
        threshold_lo, "thresholdLo" = 0.8, [0.4, 1.0];
        /// The same line's 1.6. Range 1.1..2.0: above the other edge, and 3.5 × 2.0 stays under
        /// the folds' 7.1 cmH₂O at the default pressure (see `pressure`).
        threshold_hi, "thresholdHi" = 1.6, [1.1, 2.0];
        /// Static opening at which voicing starts to fail (cm², not body-scaled). glottis.rs:176,
        /// `1.0 - smoothstep(0.08, 0.22, self.static_sm - p.chink)`.
        adduct_lo, "adductLo" = 0.08, half;
        /// The same line's 0.22. Range 0.13..0.33: above the other edge's highest (0.12).
        adduct_hi, "adductHi" = 0.22, [0.13, 0.33];
        /// Open quotient lost per unit of effort. glottis.rs:205,
        /// `clamp(p.open_quotient - 0.18 * (p.effort - 0.5), 0.25, 0.98)`; the clamp is kept literal.
        oq_effort, "oqEffort" = 0.18, half;
        /// The same line's 0.5, the effort at which the quotient is the source's own.
        oq_effort_mid, "oqEffortMid" = 0.5, half;
        /// Pulse amplitude at no effort. glottis.rs:219, `0.45 + 0.75 * p.effort`.
        effort_amp_base, "effortAmpBase" = 0.45, half;
        /// The same line's 0.75.
        effort_amp_gain, "effortAmpGain" = 0.75, half;
        /// Time constant of the per-sample smoothing of amplitude, closure and abduction (s).
        /// glottis.rs:137, `1.0 - exp(-1.0 / (0.004 * fs))`.
        smooth_seconds, "smoothSeconds" = 0.004, half;
    }

    /// The tubes' walls and ports and how fast an area may change: `HumanAirway::new`
    /// (voice/human_airway.rs).
    airway: AirwayTuning {
        /// Share of the pharynx walls' motion that radiates. human_airway.rs:123, `.radiating(0.25)`.
        pharynx_wall_radiation, "pharynxWallRadiation" = 0.25, half;
        /// Pharynx perimeter over a circle's of the same area. human_airway.rs:123,
        /// `.perimeter(1.25)`. Range from 1.0: no section has less perimeter than a circle.
        pharynx_perimeter, "pharynxPerimeter" = 1.25, [1.0, 1.875];
        /// Share of the mouth walls' motion that radiates. human_airway.rs:124, `.radiating(0.25)`.
        oral_wall_radiation, "oralWallRadiation" = 0.25, half;
        /// Mouth perimeter over a circle's. human_airway.rs:124, `.perimeter(1.5)`. From 1.0, as above.
        oral_perimeter, "oralPerimeter" = 1.5, [1.0, 2.25];
        /// Nasal perimeter over a circle's. human_airway.rs:142, `.perimeter(3.2)`.
        nasal_perimeter, "nasalPerimeter" = 3.2, half;
        /// Multiplier on the nasal tube's viscous loss. human_airway.rs:142, `.viscous(2.5)`.
        nasal_viscous, "nasalViscous" = 2.5, half;
        /// Piriform fossa perimeter over a circle's. human_airway.rs:146, `.perimeter(1.3)`. From 1.0.
        piriform_perimeter, "piriformPerimeter" = 1.3, [1.0, 1.95];
        /// The lung port's resistance as a share of the trachea's characteristic impedance.
        /// human_airway.rs:152, `(0.5 * RHO * C_SOUND) / tr_a`.
        lung_termination, "lungTermination" = 0.5, half;
        /// The lung port's entrance loss. human_airway.rs:153, the first of `.kinetic(0.0, 0.0)`.
        /// Range 0..0.5: a sharp-edged contraction loses at most about half a dynamic head.
        lung_kinetic_in, "lungKineticIn" = 0.0, [0.0, 0.5];
        /// The same call's second 0.0, the exit loss. Range 0..1: a jet loses at most all of its
        /// dynamic head.
        lung_kinetic_out, "lungKineticOut" = 0.0, [0.0, 1.0];
        /// Lower-airway resistance, male (cmH₂O per L/s). human_airway.rs:23, `LOWER_AIRWAY_R`
        /// (used at :154).
        lower_airway_r, "lowerAirwayR" = 0.8, half;
        /// Lower-airway inertance, male (g/cm⁴). human_airway.rs:26, `LOWER_AIRWAY_L` (used at :155).
        lower_airway_l, "lowerAirwayL" = 5.2e-3, half;
        /// The glottis's entrance loss. human_airway.rs:156, the first of `.kinetic(0.37, 1.0)`.
        glottis_kinetic_in, "glottisKineticIn" = 0.37, half;
        /// The same call's 1.0, the exit loss. Range 0.5..1.0: no more than the whole dynamic head.
        glottis_kinetic_out, "glottisKineticOut" = 1.0, [0.5, 1.0];
        /// The velum port's entrance loss. human_airway.rs:158, the first of `.kinetic(0.3, 0.8)`.
        velum_kinetic_in, "velumKineticIn" = 0.3, half;
        /// The same call's 0.8. Range 0.4..1.0, as for the glottis.
        velum_kinetic_out, "velumKineticOut" = 0.8, [0.4, 1.0];
        /// Fastest area change (cm²/s at scale 1). human_airway.rs:196, `max_area_rate: 45.0`.
        max_area_rate, "maxAreaRate" = 45.0, half;
        /// More rate per cm² of area (1/s). human_airway.rs:197, `area_rate_gain: 40.0`.
        area_rate_gain, "areaRateGain" = 40.0, half;
        /// Closing rate at the seal (cm²/s). human_airway.rs:198, `seal_rate: 5.0`.
        seal_rate, "sealRate" = 5.0, half;
        /// Area below which closing is at the seal rate (cm²). human_airway.rs:199, `seal_lo: 0.03`.
        seal_lo, "sealLo" = 0.03, half;
        /// Area above which closing is at the full rate (cm²). human_airway.rs:200, `seal_hi: 0.4`.
        seal_hi, "sealHi" = 0.4, half;
        /// Area below which closing slows with √A (cm²). human_airway.rs:201, `seal_ease: 0.03`.
        seal_ease, "sealEase" = 0.03, half;
        /// The lips' opening rate at contact (cm²/s). human_airway.rs:202, `lip_open_rate: 7.0`.
        lip_open_rate, "lipOpenRate" = 7.0, half;
    }

    /// Turbulence: the three sources of `HumanAirway::update_noise` (voice/human_airway.rs).
    noise: NoiseTuning {
        /// Aspiration's gain. human_airway.rs:195, `NoiseGains { glottis: 3.2e-6, .. }`.
        /// Range a quarter to four times the default (±12 dB), for this and the two gains below:
        /// a gain is a level, and half either way (−6 to +3.5 dB) was the table's default, not
        /// physics. The first fit made the source 4-6 dB louder against every consonant and had
        /// no room here to bring the noise up with it (plan-chimera-fit, T3.5).
        glottis_gain, "glottisGain" = 3.2e-6, [0.8e-6, 12.8e-6];
        /// The constriction source's gain. human_airway.rs:195, `constriction: 1.364e-6`.
        /// Range a quarter to four times the default, as `glottisGain`.
        constriction_gain, "constrictionGain" = 1.364e-6, [0.341e-6, 5.456e-6];
        /// The teeth source's gain. human_airway.rs:195, `teeth: 3.0e-6`.
        /// Range a quarter to four times the default, as `glottisGain`.
        teeth_gain, "teethGain" = 3.0e-6, [0.75e-6, 12.0e-6];
        /// Top of aspiration's band (Hz). human_airway.rs:337, the first of `.band(3500.0, 300.0)`.
        aspiration_top, "aspirationTop" = 3500.0, half;
        /// The same call's 300.0, the band's bottom (Hz).
        aspiration_low, "aspirationLow" = 300.0, half;
        /// How steeply aspiration falls above its top, in one-pole stages (6 dB an octave each).
        /// Every source has two today (network.rs:1176-1177, `lp1`, `lp2`, shared by all the
        /// sources of `compute_noise`), and the code has no such parameter: at exactly 2.0 the
        /// aspiration source's output must be `s.lp2` itself, taken by a branch and not by
        /// arithmetic. Otherwise the whole part picks the stage and the fraction mixes toward the
        /// next (`lp_n + frac * (lp_next - lp_n)`), with stages past the second computed only for
        /// a source that asks for them. Range 1..4. No lead in the author's log is about
        /// aspiration's roll-off: its lead 11 is the teeth source's (/s ʃ/, see `teethOrder`), and
        /// the nearest on aspiration is lead 34, a third of /f/'s power being aspiration, which is
        /// about its level. The reason for this field is Plint's own measurement: in a held vowel
        /// the 4-8 kHz band is noise 8 dB over the harmonics.
        aspiration_order, "aspirationOrder" = 2.0, [1.0, 4.0];
        /// Top of the constriction source's band (Hz). human_airway.rs:374, the first of
        /// `.band(2600.0, 350.0)`.
        constriction_top, "constrictionTop" = 2600.0, half;
        /// The same call's 350.0.
        constriction_low, "constrictionLow" = 350.0, half;
        /// Narrowest area at and below which frication is full (cm² at scale 1).
        /// human_airway.rs:352, `smooth(0.35 * k2, FRICATION_FADE_AREA * k2, min_a)`. Range to
        /// 0.45: under the other edge's lowest.
        frication_full, "fricationFull" = 0.35, [0.175, 0.45];
        /// Narrowest area above which there is no frication (cm² at scale 1). human_airway.rs:32,
        /// `FRICATION_FADE_AREA` (used at :352; the const stays for tests/frication_junction.rs).
        /// Range from 0.5: above the other edge's highest.
        frication_fade_area, "fricationFadeArea" = 0.9, [0.5, 1.35];
        /// Distance behind the teeth at which a jet starts to strike them (cm at scale 1).
        /// human_airway.rs:380, `smooth(1.0, 1.5, behind) * (1.0 - smooth(2.6, 4.0, behind))`:
        /// the 1.0. Range to 1.2: under the next edge's lowest.
        teeth_near_lo, "teethNearLo" = 1.0, [0.5, 1.2];
        /// The same line's 1.5. Range from 1.25.
        teeth_near_hi, "teethNearHi" = 1.5, [1.25, 2.25];
        /// The same line's 2.6. Range to 3.2: under the next edge's lowest.
        teeth_far_lo, "teethFarLo" = 2.6, [1.3, 3.2];
        /// The same line's 4.0. Range from 3.3.
        teeth_far_hi, "teethFarHi" = 4.0, [3.3, 6.0];
        /// Jet area at and below which the teeth source is full (cm² at scale 1).
        /// human_airway.rs:381, `1.0 - smooth(0.2 * k2, 0.5 * k2, min_a)`.
        teeth_jet_lo, "teethJetLo" = 0.2, half;
        /// The same line's 0.5. Range from 0.31: above the other edge's highest.
        teeth_jet_hi, "teethJetHi" = 0.5, [0.31, 0.75];
        /// Strouhal number of the teeth source's band top. human_airway.rs:16, `TEETH_STROUHAL`
        /// (used at :396).
        teeth_strouhal, "teethStrouhal" = 1.0, half;
        /// Growth of the jet's mixing layer with distance. human_airway.rs:29, `JET_SPREAD`
        /// (used at :395).
        jet_spread, "jetSpread" = 0.17, half;
        /// Bottom of the teeth source's band (Hz). human_airway.rs:35, `TEETH_HIGHPASS_HZ`
        /// (used twice at :397-398).
        teeth_highpass_hz, "teethHighpassHz" = 1200.0, half;
        /// How steeply the teeth source falls above its band's top, in one-pole stages: the log's
        /// lead 11 (the speakers' /s ʃ/ fall 25 dB an octave or more, two poles only 12). Two
        /// today: the source pushed at human_airway.rs:398 goes through the same `lp1`, `lp2` as
        /// every other (network.rs:1176-1177). The rule is `aspirationOrder`'s: at exactly 2.0 the
        /// output is `s.lp2` itself, by a branch; otherwise the whole part picks the stage and the
        /// fraction mixes toward the next. Range 1..4. (The constriction source keeps its two.)
        teeth_order, "teethOrder" = 2.0, [1.0, 4.0];
    }

    /// The planner's pitch line and the folds' gestures a listener hears (phonetics/planner.rs).
    /// With [`DurationsTuning`] this is [`ProsodyTuning`], what `PlanOptions` carries.
    prosody: PlannerTuning {
        /// Where a phrase's pitch starts (semitones above the base). planner.rs:252, `PITCH_START`.
        /// An explicit `PlanOptions::pitch_start` still wins. Range from 0, the old plan's start.
        pitch_start, "pitchStart" = 5.0, [0.0, 7.5];
        /// How far the pitch falls over a phrase (semitones). planner.rs:253, `PITCH_FALL`;
        /// `PlanOptions::pitch_fall` wins. Range from 1.8, the old plan's 10 %.
        pitch_fall, "pitchFall" = 6.5, [1.8, 9.75];
        /// How early the fall comes. planner.rs:258, `PITCH_CURVE` (three uses at :703). Range
        /// 0.01..6: at 0 the line divides by zero, and near 0 it is the old plan's straight line.
        pitch_curve, "pitchCurve" = 3.0, [0.01, 6.0];
        /// A stressed vowel's rise above the line (a fraction). planner.rs:255, `ACCENT`;
        /// `PlanOptions::accent` wins.
        accent, "accent" = 0.14, half;
        /// The nuclear accent over an ordinary one. planner.rs:724, `accent * 1.8`.
        nuclear_accent, "nuclearAccent" = 1.8, half;
        /// A secondary stress's accent over a primary one's. planner.rs:726, `accent * 0.5`.
        secondary_accent, "secondaryAccent" = 0.5, half;
        /// The share of a stressed vowel its accent's peak lasts. planner.rs:260, `ACCENT_PEAK`
        /// (used at :748). Range to 1.0: the whole vowel.
        accent_peak, "accentPeak" = 0.5, [0.25, 1.0];
        /// A statement's final low under the line's end (semitones). planner.rs:264, `FINAL_DROP`
        /// (used at :684). Range to 1.7, which holds the old plan's 1.61.
        final_drop, "finalDrop" = 1.0, [0.5, 1.7];
        /// /h/: how far the folds part (cm²). planner.rs:572, `one(ABDUCTION, 0.33 * k2)`; keep
        /// the `* k2`.
        h_opening, "hOpening" = 0.33, half;
        /// /h/: the opening's lead on the span (s). planner.rs:572, `t0 - 0.01`.
        h_lead, "hLead" = 0.01, half;
        /// /h/: how long before the span's end the folds close (s). planner.rs:572, `t1 - 0.01`.
        h_end_early, "hEndEarly" = 0.01, half;
        /// /h/: the opening's ramp (s). planner.rs:572, the 0.03.
        h_ramp_in, "hRampIn" = 0.03, half;
        /// /h/: the closing's ramp (s). planner.rs:572, the 0.035.
        h_ramp_out, "hRampOut" = 0.035, half;
        /// The default voice onset time (s). planner.rs:871, `DEFAULT_VOT` (three uses at :594-595).
        vot, "vot" = 0.05, half;
        /// The share of the voice onset time the folds stay apart past the release. planner.rs:594
        /// `DEFAULT_VOT / rate * 0.4`, :595 `0.4 * DEFAULT_VOT + v - DEFAULT_VOT` and the second
        /// factor of :596 `0.012 * 0.4`: one field, three places.
        vot_fold_share, "votFoldShare" = 0.4, half;
        /// An unaspirated stop's voice onset time (s). planner.rs:596, the first factor of
        /// `0.012 * 0.4`: the product stays two factors, this times `vot_fold_share`.
        unaspirated_vot, "unaspiratedVot" = 0.012, half;
        /// Where in a voiceless stop's span its folds start to part. planner.rs:598,
        /// `t0 + 0.25 * dur`.
        stop_folds_open_at, "stopFoldsOpenAt" = 0.25, half;
        /// A voiceless stop's folds: the opening's ramp (s). planner.rs:598, the 0.035.
        stop_ramp_in, "stopRampIn" = 0.035, half;
        /// An aspirated stop's folds: the closing's ramp (s). planner.rs:598,
        /// `if aspirated { 0.045 } else { 0.02 }`.
        aspirated_ramp_out, "aspiratedRampOut" = 0.045, half;
        /// The same expression's 0.02.
        unaspirated_ramp_out, "unaspiratedRampOut" = 0.02, half;
        /// An aspirated stop's fold opening (cm²). planner.rs:598,
        /// `if aspirated { 0.36 } else { 0.22 }`.
        aspirated_opening, "aspiratedOpening" = 0.36, half;
        /// The same expression's 0.22.
        unaspirated_opening, "unaspiratedOpening" = 0.22, half;
        /// A voiceless fricative's fold opening (cm²). planner.rs:602, `one(ABDUCTION, 0.3)`.
        voiceless_opening, "voicelessOpening" = 0.3, half;
        /// Its lead on the span (s). planner.rs:602, `t0 - 0.015`.
        voiceless_lead, "voicelessLead" = 0.015, half;
        /// Its lag past the span (s). planner.rs:602, `t1 + 0.005`.
        voiceless_lag, "voicelessLag" = 0.005, half;
        /// Its opening ramp (s). planner.rs:602, the first 0.03.
        voiceless_ramp_in, "voicelessRampIn" = 0.03, half;
        /// Its closing ramp (s). planner.rs:602, the second 0.03.
        voiceless_ramp_out, "voicelessRampOut" = 0.03, half;
        /// A voiced fricative's fold opening (cm²). planner.rs:48, `VOICED_FRICATIVE_OPENING`
        /// (used at :607 and :612).
        voiced_fricative_opening, "voicedFricativeOpening" = 0.09, half;
        /// /ð/'s after a nasal (cm²). planner.rs:51, `DH_OPENING` (used at :612).
        dh_opening, "dhOpening" = 0.035, half;
        /// A voiced fricative's lead on the span (s). planner.rs:613, `t0 - 0.015` (the stop-like
        /// /ð/'s at :607 is kept literal).
        voiced_fricative_lead, "voicedFricativeLead" = 0.015, half;
        /// Its opening ramp (s). planner.rs:613, the first 0.03.
        voiced_fricative_ramp_in, "voicedFricativeRampIn" = 0.03, half;
        /// Its closing ramp (s). planner.rs:613, the second 0.03.
        voiced_fricative_ramp_out, "voicedFricativeRampOut" = 0.03, half;
        /// The share of an affricate's span that is closure. planner.rs:770, `AFFRICATE_CLOSURE`
        /// (used at :542 and :618).
        affricate_closure, "affricateClosure" = 0.45, half;
        /// Where in a voiceless affricate's span its folds start to part. planner.rs:583,
        /// `t0 + 0.15 * dur`.
        affricate_folds_open_at, "affricateFoldsOpenAt" = 0.15, half;
        /// An affricate's fold opening (cm²). planner.rs:777, `one(ABDUCTION, 0.36)` in
        /// `affricate_folds`.
        affricate_opening, "affricateOpening" = 0.36, half;
        /// Its opening ramp (s). planner.rs:777, the 0.035.
        affricate_ramp_in, "affricateRampIn" = 0.035, half;
        /// How long its folds take to close (s). planner.rs:776, all three 0.03s of
        /// `if t1 - open > 0.03 { (t1 - 0.03, 0.03) } else { (open, t1 - open) }`.
        affricate_closing, "affricateClosing" = 0.03, half;
    } [
        /// The author's pitch line instead of Plint's (P2, `442f784`): what `Tuning::upstream()`
        /// sets. Not in [`Tuning::fields`]: a fitter never moves it. When true the planner's
        /// intonation is its code as at `01755b5`: a straight fall in Hz from the base pitch,
        /// `f0 * (1 - 0.1 * prog)` with `prog = (t0 - o.t0) / total` unclamped; a stressed vowel's
        /// accent as a factor on that (`1 + accent`, `1 + accent * 1.8` nuclear, `1 + accent *
        /// 0.5` secondary) held to 30 ms before the next vowel; one F0 gesture a vowel, of weight
        /// 1 with 0.06 ramps (an explicit `f0` keeps its weight 6 and its own ramps); no pitch
        /// gesture before the first vowel; a question's `f0 * 1.35`; and a statement's final low
        /// at `f0 * 0.82`, after a nuclear peak over the first 0.35 of a last vowel longer than
        /// 80 ms. `pitch_start`, `pitch_fall`, `pitch_curve`, `accent_peak` and `final_drop` are
        /// then read by nothing, in the tuning and in `PlanOptions`; `accent` still is. Proof for
        /// T1.3: the golden planning cases planned with `Tuning::upstream()` equal
        /// `git show 01755b5:apps/chimera/crates/chimera/tests/golden/planning.json`.
        upstream_line = false;
    ]

    /// Klatt's context factors (phonetics/durations.rs); the table of inherent and minimum
    /// durations is not here. Each is used as `p *= field` unless its comment says otherwise.
    durations: DurationsTuning {
        /// Clause-final lengthening of a vowel. durations.rs:154, `p *= 1.4`.
        vowel_final, "vowelFinal" = 1.4, half;
        /// Non-phrase-final shortening. durations.rs:156, `p *= 0.6`.
        vowel_non_final, "vowelNonFinal" = 0.6, half;
        /// Non-word-final shortening. durations.rs:159, `p *= 0.85`.
        vowel_non_word_final, "vowelNonWordFinal" = 0.85, half;
        /// Polysyllabic shortening. durations.rs:162, `p *= 0.8`.
        vowel_polysyllabic, "vowelPolysyllabic" = 0.8, half;
        /// An unstressed vowel's minimum. durations.rs:165, `min *= 0.5`.
        unstressed_min, "unstressedMin" = 0.5, half;
        /// An unstressed vowel's shortening. durations.rs:166, `p *= 0.7`.
        unstressed, "unstressed" = 0.7, half;
        /// A vowel before a voiced fricative. durations.rs:174, `if voiced { 1.6 } else { 1.0 }`
        /// (the 1.0 is kept literal).
        before_voiced_fricative, "beforeVoicedFricative" = 1.6, half;
        /// A vowel before a voiced stop or affricate. durations.rs:176,
        /// `if voiced { 1.2 } else { 0.7 }`.
        before_voiced_stop, "beforeVoicedStop" = 1.2, half;
        /// The same expression's 0.7, before a voiceless one.
        before_voiceless_stop, "beforeVoicelessStop" = 0.7, half;
        /// A vowel before a nasal. durations.rs:178, `f = 0.85`.
        before_nasal, "beforeNasal" = 0.85, half;
        /// What a non-final vowel keeps whatever follows. durations.rs:180,
        /// `if final_syl { f } else { 0.7 + 0.3 * f }`: the 0.7.
        non_final_cue_base, "nonFinalCueBase" = 0.7, half;
        /// The same expression's 0.3, the share of the following consonant's cue it takes.
        non_final_cue_share, "nonFinalCueShare" = 0.3, half;
        /// A vowel before a vowel. durations.rs:184, `p *= 1.2`.
        vowel_before_vowel, "vowelBeforeVowel" = 1.2, half;
        /// A vowel after a vowel. durations.rs:187, `p *= 0.7`.
        vowel_after_vowel, "vowelAfterVowel" = 0.7, half;
        /// What a stressed vowel gets back after a voiceless stop's aspiration (ms).
        /// durations.rs:193, `d += 20.0`.
        aspiration_give_back, "aspirationGiveBack" = 20.0, half;
        /// A nasal, lateral or approximant after the phrase's last vowel. durations.rs:200,
        /// `p *= 1.4`.
        final_sonorant, "finalSonorant" = 1.4, half;
        /// A consonant not at its word's start. durations.rs:203, `p *= 0.85`.
        consonant_non_initial, "consonantNonInitial" = 0.85, half;
        /// A consonant between two consonants. durations.rs:208, `p *= 0.5`.
        cluster_both, "clusterBoth" = 0.5, half;
        /// A consonant next to one consonant. durations.rs:210, `p *= 0.7`.
        cluster_one, "clusterOne" = 0.7, half;
    }

    /// Each articulator's natural frequency (Hz), in `ArticKey::ALL` order: the third number of
    /// its row in `ARTIC_SPECS` (control/articulators.rs:27-45), used at :80 as `2.0 * PI * s.hz`.
    /// The ranges stay with `ARTIC_SPECS`.
    artic: ArticTuning {
        /// articulators.rs:27
        jaw, "jaw" = 10.0, half;
        /// articulators.rs:28
        tb_x, "tbX" = 12.0, half;
        /// articulators.rs:29
        tb_y, "tbY" = 12.0, half;
        /// articulators.rs:30
        tt_x, "ttX" = 13.0, half;
        /// articulators.rs:31
        tt_y, "ttY" = 13.0, half;
        /// articulators.rs:32
        tt_curl, "ttCurl" = 9.0, half;
        /// articulators.rs:33
        root, "root" = 9.0, half;
        /// articulators.rs:34
        lip_ap, "lipAp" = 11.0, half;
        /// articulators.rs:35
        lip_prot, "lipProt" = 11.0, half;
        /// articulators.rs:36
        lip_width, "lipWidth" = 11.0, half;
        /// articulators.rs:37
        labiodental, "labiodental" = 10.0, half;
        /// articulators.rs:40
        velum, "velum" = 12.0, half;
        /// articulators.rs:41
        larynx, "larynx" = 8.0, half;
        /// articulators.rs:42
        epilarynx, "epilarynx" = 6.0, half;
        /// articulators.rs:43
        ventricular, "ventricular" = 6.0, half;
        /// articulators.rs:44
        lateral, "lateral" = 10.0, half;
        /// articulators.rs:45
        dorsum, "dorsum" = 9.0, half;
    }

    /// Losses and turbulence onset the human airway takes for itself: `WALL_SOFT` and `RE_CRIT`
    /// (physics.rs, which keep their consts for the creatures) and the network's own
    /// (airway/network.rs).
    physics: PhysicsTuning {
        /// Soft wall mass per area (g/cm²). physics.rs:28, `WALL_SOFT.mass` (the pharynx, mouth
        /// and piriform tubes, human_airway.rs:123, 124, 146).
        wall_mass, "wallMass" = 1.5, half;
        /// Soft wall resistance per area (dyn·s/cm³). physics.rs:28, `WALL_SOFT.resistance`.
        wall_resistance, "wallResistance" = 1600.0, half;
        /// Soft wall stiffness per area (dyn/cm³). physics.rs:28, `WALL_SOFT.stiffness`.
        wall_stiffness, "wallStiffness" = 3.0e5, half;
        /// Reynolds number at which turbulence starts. physics.rs:35, `RE_CRIT` (four uses in
        /// network.rs:1158-1168; the network gets a field for it).
        re_crit, "reCrit" = 1800.0, half;
        /// Reynolds number toward which the noise's growth levels off. network.rs:1158, both
        /// factors of `12000.0 * 12000.0 - RE_CRIT * RE_CRIT`.
        re_saturation, "reSaturation" = 12000.0, half;
        /// Scale on every viscous loss. network.rs:433, `viscous_scale: 1.0` (a pub field).
        viscous_scale, "viscousScale" = 1.0, half;
        /// The frequency at which the viscous loss is taken (Hz). network.rs:434,
        /// `viscous_omega: 2.0 * PI * 700.0`.
        viscous_hz, "viscousHz" = 700.0, half;
        /// Scale on the radiation load's inertance at the lips and nostrils. The code has no such
        /// factor: multiply the whole of network.rs:1125, `lr = (8.0 * RHO) / (3.0 * PI * sqrt(PI
        /// * a))`, by it, last (`x * 1.0` is `x`, to the bit).
        radiation_inertance, "radiationInertance" = 1.0, half;
        /// Scale on the radiation load's resistance. As above, on network.rs:1126,
        /// `rr = (128.0 * RHO * C_SOUND) / (9.0 * PI * PI * a)`.
        radiation_resistance, "radiationResistance" = 1.0, half;
    }
}

/// The whole tuning. See the module's comment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct Tuning {
    pub source: SourceTuning,
    pub airway: AirwayTuning,
    pub noise: NoiseTuning,
    pub prosody: PlannerTuning,
    pub durations: DurationsTuning,
    pub artic: ArticTuning,
    pub physics: PhysicsTuning,
}

/// What the planner needs of a [`Tuning`], for `PlanOptions::tuning`: the two members `prosody`
/// and `durations`, under the same names and so the same JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct ProsodyTuning {
    pub prosody: PlannerTuning,
    pub durations: DurationsTuning,
}

/// The old plan's straight 10 % fall, in semitones: 12·log2(1/0.9).
const UPSTREAM_PITCH_FALL: f64 = 1.8240371213406008;
/// The old plan's final low, 0.82 of the base pitch, under that line's end: 12·log2(0.9/0.82).
const UPSTREAM_FINAL_DROP: f64 = 1.6116131005390937;

impl Tuning {
    /// One row per field, grouped as the struct is.
    pub fn fields() -> &'static [Field] {
        FIELDS
    }

    /// The field at a JSON path (`"source.oqBase"`).
    pub fn field(path: &str) -> Option<&'static Field> {
        FIELDS.iter().find(|f| f.path == path)
    }

    /// The author's voice at upstream `2c96403`: the default but for the pitch line Plint's P2
    /// (`442f784`) changed, which `prosody.upstream_line` puts back as code. The five value
    /// fields set beside it are the nearest the values alone come to that line (a fall nearly
    /// straight in semitones, at most 2 cents off the old one in Hz, the accent's peak lasting
    /// the vowel, the final low at 0.82 of the base pitch); while the switch is on nothing reads
    /// them, and they are what is left if someone turns it off.
    pub fn upstream() -> Tuning {
        let mut t = Tuning::default();
        t.prosody.upstream_line = true;
        t.prosody.pitch_start = 0.0;
        t.prosody.pitch_fall = UPSTREAM_PITCH_FALL;
        t.prosody.pitch_curve = 0.01;
        t.prosody.accent_peak = 1.0;
        t.prosody.final_drop = UPSTREAM_FINAL_DROP;
        t
    }

    /// The planner's part.
    pub fn plan(&self) -> ProsodyTuning {
        ProsodyTuning { prosody: self.prosody, durations: self.durations }
    }
}

impl ArticTuning {
    /// The natural frequencies (Hz) in `ArticKey::ALL` order.
    pub fn hz(&self) -> [f64; N_ARTIC] {
        [self.jaw, self.tb_x, self.tb_y, self.tt_x, self.tt_y, self.tt_curl, self.root, self.lip_ap, self.lip_prot, self.lip_width, self.labiodental, self.velum, self.larynx, self.epilarynx, self.ventricular, self.lateral, self.dorsum]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn at<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
        path.split('.').try_fold(v, |v, key| v.get(key))
    }

    fn leaves(v: &Value, prefix: &str, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    leaves(x, &p, out);
                }
            }
            _ => out.push(prefix.to_string()),
        }
    }

    #[test]
    fn every_default_lies_inside_its_bounds() {
        let d = Tuning::default();
        for f in Tuning::fields() {
            let x = (f.get)(&d);
            assert!(f.lo < f.hi, "{}: {} is not below {}", f.path, f.lo, f.hi);
            assert!(f.lo <= x && x <= f.hi, "{}: default {} outside {}..{}", f.path, x, f.lo, f.hi);
            assert!(f.lo.is_finite() && f.hi.is_finite(), "{}", f.path);
        }
    }

    #[test]
    fn upstream_lies_inside_the_bounds() {
        let u = Tuning::upstream();
        for f in Tuning::fields() {
            let x = (f.get)(&u);
            assert!(f.lo <= x && x <= f.hi, "{}: upstream {} outside {}..{}", f.path, x, f.lo, f.hi);
        }
    }

    #[test]
    fn set_then_get_gives_the_value_back_and_moves_nothing_else() {
        let d = Tuning::default();
        for (i, f) in Tuning::fields().iter().enumerate() {
            let x = 12345.5 + i as f64;
            let mut t = d;
            (f.set)(&mut t, x);
            assert_eq!((f.get)(&t), x, "{}", f.path);
            for g in Tuning::fields() {
                if g.path != f.path {
                    assert_eq!((g.get)(&t), (g.get)(&d), "setting {} moved {}", f.path, g.path);
                }
            }
        }
    }

    #[test]
    fn paths_are_unique_and_are_the_json_paths() {
        let fields = Tuning::fields();
        for (i, f) in fields.iter().enumerate() {
            assert!(fields[..i].iter().all(|g| g.path != f.path), "{} twice", f.path);
            assert_eq!(Tuning::field(f.path).map(|g| g.path), Some(f.path));
            // a value no other field has, so a path naming the wrong field shows
            let x = 12345.5 + i as f64;
            let mut t = Tuning::default();
            (f.set)(&mut t, x);
            let v = serde_json::to_value(t).unwrap();
            assert_eq!(at(&v, f.path).and_then(Value::as_f64), Some(x), "{} is not that field's JSON path", f.path);
        }
        // and every number in the JSON is a field; a switch is a bool and in no row
        let v = serde_json::to_value(Tuning::default()).unwrap();
        let mut all = Vec::new();
        leaves(&v, "", &mut all);
        let switches: Vec<&String> = all.iter().filter(|p| at(&v, p).unwrap().is_boolean()).collect();
        assert_eq!(switches, ["prosody.upstreamLine"]);
        assert!(switches.iter().all(|p| Tuning::field(p).is_none()));
        all.retain(|p| !at(&v, p).unwrap().is_boolean());
        assert_eq!(all.len(), fields.len());
        for p in &all {
            assert!(at(&v, p).unwrap().is_number(), "{p} is not a number");
            let f = Tuning::field(p).unwrap_or_else(|| panic!("{p} is in the JSON and not in fields()"));
            assert_eq!(at(&v, p).and_then(Value::as_f64), Some((f.get)(&Tuning::default())), "{p}");
        }
    }

    #[test]
    fn json_goes_round_and_a_partial_object_leaves_the_rest_default() {
        let d = Tuning::default();
        let back: Tuning = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(back, d);
        let u: Tuning = serde_json::from_str(&serde_json::to_string(&Tuning::upstream()).unwrap()).unwrap();
        assert_eq!(u, Tuning::upstream());

        let t: Tuning = serde_json::from_value(json!({"source": {"oqBase": 0.7}})).unwrap();
        assert_eq!(t.source.oq_base, 0.7);
        let mut want = d;
        want.source.oq_base = 0.7;
        assert_eq!(t, want);
        assert_eq!(serde_json::from_value::<Tuning>(json!({})).unwrap(), d);

        let p: ProsodyTuning = serde_json::from_value(json!({"durations": {"vowelFinal": 1.5}})).unwrap();
        assert_eq!(p.durations.vowel_final, 1.5);
        assert_eq!(p.prosody, d.prosody);
        // the planner's part reads the same JSON the whole does
        let whole = serde_json::to_value(d).unwrap();
        let part = serde_json::to_value(d.plan()).unwrap();
        assert_eq!(part["prosody"], whole["prosody"]);
        assert_eq!(part["durations"], whole["durations"]);
    }

    #[test]
    fn an_unknown_name_is_refused() {
        assert!(serde_json::from_value::<Tuning>(json!({"source": {"oqBaze": 0.7}})).is_err());
        assert!(serde_json::from_value::<Tuning>(json!({"sauce": {}})).is_err());
        assert!(serde_json::from_value::<Tuning>(json!({"source": {"oq_base": 0.7}})).is_err());
        assert!(serde_json::from_value::<ProsodyTuning>(json!({"source": {}})).is_err());
        assert!(serde_json::from_value::<ProsodyTuning>(json!({"prosody": {"pitchStrat": 1.0}})).is_err());
    }

    #[test]
    fn upstream_differs_from_the_default_only_in_prosody() {
        let (u, d) = (Tuning::upstream(), Tuning::default());
        assert_ne!(u, d);
        assert_ne!(u.prosody, d.prosody);
        assert!(u.prosody.upstream_line && !d.prosody.upstream_line);
        // the switch alone is a difference, and it is in the JSON both ways
        assert_ne!(Tuning { prosody: PlannerTuning { upstream_line: true, ..d.prosody }, ..d }, d);
        let on: Tuning = serde_json::from_value(json!({"prosody": {"upstreamLine": true}})).unwrap();
        assert_eq!(on, Tuning { prosody: PlannerTuning { upstream_line: true, ..d.prosody }, ..d });
        assert_eq!(serde_json::to_value(u).unwrap()["prosody"]["upstreamLine"], json!(true));
        assert!(serde_json::from_value::<Tuning>(json!({"prosody": {"upstreamLine": 1.0}})).is_err());
        assert_eq!(Tuning { prosody: d.prosody, ..u }, d);
        // the old plan's numbers: a 10 % fall from the base pitch, and 0.82 of it at the end
        let end = crate::math::pow(2.0, (u.prosody.pitch_start - u.prosody.pitch_fall) / 12.0);
        let low = crate::math::pow(2.0, (u.prosody.pitch_start - u.prosody.pitch_fall - u.prosody.final_drop) / 12.0);
        assert!((end - 0.9).abs() < 1e-12 && (low - 0.82).abs() < 1e-12, "{end} {low}");
    }

    /// The defaults against the code's own constants, where a test can reach them.
    #[test]
    fn defaults_are_the_codes_constants() {
        use crate::anatomy::human::{ArticKey, BODY_MALE};
        use crate::control::articulators::ARTIC_SPECS;
        use crate::phonetics::planner;
        use crate::physics::{CM_H2O, RE_CRIT, WALL_SOFT};
        use crate::voice::human_airway::{FRICATION_FADE_AREA, HumanAirway};
        use crate::voice::human_voice::{DEFAULT_SOURCE, HumanVoice};
        let t = Tuning::default();

        assert_eq!(t.artic.hz(), ARTIC_SPECS.map(|s| s.hz));
        let v = serde_json::to_value(t.artic).unwrap();
        for (k, name) in ArticKey::ALL.iter().zip(ArticKey::NAMES) {
            assert_eq!(v[name].as_f64(), Some(ARTIC_SPECS[k.index()].hz), "{name}");
        }

        let p = t.prosody;
        assert_eq!((p.pitch_start, p.pitch_fall, p.pitch_curve), (planner::PITCH_START, planner::PITCH_FALL, planner::PITCH_CURVE));
        assert_eq!((p.accent, p.accent_peak, p.final_drop), (planner::ACCENT, planner::ACCENT_PEAK, planner::FINAL_DROP));

        let ph = t.physics;
        assert_eq!((ph.wall_mass, ph.wall_resistance, ph.wall_stiffness), (WALL_SOFT.mass, WALL_SOFT.resistance, WALL_SOFT.stiffness));
        assert_eq!(ph.re_crit, RE_CRIT);
        assert_eq!(t.noise.frication_fade_area, FRICATION_FADE_AREA);

        let a = HumanAirway::new(96000.0, BODY_MALE);
        assert_eq!((a.noise_gain.glottis, a.noise_gain.constriction, a.noise_gain.teeth), (t.noise.glottis_gain, t.noise.constriction_gain, t.noise.teeth_gain));
        let aw = t.airway;
        assert_eq!((a.max_area_rate, a.area_rate_gain, a.lip_open_rate), (aw.max_area_rate, aw.area_rate_gain, aw.lip_open_rate));
        assert_eq!((a.seal_rate, a.seal_lo, a.seal_hi, a.seal_ease), (aw.seal_rate, aw.seal_lo, aw.seal_hi, aw.seal_ease));
        assert_eq!(a.net.viscous_scale, ph.viscous_scale);
        assert_eq!(a.net.viscous_omega, 2.0 * crate::math::PI * ph.viscous_hz);

        // the male voice's source as `apply_source` leaves it, from the fields in the code's shapes
        let voice = HumanVoice::with_defaults(48000.0);
        let (g, s, src) = (voice.glottis.params, t.source, DEFAULT_SOURCE);
        assert_eq!(g.amplitude, s.amplitude * BODY_MALE.fold_length * BODY_MALE.scale * 1.0);
        assert_eq!(g.abduction, s.abduction_breathy * src.breathiness * src.breathiness);
        assert_eq!(g.chink, s.chink_base + s.chink_breathy * src.breathiness);
        assert_eq!(g.open_quotient, s.oq_base - s.oq_tension * src.tension + BODY_MALE.open_phase);
        assert_eq!(g.open_with_abduction, s.oq_with_abduction);
        assert_eq!(g.closure_sharpness, s.closure_tension * src.tension + s.closure_pressed * 0.0 + s.closure_bright * 0.0);
        assert_eq!(g.skew, s.skew_base + s.skew_tension * src.tension + s.skew_bright * 0.0);
        assert_eq!(g.jitter, s.jitter_base + s.jitter_rough * src.roughness);
        assert_eq!(g.shimmer, s.shimmer_base + s.shimmer_rough * src.roughness);
        assert_eq!(g.effort, (src.pressure - s.effort_threshold) / s.effort_span);
        assert_eq!(voice.glottis.threshold, s.threshold_cm * CM_H2O);
        assert_eq!((s.pressure, s.drift), (src.pressure, src.drift));
        assert_eq!((voice.source.pressure, g.drift), (s.pressure, s.drift));
        assert_eq!((t.noise.aspiration_order, t.noise.teeth_order), (2.0, 2.0));
        assert!(src.tension < s.closure_pressed_from && src.roughness < s.diplophonia_from);
    }
}
