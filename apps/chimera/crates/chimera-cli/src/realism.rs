//! `chimera realism` (Plint): how far a voice is from recorded speakers on what the author's judge
//! only advises on (timbre, the source, noise, pitch and motion) and, so that a fit cannot buy
//! those with the consonants, on the obstruents' levels. No aligner and no network: a
//! render's phone times are the plan's (moved to where the phone sounds, `PLAN_TO_SOUND`), a recording's are the committed alignment's
//! (`tools/tsp/alignment.tsv.gz`), and every measure is the same code for both.
//!
//!   chimera realism reference <tsp-dir> <out.json> [--sex male] [--speakers MA,MB,…] [--jobs n]
//!   chimera realism score <reference.json> [--tuning t.json] [--speakers MH,MI,MJ] [--jobs n]
//!                         [--json out.json] [--limit n] [--times mfa:<aligned-dir>]
//!   chimera realism score <reference.json> --recordings <tsp-dir> --speakers MK [--json out.json]
//!   chimera realism score <reference.json> --renders <dir> [--times mfa:<aligned-dir>] …
//!   (--body male|female: the body that speaks, the reference's sex by default)
//!   chimera realism offsets <reference.json> --times mfa:<aligned-dir> [--tuning t.json] [--speakers …] [--jobs n]
//!   (where the aligner hears each phone of a voice against the plan: `PLAN_TO_SOUND`'s figures)
//!
//! **What is measured.** Per phone, for vowels, nasals, fricatives and liquids over the phone's
//! middle half (a plan's times are gestures, and no duration is scored: timing stays with the
//! author's judge):
//!
//! - *timbre*: per phone the mean 40-band log-mel spectrum, the voice's own long-term spectrum
//!   subtracted (the room and the microphone go with it), compared as 13 cepstral coefficients;
//!   and the level of each nasal, each liquid and the unstressed vowels re the utterance's
//!   stressed vowels.
//! - *source*: per vowel the band levels as differences between bands, H1−H2 and the cepstral
//!   peak prominence. NOT normalised by the long-term spectrum: that would remove the very tilt
//!   that is wrong.
//! - *noise*: per vowel 4-8 kHz against 2-4 kHz, and how deep the harmonics' ripple is in three
//!   bands (a harmonic spectrum has peaks and valleys an f0 apart, noise has none).
//! - *pitch*: per utterance the shape `tools/plint/f0_shape.py` defines (10th, 90th, first and
//!   last fifth, fitted fall, in semitones re the utterance's median), and the frame-to-frame step.
//! - *motion*: per utterance the envelope's modulation spectrum (2-16 Hz), and the frame-to-frame
//!   spectral change by phone class and over the whole utterance.
//! - *consonants* (since version 2; the first fit made the vowels 4-6 dB louder against every
//!   obstruent and nothing here saw it), all re the utterance's stressed vowels, and all but the
//!   closure the author's judge's own code (`compare tsp`'s `measure`):
//!   - each fricative's level /f v θ ð s z ʃ h/, over its middle half (these were timbre terms
//!     in version 1; a phone's level is one term in one group);
//!   - each affricate's level /tʃ dʒ/, over its second half (the frication);
//!   - the spectral centroid of /s z ʃ tʃ dʒ/ (1-10 kHz, in octaves), over the same windows;
//!   - each stop's burst /p t k b d g/: the loudest 5 ms from a third into its span to its end
//!     (/d/ unscored);
//!   - each stop's closure: the quietest 10 ms inside its span, at least `CLOSURE_FLOOR`
//!     (/b d g/ unscored; /p t k/ read the floor for the men and for the voice, so they cost
//!     nothing until a closure leaks).
//!
//!   The stop measures are extremes inside the span, so they hold while the span's edges move,
//!   as long as the release is inside and the next sound is not: a recording's span is the
//!   aligner's (closure, burst and aspiration), a render's is the plan's closure to
//!   `STOP_RELEASE` past the planned release (`PLAN_TO_SOUND`). Not measured: the stops' own
//!   middle-half level (the author's `level k`: with an aligner's span it is burst and
//!   aspiration, with a plan's it is the closure, 13 dB apart on /p/), and /ʒ/ (one token in 360
//!   sentences).
//!
//! **The score.** A voice is one value per term over all its sentences (a median over tokens or
//! sentences). The reference is, per term, the median over its speakers (centre) and their MAD
//! scaled to an SD, at least a floor of about one just-noticeable difference (unit). Cost =
//! max(0, |gap| / unit − 1), at most `WORST`; group = mean of its terms; score = mean of the six
//! groups. Seven terms are measured and printed under `--json` but not scored (`UNSCORED`: their
//! cost moves with where the plan puts the phone). A term a voice has too few tokens for is left out; a source, noise or pitch term it has
//! the tokens for but cannot be measured on (no voicing) costs `WORST`.
//!
//! **For other modules (the fitter).** Everything is deterministic: the same input gives the same
//! bits whatever `jobs` is (sentences are measured in parallel, summed in the order given).
//!
//! ```ignore
//! let reference = Reference::load("ref-fit6.json")?;          // or Reference::build(…)
//! let scorer = Scorer::new(&reference)?;                      // loads the G2P once
//! let all = sentences(&reference.speakers)?;                  // (id "01_03", text), by id
//! let score = scorer.score(&tuning, &all[..12], 2)?;          // renders, measures, scores; prints nothing
//! score.score; score.groups["source"]; score.terms; score.failed
//! score.level_db; score.limited                               // not scored: how loud, and the limiter's share
//! // or in one call, by sentence id:
//! let score = score_sentences(&reference, &tuning, &["01_03".into()], 2)?;
//! // or in two steps, to keep the measures: scorer.measure(&tuning, &s)? then scorer.judge(&[…])
//! ```
//!
//! A render with a non-finite sample makes the whole score `WORST` (`Score::failed`).
//!
//! **Level.** The judge's terms are all relative (to the utterance's stressed vowels, or between
//! bands), so a voice twice as loud scores the same. A score therefore also carries, unscored,
//! the voice's absolute level (`level_db`: the rms over the stressed vowels' middle halves, the
//! spans the terms are measured re, pooled over the sentences, in dB re digital full scale;
//! `speech_db`: the same over each utterance from its first phone to its last) and the share of
//! the samples on which the voice's output limiter turned the gain down (`limited`; the voice
//! counts them, since the samples themselves stay at the limiter's ceiling). The fitter holds
//! both (`fit.rs`); a recording or somebody else's render has no `limited`. A voice whose limiter
//! worked on more than 5 % of its samples is called LIMITED in the printed score: its figures
//! are of a voice the limiter has reshaped. It is not `failed`, which stays for a render with
//! nothing to measure: the first fit's voice sits on the limiter for 42 % of its samples
//! (T3.6), and a fit that starts from it needs its terms, and a cost that falls as the limiter
//! lets go, to find the way back. (The rule this replaces, "more than 5 % of the samples over
//! 0.75", never fired: the limiter holds the samples at 0.75.)

use crate::compare::tsp::{AFFRICATES, BANDS, FRAME, HOP, Phone, SIBILANTS, STOPS, Sig, Tools, Utt, VOWELS, cepstrum, db, is_vowel, load_utts, mad_sd, mean_square, measure as judge_measure, percentile};
use crate::compare::{N_CEP, N_MEL, mcd, median, read_wav};
use crate::render::{g2p, say_with, tuning_file};
use crate::{Ctx, FS};
use chimera::analysis::fft::{Fft, hann};
use chimera::phonetics::english::{EnglishG2P, arpabet_to_segments};
use chimera::phonetics::planner::expand;
use chimera::tuning::Tuning;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bump when a measure, a term, a floor or the score changes: a reference of another version is refused.
pub const VERSION: u32 = 2;
/// The scored groups, one share each.
pub const GROUPS: [&str; 6] = ["timbre", "source", "noise", "pitch", "motion", "consonants"];
/// The largest cost a term can have, the cost of a term that cannot be measured, and the whole
/// score of a render that failed.
pub const WORST: f64 = 20.0;
/// Below this many tokens (or sentences) a voice has no value for a term.
const MIN_TOKENS: usize = 3;
/// The fitting men (plan-chimera-fit, finding 22): the validation men MH MI MJ and the held-out
/// MK ML are never read for a default reference.
const FIT_MEN: [&str; 6] = ["MA", "MB", "MC", "MD", "MF", "MG"];
const FIT_WOMEN: [&str; 7] = ["FA", "FB", "FC", "FD", "FE", "FF", "FG"];
/// A pitch shape needs this many voiced 10 ms frames (as f0_shape.py).
const MIN_VOICED: usize = 30;
/// A voice whose output limiter (human_voice.rs) turned the gain down on more than this share of
/// its samples is called limited. (The samples cannot say it: the limiter holds them at its 0.75
/// ceiling.)
pub const MAX_LIMITED: f64 = 0.05;

const NASALS: [&str; 3] = ["m", "n", "ŋ"];
const FRICATIVES: [&str; 8] = ["f", "v", "θ", "ð", "s", "z", "ʃ", "h"];
const LIQUIDS: [&str; 2] = ["l", "ɹ"];
const CLASS_NAMES: [&str; 4] = ["vowels", "nasals", "fricatives", "liquids"];
/// The bands the harmonics' ripple is read in (Hz).
const RIPPLE_BANDS: [(f64, f64); 3] = [(500.0, 2000.0), (2000.0, 4000.0), (4000.0, 8000.0)];
const RIPPLE_NAMES: [&str; 3] = [".5-2k", "2-4k", "4-8k"];
/// The envelope's modulation bands (Hz).
const MOD_BANDS: [(f64, f64); 3] = [(2.0, 4.0), (4.0, 8.0), (8.0, 16.0)];
/// The cepstrum for the peak prominence: the log spectrum's bins 0..=CEP_N/2 (0-6 kHz), mirrored.
const CEP_N: usize = 512;
/// The envelope is sampled every 10 ms and transformed over this many points.
const ENV_N: usize = 1024;
/// A mel band this far under its own loudest frame of the utterance is at its floor: a recording's
/// room noise flickers there and a render's digital silence does not, and neither is the voice.
const BAND_FLOOR: f64 = 1e-4;
/// A spectrum bin this far under the frame's strongest is at the floor (the same for a render,
/// which has no floor of its own, and a recording).
const BIN_FLOOR: f64 = 1e-7;
/// How far past the planned release a planned stop's span reaches (ms; `PLAN_TO_SOUND`).
const STOP_RELEASE: f64 = 12.0;
/// A stop's closure is its quietest stretch of this length (s).
const CLOSURE_SECS: f64 = 0.010;
/// A closure quieter than this (dB re the utterance's stressed vowels) reads as this: under it a
/// recording is its room's noise (the floor sits about 47 dB under the loud level) and a render is
/// digital silence, and neither is the voice. So "silent" is one number for both.
const CLOSURE_FLOOR: f64 = -40.0;

const TSP_NOTICE: &str = "Statistics derived from the TSP speech database, version 2 (P. Kabal, McGill University), made available under the Simplified BSD licence. \
Copyright (c) 2018 Peter Kabal. All rights reserved. \
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: \
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. \
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. \
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. \
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. \
See tools/tsp/README.md.";

/// The phones with per-phone terms: vowels, nasals, fricatives, liquids, in that order.
fn cats() -> Vec<&'static str> {
    VOWELS.iter().chain(&NASALS).chain(&FRICATIVES).chain(&LIQUIDS).copied().collect()
}

/// A phone's class (an index into `CLASS_NAMES`), or none for a phone that is not measured.
fn class_of(ipa: &str) -> Option<usize> {
    if is_vowel(ipa) {
        Some(0)
    } else if NASALS.contains(&ipa) {
        Some(1)
    } else if FRICATIVES.contains(&ipa) {
        Some(2)
    } else if LIQUIDS.contains(&ipa) {
        Some(3)
    } else {
        None
    }
}

// ---- measuring one sentence of one voice

/// The analysis tables, made once and shared by the threads.
struct Kit {
    tools: Tools,
    cep: Fft,
    env: Fft,
}

impl Kit {
    fn new() -> Self {
        Kit { tools: Tools::new(FS), cep: Fft::new(CEP_N), env: Fft::new(ENV_N) }
    }
}

/// What one voice does on one measured phone.
#[derive(Clone)]
struct Tok {
    ipa: &'static str,
    class: usize,
    stressed: bool,
    /// ln of the mel band powers averaged over the middle half
    mel: Option<[f64; N_MEL]>,
    /// middle-half level (dB) re the utterance's stressed vowels, noise floor subtracted; NaN
    /// within 3 dB of the floor
    level: f64,
    /// vowels: band levels (dB re 0-8 kHz) over the middle half
    bands: [f64; 5],
    /// vowels: first harmonic minus second (dB) at the middle
    h1h2: f64,
    /// vowels: cepstral peak prominence (dB) at the middle
    cpp: f64,
    /// vowels: depth of the harmonics' ripple (dB) in `RIPPLE_BANDS` at the middle
    ripple: [f64; 3],
    /// mean spectral change (dB per 10 ms) over the middle half
    change: f64,
}

/// What one voice does on one stop, affricate or sibilant: the author's judge's own measures
/// (`compare tsp`'s `measure`, the same code) and the closure. All in dB re the utterance's
/// stressed vowels but the centroid.
#[derive(Clone)]
struct Con {
    ipa: &'static str,
    /// affricates, the judge's `level`: the second half of the span (the frication), noise floor
    /// subtracted; NaN within 3 dB of the floor. (A fricative's level is its `Tok`'s; a stop's
    /// middle half is not scored: see `STOP_RELEASE`.)
    level: f64,
    /// stops, the judge's `burst`: the loudest 5 ms from a third into the span to its end
    burst: f64,
    /// stops: the quietest `CLOSURE_SECS` inside the span, noise floor subtracted, at least `CLOSURE_FLOOR`
    closure: f64,
    /// sibilants and affricates, the judge's `centroid`: power-weighted mean frequency 1-10 kHz, here in octaves (log2 Hz)
    centroid: f64,
}

/// A stop's closure level (dB re `vref`): the quietest `CLOSURE_SECS` (every 1 ms) inside its
/// span, the signal's noise floor subtracted, never under `CLOSURE_FLOOR`. A minimum inside the
/// span does not move when the span's edges do, as long as the closure is inside.
fn closure_db(sig: &Sig, p: &Phone, vref: f64) -> f64 {
    let seg = sig.slice(p.t0, p.t1);
    let n = ((CLOSURE_SECS * sig.fs) as usize).min(seg.len());
    if n == 0 || !vref.is_finite() {
        return f64::NAN;
    }
    let step = (0.001 * sig.fs) as usize;
    let mut quiet = f64::MAX;
    let mut k = 0;
    while k + n <= seg.len() {
        quiet = quiet.min(mean_square(&seg[k..k + n]));
        k += step;
    }
    (db((quiet - sig.noise).max(0.0)) - vref).max(CLOSURE_FLOOR)
}

/// One sentence of one voice, measured.
#[derive(Clone)]
pub struct SentMeas {
    toks: Vec<Tok>,
    cons: Vec<Con>,
    /// p10, p90, range, fall, first fifth, last fifth, step; none with too few voiced frames
    pitch: Option<[f64; 7]>,
    /// the envelope's modulation (dB) in `MOD_BANDS`
    modulation: [f64; 3],
    /// spectral change over the whole utterance: its median and its top tenth
    change: [f64; 2],
    /// the stressed vowels' middle halves: the sum of the samples' squares (full scale 1) and how many
    vowel_power: (f64, usize),
    /// the same over the speech, first phone to last
    speech_power: (f64, usize),
    /// a render of ours: the samples its limiter turned down, and all its samples; none for a
    /// recording or somebody else's render
    limited: Option<(u64, usize)>,
    /// a render that went non-finite
    failed: bool,
}

impl SentMeas {
    fn failed() -> Self {
        SentMeas { toks: Vec::new(), cons: Vec::new(), pitch: None, modulation: [f64::NAN; 3], change: [f64::NAN; 2], vowel_power: (0.0, 0), speech_power: (0.0, 0), limited: None, failed: true }
    }
}

/// The pitch shape of one utterance from f0 every 10 ms (0 = unvoiced) between t0 and t1, as
/// tools/plint/f0_shape.py: in semitones re the voiced frames' median, the 10th and 90th
/// percentiles, their distance, the slope of a line through the voiced frames (per utterance,
/// first voiced frame to last), the medians of the first and the last fifth of the voiced frames;
/// and the median step between neighbouring voiced frames. None under `MIN_VOICED` voiced frames.
fn pitch_shape(f0: &[f64], t0: f64, t1: f64) -> Option<[f64; 7]> {
    let v: Vec<(usize, f64)> = f0.iter().enumerate().filter(|(i, f)| **f > 0.0 && (*i as f64 * 0.01) >= t0 && (*i as f64 * 0.01) <= t1).map(|(i, f)| (i, *f)).collect();
    if v.len() < MIN_VOICED {
        return None;
    }
    let med = median(&v.iter().map(|p| p.1).collect::<Vec<_>>());
    let st: Vec<f64> = v.iter().map(|p| 12.0 * (p.1 / med).log2()).collect();
    let n = st.len();
    let (first, last) = (v[0].0 as f64, v[n - 1].0 as f64);
    let tt: Vec<f64> = v.iter().map(|p| (p.0 as f64 - first) / (last - first)).collect();
    let (mt, ms) = (tt.iter().sum::<f64>() / n as f64, st.iter().sum::<f64>() / n as f64);
    let (mut num, mut den) = (0.0, 0.0);
    for (t, s) in tt.iter().zip(&st) {
        num += (t - mt) * (s - ms);
        den += (t - mt) * (t - mt);
    }
    let steps: Vec<f64> = (1..n).filter(|&k| v[k].0 == v[k - 1].0 + 1).map(|k| (st[k] - st[k - 1]).abs()).collect();
    let (p10, p90) = (percentile(&st, 0.1), percentile(&st, 0.9));
    Some([p10, p90, p90 - p10, num / den, median(&st[..n / 5]), median(&st[n - n / 5..]), median(&steps)])
}

/// The envelope's modulation spectrum: `env` is the amplitude every 10 ms; per band of
/// `MOD_BANDS` the mean power (dB) of the envelope's relative swing (env / mean − 1, Hann
/// window, power over the window's energy, so the length cancels). NaN under one second.
fn modulation(fft: &Fft, env: &[f64]) -> [f64; 3] {
    let n = env.len().min(ENV_N);
    let avg = env[..n].iter().sum::<f64>() / n.max(1) as f64;
    if n < 100 || !(avg > 0.0) {
        return [f64::NAN; 3];
    }
    let w = hann(n);
    let mut re = vec![0.0; ENV_N];
    let mut im = vec![0.0; ENV_N];
    for i in 0..n {
        re[i] = (env[i] / avg - 1.0) * w[i];
    }
    fft.transform(&mut re, &mut im);
    let energy: f64 = w.iter().map(|v| v * v).sum();
    let hz = 100.0 / ENV_N as f64;
    std::array::from_fn(|b| {
        let (lo, hi) = MOD_BANDS[b];
        let ks: Vec<usize> = (1..ENV_N / 2).filter(|&k| (lo..hi).contains(&(k as f64 * hz))).collect();
        db(ks.iter().map(|&k| re[k] * re[k] + im[k] * im[k]).sum::<f64>() / ks.len() as f64 / energy)
    })
}

/// The cepstral peak prominence (dB) of a power spectrum `p` (bins `bin` Hz apart, at least
/// CEP_N/2 + 1 of them): the cepstrum of the log spectrum below 6 kHz, in dB; its peak between
/// the periods of 300 and 60 Hz over a line fitted from 1 ms up (Hillenbrand's measure).
fn peak_prominence(fft: &Fft, p: &[f64], bin: f64) -> f64 {
    let half = CEP_N / 2;
    let floor = BIN_FLOOR * p.iter().copied().fold(0.0, f64::max) + 1e-30;
    let mut re = vec![0.0; CEP_N];
    let mut im = vec![0.0; CEP_N];
    for k in 0..=half {
        re[k] = db(p[k] + floor);
        if k > 0 && k < half {
            re[CEP_N - k] = re[k];
        }
    }
    fft.transform(&mut re, &mut im);
    // index q is a period of q / (CEP_N · bin) seconds
    let per = CEP_N as f64 * bin;
    let c: Vec<f64> = (0..=half).map(|q| 2.0 * db(re[q].abs() / CEP_N as f64 + 1e-12)).collect();
    let (q0, q1) = ((per / 300.0).floor() as usize, ((per / 60.0).ceil() as usize).min(half - 1));
    let start = (per / 1000.0).ceil() as usize;
    let peak = (q0..=q1).max_by(|&a, &b| c[a].total_cmp(&c[b])).unwrap();
    let n = (half - start) as f64;
    let (mq, mc) = ((start..half).map(|q| q as f64).sum::<f64>() / n, c[start..half].iter().sum::<f64>() / n);
    let (mut num, mut den) = (0.0, 0.0);
    for q in start..half {
        num += (q as f64 - mq) * (c[q] - mc);
        den += (q as f64 - mq) * (q as f64 - mq);
    }
    c[peak] - (mc + num / den * (peak as f64 - mq))
}

/// How deep the harmonics' ripple is in a band of a power spectrum: the median, over windows one
/// f0 wide, of the strongest bin over the weakest (dB). A window that wide holds one harmonic and
/// one valley of a voiced sound (25 dB and more when clean); noise alone gives about 14 dB.
fn ripple(p: &[f64], bin: f64, f0: f64, lo: f64, hi: f64) -> f64 {
    let floor = BIN_FLOOR * p.iter().copied().fold(0.0, f64::max) + 1e-30;
    let w = ((f0 / bin).round() as usize).max(2);
    let (a, b) = ((lo / bin).ceil() as usize, ((hi / bin).floor() as usize).min(p.len() - 1));
    if a + w > b {
        return f64::NAN;
    }
    let l: Vec<f64> = p[a..=b].iter().map(|v| db(v + floor)).collect();
    let depths: Vec<f64> = (0..l.len() - w)
        .map(|s| {
            let win = &l[s..=s + w];
            win.iter().copied().fold(f64::MIN, f64::max) - win.iter().copied().fold(f64::MAX, f64::min)
        })
        .collect();
    median(&depths)
}

/// Measure one sentence of one voice: `x` at 48 kHz and its phones (a plan's or an alignment's).
fn measure_sentence(kit: &Kit, x: &[f64], phones: &[Phone], female: bool) -> SentMeas {
    let tools = &kit.tools;
    let sig = Sig::new(x, FS, female);
    let mid_db = |p: &Phone| {
        let (a, b) = Sig::middle(p);
        db(mean_square(sig.slice(a, b)))
    };
    // the utterance's level: its stressed vowels (as the author's judge), else its vowels
    let mut vref = median(&phones.iter().filter(|p| is_vowel(&p.ipa) && p.stress == 1).map(mid_db).collect::<Vec<_>>());
    if !vref.is_finite() {
        vref = median(&phones.iter().filter(|p| is_vowel(&p.ipa)).map(mid_db).collect::<Vec<_>>());
    }
    let (t_first, t_last) = (phones.first().map_or(0.0, |p| p.t0), phones.last().map_or(0.0, |p| p.t1));

    // the mel spectrum every HOP over the speech, each band floored under its own loudest frame,
    // and the change between frames 10 ms apart
    let n_frames = (x.len() as f64 / FS / HOP) as usize;
    let mut pow: Vec<Option<[f64; N_MEL]>> = vec![None; n_frames];
    let mut top = [0.0f64; N_MEL];
    let mut re = vec![0.0; FRAME];
    let mut im = vec![0.0; FRAME];
    for (n, slot) in pow.iter_mut().enumerate() {
        let t = n as f64 * HOP;
        let c = (t * FS) as isize - (FRAME / 2) as isize;
        if t < t_first - 0.02 || t > t_last + 0.02 || c < 0 || c as usize + FRAME > x.len() {
            continue;
        }
        for i in 0..FRAME {
            re[i] = x[c as usize + i] * tools.win[i];
            im[i] = 0.0;
        }
        tools.fft.transform(&mut re, &mut im);
        let m: [f64; N_MEL] = std::array::from_fn(|k| tools.mel[k].iter().map(|(i, w)| (re[*i] * re[*i] + im[*i] * im[*i]) * w).sum());
        for k in 0..N_MEL {
            top[k] = top[k].max(m[k]);
        }
        *slot = Some(m);
    }
    let step = (0.01 / HOP).round() as usize;
    let change: Vec<f64> = (0..n_frames)
        .map(|n| match (pow[n], pow.get(n + step).copied().flatten()) {
            (Some(a), Some(b)) => {
                let s: f64 = (0..N_MEL)
                    .map(|k| {
                        let d = db(b[k] + BAND_FLOOR * top[k] + 1e-30) - db(a[k] + BAND_FLOOR * top[k] + 1e-30);
                        d * d
                    })
                    .sum();
                (s / N_MEL as f64).sqrt()
            }
            _ => f64::NAN,
        })
        .collect();
    // frames n whose pair (n, n + step) lies in [a, b]; the pair nearest the centre when none does
    let change_in = |a: f64, b: f64| -> f64 {
        let (i, j) = ((a / HOP).ceil().max(0.0) as usize, (b / HOP).floor() as isize - step as isize);
        let xs: Vec<f64> = if j >= i as isize {
            (i..=j as usize).filter_map(|n| change.get(n).copied()).collect()
        } else {
            let n = ((0.5 * (a + b)) / HOP).round() as isize - (step / 2) as isize;
            change.get(n.max(0) as usize).copied().into_iter().collect()
        };
        let xs: Vec<f64> = xs.into_iter().filter(|v| v.is_finite()).collect();
        if xs.is_empty() { f64::NAN } else { xs.iter().sum::<f64>() / xs.len() as f64 }
    };
    let all_change: Vec<f64> = (0..n_frames).filter(|n| (*n as f64 * HOP) >= t_first && ((*n + step) as f64 * HOP) <= t_last).map(|n| change[n]).collect();

    // the amplitude envelope every 10 ms over the speech
    let n10 = ((t_last - t_first) / 0.01).floor().max(0.0) as usize;
    let env: Vec<f64> = (0..n10).map(|j| mean_square(sig.slice(t_first + j as f64 * 0.01, t_first + (j + 1) as f64 * 0.01)).sqrt()).filter(|v| v.is_finite()).collect();

    let bin = FS / FRAME as f64;
    let long_n = tools.long.n;
    let long_bin = FS / long_n as f64;
    let all = cats();
    let mut toks = Vec::new();
    for p in phones {
        let (Some(class), Some(ipa)) = (class_of(&p.ipa), all.iter().find(|c| **c == p.ipa)) else { continue };
        let (a, b) = Sig::middle(p);
        let ms = mean_square(sig.slice(a, b));
        let level = if ms > 2.0 * sig.noise { db(ms - sig.noise) - vref } else { f64::NAN };
        let spectrum = sig.spectrum(&tools.fft, &tools.win, a, b);
        // energy re the utterance's stressed vowels
        let mel = spectrum.as_ref().filter(|_| vref.is_finite()).map(|pw| -> [f64; N_MEL] {
            std::array::from_fn(|k| (tools.mel[k].iter().map(|(i, w)| pw[*i] * w).sum::<f64>() + 1e-20).ln() - vref * std::f64::consts::LN_10 / 10.0)
        });
        let mut tok = Tok { ipa, class, stressed: p.stress == 1, mel, level, bands: [f64::NAN; 5], h1h2: f64::NAN, cpp: f64::NAN, ripple: [f64::NAN; 3], change: change_in(a, b) };
        if class == 0 {
            if let Some(pw) = &spectrum {
                let band = |lo: f64, hi: f64| pw.iter().enumerate().filter(|(k, _)| (lo..hi).contains(&(*k as f64 * bin))).map(|(_, v)| v).sum::<f64>();
                let total = band(0.0, 8000.0) + 1e-30;
                for (k, (lo, hi)) in BANDS.iter().enumerate() {
                    tok.bands[k] = db(band(*lo, *hi) / total);
                }
            }
            let t = 0.5 * (p.t0 + p.t1);
            tok.h1h2 = sig.h1h2(tools, t);
            let f0 = sig.f0.get((t / 0.01).round() as usize).copied().unwrap_or(0.0);
            let c = (t * FS) as isize - (long_n / 2) as isize;
            if f0 > 0.0 && c >= 0 && c as usize + long_n <= x.len() {
                let mut re: Vec<f64> = (0..long_n).map(|i| x[c as usize + i] * tools.long_win[i]).collect();
                let mut im = vec![0.0; long_n];
                tools.long.transform(&mut re, &mut im);
                let pw: Vec<f64> = (0..=long_n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
                tok.cpp = peak_prominence(&kit.cep, &pw, long_bin);
                for (k, (lo, hi)) in RIPPLE_BANDS.iter().enumerate() {
                    tok.ripple[k] = ripple(&pw, long_bin, f0, *lo, *hi);
                }
            }
        }
        toks.push(tok);
    }
    // stops, affricates and sibilants: the author's judge measures them, and the closure is ours
    let want: Vec<usize> = (0..phones.len()).filter(|&i| STOPS.iter().chain(&SIBILANTS).any(|c| *c == phones[i].ipa)).collect();
    let (_, theirs) = judge_measure(tools, &sig, phones, &want, female);
    let cons = want
        .iter()
        .zip(&theirs)
        .map(|(&i, t)| {
            let ipa = STOPS.iter().chain(&SIBILANTS).find(|c| **c == phones[i].ipa).expect("a wanted phone");
            let stop = STOPS.contains(ipa);
            Con { ipa, level: t.level, burst: t.burst, closure: if stop { closure_db(&sig, &phones[i], vref) } else { f64::NAN }, centroid: t.centroid.log2() }
        })
        .collect();
    // the absolute level: the samples' own squares over the spans `vref` is taken from, and over the speech
    let power = |a: f64, b: f64| {
        let seg = sig.slice(a, b);
        (seg.iter().map(|v| v * v).sum::<f64>(), seg.len())
    };
    let vowel_power = phones.iter().filter(|p| is_vowel(&p.ipa) && p.stress == 1).fold((0.0, 0), |acc, p| {
        let (a, b) = Sig::middle(p);
        let (ss, n) = power(a, b);
        (acc.0 + ss, acc.1 + n)
    });
    SentMeas {
        toks,
        cons,
        pitch: pitch_shape(&sig.f0, t_first, t_last),
        modulation: modulation(&kit.env, &env),
        change: [percentile(&all_change, 0.5), percentile(&all_change, 0.9)],
        vowel_power,
        speech_power: power(t_first, t_last),
        limited: None,
        failed: false,
    }
}

// ---- the terms

/// What a term reads.
#[derive(Clone, Copy)]
enum Take {
    /// a nasal's, a liquid's or a fricative's level (its middle half)
    Level(&'static str),
    /// an affricate's level; a stop's burst and closure; a sibilant's or an affricate's centroid
    ConLevel(&'static str),
    Burst(&'static str),
    Closure(&'static str),
    Centroid(&'static str),
    LevelUnstressed,
    /// vowels: band k re band 0 (stressed or not)
    Tilt(usize, bool),
    H1H2(bool),
    Cpp(bool),
    /// vowels: 4-8 kHz re 2-4 kHz
    High(bool),
    Ripple(usize, bool),
    Pitch(usize),
    Modulation(usize),
    /// spectral change in the middle of a class's phones
    Change(usize),
    ChangeAll(usize),
}

struct TermDef {
    name: String,
    group: &'static str,
    take: Take,
    /// the smallest unit, about a just-noticeable difference (the idea of the author's `spread_floor`)
    floor: f64,
    /// too few measurable tokens among enough tokens costs `WORST` (the voice lost its voicing)
    needs_voice: bool,
}

/// Terms measured and reported but NOT scored: on the default voice their cost with the plan's
/// phone times (moved by `PLAN_TO_SOUND`) and their cost with the aligner's times differ by more
/// than `0.5`, on the 180 validation sentences or the 360 fitting ones (2026-11-25, `95b7092`;
/// `chimera realism score … --times mfa:<aligned-dir>` against the same without `--times`). A
/// fitter chasing them would chase where the plan's window falls, not the voice. Beside each: its
/// cost with plan times → with the aligner's, validation / fitting sentences (all measured again
/// 2026-11-26 with version 2). To put one back, measure it again both ways first.
///
/// The `consonants` terms were also compared on the first fit's voice
/// (`tools/plint/fit/stage1-failed.json`, with its own aligned renders in the rig's `hill/fit1`),
/// whose source is far from the default voice the offsets were measured on; those figures are in
/// brackets. Every scored term agrees within 0.5 there too but one: `burst k` costs 1.74 with plan
/// times against 2.24 with the aligner's on the validation sentences (0.51 apart; 1.53 against
/// 1.97 on the fitting ones, and 0.00 against 0.00 on the default voice both times). The plan's
/// span reads that burst 1 dB louder, so the judge charges a quiet /k/ about a quarter too
/// little. It stays scored: the list is decided on the default voice, as before, and the term
/// errs toward leniency on the very fault it is there to catch.
const UNSCORED: [(&str, &str); 7] = [
    ("level v", "plan 1.65 → aligner 3.30 / 1.39 → 3.46: a planned /v/ is 60 ms, its frication a part of it the offsets do not find"),
    ("tilt 2-4k re 0-.5k, unstressed", "plan 2.69 → aligner 3.63 / 2.94 → 3.87: unstressed vowels are short and the window reaches their transitions"),
    ("change in vowels", "plan 0.00 → aligner 0.62 / 0.00 → 0.42: the change inside a vowel depends on how much transition the window holds"),
    ("burst d", "plan 1.10 → aligner 0.00 / 0.86 → 0.00 (the first fit's voice 1.45 → 0.00 / 0.00 → 0.66): two thirds of the /d/s stand before a consonant, the aligner ends them early, and the plan's span reads the next sound 2 to 4 dB louder"),
    ("closure b", "plan 0.00 → aligner 1.98 / 0.00 → 0.00 (the first fit's voice 6.37 → 3.25 / 0.00 → 0.00): the men's /b/ reads the floor with a unit of 1.5 dB, so a voice bar a little over the floor in one of the two spans is a cost of 2 and more"),
    ("closure d", "plan 0.00 → aligner 0.00 / 1.07 → 0.00 (the first fit's voice 0.42 → 0.28 / 1.79 → 0.97): the aligner's /d/ is often shorter than the closure and never reaches its quietest 10 ms"),
    ("closure g", "plan 0.52 → aligner 1.09 / 0.00 → 2.79 (the first fit's voice 0.00 → 1.21 / 0.00 → 1.26): as closure d; the plan's span reads the floor, the aligner's 4 to 6 dB over it"),
];

/// Whether a term counts in its group (see `UNSCORED`).
fn scored(name: &str) -> bool {
    !UNSCORED.iter().any(|u| u.0 == name)
}

/// The smallest unit of a spectrum term (dB of cepstral distance).
const SPECTRUM_FLOOR: f64 = 1.0;

fn term_defs() -> Vec<TermDef> {
    let mut d = Vec::new();
    let mut add = |name: String, group, take, floor, needs_voice| d.push(TermDef { name, group, take, floor, needs_voice });
    for c in NASALS.iter().chain(&LIQUIDS) {
        add(format!("level {c}"), "timbre", Take::Level(c), 1.5, false);
    }
    for c in &FRICATIVES {
        add(format!("level {c}"), "consonants", Take::Level(c), 1.5, false);
    }
    for c in &STOPS {
        add(format!("closure {c}"), "consonants", Take::Closure(c), 1.5, false);
        add(format!("burst {c}"), "consonants", Take::Burst(c), 1.5, false);
    }
    for c in &AFFRICATES {
        add(format!("level {c}"), "consonants", Take::ConLevel(c), 1.5, false);
    }
    for c in &SIBILANTS {
        add(format!("centroid {c}"), "consonants", Take::Centroid(c), 0.07, false);
    }
    add("level unstressed vowels".into(), "timbre", Take::LevelUnstressed, 1.5, false);
    for stressed in [true, false] {
        let s = if stressed { "stressed" } else { "unstressed" };
        for (k, b) in [(1, ".5-1k"), (2, "1-2k"), (3, "2-4k")] {
            add(format!("tilt {b} re 0-.5k, {s}"), "source", Take::Tilt(k, stressed), 1.0, true);
        }
        add(format!("H1-H2, {s}"), "source", Take::H1H2(stressed), 1.0, true);
        add(format!("peak prominence, {s}"), "source", Take::Cpp(stressed), 1.0, true);
        add(format!("vowel 4-8k re 2-4k, {s}"), "noise", Take::High(stressed), 1.0, true);
        for (k, b) in RIPPLE_NAMES.iter().enumerate() {
            add(format!("ripple {b}, {s}"), "noise", Take::Ripple(k, stressed), 1.0, true);
        }
    }
    for (k, (n, floor)) in [("p10", 0.5), ("p90", 0.5), ("range", 1.0), ("fall", 1.0), ("first fifth", 0.5), ("last fifth", 0.5), ("step", 0.02)].iter().enumerate() {
        add(format!("pitch {n}"), "pitch", Take::Pitch(k), *floor, true);
    }
    for (k, (lo, hi)) in MOD_BANDS.iter().enumerate() {
        add(format!("modulation {lo}-{hi} Hz"), "motion", Take::Modulation(k), 1.0, false);
    }
    for (k, c) in CLASS_NAMES.iter().enumerate() {
        add(format!("change in {c}"), "motion", Take::Change(k), 0.25, false);
    }
    add("change, median".into(), "motion", Take::ChangeAll(0), 0.25, false);
    add("change, top tenth".into(), "motion", Take::ChangeAll(1), 0.25, false);
    d
}

fn group_index(g: &str) -> usize {
    GROUPS.iter().position(|x| *x == g).unwrap_or(GROUPS.len())
}

/// One voice over its sentences: per term its value (NaN when it has none) and the tokens it had
/// for it, and per phone the cepstrum of its mean log-mel spectrum, its long-term spectrum subtracted.
struct Values {
    terms: BTreeMap<String, (f64, usize)>,
    ceps: BTreeMap<&'static str, [f64; N_CEP]>,
    cep_support: BTreeMap<&'static str, usize>,
    sentences: usize,
    /// see `Score`
    level_db: f64,
    speech_db: f64,
    limited: f64,
    failed: bool,
}

/// The rms (dB re full scale) of sums of squares and counts, pooled; NaN with no sample.
fn pooled_db(parts: impl Iterator<Item = (f64, usize)>) -> f64 {
    let (ss, n) = parts.fold((0.0, 0usize), |acc, p| (acc.0 + p.0, acc.1 + p.1));
    if n > 0 { db(ss / n as f64) } else { f64::NAN }
}

fn values(defs: &[TermDef], ms: &[&SentMeas]) -> Values {
    let toks: Vec<&Tok> = ms.iter().flat_map(|m| m.toks.iter()).collect();
    let cons: Vec<&Con> = ms.iter().flat_map(|m| m.cons.iter()).collect();
    let vowel = |t: &Tok, stressed: bool| t.class == 0 && t.stressed == stressed;
    let mut terms = BTreeMap::new();
    for d in defs {
        let of_toks = |f: &dyn Fn(&Tok) -> Option<f64>| -> Vec<f64> { toks.iter().filter_map(|t| f(t)).collect() };
        let of_utts = |f: &dyn Fn(&SentMeas) -> f64| -> Vec<f64> { ms.iter().map(|m| f(m)).collect() };
        let of_cons = |c: &str, f: &dyn Fn(&Con) -> f64| -> Vec<f64> { cons.iter().filter(|t| t.ipa == c).map(|t| f(t)).collect() };
        let xs: Vec<f64> = match d.take {
            Take::Level(c) => of_toks(&|t| (t.ipa == c).then_some(t.level)),
            Take::ConLevel(c) => of_cons(c, &|t| t.level),
            Take::Burst(c) => of_cons(c, &|t| t.burst),
            Take::Closure(c) => of_cons(c, &|t| t.closure),
            Take::Centroid(c) => of_cons(c, &|t| t.centroid),
            Take::LevelUnstressed => of_toks(&|t| vowel(t, false).then_some(t.level)),
            Take::Tilt(k, s) => of_toks(&|t| vowel(t, s).then_some(t.bands[k] - t.bands[0])),
            Take::H1H2(s) => of_toks(&|t| vowel(t, s).then_some(t.h1h2)),
            Take::Cpp(s) => of_toks(&|t| vowel(t, s).then_some(t.cpp)),
            Take::High(s) => of_toks(&|t| vowel(t, s).then_some(t.bands[4] - t.bands[3])),
            Take::Ripple(k, s) => of_toks(&|t| vowel(t, s).then_some(t.ripple[k])),
            Take::Pitch(k) => of_utts(&|m| m.pitch.map_or(f64::NAN, |p| p[k])),
            Take::Modulation(k) => of_utts(&|m| m.modulation[k]),
            Take::Change(c) => of_toks(&|t| (t.class == c).then_some(t.change)),
            Take::ChangeAll(k) => of_utts(&|m| m.change[k]),
        };
        let finite = xs.iter().filter(|x| x.is_finite()).count();
        terms.insert(d.name.clone(), (if finite >= MIN_TOKENS { median(&xs) } else { f64::NAN }, xs.len()));
    }
    // the long-term spectrum: the mean log-mel over every measured phone
    let mels: Vec<(&'static str, &[f64; N_MEL])> = toks.iter().filter_map(|t| t.mel.as_ref().filter(|m| m.iter().all(|v| v.is_finite())).map(|m| (t.ipa, m))).collect();
    let lts: [f64; N_MEL] = std::array::from_fn(|k| mels.iter().map(|m| m.1[k]).sum::<f64>() / mels.len().max(1) as f64);
    let mut ceps = BTreeMap::new();
    let mut cep_support = BTreeMap::new();
    for c in cats() {
        let mine: Vec<&[f64; N_MEL]> = mels.iter().filter(|m| m.0 == c).map(|m| m.1).collect();
        cep_support.insert(c, mine.len());
        if mine.len() >= MIN_TOKENS {
            let avg: [f64; N_MEL] = std::array::from_fn(|k| mine.iter().map(|m| m[k]).sum::<f64>() / mine.len() as f64 - lts[k]);
            ceps.insert(c, cepstrum(&avg));
        }
    }
    // the limiter's share of all the samples, when every sentence is a render of ours
    let limited = ms.iter().try_fold((0u64, 0usize), |acc, m| m.limited.map(|l| (acc.0 + l.0, acc.1 + l.1))).filter(|l| l.1 > 0).map_or(f64::NAN, |l| l.0 as f64 / l.1 as f64);
    Values {
        terms,
        ceps,
        cep_support,
        sentences: ms.len(),
        level_db: pooled_db(ms.iter().map(|m| m.vowel_power)),
        speech_db: pooled_db(ms.iter().map(|m| m.speech_power)),
        limited,
        failed: ms.iter().any(|m| m.failed),
    }
}

/// A term's unit: the speakers' MAD scaled to an SD, at least the floor.
fn unit(xs: &[f64], floor: f64) -> f64 {
    let m = mad_sd(xs);
    if m.is_finite() { m.max(floor) } else { floor }
}

/// max(0, |gap| / unit − 1), at most `WORST`; NaN when either is missing.
fn cost(gap: f64, unit: f64) -> f64 {
    if gap.is_finite() && unit.is_finite() && unit > 0.0 { (gap.abs() / unit - 1.0).clamp(0.0, WORST) } else { f64::NAN }
}

// ---- the reference

/// One term of a reference.
#[derive(Clone, Debug, PartialEq)]
pub struct RefTerm {
    pub group: String,
    pub centre: f64,
    pub unit: f64,
    /// speakers with a value
    pub speakers: usize,
}

/// What real speakers do, term by term: `chimera realism reference` writes it, `score` reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    pub version: u32,
    /// "male" or "female": the body the synth is scored in
    pub sex: String,
    pub speakers: Vec<String>,
    pub terms: BTreeMap<String, RefTerm>,
    /// per phone: the centre cepstrum, the unit distance (dB), the speakers with a value
    pub spectra: BTreeMap<String, ([f64; N_CEP], f64, usize)>,
    /// each speaker's own values, for the report
    pub per_speaker: BTreeMap<String, BTreeMap<String, f64>>,
}

fn mean_ceps(cs: &[&[f64; N_CEP]]) -> [f64; N_CEP] {
    std::array::from_fn(|q| cs.iter().map(|c| c[q]).sum::<f64>() / cs.len() as f64)
}

fn reference_of(sex: &str, defs: &[TermDef], speakers: &BTreeMap<String, Values>) -> Reference {
    // a term needs a value from at least half the speakers (a rare phone that two of six have is no reference)
    let need = speakers.len().div_ceil(2).max(1);
    let mut terms = BTreeMap::new();
    for d in defs {
        let xs: Vec<f64> = speakers.values().map(|v| v.terms[&d.name].0).filter(|x| x.is_finite()).collect();
        if xs.len() >= need {
            terms.insert(d.name.clone(), RefTerm { group: d.group.into(), centre: median(&xs), unit: unit(&xs, d.floor), speakers: xs.len() });
        }
    }
    let mut spectra = BTreeMap::new();
    for c in cats() {
        let mine: Vec<(&String, &[f64; N_CEP])> = speakers.iter().filter_map(|(k, v)| v.ceps.get(c).map(|cc| (k, cc))).collect();
        if mine.len() < need {
            continue;
        }
        // a speaker's distance to the centre of the others: how far apart two real voices are
        // (as the author's judge)
        let loo: Vec<f64> = if mine.len() < 2 {
            Vec::new()
        } else {
            mine.iter()
                .map(|(k, cc)| {
                    let others: Vec<&[f64; N_CEP]> = mine.iter().filter(|m| m.0 != *k).map(|m| m.1).collect();
                    mcd(cc, &mean_ceps(&others))
                })
                .collect()
        };
        let u = median(&loo);
        let all: Vec<&[f64; N_CEP]> = mine.iter().map(|m| m.1).collect();
        spectra.insert(c.to_string(), (mean_ceps(&all), if u.is_finite() { u.max(SPECTRUM_FLOOR) } else { SPECTRUM_FLOOR }, mine.len()));
    }
    let per_speaker = speakers.iter().map(|(k, v)| (k.clone(), v.terms.iter().filter(|(_, x)| x.0.is_finite()).map(|(n, x)| (n.clone(), x.0)).collect())).collect();
    Reference { version: VERSION, sex: sex.into(), speakers: speakers.keys().cloned().collect(), terms, spectra, per_speaker }
}

impl Reference {
    /// The reference of `speakers` (their recordings under `tsp`, the committed alignment), on `jobs` threads.
    pub fn build(tsp: &str, sex: &str, speakers: &[String], jobs: usize) -> Result<Reference, String> {
        if sex != "male" && sex != "female" {
            return Err(format!("--sex {sex:?}: male or female"));
        }
        let kit = Kit::new();
        let per = measure_recordings(&kit, tsp, speakers, jobs)?;
        let defs = term_defs();
        let vals: BTreeMap<String, Values> = per.iter().map(|(k, ms)| (k.clone(), values(&defs, &ms.iter().collect::<Vec<_>>()))).collect();
        Ok(reference_of(sex, &defs, &vals))
    }

    pub fn load(path: &str) -> Result<Reference, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let bad = |what: &str| format!("{path}: not a realism reference ({what})");
        let version = v["version"].as_u64().ok_or(bad("no version"))? as u32;
        if version != VERSION {
            return Err(format!("{path}: reference of version {version}, this judge is version {VERSION}: build it again"));
        }
        let sex = v["sex"].as_str().ok_or(bad("no sex"))?.to_string();
        let speakers = v["speakers"].as_array().ok_or(bad("no speakers"))?.iter().filter_map(|s| s.as_str().map(String::from)).collect();
        let mut terms = BTreeMap::new();
        for (name, t) in v["terms"].as_object().ok_or(bad("no terms"))? {
            let (Some(group), Some(centre), Some(unit)) = (t["group"].as_str(), t["centre"].as_f64(), t["unit"].as_f64()) else { return Err(bad(name)) };
            terms.insert(name.clone(), RefTerm { group: group.into(), centre, unit, speakers: t["speakers"].as_u64().unwrap_or(0) as usize });
        }
        let mut spectra = BTreeMap::new();
        for (name, t) in v["spectra"].as_object().ok_or(bad("no spectra"))? {
            let c: Vec<f64> = t["centre"].as_array().ok_or(bad(name))?.iter().filter_map(Value::as_f64).collect();
            let (Ok(centre), Some(unit)) = (<[f64; N_CEP]>::try_from(c), t["unit"].as_f64()) else { return Err(bad(name)) };
            spectra.insert(name.clone(), (centre, unit, t["speakers"].as_u64().unwrap_or(0) as usize));
        }
        let mut per_speaker = BTreeMap::new();
        for (k, o) in v["perSpeaker"].as_object().into_iter().flatten() {
            per_speaker.insert(k.clone(), o.as_object().into_iter().flatten().filter_map(|(n, x)| x.as_f64().map(|x| (n.clone(), x))).collect());
        }
        Ok(Reference { version, sex, speakers, terms, spectra, per_speaker })
    }

    pub fn to_json(&self) -> Value {
        // every object is filled in sorted order, so the bytes do not depend on serde_json's map
        let mut terms = Map::new();
        for (n, t) in &self.terms {
            terms.insert(n.clone(), json!({ "centre": t.centre, "group": t.group, "speakers": t.speakers, "unit": t.unit }));
        }
        let mut spectra = Map::new();
        for (n, (c, u, k)) in &self.spectra {
            spectra.insert(n.clone(), json!({ "centre": c.to_vec(), "speakers": k, "unit": u }));
        }
        let mut per = Map::new();
        for (k, o) in &self.per_speaker {
            let mut m = Map::new();
            for (n, x) in o {
                m.insert(n.clone(), json!(x));
            }
            per.insert(k.clone(), Value::Object(m));
        }
        let mut out = Map::new();
        out.insert("judge".into(), json!("chimera realism"));
        out.insert("notice".into(), json!(TSP_NOTICE));
        out.insert("perSpeaker".into(), Value::Object(per));
        out.insert("sex".into(), json!(self.sex));
        out.insert("speakers".into(), json!(self.speakers));
        out.insert("spectra".into(), Value::Object(spectra));
        out.insert("terms".into(), Value::Object(terms));
        out.insert("version".into(), json!(self.version));
        Value::Object(out)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, serde_json::to_string_pretty(&self.to_json()).unwrap() + "\n").map_err(|e| format!("{path}: {e}"))
    }
}

// ---- the score

/// One term of a score.
#[derive(Clone, Debug, PartialEq)]
pub struct TermScore {
    pub name: String,
    pub group: String,
    /// the voice's value (a spectrum term: its cepstral distance to the centre, dB)
    pub value: f64,
    /// the reference's centre (a spectrum term: 0) and unit
    pub centre: f64,
    pub unit: f64,
    /// NaN when the voice has too few tokens for the term: it is left out of its group
    pub cost: f64,
    /// tokens (or sentences) the voice had for it
    pub support: usize,
    /// false for a term that is reported but left out of its group (its cost moves with the
    /// plan's phone times: the list is `UNSCORED` in realism.rs)
    pub scored: bool,
}

/// A voice against a reference.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    /// the mean of the groups; `WORST` when a render failed
    pub score: f64,
    /// each of `GROUPS`: the mean cost of its terms (NaN when it has none)
    pub groups: BTreeMap<String, f64>,
    /// every term of the reference, by group (in `GROUPS`' order) and then by name
    pub terms: Vec<TermScore>,
    pub sentences: usize,
    /// Not scored. The voice's level: the rms over the stressed vowels' middle halves, pooled over
    /// the sentences (dB re digital full scale; NaN with no stressed vowel).
    pub level_db: f64,
    /// Not scored. The rms over each utterance from its first phone to its last, pooled (dB re full scale).
    pub speech_db: f64,
    /// Not scored. The share of all the renders' samples on which the voice's output limiter
    /// turned the gain down (0..1); NaN when a sentence was not rendered here (a recording).
    pub limited: f64,
    /// a render had a non-finite sample: nothing was scored
    pub failed: bool,
}

fn score_values(r: &Reference, defs: &[TermDef], v: &Values) -> Score {
    let mut terms = Vec::new();
    for (name, t) in &r.terms {
        let Some(d) = defs.iter().find(|d| d.name == *name) else { continue };
        let (value, support) = v.terms.get(name).copied().unwrap_or((f64::NAN, 0));
        let mut c = cost(value - t.centre, t.unit);
        if !value.is_finite() && d.needs_voice && support >= MIN_TOKENS {
            c = WORST;
        }
        terms.push(TermScore { name: name.clone(), group: t.group.clone(), value, centre: t.centre, unit: t.unit, cost: c, support, scored: scored(name) });
    }
    for (name, (centre, u, _)) in &r.spectra {
        let Some((c, cep)) = v.ceps.get_key_value(name.as_str()) else {
            terms.push(TermScore { name: format!("spectrum {name}"), group: "timbre".into(), value: f64::NAN, centre: 0.0, unit: *u, cost: f64::NAN, support: 0, scored: scored(&format!("spectrum {name}")) });
            continue;
        };
        let d = mcd(cep, centre);
        terms.push(TermScore { name: format!("spectrum {name}"), group: "timbre".into(), value: d, centre: 0.0, unit: *u, cost: cost(d, *u), support: v.cep_support[c], scored: scored(&format!("spectrum {name}")) });
    }
    terms.sort_by(|a, b| (group_index(&a.group), &a.name).cmp(&(group_index(&b.group), &b.name)));
    let mut groups = BTreeMap::new();
    for g in GROUPS {
        let cs: Vec<f64> = terms.iter().filter(|t| t.group == g && t.scored && t.cost.is_finite()).map(|t| t.cost).collect();
        groups.insert(g.to_string(), if v.failed { WORST } else if cs.is_empty() { f64::NAN } else { cs.iter().sum::<f64>() / cs.len() as f64 });
    }
    let have: Vec<f64> = GROUPS.iter().map(|g| groups[*g]).filter(|x| x.is_finite()).collect();
    let score = if v.failed || have.is_empty() { WORST } else { have.iter().sum::<f64>() / have.len() as f64 };
    Score { score, groups, terms, sentences: v.sentences, level_db: v.level_db, speech_db: v.speech_db, limited: v.limited, failed: v.failed }
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

impl Score {
    pub fn to_json(&self, voice: &str) -> Value {
        let mut groups = Map::new();
        for (g, x) in &self.groups {
            groups.insert(g.clone(), num(*x));
        }
        let terms: Vec<Value> = self
            .terms
            .iter()
            .map(|t| json!({ "centre": num(t.centre), "cost": num(t.cost), "group": t.group, "name": t.name, "scored": t.scored, "support": t.support, "unit": num(t.unit), "value": num(t.value) }))
            .collect();
        json!({ "failed": self.failed, "groups": groups, "levelDb": num(self.level_db), "limited": num(self.limited), "score": num(self.score), "sentences": self.sentences, "speechDb": num(self.speech_db), "terms": terms, "voice": voice })
    }

    /// The scored terms that cost most, worst first (ties in the terms' own order).
    pub fn worst(&self, n: usize) -> Vec<&TermScore> {
        let mut t: Vec<&TermScore> = self.terms.iter().filter(|t| t.scored && t.cost.is_finite()).collect();
        t.sort_by(|a, b| b.cost.total_cmp(&a.cost));
        t.truncate(n);
        t
    }
}

// ---- voices: renders and recordings

/// A sentence to say: its id without a speaker ("01_03") and its text.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sentence {
    pub id: String,
    pub text: String,
}

/// The sentences `speakers` read (from the committed corpus), by id, each once.
pub fn sentences(speakers: &[String]) -> Result<Vec<Sentence>, String> {
    let mut out: Vec<Sentence> = load_utts()?.iter().filter(|u| speakers.contains(&u.speaker)).map(|u| Sentence { id: u.sentence().to_string(), text: u.text.clone() }).collect();
    out.sort();
    out.dedup();
    Ok(out)
}

/// `f` over `items` on `jobs` threads (the pattern of the author's `say_all`), the results in the
/// items' order whatever the threads did.
pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], jobs: usize, f: impl Fn(&T) -> Result<R, String> + Sync) -> Result<Vec<R>, String> {
    let next = AtomicUsize::new(0);
    let done: Mutex<Vec<Option<Result<R, String>>>> = Mutex::new(items.iter().map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1).min(items.len().max(1)) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let r = f(item);
                    done.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for r in done.into_inner().unwrap() {
        match r.expect("every item was taken") {
            Ok(v) => out.push(v),
            Err(e) => errors.push(e),
        }
    }
    if errors.is_empty() { Ok(out) } else { Err(errors.join("\n")) }
}

/// Every recording of `speakers` measured with the alignment's phone times: per speaker, by recording id.
fn measure_recordings(kit: &Kit, tsp: &str, speakers: &[String], jobs: usize) -> Result<BTreeMap<String, Vec<SentMeas>>, String> {
    let all = load_utts()?;
    let mut utts: Vec<&Utt> = all.iter().filter(|u| speakers.contains(&u.speaker)).collect();
    utts.sort_by(|a, b| a.id.cmp(&b.id));
    for s in speakers {
        if !utts.iter().any(|u| u.speaker == *s) {
            return Err(format!("no recordings of speaker {s} in the alignment"));
        }
    }
    let ms = par_map(&utts, jobs, |u| {
        let path = format!("{tsp}/48k/{}/{}.wav", u.speaker, u.id);
        let (x, fs) = read_wav(&path)?;
        if (fs - FS).abs() > 0.5 {
            return Err(format!("{path}: {fs} Hz, expected 48 kHz"));
        }
        Ok(measure_sentence(kit, &x, &u.phones, u.female()))
    })?;
    let mut out: BTreeMap<String, Vec<SentMeas>> = BTreeMap::new();
    for (u, m) in utts.iter().zip(ms) {
        out.entry(u.speaker.clone()).or_default().push(m);
    }
    Ok(out)
}

/// The aligner's phones for a render, from `<dir>/<stem>/<stem>.json` (what the author's judge
/// leaves in `<run>/tsp/mfa/aligned`), read as his `align_synth` reads them.
fn mfa_phones(dir: &str, stem: &str) -> Result<Vec<Phone>, String> {
    let path = format!("{dir}/{stem}/{stem}.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let entries = v["tiers"]["phones"]["entries"].as_array().ok_or(format!("{path}: no phones"))?;
    let mut phones = Vec::new();
    for e in entries {
        let (t0, t1, label) = (e[0].as_f64().unwrap_or(f64::NAN), e[1].as_f64().unwrap_or(f64::NAN), e[2].as_str().unwrap_or(""));
        if matches!(label, "" | "sil" | "sp" | "<eps>") {
            continue;
        }
        let (ipa, stress) = match arpabet_to_segments(label).pop() {
            Some(s) => (s.ipa, s.stress.unwrap_or(0)),
            None => (label.to_string(), 0),
        };
        phones.push(Phone { ipa, stress, t0, t1 });
    }
    Ok(phones)
}

/// A plan's segments as the phones the G2P said: the planner writes a diphthong as its two
/// targets ("aɪ" → "a", "ɪ") and /ɚ/ as an unstressed "ɝ", and the recordings' alignment has the
/// phones themselves; the G2P's dark "ɫ" is the alignment's /l/. `said` is the G2P's phones in order; when the plan is not their expansion
/// one for one (or `said` is not known: a dumped plan), neighbouring vowel segments that spell an
/// English diphthong are joined instead.
fn plan_phones(plan: &[Phone], said: Option<&[String]>) -> Vec<Phone> {
    let mut out = joined_phones(plan, said);
    for p in &mut out {
        if p.ipa == "ɫ" {
            p.ipa = "l".into();
        }
    }
    out
}

fn joined_phones(plan: &[Phone], said: Option<&[String]>) -> Vec<Phone> {
    if let Some(said) = said {
        let mut out = Vec::new();
        let mut j = 0;
        let exact = said.iter().all(|ipa| {
            let own = [ipa.as_str()];
            let parts: &[&str] = expand(ipa).unwrap_or(&own);
            let fits = j + parts.len() <= plan.len() && parts.iter().zip(&plan[j..]).all(|(a, b)| *a == b.ipa);
            if fits {
                out.push(Phone { ipa: ipa.clone(), stress: plan[j].stress, t0: plan[j].t0, t1: plan[j + parts.len() - 1].t1 });
                j += parts.len();
            }
            fits
        });
        if exact && j == plan.len() {
            return out;
        }
    }
    let mut out: Vec<Phone> = Vec::new();
    for p in plan {
        let joined = out.last().and_then(|q| match (q.ipa.as_str(), p.ipa.as_str()) {
            ("a", "ɪ") => Some("aɪ"),
            ("a", "ʊ") => Some("aʊ"),
            ("e", "i") => Some("eɪ"),
            ("o", "u") => Some("oʊ"),
            ("ɔ", "ɪ") => Some("ɔɪ"),
            _ => None,
        });
        match joined {
            Some(ipa) => {
                let q = out.last_mut().unwrap();
                (q.ipa, q.t1) = (ipa.into(), p.t1);
            }
            None => out.push(Phone { ipa: if p.ipa == "ɝ" && p.stress != 1 { "ɚ".into() } else { p.ipa.clone() }, ..p.clone() }),
        }
    }
    out
}

/// Where a render's phone sounds against where the plan puts it: (phone, start, end) in ms added
/// to the plan's span. A plan's times are gestures: the default male voice's vowels sound 20-35 ms
/// after their gesture starts and end 5-15 ms before it ends, /ð/ sounds 22 ms late as a whole,
/// /f s θ ʃ/ start early and end late. These are the medians of the aligner's boundary less the
/// plan's over the 360 fitting sentences of the default voice at `95b7092`
/// (`chimera realism offsets ref-fit6.json --times mfa:<hill/ref442/tsp/mfa/aligned>` prints
/// them, with their spread: 10th to 90th percentile is about ±25 ms around each). Without them
/// the middle half of a planned /ð/ sat on its onset and read 9 dB low. They are fixed numbers,
/// measured once on the default voice: a tuning that moves the articulators' speeds moves the
/// truth under them, which is what the author's judge with the aligner is for at the gates.
/// What they leave: with plan times the timbre group still reads 0.09-0.10 (17-19 %) higher than
/// with the aligner's (0.636 against 0.536 on the validation sentences, 0.616 against 0.525 on
/// the fitting ones), spread thinly over a dozen spectrum and level terms, none by more than 0.4;
/// the other groups agree within 10 % or 0.05. The middle third instead of the middle half was
/// tried and changes nothing (0.675 against 0.577).
///
/// **Stops** (2026-11-26, plan-chimera-fit T3.5). A planned stop is its closure and ends at the
/// planned release, and the burst is there: the level 5 ms at a time around it, over MA's and MB's
/// sentences, is silent until −10 ms, peaks at 0 to +10 ms, and is then aspiration (about −16 dB
/// for 60 ms before a vowel) or, 10 ms on, a voiced stop's vowel. The aligner ends a stop where
/// the voice starts: 15 ms after the release for /b d g/, 20 to 30 ms in the median for /p t k/
/// but −43 … +64 ms (10th to 90th percentile, /k/) with what follows, so no one number is its
/// end, and the burst is a maximum: a span that reaches the next vowel reads the vowel. So a
/// stop's start moves by the aligner's median as the other phones' do, and its end is
/// `STOP_RELEASE` past the planned release, for all six. 12 ms was chosen among 8, 12 and the
/// aligner's medians by how the bursts then agree with the aligner's spans, on the default voice
/// and on the first fit's (`tools/plint/fit/stage1-failed.json`): with the medians burst b and d
/// read 2.5 and 2 dB high on the default voice and 5 dB high on the fit's (the next vowel's
/// onset), with 8 ms burst p reads 2.6 dB low. The affricates take the medians at both ends.
const PLAN_TO_SOUND: [(&str, f64, f64); 38] = [
    ("ɑ", 29.5, -16.9), ("æ", 31.5, -16.1), ("ʌ", 34.8, -8.4), ("ə", 20.0, -0.1), ("ɔ", 27.1, -29.2), ("aʊ", 21.3, -15.3), ("aɪ", 31.1, -14.4), ("ɛ", 25.6, -7.1),
    ("ɝ", 23.7, -7.9), ("ɚ", 21.8, 3.8), ("eɪ", 26.1, -11.4), ("ɪ", 21.5, -5.0), ("i", 24.6, -5.9), ("oʊ", 24.8, -14.0), ("ɔɪ", 21.3, -10.4), ("ʊ", 30.0, -4.0), ("u", 49.8, -3.0),
    ("m", -2.8, 13.6), ("n", -9.2, 12.7), ("ŋ", -6.4, 10.9),
    ("f", -24.9, 31.0), ("v", 11.8, 13.9), ("θ", -29.0, 32.5), ("ð", 24.6, 20.0), ("s", -16.5, 22.3), ("z", 8.4, 16.1), ("ʃ", -6.6, 33.8), ("h", 13.0, 25.4),
    ("l", 7.9, 33.4), ("ɹ", 9.6, 9.2),
    ("p", 1.1, STOP_RELEASE), ("t", -3.6, STOP_RELEASE), ("k", -2.0, STOP_RELEASE), ("b", 13.6, STOP_RELEASE), ("d", 16.4, STOP_RELEASE), ("g", 2.7, STOP_RELEASE),
    ("tʃ", -5.9, 7.2), ("dʒ", 1.4, 6.8),
];

/// A plan's phones moved to where they sound (`PLAN_TO_SOUND`); a phone never shrinks under 10 ms.
fn sound_times(plan: &[Phone]) -> Vec<Phone> {
    plan.iter()
        .map(|p| match PLAN_TO_SOUND.iter().find(|o| o.0 == p.ipa) {
            Some((_, a, b)) => {
                let (t0, t1) = (p.t0 + a / 1000.0, p.t1 + b / 1000.0);
                if t1 - t0 >= 0.01 { Phone { t0, t1, ..p.clone() } } else { Phone { t0: 0.5 * (t0 + t1) - 0.005, t1: 0.5 * (t0 + t1) + 0.005, ..p.clone() } }
            }
            None => p.clone(),
        })
        .collect()
}

/// Renders and scores against one reference. Holds the G2P and the analysis tables, so the
/// fitter makes one and calls `score` for every candidate.
pub struct Scorer<'a> {
    reference: &'a Reference,
    /// the body that speaks: the reference's sex unless `in_body` says otherwise
    body: String,
    g2p: EnglishG2P,
    kit: Kit,
    defs: Vec<TermDef>,
}

impl<'a> Scorer<'a> {
    pub fn new(reference: &'a Reference) -> Result<Self, String> {
        Ok(Scorer { reference, body: reference.sex.clone(), g2p: g2p()?, kit: Kit::new(), defs: term_defs() })
    }

    /// Score another body's voice against this reference (the judge's own check: a man must read
    /// far from the women).
    pub fn in_body(mut self, body: &str) -> Result<Self, String> {
        crate::body(body)?;
        self.body = body.into();
        Ok(self)
    }

    fn female(&self) -> bool {
        self.body == "female"
    }

    /// Say one sentence with `tuning` in the reference's body and measure it with the plan's phone
    /// times, or with `times` when given. A render that went wrong is a failed measure, not an error.
    pub fn measure(&self, tuning: &Tuning, s: &Sentence) -> Result<SentMeas, String> {
        self.measure_at(tuning, s, None)
    }

    /// Say one sentence: its samples, the plan's phones (as the phones that were said) and how
    /// many samples the voice's limiter turned down.
    fn say(&self, tuning: &Tuning, s: &Sentence) -> Result<(Vec<f64>, Vec<Phone>, u64), String> {
        let sp = say_with(&self.g2p, &s.text, &self.body, tuning).map_err(|e| format!("{}: {e}", s.id))?;
        let plan: Vec<Phone> = sp.plan.segments.iter().map(|p| Phone { ipa: p.ipa.clone(), stress: p.stress.unwrap_or(0), t0: p.t0, t1: p.t1 }).collect();
        let said: Vec<String> = self.g2p.to_phrases(&s.text).iter().flat_map(|p| p.segments.iter().map(|g| g.ipa.clone())).collect();
        Ok((sp.out.iter().map(|&v| v as f64).collect(), plan_phones(&plan, Some(&said)), sp.limited))
    }

    fn measure_at(&self, tuning: &Tuning, s: &Sentence, times: Option<&[Phone]>) -> Result<SentMeas, String> {
        let (x, plan, limited) = self.say(tuning, s)?;
        let mut m = self.measure_samples(&x, times.unwrap_or(&sound_times(&plan)));
        m.limited = Some((limited, x.len()));
        Ok(m)
    }

    /// Measure samples somebody else rendered (48 kHz) with their phone times (whether their
    /// limiter worked cannot be read off the samples). A non-finite sample is a failed measure.
    fn measure_samples(&self, x: &[f64], phones: &[Phone]) -> SentMeas {
        if x.iter().any(|v| !v.is_finite()) {
            return SentMeas::failed();
        }
        measure_sentence(&self.kit, x, phones, self.female())
    }

    /// The score of measured sentences, taken together as one voice, in the order given.
    pub fn judge(&self, measured: &[SentMeas]) -> Score {
        score_values(self.reference, &self.defs, &values(&self.defs, &measured.iter().collect::<Vec<_>>()))
    }

    /// Render `sentences` with `tuning` on `jobs` threads and score them. Prints nothing; the same
    /// input gives the same score to the bit whatever `jobs` is.
    pub fn score(&self, tuning: &Tuning, sentences: &[Sentence], jobs: usize) -> Result<Score, String> {
        Ok(self.judge(&par_map(sentences, jobs, |s| self.measure(tuning, s))?))
    }
}

/// Render the sentences with these ids (as `sentences` gives them: "01_03") with `tuning` and
/// score them against `reference`, without printing. For many calls make a [`Scorer`] once.
#[allow(dead_code)] // the fitter's entry (T3.2)
pub fn score_sentences(reference: &Reference, tuning: &Tuning, ids: &[String], jobs: usize) -> Result<Score, String> {
    let corpus: BTreeMap<String, String> = load_utts()?.iter().filter(|u| u.female() == (reference.sex == "female")).map(|u| (u.sentence().to_string(), u.text.clone())).collect();
    let mut want = Vec::new();
    for id in ids {
        let text = corpus.get(id).ok_or(format!("no sentence {id:?} in tools/tsp/sentences.tsv"))?;
        want.push(Sentence { id: id.clone(), text: text.clone() });
    }
    Scorer::new(reference)?.score(tuning, &want, jobs)
}

// ---- the command

fn print_score(voice: &str, s: &Score, worst: usize) {
    let fx = |x: f64| if x.is_finite() { format!("{x:.3}") } else { "-".into() };
    println!("{voice}: realism {}  ({} sentences{})", fx(s.score), s.sentences, if s.failed { ", a render FAILED" } else { "" });
    println!("  {}", GROUPS.iter().map(|g| format!("{g} {}", fx(s.groups[*g]))).collect::<Vec<_>>().join("  "));
    let db = |x: f64| if x.is_finite() { format!("{x:.2} dBFS") } else { "-".into() };
    let limited = if s.limited.is_finite() { format!("{:.3} % of samples{}", 100.0 * s.limited, if s.limited > MAX_LIMITED { " (LIMITED: over 5 %, the figures are of a voice the limiter has reshaped)" } else { "" }) } else { "- (not rendered here)".into() };
    println!("  not scored: level {} over stressed vowels, {} over the speech; limiter on {limited}", db(s.level_db), db(s.speech_db));
    if worst > 0 {
        println!("  worst terms (cost: value against centre ± unit):");
        for t in s.worst(worst) {
            println!("    {:6.2}  {:<8}{:<34} {:8.2} against {:7.2} ± {:.2}", t.cost, t.group, t.name, t.value, t.centre, t.unit);
        }
    }
}

fn split(list: &str) -> Vec<String> {
    list.split(',').filter(|s| !s.is_empty()).map(String::from).collect()
}

/// `chimera realism reference|score …` (the head of this file has the arguments)
pub fn run(_ctx: &Ctx, a: &[String]) -> Result<(), String> {
    const USAGE: &str = "usage: chimera realism reference <tsp-dir> <out.json> [--sex male] [--speakers MA,MB,…] [--jobs n]\n       chimera realism score <reference.json> [--tuning t.json] [--speakers MH,MI,MJ] [--jobs n] [--json out.json] [--limit n]\n                             [--recordings <tsp-dir> | --renders <dir>] [--times mfa:<aligned-dir>] [--body male]\n       chimera realism offsets <reference.json> --times mfa:<aligned-dir> [--tuning t.json] [--speakers MA,…] [--jobs n]";
    let mut pos: Vec<&str> = Vec::new();
    let mut jobs = std::thread::available_parallelism().map_or(2, |n| n.get());
    let (mut sex, mut speakers, mut tuning, mut json_out, mut limit, mut recordings, mut renders, mut times, mut body) = (None, None, None, None, None, None, None, None, None);
    let mut it = a.iter().skip(1);
    while let Some(x) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match x.as_str() {
            "--jobs" => jobs = val("--jobs")?.parse().map_err(|e| format!("--jobs: {e}"))?,
            "--sex" => sex = Some(val("--sex")?),
            "--speakers" => speakers = Some(split(&val("--speakers")?)),
            "--tuning" => tuning = Some(val("--tuning")?),
            "--json" => json_out = Some(val("--json")?),
            "--limit" => limit = Some(val("--limit")?.parse::<usize>().map_err(|e| format!("--limit: {e}"))?),
            "--recordings" => recordings = Some(val("--recordings")?),
            "--renders" => renders = Some(val("--renders")?),
            "--times" => times = Some(val("--times")?),
            "--body" => body = Some(val("--body")?),
            f if f.starts_with("--") => return Err(format!("unknown option {f}\n{USAGE}")),
            _ => pos.push(x),
        }
    }
    match (a.first().map(String::as_str), pos.as_slice()) {
        (Some("reference"), [tsp, out]) => {
            let sex = sex.unwrap_or("male".into());
            let speakers = speakers.unwrap_or_else(|| if sex == "female" { FIT_WOMEN.iter().map(|s| s.to_string()).collect() } else { FIT_MEN.iter().map(|s| s.to_string()).collect() });
            let r = Reference::build(tsp, &sex, &speakers, jobs)?;
            r.save(out)?;
            println!("reference of {} ({sex}): {} terms and {} spectra → {out}", r.speakers.join(" "), r.terms.len(), r.spectra.len());
            Ok(())
        }
        (Some("offsets"), [reference]) => {
            // where the aligner hears each phone of the default voice against where the plan puts
            // it: the figures behind the plan-time notes in this file
            let r = Reference::load(reference)?;
            let scorer = Scorer::new(&r)?;
            let dir = times.as_deref().and_then(|t| t.strip_prefix("mfa:")).ok_or("offsets needs --times mfa:<aligned-dir>")?;
            let mut want = sentences(&speakers.unwrap_or_else(|| r.speakers.clone()))?;
            want.truncate(limit.unwrap_or(usize::MAX));
            // the plan is the default's unless the tuning moves the timing; the aligned renders must be that tuning's
            let t = match &tuning {
                Some(p) => tuning_file(p)?,
                None => Tuning::default(),
            };
            let pairs = par_map(&want, jobs, |s| Ok((scorer.say(&t, s)?.1, mfa_phones(dir, &format!("{}-{}", scorer.body, s.id))?)))?;
            let mut by: BTreeMap<String, Vec<[f64; 4]>> = BTreeMap::new();
            let mut skipped = 0;
            for (plan, mfa) in &pairs {
                if plan.len() != mfa.len() {
                    skipped += 1;
                    continue;
                }
                for (p, m) in plan.iter().zip(mfa).filter(|(p, m)| p.ipa == m.ipa) {
                    by.entry(p.ipa.clone()).or_default().push([m.t0 - p.t0, m.t1 - p.t1, 0.5 * (m.t0 + m.t1 - p.t0 - p.t1), (m.t1 - m.t0) / (p.t1 - p.t0)]);
                }
            }
            println!("aligner against plan, {} sentences ({skipped} skipped: other phone counts); ms, median (10th … 90th)", pairs.len());
            println!("{:<5}{:>6}  {:<24}{:<24}{:<24}{}", "phone", "n", "start", "end", "centre", "length ratio");
            for (ipa, v) in &by {
                let col = |k: usize, scale: f64| {
                    let xs: Vec<f64> = v.iter().map(|x| x[k] * scale).collect();
                    format!("{:6.1} ({:5.1} … {:5.1})", median(&xs), percentile(&xs, 0.1), percentile(&xs, 0.9))
                };
                println!("{ipa:<5}{:>6}  {:<24}{:<24}{:<24}{}", v.len(), col(0, 1000.0), col(1, 1000.0), col(2, 1000.0), col(3, 1.0));
            }
            Ok(())
        }
        (Some("score"), [reference]) => {
            let r = Reference::load(reference)?;
            let speakers = speakers.unwrap_or_else(|| r.speakers.clone());
            let scorer = match &body {
                Some(b) => Scorer::new(&r)?.in_body(b)?,
                None => Scorer::new(&r)?,
            };
            let mfa = match &times {
                None => None,
                Some(t) => Some(t.strip_prefix("mfa:").ok_or(format!("--times {t}: mfa:<aligned-dir>"))?.to_string()),
            };
            let mut voices: Vec<(String, Score)> = Vec::new();
            if let Some(tsp) = &recordings {
                if tuning.is_some() || renders.is_some() || mfa.is_some() || limit.is_some() {
                    return Err("--recordings scores real speakers: no --tuning, --renders, --times or --limit with it".into());
                }
                for (k, ms) in measure_recordings(&scorer.kit, tsp, &speakers, jobs)? {
                    voices.push((k, scorer.judge(&ms)));
                }
            } else {
                let mut want = sentences(&speakers)?;
                if want.is_empty() {
                    return Err(format!("no sentences of {}", speakers.join(",")));
                }
                want.truncate(limit.unwrap_or(usize::MAX));
                let t = match &tuning {
                    Some(p) => tuning_file(p)?,
                    None => Tuning::default(),
                };
                let stem = |s: &Sentence| format!("{}-{}", scorer.body, s.id);
                let clock = std::time::Instant::now();
                let score = if renders.is_none() && mfa.is_none() {
                    scorer.score(&t, &want, jobs)?
                } else {
                    scorer.judge(&par_map(&want, jobs, |s| {
                    let phones = match &mfa {
                        Some(dir) => Some(mfa_phones(dir, &stem(s))?),
                        None => None,
                    };
                    let Some(dir) = &renders else { return scorer.measure_at(&t, s, phones.as_deref()) };
                    // a folder of renders: <stem>.wav as `compare tsp --dump` writes them (with the
                    // plan's phones beside, <stem>.json), or <stem>/<stem>.wav as the aligner's corpus
                    if tuning.is_some() {
                        return Err("--renders are already rendered: no --tuning with it".into());
                    }
                    let flat = format!("{dir}/{}.wav", stem(s));
                    let path = if std::path::Path::new(&flat).exists() { flat } else { format!("{dir}/{0}/{0}.wav", stem(s)) };
                    let (x, fs) = read_wav(&path)?;
                    if (fs - FS).abs() > 0.5 {
                        return Err(format!("{path}: {fs} Hz, expected 48 kHz"));
                    }
                    let phones = match phones {
                        Some(p) => p,
                        None => {
                            let side = format!("{dir}/{}.json", stem(s));
                            let text = std::fs::read_to_string(&side).map_err(|e| format!("{side}: {e} (the plan's phones; or give --times mfa:<aligned-dir>)"))?;
                            let v: Vec<(String, u8, f64, f64)> = serde_json::from_str(&text).map_err(|e| format!("{side}: {e}"))?;
                            sound_times(&plan_phones(&v.into_iter().map(|(ipa, stress, t0, t1)| Phone { ipa, stress, t0, t1 }).collect::<Vec<_>>(), None))
                        }
                    };
                    Ok(scorer.measure_samples(&x, &phones))
                    })?)
                };
                let secs = clock.elapsed().as_secs_f64();
                println!("{} sentences in {secs:.0} s on {jobs} threads: {:.2} sentences a second", want.len(), want.len() as f64 / secs);
                voices.push(("synth".into(), score));
            }
            if let Some(path) = &json_out {
                let out = json!({ "reference": { "sex": r.sex, "speakers": r.speakers }, "version": VERSION, "voices": voices.iter().map(|(k, s)| s.to_json(k)).collect::<Vec<_>>() });
                std::fs::write(path, serde_json::to_string_pretty(&out).unwrap() + "\n").map_err(|e| format!("{path}: {e}"))?;
            }
            println!("against {reference} ({} {})", r.sex, r.speakers.join(" "));
            for (k, s) in &voices {
                print_score(k, s, 15);
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realism_cost_is_zero_within_a_unit_and_grows_by_units_after() {
        assert_eq!(cost(0.5, 1.0), 0.0);
        assert_eq!(cost(-1.0, 1.0), 0.0);
        assert!((cost(-3.0, 1.5) - 1.0).abs() < 1e-12);
        assert!((cost(4.0, 2.0) - 1.0).abs() < 1e-12);
        assert_eq!(cost(1e9, 1.0), WORST);
        assert!(cost(f64::NAN, 1.0).is_nan() && cost(1.0, f64::NAN).is_nan() && cost(1.0, 0.0).is_nan());
    }

    #[test]
    fn realism_unit_is_the_scaled_mad_and_never_under_its_floor() {
        // median 3, absolute deviations 2 1 0 1 2 → MAD 1 → 1.4826
        assert!((unit(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5) - 1.4826).abs() < 1e-12);
        assert_eq!(unit(&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0), 2.0);
        // speakers who agree, and a single speaker: the floor
        assert_eq!(unit(&[7.0, 7.0, 7.0], 0.3), 0.3);
        assert_eq!(unit(&[7.0], 0.3), 0.3);
        assert_eq!(unit(&[], 0.3), 0.3);
    }

    #[test]
    fn realism_pitch_shape_reads_a_falling_line() {
        // 101 voiced frames from +2 to −2 semitones around 100 Hz, inside silence
        let mut f0 = vec![0.0; 10];
        f0.extend((0..101).map(|i| 100.0 * 2f64.powf((2.0 - 4.0 * i as f64 / 100.0) / 12.0)));
        f0.extend([0.0; 10]);
        let p = pitch_shape(&f0, 0.0, 10.0).unwrap();
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(near(p[0], -1.6) && near(p[1], 1.6) && near(p[2], 3.2), "{p:?}");
        assert!(near(p[3], -4.0), "fall {}", p[3]);
        // the first fifth is frames 0..20 (median at 9.5), the last 81..101
        assert!(near(p[4], 2.0 - 4.0 * 0.095) && near(p[5], -(2.0 - 4.0 * 0.095)), "{p:?}");
        assert!(near(p[6], 0.04), "step {}", p[6]);
        // a gap in voicing is not a step, and frames outside the phones are not read
        let mut gap = f0.clone();
        gap[40] = 0.0;
        assert!(near(pitch_shape(&gap, 0.0, 10.0).unwrap()[6], 0.04));
        assert!(pitch_shape(&f0, 0.0, 0.3).is_none());
        let inside = pitch_shape(&f0, 0.10, 0.60).unwrap();
        assert!(near(inside[3], -2.0), "half the line falls half as far: {}", inside[3]);
    }

    #[test]
    fn realism_modulation_finds_the_band_an_envelope_swings_in() {
        let fft = Fft::new(ENV_N);
        let env = |hz: f64| -> Vec<f64> { (0..300).map(|i| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * hz * i as f64 * 0.01).sin()).collect() };
        let (slow, mid, fast) = (modulation(&fft, &env(3.0)), modulation(&fft, &env(6.0)), modulation(&fft, &env(12.0)));
        assert!(slow[0] > slow[1] + 20.0 && slow[0] > slow[2] + 20.0, "{slow:?}");
        assert!(mid[1] > mid[0] + 20.0 && mid[1] > mid[2] + 20.0, "{mid:?}");
        assert!(fast[2] > fast[0] + 20.0 && fast[2] > fast[1] + 20.0, "{fast:?}");
        // the length cancels: twice as long, the same level
        let long: Vec<f64> = (0..600).map(|i| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * 3.0 * i as f64 * 0.01).sin()).collect();
        assert!((modulation(&fft, &long)[0] - slow[0]).abs() < 3.5, "{} {}", modulation(&fft, &long)[0], slow[0]);
        assert!(modulation(&fft, &[1.0; 50])[0].is_nan());
    }

    /// A 43 ms frame's power spectrum: harmonics of f0 falling 6 dB an octave, plus white noise.
    fn frame_spectrum(f0: f64, harmonics: f64, noise: f64) -> Vec<f64> {
        let n = 2 * FRAME;
        let w = hann(n);
        let mut state: u32 = 0x2545_f491;
        let mut re: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / FS;
                let h: f64 = (1..=(8000.0 / f0) as usize).map(|k| (2.0 * std::f64::consts::PI * f0 * k as f64 * t).sin() / k as f64).sum();
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (harmonics * h + noise * (state as f64 / 2_147_483_648.0 - 1.0)) * w[i]
            })
            .collect();
        let mut im = vec![0.0; n];
        Fft::new(n).transform(&mut re, &mut im);
        (0..=n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
    }

    #[test]
    fn realism_ripple_and_peak_prominence_tell_harmonics_from_noise() {
        let bin = FS / (2 * FRAME) as f64;
        let (clean, noisy, hiss) = (frame_spectrum(110.0, 1.0, 0.0), frame_spectrum(110.0, 1.0, 0.3), frame_spectrum(110.0, 0.0, 0.3));
        let r = |p: &[f64]| ripple(p, bin, 110.0, 2000.0, 4000.0);
        assert!(r(&clean) > 25.0, "clean harmonics {}", r(&clean));
        assert!(r(&hiss) < 17.0, "noise alone {}", r(&hiss));
        assert!(r(&clean) > r(&noisy) && r(&noisy) > r(&hiss), "{} {} {}", r(&clean), r(&noisy), r(&hiss));
        let cep = Fft::new(CEP_N);
        let c = |p: &[f64]| peak_prominence(&cep, p, bin);
        assert!(c(&clean) > c(&noisy) + 3.0 && c(&noisy) > c(&hiss) + 3.0, "{} {} {}", c(&clean), c(&noisy), c(&hiss));
    }

    fn sent(level: f64, pitch: f64) -> SentMeas {
        let tok = |ipa, class, stressed| Tok { ipa, class, stressed, mel: Some(std::array::from_fn(|k| level + k as f64 * 0.1)), level, bands: [0.0, -3.0, -10.0, -18.0, -30.0], h1h2: -4.0, cpp: 15.0, ripple: [20.0; 3], change: 1.0 };
        SentMeas {
            toks: vec![tok("ɑ", 0, true), tok("ə", 0, false), tok("s", 2, false), tok("n", 1, false), tok("l", 3, false)],
            cons: vec![Con { ipa: "k", level: level - 10.0, burst: level, closure: CLOSURE_FLOOR, centroid: f64::NAN }, Con { ipa: "s", level, burst: f64::NAN, closure: f64::NAN, centroid: 12.5 }],
            pitch: Some([-2.0, 2.0, 4.0, pitch, 1.0, -2.0, 0.1]),
            modulation: [-10.0; 3],
            change: [1.0, 3.0],
            vowel_power: (0.01 * 100.0, 100),
            speech_power: (0.0025 * 400.0, 400),
            limited: Some((0, 4000)),
            failed: false,
        }
    }

    #[test]
    fn realism_scores_come_out_in_one_order_and_a_lost_voice_costs_the_worst() {
        let defs = term_defs();
        let speaker = |pitch: f64| -> Vec<SentMeas> { (0..4).map(|i| sent(-10.0 + i as f64, pitch)).collect() };
        let vals: BTreeMap<String, Values> = [("MA", -3.0), ("MB", -4.0), ("MC", -5.0)].iter().map(|(k, p)| (k.to_string(), values(&defs, &speaker(*p).iter().collect::<Vec<_>>()))).collect();
        let r = reference_of("male", &defs, &vals);
        assert_eq!(r.terms["pitch fall"].centre, -4.0);
        assert_eq!(r.terms["pitch fall"].unit, 1.4826);
        // the reference survives its JSON to the bit
        let path = std::env::temp_dir().join(format!("chimera-realism-test-{}.json", std::process::id()));
        r.save(path.to_str().unwrap()).unwrap();
        let back = Reference::load(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(back, r);

        // a speaker at the centre costs nothing; terms are by group, then by name
        let centre = score_values(&r, &defs, &values(&defs, &speaker(-4.0).iter().collect::<Vec<_>>()));
        assert_eq!(centre.score, 0.0);
        let order: Vec<(usize, &str)> = centre.terms.iter().map(|t| (group_index(&t.group), t.name.as_str())).collect();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(order, sorted);
        assert!(centre.terms.len() > 40);

        // a fall three units off costs 2 on that term alone
        let off = score_values(&r, &defs, &values(&defs, &speaker(-4.0 - 3.0 * 1.4826).iter().collect::<Vec<_>>()));
        let fall = off.terms.iter().find(|t| t.name == "pitch fall").unwrap();
        assert!((fall.cost - 2.0).abs() < 1e-9, "{}", fall.cost);
        assert!((off.groups["pitch"] - 2.0 / 7.0).abs() < 1e-9 && off.groups["timbre"] == 0.0);
        assert!((off.score - 2.0 / 7.0 / 6.0).abs() < 1e-9);

        // a term on the unscored list is reported with its cost and left out of its group
        let mut loud = speaker(-4.0);
        for m in &mut loud {
            m.toks.push(Tok { ipa: "v", class: 2, stressed: false, mel: None, level: 30.0, bands: [f64::NAN; 5], h1h2: f64::NAN, cpp: f64::NAN, ripple: [f64::NAN; 3], change: 1.0 });
        }
        let with_v: BTreeMap<String, Values> = [("MA", 0.0), ("MB", 0.0)].iter().map(|(k, _)| (k.to_string(), values(&defs, &loud.iter().collect::<Vec<_>>()))).collect();
        let rv = reference_of("male", &defs, &with_v);
        for m in &mut loud {
            m.toks.last_mut().unwrap().level = 60.0;
        }
        let s = score_values(&rv, &defs, &values(&defs, &loud.iter().collect::<Vec<_>>()));
        let v = s.terms.iter().find(|t| t.name == "level v").unwrap();
        assert!(!v.scored && v.cost == 19.0 && s.groups["consonants"] == 0.0, "{} {}", v.cost, s.groups["consonants"]);
        assert!(s.worst(5).iter().all(|t| t.name != "level v"));
        assert!(UNSCORED.iter().all(|u| defs.iter().any(|d| d.name == u.0)), "an unscored name that is no term");

        // a voice with the vowels but no voicing in them: the source terms cost the worst
        let mut mute = speaker(-4.0);
        for m in &mut mute {
            for t in &mut m.toks {
                t.h1h2 = f64::NAN;
            }
        }
        let s = score_values(&r, &defs, &values(&defs, &mute.iter().collect::<Vec<_>>()));
        assert_eq!(s.terms.iter().find(|t| t.name == "H1-H2, stressed").unwrap().cost, WORST);
        // too few tokens is not a lost voice: the term is left out
        let few = score_values(&r, &defs, &values(&defs, &speaker(-4.0)[..2].iter().collect::<Vec<_>>()));
        assert!(few.terms.iter().find(|t| t.name == "H1-H2, stressed").unwrap().cost.is_nan());
        // a failed render: the whole score
        let mut bad = speaker(-4.0);
        bad.push(SentMeas::failed());
        let s = score_values(&r, &defs, &values(&defs, &bad.iter().collect::<Vec<_>>()));
        assert!(s.failed && s.score == WORST && s.groups["timbre"] == WORST);
        // and no limiter's share: the failed one was not counted
        assert!(s.limited.is_nan());

        // the level and the limiter's share are pooled over the sentences and cost nothing
        assert!((centre.level_db + 20.0).abs() < 1e-9 && (centre.speech_db - db(0.0025)).abs() < 1e-9 && centre.limited == 0.0);
        let mut hot = speaker(-4.0);
        hot[0].vowel_power = (0.04 * 300.0, 300);
        hot[1].limited = Some((40, 4000));
        let s = score_values(&r, &defs, &values(&defs, &hot.iter().collect::<Vec<_>>()));
        assert!((s.level_db - db((0.04 * 300.0 + 3.0) / 600.0)).abs() < 1e-9 && s.limited == 40.0 / 16000.0);
        assert!(s.score == 0.0 && !s.failed);
        let j = s.to_json("synth");
        assert_eq!((j["limited"].as_f64(), j["levelDb"].as_f64()), (Some(0.0025), Some(s.level_db)));
    }

    #[test]
    fn realism_a_stops_closure_is_its_quietest_stretch_wherever_the_span_edges_fall() {
        // silence, a vowel from 0.1 until 0.30 s, a closure with a voice bar 26 dB down until 0.36, a burst, the vowel again
        let tone = |i: usize, amp: f64| amp * (2.0 * std::f64::consts::PI * 150.0 * i as f64 / FS).sin();
        let voiced: Vec<f64> = (0..(0.6 * FS) as usize)
            .map(|i| {
                let t = i as f64 / FS;
                if t < 0.1 { 0.0 } else if (0.30..0.36).contains(&t) { tone(i, 0.005) } else if (0.36..0.365).contains(&t) { if i % 2 == 0 { 0.03 } else { -0.03 } } else { tone(i, 0.1) }
            })
            .collect();
        let sig = Sig::new(&voiced, FS, false);
        let vref = db(0.1 * 0.1 / 2.0);
        let stop = |t0: f64, t1: f64| Phone { ipa: "b".into(), stress: 0, t0, t1 };
        let exact = closure_db(&sig, &stop(0.30, 0.37), vref);
        assert!((exact + 26.0).abs() < 0.5, "{exact}");
        // the span 25 ms early, 25 ms late, and 25 ms wider each way: the same closure
        for (t0, t1) in [(0.275, 0.345), (0.325, 0.395), (0.275, 0.395)] {
            assert!((closure_db(&sig, &stop(t0, t1), vref) - exact).abs() < 0.5, "{t0} {t1}: {}", closure_db(&sig, &stop(t0, t1), vref));
        }
        // a silent closure reads the floor, a render's digital silence and a recording's room alike
        let silent: Vec<f64> = voiced.iter().enumerate().map(|(i, v)| if (0.30..0.36).contains(&(i as f64 / FS)) { 0.0 } else { *v }).collect();
        assert_eq!(closure_db(&Sig::new(&silent, FS, false), &stop(0.30, 0.37), vref), CLOSURE_FLOOR);
        let mut state: u32 = 0x9e37_79b9;
        let room: Vec<f64> = silent
            .iter()
            .map(|v| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                v + 0.0003 * (state as f64 / 2_147_483_648.0 - 1.0)
            })
            .collect();
        assert_eq!(closure_db(&Sig::new(&room, FS, false), &stop(0.30, 0.37), vref), CLOSURE_FLOOR);
        // a span with nothing in it, and an utterance with no vowel level: no value
        assert!(closure_db(&sig, &stop(0.7, 0.8), vref).is_nan() && closure_db(&sig, &stop(0.30, 0.37), f64::NAN).is_nan());
    }

    #[test]
    fn realism_the_consonants_group_holds_the_obstruents_and_each_level_lives_in_one_group() {
        let defs = term_defs();
        let group = |name: &str| defs.iter().find(|d| d.name == name).unwrap_or_else(|| panic!("no term {name}")).group;
        for c in STOPS {
            for kind in ["closure", "burst"] {
                assert_eq!(group(&format!("{kind} {c}")), "consonants");
            }
            assert!(defs.iter().all(|d| d.name != format!("level {c}")), "a stop's middle half cannot be had from a plan");
        }
        for c in FRICATIVES.iter().chain(&AFFRICATES) {
            assert_eq!(group(&format!("level {c}")), "consonants");
        }
        for c in SIBILANTS {
            assert_eq!(group(&format!("centroid {c}")), "consonants");
        }
        for c in NASALS.iter().chain(&LIQUIDS) {
            assert_eq!(group(&format!("level {c}")), "timbre");
        }
        // no name twice: a phone's level is one term in one group
        let mut names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        names.sort();
        assert!(names.windows(2).all(|w| w[0] != w[1]));
        assert!(defs.iter().all(|d| GROUPS.contains(&d.group)));

        // a voice whose bursts are 6 dB down pays in `consonants` and nowhere else
        let speaker = |burst: f64| -> Vec<SentMeas> {
            (0..4)
                .map(|i| {
                    let mut m = sent(-10.0 + i as f64, -4.0);
                    m.cons[0].burst = burst + i as f64;
                    m
                })
                .collect()
        };
        let vals: BTreeMap<String, Values> = [("MA", -10.0), ("MB", -11.0), ("MC", -12.0)].iter().map(|(k, b)| (k.to_string(), values(&defs, &speaker(*b).iter().collect::<Vec<_>>()))).collect();
        let r = reference_of("male", &defs, &vals);
        assert_eq!((r.terms["burst k"].centre, r.terms["burst k"].unit, r.terms["burst k"].group.as_str()), (-9.5, 1.5, "consonants"));
        assert_eq!(r.terms["closure k"].centre, CLOSURE_FLOOR);
        assert_eq!(r.terms["centroid s"].centre, 12.5);
        let s = score_values(&r, &defs, &values(&defs, &speaker(-17.0).iter().collect::<Vec<_>>()));
        let burst = s.terms.iter().find(|t| t.name == "burst k").unwrap();
        assert!((burst.cost - 3.0).abs() < 1e-9, "{}", burst.cost);
        assert!(s.groups["consonants"] > 0.0 && GROUPS.iter().filter(|g| **g != "consonants").all(|g| s.groups[*g] == 0.0), "{:?}", s.groups);
    }

    #[test]
    fn realism_plan_segments_become_the_phones_that_were_said() {
        let ph = |ipa: &str, stress, t0: f64| Phone { ipa: ipa.into(), stress, t0, t1: t0 + 0.1 };
        // "my saw is" + "her": a diphthong, a true ɔ before ɪ, an unstressed ɝ
        let plan = [ph("m", 0, 0.0), ph("a", 1, 0.1), ph("ɪ", 1, 0.2), ph("s", 0, 0.3), ph("ɔ", 1, 0.4), ph("ɪ", 0, 0.5), ph("z", 0, 0.6), ph("h", 0, 0.7), ph("ɝ", 0, 0.8), ph("ɫ", 0, 0.9)];
        let said: Vec<String> = ["m", "aɪ", "s", "ɔ", "ɪ", "z", "h", "ɚ", "ɫ"].iter().map(|s| s.to_string()).collect();
        let got = plan_phones(&plan, Some(&said));
        assert_eq!(got.iter().map(|p| p.ipa.as_str()).collect::<Vec<_>>(), ["m", "aɪ", "s", "ɔ", "ɪ", "z", "h", "ɚ", "l"]);
        assert_eq!((got[1].t0, got[1].t1, got[1].stress), (0.1, 0.2 + 0.1, 1));
        // without what was said, the pattern alone: the same but for the ɔ + ɪ it cannot tell apart
        let blind = plan_phones(&plan, None);
        assert_eq!(blind.iter().map(|p| p.ipa.as_str()).collect::<Vec<_>>(), ["m", "aɪ", "s", "ɔɪ", "z", "h", "ɚ", "l"]);
        // a plan that is not what was said one for one falls back to the pattern
        assert_eq!(plan_phones(&plan, Some(&said[1..])).len(), blind.len());
    }

    #[test]
    fn realism_planned_phones_move_to_where_they_sound() {
        let plan = [Phone { ipa: "ð".into(), stress: 0, t0: 1.0, t1: 1.04 }, Phone { ipa: "ɑ".into(), stress: 1, t0: 1.04, t1: 1.2 }, Phone { ipa: "w".into(), stress: 0, t0: 1.2, t1: 1.26 }, Phone { ipa: "ɔ".into(), stress: 0, t0: 2.0, t1: 2.03 }, Phone { ipa: "t".into(), stress: 0, t0: 2.03, t1: 2.09 }];
        let s = sound_times(&plan);
        let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(near(s[0].t0, 1.0246) && near(s[0].t1, 1.06));
        assert!(near(s[1].t0, 1.0695) && near(s[1].t1, 1.1831));
        // a glide has no offset; a phone the offsets would turn inside out keeps 10 ms at its new centre
        assert!(s[2].t0 == 1.2 && s[2].t1 == 1.26);
        // a stop's span ends a little past the planned release, wherever the aligner would end it
        assert!(near(s[4].t0, 2.0264) && near(s[4].t1, 2.09 + STOP_RELEASE / 1000.0));
        assert!(near(s[3].t1 - s[3].t0, 0.01) && near(0.5 * (s[3].t0 + s[3].t1), 0.5 * (2.0271 + 2.0008)));
        let measured: Vec<&str> = cats().into_iter().chain(STOPS).chain(AFFRICATES).collect();
        assert!(PLAN_TO_SOUND.iter().all(|o| measured.contains(&o.0)) && PLAN_TO_SOUND.len() == measured.len());
    }

    #[test]
    fn realism_par_map_keeps_the_items_order_on_any_number_of_threads() {
        let items: Vec<usize> = (0..200).collect();
        let one = par_map(&items, 1, |i| Ok(i * i)).unwrap();
        for jobs in [2, 3, 8] {
            assert_eq!(par_map(&items, jobs, |i| Ok(i * i)).unwrap(), one);
        }
        let e = par_map(&items, 4, |i| if i % 50 == 7 { Err(format!("{i}")) } else { Ok(*i) }).unwrap_err();
        assert_eq!(e, "7\n57\n107\n157");
    }
}
