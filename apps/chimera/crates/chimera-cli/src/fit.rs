//! `chimera fit` (Plint): tune chosen `Tuning` fields against the realism judge (`realism.rs`)
//! with a separable CMA-ES (`chimera::util::cma`), on `--cores` threads, resumable to the bit.
//!
//!   chimera fit <reference.json> --fields source.*,noise.* --out run/ [--cores 2] [--minutes n]
//!               [--generations n] [--resume] [--start t.json] [--seed n] [--lambda n] [--sigma s]
//!               [--speakers MA,MB,MC,MD,MF,MG] [--groups timbre,source,noise,pitch,motion,consonants]
//!               [--weights consonants=2,…] [--pull λ] [--screen]
//!   chimera fit bench <reference.json> [--cores n]
//!
//! **Fields.** `--fields` is a comma-separated list of `Tuning::fields()` paths, `*` standing for
//! any run of characters (`source.*`, `noise.aspiration*`) and `!pattern` taking fields out again.
//! The optimiser moves only these, each inside its own `lo..hi`, from `--start` (a tuning file as
//! `render text --tuning` reads it) or the default.
//!
//! **One evaluation** is the realism score of a candidate on a minibatch of `BATCH` sentences of
//! the fitting men, plus a pull toward the start, `λ·Σ((x − x0)/range)²` (`--pull`, default
//! `DEFAULT_PULL`): a number moves only if the judge pays for it. `--groups` keeps only those
//! groups of the judge in the score (the mean of them), and `--weights group=w,…` makes that mean
//! a weighted one (every group 1 unless named; with all at 1 it is the judge's score to the bit).
//! The log's `fixed` and `best_fixed` are then the weighted number; its group columns are the
//! judge's own. Every candidate of a generation says the
//! same sentences (no racing), two of each speaker, and the sentences rotate from generation to
//! generation by a schedule that is a function of the seed and the generation alone.
//! The generation's best candidate is then scored on a fixed set of `FIXED` sentences, chosen
//! once from the seed; `best.json` is the best so far on that fixed set, never on a minibatch,
//! and without the pull, so that number is the judge's own.
//!
//! **Small sets.** The judge's score is not a mean over sentences: the sentences' tokens are
//! pooled and each term is a median over them, and a term with fewer than three tokens has no
//! value and is left out of its group. So a minibatch scores fewer terms than the whole corpus
//! (rare phones), and which terms it scores could differ between two candidates of one generation
//! (a candidate that makes a phone too short or too quiet to measure loses the term, and with it
//! the term's cost). That must not pay, so the fitter does the judge's last step itself, from
//! the judge's own terms, over one list of terms for all:
//! - on a minibatch, every term that any candidate of the generation could be measured on; a
//!   candidate without it is charged the highest cost another candidate paid for it;
//! - on the fixed set, exactly the terms the start had; a candidate without one is charged
//!   `WORST`, and a term the start did not have is not counted.
//!
//! With nothing missing both are the judge's score to the bit. The log's `unmeasured` column
//! counts the charges.
//!
//! **Blow-ups.** A candidate with a failed render (a non-finite sample, or in the limiter) gets
//! the worst value seen so far in the run (`WORST` when there is none) and is counted in the log.
//!
//! **Stopping.** `--generations n` (the run's generation count, so a resumed run goes on to n),
//! `--minutes n` (of this invocation), or when the fixed-set best has not improved by
//! `STALL_GAIN` in `STALL_GENERATIONS` generations. A kill at any moment loses the generation in
//! flight and nothing else: `state.json` is written whole to a temporary file and renamed, after
//! every generation.
//!
//! **Files in `--out`.**
//! - `state.json`: everything a resume needs (the optimiser, the seed, the fields, the start, the
//!   speakers, the fixed set, the best so far, the pull, the groups and their weights). `--resume` refuses when the
//!   command line names a value that differs from the state's, and says which; an option left out
//!   takes the state's.
//! - `best.json`: a partial tuning object for `--tuning`, `CHIMERA_TUNING` or a script's
//!   `tuning`: the fitted fields, and whatever else the start had off the default.
//! - `log.tsv`: one row a generation (row 0 is the start). The last two columns are timing; the
//!   rest is the same for the same command whatever `--cores` is and however often it was resumed.
//! - `screen.tsv` when `--screen` ran: each field moved to both ends of its range, alone, scored
//!   on the fixed set. Fields that move the score by under `SCREEN_KEEP` of the largest move are
//!   left out of the fit (a field with a failed end stays). `--screen --generations 0` screens only.
//!
//! **Priority.** The command lowers its own niceness to 19 before it starts a thread.
//!
//! Never inside this loop: the neural gate, the held-out men (`GATE_ONLY` is refused outright).

use crate::realism::{GROUPS, Reference, Score, Scorer, SentMeas, Sentence, TermScore, WORST, par_map, sentences};
use crate::render::{g2p, say_with, tuning_file};
use crate::{Ctx, FS};
use chimera::math::Rng;
use chimera::tuning::{Field, Tuning};
use chimera::util::cma::{Cma, CmaOpts};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// Sentences a candidate says in one generation.
const BATCH: usize = 12;
/// Sentences of the fixed set the generation's best is scored on.
const FIXED: usize = 40;
/// The pull's λ. The judge's score is about 1 and a gain of 1 % of it is what the stopping rule
/// counts, 0.01. With 0.01 here a field at the far end of a ±50 % range (half its range from the
/// start) costs 0.0025, a quarter of such a gain, and sixty fields all moved a quarter of their
/// ranges cost 0.0375: enough to hold still a number the judge is indifferent to, too little to
/// hold one that pays.
const DEFAULT_PULL: f64 = 0.01;
/// The first step, as a share of each field's range: 0.15 of a ±50 % range is ±15 % of the value.
const DEFAULT_SIGMA: f64 = 0.15;
const STALL_GENERATIONS: usize = 30;
const STALL_GAIN: f64 = 0.01;
/// `--screen` keeps a field whose larger change is at least this share of the largest one's.
const SCREEN_KEEP: f64 = 0.01;
/// The fitting men (plan-chimera-fit, finding 22). MH MI MJ are the validation men: only when named.
const FIT_MEN: [&str; 6] = ["MA", "MB", "MC", "MD", "MF", "MG"];
/// The held-out men: read at the Gate and nowhere else.
const GATE_ONLY: [&str; 2] = ["MK", "ML"];
/// 2 since the judge has six groups (2026-11-26): a version-1 state holds five-group numbers of
/// another objective and its log has one column fewer, so it is refused, never read.
const STATE_VERSION: u64 = 2;
const LOG_HEAD: &str = "generation\tevaluations\tbatch_best\tbatch_mean\tfixed\tbest_fixed\ttimbre\tsource\tnoise\tpitch\tmotion\tconsonants\tfailed\tunmeasured\tsigma\tsentences_per_s\telapsed_s";

const USAGE: &str = "usage: chimera fit <reference.json> --fields source.*,noise.* --out run/ [--cores 2] [--minutes n] [--generations n] [--resume]\n                   [--start t.json] [--seed n] [--lambda n] [--sigma s] [--speakers MA,MB,MC,MD,MF,MG]\n                   [--groups timbre,source,noise,pitch,motion,consonants] [--weights consonants=2,…] [--pull λ] [--screen]\n       chimera fit bench <reference.json> [--cores n]";

#[cfg(unix)]
unsafe extern "C" {
    /// libc's, which std already links: no crate is needed for one call.
    fn nice(inc: i32) -> i32;
}

/// Niceness 19 for this process and every thread it starts from here on.
fn lower_priority() {
    #[cfg(unix)]
    // SAFETY: nice(2) takes an integer and touches no memory.
    unsafe {
        nice(19);
    }
}

// ---- which fields

/// `pat` against `s`, `*` standing for any run of characters.
fn glob(pat: &str, s: &str) -> bool {
    let parts: Vec<&str> = pat.split('*').collect();
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if parts.len() == 1 {
        return pat == s;
    }
    if s.len() < first.len() + last.len() || !s.starts_with(first) || !s.ends_with(last) {
        return false;
    }
    let mut rest = &s[first.len()..s.len() - last.len()];
    for p in &parts[1..parts.len() - 1] {
        match rest.find(p) {
            Some(i) => rest = &rest[i + p.len()..],
            None => return false,
        }
    }
    true
}

/// The fields a `--fields` list names, in `Tuning::fields()` order. A pattern that matches no
/// field is an error (a misspelt name must not silently fit less).
fn select_fields(spec: &str) -> Result<Vec<&'static Field>, String> {
    let pats: Vec<&str> = spec.split(',').map(str::trim).filter(|p| !p.is_empty()).collect();
    for p in &pats {
        let bare = p.strip_prefix('!').unwrap_or(p);
        if !Tuning::fields().iter().any(|f| glob(bare, f.path)) {
            return Err(format!("--fields: {p:?} matches no tuning field"));
        }
    }
    let wanted = |path: &str| pats.iter().any(|p| !p.starts_with('!') && glob(p, path)) && !pats.iter().any(|p| p.strip_prefix('!').is_some_and(|b| glob(b, path)));
    let out: Vec<&'static Field> = Tuning::fields().iter().filter(|f| wanted(f.path)).collect();
    if out.is_empty() { Err(format!("--fields {spec:?} leaves no field to fit")) } else { Ok(out) }
}

fn fields_of(paths: &[String]) -> Result<Vec<&'static Field>, String> {
    paths.iter().map(|p| Tuning::field(p).ok_or(format!("no tuning field {p:?}"))).collect()
}

fn paths_of(fields: &[&'static Field]) -> Vec<String> {
    fields.iter().map(|f| f.path.to_string()).collect()
}

/// `start` with `x` in `fields`.
fn with_values(start: &Tuning, fields: &[&'static Field], x: &[f64]) -> Tuning {
    let mut t = *start;
    for (f, v) in fields.iter().zip(x) {
        (f.set)(&mut t, *v);
    }
    t
}

/// λ·Σ((x − x0)/range)².
fn pull(lambda: f64, fields: &[&'static Field], x: &[f64], x0: &[f64]) -> f64 {
    lambda * fields.iter().zip(x.iter().zip(x0)).map(|(f, (a, b))| ((a - b) / (f.hi - f.lo)) * ((a - b) / (f.hi - f.lo))).sum::<f64>()
}

/// A tuning as the partial object `--tuning` reads: the `moved` fields, and every other field
/// that is off the default (what a start from an earlier stage carries).
fn partial(t: &Tuning, moved: &[&'static Field]) -> Value {
    let d = Tuning::default();
    let mut out = Map::new();
    let mut put = |path: &str, v: Value| {
        let (group, name) = path.split_once('.').expect("a field's path is group.name");
        out.entry(group).or_insert_with(|| json!({})).as_object_mut().unwrap().insert(name.into(), v);
    };
    for f in Tuning::fields() {
        let x = (f.get)(t);
        if x != (f.get)(&d) || moved.iter().any(|m| m.path == f.path) {
            put(f.path, json!(x));
        }
    }
    if t.prosody.upstream_line {
        put("prosody.upstreamLine", json!(true));
    }
    Value::Object(out)
}

// ---- which sentences

fn check_speakers(speakers: &[String]) -> Result<(), String> {
    if let Some(s) = speakers.iter().find(|s| GATE_ONLY.contains(&s.as_str())) {
        return Err(format!("--speakers {s}: {} are the held-out men, read at the Gate only; a fit never says their sentences", GATE_ONLY.join(" and ")));
    }
    if speakers.is_empty() {
        return Err("--speakers: none named".into());
    }
    if let Some(s) = speakers.iter().enumerate().find(|(i, s)| speakers[..*i].contains(s)) {
        return Err(format!("--speakers: {} twice", s.1));
    }
    Ok(())
}

/// Each speaker's sentences, in `speakers`' order.
fn pools(speakers: &[String]) -> Result<Vec<Vec<Sentence>>, String> {
    check_speakers(speakers)?;
    speakers
        .iter()
        .map(|s| {
            let own = sentences(std::slice::from_ref(s))?;
            if own.is_empty() { Err(format!("--speakers {s}: no sentences in the corpus")) } else { Ok(own) }
        })
        .collect()
}

/// A seed for one use of the run's seed (a use, and two numbers of its own).
fn stream(seed: u32, kind: u32, a: u32, b: u32) -> u32 {
    let mut h = seed ^ 0x9e37_79b9;
    for v in [kind, a, b] {
        h = (h ^ v).wrapping_mul(0x85eb_ca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2_ae35);
        h ^= h >> 16;
    }
    h
}

/// 0..n in a shuffled order (Fisher–Yates on the crate's generator).
fn shuffled(n: usize, seed: u32) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = ((rng.next() * (i + 1) as f64) as usize).min(i);
        v.swap(i, j);
    }
    v
}

/// `total` shared over `k` speakers as evenly as it goes; who gets the odd ones turns with `turn`.
fn shares(total: usize, k: usize, turn: usize) -> Vec<usize> {
    (0..k).map(|i| total / k + usize::from((i + k - turn % k) % k < total % k)).collect()
}

/// The sentences every candidate of `generation` says: each speaker's share (two each of six),
/// taken in turn from a shuffle of his sentences that is made anew each time he has said them all.
/// A function of the seed and the generation alone.
fn minibatch(pools: &[Vec<Sentence>], seed: u32, generation: usize) -> Vec<Sentence> {
    let k = pools.len();
    let mut out = Vec::new();
    for (i, pool) in pools.iter().enumerate() {
        let before: usize = (0..generation).map(|g| shares(BATCH, k, g)[i]).sum();
        for pos in before..before + shares(BATCH, k, generation)[i] {
            let order = shuffled(pool.len(), stream(seed, 1, i as u32, (pos / pool.len()) as u32));
            out.push(pool[order[pos % pool.len()]].clone());
        }
    }
    out
}

/// The fixed set: each speaker's share of `FIXED`, by a shuffle of its own from the seed.
fn fixed_set(pools: &[Vec<Sentence>], seed: u32) -> Vec<Sentence> {
    let counts = shares(FIXED, pools.len(), 0);
    let mut out = Vec::new();
    for (i, pool) in pools.iter().enumerate() {
        let order = shuffled(pool.len(), stream(seed, 2, i as u32, 0));
        out.extend(order.iter().take(counts[i]).map(|&j| pool[j].clone()));
    }
    out
}

// ---- the objective

/// The judge's last step over a list of terms of the fitter's choosing: per group of `groups`
/// the mean of `cost_of` over its scored terms (a term it gives none for is left out), then the
/// mean of the groups. `WORST` for a failed render. With `cost_of` the term's own cost and every
/// group this is `Score::score` to the bit.
fn objective(score: &Score, mix: &Mix, cost_of: &dyn Fn(&TermScore) -> Option<f64>) -> f64 {
    let (mut sum, mut weight) = (0.0, 0.0);
    for (k, g) in GROUPS.iter().enumerate().filter(|(_, g)| mix.groups.iter().any(|x| x == *g)) {
        let cs: Vec<f64> = score.terms.iter().filter(|t| t.group == *g && t.scored).filter_map(cost_of).collect();
        if !cs.is_empty() {
            sum += mix.weights[k] * (cs.iter().sum::<f64>() / cs.len() as f64);
            weight += mix.weights[k];
        }
    }
    if score.failed || !(weight > 0.0) { WORST } else { sum / weight }
}

/// Which groups of the judge a fit counts, and how much each (`--groups`, `--weights`).
#[derive(Clone, Debug, PartialEq)]
struct Mix {
    groups: Vec<String>,
    /// one per group of `GROUPS`, 1 unless `--weights` named it
    weights: [f64; GROUPS.len()],
}

/// `--weights consonants=2,source=0.5` as one weight per group of `GROUPS` (1 for a group not named).
fn parse_weights(spec: &str) -> Result<[f64; GROUPS.len()], String> {
    let mut w = [1.0; GROUPS.len()];
    for part in split(spec) {
        let (g, v) = part.split_once('=').ok_or(format!("--weights {part}: group=weight"))?;
        let k = GROUPS.iter().position(|x| *x == g).ok_or(format!("--weights {g}: the groups are {}", GROUPS.join(", ")))?;
        w[k] = v.parse().ok().filter(|x: &f64| x.is_finite() && *x > 0.0).ok_or(format!("--weights {part}: a weight is a number above 0 (leave a group out with --groups)"))?;
    }
    Ok(w)
}

/// The scored terms a score has a cost for.
fn measured_terms(score: &Score) -> BTreeSet<String> {
    score.terms.iter().filter(|t| t.scored && t.cost.is_finite()).map(|t| t.name.clone()).collect()
}

/// A score on the fixed set: over the start's terms, one that cannot be measured costing `WORST`.
fn fixed_objective(score: &Score, mix: &Mix, start_terms: &BTreeSet<String>) -> f64 {
    objective(score, mix, &|t| start_terms.contains(&t.name).then_some(if t.cost.is_finite() { t.cost } else { WORST }))
}

/// A generation's candidates on their minibatch, before the pull: over every term any of them
/// was measured on, a candidate without it paying the highest cost another paid; and how many
/// such charges there were. A failed candidate reads `WORST` here.
fn batch_objectives(scores: &[Score], mix: &Mix) -> (Vec<f64>, usize) {
    let mut peers: BTreeMap<&str, f64> = BTreeMap::new();
    for s in scores.iter().filter(|s| !s.failed) {
        for t in s.terms.iter().filter(|t| t.scored && t.cost.is_finite()) {
            let e = peers.entry(t.name.as_str()).or_insert(t.cost);
            *e = e.max(t.cost);
        }
    }
    let charged = |s: &Score| s.terms.iter().filter(|t| t.scored && !t.cost.is_finite() && peers.contains_key(t.name.as_str()) && mix.groups.contains(&t.group)).count();
    let values = scores.iter().map(|s| objective(s, mix, &|t| if t.cost.is_finite() { Some(t.cost) } else { peers.get(t.name.as_str()).copied() })).collect();
    (values, scores.iter().filter(|s| !s.failed).map(charged).sum())
}

fn group_values(score: &Score) -> [f64; GROUPS.len()] {
    std::array::from_fn(|k| score.groups.get(GROUPS[k]).copied().unwrap_or(f64::NAN))
}

// ---- the state

/// Everything a run is, between two generations.
#[derive(Clone, Debug)]
struct State {
    reference: String,
    seed: u32,
    /// the fields `--fields` named, and the ones fitted (fewer after a screen)
    selected: Vec<String>,
    fields: Vec<String>,
    screened: bool,
    start: Tuning,
    speakers: Vec<String>,
    mix: Mix,
    pull: f64,
    sigma: f64,
    /// the fixed set's sentence ids
    fixed: Vec<String>,
    start_score: f64,
    start_groups: [f64; GROUPS.len()],
    start_terms: BTreeSet<String>,
    best_generation: usize,
    best_x: Vec<f64>,
    best_score: f64,
    best_groups: [f64; GROUPS.len()],
    /// the worst value a candidate that did not fail has had
    worst_seen: Option<f64>,
    failed: usize,
    /// the stopping rule: the best when it last improved by `STALL_GAIN`, and when that was
    stall_score: f64,
    stall_generation: usize,
    elapsed: f64,
    rendered: usize,
    cma: Cma,
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

impl State {
    fn generation(&self) -> usize {
        self.cma.generation()
    }

    fn to_json(&self) -> Value {
        json!({
            "version": STATE_VERSION,
            "reference": self.reference,
            "seed": self.seed,
            "selected": self.selected,
            "fields": self.fields,
            "screened": self.screened,
            "start": self.start,
            "speakers": self.speakers,
            "groups": self.mix.groups,
            "weights": self.mix.weights,
            "pull": self.pull,
            "sigma": self.sigma,
            "batch": BATCH,
            "fixed": self.fixed,
            "startScore": { "score": num(self.start_score), "groups": self.start_groups.map(num), "terms": self.start_terms },
            "best": { "generation": self.best_generation, "x": self.best_x, "score": num(self.best_score), "groups": self.best_groups.map(num) },
            "worstSeen": self.worst_seen,
            "failed": self.failed,
            "stall": { "score": num(self.stall_score), "generation": self.stall_generation },
            "elapsed": self.elapsed,
            "rendered": self.rendered,
            "cma": self.cma,
        })
    }

    fn from_json(v: &Value) -> Result<State, String> {
        let bad = |what: &str| format!("not a fit's state ({what})");
        let f = |v: &Value, what: &str| v.as_f64().ok_or(bad(what));
        let nan = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
        let n = |v: &Value, what: &str| v.as_u64().map(|x| x as usize).ok_or(bad(what));
        let strings = |v: &Value, what: &str| -> Result<Vec<String>, String> { v.as_array().ok_or(bad(what))?.iter().map(|s| s.as_str().map(String::from).ok_or(bad(what))).collect() };
        let per_group = |v: &Value, what: &str| -> Result<[f64; GROUPS.len()], String> { <[f64; GROUPS.len()]>::try_from(v.as_array().ok_or(bad(what))?.iter().map(nan).collect::<Vec<_>>()).map_err(|_| bad(what)) };
        if v["version"].as_u64() != Some(STATE_VERSION) {
            return Err(format!(
                "a fit's state of version {}, this fitter writes version {STATE_VERSION}: that run was scored by a judge with other groups (the `consonants` group came with version 2) and cannot go on; start a new run in another --out, with --start <its best.json> to go on from where it got",
                v["version"]
            ));
        }
        if v["batch"].as_u64() != Some(BATCH as u64) {
            return Err(format!("a fit with minibatches of {}, this fitter's are {BATCH}", v["batch"]));
        }
        Ok(State {
            reference: v["reference"].as_str().ok_or(bad("reference"))?.into(),
            seed: v["seed"].as_u64().and_then(|s| u32::try_from(s).ok()).ok_or(bad("seed"))?,
            selected: strings(&v["selected"], "selected")?,
            fields: strings(&v["fields"], "fields")?,
            screened: v["screened"].as_bool().ok_or(bad("screened"))?,
            start: serde_json::from_value(v["start"].clone()).map_err(|e| bad(&format!("start: {e}")))?,
            speakers: strings(&v["speakers"], "speakers")?,
            mix: Mix { groups: strings(&v["groups"], "groups")?, weights: per_group(&v["weights"], "weights")? },
            pull: f(&v["pull"], "pull")?,
            sigma: f(&v["sigma"], "sigma")?,
            fixed: strings(&v["fixed"], "fixed")?,
            start_score: nan(&v["startScore"]["score"]),
            start_groups: per_group(&v["startScore"]["groups"], "startScore")?,
            start_terms: strings(&v["startScore"]["terms"], "startScore")?.into_iter().collect(),
            best_generation: n(&v["best"]["generation"], "best")?,
            best_x: v["best"]["x"].as_array().ok_or(bad("best"))?.iter().map(|x| f(x, "best")).collect::<Result<_, _>>()?,
            best_score: nan(&v["best"]["score"]),
            best_groups: per_group(&v["best"]["groups"], "best")?,
            worst_seen: v["worstSeen"].as_f64(),
            failed: n(&v["failed"], "failed")?,
            stall_score: nan(&v["stall"]["score"]),
            stall_generation: n(&v["stall"]["generation"], "stall")?,
            elapsed: f(&v["elapsed"], "elapsed")?,
            rendered: n(&v["rendered"], "rendered")?,
            cma: serde_json::from_value(v["cma"].clone()).map_err(|e| bad(&format!("cma: {e}")))?,
        })
    }

    fn load(path: &str) -> Result<State, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        State::from_json(&serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))
    }

    /// What the command line names differently from this state, if anything.
    fn disagreement(&self, given: &Given) -> Option<String> {
        fn differ<T: PartialEq + std::fmt::Debug>(name: &str, given: &Option<T>, mine: &T) -> Option<String> {
            given.as_ref().filter(|g| *g != mine).map(|g| format!("{name} is {g:?} on the command line and {mine:?} in state.json"))
        }
        differ("the reference", &given.reference, &self.reference)
            .or_else(|| differ("--fields", &given.selected, &self.selected))
            .or_else(|| given.start.filter(|t| *t != self.start).map(|_| "--start is another tuning than the start in state.json".to_string()))
            .or_else(|| differ("--seed", &given.seed, &self.seed))
            .or_else(|| differ("--speakers", &given.speakers, &self.speakers))
            .or_else(|| differ("--groups", &given.groups, &self.mix.groups))
            .or_else(|| differ("--weights", &given.weights, &self.mix.weights))
            .or_else(|| differ("--pull", &given.pull, &self.pull))
            .or_else(|| differ("--sigma", &given.sigma, &self.sigma))
            .or_else(|| differ("--lambda", &given.lambda, &self.cma.lambda()))
    }
}

/// What a command line says about a run; `None` is "not said".
#[derive(Clone, Debug, Default)]
struct Given {
    reference: Option<String>,
    selected: Option<Vec<String>>,
    start: Option<Tuning>,
    seed: Option<u32>,
    speakers: Option<Vec<String>>,
    groups: Option<Vec<String>>,
    weights: Option<[f64; GROUPS.len()]>,
    pull: Option<f64>,
    sigma: Option<f64>,
    lambda: Option<usize>,
}

/// Write a whole file or none of it: to a temporary name beside it, then renamed over it.
fn write_atomic(path: &str, text: &str) -> Result<(), String> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{path}: {e}"))
}

fn fx(x: f64) -> String {
    if x.is_finite() { format!("{x:.6}") } else { "-".into() }
}

/// One row of log.tsv (without its newline).
fn log_row(state: &State, batch_best: f64, batch_mean: f64, fixed: f64, failed: usize, unmeasured: usize, rate: f64) -> String {
    let groups: Vec<String> = state.best_groups.iter().map(|g| fx(*g)).collect();
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{failed}\t{unmeasured}\t{:.6e}\t{rate:.2}\t{:.0}",
        state.generation(),
        state.cma.evaluations(),
        fx(batch_best),
        fx(batch_mean),
        fx(fixed),
        fx(state.best_score),
        groups.join("\t"),
        state.cma.sigma(),
        state.elapsed
    )
}

// ---- the run

/// A scorer on `cores` threads.
struct Lab<'a> {
    scorer: Scorer<'a>,
    cores: usize,
}

impl Lab<'_> {
    /// `tuning` on `set`, the sentences of `have` (already measured with this tuning) not said again.
    fn on_set(&self, tuning: &Tuning, set: &[Sentence], have: &BTreeMap<&str, &SentMeas>) -> Result<(Score, usize), String> {
        let todo: Vec<&Sentence> = set.iter().filter(|s| !have.contains_key(s.id.as_str())).collect();
        let fresh = par_map(&todo, self.cores, |s| self.scorer.measure(tuning, s))?;
        let mut fresh = fresh.into_iter();
        let all: Vec<SentMeas> = set.iter().map(|s| have.get(s.id.as_str()).map(|m| (*m).clone()).unwrap_or_else(|| fresh.next().expect("one measure for each sentence said"))).collect();
        Ok((self.scorer.judge(&all), todo.len()))
    }
}

fn split(list: &str) -> Vec<String> {
    list.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect()
}

/// `--screen`: each field to both ends of its range, alone, on the fixed set. Writes screen.tsv,
/// prints the table and gives the fields to keep.
fn screen(lab: &Lab, out: &str, state: &State, fields: &[&'static Field], fixed: &[Sentence]) -> Result<(Vec<&'static Field>, usize), String> {
    struct Row {
        field: &'static Field,
        at: [f64; 2],
        failed: [bool; 2],
        change: f64,
        group: &'static str,
        group_change: f64,
    }
    let mut rows = Vec::new();
    let mut rendered = 0;
    let clock = Instant::now();
    for (i, f) in fields.iter().enumerate() {
        let x0 = (f.get)(&state.start);
        let mut row = Row { field: f, at: [state.start_score; 2], failed: [false; 2], change: 0.0, group: "-", group_change: 0.0 };
        for (e, end) in [f.lo, f.hi].into_iter().enumerate() {
            if end == x0 {
                continue;
            }
            let (score, said) = lab.on_set(&with_values(&state.start, &[*f], &[end]), fixed, &BTreeMap::new())?;
            rendered += said;
            row.at[e] = fixed_objective(&score, &state.mix, &state.start_terms);
            row.failed[e] = score.failed;
            let change = row.at[e] - state.start_score;
            if !score.failed && change.abs() > row.change.abs() {
                row.change = change;
                let moved = group_values(&score);
                // the group that moved most at this end (of the groups in the score)
                for k in (0..GROUPS.len()).filter(|k| state.mix.groups.iter().any(|g| g == GROUPS[*k])) {
                    let d = moved[k] - state.start_groups[k];
                    if d.is_finite() && (row.group == "-" || d.abs() > row.group_change.abs()) {
                        (row.group, row.group_change) = (GROUPS[k], d);
                    }
                }
            }
        }
        eprintln!("screen {}/{} {}: lo {} hi {} ({:.0} s)", i + 1, fields.len(), f.path, fx(row.at[0]), fx(row.at[1]), clock.elapsed().as_secs_f64());
        rows.push(row);
    }
    let largest = rows.iter().map(|r| r.change.abs()).fold(0.0, f64::max);
    let keep = |r: &Row| r.failed[0] || r.failed[1] || (largest > 0.0 && r.change.abs() >= SCREEN_KEEP * largest);
    let mut text = String::from("field\tlo\thi\tat_lo\tat_hi\tat_start\tchange\tgroup\tgroup_change\tkept\n");
    for r in &rows {
        let at = |e: usize| if r.failed[e] { "failed".to_string() } else { fx(r.at[e]) };
        text += &format!("{}\t{}\t{}\t{}\t{}\t{}\t{:+.6}\t{}\t{:+.6}\t{}\n", r.field.path, r.field.lo, r.field.hi, at(0), at(1), fx(state.start_score), r.change, r.group, r.group_change, if keep(r) { "yes" } else { "no" });
    }
    write_atomic(&format!("{out}/screen.tsv"), &text)?;
    let mut order: Vec<&Row> = rows.iter().collect();
    order.sort_by(|a, b| b.change.abs().total_cmp(&a.change.abs()));
    println!("screen: each field to both ends of its range, alone, on the fixed {} (start {})", fixed.len(), fx(state.start_score));
    println!("  {:<34}{:>10}{:>10}{:>10}  {}", "field", "at lo", "at hi", "change", "the group that moved most");
    for r in &order {
        let at = |e: usize| if r.failed[e] { "failed".to_string() } else { format!("{:.4}", r.at[e]) };
        println!("  {:<34}{:>10}{:>10}{:>+10.4}  {} {:+.3}{}", r.field.path, at(0), at(1), r.change, r.group, r.group_change, if keep(r) { "" } else { "   (dropped)" });
    }
    let kept: Vec<&'static Field> = rows.iter().filter(|r| keep(r)).map(|r| r.field).collect();
    println!("screen: {} of {} fields kept; {} move the score by under {} % of the largest change ({:.4}) and are left out", kept.len(), rows.len(), rows.len() - kept.len(), SCREEN_KEEP * 100.0, largest);
    Ok((kept, rendered))
}

/// `chimera fit bench`: what one core and `cores` threads do.
fn bench(reference: &str, cores: usize) -> Result<(), String> {
    let r = Reference::load(reference)?;
    let scorer = Scorer::new(&r)?;
    let men: Vec<String> = FIT_MEN.iter().map(|s| s.to_string()).collect();
    let batch = minibatch(&pools(&men)?, 1, 0);
    let (g, t) = (g2p()?, Tuning::default());
    // one thread: the render alone, then the render and the measures
    let clock = Instant::now();
    let mut audio = 0.0;
    for s in &batch {
        audio += say_with(&g, &s.text, &r.sex, &t)?.out.len() as f64 / FS;
    }
    let render = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    for s in &batch {
        scorer.measure(&t, s)?;
    }
    let both = clock.elapsed().as_secs_f64();
    let n = batch.len() as f64;
    println!("one core, {} sentences ({:.1} s of speech): the render alone {:.3} s a sentence ({:.2} of real time), the render and the measures {:.3} s; {:.3} sentences a second", batch.len(), audio, render / n, render / audio, both / n, n / both);
    let clock = Instant::now();
    let measured = par_map(&batch, cores, |s| scorer.measure(&t, s))?;
    let wall = clock.elapsed().as_secs_f64();
    let score = scorer.judge(&measured);
    println!("{cores} threads: one candidate on {} sentences in {:.2} s: {:.3} sentences a second, {:.3} a core (score {:.3})", batch.len(), wall, n / wall, n / wall / cores as f64, score.score);
    Ok(())
}

/// `chimera fit …` (the head of this file has the arguments)
pub fn run(_ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let mut pos: Vec<&str> = Vec::new();
    let mut cores = 2usize;
    let (mut fields_arg, mut out, mut minutes, mut generations, mut start_path) = (None, None, None, None, None);
    let (mut resume, mut screening) = (false, false);
    let mut given = Given::default();
    let mut it = a.iter();
    while let Some(x) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        fn parse<T: std::str::FromStr>(name: &str, v: String) -> Result<T, String>
        where
            T::Err: std::fmt::Display,
        {
            v.parse().map_err(|e| format!("{name} {v}: {e}"))
        }
        match x.as_str() {
            "--fields" => fields_arg = Some(val("--fields")?),
            "--out" => out = Some(val("--out")?),
            "--cores" => cores = parse("--cores", val("--cores")?)?,
            "--minutes" => minutes = Some(parse::<f64>("--minutes", val("--minutes")?)?),
            "--generations" => generations = Some(parse::<usize>("--generations", val("--generations")?)?),
            "--resume" => resume = true,
            "--screen" => screening = true,
            "--start" => start_path = Some(val("--start")?),
            "--seed" => given.seed = Some(parse("--seed", val("--seed")?)?),
            "--lambda" => given.lambda = Some(parse("--lambda", val("--lambda")?)?),
            "--sigma" => given.sigma = Some(parse("--sigma", val("--sigma")?)?),
            "--pull" => given.pull = Some(parse("--pull", val("--pull")?)?),
            "--speakers" => given.speakers = Some(split(&val("--speakers")?)),
            "--groups" => given.groups = Some(split(&val("--groups")?)),
            "--weights" => given.weights = Some(parse_weights(&val("--weights")?)?),
            f if f.starts_with("--") => return Err(format!("unknown option {f}\n{USAGE}")),
            _ => pos.push(x),
        }
    }
    if cores == 0 {
        return Err("--cores 0".into());
    }
    lower_priority();
    let reference = match pos.as_slice() {
        ["bench", reference] => return bench(reference, cores),
        [reference] => *reference,
        _ => return Err(USAGE.into()),
    };
    let out = out.ok_or(format!("--out names the run's folder\n{USAGE}"))?;
    let out = out.trim_end_matches('/').to_string();

    // what the command line says, checked before anything is read or written
    if let Some(s) = &given.speakers {
        check_speakers(s)?;
    }
    if let Some(gs) = &given.groups {
        if let Some(g) = gs.iter().find(|g| !GROUPS.contains(&g.as_str())) {
            return Err(format!("--groups {g}: the groups are {}", GROUPS.join(", ")));
        }
        if gs.is_empty() {
            return Err("--groups: none named".into());
        }
        // in the judge's order, each once
        given.groups = Some(GROUPS.iter().filter(|g| gs.iter().any(|x| x == *g)).map(|g| g.to_string()).collect());
    }
    if given.pull.is_some_and(|p| !(p >= 0.0) || !p.is_finite()) || given.sigma.is_some_and(|s| !(s > 0.0) || !s.is_finite()) || given.lambda.is_some_and(|l| l < 2) {
        return Err("--pull is 0 or more, --sigma above 0, --lambda 2 or more".into());
    }
    if let Some(spec) = &fields_arg {
        given.selected = Some(paths_of(&select_fields(spec)?));
    }
    if let Some(p) = &start_path {
        given.start = Some(tuning_file(p)?);
    }
    let r = Reference::load(reference)?;
    given.reference = Some(std::fs::canonicalize(reference).map_err(|e| format!("{reference}: {e}"))?.to_string_lossy().into_owned());

    let state_path = format!("{out}/state.json");
    let log_path = format!("{out}/log.tsv");
    let best_path = format!("{out}/best.json");
    let lab = Lab { scorer: Scorer::new(&r)?, cores };
    let run_clock = Instant::now();
    let save_best = |state: &State, fields: &[&'static Field]| write_atomic(&best_path, &(serde_json::to_string_pretty(&partial(&with_values(&state.start, fields, &state.best_x), fields)).unwrap() + "\n"));

    let (mut state, fields, pools, fixed) = if resume {
        let state = State::load(&state_path)?;
        if let Some(d) = state.disagreement(&given) {
            return Err(format!("cannot resume {out}: {d}"));
        }
        if screening && !state.screened {
            return Err(format!("cannot resume {out}: --screen, and the run in state.json was started without a screen"));
        }
        let fields = fields_of(&state.fields)?;
        let pools = pools(&state.speakers)?;
        let all: BTreeMap<&str, &Sentence> = pools.iter().flatten().map(|s| (s.id.as_str(), s)).collect();
        let fixed: Vec<Sentence> = state.fixed.iter().map(|id| all.get(id.as_str()).map(|s| (*s).clone()).ok_or(format!("{state_path}: no sentence {id} among the speakers'"))).collect::<Result<_, _>>()?;
        // the log may hold a row of a generation the state never saw (killed between the two writes)
        let log = std::fs::read_to_string(&log_path).map_err(|e| format!("{log_path}: {e}"))?;
        let kept: Vec<&str> = log.lines().filter(|l| l.split('\t').next().and_then(|g| g.parse::<usize>().ok()).is_none_or(|g| g <= state.generation())).collect();
        write_atomic(&log_path, &(kept.join("\n") + "\n"))?;
        save_best(&state, &fields)?;
        println!("resuming {out} at generation {}: best {} on the fixed {}", state.generation(), fx(state.best_score), fixed.len());
        (state, fields, pools, fixed)
    } else {
        if std::path::Path::new(&state_path).exists() {
            return Err(format!("{state_path} exists: --resume goes on with that run, another --out starts a new one"));
        }
        let selected = given.selected.clone().ok_or(format!("--fields names what to fit\n{USAGE}"))?;
        let mut fields = fields_of(&selected)?;
        let start = given.start.unwrap_or_default();
        for f in &fields {
            let x = (f.get)(&start);
            if !(f.lo <= x && x <= f.hi) {
                return Err(format!("the start's {} is {x}, outside its range {}..{}", f.path, f.lo, f.hi));
            }
        }
        let speakers = given.speakers.clone().unwrap_or_else(|| FIT_MEN.iter().map(|s| s.to_string()).collect());
        let mix = Mix { groups: given.groups.clone().unwrap_or_else(|| GROUPS.iter().map(|g| g.to_string()).collect()), weights: given.weights.unwrap_or([1.0; GROUPS.len()]) };
        let seed = given.seed.unwrap_or(1);
        let sigma = given.sigma.unwrap_or(DEFAULT_SIGMA);
        let pools = pools(&speakers)?;
        let fixed = fixed_set(&pools, seed);
        std::fs::create_dir_all(&out).map_err(|e| format!("{out}: {e}"))?;
        // the start on the fixed set, once
        let (score, said) = lab.on_set(&start, &fixed, &BTreeMap::new())?;
        if score.failed {
            return Err("the start's own render failed on the fixed set: nothing to fit from".into());
        }
        let start_terms = measured_terms(&score);
        let start_score = fixed_objective(&score, &mix, &start_terms);
        let x0: Vec<f64> = fields.iter().map(|f| (f.get)(&start)).collect();
        let cma_of = |fields: &[&'static Field]| {
            Cma::new(
                &fields.iter().map(|f| (f.get)(&start)).collect::<Vec<_>>(),
                CmaOpts { lo: Some(fields.iter().map(|f| f.lo).collect()), hi: Some(fields.iter().map(|f| f.hi).collect()), sigma: Some(sigma), lambda: given.lambda, seed: Some(seed), ..Default::default() },
            )
        };
        let mut state = State {
            reference: given.reference.clone().unwrap(),
            seed,
            selected,
            fields: paths_of(&fields),
            screened: false,
            start,
            speakers,
            mix,
            pull: given.pull.unwrap_or(DEFAULT_PULL),
            sigma,
            fixed: fixed.iter().map(|s| s.id.clone()).collect(),
            start_score,
            start_groups: group_values(&score),
            start_terms,
            best_generation: 0,
            best_x: x0,
            best_score: start_score,
            best_groups: group_values(&score),
            worst_seen: None,
            failed: 0,
            stall_score: start_score,
            stall_generation: 0,
            elapsed: 0.0,
            rendered: said,
            cma: cma_of(&fields),
        };
        println!("start: {} on the fixed {} ({} terms; {})", fx(start_score), fixed.len(), state.start_terms.len(), GROUPS.iter().zip(&state.start_groups).map(|(g, x)| format!("{g} {}", fx(*x))).collect::<Vec<_>>().join("  "));
        if screening {
            let (kept, said) = screen(&lab, &out, &state, &fields, &fixed)?;
            if kept.is_empty() {
                return Err("the screen kept no field: nothing moves the score".into());
            }
            fields = kept;
            state.fields = paths_of(&fields);
            state.best_x = fields.iter().map(|f| (f.get)(&state.start)).collect();
            state.cma = cma_of(&fields);
            state.screened = true;
            state.rendered += said;
        }
        state.elapsed = run_clock.elapsed().as_secs_f64();
        let rate = state.rendered as f64 / state.elapsed.max(1e-9);
        write_atomic(&log_path, &format!("{LOG_HEAD}\n{}\n", log_row(&state, f64::NAN, f64::NAN, start_score, 0, 0, rate)))?;
        save_best(&state, &fields)?;
        write_atomic(&state_path, &(serde_json::to_string(&state.to_json()).unwrap() + "\n"))?;
        (state, fields, pools, fixed)
    };
    let start_terms = state.start_terms.clone();
    let x0: Vec<f64> = fields.iter().map(|f| (f.get)(&state.start)).collect();
    println!("fitting {} fields, {} candidates a generation on {} sentences, {cores} threads, seed {}, pull {}", fields.len(), state.cma.lambda(), BATCH, state.seed, state.pull);
    if state.mix.weights != [1.0; GROUPS.len()] {
        println!("weights: {} (the fixed-set numbers are the weighted mean, not the judge's score)", GROUPS.iter().zip(&state.mix.weights).map(|(g, w)| format!("{g} {w}")).collect::<Vec<_>>().join("  "));
    }

    let elapsed_before = if resume { state.elapsed } else { 0.0 };
    let stopped = loop {
        let g = state.generation();
        if generations.is_some_and(|n| g >= n) {
            break format!("--generations {}", generations.unwrap());
        }
        if minutes.is_some_and(|m| run_clock.elapsed().as_secs_f64() >= m * 60.0) {
            break format!("--minutes {}", minutes.unwrap());
        }
        if g - state.stall_generation >= STALL_GENERATIONS {
            break format!("the fixed-set best has not improved by {} % in {STALL_GENERATIONS} generations", STALL_GAIN * 100.0);
        }
        let clock = Instant::now();
        let xs = state.cma.ask();
        let tunings: Vec<Tuning> = xs.iter().map(|x| with_values(&state.start, &fields, x)).collect();
        let batch = minibatch(&pools, state.seed, g);
        // every (candidate, sentence) is one job, so the threads stay busy to the generation's end
        let jobs: Vec<(usize, &Sentence)> = (0..tunings.len()).flat_map(|c| batch.iter().map(move |s| (c, s))).collect();
        let measured = par_map(&jobs, cores, |(c, s)| lab.scorer.measure(&tunings[*c], s))?;
        let scores: Vec<Score> = measured.chunks(batch.len()).map(|m| lab.scorer.judge(m)).collect();
        let (raw, unmeasured) = batch_objectives(&scores, &state.mix);
        let mut values: Vec<f64> = raw.iter().zip(&xs).map(|(v, x)| v + pull(state.pull, &fields, x, &x0)).collect();
        // a blow-up gets the worst value seen so far in the run, this generation's included
        let failed = scores.iter().filter(|s| s.failed).count();
        for (v, _) in values.iter().zip(&scores).filter(|(_, s)| !s.failed) {
            state.worst_seen = Some(state.worst_seen.map_or(*v, |w| w.max(*v)));
        }
        for (v, _) in values.iter_mut().zip(&scores).filter(|(_, s)| s.failed) {
            *v = state.worst_seen.unwrap_or(WORST);
        }
        state.failed += failed;
        state.cma.tell(&values);

        // the generation's best (the first of equals), on the fixed set
        let best = (0..values.len()).min_by(|&a, &b| values[a].total_cmp(&values[b])).unwrap();
        let have: BTreeMap<&str, &SentMeas> = batch.iter().map(|s| s.id.as_str()).zip(&measured[best * batch.len()..(best + 1) * batch.len()]).collect();
        let (score, said) = lab.on_set(&tunings[best], &fixed, &have)?;
        let on_fixed = fixed_objective(&score, &state.mix, &start_terms);
        if !score.failed && on_fixed < state.best_score {
            state.best_generation = g + 1;
            state.best_x = xs[best].clone();
            state.best_score = on_fixed;
            state.best_groups = group_values(&score);
            save_best(&state, &fields)?;
        }
        if state.best_score < state.stall_score * (1.0 - STALL_GAIN) {
            state.stall_score = state.best_score;
            state.stall_generation = g + 1;
        }
        let rendered = jobs.len() + said;
        state.rendered += rendered;
        state.elapsed = elapsed_before + run_clock.elapsed().as_secs_f64();
        let rate = rendered as f64 / clock.elapsed().as_secs_f64().max(1e-9);
        let batch_best = values.iter().copied().fold(f64::INFINITY, f64::min);
        let row = log_row(&state, batch_best, values.iter().sum::<f64>() / values.len() as f64, on_fixed, failed, unmeasured, rate);
        // the log's row first, the state last: a kill between the two leaves a row the resume removes
        {
            use std::io::Write;
            let mut log = std::fs::OpenOptions::new().append(true).open(&log_path).map_err(|e| format!("{log_path}: {e}"))?;
            writeln!(log, "{row}").map_err(|e| format!("{log_path}: {e}"))?;
        }
        write_atomic(&state_path, &(serde_json::to_string(&state.to_json()).unwrap() + "\n"))?;
        println!("generation {}: batch best {} fixed {} best {} (generation {}){} sigma {:.4} {:.2} sentences/s", g + 1, fx(batch_best), fx(on_fixed), fx(state.best_score), state.best_generation, if failed > 0 { format!(" {failed} failed") } else { String::new() }, state.cma.sigma(), rate);
    };
    println!("stopped at generation {}: {stopped}", state.generation());
    println!("best on the fixed {}: {} (start {}), from generation {} → {best_path}", fixed.len(), fx(state.best_score), fx(state.start_score), state.best_generation);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn fit_fields_are_chosen_by_globs_and_taken_out_by_a_bang() {
        assert!(glob("source.*", "source.oqBase") && !glob("source.*", "noise.teethGain"));
        assert!(glob("noise.aspiration*", "noise.aspirationTop") && !glob("noise.aspiration*", "noise.teethOrder"));
        assert!(glob("*.re*", "physics.reCrit") && glob("*", "artic.jaw") && glob("s*.oq*e", "source.oqBase") && !glob("s*.oq*e", "source.oqEffort"));
        assert!(glob("source.oqBase", "source.oqBase") && !glob("source.oqBase", "source.oqBaseX") && !glob("ab*ba", "aba"));

        let all = |g: &str| Tuning::fields().iter().filter(|f| f.path.starts_with(g)).count();
        let source = select_fields("source.*").unwrap();
        assert_eq!(source.len(), all("source."));
        assert!(source.iter().all(|f| f.path.starts_with("source.")));
        let two = select_fields("source.*,noise.*").unwrap();
        assert_eq!(two.len(), all("source.") + all("noise."));
        // in the table's order whatever the list's, each once
        assert_eq!(paths_of(&select_fields("noise.*, source.*,source.oqBase").unwrap()), paths_of(&two));
        let some = paths_of(&select_fields("noise.aspiration*,source.oqBase,!noise.aspirationLow").unwrap());
        assert_eq!(some, ["source.oqBase", "noise.aspirationTop", "noise.aspirationOrder"]);
        assert_eq!(select_fields("source.*,!source.pressure").unwrap().len(), all("source.") - 1);
        // a name that is no field, an exclusion that excludes nothing, and nothing left
        assert!(select_fields("source.oqBaze").unwrap_err().contains("oqBaze"));
        assert!(select_fields("source.*,!sauce.*").is_err());
        assert!(select_fields("source.oqBase,!source.*").is_err());
        assert!(select_fields("").is_err() && select_fields("!source.*").is_err());
    }

    fn fake_pools(k: usize) -> Vec<Vec<Sentence>> {
        (0..k).map(|i| (0..60).map(|j| Sentence { id: format!("{:02}_{:02}", i + 1, j), text: String::new() }).collect()).collect()
    }

    #[test]
    fn fit_minibatches_are_balanced_and_a_function_of_the_seed_and_the_generation() {
        let pools = fake_pools(6);
        let speaker = |s: &Sentence| s.id[..2].parse::<usize>().unwrap() - 1;
        let mut said: Vec<BTreeSet<String>> = vec![BTreeSet::new(); 6];
        for g in 0..90 {
            let batch = minibatch(&pools, 7, g);
            assert_eq!(batch.len(), BATCH);
            for i in 0..6 {
                assert_eq!(batch.iter().filter(|s| speaker(s) == i).count(), 2, "generation {g}, speaker {i}");
            }
            assert_eq!(batch.iter().collect::<BTreeSet<_>>().len(), BATCH, "a sentence twice in generation {g}");
            assert_eq!(batch, minibatch(&pools, 7, g));
            if g < 30 {
                for s in &batch {
                    assert!(said[speaker(s)].insert(s.id.clone()), "{} twice before every sentence was said", s.id);
                }
            }
        }
        // thirty generations say each of a speaker's sixty sentences once
        assert!(said.iter().all(|s| s.len() == 60));
        assert_ne!(minibatch(&pools, 7, 0), minibatch(&pools, 7, 1));
        assert_ne!(minibatch(&pools, 7, 0), minibatch(&pools, 7, 30));
        assert_ne!(minibatch(&pools, 7, 0), minibatch(&pools, 8, 0));

        // the fixed set: 40 over six speakers, seven or six each, one set a seed
        let fixed = fixed_set(&pools, 7);
        assert_eq!(fixed.len(), FIXED);
        assert_eq!((0..6).map(|i| fixed.iter().filter(|s| speaker(s) == i).count()).collect::<Vec<_>>(), [7, 7, 7, 7, 6, 6]);
        assert_eq!(fixed.iter().collect::<BTreeSet<_>>().len(), FIXED);
        assert_eq!(fixed, fixed_set(&pools, 7));
        assert_ne!(fixed, fixed_set(&pools, 8));

        // speakers that do not divide the batch: the odd sentences take turns
        assert_eq!(shares(12, 6, 3), [2; 6]);
        assert_eq!(shares(12, 5, 0), [3, 3, 2, 2, 2]);
        assert_eq!(shares(12, 5, 1), [2, 3, 3, 2, 2]);
        let five = fake_pools(5);
        for g in 0..20 {
            assert_eq!(minibatch(&five, 1, g).len(), BATCH);
            assert_eq!(minibatch(&five, 1, g).iter().collect::<BTreeSet<_>>().len(), BATCH);
        }
    }

    #[test]
    fn fit_pull_grows_with_the_square_of_the_share_of_the_range_moved() {
        let fields = select_fields("source.oqBase,source.amplitude").unwrap();
        let d = Tuning::default();
        let x0: Vec<f64> = fields.iter().map(|f| (f.get)(&d)).collect();
        assert_eq!(pull(0.01, &fields, &x0, &x0), 0.0);
        // amplitude to the top of its range: half the range from its default
        let amp = fields.iter().position(|f| f.path == "source.amplitude").unwrap();
        let mut x = x0.clone();
        x[amp] = fields[amp].hi;
        assert!((pull(0.01, &fields, &x, &x0) - 0.01 * 0.25).abs() < 1e-15);
        assert!((pull(2.0, &fields, &x, &x0) - 0.5).abs() < 1e-12);
        // and both moved: the sum
        let oq = 1 - amp;
        x[oq] = x0[oq] + 0.1 * (fields[oq].hi - fields[oq].lo);
        assert!((pull(1.0, &fields, &x, &x0) - (0.25 + 0.01)).abs() < 1e-12);
        assert_eq!(pull(0.0, &fields, &x, &x0), 0.0);
        // the tuning a candidate is: the start with those numbers
        let t = with_values(&d, &fields, &x);
        assert_eq!(t.source.amplitude, fields[amp].hi);
        let mut want = d;
        want.source.amplitude = x[amp];
        want.source.oq_base = x[oq];
        assert_eq!(t, want);
    }

    #[test]
    fn fit_best_json_is_a_partial_tuning_that_reads_back_as_the_whole() {
        let fields = select_fields("source.oqBase,noise.glottisGain").unwrap();
        // a start from an earlier stage: one number off the default that this fit does not move
        let mut start = Tuning::default();
        start.artic.jaw = 11.5;
        let t = with_values(&start, &fields, &[0.61, 2.5e-6]);
        let v = partial(&t, &fields);
        assert_eq!(v, json!({ "artic": { "jaw": 11.5 }, "noise": { "glottisGain": 2.5e-6 }, "source": { "oqBase": 0.61 } }));
        assert_eq!(chimera_serve::script::parse_tuning(v).unwrap(), t);
        // a moved field is written even when it ended on the default; nothing else is
        assert_eq!(partial(&Tuning::default(), &fields), json!({ "noise": { "glottisGain": 3.2e-6 }, "source": { "oqBase": 0.78 } }));
        assert_eq!(partial(&Tuning::default(), &[]), json!({}));
        let up = partial(&Tuning::upstream(), &[]);
        assert_eq!(up["prosody"]["upstreamLine"], json!(true));
        assert_eq!(serde_json::from_value::<Tuning>(up).unwrap(), Tuning::upstream());
    }

    fn term(name: &str, group: &str, cost: f64) -> TermScore {
        TermScore { name: name.into(), group: group.into(), value: 0.0, centre: 0.0, unit: 1.0, cost, support: 3, scored: true }
    }

    fn score_of(terms: Vec<TermScore>) -> Score {
        let mut groups = BTreeMap::new();
        for g in GROUPS {
            let cs: Vec<f64> = terms.iter().filter(|t| t.group == g && t.scored && t.cost.is_finite()).map(|t| t.cost).collect();
            groups.insert(g.to_string(), if cs.is_empty() { f64::NAN } else { cs.iter().sum::<f64>() / cs.len() as f64 });
        }
        let have: Vec<f64> = GROUPS.iter().map(|g| groups[*g]).filter(|x| x.is_finite()).collect();
        Score { score: have.iter().sum::<f64>() / have.len() as f64, groups, terms, sentences: 12, failed: false }
    }

    #[test]
    fn fit_a_term_that_cannot_be_measured_does_not_pay() {
        let all = Mix { groups: GROUPS.iter().map(|g| g.to_string()).collect(), weights: [1.0; GROUPS.len()] };
        let whole = score_of(vec![term("spectrum s", "timbre", 2.0), term("level s", "timbre", 1.0), term("H1-H2", "source", 3.0), term("pitch fall", "pitch", 0.5)]);
        // nothing missing: the judge's own score, to the bit, on a minibatch and on the fixed set
        let (values, charged) = batch_objectives(std::slice::from_ref(&whole), &all);
        assert_eq!((values[0], charged), (whole.score, 0));
        assert_eq!(fixed_objective(&whole, &all, &measured_terms(&whole)), whole.score);
        assert_eq!(whole.score, (1.5 + 3.0 + 0.5) / 3.0);

        // a candidate that lost its /s/ (no cost for the spectrum): the judge alone would read it better
        let lost = score_of(vec![term("spectrum s", "timbre", f64::NAN), term("level s", "timbre", 1.0), term("H1-H2", "source", 3.0), term("pitch fall", "pitch", 0.5)]);
        assert!(lost.score < whole.score);
        // on a minibatch it pays what the worst of the others paid for that term
        let other = score_of(vec![term("spectrum s", "timbre", 5.0), term("level s", "timbre", 1.0), term("H1-H2", "source", 3.0), term("pitch fall", "pitch", 0.5)]);
        let (values, charged) = batch_objectives(&[whole.clone(), lost.clone(), other.clone()], &all);
        assert_eq!(charged, 1);
        assert_eq!(values[0], whole.score);
        assert_eq!(values[1], values[2]);
        assert!(values[1] > values[0]);
        // a term nobody could be measured on is nobody's cost
        let (alone, charged) = batch_objectives(&[lost.clone(), lost.clone()], &all);
        assert_eq!((alone[0], charged), (lost.score, 0));
        // on the fixed set it costs the worst, and a term the start did not have is not counted
        let start_terms = measured_terms(&whole);
        assert_eq!(fixed_objective(&lost, &all, &start_terms), ((WORST + 1.0) / 2.0 + 3.0 + 0.5) / 3.0);
        let mut more = whole.clone();
        more.terms.push(term("spectrum ʒ", "timbre", 0.0));
        assert_eq!(fixed_objective(&more, &all, &start_terms), whole.score);
        // an unscored term never counts
        let mut unscored = whole.clone();
        unscored.terms.push(TermScore { scored: false, ..term("level v", "timbre", 9.0) });
        assert_eq!(batch_objectives(&[unscored], &all).0[0], whole.score);

        // --groups: the mean of those groups alone
        let two = Mix { groups: names(&["source", "pitch"]), weights: [1.0; GROUPS.len()] };
        assert_eq!(batch_objectives(std::slice::from_ref(&whole), &two).0[0], (3.0 + 0.5) / 2.0);
        assert_eq!(batch_objectives(&[whole.clone(), lost.clone()], &two), (vec![1.75, 1.75], 0));
        // --weights: a weighted mean of the groups that have terms (timbre 1.5, source 3, pitch 0.5)
        let heavy = Mix { weights: parse_weights("source=3,motion=5").unwrap(), ..all.clone() };
        assert_eq!(heavy.weights, [1.0, 3.0, 1.0, 1.0, 5.0, 1.0]);
        assert_eq!(batch_objectives(std::slice::from_ref(&whole), &heavy).0[0], (1.5 + 3.0 * 3.0 + 0.5) / 5.0);
        assert_eq!(fixed_objective(&whole, &heavy, &measured_terms(&whole)), (1.5 + 3.0 * 3.0 + 0.5) / 5.0);
        assert_eq!(parse_weights("").unwrap(), [1.0; GROUPS.len()]);
        assert!(parse_weights("sauce=2").unwrap_err().contains("the groups are"));
        assert!(parse_weights("source").is_err() && parse_weights("source=0").is_err() && parse_weights("source=-1").is_err() && parse_weights("source=x").is_err());
        // a failed render is the worst whatever its terms say
        let failed = Score { failed: true, ..whole.clone() };
        assert_eq!(batch_objectives(&[failed.clone(), whole.clone()], &all).0, [WORST, whole.score]);
        assert_eq!(fixed_objective(&failed, &all, &start_terms), WORST);
    }

    fn a_state() -> State {
        let fields = select_fields("source.oqBase,source.amplitude,noise.glottisGain").unwrap();
        let mut start = Tuning::default();
        start.source.oq_base = 0.7;
        let x0: Vec<f64> = fields.iter().map(|f| (f.get)(&start)).collect();
        let mut cma = Cma::new(&x0, CmaOpts { lo: Some(fields.iter().map(|f| f.lo).collect()), hi: Some(fields.iter().map(|f| f.hi).collect()), sigma: Some(DEFAULT_SIGMA), seed: Some(9), ..Default::default() });
        for _ in 0..3 {
            let xs = cma.ask();
            cma.tell(&xs.iter().map(|x| pull(1.0, &fields, x, &x0) + 0.1 * x[0]).collect::<Vec<_>>());
        }
        State {
            reference: "/somewhere/ref-fit6.json".into(),
            seed: 9,
            selected: names(&["source.oqBase", "source.amplitude", "source.drift", "noise.glottisGain"]),
            fields: paths_of(&fields),
            screened: true,
            start,
            speakers: names(&FIT_MEN),
            mix: Mix { groups: names(&["source", "noise"]), weights: [1.0, 1.0, 2.0, 1.0, 1.0, 1.0] },
            pull: DEFAULT_PULL,
            sigma: DEFAULT_SIGMA,
            fixed: names(&["01_03", "07_10"]),
            start_score: 1.0625,
            start_groups: [0.6, 2.4, 1.5, f64::NAN, 0.1, 0.9],
            start_terms: ["H1-H2, stressed", "spectrum s"].iter().map(|s| s.to_string()).collect(),
            best_generation: 2,
            best_x: cma.ask()[0].clone(),
            best_score: 0.987_654_321_012_345_6,
            best_groups: [0.5, 2.0, 1.4, f64::NAN, 0.1, 0.8],
            worst_seen: Some(1.3),
            failed: 1,
            stall_score: 1.0,
            stall_generation: 2,
            elapsed: 123.5,
            rendered: 616,
            cma,
        }
    }

    #[test]
    fn fit_state_survives_its_json_to_the_bit_and_goes_on_as_it_would_have() {
        let state = a_state();
        let text = serde_json::to_string(&state.to_json()).unwrap();
        let mut back = State::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
        assert_eq!(serde_json::to_string(&back.to_json()).unwrap(), text);
        assert_eq!((back.start, back.best_score.to_bits(), back.best_x.clone()), (state.start, state.best_score.to_bits(), state.best_x.clone()));
        assert_eq!((back.generation(), back.worst_seen, back.start_terms.len()), (3, Some(1.3), 2));
        assert!(back.start_groups[3].is_nan() && back.start_groups[1] == 2.4);
        // the optimiser asks for the same candidates after the round trip
        let mut whole = state.clone();
        assert_eq!(back.cma.ask(), whole.cma.ask());
        // a run that never had a candidate that did not fail
        let none = State { worst_seen: None, ..a_state() };
        assert_eq!(State::from_json(&none.to_json()).unwrap().worst_seen, None);
        // another fitter's state, and something that is no state
        let mut v = state.to_json();
        v["version"] = json!(99);
        assert!(State::from_json(&v).unwrap_err().contains("version 99"));
        // a state from before the judge's sixth group is refused by its version, not misread
        v["version"] = json!(1);
        let old = State::from_json(&v).unwrap_err();
        assert!(old.contains("version 1,") && old.contains("consonants") && old.contains("--start"), "{old}");
        assert!(State::from_json(&json!({ "version": STATE_VERSION, "batch": BATCH })).is_err());
    }

    #[test]
    fn fit_refuses_the_held_out_men_and_a_resume_that_disagrees_with_its_state() {
        for held in GATE_ONLY {
            let e = check_speakers(&names(&["MA", held])).unwrap_err();
            assert!(e.contains(held) && e.contains("held-out"), "{e}");
            assert!(pools(&names(&[held])).is_err());
        }
        assert!(check_speakers(&names(&FIT_MEN)).is_ok() && check_speakers(&names(&["MH", "MI", "MJ"])).is_ok());
        assert!(check_speakers(&[]).is_err() && check_speakers(&names(&["MA", "MA"])).is_err());
        assert!(FIT_MEN.iter().all(|m| !GATE_ONLY.contains(m)));

        let state = a_state();
        // nothing said, or the same said again: it agrees
        assert_eq!(state.disagreement(&Given::default()), None);
        let same = Given {
            reference: Some(state.reference.clone()),
            selected: Some(state.selected.clone()),
            start: Some(state.start),
            seed: Some(9),
            speakers: Some(state.speakers.clone()),
            groups: Some(state.mix.groups.clone()),
            weights: Some(state.mix.weights),
            pull: Some(DEFAULT_PULL),
            sigma: Some(DEFAULT_SIGMA),
            lambda: Some(state.cma.lambda()),
        };
        assert_eq!(state.disagreement(&same), None);
        // each difference is named
        let cases: Vec<(Given, &str)> = vec![
            (Given { reference: Some("/elsewhere/ref.json".into()), ..same.clone() }, "the reference"),
            (Given { selected: Some(names(&["source.oqBase"])), ..same.clone() }, "--fields"),
            (Given { start: Some(Tuning::default()), ..same.clone() }, "--start"),
            (Given { seed: Some(10), ..same.clone() }, "--seed"),
            (Given { speakers: Some(names(&["MA", "MB"])), ..same.clone() }, "--speakers"),
            (Given { groups: Some(names(&["pitch", "motion"])), ..same.clone() }, "--groups"),
            (Given { weights: Some([1.0; GROUPS.len()]), ..same.clone() }, "--weights"),
            (Given { pull: Some(0.02), ..same.clone() }, "--pull"),
            (Given { sigma: Some(0.3), ..same.clone() }, "--sigma"),
            (Given { lambda: Some(state.cma.lambda() + 2), ..same.clone() }, "--lambda"),
        ];
        for (given, name) in cases {
            let d = state.disagreement(&given).unwrap_or_else(|| panic!("{name} differs and was not refused"));
            assert!(d.starts_with(name), "{name}: {d}");
        }
        assert!(state.disagreement(&Given { seed: Some(10), ..Given::default() }).unwrap().contains("10") );
    }

    #[test]
    fn fit_log_rows_carry_no_timing_before_their_last_two_columns() {
        let state = a_state();
        let cols = LOG_HEAD.split('\t').count();
        let (a, b) = (log_row(&state, 0.9, 1.1, 1.0, 1, 2, 3.25), log_row(&State { elapsed: 999.0, ..a_state() }, 0.9, 1.1, 1.0, 1, 2, 7.5));
        assert_eq!(a.split('\t').count(), cols);
        assert_ne!(a, b);
        let cut = |r: &str| r.split('\t').take(cols - 2).collect::<Vec<_>>().join("\t");
        assert_eq!(cut(&a), cut(&b));
        assert!(LOG_HEAD.ends_with("sentences_per_s\telapsed_s"));
        assert!(a.starts_with("3\t") && a.contains("\t-\t"), "{a}");
    }
}
