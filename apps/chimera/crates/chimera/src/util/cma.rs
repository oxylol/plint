//! Separable CMA-ES (Ros & Hansen 2008: the covariance is a diagonal) with box bounds, for
//! minimising a noisy function of many numbers whose scales differ. Ask-and-tell: the caller
//! evaluates each generation's candidates however it likes (in parallel, on a minibatch).
//!
//! - **Normalised coordinates.** A bounded dimension is searched as 0..1 over its range, so one
//!   step size serves numbers of any magnitude; an unbounded one is searched as it is.
//! - **Bounds by mirroring.** A sample that leaves a bound is reflected back inside, and the
//!   reflected point is the one that is evaluated and the one the update learns from, so the mean
//!   never leaves the box.
//! - **Deterministic and resumable.** The only randomness is the crate's [`Rng`] from an explicit
//!   `u32` seed. The whole state is plain serde data (finite `f64`s, which JSON carries exactly):
//!   the same seed and the same `tell`s give the same `ask`s, across a save and a load at any
//!   point. The generator itself is not stored: the state keeps how many numbers it has drawn and
//!   a loaded state draws that many again before it goes on.
//!
//! ```ignore
//! let mut es = Cma::new(&x0, CmaOpts { lo: Some(lo), hi: Some(hi), seed: Some(7), ..Default::default() });
//! loop {
//!     let xs = es.ask();                                   // λ candidates, inside the bounds
//!     let fs: Vec<f64> = xs.iter().map(|x| f(x)).collect();
//!     es.tell(&fs);                                        // lower is better; NaN ranks last
//!     std::fs::write("state.json", serde_json::to_string(&es)?)?;
//! }
//! ```

use crate::math::{Rng, TAU, cos, exp, ln, pow, sqrt};
use serde::{Deserialize, Serialize};

/// Options of [`Cma::new`] and [`cma_es`]; `None` takes the default noted on each field.
#[derive(Clone, Debug, Default)]
pub struct CmaOpts {
    /// lower bounds (default −∞)
    pub lo: Option<Vec<f64>>,
    /// upper bounds (default +∞)
    pub hi: Option<Vec<f64>>,
    /// initial step size, as a share of each bounded dimension's range (and in the number's own
    /// unit for an unbounded one). Default 0.3
    pub sigma: Option<f64>,
    /// candidates a generation (default 4 + ⌊3·ln n⌋)
    pub lambda: Option<usize>,
    /// default 1
    pub seed: Option<u32>,
    /// [`cma_es`] only: generations at most (default 1000)
    pub max_iter: Option<usize>,
    /// [`cma_es`] only: stop when the largest step, in normalised coordinates, is under this
    /// (default 1e-9)
    pub tol: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CmaResult {
    /// the best candidate that was evaluated
    pub x: Vec<f64>,
    pub fx: f64,
    /// where the search's mean ended: on a noisy function this is the better answer, the best
    /// single evaluation being partly luck
    pub mean: Vec<f64>,
    pub iters: usize,
}

/// The optimiser's whole state. See the module's comment.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Cma {
    /// per dimension: the bounds, when it has both
    bounds: Vec<Option<(f64, f64)>>,
    lambda: usize,
    seed: u32,
    /// uniform numbers drawn from the seed's generator so far
    draws: u64,
    generation: usize,
    evaluations: usize,
    /// all below in normalised coordinates
    mean: Vec<f64>,
    sigma: f64,
    /// the diagonal covariance
    variance: Vec<f64>,
    path_sigma: Vec<f64>,
    path_c: Vec<f64>,
    /// the candidates handed out by `ask` and not yet told
    pending: Vec<Vec<f64>>,
    best: Option<(Vec<f64>, f64)>,
    #[serde(skip)]
    rng: Option<Rng>,
}

/// The strategy's constants for `n` dimensions and `lambda` candidates.
struct Consts {
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    damps: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    chi_n: f64,
}

fn consts(n: usize, lambda: usize) -> Consts {
    let nf = n as f64;
    let mu = (lambda / 2).max(1);
    let raw: Vec<f64> = (0..mu).map(|i| ln(mu as f64 + 0.5) - ln(i as f64 + 1.0)).collect();
    let sum: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / sum).collect();
    let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
    let damps = 1.0 + 2.0 * (sqrt((mu_eff - 1.0) / (nf + 1.0)) - 1.0).max(0.0) + c_sigma;
    let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
    // a diagonal has n numbers to learn where a full matrix has n²/2: the learning rates are
    // (n + 2)/3 times the full CMA's
    let sep = (nf + 2.0) / 3.0;
    let c_1 = (sep * 2.0 / ((nf + 1.3) * (nf + 1.3) + mu_eff)).min(1.0);
    let c_mu = (sep * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0) * (nf + 2.0) + mu_eff)).min(1.0 - c_1);
    let chi_n = sqrt(nf) * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
    Consts { weights, mu_eff, c_sigma, damps, c_c, c_1, c_mu, chi_n }
}

/// `u` reflected into 0..1.
fn mirror(u: f64) -> f64 {
    if (0.0..=1.0).contains(&u) {
        return u;
    }
    let t = u.rem_euclid(2.0);
    if t > 1.0 { 2.0 - t } else { t }
}

impl Cma {
    /// A search starting at `x0` (put inside the bounds if it is not).
    pub fn new(x0: &[f64], opts: CmaOpts) -> Cma {
        let n = x0.len();
        assert!(n > 0, "cma: no dimensions");
        let lo = opts.lo.unwrap_or_else(|| vec![f64::NEG_INFINITY; n]);
        let hi = opts.hi.unwrap_or_else(|| vec![f64::INFINITY; n]);
        assert!(lo.len() == n && hi.len() == n, "cma: bounds of another length than x0");
        let bounds: Vec<Option<(f64, f64)>> = (0..n)
            .map(|i| {
                assert!(x0[i].is_finite(), "cma: x0[{i}] is not finite");
                let both = lo[i].is_finite() && hi[i].is_finite();
                assert!(both || (lo[i] == f64::NEG_INFINITY && hi[i] == f64::INFINITY), "cma: dimension {i} is bounded on one side only");
                assert!(!both || lo[i] < hi[i], "cma: dimension {i}: {} is not below {}", lo[i], hi[i]);
                both.then_some((lo[i], hi[i]))
            })
            .collect();
        let lambda = opts.lambda.unwrap_or(4 + (3.0 * ln(n as f64)).floor() as usize).max(2);
        let sigma = opts.sigma.unwrap_or(0.3);
        assert!(sigma > 0.0 && sigma.is_finite(), "cma: sigma {sigma}");
        let mut es = Cma {
            bounds,
            lambda,
            seed: opts.seed.unwrap_or(1),
            draws: 0,
            generation: 0,
            evaluations: 0,
            mean: Vec::new(),
            sigma,
            variance: vec![1.0; n],
            path_sigma: vec![0.0; n],
            path_c: vec![0.0; n],
            pending: Vec::new(),
            best: None,
            rng: None,
        };
        es.mean = (0..n).map(|i| es.encode(i, x0[i])).collect();
        es
    }

    fn encode(&self, i: usize, x: f64) -> f64 {
        match self.bounds[i] {
            Some((lo, hi)) => ((x - lo) / (hi - lo)).clamp(0.0, 1.0),
            None => x,
        }
    }

    fn decode(&self, u: &[f64]) -> Vec<f64> {
        u.iter()
            .enumerate()
            .map(|(i, &u)| match self.bounds[i] {
                // the ends exactly, whatever the rounding of lo + (hi − lo)
                Some((lo, hi)) => (lo + (hi - lo) * u).clamp(lo, hi),
                None => u,
            })
            .collect()
    }

    /// One standard normal number (Box–Muller on two uniforms).
    fn normal(&mut self) -> f64 {
        let (seed, draws) = (self.seed, self.draws);
        let rng = self.rng.get_or_insert_with(|| {
            // a loaded state: back to where the generator was
            let mut r = Rng::new(seed);
            for _ in 0..draws {
                r.next();
            }
            r
        });
        let (a, b) = (rng.next(), rng.next());
        self.draws += 2;
        sqrt(-2.0 * ln(1.0 - a)) * cos(TAU * b)
    }

    /// This generation's candidates, each inside the bounds. Asking again before `tell` gives the
    /// same ones.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        if self.pending.is_empty() {
            let n = self.mean.len();
            for _ in 0..self.lambda {
                let mut u = Vec::with_capacity(n);
                for i in 0..n {
                    let v = self.mean[i] + self.sigma * sqrt(self.variance[i]) * self.normal();
                    u.push(if self.bounds[i].is_some() { mirror(v) } else { v });
                }
                self.pending.push(u);
            }
        }
        self.pending.iter().map(|u| self.decode(u)).collect()
    }

    /// The values of the candidates `ask` gave, in its order (lower is better; a NaN ranks last).
    pub fn tell(&mut self, values: &[f64]) {
        assert!(!self.pending.is_empty(), "cma: tell without ask");
        assert_eq!(values.len(), self.pending.len(), "cma: one value for each candidate");
        let n = self.mean.len();
        let k = consts(n, self.lambda);
        let key = |i: usize| if values[i].is_nan() { f64::INFINITY } else { values[i] };
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
        let first = order[0];
        if values[first].is_finite() && self.best.as_ref().is_none_or(|b| values[first] < b.1) {
            self.best = Some((self.decode(&self.pending[first]), values[first]));
        }

        // the steps that were taken, in units of sigma, best first
        let steps: Vec<Vec<f64>> = order.iter().take(k.weights.len()).map(|&c| (0..n).map(|i| (self.pending[c][i] - self.mean[i]) / self.sigma).collect()).collect();
        let step: Vec<f64> = (0..n).map(|i| steps.iter().zip(&k.weights).map(|(y, w)| w * y[i]).sum()).collect();
        for i in 0..n {
            self.mean[i] += self.sigma * step[i];
            if self.bounds[i].is_some() {
                self.mean[i] = self.mean[i].clamp(0.0, 1.0);
            }
        }
        let cs = sqrt(k.c_sigma * (2.0 - k.c_sigma) * k.mu_eff);
        for i in 0..n {
            self.path_sigma[i] = (1.0 - k.c_sigma) * self.path_sigma[i] + cs * step[i] / sqrt(self.variance[i]);
        }
        let norm = sqrt(self.path_sigma.iter().map(|p| p * p).sum::<f64>());
        let decay = sqrt(1.0 - pow(1.0 - k.c_sigma, 2.0 * (self.generation + 1) as f64));
        let h = if norm / decay / k.chi_n < 1.4 + 2.0 / (n as f64 + 1.0) { 1.0 } else { 0.0 };
        let cc = sqrt(k.c_c * (2.0 - k.c_c) * k.mu_eff);
        for i in 0..n {
            self.path_c[i] = (1.0 - k.c_c) * self.path_c[i] + h * cc * step[i];
            let rank_mu: f64 = steps.iter().zip(&k.weights).map(|(y, w)| w * y[i] * y[i]).sum();
            self.variance[i] = (1.0 - k.c_1 - k.c_mu) * self.variance[i] + k.c_1 * (self.path_c[i] * self.path_c[i] + (1.0 - h) * k.c_c * (2.0 - k.c_c) * self.variance[i]) + k.c_mu * rank_mu;
        }
        self.sigma *= exp(((k.c_sigma / k.damps) * (norm / k.chi_n - 1.0)).min(1.0));
        // inside a box no step need be longer than the box (on a flat or very noisy function the
        // step size would otherwise grow without end)
        let widest = (0..n).filter(|&i| self.bounds[i].is_some()).map(|i| sqrt(self.variance[i])).fold(0.0, f64::max);
        if widest > 0.0 && self.sigma * widest > 1.0 {
            self.sigma = 1.0 / widest;
        }
        // and none may vanish or stop being a number
        self.sigma = self.sigma.max(1e-300);
        for v in &mut self.variance {
            *v = v.clamp(1e-300, 1e300);
        }
        self.evaluations += values.len();
        self.generation += 1;
        self.pending.clear();
    }

    /// Generations told so far.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Candidates told so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Candidates a generation.
    pub fn lambda(&self) -> usize {
        self.lambda
    }

    /// The step size (a share of a bounded dimension's range, before the dimension's own factor).
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Each dimension's step, in normalised coordinates: sigma times its own factor.
    pub fn steps(&self) -> Vec<f64> {
        self.variance.iter().map(|v| self.sigma * sqrt(*v)).collect()
    }

    /// The search's mean, as the caller's numbers.
    pub fn mean(&self) -> Vec<f64> {
        self.decode(&self.mean)
    }

    /// The best candidate told so far and its value.
    pub fn best(&self) -> Option<(&[f64], f64)> {
        self.best.as_ref().map(|b| (b.0.as_slice(), b.1))
    }
}

/// Separable CMA-ES minimiser with box bounds (mirrored), run to `max_iter` generations or `tol`.
pub fn cma_es(mut f: impl FnMut(&[f64]) -> f64, x0: &[f64], opts: CmaOpts) -> CmaResult {
    let max_iter = opts.max_iter.unwrap_or(1000);
    let tol = opts.tol.unwrap_or(1e-9);
    let mut es = Cma::new(x0, opts);
    while es.generation() < max_iter && es.steps().iter().copied().fold(0.0, f64::max) >= tol {
        let values: Vec<f64> = es.ask().iter().map(|x| f(x)).collect();
        es.tell(&values);
    }
    let (x, fx) = match es.best() {
        Some((x, fx)) => (x.to_vec(), fx),
        None => (es.mean(), f64::NAN),
    };
    CmaResult { x, fx, mean: es.mean(), iters: es.generation() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Σ 10^(6·i/(n−1))·xᵢ²: a valley a million times steeper one way than the other.
    fn ellipsoid(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        x.iter().enumerate().map(|(i, v)| pow(10.0, 6.0 * i as f64 / (n - 1.0)) * v * v).sum()
    }

    #[test]
    fn cma_finds_the_minimum_of_a_40_dimensional_ellipsoid() {
        let r = cma_es(ellipsoid, &[3.0; 40], CmaOpts { lo: Some(vec![-5.0; 40]), hi: Some(vec![5.0; 40]), max_iter: Some(6000), tol: Some(1e-12), ..Default::default() });
        assert!(r.fx < 1e-8, "f {} after {} generations", r.fx, r.iters);
        assert!(r.mean.iter().all(|v| v.abs() < 1e-3), "{:?}", r.mean);
        // and without bounds, from a step in the numbers' own unit
        let free = cma_es(ellipsoid, &[3.0; 40], CmaOpts { sigma: Some(2.0), max_iter: Some(6000), tol: Some(1e-12), ..Default::default() });
        assert!(free.fx < 1e-8, "unbounded: f {} after {} generations", free.fx, free.iters);
    }

    /// A sphere around a point whose first coordinate is on the lower bound and whose second is
    /// outside the upper one, with noise on every evaluation.
    fn noisy_sphere(noise: &mut Rng, x: &[f64]) -> f64 {
        let target = |i: usize| match i {
            0 => 2.0,
            1 => 9.0,
            _ => 4.0 + 0.1 * i as f64,
        };
        x.iter().enumerate().map(|(i, v)| (v - target(i)) * (v - target(i))).sum::<f64>() + 0.02 * noise.gauss()
    }

    #[test]
    fn cma_finds_the_minimum_of_a_noisy_sphere_with_the_optimum_on_a_bound() {
        let n = 10;
        let (lo, hi) = (vec![2.0; n], vec![6.0; n]);
        let mut noise = Rng::new(99);
        let mut es = Cma::new(&vec![5.5; n], CmaOpts { lo: Some(lo.clone()), hi: Some(hi.clone()), seed: Some(3), ..Default::default() });
        for _ in 0..400 {
            let xs = es.ask();
            assert!(xs.iter().all(|x| x.iter().zip(lo.iter().zip(&hi)).all(|(v, (l, h))| l <= v && v <= h)), "a candidate outside the bounds");
            let values: Vec<f64> = xs.iter().map(|x| noisy_sphere(&mut noise, x)).collect();
            es.tell(&values);
        }
        let m = es.mean();
        assert!(m[0] < 2.15, "the coordinate whose optimum is the lower bound: {}", m[0]);
        assert!(m[1] > 5.85, "the coordinate whose optimum is past the upper bound: {}", m[1]);
        for i in 2..n {
            assert!((m[i] - (4.0 + 0.1 * i as f64)).abs() < 0.15, "coordinate {i}: {}", m[i]);
        }
    }

    #[test]
    fn cma_takes_the_same_path_from_the_same_seed_and_another_from_another() {
        let path = |seed: u32| -> Vec<Vec<f64>> {
            let mut es = Cma::new(&[1.0; 6], CmaOpts { lo: Some(vec![-2.0; 6]), hi: Some(vec![2.0; 6]), seed: Some(seed), ..Default::default() });
            let mut seen = Vec::new();
            for _ in 0..30 {
                let xs = es.ask();
                es.tell(&xs.iter().map(|x| ellipsoid(x)).collect::<Vec<_>>());
                seen.extend(xs);
            }
            seen
        };
        assert_eq!(path(11), path(11));
        assert_ne!(path(11), path(12));
    }

    #[test]
    fn cma_continues_identically_after_a_save_and_a_load_mid_run() {
        let opts = || CmaOpts { lo: Some(vec![-2.0; 8]), hi: Some(vec![2.0; 8]), seed: Some(5), ..Default::default() };
        let f = |x: &[f64]| ellipsoid(x) + x[0];
        let mut whole = Cma::new(&[1.0; 8], opts());
        let mut broken = Cma::new(&[1.0; 8], opts());
        for g in 0..40 {
            if g == 17 {
                // between generations
                broken = serde_json::from_str(&serde_json::to_string(&broken).unwrap()).unwrap();
            }
            let (a, mut b) = (whole.ask(), broken.ask());
            if g == 23 {
                // and between an ask and its tell: the candidates come back as they were
                broken = serde_json::from_str(&serde_json::to_string(&broken).unwrap()).unwrap();
                b = broken.ask();
            }
            assert_eq!(a, b, "generation {g}");
            whole.tell(&a.iter().map(|x| f(x)).collect::<Vec<_>>());
            broken.tell(&b.iter().map(|x| f(x)).collect::<Vec<_>>());
        }
        assert_eq!(serde_json::to_string(&whole).unwrap(), serde_json::to_string(&broken).unwrap());
        assert_eq!(whole.best().unwrap().1, broken.best().unwrap().1);
        assert_eq!((whole.generation(), whole.evaluations()), (40, 40 * whole.lambda()));
    }

    #[test]
    fn cma_ranks_a_nan_last_and_keeps_its_state_finite() {
        let mut es = Cma::new(&[0.5; 4], CmaOpts { lo: Some(vec![0.0; 4]), hi: Some(vec![1.0; 4]), seed: Some(2), ..Default::default() });
        for g in 0..50 {
            let xs = es.ask();
            let values: Vec<f64> = xs.iter().enumerate().map(|(i, x)| if (i + g) % 3 == 0 { f64::NAN } else { x.iter().map(|v| (v - 0.2) * (v - 0.2)).sum() }).collect();
            es.tell(&values);
        }
        assert!(es.best().unwrap().1.is_finite());
        assert!(es.mean().iter().all(|v| (v - 0.2).abs() < 0.05), "{:?}", es.mean());
        // every number of the state is finite, so JSON holds it
        let back: Cma = serde_json::from_str(&serde_json::to_string(&es).unwrap()).unwrap();
        assert_eq!(back.mean(), es.mean());
        assert_eq!(mirror(1.25), 0.75);
        assert_eq!(mirror(-0.25), 0.25);
        assert_eq!(mirror(2.5), 0.5);
    }
}
