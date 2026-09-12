//! A paired, interleaved A/B harness: every round times each arm once, in
//! order, so `A B C A B C …` — never `A A A B B B`. The report is the
//! distribution of the *per-round* ratios against a named reference arm,
//! which cancels whatever the machine did during that round (thermal state,
//! a background task), plus each arm's own spread so a difference smaller
//! than the spread reads as "not resolved" rather than as a change.
//!
//! Arms run in one binary, so a comparison here carries no build-to-build
//! layout term; put the old and the new implementation behind two arms and
//! keep an arm the change cannot touch as the control.
//!
//! Included by a bench via `#[path = "support/paired.rs"] mod paired;`.
//! `PAIRED_ROUNDS` (default 15) and `PAIRED_REPS` (default 5, the timed
//! repetitions per arm per round, of which the minimum is kept) are env
//! knobs; pair it with `BENCH_PIN_CORE` from `pin.rs`.

use std::time::Instant;

#[allow(dead_code)]
pub struct Arm<'a> {
    pub name: &'a str,
    pub run: Box<dyn FnMut() -> usize + 'a>,
}

#[allow(dead_code)]
pub fn arm<'a>(name: &'a str, run: impl FnMut() -> usize + 'a) -> Arm<'a> {
    Arm {
        name,
        run: Box::new(run),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Time every arm `rounds` times, interleaved, and print the report.
/// `reference` names the arm the ratios are taken against.
#[allow(dead_code)]
pub fn run(title: &str, arms: &mut [Arm<'_>], reference: &str) {
    let rounds = env_usize("PAIRED_ROUNDS", 15);
    let reps = env_usize("PAIRED_REPS", 5);
    let n = arms.len();
    let mut times = vec![Vec::with_capacity(rounds); n];
    let mut checksum = vec![0usize; n];

    // One untimed pass per arm: page in the data and warm the predictor.
    for arm in arms.iter_mut() {
        std::hint::black_box((arm.run)());
    }

    for _ in 0..rounds {
        for (i, arm) in arms.iter_mut().enumerate() {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t0 = Instant::now();
                let out = std::hint::black_box((arm.run)());
                let dt = t0.elapsed().as_secs_f64() * 1e6;
                checksum[i] = out;
                best = best.min(dt);
            }
            times[i].push(best);
        }
    }

    let r = arms
        .iter()
        .position(|a| a.name == reference)
        .expect("reference arm exists");
    println!("== {title}: {rounds} rounds x {reps} reps, min per round, us per arm run");
    println!(
        "{:<36} {:>10} {:>10} {:>10}   {:<22} {:>8}",
        "arm", "median", "min", "max", "ratio vs ref [min..max]", "result"
    );
    for (i, arm) in arms.iter().enumerate() {
        let mut t = times[i].clone();
        let med = median(&mut t);
        let (lo, hi) = (t[0], t[t.len() - 1]);
        let mut ratios: Vec<f64> = times[i].iter().zip(&times[r]).map(|(a, b)| a / b).collect();
        let rmed = median(&mut ratios);
        let (rlo, rhi) = (ratios[0], ratios[ratios.len() - 1]);
        println!(
            "{:<36} {:>10.1} {:>10.1} {:>10.1}   {:>6.3} [{:.3}..{:.3}]   {:>8}",
            arm.name, med, lo, hi, rmed, rlo, rhi, checksum[i]
        );
    }
}
