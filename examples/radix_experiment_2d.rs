//! Experiment: radix-sort digit width (probing the memory-bound build ceiling).
//! Sort an isolated array of random `(key_u32, index)` pairs.
//! Run: `cargo run --release --example radix_experiment`

use std::time::Instant;

use packed_spatial_index::experimental::radix_sort_pairs;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn gen_pairs(n: usize) -> Vec<(u32, u32)> {
    let mut rng = StdRng::seed_from_u64(0x5A17);
    (0..n).map(|i| (rng.random::<u32>(), i as u32)).collect()
}

fn time<F: FnMut(&mut Vec<(u32, u32)>)>(base: &[(u32, u32)], reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let mut a = base.to_vec();
        let t = Instant::now();
        f(&mut a);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(a.first());
    }
    best
}

fn main() {
    println!(
        "{:>10} | {:>10} | {:>12} | {:>12} | {:>12}",
        "N", "pdqsort", "radix-8 (4p)", "radix-11 (3p)", "radix-16 (2p)"
    );
    println!("{}", "-".repeat(66));
    for &n in &[100_000usize, 1_000_000, 5_000_000] {
        let base = gen_pairs(n);
        let reps = if n >= 1_000_000 { 15 } else { 150 };
        // correctness baseline: pdqsort-sorted data
        let mut sorted = base.clone();
        sorted.sort_unstable_by_key(|&(k, _)| k);

        let pdq = time(&base, reps, |a| a.sort_unstable_by_key(|&(k, _)| k));
        let r8 = time(&base, reps, |a| radix_sort_pairs(a, 8));
        let r11 = time(&base, reps, |a| radix_sort_pairs(a, 11));
        let r16 = time(&base, reps, |a| radix_sort_pairs(a, 16));

        // sanity: radix produces the same key order as pdqsort
        for bits in [8u32, 11, 16] {
            let mut a = base.clone();
            radix_sort_pairs(&mut a, bits);
            let ka: Vec<u32> = a.iter().map(|p| p.0).collect();
            let ks: Vec<u32> = sorted.iter().map(|p| p.0).collect();
            assert_eq!(ka, ks, "radix-{bits} produced an invalid order");
        }

        println!(
            "{:>10} | {:>7.3}ms | {:>9.3}ms | {:>9.3}ms | {:>9.3}ms",
            n, pdq, r8, r11, r16
        );
    }
}
