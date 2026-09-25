//! `search_heaviest` (top-k by the aggregate scalar) against collecting the
//! window and sorting it, interleaved in one binary. Two baselines: a full sort
//! of the hits by weight; `select_nth_unstable` plus a sort of the `k`
//! (what a careful caller would write). Ratios are against the full sort.
//!
//! Run:
//!   BENCH_PIN_CORE=2 cargo bench --bench paired_heaviest

use std::hint::black_box;

use packed_spatial_index::{Box2D, Index2D, Index2DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const NODE_SIZE: usize = 16;
const COUNT: usize = 1_000_000;
const EXTENT: f64 = 1_000_000.0;
const MAX_SIZE: f64 = 50.0;

fn build() -> (Index2D, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(1);
    let mut builder = Index2DBuilder::new(COUNT).node_size(NODE_SIZE);
    for _ in 0..COUNT {
        let x: f64 = rng.random_range(0.0..EXTENT);
        let y: f64 = rng.random_range(0.0..EXTENT);
        let w: f64 = rng.random_range(0.0..MAX_SIZE);
        let h: f64 = rng.random_range(0.0..MAX_SIZE);
        builder.add(Box2D::new(x, y, x + w, y + h));
    }
    let weights: Vec<f64> = (0..COUNT).map(|_| rng.random_range(0.0..1.0)).collect();
    let index = builder.aggregate_scalar(&weights).finish().unwrap();
    (index, weights)
}

/// Heaviest first, ties by index: the order `search_heaviest` promises.
fn by_weight(weights: &[f64]) -> impl Fn(&usize, &usize) -> std::cmp::Ordering + '_ {
    |&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b))
}

fn main() {
    pin::pin_from_env();
    let (index, weights) = build();
    for &(side, windows_n) in &[
        (10_000.0f64, 2000usize),
        (100_000.0, 200),
        (300_000.0, 20),
        (1_000_000.0, 4),
    ] {
        let mut rng = StdRng::seed_from_u64(2);
        let windows: Vec<Box2D> = (0..windows_n)
            .map(|_| {
                let x: f64 = rng.random_range(0.0..=EXTENT - side);
                let y: f64 = rng.random_range(0.0..=EXTENT - side);
                Box2D::new(x, y, x + side, y + side)
            })
            .collect();
        let hits: usize = windows.iter().map(|&w| index.count(w)).sum::<usize>() / windows.len();
        for k in [10usize, 100] {
            // The three arms agree before they are timed.
            for &w in &windows {
                let mut all = index.search(w);
                all.sort_by(by_weight(&weights));
                all.truncate(k);
                assert_eq!(index.search_heaviest(w, k).unwrap(), all);
            }
            let label = format!("side {side} (~{hits} hits) x{windows_n}, k = {k}");
            let mut buf = Vec::new();
            let mut buf2 = Vec::new();
            let mut arms = vec![
                paired::arm("search + sort", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        index.search_into(w, &mut buf);
                        buf.sort_by(by_weight(&weights));
                        t += buf.iter().take(k).sum::<usize>();
                    }
                    t
                }),
                paired::arm("search + select_nth", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        index.search_into(w, &mut buf2);
                        if buf2.len() > k {
                            buf2.select_nth_unstable_by(k, by_weight(&weights));
                            buf2.truncate(k);
                        }
                        buf2.sort_by(by_weight(&weights));
                        t += buf2.iter().sum::<usize>();
                    }
                    t
                }),
                paired::arm("search_heaviest", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        t += index.search_heaviest(w, k).unwrap().iter().sum::<usize>();
                    }
                    t
                }),
            ];
            paired::run(&label, &mut arms, "search + sort");
        }
    }
}
