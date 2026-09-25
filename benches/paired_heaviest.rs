//! `search_heaviest` (top-k by the aggregate scalar) and the two forms behind
//! its switch, against collecting the window and selecting the `k` heaviest
//! from it, interleaved in one binary.
//!
//! Arms, all against `search + select_nth` (`search_into`, then
//! `select_nth_unstable` and a sort of the `k`, what a careful caller writes):
//! the shipping `search_heaviest`; its best-first descent and its
//! collect-and-select form, forced; and `search_heaviest_each` in both forms
//! with a visitor that stops at `k`.
//!
//! Windows run from a fraction of a hit to tens of thousands, `k` over
//! 1 / 10 / 100 / 1000. Each window class gets its own query set, big enough
//! that the branch predictor cannot learn it (kb:observation/534): 10 000
//! windows below 100 expected hits, 2000 up to 3000, 400 above.
//!
//! Run:
//!   BENCH_PIN_CORE=2 cargo bench --bench paired_heaviest
//! `HEAVIEST_HITS` (e.g. "100 1000") and `HEAVIEST_K` (e.g. "10") narrow the
//! grid.

use std::hint::black_box;
use std::ops::ControlFlow;

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
/// Expected hits per window: a window of side `s` meets about
/// `(s + MAX_SIZE / 2)^2 * COUNT / EXTENT^2` items.
const HITS: [f64; 12] = [
    0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0,
];
const KS: [usize; 4] = [1, 10, 100, 1000];

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

fn env_list<T: std::str::FromStr>(key: &str) -> Option<Vec<T>> {
    let v = std::env::var(key).ok()?;
    Some(
        v.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect(),
    )
}

/// The first `k` ids of `search_heaviest_each` in the form `COLLECT` names.
fn each_k<const COLLECT: bool>(index: &Index2D, w: Box2D, k: usize) -> usize {
    let (mut t, mut n) = (0, 0);
    let _ = index.search_heaviest_each_forced::<COLLECT, _, _, _>(w, |id, _| {
        t += id;
        n += 1;
        if n == k {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    t
}

fn main() {
    pin::pin_from_env();
    let (index, weights) = build();
    let hits_grid = env_list::<f64>("HEAVIEST_HITS").unwrap_or(HITS.to_vec());
    let ks = env_list::<usize>("HEAVIEST_K").unwrap_or(KS.to_vec());
    for &target in &hits_grid {
        let side = ((target * EXTENT * EXTENT / COUNT as f64).sqrt() - MAX_SIZE / 2.0).max(0.0);
        let windows_n = if target < 100.0 {
            10_000
        } else if target <= 3_000.0 {
            2_000
        } else {
            400
        };
        let mut rng = StdRng::seed_from_u64(2);
        let windows: Vec<Box2D> = (0..windows_n)
            .map(|_| {
                let x: f64 = rng.random_range(0.0..=EXTENT - side);
                let y: f64 = rng.random_range(0.0..=EXTENT - side);
                Box2D::new(x, y, x + side, y + side)
            })
            .collect();
        let hits = windows.iter().map(|&w| index.count(w)).sum::<usize>() as f64 / windows_n as f64;
        for &k in &ks {
            // Every arm agrees before any is timed.
            for &w in windows.iter().step_by(7) {
                let mut all = index.search(w);
                all.sort_by(by_weight(&weights));
                all.truncate(k);
                assert_eq!(index.search_heaviest(w, k).unwrap(), all);
                assert_eq!(index.search_heaviest_forced::<true, _>(w, k).unwrap(), all);
                assert_eq!(index.search_heaviest_forced::<false, _>(w, k).unwrap(), all);
                let sum: usize = all.iter().sum();
                assert_eq!(each_k::<true>(&index, w, k), sum);
                assert_eq!(each_k::<false>(&index, w, k), sum);
            }
            let label = format!("side {side:.0} (~{hits:.1} hits) x{windows_n}, k = {k}");
            let mut buf = Vec::new();
            let mut arms = vec![
                paired::arm("search + select_nth", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        index.search_into(w, &mut buf);
                        if buf.len() > k {
                            buf.select_nth_unstable_by(k, by_weight(&weights));
                            buf.truncate(k);
                        }
                        buf.sort_by(by_weight(&weights));
                        t += buf.iter().sum::<usize>();
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
                paired::arm("heaviest: best-first", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        let ids = index.search_heaviest_forced::<false, _>(w, k).unwrap();
                        t += ids.iter().sum::<usize>();
                    }
                    t
                }),
                paired::arm("heaviest: collect", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        let ids = index.search_heaviest_forced::<true, _>(w, k).unwrap();
                        t += ids.iter().sum::<usize>();
                    }
                    t
                }),
                paired::arm("each: best-first", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        t += each_k::<false>(&index, w, k);
                    }
                    t
                }),
                paired::arm("each: collect", || {
                    let mut t = 0;
                    for &w in black_box(&windows) {
                        t += each_k::<true>(&index, w, k);
                    }
                    t
                }),
            ];
            paired::run(&label, &mut arms, "search + select_nth");
        }
    }
}
