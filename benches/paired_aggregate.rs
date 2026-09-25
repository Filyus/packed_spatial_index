//! `Index2D::aggregate` against `count` (a control the aggregate kernel does
//! not touch) and the search-and-fold workaround, interleaved in one binary.
//! To compare two versions of the kernel, build this bench in both trees and
//! alternate the two binaries (A B A B): the `count` arm measures the
//! binary-to-binary bias, and the aggregate/count ratio is the figure.
//!
//! Run:
//!   BENCH_PIN_CORE=8 cargo bench --bench paired_aggregate

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

fn build() -> Index2D {
    let mut rng = StdRng::seed_from_u64(1);
    let mut builder = Index2DBuilder::new(COUNT).node_size(NODE_SIZE);
    for _ in 0..COUNT {
        let x: f64 = rng.random_range(0.0..EXTENT);
        let y: f64 = rng.random_range(0.0..EXTENT);
        let w: f64 = rng.random_range(0.0..MAX_SIZE);
        let h: f64 = rng.random_range(0.0..MAX_SIZE);
        builder.add(Box2D::new(x, y, x + w, y + h));
    }
    let scalars: Vec<f64> = (0..COUNT).map(|i| (i % 101) as f64).collect();
    let masks: Vec<u64> = (0..COUNT).map(|i| 1u64 << (i % 61)).collect();
    builder
        .aggregate_scalar(&scalars)
        .aggregate_mask(&masks)
        .finish()
        .unwrap()
}

fn main() {
    pin::pin_from_env();
    let index = build();
    // Small outputs get 2000 windows so a Zen 4 / Zen 5 predictor cannot
    // learn the set (kb:task/191); the ~10 000-hit side keeps 200.
    for &(side, count) in &[(1_000.0f64, 2000usize), (10_000.0, 2000), (100_000.0, 200)] {
        let mut rng = StdRng::seed_from_u64(2);
        let windows: Vec<Box2D> = (0..count)
            .map(|_| {
                let x: f64 = rng.random_range(0.0..EXTENT - side);
                let y: f64 = rng.random_range(0.0..EXTENT - side);
                Box2D::new(x, y, x + side, y + side)
            })
            .collect();
        let hits: usize = windows.iter().map(|&w| index.count(w)).sum::<usize>() / windows.len();
        let label = format!("side {side} (~{hits} hits) x{count}");
        let mut arms = vec![
            paired::arm("count (control)", || {
                let mut t = 0;
                for &w in black_box(&windows) {
                    t += index.count(w);
                }
                t
            }),
            paired::arm("search + fold", || {
                let mut t = 0.0f64;
                for &w in black_box(&windows) {
                    for id in index.search(w) {
                        t += (id % 101) as f64;
                    }
                }
                t as usize
            }),
            paired::arm("aggregate", || {
                let mut t = 0u64;
                for &w in black_box(&windows) {
                    let a = index.aggregate(w).unwrap();
                    t += a.count
                        + a.sum.unwrap_or(0.0) as u64
                        + a.mask.unwrap_or(0).count_ones() as u64;
                }
                t as usize
            }),
        ];
        paired::run(&label, &mut arms, "count (control)");
    }
}
