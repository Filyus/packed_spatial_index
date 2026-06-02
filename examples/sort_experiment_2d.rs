//! Experiment: compare sorting methods during index build:
//! comparison-based `sort_unstable_by_key` (pdqsort) versus LSD radix sort.
//! Run: `cargo run --release --example sort_experiment`

use std::time::Instant;

use packed_spatial_index::experimental::ExperimentalSortKey2D;
use packed_spatial_index::{Box2D, Index2DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;

fn gen_boxes(n: usize) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(0xB0B);
    (0..n)
        .map(|_| {
            let cx: f64 = rng.random_range(0.0..10_000.0);
            let cy: f64 = rng.random_range(0.0..10_000.0);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            [cx, cy, cx + w, cy + h]
        })
        .collect()
}

fn time_build(boxes: &[[f64; 4]], radix: bool, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let mut b = Index2DBuilder::new(boxes.len())
            .node_size(NODE_SIZE)
            .experimental_sort_key(ExperimentalSortKey2D::HilbertLut)
            .radix(radix);
        for r in boxes {
            b.add(Box2D::new(r[0], r[1], r[2], r[3]));
        }
        let idx = b.finish().unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(idx.num_items());
        best = best.min(el);
    }
    best
}

fn main() {
    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "N", "pdqsort", "radix", "speedup"
    );
    println!("{}", "-".repeat(52));
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let boxes = gen_boxes(n);
        let reps = if n >= 1_000_000 { 10 } else { 200 };
        let pdq = time_build(&boxes, false, reps);
        let radix = time_build(&boxes, true, reps);
        println!(
            "{:>10} | {:>9.3} ms | {:>9.3} ms | {:>8.2}x",
            n,
            pdq,
            radix,
            pdq / radix
        );
    }
}
