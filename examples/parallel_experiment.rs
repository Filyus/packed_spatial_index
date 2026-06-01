//! Experiment: adaptive and forced parallel index builds (rayon)
//! versus single-threaded radix builds.
//! NOTE: the multi-threaded variant changes the comparison base from the single-threaded crate;
//! this demonstrates the speedup ceiling, not a strict algorithm-to-algorithm comparison.
//! Run: `cargo run --release --example parallel_experiment`

use std::time::Instant;

use packed_spatial_index::experimental::ExperimentalSortKey2D;
use packed_spatial_index::{Bounds2D, Index2DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;

#[derive(Clone, Copy)]
enum BuildMode {
    Serial,
    ParallelAuto,
    ParallelForced,
}

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

fn build(boxes: &[[f64; 4]], mode: BuildMode) -> packed_spatial_index::Index2D {
    let mut b = Index2DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey2D::HilbertLut);
    b = match mode {
        BuildMode::Serial => b.parallel(false),
        BuildMode::ParallelAuto => b.parallel(true),
        BuildMode::ParallelForced => b.parallel(true).parallel_min_items(0),
    };
    for r in boxes {
        b.add(Bounds2D::new(r[0], r[1], r[2], r[3]));
    }
    b.finish().unwrap()
}

fn time_build(boxes: &[[f64; 4]], mode: BuildMode, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let idx = build(boxes, mode);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(idx.num_items());
    }
    best
}

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());
    println!("auto parallel threshold: 50000 items\n");
    // sanity: parallel and serial builds produce identical query results
    {
        let boxes = gen_boxes(20_000);
        let s = build(&boxes, BuildMode::Serial);
        let p = build(&boxes, BuildMode::ParallelForced);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..300 {
            let qx: f64 = rng.random_range(0.0..10_000.0);
            let qy: f64 = rng.random_range(0.0..10_000.0);
            let query = Bounds2D::new(qx, qy, qx + 150.0, qy + 150.0);
            let mut a = s.search(query);
            let mut b = p.search(query);
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "parallel build produced different results!");
        }
        println!("correctness: parallel build == serial build OK\n");
    }

    println!(
        "{:>10} | {:>12} | {:>12} | {:>12} | {:>10}",
        "N", "serial", "auto", "forced", "auto/serial"
    );
    println!("{}", "-".repeat(67));
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000, 5_000_000] {
        let boxes = gen_boxes(n);
        let reps = if n >= 1_000_000 {
            10
        } else if n >= 100_000 {
            50
        } else {
            200
        };
        let s = time_build(&boxes, BuildMode::Serial, reps);
        let auto = time_build(&boxes, BuildMode::ParallelAuto, reps);
        let forced = time_build(&boxes, BuildMode::ParallelForced, reps);
        println!(
            "{:>10} | {:>9.3} ms | {:>9.3} ms | {:>9.3} ms | {:>8.2}x",
            n,
            s,
            auto,
            forced,
            s / auto
        );
    }
}
