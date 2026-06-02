//! Experiment: adaptive and forced parallel 3D index builds versus serial builds.
//! Run: `cargo run --release --example parallel_experiment_3d`

use std::time::Instant;

use packed_spatial_index::experimental::ExperimentalSortKey3D;
use packed_spatial_index::{Bounds3D, Index3D, Index3DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;

#[derive(Clone, Copy)]
enum BuildMode {
    Serial,
    ParallelAuto,
    ParallelForced,
}

fn gen_boxes(n: usize) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(0x3D0B);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Bounds3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn build(boxes: &[Bounds3D], mode: BuildMode) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey3D::Hilbert);
    builder = match mode {
        BuildMode::Serial => builder.parallel(false),
        BuildMode::ParallelAuto => builder.parallel(true),
        BuildMode::ParallelForced => builder.parallel(true).parallel_min_items(0),
    };
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

fn time_build(boxes: &[Bounds3D], mode: BuildMode, reps: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        let index = build(boxes, mode);
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(index.num_items());
    }
    best
}

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());
    println!("auto parallel threshold: 50000 items\n");

    {
        let boxes = gen_boxes(20_000);
        let serial = build(&boxes, BuildMode::Serial);
        let parallel = build(&boxes, BuildMode::ParallelForced);
        let mut rng = StdRng::seed_from_u64(0x3D77);
        for _ in 0..300 {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let query = Bounds3D::new(x, y, z, x + 150.0, y + 150.0, z + 150.0);
            let mut a = serial.search(query);
            let mut b = parallel.search(query);
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "parallel build produced different results");
        }
        println!("correctness: parallel build == serial build OK\n");
    }

    println!(
        "{:>10} | {:>12} | {:>12} | {:>12} | {:>10}",
        "N", "serial", "auto", "forced", "auto/serial"
    );
    println!("{}", "-".repeat(67));
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let boxes = gen_boxes(n);
        let reps = if n >= 1_000_000 {
            10
        } else if n >= 100_000 {
            50
        } else {
            200
        };
        let serial = time_build(&boxes, BuildMode::Serial, reps);
        let auto = time_build(&boxes, BuildMode::ParallelAuto, reps);
        let forced = time_build(&boxes, BuildMode::ParallelForced, reps);
        println!(
            "{n:>10} | {serial:>9.3} ms | {auto:>9.3} ms | {forced:>9.3} ms | {:>8.2}x",
            serial / auto
        );
    }
}
