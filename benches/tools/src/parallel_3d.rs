//! Local performance tool: adaptive and forced parallel 3D index builds versus serial builds.
//! Run: `cargo run --release --manifest-path benches/tools/Cargo.toml --bin parallel_3d`

use std::time::Instant;

use packed_spatial_index::benchmark_support::SortKey3DStrategy;
use packed_spatial_index::{Box3D, Index3D, Index3DBuilder};
use psi_perf::emit;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;

#[derive(Clone, Copy)]
enum BuildMode {
    Serial,
    ParallelAuto,
    ParallelForced,
}

fn gen_boxes(n: usize) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(0x3D0B);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn build(boxes: &[Box3D], mode: BuildMode) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey3DStrategy::Hilbert);
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

fn time_build(boxes: &[Box3D], mode: BuildMode, reps: usize) -> f64 {
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
    psi_perf::pin_from_env();
    emit(&serde_json::json!({
        "tool": "parallel_3d_meta",
        "rayon_threads": rayon::current_num_threads(),
        "node_size": NODE_SIZE,
    }));

    {
        let boxes = gen_boxes(20_000);
        let serial = build(&boxes, BuildMode::Serial);
        let parallel = build(&boxes, BuildMode::ParallelForced);
        let mut rng = StdRng::seed_from_u64(0x3D77);
        for _ in 0..300 {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let query = Box3D::new(x, y, z, x + 150.0, y + 150.0, z + 150.0);
            let mut a = serial.search(query);
            let mut b = parallel.search(query);
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "parallel build produced different results");
        }
        emit(&serde_json::json!({ "tool": "parallel_3d_check", "ok": true }));
    }

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
        emit(&serde_json::json!({
            "tool": "parallel_3d",
            "n": n,
            "serial_ms": serial,
            "auto_ms": auto,
            "forced_ms": forced,
        }));
    }
}
