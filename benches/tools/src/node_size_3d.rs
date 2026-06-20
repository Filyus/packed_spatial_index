//! Local performance tool: effect of `node_size` on 3D builds and SIMD queries.
//! Run: `cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_3d`

use std::time::Instant;

use packed_spatial_index::benchmark_support::SortKey3DStrategy;
use packed_spatial_index::{Box3D, Index3DBuilder};
use psi_perf::emit;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 100_000;
const NQ: usize = 1_000;
const REPS_Q: usize = 200;
const REPS_B: usize = 50;

fn main() {
    psi_perf::pin_from_env();
    let mut rng = StdRng::seed_from_u64(0x3D0B);
    let boxes: Vec<Box3D> = (0..N)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect();

    let mut qrng = StdRng::seed_from_u64(0x3ACE);
    let queries: Vec<Box3D> = (0..NQ)
        .map(|_| {
            let x: f64 = qrng.random_range(0.0..10_000.0);
            let y: f64 = qrng.random_range(0.0..10_000.0);
            let z: f64 = qrng.random_range(0.0..10_000.0);
            let w: f64 = qrng.random_range(10.0..200.0);
            Box3D::new(x, y, z, x + w, y + w, z + w)
        })
        .collect();

    emit(&serde_json::json!({ "tool": "node_size_3d_meta", "n": N, "nq": NQ }));

    for &node_size in &[4usize, 8, 16, 32, 64] {
        let mut build_best = f64::INFINITY;
        for _ in 0..REPS_B {
            let start = Instant::now();
            let mut builder = Index3DBuilder::new(N)
                .node_size(node_size)
                .sort_key_strategy(SortKey3DStrategy::Hilbert);
            for &bounds in &boxes {
                builder.add(bounds);
            }
            let index = builder.finish().unwrap();
            build_best = build_best.min(start.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(index.num_items());
        }

        let mut builder = Index3DBuilder::new(N)
            .node_size(node_size)
            .sort_key_strategy(SortKey3DStrategy::Hilbert);
        for &bounds in &boxes {
            builder.add(bounds);
        }
        let simd = builder.finish_simd().unwrap();
        let (mut out, mut stack) = (Vec::new(), Vec::new());

        let mut simd_best = f64::INFINITY;
        for _ in 0..REPS_Q {
            let start = Instant::now();
            let mut total = 0usize;
            for &query in &queries {
                simd.search_simd(query, &mut out, &mut stack);
                total += out.len();
            }
            std::hint::black_box(total);
            simd_best = simd_best.min(start.elapsed().as_secs_f64() * 1e6);
        }

        let mut avx_best = f64::INFINITY;
        for _ in 0..REPS_Q {
            let start = Instant::now();
            let mut total = 0usize;
            for &query in &queries {
                simd.search_avx512(query, &mut out, &mut stack);
                total += out.len();
            }
            std::hint::black_box(total);
            avx_best = avx_best.min(start.elapsed().as_secs_f64() * 1e6);
        }

        emit(&serde_json::json!({
            "tool": "node_size_3d",
            "node_size": node_size,
            "build_serial_ms": build_best,
            "query_simd_us": simd_best,
            "query_avx512_us": avx_best,
        }));
    }
}
