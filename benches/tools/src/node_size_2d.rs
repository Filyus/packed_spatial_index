//! Local performance tool: effect of `node_size` on builds and queries (AoS / AVX2 / AVX-512).
//! Larger node_size makes the tree shallower (less traversal), but puts more boxes per node (more
//! checks); for SIMD, larger nodes amortize better. Search for the optimum.
//! Run: `cargo run --release --manifest-path benches/tools/Cargo.toml --bin node_size_2d`

use std::time::Instant;

use packed_spatial_index::benchmark_support::SortKey2DStrategy;
use packed_spatial_index::{Box2D, Index2DBuilder};
use psi_perf::emit;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 100_000;
const NQ: usize = 1_000;
const REPS_Q: usize = 200;
const REPS_B: usize = 50;

fn main() {
    psi_perf::pin_from_env();
    let mut rng = StdRng::seed_from_u64(0xB0B);
    let boxes: Vec<[f64; 4]> = (0..N)
        .map(|_| {
            let cx: f64 = rng.random_range(0.0..10_000.0);
            let cy: f64 = rng.random_range(0.0..10_000.0);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            [cx, cy, cx + w, cy + h]
        })
        .collect();
    let mut qrng = StdRng::seed_from_u64(0xACE);
    let queries: Vec<[f64; 4]> = (0..NQ)
        .map(|_| {
            let qx: f64 = qrng.random_range(0.0..10_000.0);
            let qy: f64 = qrng.random_range(0.0..10_000.0);
            let qw: f64 = qrng.random_range(10.0..200.0);
            let qh: f64 = qrng.random_range(10.0..200.0);
            [qx, qy, qx + qw, qy + qh]
        })
        .collect();

    emit(&serde_json::json!({ "tool": "node_size_2d_meta", "n": N, "nq": NQ }));

    for &ns in &[4usize, 8, 16, 32, 64] {
        // build (AoS serial, lut+radix)
        let mut bbest = f64::INFINITY;
        for _ in 0..REPS_B {
            let t = Instant::now();
            let mut b = Index2DBuilder::new(N)
                .node_size(ns)
                .sort_key_strategy(SortKey2DStrategy::HilbertLut);
            for r in &boxes {
                b.add(Box2D::new(r[0], r[1], r[2], r[3]));
            }
            let idx = b.finish().unwrap();
            bbest = bbest.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(idx.num_items());
        }

        // SoA index for queries
        let mut sb = Index2DBuilder::new(N)
            .node_size(ns)
            .sort_key_strategy(SortKey2DStrategy::HilbertLut);
        for r in &boxes {
            sb.add(Box2D::new(r[0], r[1], r[2], r[3]));
        }
        let soa = sb.finish_simd().unwrap();
        let (mut buf, mut st) = (Vec::new(), Vec::new());

        let mut q2 = f64::INFINITY;
        for _ in 0..REPS_Q {
            let t = Instant::now();
            let mut tot = 0;
            for x in &queries {
                soa.search_simd(Box2D::new(x[0], x[1], x[2], x[3]), &mut buf, &mut st);
                tot += buf.len();
            }
            std::hint::black_box(tot);
            q2 = q2.min(t.elapsed().as_secs_f64() * 1e6);
        }

        let mut q8 = f64::INFINITY;
        for _ in 0..REPS_Q {
            let t = Instant::now();
            let mut tot = 0;
            for x in &queries {
                soa.search_avx512(Box2D::new(x[0], x[1], x[2], x[3]), &mut buf, &mut st);
                tot += buf.len();
            }
            std::hint::black_box(tot);
            q8 = q8.min(t.elapsed().as_secs_f64() * 1e6);
        }

        emit(&serde_json::json!({
            "tool": "node_size_2d",
            "node_size": ns,
            "build_serial_ms": bbest,
            "query_avx2_us": q2,
            "query_avx512_us": q8,
        }));
    }
}
