//! Compare sort-key strategies for the packed Hilbert R-tree.
//!
//! Shows why the space-filling curve matters: measures build time,
//! query time, and average intersection checks per query (a locality metric).
//! Run: `cargo run --release --manifest-path benches/tools/Cargo.toml --bin sortkey_quality_2d`

use std::time::{Duration, Instant};

use packed_spatial_index::benchmark_support::SortKey2DStrategy;
use packed_spatial_index::{Box2D, Index2DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 100_000;
const NODE_SIZE: usize = 16;
const QUERIES: usize = 2_000;
const BUILD_REPEATS: usize = 5;
const REPEATS: usize = 50; // repetitions for stable query timing

fn main() {
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
    let queries: Vec<[f64; 4]> = (0..QUERIES)
        .map(|_| {
            let qx: f64 = qrng.random_range(0.0..10_000.0);
            let qy: f64 = qrng.random_range(0.0..10_000.0);
            let qw: f64 = qrng.random_range(10.0..200.0);
            let qh: f64 = qrng.random_range(10.0..200.0);
            [qx, qy, qx + qw, qy + qh]
        })
        .collect();

    let keys = [
        ("Hilbert (magic_bits)", SortKey2DStrategy::HilbertMagicBits),
        ("Hilbert (lut)", SortKey2DStrategy::HilbertLut),
        ("Hilbert (loop)", SortKey2DStrategy::HilbertLoopRotation),
        ("Morton (Z-order)", SortKey2DStrategy::Morton),
    ];

    println!(
        "N={N}, node_size={NODE_SIZE}, queries={QUERIES} (build best of {BUILD_REPEATS}, query x{REPEATS})\n"
    );
    println!(
        "{:<22} | {:>10} | {:>12} | {:>16} | {:>14}",
        "Sort key", "build", "query (all)", "checks/query", "results/query"
    );
    println!("{}", "-".repeat(86));

    let mut baseline_visited = 0f64;
    for (i, (name, key)) in keys.iter().enumerate() {
        // build
        let mut build_t = Duration::MAX;
        let mut best_index = None;
        for _ in 0..BUILD_REPEATS {
            let t0 = Instant::now();
            let mut b = Index2DBuilder::new(N)
                .node_size(NODE_SIZE)
                .sort_key_strategy(*key);
            for r in &boxes {
                b.add(Box2D::new(r[0], r[1], r[2], r[3]));
            }
            let built_index = b.finish().unwrap();
            let build_time = t0.elapsed();
            if build_time < build_t {
                build_t = build_time;
                best_index = Some(built_index);
            }
        }
        let index = best_index.unwrap();

        // quality: average check and result counts
        let mut total_visited = 0usize;
        let mut total_results = 0usize;
        for q in &queries {
            let (res, vis) = index.search_visited(Box2D::new(q[0], q[1], q[2], q[3]));
            total_results += res;
            total_visited += vis;
        }
        let avg_visited = total_visited as f64 / QUERIES as f64;
        let avg_results = total_results as f64 / QUERIES as f64;
        if i == 0 {
            baseline_visited = avg_visited;
        }

        // query timing
        let t1 = Instant::now();
        let mut sink = 0usize;
        let mut buf = Vec::new();
        for _ in 0..REPEATS {
            for q in &queries {
                index.search_into(Box2D::new(q[0], q[1], q[2], q[3]), &mut buf);
                sink += buf.len();
            }
        }
        let query_t = t1.elapsed();
        std::hint::black_box(sink);

        let factor = avg_visited / baseline_visited;
        println!(
            "{:<22} | {:>8.2?} | {:>12.2?} | {:>10.0} (x{:.1}) | {:>14.1}",
            name, build_t, query_t, avg_visited, factor, avg_results
        );
    }

    println!(
        "\nNote: the number of results is the same for all keys (correctness does not depend\n\
         on order). Only the amount of work differs: worse key locality means more\n\
         intersection checks and slower queries."
    );
}
