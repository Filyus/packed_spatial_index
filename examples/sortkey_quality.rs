//! Compare sort-key strategies for the packed Hilbert R-tree.
//!
//! Shows why the space-filling curve matters: measures build time,
//! query time, and average intersection checks per query (a locality metric).
//! Run: `cargo run --release --example sortkey_quality`

use std::time::Instant;

use packed_spatial_index::experimental::ExperimentalSortKey;
use packed_spatial_index::{IndexBuilder, Rect};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 100_000;
const NODE_SIZE: usize = 16;
const QUERIES: usize = 2_000;
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
        (
            "Hilbert (magic_bits)",
            ExperimentalSortKey::HilbertMagicBits,
        ),
        ("Morton (Z-order)", ExperimentalSortKey::Morton),
    ];

    println!("N={N}, node_size={NODE_SIZE}, queries={QUERIES} (x{REPEATS} repetitions)\n");
    println!(
        "{:<22} | {:>10} | {:>12} | {:>16} | {:>14}",
        "Sort key", "build", "query (all)", "checks/query", "results/query"
    );
    println!("{}", "-".repeat(86));

    let mut baseline_visited = 0f64;
    for (i, (name, key)) in keys.iter().enumerate() {
        // build
        let t0 = Instant::now();
        let mut b = IndexBuilder::new(N)
            .node_size(NODE_SIZE)
            .experimental_sort_key(*key);
        for r in &boxes {
            b.add_bounds(r[0], r[1], r[2], r[3]);
        }
        let index = b.finish().unwrap();
        let build_t = t0.elapsed();

        // quality: average check and result counts
        let mut total_visited = 0usize;
        let mut total_results = 0usize;
        for q in &queries {
            let (res, vis) = index.search_visited(Rect::new(q[0], q[1], q[2], q[3]));
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
                index.search_into(Rect::new(q[0], q[1], q[2], q[3]), &mut buf);
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
