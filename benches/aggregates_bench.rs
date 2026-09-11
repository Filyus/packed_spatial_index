//! Node aggregates: the exact `aggregate` fold against the search-and-fold
//! workaround it replaces. Both walk the same hits; the fold wins when the
//! number of hits is large relative to the nodes the window cuts, which is
//! what the case spread shows.
//!
//! Run:
//!   cargo bench --bench aggregates_bench

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use packed_spatial_index::{Box2D, Index2DBuilder};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;
const COUNT: usize = 1_000_000;
const EXTENT: f64 = 1_000_000.0;
const MAX_SIZE: f64 = 50.0;

fn build() -> packed_spatial_index::Index2D {
    let mut rng = StdRng::seed_from_u64(1);
    let mut builder = Index2DBuilder::new(COUNT).node_size(NODE_SIZE);
    for i in 0..COUNT {
        let x: f64 = rng.random_range(0.0..EXTENT);
        let y: f64 = rng.random_range(0.0..EXTENT);
        let w: f64 = rng.random_range(0.0..MAX_SIZE);
        let h: f64 = rng.random_range(0.0..MAX_SIZE);
        builder.add(Box2D::new(x, y, x + w, y + h));
        let _ = i;
    }
    let scalars: Vec<f64> = (0..COUNT).map(|i| (i % 101) as f64).collect();
    builder.aggregate_scalar(&scalars).finish().unwrap()
}

fn bench_aggregate(c: &mut Criterion) {
    let index = build();

    let mut group = c.benchmark_group("aggregate_vs_search_fold");
    for &side in &[1_000.0f64, 10_000.0, 100_000.0] {
        let mut rng = StdRng::seed_from_u64(2);
        let windows: Vec<Box2D> = (0..10)
            .map(|_| {
                let x: f64 = rng.random_range(0.0..EXTENT - side);
                let y: f64 = rng.random_range(0.0..EXTENT - side);
                Box2D::new(x, y, x + side, y + side)
            })
            .collect();
        let hits: usize = windows.iter().map(|&w| index.count(w)).sum::<usize>() / windows.len();

        group.bench_with_input(
            BenchmarkId::new("aggregate", format!("side_{side}_hits~{hits}")),
            &windows,
            |b, windows| {
                b.iter(|| {
                    for &w in windows {
                        black_box(index.aggregate(black_box(w)));
                    }
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("search_fold", format!("side_{side}_hits~{hits}")),
            &windows,
            |b, windows| {
                b.iter(|| {
                    for &w in windows {
                        let mut sum = 0.0f64;
                        for id in index.search(black_box(w)) {
                            sum += (id % 101) as f64;
                        }
                        black_box(sum);
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_aggregate);
criterion_main!(benches);
