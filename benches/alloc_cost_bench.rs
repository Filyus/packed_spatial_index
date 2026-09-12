//! What one traversal-stack allocation per query costs. `search_into` and
//! `search_into_stack` run the identical traversal into a caller-owned result
//! buffer; the only difference is that `search_into` allocates its stack `Vec`
//! per call. Same binary, same data, so the gap is the allocation.

use criterion::{Criterion, criterion_group};
use packed_spatial_index::{Box2D, Index2D, Index2DBuilder, SearchWorkspace};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

const NODE_SIZE: usize = 16;
const N: usize = 100_000;

fn build(seed: u64) -> Index2D {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = Index2DBuilder::new(N).node_size(NODE_SIZE);
    for _ in 0..N {
        let cx: f64 = rng.random_range(0.0..10_000.0);
        let cy: f64 = rng.random_range(0.0..10_000.0);
        let w: f64 = rng.random_range(0.1..20.0);
        let h: f64 = rng.random_range(0.1..20.0);
        b.add(Box2D::new(cx, cy, cx + w, cy + h));
    }
    b.finish().unwrap()
}

fn queries(count: usize, seed: u64, span: f64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0 - span);
            let y: f64 = rng.random_range(0.0..10_000.0 - span);
            Box2D::new(x, y, x + span, y + span)
        })
        .collect()
}

fn bench(c: &mut Criterion, label: &str, span: f64) {
    let index = build(0xB0B);
    let qs = queries(1_000, 0xACE, span);

    let mut group = c.benchmark_group(format!("alloc/{label}"));

    group.bench_function("search_into_stack (no alloc)", |b| {
        let (mut out, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &qs {
                index.search_into_stack(*q, &mut out, &mut stack);
                total += out.len();
            }
            black_box(total)
        })
    });

    group.bench_function("search_into (stack alloc per query)", |b| {
        let mut out = Vec::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &qs {
                index.search_into(*q, &mut out);
                total += out.len();
            }
            black_box(total)
        })
    });

    group.bench_function("search_with (workspace)", |b| {
        let mut ws = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &qs {
                total += index.search_with(*q, &mut ws).len();
            }
            black_box(total)
        })
    });

    group.bench_function("count (stack alloc per query)", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &qs {
                total += index.count(*q);
            }
            black_box(total)
        })
    });

    group.bench_function("search (result Vec + stack alloc)", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &qs {
                total += index.search(*q).len();
            }
            black_box(total)
        })
    });

    group.finish();
}

fn benches(c: &mut Criterion) {
    bench(c, "wide", 200.0);
    bench(c, "narrow", 20.0);
    bench(c, "point", 2.0);
}

criterion_group! {
    name = alloc;
    config = pin::criterion();
    targets = benches
}
#[path = "support/pin.rs"]
mod pin;
fn main() {
    pin::pin_from_env();
    alloc();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
