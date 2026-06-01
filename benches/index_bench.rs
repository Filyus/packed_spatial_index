//! Spatial-index benchmark in **two modes**: single-threaded and parallel.
//!
//!  * Single-threaded: `Index` (serial) against the `static_aabb2d_index` baseline.
//!  * Parallel build: thresholded auto mode versus forced rayon. The baseline is single-threaded
//!    (it has no parallel build), so parallel numbers are the implementation ceiling,
//!    not a one-to-one algorithm comparison.
//!  * For queries, the query batch itself is parallelized (read-only), so the comparison is symmetric:
//!    both the baseline crate and `Index` benefit.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use packed_spatial_index::experimental::ExperimentalSortKey;
use packed_spatial_index::{Index, IndexBuilder, Rect};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use static_aabb2d_index::{StaticAABB2DIndex, StaticAABB2DIndexBuilder};

const NODE_SIZE: usize = 16;

#[derive(Clone, Copy)]
enum BuildMode {
    Serial,
    ParallelAuto,
    ParallelForced,
}

fn gen_boxes(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
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

fn build_reference(boxes: &[[f64; 4]]) {
    let mut b = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(boxes.len(), NODE_SIZE);
    for r in boxes {
        b.add(r[0], r[1], r[2], r[3]);
    }
    black_box(b.build().unwrap());
}

fn build_mine(boxes: &[[f64; 4]], mode: BuildMode) {
    let mut b = IndexBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    b = match mode {
        BuildMode::Serial => b.parallel(false),
        BuildMode::ParallelAuto => b.parallel(true),
        BuildMode::ParallelForced => b.parallel(true).parallel_min_items(0),
    };
    for r in boxes {
        b.add(Rect::new(r[0], r[1], r[2], r[3]));
    }
    black_box(b.finish().unwrap());
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    for &n in &[NODE_SIZE, 1_000, 100_000, 1_000_000] {
        let boxes = gen_boxes(n, 0xB0B);
        // single-threaded mode
        group.bench_with_input(BenchmarkId::new("crate", n), &boxes, |b, boxes| {
            b.iter(|| build_reference(boxes))
        });
        group.bench_with_input(BenchmarkId::new("index_serial", n), &boxes, |b, boxes| {
            b.iter(|| build_mine(boxes, BuildMode::Serial))
        });
        // auto: parallel(true) enables rayon only above the threshold
        group.bench_with_input(
            BenchmarkId::new("index_parallel_auto", n),
            &boxes,
            |b, boxes| b.iter(|| build_mine(boxes, BuildMode::ParallelAuto)),
        );
        // forced: previous behavior, useful for measuring rayon overhead
        group.bench_with_input(
            BenchmarkId::new("index_parallel_forced", n),
            &boxes,
            |b, boxes| b.iter(|| build_mine(boxes, BuildMode::ParallelForced)),
        );
    }
    group.finish();
}

fn make_queries(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let qx: f64 = rng.random_range(0.0..10_000.0);
            let qy: f64 = rng.random_range(0.0..10_000.0);
            let qw: f64 = rng.random_range(10.0..200.0);
            let qh: f64 = rng.random_range(10.0..200.0);
            [qx, qy, qx + qw, qy + qh]
        })
        .collect()
}

fn to_rect(q: &[f64; 4]) -> Rect {
    Rect::new(q[0], q[1], q[2], q[3])
}

fn bench_query(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xB0B);

    let mut rb = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, NODE_SIZE);
    let mut mb = IndexBuilder::new(n)
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    let mut sb = IndexBuilder::new(n)
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey::HilbertLut);
    for r in &boxes {
        rb.add(r[0], r[1], r[2], r[3]);
        mb.add(Rect::new(r[0], r[1], r[2], r[3]));
        sb.add(Rect::new(r[0], r[1], r[2], r[3]));
    }
    let reference: StaticAABB2DIndex<f64> = rb.build().unwrap();
    let packed: Index = mb.finish().unwrap();
    let simd = sb.finish_simd().unwrap();
    let queries = make_queries(1_000, 0xACE);

    let mut group = c.benchmark_group("query");

    // --- single-threaded mode ---
    group.bench_function("crate_serial", |b| {
        let mut stack = Vec::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += reference
                    .query_with_stack(q[0], q[1], q[2], q[3], &mut stack)
                    .len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                packed.search_into_stack(to_rect(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_prefetch_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                packed.search_into_stack_prefetch(to_rect(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_any_serial", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += usize::from(packed.any(to_rect(q)));
            }
            black_box(total)
        })
    });
    group.bench_function("simd_simd_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                simd.search_simd(to_rect(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });
    group.bench_function("simd_any_serial", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += usize::from(simd.any(to_rect(q)));
            }
            black_box(total)
        })
    });
    group.bench_function("simd_simd_prefetch_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                simd.search_simd_prefetch(to_rect(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });
    group.bench_function("simd_avx512_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                simd.search_avx512(to_rect(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });

    // --- parallel mode: the query batch is spread across threads (read-only) ---
    group.bench_function("crate_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(Vec::new, |stack, q| {
                    reference
                        .query_with_stack(q[0], q[1], q[2], q[3], stack)
                        .len()
                })
                .sum();
            black_box(total)
        })
    });
    group.bench_function("index_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(buf, stack), q| {
                        packed.search_into_stack(to_rect(q), buf, stack);
                        buf.len()
                    },
                )
                .sum();
            black_box(total)
        })
    });
    group.bench_function("index_prefetch_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(buf, stack), q| {
                        packed.search_into_stack_prefetch(to_rect(q), buf, stack);
                        buf.len()
                    },
                )
                .sum();
            black_box(total)
        })
    });
    group.bench_function("index_any_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map(|q| usize::from(packed.any(to_rect(q))))
                .sum();
            black_box(total)
        })
    });
    group.bench_function("simd_simd_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(buf, stack), q| {
                        simd.search_simd(to_rect(q), buf, stack);
                        buf.len()
                    },
                )
                .sum();
            black_box(total)
        })
    });
    group.bench_function("simd_any_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map(|q| usize::from(simd.any(to_rect(q))))
                .sum();
            black_box(total)
        })
    });
    group.bench_function("simd_simd_prefetch_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(buf, stack), q| {
                        simd.search_simd_prefetch(to_rect(q), buf, stack);
                        buf.len()
                    },
                )
                .sum();
            black_box(total)
        })
    });
    group.bench_function("simd_avx512_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(
                    || (Vec::new(), Vec::new()),
                    |(buf, stack), q| {
                        simd.search_avx512(to_rect(q), buf, stack);
                        buf.len()
                    },
                )
                .sum();
            black_box(total)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_build, bench_query);
criterion_main!(benches);
