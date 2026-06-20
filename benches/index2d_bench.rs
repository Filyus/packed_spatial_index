//! Spatial-index benchmark in **two modes**: single-threaded and parallel.
//!
//!  * Single-threaded: `Index2D` (serial) against the `static_aabb2d_index` baseline.
//!  * Parallel build: thresholded auto mode versus forced rayon. The baseline is single-threaded
//!    (it has no parallel build), so parallel numbers are the implementation ceiling,
//!    not a one-to-one algorithm comparison.
//!  * For queries, the query batch itself is parallelized (read-only), so the comparison is symmetric:
//!    both the baseline crate and `Index2D` benefit.

use std::{hint::black_box, ops::ControlFlow};

use criterion::{BenchmarkId, Criterion, criterion_group};
use packed_spatial_index::benchmark_support::SortKey2DStrategy;
use packed_spatial_index::{Box2D, Index2D, Index2DBuilder};
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
    let mut b = Index2DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey2DStrategy::HilbertLut);
    b = match mode {
        BuildMode::Serial => b.parallel(false),
        BuildMode::ParallelAuto => b.parallel(true),
        BuildMode::ParallelForced => b.parallel(true).parallel_min_items(0),
    };
    for r in boxes {
        b.add(Box2D::new(r[0], r[1], r[2], r[3]));
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
    make_queries_with_size(n, seed, 10.0..200.0)
}

fn make_queries_with_size(n: usize, seed: u64, size_range: std::ops::Range<f64>) -> Vec<[f64; 4]> {
    make_queries_with_ranges(n, seed, size_range.clone(), size_range)
}

fn make_queries_with_ranges(
    n: usize,
    seed: u64,
    width_range: std::ops::Range<f64>,
    height_range: std::ops::Range<f64>,
) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let qx: f64 = rng.random_range(0.0..10_000.0);
            let qy: f64 = rng.random_range(0.0..10_000.0);
            let qw: f64 = rng.random_range(width_range.clone());
            let qh: f64 = rng.random_range(height_range.clone());
            [qx, qy, qx + qw, qy + qh]
        })
        .collect()
}

fn to_bounds(q: &[f64; 4]) -> Box2D {
    Box2D::new(q[0], q[1], q[2], q[3])
}

fn bench_query(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xB0B);

    let mut rb = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(n, NODE_SIZE);
    let mut mb = Index2DBuilder::new(n)
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey2DStrategy::HilbertLut);
    let mut sb = Index2DBuilder::new(n)
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey2DStrategy::HilbertLut);
    for r in &boxes {
        rb.add(r[0], r[1], r[2], r[3]);
        mb.add(Box2D::new(r[0], r[1], r[2], r[3]));
        sb.add(Box2D::new(r[0], r[1], r[2], r[3]));
    }
    let reference: StaticAABB2DIndex<f64> = rb.build().unwrap();
    let packed: Index2D = mb.finish().unwrap();
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
                packed.search_into_stack(to_bounds(q), &mut buf, &mut stack);
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
                packed.search_into_stack_prefetch(to_bounds(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_any_serial", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += usize::from(packed.any(to_bounds(q)));
            }
            black_box(total)
        })
    });
    group.bench_function("simd_simd_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                simd.search_simd(to_bounds(q), &mut buf, &mut stack);
                total += buf.len();
            }
            black_box(total)
        })
    });
    group.bench_function("simd_any_serial", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += usize::from(simd.any(to_bounds(q)));
            }
            black_box(total)
        })
    });
    group.bench_function("simd_any_wide4_serial", |b| {
        let mut stack = Vec::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += usize::from(
                    simd.visit_simd(to_bounds(q), &mut stack, |_| ControlFlow::Break(()))
                        .is_break(),
                );
            }
            black_box(total)
        })
    });
    group.bench_function("simd_any_wide4_alloc_serial", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                let mut stack = Vec::with_capacity(NODE_SIZE);
                total += usize::from(
                    simd.visit_simd(to_bounds(q), &mut stack, |_| ControlFlow::Break(()))
                        .is_break(),
                );
            }
            black_box(total)
        })
    });
    group.bench_function("simd_any_avx512_reused_serial", |b| {
        let mut stack = Vec::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += usize::from(
                    simd.visit_avx512(to_bounds(q), &mut stack, |_| ControlFlow::Break(()))
                        .is_break(),
                );
            }
            black_box(total)
        })
    });
    group.bench_function("simd_simd_prefetch_serial", |b| {
        let (mut buf, mut stack) = (Vec::new(), Vec::new());
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                simd.search_simd_prefetch(to_bounds(q), &mut buf, &mut stack);
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
                simd.search_avx512(to_bounds(q), &mut buf, &mut stack);
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
                        packed.search_into_stack(to_bounds(q), buf, stack);
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
                        packed.search_into_stack_prefetch(to_bounds(q), buf, stack);
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
                .map(|q| usize::from(packed.any(to_bounds(q))))
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
                        simd.search_simd(to_bounds(q), buf, stack);
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
                .map(|q| usize::from(simd.any(to_bounds(q))))
                .sum();
            black_box(total)
        })
    });
    group.bench_function("simd_any_wide4_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(Vec::new, |stack, q| {
                    usize::from(
                        simd.visit_simd(to_bounds(q), stack, |_| ControlFlow::Break(()))
                            .is_break(),
                    )
                })
                .sum();
            black_box(total)
        })
    });
    group.bench_function("simd_any_avx512_reused_parallel", |b| {
        b.iter(|| {
            let total: usize = queries
                .par_iter()
                .map_init(Vec::new, |stack, q| {
                    usize::from(
                        simd.visit_avx512(to_bounds(q), stack, |_| ControlFlow::Break(()))
                            .is_break(),
                    )
                })
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
                        simd.search_simd_prefetch(to_bounds(q), buf, stack);
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
                        simd.search_avx512(to_bounds(q), buf, stack);
                        buf.len()
                    },
                )
                .sum();
            black_box(total)
        })
    });

    group.finish();
}

fn bench_query_windows(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xB0B);

    let mut mb = Index2DBuilder::new(n)
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey2DStrategy::HilbertLut);
    for r in &boxes {
        mb.add(Box2D::new(r[0], r[1], r[2], r[3]));
    }
    let packed: Index2D = mb.finish().unwrap();

    let mut sb = Index2DBuilder::new(n)
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey2DStrategy::HilbertLut);
    for r in &boxes {
        sb.add(Box2D::new(r[0], r[1], r[2], r[3]));
    }
    let simd = sb.finish_simd().unwrap();

    let small_queries = make_queries_with_size(1_000, 0x51A11, 10.0..200.0);
    let large_queries = make_queries_with_size(1_000, 0x1A96E, 2_000.0..5_000.0);
    let sliver_queries = make_queries_with_ranges(1_000, 0x5111E, 3_000.0..7_000.0, 10.0..40.0);
    let extent = packed.extent().unwrap();
    let full_extent_queries = vec![[extent.min_x, extent.min_y, extent.max_x, extent.max_y]; 1_000];

    let mut group = c.benchmark_group("query_windows");
    for (name, queries) in [
        ("small", &small_queries),
        ("large", &large_queries),
        ("wide_sliver", &sliver_queries),
        ("full_extent", &full_extent_queries),
    ] {
        group.bench_function(format!("index_serial_{name}"), |b| {
            let (mut buf, mut stack) = (Vec::new(), Vec::new());
            b.iter(|| {
                let mut total = 0usize;
                for q in queries {
                    packed.search_into_stack(to_bounds(q), &mut buf, &mut stack);
                    total += buf.len();
                }
                black_box(total)
            })
        });
        group.bench_function(format!("index_prefetch_serial_{name}"), |b| {
            let (mut buf, mut stack) = (Vec::new(), Vec::new());
            b.iter(|| {
                let mut total = 0usize;
                for q in queries {
                    packed.search_into_stack_prefetch(to_bounds(q), &mut buf, &mut stack);
                    total += buf.len();
                }
                black_box(total)
            })
        });
        group.bench_function(format!("simd_avx512_serial_{name}"), |b| {
            let (mut buf, mut stack) = (Vec::new(), Vec::new());
            b.iter(|| {
                let mut total = 0usize;
                for q in queries {
                    simd.search_avx512(to_bounds(q), &mut buf, &mut stack);
                    total += buf.len();
                }
                black_box(total)
            })
        });
        group.bench_function(format!("simd_simd_serial_{name}"), |b| {
            let (mut buf, mut stack) = (Vec::new(), Vec::new());
            b.iter(|| {
                let mut total = 0usize;
                for q in queries {
                    simd.search_simd(to_bounds(q), &mut buf, &mut stack);
                    total += buf.len();
                }
                black_box(total)
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = pin::criterion();
    targets = bench_build, bench_query, bench_query_windows
}
#[path = "support/pin.rs"]
mod pin;

fn main() {
    pin::pin_from_env();
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
