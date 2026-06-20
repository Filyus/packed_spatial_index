//! f64 vs f32 storage A/B: range search with small and large query boxes, plus
//! a check that f32 rounded search does not miss exact f64 hits.
//!
//! Run:
//!   cargo bench --bench coord_precision --no-default-features --features f32-storage

use std::collections::HashSet;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use packed_spatial_index::{
    Box2D, Index2DBuilder, NeighborWorkspace, Point2D, SimdIndex2D, SimdIndex2DF32,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const NODE_SIZE: usize = 16;
const EXTENT: f64 = 1_000_000.0;
const MAX_SIZE: f64 = 50.0;
const QUERY_COUNT: usize = 1_000;
const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];
const KNN_QUERY_COUNT: usize = 200;
const KNN_LIMIT: usize = 8;
const KNN_SIZES: &[usize] = &[10_000, 100_000];

#[derive(Clone, Copy)]
struct QueryCase {
    name: &'static str,
    query_box_fraction: f64,
}

const QUERY_CASES: &[QueryCase] = &[
    QueryCase {
        name: "small_query_boxes",
        query_box_fraction: 0.001,
    },
    QueryCase {
        name: "large_query_boxes",
        query_box_fraction: 0.05,
    },
];

fn gen_boxes(n: usize, seed: u64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x = rng.random::<f64>() * EXTENT;
            let y = rng.random::<f64>() * EXTENT;
            let w = rng.random::<f64>() * MAX_SIZE;
            let h = rng.random::<f64>() * MAX_SIZE;
            Box2D::new(x, y, x + w, y + h)
        })
        .collect()
}

fn make_query_boxes(n: usize, seed: u64, size: f64) -> Vec<Box2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x = rng.random::<f64>() * EXTENT;
            let y = rng.random::<f64>() * EXTENT;
            Box2D::new(x, y, x + size, y + size)
        })
        .collect()
}

fn make_points(n: usize, seed: u64) -> Vec<Point2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| Point2D::new(rng.random::<f64>() * EXTENT, rng.random::<f64>() * EXTENT))
        .collect()
}

fn build_f64(boxes: &[Box2D]) -> SimdIndex2D {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(NODE_SIZE);
    for &it in boxes {
        b.add(it);
    }
    b.finish_simd().unwrap()
}

fn build_f32(boxes: &[Box2D]) -> SimdIndex2DF32 {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(NODE_SIZE);
    for &it in boxes {
        b.add(it);
    }
    b.finish_simd_f32().unwrap()
}

fn check_rounded_search(
    query_boxes: &[Box2D],
    exact_index: &SimdIndex2D,
    rounded_index: &SimdIndex2DF32,
) -> (bool, usize, usize, usize) {
    let mut no_missed_hits = true;
    let mut extra_hits = 0usize;
    let mut rounded_hit_count = 0usize;
    let mut exact_hit_count = 0usize;
    let mut rounded_hits = Vec::new();
    let mut exact_hits = Vec::new();
    for &query in query_boxes {
        exact_index.search_into(query, &mut exact_hits);
        rounded_index.search_into(query, &mut rounded_hits);
        rounded_hit_count += rounded_hits.len();
        exact_hit_count += exact_hits.len();
        let rounded_set: HashSet<usize> = rounded_hits.iter().copied().collect();
        if exact_hits.iter().any(|h| !rounded_set.contains(h)) {
            no_missed_hits = false;
        }
        extra_hits += rounded_hits.len() - exact_hits.len();
    }
    (
        no_missed_hits,
        extra_hits,
        rounded_hit_count,
        exact_hit_count,
    )
}

fn bench_search(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    eprintln!(
        "[cpu] avx512f = {}",
        std::is_x86_feature_detected!("avx512f")
    );

    for &n in SIZES {
        let boxes = gen_boxes(n, 0xC0FFEE ^ n as u64);
        let idx64 = build_f64(&boxes);
        let idx32 = build_f32(&boxes);

        for case in QUERY_CASES {
            let query_box_size = EXTENT * case.query_box_fraction;
            let queries = make_query_boxes(QUERY_COUNT, 0xBEEF ^ n as u64, query_box_size);

            let (no_missed_hits, extra_hits, rounded_hit_count, exact_hit_count) =
                check_rounded_search(&queries, &idx64, &idx32);
            eprintln!(
                "[check] items={n} {}: no_missed_hits={no_missed_hits} extra_hits={extra_hits} avg_rounded_hits={:.1} avg_exact_hits={:.1}",
                case.name,
                rounded_hit_count as f64 / queries.len() as f64,
                exact_hit_count as f64 / queries.len() as f64,
            );

            let mut group = c.benchmark_group(format!("range_{}", case.name));
            group.throughput(Throughput::Elements(QUERY_COUNT as u64));

            let mut buf = Vec::new();
            group.bench_with_input(BenchmarkId::new("f64_exact", n), &queries, |b, qs| {
                b.iter(|| {
                    for &q in qs {
                        idx64.search_into(q, &mut buf);
                        black_box(buf.len());
                    }
                })
            });
            group.bench_with_input(BenchmarkId::new("f32_rounded", n), &queries, |b, qs| {
                b.iter(|| {
                    for &q in qs {
                        idx32.search_into(q, &mut buf);
                        black_box(buf.len());
                    }
                })
            });
            group.bench_with_input(BenchmarkId::new("f32_exact", n), &queries, |b, qs| {
                b.iter(|| {
                    for &q in qs {
                        idx32.search_exact_into(q, |i| boxes[i], &mut buf);
                        black_box(buf.len());
                    }
                })
            });
            group.finish();
        }
    }
}

fn bench_neighbors(c: &mut Criterion) {
    for &n in KNN_SIZES {
        let boxes = gen_boxes(n, 0xD15EA5E ^ n as u64);
        let idx64 = build_f64(&boxes);
        let idx32 = build_f32(&boxes);
        let points = make_points(KNN_QUERY_COUNT, 0xFACE ^ n as u64);

        let mut group = c.benchmark_group("nearest_neighbors_top_8");
        group.throughput(Throughput::Elements(KNN_QUERY_COUNT as u64));

        let mut workspace = NeighborWorkspace::new();
        group.bench_with_input(BenchmarkId::new("f64_exact", n), &points, |b, ps| {
            b.iter(|| {
                for &p in ps {
                    let hits = idx64.neighbors_with(p, KNN_LIMIT, f64::INFINITY, &mut workspace);
                    black_box(hits.len());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("f32_rounded", n), &points, |b, ps| {
            b.iter(|| {
                for &p in ps {
                    let hits = idx32.neighbors_with(p, KNN_LIMIT, f64::INFINITY, &mut workspace);
                    black_box(hits.len());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("f32_exact", n), &points, |b, ps| {
            b.iter(|| {
                for &p in ps {
                    let hits = idx32.neighbors_exact_with(
                        p,
                        KNN_LIMIT,
                        f64::INFINITY,
                        |i| boxes[i],
                        &mut workspace,
                    );
                    black_box(hits.len());
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_search, bench_neighbors);
#[path = "support/pin.rs"]
mod pin;

fn main() {
    pin::pin_from_env();
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
