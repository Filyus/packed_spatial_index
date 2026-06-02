//! Persistence and nearest-neighbor benchmarks.
//!
//! Run:
//!   cargo bench --bench persistence_knn2d_bench --no-default-features --features simd

use std::ops::ControlFlow;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use packed_spatial_index::experimental::ExperimentalSortKey2D;
use packed_spatial_index::{
    Bounds2D, Index2D, Index2DBuilder, Index2DView, NeighborWorkspace, Point2D, SearchWorkspace,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use static_aabb2d_index::{Control, StaticAABB2DIndex, StaticAABB2DIndexBuilder};

const NODE_SIZE: usize = 16;
const QUERY_COUNT: usize = 1_000;
const LOADED_KNN_LIMIT: usize = 10;
const LOADED_KNN_MAX_DISTANCE: f64 = f64::INFINITY;

#[derive(Clone, Copy)]
struct KnnCase {
    name: &'static str,
    limit: usize,
    max_distance: f64,
}

const KNN_CASES: &[KnnCase] = &[
    KnnCase {
        name: "top_1",
        limit: 1,
        max_distance: f64::INFINITY,
    },
    KnnCase {
        name: "top_10",
        limit: 10,
        max_distance: f64::INFINITY,
    },
    KnnCase {
        name: "top_100",
        limit: 100,
        max_distance: f64::INFINITY,
    },
    KnnCase {
        name: "top_10_radius_100",
        limit: 10,
        max_distance: 100.0,
    },
];

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

fn make_points(n: usize, seed: u64) -> Vec<Point2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Point2D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
            )
        })
        .collect()
}

fn to_bounds(q: &[f64; 4]) -> Bounds2D {
    Bounds2D::new(q[0], q[1], q[2], q[3])
}

fn build_index(boxes: &[[f64; 4]]) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey2D::HilbertLut);
    for b in boxes {
        builder.add(Bounds2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish().unwrap()
}

fn build_simd_index(boxes: &[[f64; 4]]) -> packed_spatial_index::SimdIndex2D {
    let mut builder = Index2DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .experimental_sort_key(ExperimentalSortKey2D::HilbertLut);
    for b in boxes {
        builder.add(Bounds2D::new(b[0], b[1], b[2], b[3]));
    }
    builder.finish_simd().unwrap()
}

fn build_reference(boxes: &[[f64; 4]]) -> StaticAABB2DIndex<f64> {
    let mut builder = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(boxes.len(), NODE_SIZE);
    for b in boxes {
        builder.add(b[0], b[1], b[2], b[3]);
    }
    builder.build().unwrap()
}

fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");

    for &n in &[1_000usize, 100_000, 1_000_000] {
        let boxes = gen_boxes(n, 0xB0B);
        let index = build_index(&boxes);
        let bytes = index.to_bytes();

        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("to_bytes", n), &index, |b, index| {
            b.iter(|| black_box(index.to_bytes()))
        });
        group.bench_with_input(
            BenchmarkId::new("from_bytes_owned", n),
            &bytes,
            |b, bytes| b.iter(|| black_box(Index2D::from_bytes(bytes).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("from_bytes_view", n),
            &bytes,
            |b, bytes| b.iter(|| black_box(Index2DView::from_bytes(bytes).unwrap())),
        );
    }

    group.finish();
}

fn bench_loaded_query(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xB0B);
    let index = build_index(&boxes);
    let bytes = index.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    let queries = make_queries(QUERY_COUNT, 0xACE);
    let points = make_points(QUERY_COUNT, 0xD15C);

    let mut group = c.benchmark_group("loaded_index");
    group.bench_function("index_search", |b| {
        let mut workspace = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += index.search_with(to_bounds(q), &mut workspace).len();
            }
            black_box(total)
        })
    });
    group.bench_function("view_search", |b| {
        let mut workspace = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for q in &queries {
                total += view.search_with(to_bounds(q), &mut workspace).len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_neighbors", |b| {
        let mut workspace = NeighborWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &p in &points {
                total += index
                    .neighbors_with(p, LOADED_KNN_LIMIT, LOADED_KNN_MAX_DISTANCE, &mut workspace)
                    .len();
            }
            black_box(total)
        })
    });
    group.bench_function("view_neighbors", |b| {
        let mut workspace = NeighborWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &p in &points {
                total += view
                    .neighbors_with(p, LOADED_KNN_LIMIT, LOADED_KNN_MAX_DISTANCE, &mut workspace)
                    .len();
            }
            black_box(total)
        })
    });
    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0xB0B);
    let index = build_index(&boxes);
    let simd = build_simd_index(&boxes);
    let reference = build_reference(&boxes);
    let bytes = index.to_bytes();
    let view = Index2DView::from_bytes(&bytes).unwrap();
    let points = make_points(QUERY_COUNT, 0xD15C);

    let mut group = c.benchmark_group("knn");
    for case in KNN_CASES {
        group.bench_with_input(
            BenchmarkId::new("crate_visit_neighbors", case.name),
            case,
            |b, case| {
                let max_dist_sq = case.max_distance * case.max_distance;
                b.iter(|| {
                    let mut total = 0usize;
                    let mut results = Vec::with_capacity(case.limit);
                    for &p in &points {
                        results.clear();
                        let _ = reference.visit_neighbors(p.x, p.y, &mut |idx, dist| {
                            if dist > max_dist_sq {
                                return Control::Break(());
                            }
                            results.push(idx);
                            if results.len() == case.limit {
                                Control::Break(())
                            } else {
                                Control::Continue
                            }
                        });
                        total += results.len();
                    }
                    black_box(total)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("index_neighbors_with", case.name),
            case,
            |b, case| {
                let mut workspace = NeighborWorkspace::new();
                b.iter(|| {
                    let mut total = 0usize;
                    for &p in &points {
                        total += index
                            .neighbors_with(p, case.limit, case.max_distance, &mut workspace)
                            .len();
                    }
                    black_box(total)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("view_neighbors_with", case.name),
            case,
            |b, case| {
                let mut workspace = NeighborWorkspace::new();
                b.iter(|| {
                    let mut total = 0usize;
                    for &p in &points {
                        total += view
                            .neighbors_with(p, case.limit, case.max_distance, &mut workspace)
                            .len();
                    }
                    black_box(total)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_neighbors_with", case.name),
            case,
            |b, case| {
                let mut workspace = NeighborWorkspace::new();
                b.iter(|| {
                    let mut total = 0usize;
                    for &p in &points {
                        total += simd
                            .neighbors_with(p, case.limit, case.max_distance, &mut workspace)
                            .len();
                    }
                    black_box(total)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("index_visit_neighbors", case.name),
            case,
            |b, case| {
                b.iter(|| {
                    let mut total = 0usize;
                    for &p in &points {
                        let mut count = 0usize;
                        let _: ControlFlow<()> =
                            index.visit_neighbors(p, case.max_distance, |_idx, _dist| {
                                count += 1;
                                if count == case.limit {
                                    ControlFlow::Break(())
                                } else {
                                    ControlFlow::Continue(())
                                }
                            });
                        total += count;
                    }
                    black_box(total)
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_persistence, bench_loaded_query, bench_knn);
criterion_main!(benches);
