//! 3D persistence and nearest-neighbor benchmarks.
//!
//! Run:
//!   cargo bench --bench persistence_knn3d_bench --no-default-features --features simd

use std::hint::black_box;
use std::ops::ControlFlow;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use packed_spatial_index::benchmark_support::SortKey3DStrategy;
use packed_spatial_index::{
    Box3D, Index3D, Index3DBuilder, Index3DView, NeighborWorkspace, Point3D, SearchWorkspace,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

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

fn gen_boxes(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let dx: f64 = rng.random_range(0.1..20.0);
            let dy: f64 = rng.random_range(0.1..20.0);
            let dz: f64 = rng.random_range(0.1..20.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn make_queries(n: usize, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..10_000.0);
            let y: f64 = rng.random_range(0.0..10_000.0);
            let z: f64 = rng.random_range(0.0..10_000.0);
            let w: f64 = rng.random_range(10.0..200.0);
            Box3D::new(x, y, z, x + w, y + w, z + w)
        })
        .collect()
}

fn make_points(n: usize, seed: u64) -> Vec<Point3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
            )
        })
        .collect()
}

fn build_index(boxes: &[Box3D]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey3DStrategy::Hilbert);
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

fn build_simd_index(boxes: &[Box3D]) -> packed_spatial_index::SimdIndex3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(NODE_SIZE)
        .sort_key_strategy(SortKey3DStrategy::Hilbert);
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish_simd().unwrap()
}

fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence3d");

    for &n in &[1_000usize, 100_000, 1_000_000] {
        let boxes = gen_boxes(n, 0x3D0B);
        let index = build_index(&boxes);
        let simd = build_simd_index(&boxes);
        let bytes = index.to_bytes();

        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("to_bytes", n), &index, |b, index| {
            b.iter(|| black_box(index.to_bytes()))
        });
        group.bench_with_input(BenchmarkId::new("to_bytes_into", n), &index, |b, index| {
            let mut out = Vec::with_capacity(bytes.len());
            b.iter(|| {
                index.to_bytes_into(&mut out);
                black_box(out.len())
            })
        });
        group.bench_with_input(
            BenchmarkId::new("from_bytes_owned", n),
            &bytes,
            |b, bytes| b.iter(|| black_box(Index3D::from_bytes(bytes).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("from_bytes_view", n),
            &bytes,
            |b, bytes| b.iter(|| black_box(Index3DView::from_bytes(bytes).unwrap())),
        );
        group.bench_with_input(BenchmarkId::new("simd_to_bytes", n), &simd, |b, simd| {
            b.iter(|| black_box(simd.to_bytes()))
        });
        group.bench_with_input(
            BenchmarkId::new("simd_to_bytes_into", n),
            &simd,
            |b, simd| {
                let mut out = Vec::with_capacity(bytes.len());
                b.iter(|| {
                    simd.to_bytes_into(&mut out);
                    black_box(out.len())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("simd_from_bytes_owned", n),
            &bytes,
            |b, bytes| {
                b.iter(|| black_box(packed_spatial_index::SimdIndex3D::from_bytes(bytes).unwrap()))
            },
        );
    }

    group.finish();
}

fn bench_loaded_query(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0x3D0B);
    let index = build_index(&boxes);
    let bytes = index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let queries = make_queries(QUERY_COUNT, 0x3ACE);
    let points = make_points(QUERY_COUNT, 0x3D15);

    let mut group = c.benchmark_group("loaded_index3d");
    group.bench_function("index_search", |b| {
        let mut workspace = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &query in &queries {
                total += index.search_with(query, &mut workspace).len();
            }
            black_box(total)
        })
    });
    group.bench_function("view_search", |b| {
        let mut workspace = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &query in &queries {
                total += view.search_with(query, &mut workspace).len();
            }
            black_box(total)
        })
    });
    group.bench_function("index_neighbors", |b| {
        let mut workspace = NeighborWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &point in &points {
                total += index
                    .neighbors_with(
                        point,
                        LOADED_KNN_LIMIT,
                        LOADED_KNN_MAX_DISTANCE,
                        &mut workspace,
                    )
                    .len();
            }
            black_box(total)
        })
    });
    group.bench_function("view_neighbors", |b| {
        let mut workspace = NeighborWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &point in &points {
                total += view
                    .neighbors_with(
                        point,
                        LOADED_KNN_LIMIT,
                        LOADED_KNN_MAX_DISTANCE,
                        &mut workspace,
                    )
                    .len();
            }
            black_box(total)
        })
    });
    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(n, 0x3D0B);
    let index = build_index(&boxes);
    let simd = build_simd_index(&boxes);
    let bytes = index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let points = make_points(QUERY_COUNT, 0x3D15);

    let mut group = c.benchmark_group("knn3d");
    for case in KNN_CASES {
        group.bench_with_input(
            BenchmarkId::new("index_neighbors_with", case.name),
            case,
            |b, case| {
                let mut workspace = NeighborWorkspace::new();
                b.iter(|| {
                    let mut total = 0usize;
                    for &point in &points {
                        total += index
                            .neighbors_with(point, case.limit, case.max_distance, &mut workspace)
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
                    for &point in &points {
                        total += view
                            .neighbors_with(point, case.limit, case.max_distance, &mut workspace)
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
                    for &point in &points {
                        total += simd
                            .neighbors_with(point, case.limit, case.max_distance, &mut workspace)
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
                    for &point in &points {
                        let mut count = 0usize;
                        let _: ControlFlow<()> =
                            index.visit_neighbors(point, case.max_distance, |_idx, _dist| {
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

criterion_group! {
    name = benches;
    config = pin::criterion();
    targets = bench_persistence, bench_loaded_query, bench_knn
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
