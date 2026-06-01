//! Scalar 3D index benchmarks.
//!
//! These benches focus on the first production 3D path: `Index3DBuilder` ->
//! `Index3D` -> search/KNN. They intentionally keep Morton hidden as an
//! experimental baseline for layout/build-speed decisions.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use packed_spatial_index::experimental::ExperimentalSortKey3D;
use packed_spatial_index::{
    Bounds3D, Index3D, Index3DBuilder, Index3DView, NeighborWorkspace, Point3D, SearchWorkspace,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const QUERY_COUNT: usize = 1_000;
const KNN_COUNT: usize = 1_000;
const PERSISTENCE_NODE_SIZE: usize = 16;
const LOADED_KNN_LIMIT: usize = 10;
const LOADED_KNN_MAX_DISTANCE: f64 = f64::INFINITY;

#[derive(Clone, Copy)]
enum DatasetKind {
    Uniform,
    FlatZ,
    Clustered,
}

impl DatasetKind {
    fn name(self) -> &'static str {
        match self {
            DatasetKind::Uniform => "uniform",
            DatasetKind::FlatZ => "flat_z",
            DatasetKind::Clustered => "clustered",
        }
    }
}

fn build_index(
    boxes: &[Bounds3D],
    node_size: usize,
    sort_key: ExperimentalSortKey3D,
    parallel: bool,
) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len())
        .node_size(node_size)
        .experimental_sort_key(sort_key);
    #[cfg(feature = "parallel")]
    {
        builder = builder.parallel(parallel);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

fn gen_boxes(kind: DatasetKind, n: usize, seed: u64) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    match kind {
        DatasetKind::Uniform => (0..n)
            .map(|_| {
                random_box(
                    &mut rng,
                    0.0..10_000.0,
                    0.0..10_000.0,
                    0.0..10_000.0,
                    1.0..40.0,
                )
            })
            .collect(),
        DatasetKind::FlatZ => (0..n)
            .map(|_| random_box(&mut rng, 0.0..10_000.0, 0.0..10_000.0, 0.0..20.0, 1.0..35.0))
            .collect(),
        DatasetKind::Clustered => (0..n)
            .map(|i| {
                let cluster = (i % 8) as f64;
                let base_x = 1_000.0 + cluster * 900.0;
                let base_y = 1_000.0 + (cluster % 4.0) * 1_200.0;
                let base_z = 1_000.0 + (cluster / 2.0).floor() * 700.0;
                random_box(
                    &mut rng,
                    base_x..base_x + 250.0,
                    base_y..base_y + 250.0,
                    base_z..base_z + 250.0,
                    1.0..30.0,
                )
            })
            .collect(),
    }
}

fn gen_queries(kind: DatasetKind, n: usize, seed: u64) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| match kind {
            DatasetKind::Uniform => random_box(
                &mut rng,
                0.0..10_000.0,
                0.0..10_000.0,
                0.0..10_000.0,
                50.0..300.0,
            ),
            DatasetKind::FlatZ => random_box(
                &mut rng,
                0.0..10_000.0,
                0.0..10_000.0,
                0.0..20.0,
                30.0..250.0,
            ),
            DatasetKind::Clustered => random_box(
                &mut rng,
                800.0..8_500.0,
                800.0..6_000.0,
                800.0..4_000.0,
                50.0..300.0,
            ),
        })
        .collect()
}

fn gen_points(kind: DatasetKind, n: usize, seed: u64) -> Vec<Point3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| match kind {
            DatasetKind::Uniform => Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
            ),
            DatasetKind::FlatZ => Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..20.0),
            ),
            DatasetKind::Clustered => Point3D::new(
                rng.random_range(800.0..8_500.0),
                rng.random_range(800.0..6_000.0),
                rng.random_range(800.0..4_000.0),
            ),
        })
        .collect()
}

fn random_box(
    rng: &mut StdRng,
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
    z_range: std::ops::Range<f64>,
    size_range: std::ops::Range<f64>,
) -> Bounds3D {
    let x = rng.random_range(x_range);
    let y = rng.random_range(y_range);
    let z = rng.random_range(z_range);
    let dx = rng.random_range(size_range.clone());
    let dy = rng.random_range(size_range.clone());
    let dz = rng.random_range(size_range);
    Bounds3D::new(x, y, z, x + dx, y + dy, z + dz)
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("index3d_build");
    for &n in &[17usize, 1_000, 100_000] {
        let boxes = gen_boxes(DatasetKind::Uniform, n, 0x3D00);
        group.throughput(Throughput::Elements(n as u64));
        for &node_size in &[8usize, 16, 32] {
            for &(name, sort_key) in &[
                ("hilbert", ExperimentalSortKey3D::Hilbert),
                ("morton", ExperimentalSortKey3D::Morton),
            ] {
                group.bench_with_input(
                    BenchmarkId::new(format!("{name}_node{node_size}"), n),
                    &boxes,
                    |b, boxes| {
                        b.iter(|| {
                            black_box(build_index(boxes, node_size, sort_key, false));
                        });
                    },
                );
            }

            #[cfg(feature = "parallel")]
            group.bench_with_input(
                BenchmarkId::new(format!("hilbert_parallel_node{node_size}"), n),
                &boxes,
                |b, boxes| {
                    b.iter(|| {
                        black_box(build_index(
                            boxes,
                            node_size,
                            ExperimentalSortKey3D::Hilbert,
                            true,
                        ));
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let n = 100_000usize;
    let mut group = c.benchmark_group("index3d_search");
    for kind in [
        DatasetKind::Uniform,
        DatasetKind::FlatZ,
        DatasetKind::Clustered,
    ] {
        let boxes = gen_boxes(kind, n, 0x3D10);
        let queries = gen_queries(kind, QUERY_COUNT, 0x3D11);
        for &node_size in &[8usize, 16, 32] {
            for &(name, sort_key) in &[
                ("hilbert", ExperimentalSortKey3D::Hilbert),
                ("morton", ExperimentalSortKey3D::Morton),
            ] {
                let index = build_index(&boxes, node_size, sort_key, false);
                let id = format!("{}_{}_node{}", kind.name(), name, node_size);
                group.bench_function(id, |b| {
                    let mut workspace = SearchWorkspace::new();
                    b.iter(|| {
                        let mut total = 0usize;
                        for &query in &queries {
                            total += index.search_with(query, &mut workspace).len();
                        }
                        black_box(total);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("index3d_persistence");

    for &n in &[1_000usize, 100_000, 1_000_000] {
        let boxes = gen_boxes(DatasetKind::Uniform, n, 0x3D30);
        let index = build_index(
            &boxes,
            PERSISTENCE_NODE_SIZE,
            ExperimentalSortKey3D::Hilbert,
            false,
        );
        let bytes = index.to_bytes();

        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("to_bytes", n), &index, |b, index| {
            b.iter(|| black_box(index.to_bytes()))
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
    }

    group.finish();
}

fn bench_loaded_view(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes = gen_boxes(DatasetKind::Uniform, n, 0x3D40);
    let index = build_index(
        &boxes,
        PERSISTENCE_NODE_SIZE,
        ExperimentalSortKey3D::Hilbert,
        false,
    );
    let bytes = index.to_bytes();
    let view = Index3DView::from_bytes(&bytes).unwrap();
    let queries = gen_queries(DatasetKind::Uniform, QUERY_COUNT, 0x3D41);
    let points = gen_points(DatasetKind::Uniform, KNN_COUNT, 0x3D42);

    let mut group = c.benchmark_group("index3d_loaded_view");
    group.bench_function("index_search", |b| {
        let mut workspace = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &query in &queries {
                total += index.search_with(query, &mut workspace).len();
            }
            black_box(total);
        });
    });
    group.bench_function("view_search", |b| {
        let mut workspace = SearchWorkspace::new();
        b.iter(|| {
            let mut total = 0usize;
            for &query in &queries {
                total += view.search_with(query, &mut workspace).len();
            }
            black_box(total);
        });
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
            black_box(total);
        });
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
            black_box(total);
        });
    });
    group.finish();
}

fn bench_knn(c: &mut Criterion) {
    let n = 100_000usize;
    let mut group = c.benchmark_group("index3d_knn");
    for kind in [
        DatasetKind::Uniform,
        DatasetKind::FlatZ,
        DatasetKind::Clustered,
    ] {
        let boxes = gen_boxes(kind, n, 0x3D20);
        let points = gen_points(kind, KNN_COUNT, 0x3D21);
        for &node_size in &[8usize, 16, 32] {
            let index = build_index(&boxes, node_size, ExperimentalSortKey3D::Hilbert, false);
            for &(name, max_results, max_distance) in &[
                ("top_1", 1usize, f64::INFINITY),
                ("top_10", 10usize, f64::INFINITY),
                ("top_10_radius_100", 10usize, 100.0),
            ] {
                let id = format!("{}_{}_node{}", kind.name(), name, node_size);
                group.bench_function(id, |b| {
                    let mut workspace = NeighborWorkspace::new();
                    b.iter(|| {
                        let mut total = 0usize;
                        for &point in &points {
                            total += index
                                .neighbors_with(point, max_results, max_distance, &mut workspace)
                                .len();
                        }
                        black_box(total);
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_search,
    bench_persistence,
    bench_loaded_view,
    bench_knn
);
criterion_main!(benches);
