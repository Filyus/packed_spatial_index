//! Scalar 3D index benchmarks.
//!
//! These benches focus on the first production 3D path: `Index3DBuilder` ->
//! `Index3D` -> search/KNN. They intentionally keep Morton hidden as an
//! experimental baseline for layout/build-speed decisions.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use packed_spatial_index::experimental::{self, ExperimentalSortKey2D, ExperimentalSortKey3D};
use packed_spatial_index::{
    Bounds2D, Bounds3D, Index2D, Index2DBuilder, Index2DView, Index3D, Index3DBuilder, Index3DView,
    NeighborWorkspace, Point2D, Point3D, SearchWorkspace,
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
    PlanarXY,
    Uniform,
    FlatZ,
    Clustered,
}

impl DatasetKind {
    fn name(self) -> &'static str {
        match self {
            DatasetKind::PlanarXY => "planar_xy",
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

fn build_index2d(boxes: &[Bounds2D], node_size: usize) -> Index2D {
    let mut builder = Index2DBuilder::new(boxes.len())
        .node_size(node_size)
        .experimental_sort_key(ExperimentalSortKey2D::HilbertLut);
    for &bounds in boxes {
        builder.add(bounds);
    }
    builder.finish().unwrap()
}

fn gen_boxes(kind: DatasetKind, n: usize, seed: u64) -> Vec<Bounds3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    match kind {
        DatasetKind::PlanarXY => (0..n)
            .map(|_| {
                let x = rng.random_range(0.0..10_000.0);
                let y = rng.random_range(0.0..10_000.0);
                let dx = rng.random_range(1.0..40.0);
                let dy = rng.random_range(1.0..40.0);
                Bounds3D::new(x, y, 0.0, x + dx, y + dy, 0.0)
            })
            .collect(),
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
            DatasetKind::PlanarXY => {
                let x = rng.random_range(0.0..10_000.0);
                let y = rng.random_range(0.0..10_000.0);
                let dx = rng.random_range(50.0..300.0);
                let dy = rng.random_range(50.0..300.0);
                Bounds3D::new(x, y, 0.0, x + dx, y + dy, 0.0)
            }
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
            DatasetKind::PlanarXY => Point3D::new(
                rng.random_range(0.0..10_000.0),
                rng.random_range(0.0..10_000.0),
                0.0,
            ),
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

fn project_boxes_2d(boxes: &[Bounds3D]) -> Vec<Bounds2D> {
    boxes
        .iter()
        .map(|b| Bounds2D::new(b.min_x, b.min_y, b.max_x, b.max_y))
        .collect()
}

fn project_queries_2d(queries: &[Bounds3D]) -> Vec<Bounds2D> {
    queries
        .iter()
        .map(|b| Bounds2D::new(b.min_x, b.min_y, b.max_x, b.max_y))
        .collect()
}

fn project_points_2d(points: &[Point3D]) -> Vec<Point2D> {
    points.iter().map(|p| Point2D::new(p.x, p.y)).collect()
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

fn bench_dimension_encode(c: &mut Criterion) {
    let n = 262_144usize;
    let mut rng = StdRng::seed_from_u64(0x2D3D);
    let coords2d: Vec<(u16, u16)> = (0..n)
        .map(|_| (rng.random::<u16>(), rng.random::<u16>()))
        .collect();
    let coords3d: Vec<(u32, u32, u32)> = coords2d
        .iter()
        .map(|&(x, y)| (u32::from(x), u32::from(y), u32::from(rng.random::<u16>())))
        .collect();

    let mut group = c.benchmark_group("dimension_compare_encode");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("hilbert2d_lut", |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(x, y) in &coords2d {
                checksum ^= u64::from(experimental::lut(black_box(x), black_box(y)));
            }
            black_box(checksum);
        });
    });
    group.bench_function("hilbert3d_pair_lut", |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(x, y, z) in &coords3d {
                checksum ^= experimental::encode_hilbert3_pair_lut(
                    black_box(x),
                    black_box(y),
                    black_box(z),
                );
            }
            black_box(checksum);
        });
    });
    group.bench_function("hilbert3d_nibble_lut", |b| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(x, y, z) in &coords3d {
                checksum ^= experimental::encode_hilbert3_nibble_lut(
                    black_box(x),
                    black_box(y),
                    black_box(z),
                );
            }
            black_box(checksum);
        });
    });
    group.finish();
}

fn bench_dimension_radix(c: &mut Criterion) {
    let n = 262_144usize;
    let mut rng = StdRng::seed_from_u64(0x2D3D);
    let pairs: Vec<(u64, usize)> = (0..n)
        .map(|i| (rng.random::<u64>() & ((1u64 << 48) - 1), i))
        .collect();

    let mut group = c.benchmark_group("dimension_compare_radix");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("hilbert3d_radix_8bit_known_48", |b| {
        b.iter_batched_ref(
            || pairs.clone(),
            |a| radix_sort_pairs_u64_known_bits(a, black_box(8), 48),
            criterion::BatchSize::LargeInput,
        );
    });
    group.bench_function("hilbert3d_radix_8bit_data_driven", |b| {
        b.iter_batched_ref(
            || pairs.clone(),
            |a| experimental::radix_sort_pairs_u64(a, black_box(8)),
            criterion::BatchSize::LargeInput,
        );
    });
    group.bench_function("hilbert3d_radix_8bit_full_64", |b| {
        b.iter_batched_ref(
            || pairs.clone(),
            |a| radix_sort_pairs_u64_full_passes(a, black_box(8)),
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn radix_sort_pairs_u64_known_bits(a: &mut [(u64, usize)], bits: u32, used_bits: u32) {
    radix_sort_pairs_u64_fixed_passes(a, bits, used_bits.div_ceil(bits));
}

fn radix_sort_pairs_u64_full_passes(a: &mut [(u64, usize)], bits: u32) {
    radix_sort_pairs_u64_fixed_passes(a, bits, 64u32.div_ceil(bits));
}

fn radix_sort_pairs_u64_fixed_passes(a: &mut [(u64, usize)], bits: u32, passes: u32) {
    let n = a.len();
    if n <= 1 || passes == 0 {
        return;
    }

    let bits = bits.clamp(4, 12);
    let buckets = 1usize << bits;
    let mask = (buckets as u64) - 1;
    let mut tmp = vec![(0u64, 0usize); n];
    let mut counts = vec![0usize; buckets];

    fn pass(
        src: &[(u64, usize)],
        dst: &mut [(u64, usize)],
        shift: u32,
        mask: u64,
        counts: &mut [usize],
    ) {
        counts.fill(0);
        for &(key, _) in src {
            counts[((key >> shift) & mask) as usize] += 1;
        }

        let mut sum = 0usize;
        for count in counts.iter_mut() {
            let current = *count;
            *count = sum;
            sum += current;
        }

        for &pair in src {
            let bucket = ((pair.0 >> shift) & mask) as usize;
            dst[counts[bucket]] = pair;
            counts[bucket] += 1;
        }
    }

    for pass_idx in 0..passes {
        let shift = pass_idx * bits;
        if pass_idx % 2 == 0 {
            pass(a, &mut tmp, shift, mask, &mut counts);
        } else {
            pass(&tmp, a, shift, mask, &mut counts);
        }
    }
    if passes % 2 == 1 {
        a.copy_from_slice(&tmp);
    }
}

fn bench_dimension_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_compare_build");

    for kind in [DatasetKind::PlanarXY, DatasetKind::Uniform] {
        for &n in &[1_000usize, 100_000] {
            let boxes3d = gen_boxes(kind, n, 0xD1B0);
            let boxes2d = project_boxes_2d(&boxes3d);
            group.throughput(Throughput::Elements(n as u64));
            for &node_size in &[8usize, 16] {
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_2d_node{}", kind.name(), node_size), n),
                    &boxes2d,
                    |b, boxes| b.iter(|| black_box(build_index2d(boxes, node_size))),
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_3d_node{}", kind.name(), node_size), n),
                    &boxes3d,
                    |b, boxes| {
                        b.iter(|| {
                            black_box(build_index(
                                boxes,
                                node_size,
                                ExperimentalSortKey3D::Hilbert,
                                false,
                            ));
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_dimension_search(c: &mut Criterion) {
    let n = 100_000usize;
    let mut group = c.benchmark_group("dimension_compare_search");

    for kind in [DatasetKind::PlanarXY, DatasetKind::Uniform] {
        let boxes3d = gen_boxes(kind, n, 0xD2B0);
        let boxes2d = project_boxes_2d(&boxes3d);
        let queries3d = gen_queries(kind, QUERY_COUNT, 0xD2B1);
        let queries2d = project_queries_2d(&queries3d);
        for &node_size in &[8usize, 16] {
            let index2d = build_index2d(&boxes2d, node_size);
            let index3d = build_index(&boxes3d, node_size, ExperimentalSortKey3D::Hilbert, false);
            group.bench_function(format!("{}_2d_node{}", kind.name(), node_size), |b| {
                let mut workspace = SearchWorkspace::new();
                b.iter(|| {
                    let mut total = 0usize;
                    for &query in &queries2d {
                        total += index2d.search_with(query, &mut workspace).len();
                    }
                    black_box(total);
                });
            });
            group.bench_function(format!("{}_3d_node{}", kind.name(), node_size), |b| {
                let mut workspace = SearchWorkspace::new();
                b.iter(|| {
                    let mut total = 0usize;
                    for &query in &queries3d {
                        total += index3d.search_with(query, &mut workspace).len();
                    }
                    black_box(total);
                });
            });
        }
    }

    group.finish();
}

fn bench_dimension_knn(c: &mut Criterion) {
    let n = 100_000usize;
    let mut group = c.benchmark_group("dimension_compare_knn");

    for kind in [DatasetKind::PlanarXY, DatasetKind::Uniform] {
        let boxes3d = gen_boxes(kind, n, 0xD3B0);
        let boxes2d = project_boxes_2d(&boxes3d);
        let points3d = gen_points(kind, KNN_COUNT, 0xD3B1);
        let points2d = project_points_2d(&points3d);
        for &node_size in &[8usize, 16] {
            let index2d = build_index2d(&boxes2d, node_size);
            let index3d = build_index(&boxes3d, node_size, ExperimentalSortKey3D::Hilbert, false);
            for &(case, max_results) in &[("top_1", 1usize), ("top_10", 10usize)] {
                group.bench_function(
                    format!("{}_2d_{}_node{}", kind.name(), case, node_size),
                    |b| {
                        let mut workspace = NeighborWorkspace::new();
                        b.iter(|| {
                            let mut total = 0usize;
                            for &point in &points2d {
                                total += index2d
                                    .neighbors_with(
                                        point,
                                        max_results,
                                        f64::INFINITY,
                                        &mut workspace,
                                    )
                                    .len();
                            }
                            black_box(total);
                        });
                    },
                );
                group.bench_function(
                    format!("{}_3d_{}_node{}", kind.name(), case, node_size),
                    |b| {
                        let mut workspace = NeighborWorkspace::new();
                        b.iter(|| {
                            let mut total = 0usize;
                            for &point in &points3d {
                                total += index3d
                                    .neighbors_with(
                                        point,
                                        max_results,
                                        f64::INFINITY,
                                        &mut workspace,
                                    )
                                    .len();
                            }
                            black_box(total);
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_dimension_persistence(c: &mut Criterion) {
    let n = 100_000usize;
    let boxes3d = gen_boxes(DatasetKind::PlanarXY, n, 0xD4B0);
    let boxes2d = project_boxes_2d(&boxes3d);
    let index2d = build_index2d(&boxes2d, 16);
    let index3d = build_index(&boxes3d, 16, ExperimentalSortKey3D::Hilbert, false);
    let bytes2d = index2d.to_bytes();
    let bytes3d = index3d.to_bytes();

    let mut group = c.benchmark_group("dimension_compare_persistence");
    group.bench_function("2d_to_bytes_100000", |b| {
        b.iter(|| black_box(index2d.to_bytes()))
    });
    group.bench_function("2d_to_bytes_into_100000", |b| {
        let mut out = Vec::with_capacity(bytes2d.len());
        b.iter(|| {
            index2d.to_bytes_into(&mut out);
            black_box(out.len())
        })
    });
    group.bench_function("3d_to_bytes_100000", |b| {
        b.iter(|| black_box(index3d.to_bytes()))
    });
    group.bench_function("3d_to_bytes_into_100000", |b| {
        let mut out = Vec::with_capacity(bytes3d.len());
        b.iter(|| {
            index3d.to_bytes_into(&mut out);
            black_box(out.len())
        })
    });
    group.bench_function("2d_from_bytes_owned_100000", |b| {
        b.iter(|| black_box(Index2D::from_bytes(&bytes2d).unwrap()))
    });
    group.bench_function("3d_from_bytes_owned_100000", |b| {
        b.iter(|| black_box(Index3D::from_bytes(&bytes3d).unwrap()))
    });
    group.bench_function("2d_from_bytes_view_100000", |b| {
        b.iter(|| black_box(Index2DView::from_bytes(&bytes2d).unwrap()))
    });
    group.bench_function("3d_from_bytes_view_100000", |b| {
        b.iter(|| black_box(Index3DView::from_bytes(&bytes3d).unwrap()))
    });
    group.finish();
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

#[cfg(not(feature = "simd"))]
fn bench_simd_search(_c: &mut Criterion) {}

#[cfg(feature = "simd")]
fn bench_simd_search(c: &mut Criterion) {
    let n = 100_000usize;
    let mut group = c.benchmark_group("index3d_simd_search");
    for kind in [DatasetKind::Uniform, DatasetKind::FlatZ] {
        let boxes = gen_boxes(kind, n, 0x5D10);
        let queries = gen_queries(kind, QUERY_COUNT, 0x5D11);
        let node_size = 16usize;

        let mut scalar_builder = Index3DBuilder::new(boxes.len()).node_size(node_size);
        let mut simd_builder = Index3DBuilder::new(boxes.len()).node_size(node_size);
        for &b in &boxes {
            scalar_builder.add(b);
            simd_builder.add(b);
        }
        let scalar = scalar_builder.finish().unwrap();
        let simd = simd_builder.finish_simd().unwrap();

        group.bench_function(format!("{}_scalar", kind.name()), |b| {
            let mut workspace = SearchWorkspace::new();
            b.iter(|| {
                let mut total = 0usize;
                for &query in &queries {
                    total += scalar.search_with(query, &mut workspace).len();
                }
                black_box(total);
            });
        });
        group.bench_function(format!("{}_simd", kind.name()), |b| {
            let mut workspace = SearchWorkspace::new();
            b.iter(|| {
                let mut total = 0usize;
                for &query in &queries {
                    total += simd.search_with(query, &mut workspace).len();
                }
                black_box(total);
            });
        });
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
    bench_dimension_encode,
    bench_dimension_radix,
    bench_dimension_build,
    bench_dimension_search,
    bench_dimension_knn,
    bench_dimension_persistence,
    bench_build,
    bench_search,
    bench_simd_search,
    bench_persistence,
    bench_loaded_view,
    bench_knn
);
criterion_main!(benches);
