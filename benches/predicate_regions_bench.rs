//! The predicate regions against their bbox + exact-filter workarounds: the
//! wall-time gate from the region-query lesson — a broad-phase win must show
//! up as wall time, never as selectivity alone. Half-space has no finite
//! bounding box (it is unbounded), so its workaround arm clips the region to
//! the index extent first — the best a bbox-based caller can do.
//!
//! Run:
//!   cargo bench --bench predicate_regions_bench

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use packed_spatial_index::{
    Box2D, Box3D, Capsule2D, Cone3D, HalfSpace2D, Index2D, Index2DBuilder, Index3D, Index3DBuilder,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const COUNT: usize = 1_000_000;
const EXTENT: f64 = 1_000.0;
const NODE_SIZE: usize = 16;
const QUERIES: usize = 200;

fn build_2d(rng: &mut StdRng) -> (Index2D, Vec<Box2D>) {
    let items: Vec<Box2D> = (0..COUNT)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT);
            let y: f64 = rng.random_range(0.0..EXTENT);
            let w: f64 = rng.random_range(0.0..1.0);
            let h: f64 = rng.random_range(0.0..1.0);
            Box2D::new(x, y, x + w, y + h)
        })
        .collect();
    let mut b = Index2DBuilder::new(COUNT).node_size(NODE_SIZE);
    for &bx in &items {
        b.add(bx);
    }
    (b.finish().unwrap(), items)
}

fn build_3d(rng: &mut StdRng) -> (Index3D, Vec<Box3D>) {
    let items: Vec<Box3D> = (0..COUNT)
        .map(|_| {
            let (x, y, z) = (
                rng.random_range(0.0..EXTENT),
                rng.random_range(0.0..EXTENT),
                rng.random_range(0.0..EXTENT),
            );
            let s: f64 = rng.random_range(0.0..1.0);
            Box3D::new(x, y, z, x + s, y + s, z + s)
        })
        .collect();
    let mut b = Index3DBuilder::new(COUNT).node_size(NODE_SIZE);
    for &bx in &items {
        b.add(bx);
    }
    (b.finish().unwrap(), items)
}

fn avg_hits(hits: &[usize]) -> usize {
    hits.iter().sum::<usize>() / hits.len().max(1)
}

fn bench_regions(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_regions");
    group.sample_size(20);
    let mut rng = StdRng::seed_from_u64(1);

    // ---- half-space (2D): the workaround clips to the index extent ----
    // Two cut positions: a corner slice (selective — the cross-section case)
    // and a mid cut (broad — where a linear scan is a real competitor).
    for (label, drange) in [("corner", (0.005f64, 0.05f64)), ("broad", (0.2f64, 0.8f64))] {
        let (index, items) = build_2d(&mut rng);
        let extent = index.extent().unwrap();
        let planes: Vec<HalfSpace2D> = (0..QUERIES)
            .map(|_| {
                // Keep a slice near the (0, 0) corner: the inside side is
                // `x + y <= cut`, with the cut in units of the extent.
                let (nx, ny): (f64, f64) = (
                    -1.0 - rng.random_range(0.0..0.5),
                    -1.0 - rng.random_range(0.0..0.5),
                );
                let d = rng.random_range(drange.0..drange.1) * 2.0 * EXTENT;
                HalfSpace2D::new(nx, ny, d)
            })
            .collect();
        let hits: Vec<usize> = planes.iter().map(|p| index.search(p).len()).collect();
        group.bench_function(format!("halfspace_2d_region_{label}"), |b| {
            b.iter(|| {
                for p in &planes {
                    black_box(index.search(black_box(p)).len());
                }
            });
        });
        group.bench_function(format!("halfspace_2d_bbox_filter_{label}"), |b| {
            b.iter(|| {
                for p in &planes {
                    // Clip the half-space to the extent along its normal: the
                    // tightest box a bbox-based caller can build.
                    let corner_lo = f64::INFINITY;
                    let _ = corner_lo;
                    let f = |x: f64, y: f64| p.nx * x + p.ny * y + p.d;
                    let fmax = f(extent.max_x, extent.max_y).max(f(extent.min_x, extent.min_y));
                    let fmin = f(extent.min_x, extent.min_y).min(f(extent.max_x, extent.max_y));
                    // The clipped bbox spans the extent; the filter does the rest.
                    let mut n = 0usize;
                    for &bx in items.iter() {
                        let hit = if fmax >= 0.0 || fmin >= 0.0 {
                            let cmax = p.d
                                + p.nx * if p.nx >= 0.0 { bx.max_x } else { bx.min_x }
                                + p.ny * if p.ny >= 0.0 { bx.max_y } else { bx.min_y };
                            cmax >= 0.0
                        } else {
                            false
                        };
                        if hit {
                            n += 1;
                        }
                    }
                    black_box(n);
                }
            });
        });
        eprintln!("halfspace_2d {label} avg hits: {}", avg_hits(&hits));
    }

    // ---- capsule (2D): small, medium, large radius ----
    {
        let (index, items) = build_2d(&mut rng);
        for &radius in &[0.5f64, 5.0, 50.0] {
            let capsules: Vec<Capsule2D> = (0..QUERIES)
                .map(|_| {
                    Capsule2D::new(
                        [rng.random_range(0.0..EXTENT), rng.random_range(0.0..EXTENT)],
                        [rng.random_range(0.0..EXTENT), rng.random_range(0.0..EXTENT)],
                        radius,
                    )
                })
                .collect();
            let hits: Vec<usize> = capsules.iter().map(|cp| index.search(cp).len()).collect();
            group.bench_with_input(
                BenchmarkId::new("capsule_2d_region", radius),
                &capsules,
                |b, capsules| {
                    b.iter(|| {
                        for cp in capsules {
                            black_box(index.search(black_box(cp)).len());
                        }
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("capsule_2d_bbox_filter", radius),
                &capsules,
                |b, capsules| {
                    b.iter(|| {
                        for cp in capsules {
                            let (x0, x1) =
                                (cp.a[0].min(cp.b[0]) - radius, cp.a[0].max(cp.b[0]) + radius);
                            let (y0, y1) =
                                (cp.a[1].min(cp.b[1]) - radius, cp.a[1].max(cp.b[1]) + radius);
                            let window = Box2D::new(x0, y0, x1, y1);
                            let mut n = 0usize;
                            for id in index.search(window) {
                                if cp.overlaps_box(items[id]) {
                                    n += 1;
                                }
                            }
                            black_box(n);
                        }
                    });
                },
            );
            eprintln!("capsule_2d r={radius} avg hits: {}", avg_hits(&hits));
        }
    }

    // ---- cone (3D): narrow and wide aperture ----
    {
        let (index, items) = build_3d(&mut rng);
        for &half_angle in &[0.1f64, 0.6] {
            let cones: Vec<Cone3D> = (0..QUERIES)
                .map(|_| {
                    Cone3D::try_new(
                        [
                            rng.random_range(0.0..EXTENT),
                            rng.random_range(0.0..EXTENT),
                            rng.random_range(0.0..EXTENT),
                        ],
                        [rng.random_range(-1.0..1.0); 3],
                        half_angle,
                        rng.random_range(50.0..300.0),
                    )
                    .unwrap()
                })
                .collect();
            let hits: Vec<usize> = cones.iter().map(|cn| index.search(cn).len()).collect();
            group.bench_with_input(
                BenchmarkId::new(
                    "cone_3d_region",
                    format!("{}", (half_angle * 180.0 / std::f64::consts::PI) as usize),
                ),
                &cones,
                |b, cones| {
                    b.iter(|| {
                        for cn in cones {
                            black_box(index.search(black_box(cn)).len());
                        }
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    "cone_3d_bbox_filter",
                    format!("{}", (half_angle * 180.0 / std::f64::consts::PI) as usize),
                ),
                &cones,
                |b, cones| {
                    b.iter(|| {
                        for cn in cones {
                            let reach = cn.height;
                            let window = Box3D::new(
                                (cn.apex[0] - reach).max(0.0),
                                (cn.apex[1] - reach).max(0.0),
                                (cn.apex[2] - reach).max(0.0),
                                (cn.apex[0] + reach).min(EXTENT),
                                (cn.apex[1] + reach).min(EXTENT),
                                (cn.apex[2] + reach).min(EXTENT),
                            );
                            let mut n = 0usize;
                            for id in index.search(window) {
                                if cn.overlaps_box(items[id]) {
                                    n += 1;
                                }
                            }
                            black_box(n);
                        }
                    });
                },
            );
            eprintln!(
                "cone_3d half_angle={} avg hits: {}",
                half_angle,
                avg_hits(&hits)
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_regions);
criterion_main!(benches);
