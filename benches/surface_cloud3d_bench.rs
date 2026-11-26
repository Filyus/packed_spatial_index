//! 3D kNN and radius (`search_within`) benchmarks on a 2D-manifold point cloud
//! embedded in 3D (walls, floor, ceiling, partitions, and curved surfaces).
//!
//! Uniform volumetric 3D boxes produce a very different tree density, aspect
//! ratio, and Hilbert-curve locality from real 3D surface clouds (LiDAR, room
//! scans, mesh vertex clouds: 1–5M points on thin 2D surfaces, `k = 24–40`,
//! radius a few sample spacings).
//!
//! Run:
//!   cargo bench --all-features --bench surface_cloud3d_bench
//!
//! Environment knobs:
//!   SURFACE_CLOUD_N      number of surface points (default: 1_000_000)
//!   SURFACE_CLOUD_Q      number of query points   (default: 2_000)
//!   PAIRED_ROUNDS        interleaved rounds       (default: 15)
//!   PAIRED_REPS          reps per arm per round   (default: 5)

use std::hint::black_box;
use std::ops::ControlFlow;

use packed_spatial_index::{Box3D, Index3DBuilder, NeighborWorkspace, Point3D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[path = "support/paired.rs"]
mod paired;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// Generate `n` points sampled across 2D surfaces inside a 10 × 8 × 3 m room:
/// floor, ceiling, 4 outer walls, 2 interior partitions, and 4 spherical
/// objects. Returns `(point_boxes, total_surface_area)`.
fn generate_surface_cloud(n: usize, seed: u64) -> (Vec<Box3D>, f64) {
    let (lx, ly, lz) = (10.0_f64, 8.0_f64, 3.0_f64);
    let area_floor = lx * ly;
    let area_ceil = lx * ly;
    let area_wall_x = 2.0 * (lx * lz);
    let area_wall_y = 2.0 * (ly * lz);
    let area_partitions = 2.0 * (4.0 * lz);
    let sphere_r = 0.7_f64;
    let area_spheres = 4.0 * 4.0 * std::f64::consts::PI * sphere_r * sphere_r;
    let total_area =
        area_floor + area_ceil + area_wall_x + area_wall_y + area_partitions + area_spheres;

    let w_floor = area_floor / total_area;
    let w_ceil = w_floor + area_ceil / total_area;
    let w_wx = w_ceil + area_wall_x / total_area;
    let w_wy = w_wx + area_wall_y / total_area;
    let w_part = w_wy + area_partitions / total_area;

    let sphere_centers = [
        [2.5, 2.2, 0.9],
        [7.2, 2.5, 1.1],
        [3.1, 6.0, 0.8],
        [7.8, 5.8, 1.0],
    ];

    let mut rng = StdRng::seed_from_u64(seed);
    let mut boxes = Vec::with_capacity(n);
    for _ in 0..n {
        let u: f64 = rng.random();
        let jitter = rng.random_range(-0.002..0.002);
        let (x, y, z) = if u < w_floor {
            (rng.random_range(0.0..lx), rng.random_range(0.0..ly), jitter)
        } else if u < w_ceil {
            (
                rng.random_range(0.0..lx),
                rng.random_range(0.0..ly),
                lz + jitter,
            )
        } else if u < w_wx {
            let y_wall = if rng.random_bool(0.5) { 0.0 } else { ly };
            (
                rng.random_range(0.0..lx),
                y_wall + jitter,
                rng.random_range(0.0..lz),
            )
        } else if u < w_wy {
            let x_wall = if rng.random_bool(0.5) { 0.0 } else { lx };
            (
                x_wall + jitter,
                rng.random_range(0.0..ly),
                rng.random_range(0.0..lz),
            )
        } else if u < w_part {
            if rng.random_bool(0.5) {
                (
                    4.5 + jitter,
                    rng.random_range(1.0..5.0),
                    rng.random_range(0.0..lz),
                )
            } else {
                (
                    rng.random_range(5.0..9.0),
                    4.2 + jitter,
                    rng.random_range(0.0..lz),
                )
            }
        } else {
            let c = sphere_centers[rng.random_range(0..sphere_centers.len())];
            let z_u: f64 = rng.random_range(-1.0..1.0);
            let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let r_xy = (1.0 - z_u * z_u).max(0.0).sqrt();
            let r = sphere_r + jitter;
            (
                c[0] + r * r_xy * phi.cos(),
                c[1] + r * r_xy * phi.sin(),
                c[2] + r * z_u,
            )
        };
        boxes.push(Box3D::from_point(Point3D::new(x, y, z)));
    }
    (boxes, total_area)
}

fn main() {
    let n = env_usize("SURFACE_CLOUD_N", 1_000_000);
    let q = env_usize("SURFACE_CLOUD_Q", 2_000);
    let (boxes, total_area) = generate_surface_cloud(n, 0x50AF_ACE1);
    let spacing = (total_area / n as f64).sqrt();

    // Pick `q` distinct on-surface query points spread across the cloud.
    let stride = (n / q).max(1);
    let queries: Vec<Point3D> = (0..q)
        .map(|i| {
            let b = boxes[(i * stride) % n];
            Point3D::new(b.min_x, b.min_y, b.min_z)
        })
        .collect();

    println!(
        "3D surface cloud: n={n}, queries={q}, area={total_area:.1} m², mean spacing={:.4} m",
        spacing
    );

    let mut builder_f64 = Index3DBuilder::new(n);
    let mut builder_simd = Index3DBuilder::new(n);
    let mut builder_f32 = Index3DBuilder::new(n);
    for &b in &boxes {
        builder_f64.add(b);
        builder_simd.add(b);
        builder_f32.add(b);
    }
    let idx_f64 = builder_f64.finish().unwrap();
    let idx_simd = builder_simd.finish_simd().unwrap();
    let idx_f32 = builder_f32.finish_simd_f32().unwrap();

    for &k in &[24usize, 32, 40] {
        let mut ws_f64 = NeighborWorkspace::new();
        let mut ws_f64_each = NeighborWorkspace::new();
        let mut ws_simd = NeighborWorkspace::new();
        let mut ws_simd_each = NeighborWorkspace::new();
        let mut ws_f32 = NeighborWorkspace::new();
        let mut ws_f32_each = NeighborWorkspace::new();

        let title = format!("surface_cloud3d kNN (n={n}, q={q}, k={k})");
        let mut arms = [
            paired::arm("Index3D::neighbors_with", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let hits = idx_f64.neighbors_with(pt, k, f64::INFINITY, &mut ws_f64);
                    acc = acc.wrapping_add(hits.iter().copied().sum::<usize>());
                }
                black_box(acc)
            }),
            paired::arm("Index3D::neighbors_each", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let mut count = 0usize;
                    let _ = idx_f64.neighbors_each(pt, f64::INFINITY, |id, _d2| {
                        acc = acc.wrapping_add(id);
                        count += 1;
                        if count >= k {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    });
                }
                black_box(acc)
            }),
            paired::arm("Index3D::neighbors_each_with", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let mut count = 0usize;
                    let _ = idx_f64.neighbors_each_with(
                        pt,
                        f64::INFINITY,
                        &mut ws_f64_each,
                        |id, _d2| {
                            acc = acc.wrapping_add(id);
                            count += 1;
                            if count >= k {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        },
                    );
                }
                black_box(acc)
            }),
            paired::arm("SimdIndex3D::neighbors_with", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let hits = idx_simd.neighbors_with(pt, k, f64::INFINITY, &mut ws_simd);
                    acc = acc.wrapping_add(hits.iter().copied().sum::<usize>());
                }
                black_box(acc)
            }),
            paired::arm("SimdIndex3D::neighbors_each", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let mut count = 0usize;
                    let _ = idx_simd.neighbors_each(pt, f64::INFINITY, |id, _d2| {
                        acc = acc.wrapping_add(id);
                        count += 1;
                        if count >= k {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    });
                }
                black_box(acc)
            }),
            paired::arm("SimdIndex3D::neighbors_each_with", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let mut count = 0usize;
                    let _ = idx_simd.neighbors_each_with(
                        pt,
                        f64::INFINITY,
                        &mut ws_simd_each,
                        |id, _d2| {
                            acc = acc.wrapping_add(id);
                            count += 1;
                            if count >= k {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        },
                    );
                }
                black_box(acc)
            }),
            paired::arm("SimdIndex3DF32::neighbors_with", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let hits = idx_f32.neighbors_with(pt, k, f64::INFINITY, &mut ws_f32);
                    acc = acc.wrapping_add(hits.iter().copied().sum::<usize>());
                }
                black_box(acc)
            }),
            paired::arm("SimdIndex3DF32::neighbors_each", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let mut count = 0usize;
                    let _ = idx_f32.neighbors_each(pt, f64::INFINITY, |id, _d2| {
                        acc = acc.wrapping_add(id);
                        count += 1;
                        if count >= k {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    });
                }
                black_box(acc)
            }),
            paired::arm("SimdIndex3DF32::neighbors_each_with", || {
                let mut acc = 0usize;
                for &pt in &queries {
                    let mut count = 0usize;
                    let _ = idx_f32.neighbors_each_with(
                        pt,
                        f64::INFINITY,
                        &mut ws_f32_each,
                        |id, _d2| {
                            acc = acc.wrapping_add(id);
                            count += 1;
                            if count >= k {
                                ControlFlow::Break(())
                            } else {
                                ControlFlow::Continue(())
                            }
                        },
                    );
                }
                black_box(acc)
            }),
        ];
        paired::run(&title, &mut arms, "Index3D::neighbors_with");
    }

    for &(mult, label) in &[(3.0_f64, "3x_spacing"), (6.0_f64, "6x_spacing")] {
        let radius = mult * spacing;
        let mut out_f64 = Vec::new();
        let mut out_simd = Vec::new();

        let title = format!("surface_cloud3d search_within (n={n}, q={q}, r={label})");
        let mut arms = [
            paired::arm("Index3D::search_within_into", || {
                let mut total = 0usize;
                for &pt in &queries {
                    idx_f64.search_within_into(Box3D::from_point(pt), radius, &mut out_f64);
                    total += out_f64.len();
                }
                black_box(total)
            }),
            paired::arm("SimdIndex3D::search_within_into", || {
                let mut total = 0usize;
                for &pt in &queries {
                    idx_simd.search_within_into(Box3D::from_point(pt), radius, &mut out_simd);
                    total += out_simd.len();
                }
                black_box(total)
            }),
        ];
        paired::run(&title, &mut arms, "Index3D::search_within_into");
    }
}
