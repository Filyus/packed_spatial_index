//! `raycast_any` on the SIMD frontends: the depth-first descent against the
//! 0.33.0 form, `raycast_each` stopped at its first hit, in ONE binary,
//! interleaved.
//!
//! `any` promises no order, so the priority queue the old form kept was pure
//! overhead: every child of every popped node went through `enter_t` and a heap
//! push. The new form walks a stack and returns at the first leaf box the
//! segment enters, with the owned index testing children through the SoA
//! vector kernel (AVX-512 eight at a time for an oblique ray where the CPU has
//! it, `wide::f64x4` otherwise) and the view through the scalar hit mask its
//! interleaved records allow.
//!
//! Each row is one ray set on one scene, every ratio against the old owned
//! form; read a view's gain as its depth-first ratio over its queue ratio. The scalar
//! `Index3D` / `Index2D` `raycast_any`, already depth-first, is the control the
//! change cannot touch. Oblique rays take the AVX-512 path where it exists;
//! axis-parallel rays always take `wide`, so both kernels are timed on an
//! AVX-512 machine. On oblique rows the `kernel` arms cap the dispatch at
//! `wide` and at AVX2, so the shipped arm (AVX-512 where the CPU has it) reads
//! against both: that is what the dispatch order rests on. Scenes and oblique rays are the ones of
//! `paired_competitors` (`support/competitors.rs`); the checksum is the number
//! of rays that hit, and every arm must agree on it.
//!
//! Run:
//!   taskset -c 1 cargo bench --features simd --bench paired_raycast_any

use std::hint::black_box;

use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder, Point2D, Point3D, Ray2D, Ray3D,
    SimdIndex2D, SimdIndex2DView, SimdIndex3D, SimdIndex3DView,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use competitors::{
    EARLY_EXIT_QUERIES, EXTENT_2D, RAY_LENGTH, SCENE_CLASSES, WORLD_3D, clustered_boxes_3d,
    random_rays, set_size, uniform_boxes_3d,
};

#[path = "support/competitors.rs"]
mod competitors;
#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const NODE_SIZE: usize = 16;
/// 2D rays are short enough that a good share of them miss.
const RAY_LENGTH_2D: f64 = 50.0;

/// `n` rays of [`RAY_LENGTH`] from random origins along a random axis, either
/// sign: every one takes the `wide` path.
fn axis_rays_3d(n: usize, seed: u64) -> Vec<Ray3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..set_size(n))
        .map(|_| {
            let mut d = [0.0; 3];
            d[rng.random_range(0..3usize)] = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            Ray3D::new(
                Point3D::new(
                    rng.random_range(0.0..WORLD_3D),
                    rng.random_range(0.0..WORLD_3D),
                    rng.random_range(0.0..WORLD_3D),
                ),
                d[0],
                d[1],
                d[2],
                RAY_LENGTH,
            )
        })
        .collect()
}

fn rays_2d(n: usize, seed: u64, axis: bool) -> Vec<Ray2D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..set_size(n))
        .map(|_| {
            let origin = Point2D::new(
                rng.random_range(0.0..EXTENT_2D),
                rng.random_range(0.0..EXTENT_2D),
            );
            let (dx, dy) = if axis {
                let s = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                if rng.random_bool(0.5) {
                    (s, 0.0)
                } else {
                    (0.0, s)
                }
            } else {
                let theta: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                (theta.cos(), theta.sin())
            };
            Ray2D::new(origin, dx, dy, RAY_LENGTH_2D)
        })
        .collect()
}

fn rows_3d(tag: &str, boxes: &[Box3D], seed: u64) {
    let build = || {
        let mut b = Index3DBuilder::new(boxes.len()).node_size(NODE_SIZE);
        for &r in boxes {
            b.add(r);
        }
        b
    };
    let scalar: Index3D = build().finish().unwrap();
    let simd: SimdIndex3D = build().finish_simd().unwrap();
    let bytes = simd.to_bytes();
    let view = SimdIndex3DView::from_bytes(&bytes).unwrap();

    for (kind, rays) in [
        ("oblique", random_rays(EARLY_EXIT_QUERIES, seed ^ 0xC105E)),
        ("axis", axis_rays_3d(EARLY_EXIT_QUERIES, seed ^ 0xA715)),
    ] {
        let hits = rays.iter().filter(|&&r| scalar.raycast_any(r)).count();
        let label = format!(
            "3d {tag}, {kind} rays, {} rays ({:.0}% hit)",
            rays.len(),
            100.0 * hits as f64 / rays.len() as f64
        );
        let count = |f: &dyn Fn(Ray3D) -> bool| {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += usize::from(f(r));
            }
            t
        };
        let mut arms = vec![
            paired::arm("SimdIndex3D queue (0.33.0)", || {
                count(&|r| simd.raycast_any_queue(r))
            }),
            paired::arm("SimdIndex3D depth-first", || {
                count(&|r| simd.raycast_any(r))
            }),
            paired::arm("Index3D raycast_any (control)", || {
                count(&|r| scalar.raycast_any(r))
            }),
            paired::arm("SimdIndex3DView queue (0.33.0)", || {
                count(&|r| view.raycast_any_queue(r))
            }),
            paired::arm("SimdIndex3DView depth-first", || {
                count(&|r| view.raycast_any(r))
            }),
        ];
        if kind == "oblique" {
            // The shipped dispatch against the narrower kernels; axis rays
            // take `wide` whatever is asked, so the rows would repeat.
            arms.push(paired::arm("SimdIndex3D kernel wide", || {
                count(&|r| simd.raycast_any_kernel::<0>(r))
            }));
            arms.push(paired::arm("SimdIndex3D kernel avx2", || {
                count(&|r| simd.raycast_any_kernel::<1>(r))
            }));
        }
        paired::run(&label, &mut arms, "SimdIndex3D queue (0.33.0)");
    }
}

fn rows_2d() {
    let boxes = competitors::boxes_2d(100_000, 0xF6B);
    let build = || {
        let mut b = Index2DBuilder::new(boxes.len()).node_size(NODE_SIZE);
        for r in &boxes {
            b.add(Box2D::new(r[0], r[1], r[2], r[3]));
        }
        b
    };
    let scalar: Index2D = build().finish().unwrap();
    let simd: SimdIndex2D = build().finish_simd().unwrap();
    let bytes = simd.to_bytes();
    let view = SimdIndex2DView::from_bytes(&bytes).unwrap();

    for (kind, axis) in [("oblique", false), ("axis", true)] {
        let rays = rays_2d(EARLY_EXIT_QUERIES, 0x2DA7 ^ u64::from(axis), axis);
        let hits = rays.iter().filter(|&&r| scalar.raycast_any(r)).count();
        let label = format!(
            "2d, {kind} rays, {} rays ({:.0}% hit)",
            rays.len(),
            100.0 * hits as f64 / rays.len() as f64
        );
        let count = |f: &dyn Fn(Ray2D) -> bool| {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += usize::from(f(r));
            }
            t
        };
        let mut arms = vec![
            paired::arm("SimdIndex2D queue (0.33.0)", || {
                count(&|r| simd.raycast_any_queue(r))
            }),
            paired::arm("SimdIndex2D depth-first", || {
                count(&|r| simd.raycast_any(r))
            }),
            paired::arm("Index2D raycast_any (control)", || {
                count(&|r| scalar.raycast_any(r))
            }),
            paired::arm("SimdIndex2DView queue (0.33.0)", || {
                count(&|r| view.raycast_any_queue(r))
            }),
            paired::arm("SimdIndex2DView depth-first", || {
                count(&|r| view.raycast_any(r))
            }),
        ];
        if kind == "oblique" {
            // The shipped dispatch against the narrower kernels; axis rays
            // take `wide` whatever is asked, so the rows would repeat.
            arms.push(paired::arm("SimdIndex2D kernel wide", || {
                count(&|r| simd.raycast_any_kernel::<0>(r))
            }));
            arms.push(paired::arm("SimdIndex2D kernel avx2", || {
                count(&|r| simd.raycast_any_kernel::<1>(r))
            }));
        }
        paired::run(&label, &mut arms, "SimdIndex2D queue (0.33.0)");
    }
}

fn main() {
    pin::pin_from_env();
    for (i, class) in SCENE_CLASSES.iter().enumerate() {
        let boxes = uniform_boxes_3d(class.max_side, 0x3D00_0F01 + i as u64);
        rows_3d(class.label, &boxes, 0x3D0A_11A7 + i as u64);
    }
    rows_3d("clustered", &clustered_boxes_3d(0x3D00_C1A5), 0x3D0A_C1A5);
    rows_2d();
}
