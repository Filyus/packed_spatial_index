//! Closest-hit raycast: packed index (AoS + SoA/SIMD) vs the `bvh` crate.
//!
//! The `bvh` column is a *fair* hand-rolled ordered closest-hit traversal over
//! the `bvh` crate's SAH tree (front-to-back priority queue with pruning) — the
//! same algorithm class as the packed closest-hit path, scalar. The crate's own
//! API only offers a broad-phase `traverse_iterator`, which is not a closest-hit
//! baseline, so we drive its tree directly (`support/competitors.rs`).
//!
//! Scenes: three uniform ones whose box size sets how many boxes a ray crosses
//! (`sparse`, `mid`, `dense`: a few, tens, hundreds), and `clustered` (boxes in
//! four dense blobs), because the SAH tree's advantage shows up only on
//! clustered scenes. All-hits runs 10 000 / 2000 / 400 rays by density,
//! closest hit 10 000 in every scene, so no scene's set is small enough for the
//! branch predictor to learn. `paired_competitors` times the same comparison
//! interleaved in one binary. Build time is reported separately.

use std::collections::BinaryHeap;
use std::hint::black_box;

use bvh::bvh::Bvh;
use criterion::{Criterion, criterion_group};
use packed_spatial_index::{
    Box3D, Index3D, Index3DBuilder, NeighborWorkspace, Ray3D, SearchWorkspace, SimdIndex3D,
};

use competitors::{
    BvhBox, BvhRay, EARLY_EXIT_QUERIES, SCENE_CLASSES, build_bvh, bvh_ordered_closest,
    clustered_boxes_3d, random_rays, to_bvh_boxes, uniform_boxes_3d,
};

#[path = "support/competitors.rs"]
mod competitors;

fn build_packed(boxes: &[Box3D]) -> Index3D {
    let mut builder = Index3DBuilder::new(boxes.len());
    for &b in boxes {
        builder.add(b);
    }
    builder.finish().unwrap()
}

fn time_build(c: &mut Criterion, name: &str, boxes: &[Box3D]) {
    let mut group = c.benchmark_group(name);
    group.bench_function("packed_index", |b| {
        b.iter(|| black_box(build_packed(boxes)));
    });
    group.bench_function("bvh_crate", |b| {
        b.iter(|| {
            let mut shapes = to_bvh_boxes(boxes);
            black_box(Bvh::<f64, 3>::build(&mut shapes))
        });
    });
    group.finish();
}

fn bench_dataset(
    c: &mut Criterion,
    label: &str,
    boxes: Vec<Box3D>,
    rays: &[Ray3D],
    all_hits_rays: &[Ray3D],
) {
    let packed = build_packed(&boxes);
    let simd = {
        let mut builder = Index3DBuilder::new(boxes.len());
        for &b in &boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    };
    let (bvh, shapes) = build_bvh(&boxes);

    let mut group = c.benchmark_group(format!("closest_{label}"));
    group.bench_function("packed_aos", |b| {
        let mut ws = NeighborWorkspace::new();
        b.iter(|| {
            let mut acc = 0usize;
            for &ray in rays {
                if let Some((i, _)) = packed.raycast_closest_with(ray, &mut ws) {
                    acc ^= i;
                }
            }
            black_box(acc)
        });
    });
    group.bench_function("packed_soa_simd", |b| {
        let mut ws = NeighborWorkspace::new();
        b.iter(|| {
            let mut acc = 0usize;
            for &ray in rays {
                if let Some((i, _)) = simd.raycast_closest_with(ray, &mut ws) {
                    acc ^= i;
                }
            }
            black_box(acc)
        });
    });
    group.bench_function("bvh_crate_ordered", |b| {
        let mut heap = BinaryHeap::new();
        b.iter(|| {
            let mut acc = 0usize;
            for &ray in rays {
                if let Some((i, _)) = bvh_ordered_closest(&bvh, &shapes, ray, &mut heap) {
                    acc ^= i;
                }
            }
            black_box(acc)
        });
    });
    group.finish();

    bench_all_hits(c, label, &simd, &bvh, &shapes, all_hits_rays);
}

/// All-hits raycast: packed SoA/SIMD vs the `bvh` crate's broad-phase
/// `traverse_iterator` (the crate's natural all-candidates query).
fn bench_all_hits(
    c: &mut Criterion,
    label: &str,
    simd: &SimdIndex3D,
    bvh: &Bvh<f64, 3>,
    shapes: &[BvhBox],
    rays: &[Ray3D],
) {
    let mut group = c.benchmark_group(format!("all_hits_{label}"));
    group.bench_function("packed_soa_simd", |b| {
        let mut ws = SearchWorkspace::new();
        b.iter(|| {
            let mut acc = 0usize;
            for &ray in rays {
                acc += simd.raycast_with(ray, &mut ws).len();
            }
            black_box(acc)
        });
    });
    group.bench_function("bvh_crate_broad", |b| {
        b.iter(|| {
            let mut acc = 0usize;
            for &ray in rays {
                acc += bvh.traverse_iterator(&BvhRay(ray), shapes).count();
            }
            black_box(acc)
        });
    });
    group.finish();
}

fn raycast_benches(c: &mut Criterion) {
    let uniform = uniform_boxes_3d(SCENE_CLASSES[0].max_side, 0x3D00_0F01);
    let rays = random_rays(EARLY_EXIT_QUERIES, 0x3D0A_11A7);

    // Sanity: all three closest-hit paths must agree before timing them.
    {
        let packed = build_packed(&uniform);
        let (bvh, shapes) = build_bvh(&uniform);
        let mut heap = BinaryHeap::new();
        let mut ws = NeighborWorkspace::new();
        for &ray in rays.iter().take(200) {
            let a = packed.raycast_closest_with(ray, &mut ws).map(|(_, t)| t);
            let b = bvh_ordered_closest(&bvh, &shapes, ray, &mut heap).map(|(_, t)| t);
            let agree = match (a, b) {
                (None, None) => true,
                (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * y.abs().max(1.0),
                _ => false,
            };
            assert!(agree, "closest-hit disagreement: packed {a:?} vs bvh {b:?}");

            // All-hits sets must match exactly (both test the box AABB).
            let mut packed_hits = packed.raycast(ray);
            packed_hits.sort_unstable();
            let mut bvh_hits: Vec<usize> = bvh
                .traverse_iterator(&BvhRay(ray), &shapes)
                .map(|s| s.id)
                .collect();
            bvh_hits.sort_unstable();
            assert_eq!(packed_hits, bvh_hits, "all-hits set disagreement");
        }
    }

    time_build(c, "build_uniform", &uniform);
    for (i, class) in SCENE_CLASSES.iter().enumerate() {
        let boxes = if i == 0 {
            uniform.clone()
        } else {
            uniform_boxes_3d(class.max_side, 0x3D00_0F01 + i as u64)
        };
        let all_hits = random_rays(class.rays, 0x3D0A_AA11 + i as u64);
        bench_dataset(c, class.label, boxes, &rays, &all_hits);
    }
    let all_hits = random_rays(2_000, 0x3D0A_C1A5);
    bench_dataset(
        c,
        "clustered",
        clustered_boxes_3d(0x3D00_C1A5),
        &rays,
        &all_hits,
    );
}

criterion_group! {
    name = benches;
    config = pin::criterion();
    targets = raycast_benches
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
