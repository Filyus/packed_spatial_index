//! Closest-hit raycast: packed index (AoS + SoA/SIMD) vs the `bvh` crate.
//!
//! The `bvh` column is a *fair* hand-rolled ordered closest-hit traversal over
//! the `bvh` crate's SAH tree (front-to-back priority queue with pruning) — the
//! same algorithm class as the packed closest-hit path, scalar. The crate's own
//! API only offers a broad-phase `traverse_iterator`, which is not a closest-hit
//! baseline, so we drive its tree directly.
//!
//! Two datasets: `uniform` (boxes spread across the world) and `clustered`
//! (boxes in four dense blobs), because the SAH tree's advantage shows up only
//! on clustered scenes. Build time is reported separately.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::time::Instant;

use bvh::aabb::{Aabb, Bounded, IntersectsAabb};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::{Bvh, BvhNode};
use criterion::{Criterion, criterion_group};
use nalgebra::Point3;
use packed_spatial_index::{
    Box3D, Index3D, Index3DBuilder, NeighborWorkspace, Point3D, Ray3D, SearchWorkspace, SimdIndex3D,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const WORLD: f64 = 10_000.0;
const NUM_BOXES: usize = 100_000;
const NUM_RAYS: usize = 1_000;
const RAY_LENGTH: f64 = 4_000.0;

fn random_boxes(seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..NUM_BOXES)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..WORLD);
            let y: f64 = rng.random_range(0.0..WORLD);
            let z: f64 = rng.random_range(0.0..WORLD);
            let dx: f64 = rng.random_range(1.0..40.0);
            let dy: f64 = rng.random_range(1.0..40.0);
            let dz: f64 = rng.random_range(1.0..40.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn clustered_boxes(seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = [
        (2_000.0, 2_000.0, 2_000.0),
        (8_000.0, 2_500.0, 7_000.0),
        (3_000.0, 8_000.0, 5_000.0),
        (7_000.0, 7_000.0, 2_500.0),
    ];
    (0..NUM_BOXES)
        .map(|i| {
            let (cx, cy, cz): (f64, f64, f64) = centers[i % centers.len()];
            let x = (cx + rng.random_range(-700.0..700.0)).clamp(0.0, WORLD);
            let y = (cy + rng.random_range(-700.0..700.0)).clamp(0.0, WORLD);
            let z = (cz + rng.random_range(-700.0..700.0)).clamp(0.0, WORLD);
            let dx: f64 = rng.random_range(1.0..40.0);
            let dy: f64 = rng.random_range(1.0..40.0);
            let dz: f64 = rng.random_range(1.0..40.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

fn random_rays(seed: u64) -> Vec<Ray3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..NUM_RAYS)
        .map(|_| {
            let z: f64 = rng.random_range(-1.0..1.0);
            let theta: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let radius = (1.0 - z * z).sqrt();
            Ray3D::new(
                Point3D::new(
                    rng.random_range(0.0..WORLD),
                    rng.random_range(0.0..WORLD),
                    rng.random_range(0.0..WORLD),
                ),
                radius * theta.cos(),
                radius * theta.sin(),
                z,
                RAY_LENGTH,
            )
        })
        .collect()
}

#[derive(Clone, Copy)]
struct BvhBox {
    id: usize,
    bounds: Box3D,
    node_index: usize,
}

impl Bounded<f64, 3> for BvhBox {
    fn aabb(&self) -> Aabb<f64, 3> {
        Aabb::with_bounds(
            Point3::new(self.bounds.min_x, self.bounds.min_y, self.bounds.min_z),
            Point3::new(self.bounds.max_x, self.bounds.max_y, self.bounds.max_z),
        )
    }
}

impl BHShape<f64, 3> for BvhBox {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

fn to_bvh_boxes(boxes: &[Box3D]) -> Vec<BvhBox> {
    boxes
        .iter()
        .enumerate()
        .map(|(id, &bounds)| BvhBox {
            id,
            bounds,
            node_index: 0,
        })
        .collect()
}

struct NodeT {
    t: f64,
    idx: usize,
}
impl PartialEq for NodeT {
    fn eq(&self, o: &Self) -> bool {
        self.t == o.t
    }
}
impl Eq for NodeT {}
impl Ord for NodeT {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reverse so the max-heap yields the nearest node first.
        o.t.total_cmp(&self.t)
    }
}
impl PartialOrd for NodeT {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

fn aabb_to_box(a: &Aabb<f64, 3>) -> Box3D {
    Box3D::new(a.min.x, a.min.y, a.min.z, a.max.x, a.max.y, a.max.z)
}

/// Ray adapter for the `bvh` crate's broad-phase `traverse_iterator` (all-hits):
/// a leaf is yielded when the ray segment intersects its AABB.
struct BvhRay(Ray3D);

impl IntersectsAabb<f64, 3> for BvhRay {
    fn intersects_aabb(&self, aabb: &Aabb<f64, 3>) -> bool {
        self.0.intersects_box(aabb_to_box(aabb))
    }
}

/// Fair ordered closest-hit traversal over the `bvh` crate's SAH tree.
fn bvh_ordered_closest(
    bvh: &Bvh<f64, 3>,
    shapes: &[BvhBox],
    ray: Ray3D,
    heap: &mut BinaryHeap<NodeT>,
) -> Option<(usize, f64)> {
    heap.clear();
    if bvh.nodes.is_empty() {
        return None;
    }
    heap.push(NodeT { t: 0.0, idx: 0 });
    let mut best_t = ray.max_distance;
    let mut best = None;
    while let Some(NodeT { t, idx }) = heap.pop() {
        if t >= best_t {
            break;
        }
        match &bvh.nodes[idx] {
            BvhNode::Leaf { shape_index, .. } => {
                let s = &shapes[*shape_index];
                if let Some(et) = ray.enter_t(s.bounds)
                    && et < best_t
                {
                    best_t = et;
                    best = Some(s.id);
                }
            }
            BvhNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } => {
                if let Some(lt) = ray.enter_t(aabb_to_box(child_l_aabb))
                    && lt < best_t
                {
                    heap.push(NodeT {
                        t: lt,
                        idx: *child_l_index,
                    });
                }
                if let Some(rt) = ray.enter_t(aabb_to_box(child_r_aabb))
                    && rt < best_t
                {
                    heap.push(NodeT {
                        t: rt,
                        idx: *child_r_index,
                    });
                }
            }
        }
    }
    best.map(|id| (id, best_t))
}

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

fn bench_dataset(c: &mut Criterion, label: &str, boxes: Vec<Box3D>, rays: &[Ray3D]) {
    let packed = build_packed(&boxes);
    let simd = {
        let mut builder = Index3DBuilder::new(boxes.len());
        for &b in &boxes {
            builder.add(b);
        }
        builder.finish_simd().unwrap()
    };
    let mut shapes = to_bvh_boxes(&boxes);
    let bvh = Bvh::<f64, 3>::build(&mut shapes);

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

    bench_all_hits(c, label, &simd, &bvh, &shapes, rays);
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
    let uniform = random_boxes(0x3D00_0F01);
    let clustered = clustered_boxes(0x3D00_C1A5);
    let rays = random_rays(0x3D0A_11A7);

    // Sanity: all three closest-hit paths must agree before timing them.
    {
        let packed = build_packed(&uniform);
        let mut shapes = to_bvh_boxes(&uniform);
        let bvh = Bvh::<f64, 3>::build(&mut shapes);
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
        let _ = Instant::now();
    }

    time_build(c, "build_uniform", &uniform);
    bench_dataset(c, "uniform", uniform, &rays);
    bench_dataset(c, "clustered", clustered, &rays);
}

criterion_group!(benches, raycast_benches);
#[path = "support/pin.rs"]
mod pin;

fn main() {
    pin::pin_from_env();
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
