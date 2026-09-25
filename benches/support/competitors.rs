//! Query sets and adapters shared by the benches that time this crate against
//! other libraries (`flatgeobuf2d_bench`, `index2d_bench`, `raycast3d_bench`,
//! `paired_competitors`).
//!
//! Every participant gets the same queries. The sets are sized by output class,
//! as in `paired_mask_forms`: a repeated set small enough for the branch
//! predictor to learn flatters branchy traversals, and a Zen 4 or Zen 5 learns
//! 400 small windows (kb:observation/534). Small outputs get 10 000 queries,
//! mid 2000, large 400; an early exit costs little per query, so it gets
//! 10 000 in every class. `PAIRED_QUERIES` overrides every class.
//!
//! Included by a bench via `#[path = "support/competitors.rs"] mod competitors;`.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use bvh::aabb::{Aabb, Bounded, IntersectsAabb};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::{Bvh, BvhNode};
use nalgebra::Point3;
use packed_spatial_index::{Box3D, Point3D, Ray3D};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Side of the square world both the 2D boxes and the windows live in.
pub const EXTENT_2D: f64 = 10_000.0;

/// Early-exit queries (`any`, `first`, closest hit) per class.
pub const EARLY_EXIT_QUERIES: usize = 10_000;

/// A query class: a label, the window side range, and the set size for a
/// full traversal.
pub struct WindowClass {
    pub label: &'static str,
    pub lo: f64,
    pub hi: f64,
    pub queries: usize,
}

/// Small windows meet a handful of the 100 000 boxes, mid ones about a hundred
/// to a thousand, large ones thousands to tens of thousands.
pub const WINDOW_CLASSES: [WindowClass; 3] = [
    WindowClass {
        label: "small",
        lo: 10.0,
        hi: 200.0,
        queries: 10_000,
    },
    WindowClass {
        label: "mid",
        lo: 200.0,
        hi: 1000.0,
        queries: 2_000,
    },
    WindowClass {
        label: "large",
        lo: 2000.0,
        hi: 5000.0,
        queries: 400,
    },
];

/// `default`, unless `PAIRED_QUERIES` sets every class.
pub fn set_size(default: usize) -> usize {
    std::env::var("PAIRED_QUERIES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// `n` random boxes with sides in `0.1..20` over the 2D world, as `[min_x,
/// min_y, max_x, max_y]`.
pub fn boxes_2d(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..EXTENT_2D);
            let y: f64 = rng.random_range(0.0..EXTENT_2D);
            let w: f64 = rng.random_range(0.1..20.0);
            let h: f64 = rng.random_range(0.1..20.0);
            [x, y, x + w, y + h]
        })
        .collect()
}

/// `n` windows of `class` (width and height drawn apart) inside the world.
pub fn windows_2d(class: &WindowClass, n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..set_size(n))
        .map(|_| {
            let w: f64 = rng.random_range(class.lo..class.hi);
            let h: f64 = rng.random_range(class.lo..class.hi);
            let x: f64 = rng.random_range(0.0..EXTENT_2D - w);
            let y: f64 = rng.random_range(0.0..EXTENT_2D - h);
            [x, y, x + w, y + h]
        })
        .collect()
}

/// Side of the cubic 3D world of the raycast scenes.
pub const WORLD_3D: f64 = 10_000.0;
/// Boxes per raycast scene.
pub const BOXES_3D: usize = 100_000;
/// Length of every ray.
pub const RAY_LENGTH: f64 = 4_000.0;

/// A raycast scene: its label, the largest box side (sides are drawn from
/// `1..max_side`), and the all-hits set size.
pub struct SceneClass {
    pub label: &'static str,
    pub max_side: f64,
    pub rays: usize,
}

/// Uniform scenes whose box size sets how many boxes a 4000-long ray crosses:
/// a few, tens, and hundreds on average.
pub const SCENE_CLASSES: [SceneClass; 3] = [
    SceneClass {
        label: "sparse",
        max_side: 150.0,
        rays: 10_000,
    },
    SceneClass {
        label: "mid",
        max_side: 450.0,
        rays: 2_000,
    },
    SceneClass {
        label: "dense",
        max_side: 1400.0,
        rays: 400,
    },
];

/// Uniform boxes with sides in `1..max_side`.
pub fn uniform_boxes_3d(max_side: f64, seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..BOXES_3D)
        .map(|_| {
            let x: f64 = rng.random_range(0.0..WORLD_3D);
            let y: f64 = rng.random_range(0.0..WORLD_3D);
            let z: f64 = rng.random_range(0.0..WORLD_3D);
            let dx: f64 = rng.random_range(1.0..max_side);
            let dy: f64 = rng.random_range(1.0..max_side);
            let dz: f64 = rng.random_range(1.0..max_side);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

/// Boxes with sides in `1..40` in four dense blobs, where a SAH tree has the
/// structural edge over a Hilbert-packed one.
pub fn clustered_boxes_3d(seed: u64) -> Vec<Box3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = [
        (2_000.0, 2_000.0, 2_000.0),
        (8_000.0, 2_500.0, 7_000.0),
        (3_000.0, 8_000.0, 5_000.0),
        (7_000.0, 7_000.0, 2_500.0),
    ];
    (0..BOXES_3D)
        .map(|i| {
            let (cx, cy, cz): (f64, f64, f64) = centers[i % centers.len()];
            let x = (cx + rng.random_range(-700.0..700.0)).clamp(0.0, WORLD_3D);
            let y = (cy + rng.random_range(-700.0..700.0)).clamp(0.0, WORLD_3D);
            let z = (cz + rng.random_range(-700.0..700.0)).clamp(0.0, WORLD_3D);
            let dx: f64 = rng.random_range(1.0..40.0);
            let dy: f64 = rng.random_range(1.0..40.0);
            let dz: f64 = rng.random_range(1.0..40.0);
            Box3D::new(x, y, z, x + dx, y + dy, z + dz)
        })
        .collect()
}

/// `n` rays of [`RAY_LENGTH`] from random origins in random directions.
pub fn random_rays(n: usize, seed: u64) -> Vec<Ray3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..set_size(n))
        .map(|_| {
            let z: f64 = rng.random_range(-1.0..1.0);
            let theta: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            let radius = (1.0 - z * z).sqrt();
            Ray3D::new(
                Point3D::new(
                    rng.random_range(0.0..WORLD_3D),
                    rng.random_range(0.0..WORLD_3D),
                    rng.random_range(0.0..WORLD_3D),
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
pub struct BvhBox {
    pub id: usize,
    pub bounds: Box3D,
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

pub fn to_bvh_boxes(boxes: &[Box3D]) -> Vec<BvhBox> {
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

/// The `bvh` crate's SAH tree over `boxes`, with the shapes it indexes.
pub fn build_bvh(boxes: &[Box3D]) -> (Bvh<f64, 3>, Vec<BvhBox>) {
    let mut shapes = to_bvh_boxes(boxes);
    let bvh = Bvh::<f64, 3>::build(&mut shapes);
    (bvh, shapes)
}

pub struct NodeT {
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
pub struct BvhRay(pub Ray3D);

impl IntersectsAabb<f64, 3> for BvhRay {
    fn intersects_aabb(&self, aabb: &Aabb<f64, 3>) -> bool {
        self.0.intersects_box(aabb_to_box(aabb))
    }
}

/// Fair ordered closest-hit traversal over the `bvh` crate's SAH tree: the
/// crate's own API only offers the broad-phase `traverse_iterator`, which is
/// not a closest-hit baseline, so this drives its tree front to back with the
/// same pruning as the packed closest-hit path.
pub fn bvh_ordered_closest(
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
