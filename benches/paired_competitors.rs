//! This crate against `static_aabb2d_index`, FlatGeobuf and the `bvh` crate, in
//! ONE binary, interleaved, on query sets sized by output class.
//!
//! The Criterion suites (`flatgeobuf2d_bench`, `index2d_bench`,
//! `raycast3d_bench`) time each participant in its own window, one after the
//! other, so a shared runner's drift between those windows lands in the
//! comparison. Here every round times every participant once, and the report
//! is the per-round ratio against the other library, which is what
//! `docs/performance.md` quotes across machines.
//!
//! Query sets come from `support/competitors.rs`: small windows 10 000, mid
//! 2000, large 400, early exits 10 000 in every class; raycast scenes sparse,
//! mid and dense by box size (a few to hundreds of boxes per ray) plus the
//! clustered scene where a SAH tree has the edge. Every participant runs the
//! same set, and the checksum column pins that they return the same hits
//! (`first` and `any` return whether a hit exists, which every participant
//! agrees on while their choice of item differs).
//!
//! 2D rows, reference `static_aabb2d_index`:
//! - collect: its `query_with_stack` (a fresh `Vec` per query, its collecting
//!   API) and `visit_query_with_stack` pushing into a reused `Vec`, against
//!   FlatGeobuf's `search` and this crate's `search_with`;
//! - visit: its callback `visit_query_with_stack` (a unit visitor, so no break
//!   test) against `visit` summing indices;
//! - first: its visitor breaking on the first hit against `first` and `any`.
//!
//! FlatGeobuf's `PackedRTree` has no callback or early-exit query, so it
//! appears in the collect rows only. For `bvh`, closest hit is a hand-rolled
//! ordered traversal of its SAH tree (its API has none), all hits is its
//! broad-phase `traverse_iterator`, and the occlusion test is that iterator's
//! first item, which it yields lazily.
//!
//! Run:
//!   taskset -c 1 cargo bench --features simd --bench paired_competitors

use std::collections::BinaryHeap;
use std::hint::black_box;
use std::ops::ControlFlow;

use flatgeobuf::packed_r_tree::{NodeItem, PackedRTree, calc_extent, hilbert_sort};
use packed_spatial_index::{
    Box2D, Box3D, Index2D, Index2DBuilder, Index3D, Index3DBuilder, NeighborWorkspace,
    SearchWorkspace, SimdIndex2D, SimdIndex3D,
};
use static_aabb2d_index::{Control, StaticAABB2DIndex, StaticAABB2DIndexBuilder};

use competitors::{
    BvhRay, EARLY_EXIT_QUERIES, SCENE_CLASSES, WINDOW_CLASSES, build_bvh, bvh_ordered_closest,
    clustered_boxes_3d, random_rays, uniform_boxes_3d, windows_2d,
};

#[path = "support/competitors.rs"]
mod competitors;
#[path = "support/paired.rs"]
mod paired;
#[path = "support/pin.rs"]
mod pin;

const N: usize = 100_000;
const NODE_SIZE: usize = 16;

fn to_box(q: &[f64; 4]) -> Box2D {
    Box2D::new(q[0], q[1], q[2], q[3])
}

fn build_flatgeobuf(boxes: &[[f64; 4]]) -> PackedRTree {
    let mut nodes: Vec<NodeItem> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut node = NodeItem::bounds(b[0], b[1], b[2], b[3]);
            node.offset = i as u64;
            node
        })
        .collect();
    let extent = calc_extent(&nodes);
    hilbert_sort(&mut nodes, &extent);
    PackedRTree::build(&nodes, &extent, NODE_SIZE as u16).unwrap()
}

fn build_static_aabb(boxes: &[[f64; 4]]) -> StaticAABB2DIndex<f64> {
    let mut b = StaticAABB2DIndexBuilder::<f64>::new_with_node_size(boxes.len(), NODE_SIZE);
    for r in boxes {
        b.add(r[0], r[1], r[2], r[3]);
    }
    b.build().unwrap()
}

fn builder_2d(boxes: &[[f64; 4]]) -> Index2DBuilder {
    let mut b = Index2DBuilder::new(boxes.len()).node_size(NODE_SIZE);
    for r in boxes {
        b.add(to_box(r));
    }
    b
}

fn builder_3d(boxes: &[Box3D]) -> Index3DBuilder {
    let mut b = Index3DBuilder::new(boxes.len()).node_size(NODE_SIZE);
    for &r in boxes {
        b.add(r);
    }
    b
}

struct Scene2D {
    fgb: PackedRTree,
    saabb: StaticAABB2DIndex<f64>,
    owned: Index2D,
    simd: SimdIndex2D,
}

fn collect_rows(s: &Scene2D, label: &str, qs: &[[f64; 4]]) {
    let (mut stack_a, mut stack_b, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    let (mut ws_o, mut ws_s) = (SearchWorkspace::new(), SearchWorkspace::new());
    let mut arms = vec![
        paired::arm("static_aabb query_with_stack", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += s
                    .saabb
                    .query_with_stack(q[0], q[1], q[2], q[3], &mut stack_a)
                    .len();
            }
            t
        }),
        paired::arm("static_aabb visit into reused Vec", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                buf.clear();
                s.saabb.visit_query_with_stack(
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                    &mut |i| buf.push(i),
                    &mut stack_b,
                );
                t += buf.len();
            }
            t
        }),
        paired::arm("flatgeobuf search", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += s.fgb.search(q[0], q[1], q[2], q[3]).unwrap().len();
            }
            t
        }),
        paired::arm("Index2D search_with", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += s.owned.search_with(to_box(q), &mut ws_o).len();
            }
            t
        }),
        paired::arm("SimdIndex2D search_with", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += s.simd.search_with(to_box(q), &mut ws_s).len();
            }
            t
        }),
    ];
    paired::run(label, &mut arms, "static_aabb query_with_stack");
}

fn visit_rows(s: &Scene2D, label: &str, qs: &[[f64; 4]]) {
    let mut stack = Vec::new();
    let mut arms = vec![
        paired::arm("static_aabb visit_query_with_stack", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                s.saabb
                    .visit_query_with_stack(q[0], q[1], q[2], q[3], &mut |i| t += i, &mut stack);
            }
            t
        }),
        paired::arm("Index2D visit", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                let _ = s.owned.visit(to_box(q), |i| {
                    t += i;
                    ControlFlow::<()>::Continue(())
                });
            }
            t
        }),
        paired::arm("SimdIndex2D visit", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                let _ = s.simd.visit(to_box(q), |i| {
                    t += i;
                    ControlFlow::<()>::Continue(())
                });
            }
            t
        }),
    ];
    paired::run(label, &mut arms, "static_aabb visit_query_with_stack");
}

fn first_rows(s: &Scene2D, label: &str, qs: &[[f64; 4]]) {
    let mut stack = Vec::new();
    let mut arms = vec![
        paired::arm("static_aabb visit, break on first", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                let found: Control<usize> = s.saabb.visit_query_with_stack(
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                    &mut |i| Control::Break(i),
                    &mut stack,
                );
                t += usize::from(matches!(found, Control::Break(_)));
            }
            t
        }),
        paired::arm("Index2D first", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += usize::from(s.owned.first(to_box(q)).is_some());
            }
            t
        }),
        paired::arm("Index2D any", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += usize::from(s.owned.any(to_box(q)));
            }
            t
        }),
        paired::arm("SimdIndex2D first", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += usize::from(s.simd.first(to_box(q)).is_some());
            }
            t
        }),
        paired::arm("SimdIndex2D any", || {
            let mut t = 0usize;
            for q in black_box(qs) {
                t += usize::from(s.simd.any(to_box(q)));
            }
            t
        }),
    ];
    paired::run(label, &mut arms, "static_aabb visit, break on first");
}

fn hits_per_query(qs: &[[f64; 4]], index: &Index2D) -> f64 {
    let hits: usize = qs.iter().map(|q| index.count(to_box(q))).sum();
    hits as f64 / qs.len() as f64
}

fn raycast_rows(tag: &str, boxes: &[Box3D], all_hits_rays: usize, seed: u64) {
    let owned: Index3D = builder_3d(boxes).finish().unwrap();
    let simd: SimdIndex3D = builder_3d(boxes).finish_simd().unwrap();
    let (bvh, shapes) = build_bvh(boxes);

    let rays = random_rays(all_hits_rays, seed);
    let hits: usize = rays.iter().map(|&r| owned.raycast(r).len()).sum();
    let label = format!(
        "raycast {tag}, all hits ({:.1} boxes/ray)",
        hits as f64 / rays.len() as f64
    );
    let (mut ws_o, mut ws_s) = (SearchWorkspace::new(), SearchWorkspace::new());
    let mut arms = vec![
        paired::arm("bvh traverse_iterator", || {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += bvh.traverse_iterator(&BvhRay(r), &shapes).count();
            }
            t
        }),
        paired::arm("Index3D raycast_with", || {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += owned.raycast_with(r, &mut ws_o).len();
            }
            t
        }),
        paired::arm("SimdIndex3D raycast_with", || {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += simd.raycast_with(r, &mut ws_s).len();
            }
            t
        }),
    ];
    paired::run(&label, &mut arms, "bvh traverse_iterator");

    let rays = random_rays(EARLY_EXIT_QUERIES, seed ^ 0xC105E);
    let label = format!("raycast {tag}, closest hit");
    let mut heap = BinaryHeap::new();
    let (mut nw_o, mut nw_s) = (NeighborWorkspace::new(), NeighborWorkspace::new());
    // Checksum: the sum of hit distances, rounded, which pins the same closest
    // hit across participants whatever id a tie resolves to.
    let mut arms = vec![
        paired::arm("bvh ordered closest", || {
            let mut t = 0.0f64;
            for &r in black_box(&rays) {
                if let Some((_, d)) = bvh_ordered_closest(&bvh, &shapes, r, &mut heap) {
                    t += d;
                }
            }
            t as usize
        }),
        paired::arm("Index3D raycast_closest_with", || {
            let mut t = 0.0f64;
            for &r in black_box(&rays) {
                if let Some((_, d)) = owned.raycast_closest_with(r, &mut nw_o) {
                    t += d;
                }
            }
            t as usize
        }),
        paired::arm("SimdIndex3D raycast_closest_with", || {
            let mut t = 0.0f64;
            for &r in black_box(&rays) {
                if let Some((_, d)) = simd.raycast_closest_with(r, &mut nw_s) {
                    t += d;
                }
            }
            t as usize
        }),
    ];
    paired::run(&label, &mut arms, "bvh ordered closest");

    let label = format!("raycast {tag}, any hit (occlusion)");
    let mut arms = vec![
        paired::arm("bvh traverse_iterator first item", || {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += usize::from(bvh.traverse_iterator(&BvhRay(r), &shapes).next().is_some());
            }
            t
        }),
        paired::arm("Index3D raycast_any", || {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += usize::from(owned.raycast_any(r));
            }
            t
        }),
        paired::arm("SimdIndex3D raycast_any", || {
            let mut t = 0usize;
            for &r in black_box(&rays) {
                t += usize::from(simd.raycast_any(r));
            }
            t
        }),
    ];
    paired::run(&label, &mut arms, "bvh traverse_iterator first item");
}

fn main() {
    pin::pin_from_env();

    let boxes = competitors::boxes_2d(N, 0xF6B);
    let scene = Scene2D {
        fgb: build_flatgeobuf(&boxes),
        saabb: build_static_aabb(&boxes),
        owned: builder_2d(&boxes).finish().unwrap(),
        simd: builder_2d(&boxes).finish_simd().unwrap(),
    };
    for (i, class) in WINDOW_CLASSES.iter().enumerate() {
        let seed = 0xC0 + i as u64;
        let qs = windows_2d(class, class.queries, seed);
        let hits = hits_per_query(&qs, &scene.owned);
        let tag = format!(
            "{} windows, {} queries ({hits:.0} hits/query)",
            class.label,
            qs.len()
        );
        collect_rows(&scene, &format!("2d collect, {tag}"), &qs);
        visit_rows(&scene, &format!("2d visit, {tag}"), &qs);

        let qs = windows_2d(class, EARLY_EXIT_QUERIES, seed ^ 0xF1257);
        let hit = qs.iter().filter(|q| scene.owned.any(to_box(q))).count();
        let label = format!(
            "2d first, {} windows, {} queries ({:.0}% hit)",
            class.label,
            qs.len(),
            100.0 * hit as f64 / qs.len() as f64
        );
        first_rows(&scene, &label, &qs);
    }

    for (i, class) in SCENE_CLASSES.iter().enumerate() {
        let boxes = uniform_boxes_3d(class.max_side, 0x3D00_0F01 + i as u64);
        raycast_rows(class.label, &boxes, class.rays, 0x3D0A_11A7 + i as u64);
    }
    raycast_rows(
        "clustered",
        &clustered_boxes_3d(0x3D00_C1A5),
        2_000,
        0x3D0A_C1A5,
    );
}
